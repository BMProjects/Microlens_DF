"""本地 HRNet-OCR 实现.

基于 HRNet-Semantic-Segmentation 官方 `seg_hrnet_ocr.py` 结构改写，
目标是无缝接入当前项目的 PyTorch 训练/推理主线。

与官方实现相比，这里做了两点工程化适配：
1. 支持灰度单通道输入 (`in_channels=1`)
2. `forward()` 直接返回与输入同分辨率的最终 logits，方便复用现有 trainer/predictor
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ALIGN_CORNERS = True
BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class SpatialGatherModule(nn.Module):
    def __init__(self, scale: int = 1):
        super().__init__()
        self.scale = scale

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes, _, _ = probs.shape
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, feats.size(1), -1).permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        return torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)


class ObjectAttentionBlock2D(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, scale: int = 1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else None

        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        if self.pool is not None:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        sim_map = F.softmax(torch.matmul(query, key) * (self.key_channels ** -0.5), dim=-1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.shape[2:])
        context = self.f_up(context)
        if self.pool is not None:
            context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS)
        return context


class SpatialOCRModule(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, out_channels: int, dropout: float = 0.05):
        super().__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale=1)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats: torch.Tensor, proxy_feats: torch.Tensor) -> torch.Tensor:
        context = self.object_context_block(feats, proxy_feats)
        return self.conv_bn_dropout(torch.cat([context, feats], dim=1))


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        block: type[nn.Module],
        num_blocks: list[int],
        num_inchannels: list[int],
        num_channels: list[int],
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.num_inchannels = list(num_inchannels)
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index: int, block: type[nn.Module], num_blocks: list[int], num_channels: list[int], stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = num_channels[branch_index] * block.expansion
        if stride != 1 or self.num_inchannels[branch_index] != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = out_channels
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, block: type[nn.Module], num_blocks: list[int], num_channels: list[int]) -> nn.ModuleList:
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(self.num_branches)])

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        out_channels = self.num_inchannels[i] if k == i - j - 1 else self.num_inchannels[j]
                        seq = [
                            nn.Conv2d(self.num_inchannels[j], out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        ]
                        if k != i - j - 1:
                            seq.append(nn.ReLU(inplace=True))
                        conv3x3s.append(nn.Sequential(*seq))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        return self.num_inchannels

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        assert self.fuse_layers is not None
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=x[i].shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


@dataclass(frozen=True)
class HRNetOCRW18Config:
    stage1_block: str = "BOTTLENECK"
    stage1_num_blocks: tuple[int, ...] = (4,)
    stage1_num_channels: tuple[int, ...] = (64,)
    stage2_num_modules: int = 1
    stage2_num_branches: int = 2
    stage2_block: str = "BASIC"
    stage2_num_blocks: tuple[int, ...] = (4, 4)
    stage2_num_channels: tuple[int, ...] = (18, 36)
    stage3_num_modules: int = 4
    stage3_num_branches: int = 3
    stage3_block: str = "BASIC"
    stage3_num_blocks: tuple[int, ...] = (4, 4, 4)
    stage3_num_channels: tuple[int, ...] = (18, 36, 72)
    stage4_num_modules: int = 3
    stage4_num_branches: int = 4
    stage4_block: str = "BASIC"
    stage4_num_blocks: tuple[int, ...] = (4, 4, 4, 4)
    stage4_num_channels: tuple[int, ...] = (18, 36, 72, 144)
    ocr_mid_channels: int = 512
    ocr_key_channels: int = 256
    ocr_dropout: float = 0.05


class HRNetOCRW18(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, config: HRNetOCRW18Config | None = None):
        super().__init__()
        cfg = config or HRNetOCRW18Config()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        stage1_block = blocks_dict[cfg.stage1_block]
        self.layer1 = self._make_layer(stage1_block, 64, cfg.stage1_num_channels[0], cfg.stage1_num_blocks[0])
        stage1_out = stage1_block.expansion * cfg.stage1_num_channels[0]

        stage2_block = blocks_dict[cfg.stage2_block]
        stage2_channels = [c * stage2_block.expansion for c in cfg.stage2_num_channels]
        self.transition1 = self._make_transition_layer([stage1_out], stage2_channels)
        self.stage2, pre_stage_channels = self._make_stage(cfg.stage2_num_modules, cfg.stage2_num_branches, stage2_block, list(cfg.stage2_num_blocks), stage2_channels, stage2_channels)

        stage3_block = blocks_dict[cfg.stage3_block]
        stage3_channels = [c * stage3_block.expansion for c in cfg.stage3_num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, stage3_channels)
        self.stage3, pre_stage_channels = self._make_stage(cfg.stage3_num_modules, cfg.stage3_num_branches, stage3_block, list(cfg.stage3_num_blocks), stage3_channels, stage3_channels)

        stage4_block = blocks_dict[cfg.stage4_block]
        stage4_channels = [c * stage4_block.expansion for c in cfg.stage4_num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, stage4_channels)
        self.stage4, pre_stage_channels = self._make_stage(cfg.stage4_num_modules, cfg.stage4_num_branches, stage4_block, list(cfg.stage4_num_blocks), stage4_channels, stage4_channels)

        last_inp_channels = sum(pre_stage_channels)
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, cfg.ocr_mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cfg.ocr_mid_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGatherModule()
        self.ocr_distri_head = SpatialOCRModule(cfg.ocr_mid_channels, cfg.ocr_key_channels, cfg.ocr_mid_channels, cfg.ocr_dropout)
        self.cls_head = nn.Conv2d(cfg.ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self._init_weights()

    def _make_transition_layer(self, num_channels_pre_layer: list[int], num_channels_cur_layer: list[int]) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block: type[nn.Module], inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, num_modules: int, num_branches: int, block: type[nn.Module], num_blocks: list[int], num_inchannels: list[int], num_channels: list[int], multi_scale_output: bool = True) -> tuple[nn.Sequential, list[int]]:
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i != num_modules - 1
            module = HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, reset_multi_scale_output)
            modules.append(module)
            num_inchannels = module.get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x for i in range(len(self.transition1))]
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                source = y_list[i] if i < len(y_list) else y_list[-1]
                x_list.append(self.transition2[i](source))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                source = y_list[i] if i < len(y_list) else y_list[-1]
                x_list.append(self.transition3[i](source))
            else:
                x_list.append(y_list[i])
        x_list = self.stage4(x_list)

        h, w = x_list[0].shape[-2:]
        upsampled = [x_list[0]]
        for feat in x_list[1:]:
            upsampled.append(F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=ALIGN_CORNERS))
        feats = torch.cat(upsampled, dim=1)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        out = self.cls_head(self.ocr_distri_head(feats, context))
        return out_aux, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, out = self._forward_features(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)
        return out

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_aux, out = self._forward_features(x)
        if out_aux.shape[-2:] != x.shape[-2:]:
            out_aux = F.interpolate(out_aux, size=x.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=ALIGN_CORNERS)
        return out_aux, out

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if any(part in name for part in {"cls_head", "aux_head", "ocr"}):
                    nn.init.normal_(module.weight, std=0.001)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
