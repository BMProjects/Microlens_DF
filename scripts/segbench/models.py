"""分割候选模型注册表：统一为 model(x[B,3,H,W]) → logits[B,C,H,W]（H、W 为 32 的倍数）。

第三方实现直接引用 output/segbench/repos/ 下的官方代码（版本见 repos_versions.txt），只做输入输出适配。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOS = PROJECT_ROOT / "output/segbench/repos"
WEIGHTS = PROJECT_ROOT / "output/segbench/weights"


def conv_bn(i: int, o: int, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, i: int, skip: int, o: int):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(i + skip, o), conv_bn(o, o))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class DINOv3UNet(nn.Module):
    """DINOv3 ConvNeXt 编码器（1/4–1/32）+ 全分辨率细节分支（1/1、1/2）+ U 形解码器。
    细节分支补足 ConvNeXt 起始 4 倍下采样丢失的 2–5 像素细线；编码器用较小学习率微调。"""

    def __init__(self, n_classes: int, weights: str = "dinov3-convnext-base-pretrain-lvd1689m"):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(str(WEIGHTS / weights))
        ch = self.enc.config.hidden_sizes          # 128, 256, 512, 1024（base）
        self.d1 = nn.Sequential(conv_bn(3, 32), conv_bn(32, 32))
        self.d2 = nn.Sequential(conv_bn(32, 64, 2), conv_bn(64, 64))
        self.u3 = UpBlock(ch[3], ch[2], 256)
        self.u2 = UpBlock(256, ch[1], 128)
        self.u1 = UpBlock(128, ch[0], 96)
        self.u0 = UpBlock(96, 64, 64)
        self.uf = UpBlock(64, 32, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def encoder_params(self):
        return self.enc.parameters()

    def forward(self, x):
        h = self.enc(x, output_hidden_states=True).hidden_states   # [stem 输出, 1/4, 1/8, 1/16, 1/32]
        f1 = self.d1(x)
        f2 = self.d2(f1)
        y = self.u3(h[4], h[3])
        y = self.u2(y, h[2])
        y = self.u1(y, h[1])
        y = self.u0(y, f2)
        y = self.uf(y, f1)
        return self.head(y)


class DSCNetMulti(nn.Module):
    """官方 DSCNet（ICCV 2023，2D 版）：去掉末端 sigmoid，输出多类 logits。"""

    def __init__(self, n_classes: int, device: str):
        super().__init__()
        sys.path.insert(0, str(REPOS / "DSCNet/DSCNet_2D_opensource/Code/DRIVE/DSCNet"))
        from S3_DSCNet import DSCNet
        self.net = DSCNet(3, n_classes, kernel_size=9, extend_scope=1.0, if_offset=True, device=device,
                          number=16, dim=1)
        self.net.sigmoid = nn.Identity()

    def forward(self, x):
        return self.net(x)


def build(name: str, n_classes: int, device: str = "cuda") -> nn.Module:
    if name in ("unet_r34", "fpn_r34"):
        import segmentation_models_pytorch as smp
        cls = smp.Unet if name == "unet_r34" else smp.FPN
        return cls("resnet34", encoder_weights="imagenet", classes=n_classes)
    if name == "dsgnet":
        sys.path.insert(0, str(REPOS / "DSGNet"))
        from dsgnet.dsgnet import DSGNet
        return DSGNet(in_channels=3, n_classes=n_classes)
    if name == "dscnet":
        return DSCNetMulti(n_classes, device)
    if name == "dinov3_unet":
        return DINOv3UNet(n_classes)
    raise KeyError(f"未知模型 {name}")


def param_groups(model: nn.Module, lr: float, enc_lr_mult: float = 0.1) -> list[dict]:
    """预训练编码器用较小学习率，其余部分用基础学习率。"""
    enc = list(model.encoder_params()) if hasattr(model, "encoder_params") else (
        list(model.encoder.parameters()) if hasattr(model, "encoder") and hasattr(model, "decoder") else [])
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in ids]
    groups = [{"params": rest, "lr": lr}]
    if enc:
        groups.append({"params": enc, "lr": lr * enc_lr_mult})
    return groups
