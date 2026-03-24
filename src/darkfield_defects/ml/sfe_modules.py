"""
SFE (Shape-aware Feature Extraction) 模块
==========================================
从 GSDNet 论文提取并适配的方向性特征提取模块，用于增强细长缺陷 (scratch) 的检测。

核心思想:
  - LODConv (Linear Offset Deformable Conv): 沿 x 或 y 方向的可变形卷积，
    通过学习偏移量自适应调整采样位置，捕获细长结构的方向性特征。
  - SFEModule: 并行4分支 — 标准 3×3 + 二阶 3×3 + LODConv(x) + LODConv(y)，
    concat 后输出，同时捕获各方向的形状信息。

适配变更 (相对于 GSDNet 原始代码):
  - 移除 einops 依赖，改用纯 PyTorch tensor ops
  - 移除 @autocast 类装饰器 (PyTorch 2.x 已弃用)
  - 设备自适应 (跟随 input tensor，不硬编码 'cuda')
  - 兼容 ultralytics 8.4.24 的 Conv/C2f/Bottleneck API

参考: GSDNet (https://github.com/FisherYuuri/GSDNet)
论文: "A lightweight and robust detection network for diverse glass surface defects
       via scale- and shape-aware feature extraction"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C2f
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "LODConv",
    "SFEModule",
    "SFEModuleN",
    "Bottleneck_SFE",
    "Bottleneck_SFEN",
    "SFEBlock",
    "SFEBlockN",
    "SA",
]


# ── LODConv: 方向约束可变形卷积 ──────────────────────────────────


def _get_coordinate_map_2d(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 LODConv 的 2D 坐标映射。

    Args:
        offset: 偏移预测 [B, 2*K, W, H]
        morph: 0=沿x轴方向, 1=沿y轴方向
        extend_scope: 偏移扩展范围

    Returns:
        (y_coordinate_map, x_coordinate_map)
    """
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = offset.device

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    # 构建基础网格坐标
    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = y_center_.view(1, width, 1).expand(kernel_size, width, height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = x_center_.view(1, 1, height).expand(kernel_size, width, height)

    if morph == 0:
        # x 方向: 核沿 x 轴展开
        y_spread_ = torch.zeros(kernel_size, device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = y_spread_.view(-1, 1, 1).expand(kernel_size, width, height)
        x_grid_ = x_spread_.view(-1, 1, 1).expand(kernel_size, width, height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = y_new_.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_new_ = x_new_.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 迭代偏移: 中心不动，向两侧累积
        y_offset_t = y_offset_.permute(1, 0, 2, 3)  # [K, B, W, H]
        y_offset_new_ = y_offset_t.detach().clone()
        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_t[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_t[center - index]
            )

        y_offset_new_ = y_offset_new_.permute(1, 0, 2, 3)  # [B, K, W, H]
        y_new_ = y_new_ + y_offset_new_ * extend_scope

        # 展平: [B, K, W, H] → [B, W*K, H]
        y_coordinate_map = y_new_.permute(0, 2, 1, 3).reshape(batch_size, width * kernel_size, height)
        x_coordinate_map = x_new_.permute(0, 2, 1, 3).reshape(batch_size, width * kernel_size, height)

    elif morph == 1:
        # y 方向: 核沿 y 轴展开
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros(kernel_size, device=device)

        y_grid_ = y_spread_.view(-1, 1, 1).expand(kernel_size, width, height)
        x_grid_ = x_spread_.view(-1, 1, 1).expand(kernel_size, width, height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = y_new_.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_new_ = x_new_.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 迭代偏移
        x_offset_t = x_offset_.permute(1, 0, 2, 3)
        x_offset_new_ = x_offset_t.detach().clone()
        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_t[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_t[center - index]
            )

        x_offset_new_ = x_offset_new_.permute(1, 0, 2, 3)
        x_new_ = x_new_ + x_offset_new_ * extend_scope

        # 展平: [B, K, W, H] → [B, W, H*K]
        y_coordinate_map = y_new_.permute(0, 2, 3, 1).reshape(batch_size, width, height * kernel_size)
        x_coordinate_map = x_new_.permute(0, 2, 3, 1).reshape(batch_size, width, height * kernel_size)

    else:
        raise ValueError("morph should be 0 or 1")

    return y_coordinate_map, x_coordinate_map


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin_min: float,
    origin_max: float,
) -> torch.Tensor:
    """将坐标映射从 [origin_min, origin_max] 缩放到 [-1, 1]（grid_sample 要求）。"""
    coordinate_map = torch.clamp(coordinate_map, origin_min, origin_max)
    scale_factor = 2.0 / (origin_max - origin_min)
    return -1.0 + scale_factor * (coordinate_map - origin_min)


def _get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
) -> torch.Tensor:
    """基于坐标映射进行双线性插值采样。"""
    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_scaled = _coordinate_map_scaling(y_coordinate_map, 0, y_max)
    x_scaled = _coordinate_map_scaling(x_coordinate_map, 0, x_max)

    # grid_sample 需要 [B, H, W, 2] 格式，(x, y) 顺序
    # grid 必须与 input_feature 类型一致（AMP 下 input 为 float16，grid 默认 float32）
    grid = torch.stack([x_scaled, y_scaled], dim=-1).unsqueeze(0) if y_scaled.dim() == 2 else \
           torch.stack([x_scaled, y_scaled], dim=-1)
    grid = grid.to(dtype=input_feature.dtype)

    return F.grid_sample(
        input=input_feature,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class LODConv(nn.Module):
    """Linear Offset Deformable Convolution — 方向约束可变形卷积。

    沿单一方向（x 或 y）进行可变形采样，增强对细长结构的特征提取。
    morph=0: 沿 x 轴（捕获水平方向特征）
    morph=1: 沿 y 轴（捕获垂直方向特征）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        extend_scope: float = 1.0,
        morph: int = 0,
    ):
        super().__init__()
        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph

        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预测偏移 [-1, 1]
        offset = self.tanh(self.gn_offset(self.offset_conv(x)))

        # 计算坐标映射
        y_coord, x_coord = _get_coordinate_map_2d(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
        )

        # 双线性插值采样
        deformed = _get_interpolated_feature(x, y_coord, x_coord)

        # 方向卷积
        if self.morph == 0:
            output = self.dsc_conv_x(deformed)
        else:
            output = self.dsc_conv_y(deformed)

        return self.relu(self.gn(output))


# ── SFE 模块 ────────────────────────────────────────────────────


class SFEModule(nn.Module):
    """Shape-aware Feature Extraction — 4 分支并行特征提取。

    分支 0: 标准 Conv (局部特征)
    分支 1: 二阶 Conv (更大感受野)
    分支 2: LODConv(morph=0) (x 方向可变形)
    分支 3: LODConv(morph=1) (y 方向可变形)

    输出通道 = 4 × (ouc // 4) = ouc
    """

    def __init__(self, inc: int, ouc: int, k: int = 3) -> None:
        super().__init__()
        c_ = ouc // 4
        self.conv_0 = Conv(inc, c_, k)
        self.conv_1 = Conv(c_, c_, k)
        self.conv_x = LODConv(c_, c_, k, morph=0)
        self.conv_y = LODConv(c_, c_, k, morph=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.conv_0(x)
        y1 = self.conv_1(y0)
        y2 = self.conv_x(y0)
        y3 = self.conv_y(y0)
        return torch.cat([y0, y1, y2, y3], dim=1)


class SFEModuleN(nn.Module):
    """SFEModule 轻量版 (无 LODConv) — 用于 detection head，减少计算量。"""

    def __init__(self, inc: int, ouc: int, k: int = 3) -> None:
        super().__init__()
        c_ = ouc // 4
        self.conv_0 = Conv(inc, c_, k)
        self.conv_1 = Conv(c_, c_, k)
        self.conv_x = Conv(c_, c_, k)
        self.conv_y = Conv(c_, c_, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.conv_0(x)
        y1 = self.conv_1(y0)
        y2 = self.conv_x(y0)
        y3 = self.conv_y(y0)
        return torch.cat([y0, y1, y2, y3], dim=1)


# ── Bottleneck 变体 ──────────────────────────────────────────────


class Bottleneck_SFE(Bottleneck):
    """SFE 增强的 Bottleneck — backbone 使用 (含 LODConv)。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cv1 = SFEModule(c1, c2, k[1])

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


class Bottleneck_SFEN(Bottleneck):
    """SFE 轻量 Bottleneck — head 使用 (无 LODConv)。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cv1 = SFEModuleN(c1, c2, k[1])

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


# ── C2f 变体 (可直接在 YAML 中使用) ─────────────────────────────


class SFEBlock(C2f):
    """C2f + SFE — 替代 backbone 中的 C3k2/C2f (含 LODConv)。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_SFE(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


class SFEBlockN(C2f):
    """C2f + SFE 轻量版 — 替代 head 中的 C2f (无 LODConv)。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_SFEN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


# ── SA: Scale-Aware Attention (SPPF 变体) ───────────────────────


class SA(nn.Module):
    """Scale-Aware attention — 多尺度池化 + 空间注意力门控。

    替代 SPPF，在不同尺度的特征上学习注意力权重。
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv_squeeze = nn.Conv2d(2, 4, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # 空间注意力
        attn = torch.cat([y0, y1, y2, y3], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        # 加权聚合
        attn = (
            y0 * sig[:, 0:1, :, :]
            + y1 * sig[:, 1:2, :, :]
            + y2 * sig[:, 2:3, :, :]
            + y3 * sig[:, 3:4, :, :]
        )
        attn = self.cv2(attn)
        return x * attn


# ── 模块注册辅助函数 ─────────────────────────────────────────────


def _patch_parse_model_for_sfe(tasks) -> bool:
    """将 SFEBlock/SFEBlockN/SA 注入 parse_model 的本地 frozenset。

    parse_model 在函数体内定义 base_modules 和 repeat_modules 两个 frozenset，
    外部无法直接修改。唯一的方法是通过 inspect.getsource 获取源码，字符串替换后
    re-exec，覆盖模块中的 parse_model 定义。

    base_modules: 告诉 parse_model 此类模块的 args 格式为 (c1, c2, *extra)，
                  使其自动注入 c1（前一层通道数）。
    repeat_modules: 告诉 parse_model 第一个非通道 arg 是 depth repeat 数 n。
    """
    import inspect
    import textwrap

    src = inspect.getsource(tasks.parse_model)
    # getsource 对缩进方法返回带缩进的代码，dedent 后才能 compile
    src = textwrap.dedent(src)

    # 已 patch 过则跳过
    if "SFEBlock" in src:
        return False

    # 1. 注入 base_modules
    target1 = "base_modules = frozenset(\n        {\n            Classify,"
    inject1 = "base_modules = frozenset(\n        {\n            SFEBlock, SFEBlockN, SA,\n            Classify,"
    src = src.replace(target1, inject1, 1)

    # 2. 注入 repeat_modules
    target2 = "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n            Bottl"
    inject2 = "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n            SFEBlock, SFEBlockN,\n            Bottl"
    src = src.replace(target2, inject2, 1)

    # compile 并覆盖 tasks 模块中的 parse_model
    code = compile(src, tasks.__file__ or "<tasks>", "exec")
    ns = tasks.__dict__.copy()
    exec(code, ns)  # noqa: S102
    tasks.parse_model = ns["parse_model"]

    return True


def register_sfe_modules():
    """向 ultralytics 注册自定义 SFE 模块，使其可在 YAML 配置中使用。

    完成两件事:
    1. tasks.__dict__ 注入模块名 → parse_model 的 globals()[m] 可找到它们
    2. 重写 parse_model → 将 SFEBlock/SFEBlockN 加入 base_modules/repeat_modules，
       确保 parse_model 正确注入 c1（前一层通道数）并处理 depth scaling。
    """
    import ultralytics.nn.tasks as tasks

    custom_modules = {
        "SFEBlock": SFEBlock,
        "SFEBlockN": SFEBlockN,
        "SA": SA,
        "LODConv": LODConv,
        "SFEModule": SFEModule,
        "SFEModuleN": SFEModuleN,
    }

    for name, module in custom_modules.items():
        tasks.__dict__[name] = module

    # 重写 parse_model 以将我们的模块加入 base_modules/repeat_modules frozenset
    patched = _patch_parse_model_for_sfe(tasks)
    if patched:
        print("  ✓ parse_model frozensets 已更新 (SFEBlock/SFEBlockN 加入 base/repeat_modules)")
    else:
        print("  ✓ parse_model 已包含 SFE 模块 (无需重复 patch)")

    return custom_modules
