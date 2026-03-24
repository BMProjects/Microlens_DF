"""轻量 U-Net 语义分割模型 — 多类缺陷检测."""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required: pip install darkfield-defects[ml]")


class ConvBlock(nn.Module):
    """双卷积块: Conv-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightUNet(nn.Module):
    """轻量 U-Net 用于多类缺陷语义分割.

    Architecture:
        Encoder: 4 层下采样 (64→128→256→512)
        Bottleneck: 512→512
        Decoder: 4 层上采样 + skip connections
        Output: num_classes 通道 (background + scratch + spot + damage)

    Args:
        in_channels: 输入通道数 (1=仅灰度, 2=灰度+候选图).
        num_classes: 输出类别数 (默认 4).
        base_features: 第一层特征数 (默认 64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_features: int = 64,
    ):
        super().__init__()
        f = base_features

        # ── Encoder ──
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.2)

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(f * 8, f * 8)

        # ── Decoder ──
        self.up4 = nn.ConvTranspose2d(f * 8, f * 8, 2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        # ── Head ──
        self.head = nn.Conv2d(f, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: (B, C, H, W) 输入张量.

        Returns:
            (B, num_classes, H, W) logits (未经 softmax).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.dropout(self.pool(e3)))

        # Bottleneck
        b = self.bottleneck(self.dropout(self.pool(e4)))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理: 返回类别预测.

        Args:
            x: (B, C, H, W).

        Returns:
            (B, H, W) 类别 ID.
        """
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=1)


class DiceLoss(nn.Module):
    """多类 Dice Loss."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W).
            targets: (B, H, W) class IDs.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets
        targets_oh = F.one_hot(targets.long(), num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Dice per class
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # 排除 background (class 0) 的 dice, 避免主导 loss
        return 1 - dice[1:].mean()


class CombinedLoss(nn.Module):
    """Dice + Focal Loss 组合, 处理极度类不平衡."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dice
        d_loss = self.dice_loss(logits, targets)

        # Focal Loss
        ce_kwargs: dict = {}
        if self.class_weights is not None:
            ce_kwargs["weight"] = torch.tensor(
                self.class_weights, device=logits.device, dtype=logits.dtype
            )
        ce = F.cross_entropy(logits, targets.long(), reduction="none", **ce_kwargs)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.focal_gamma * ce).mean()

        return self.dice_weight * d_loss + self.focal_weight * focal
