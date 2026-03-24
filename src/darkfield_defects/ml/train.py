"""训练脚本 — 多类缺陷分割模型训练."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch is required: pip install darkfield-defects[ml]")

from darkfield_defects.data.dataset import DefectPatchDataset, NUM_CLASSES
from darkfield_defects.logging import get_logger
from darkfield_defects.ml.models import CombinedLoss
from darkfield_defects.ml.segmentation_factory import (
    SegmentationModelSpec,
    build_segmentation_model,
)

logger = get_logger(__name__)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path = "checkpoints",
    *,
    val_image_dir: str | Path | None = None,
    val_mask_dir: str | Path | None = None,
    candidate_dir: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-3,
    patch_size: int = 512,
    patches_per_image: int = 8,
    base_features: int = 64,
    device: str | None = None,
    class_weights: Optional[list[float]] = None,
    model_name: str = "light_unet",
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
    max_minutes: float | None = None,
    num_workers: int = 0,
    save_metric: str = "loss",
) -> Path:
    """训练多类缺陷分割模型.

    Args:
        image_dir: 图像目录.
        mask_dir: 掩码目录 (像素值=类别 ID).
        output_dir: 模型保存目录.
        val_image_dir: 验证图像目录.
        val_mask_dir: 验证掩码目录.
        candidate_dir: 可选候选图目录 (第二通道).
        epochs: 训练轮数.
        batch_size: 批大小.
        lr: 初始学习率.
        patch_size: 随机裁切尺寸.
        patches_per_image: 每张图随机采样数.
        base_features: U-Net 第一层特征数.
        device: 训练设备 (auto/cpu/cuda).
        class_weights: 类别权重 [bg, scratch, spot, damage].
        model_name: 分割模型名称.
        encoder_name: 若为 SMP 模型，所用 encoder 名称.
        encoder_weights: 若为 SMP 模型，encoder 初始化权重.
        max_minutes: 固定训练时长预算；到时后在 epoch 边界提前停止.
        num_workers: DataLoader 进程数.
        save_metric: 最佳模型保存准则，支持 loss / miou.

    Returns:
        最佳模型权重路径.
    """
    # 设备选择
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"训练设备: {dev}")

    # 数据集
    dataset = DefectPatchDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        augment=True,
        candidate_dir=candidate_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"训练集: {len(dataset)} patches ({len(dataset.pairs)} images)")

    val_loader: DataLoader | None = None
    if val_image_dir and val_mask_dir:
        val_dataset = DefectPatchDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            patch_size=patch_size,
            patches_per_image=max(1, patches_per_image // 2),
            augment=False,
            candidate_dir=candidate_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        logger.info(f"验证集: {len(val_dataset)} patches ({len(val_dataset.pairs)} images)")

    # 模型
    in_ch = dataset.in_channels
    model_spec = SegmentationModelSpec(
        model_name=model_name,
        in_channels=in_ch,
        num_classes=NUM_CLASSES,
        base_features=base_features,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )
    model = build_segmentation_model(model_spec)
    model.to(dev)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"模型: {model_name}, in_ch={in_ch}, params={n_params / 1e6:.2f}M, "
        f"encoder={encoder_name or 'n/a'}"
    )

    # 损失函数
    if class_weights is None:
        class_weights = [0.1, 1.0, 1.0, 1.0]  # 低权重 background
    criterion = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        class_weights=class_weights,
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # 输出目录
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_miou = float("-inf")
    best_path = out_dir / "best.pt"
    history: list[dict] = []
    start_time = time.time()
    num_batches = len(loader)
    log_interval = max(1, num_batches // 5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(dev)
            masks = batch["mask"].to(dev)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step == 1 or step % log_interval == 0 or step == num_batches:
                epoch_elapsed = time.time() - epoch_start
                step_rate = step / max(epoch_elapsed, 1e-6)
                epoch_eta = (num_batches - step) / max(step_rate, 1e-6)
                overall_elapsed = time.time() - start_time
                completed_epochs = epoch + step / num_batches
                projected_total = (
                    overall_elapsed / max(completed_epochs, 1e-6) * epochs
                    if max_minutes is None
                    else max_minutes * 60
                )
                overall_eta = max(0.0, projected_total - overall_elapsed)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | step {step}/{num_batches} "
                    f"({step / num_batches:.0%}) | loss={running_loss / step:.4f} | "
                    f"epoch_eta={_format_duration(epoch_eta)} | "
                    f"overall_eta={_format_duration(overall_eta)}"
                )

        scheduler.step()
        avg_loss = running_loss / len(loader)

        # 训练集快速指标
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            logits = model(sample_batch["image"].to(dev))
            preds = logits.argmax(dim=1).cpu().numpy()
            gt = sample_batch["mask"].numpy()
            train_miou = _compute_miou(preds, gt, NUM_CLASSES)

        val_miou = None
        if val_loader is not None:
            val_miou = _evaluate_loader_miou(model, val_loader, dev)

        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "train_miou": round(train_miou, 4),
            "val_miou": round(val_miou, 4) if val_miou is not None else None,
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        metric_text = (
            f"train_mIoU={train_miou:.4f}, val_mIoU={val_miou:.4f}"
            if val_miou is not None
            else f"train_mIoU={train_miou:.4f}"
        )
        logger.info(
            f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
            f"{metric_text}, lr={scheduler.get_last_lr()[0]:.6f}"
        )

        is_better = False
        if save_metric == "miou" and val_miou is not None:
            if val_miou > best_miou:
                best_miou = val_miou
                is_better = True
        elif save_metric == "loss":
            if avg_loss < best_loss:
                best_loss = avg_loss
                is_better = True
        else:
            raise ValueError("save_metric 仅支持 'loss' 或 'miou'")

        # 保存最佳模型
        if is_better:
            best_loss = min(best_loss, avg_loss)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "best_miou": best_miou if best_miou > float("-inf") else None,
                "model_name": model_name,
                "in_channels": in_ch,
                "num_classes": NUM_CLASSES,
                "base_features": base_features,
                "encoder_name": encoder_name,
                "encoder_weights": encoder_weights,
                "save_metric": save_metric,
            }, best_path)
            if save_metric == "miou" and val_miou is not None:
                logger.info(f"  ✓ 最佳模型已保存: {best_path} (val_mIoU={val_miou:.4f})")
            else:
                logger.info(f"  ✓ 最佳模型已保存: {best_path} (loss={avg_loss:.4f})")

        if max_minutes is not None and (time.time() - start_time) >= max_minutes * 60:
            logger.info(f"达到固定训练预算 {max_minutes} 分钟，提前停止。")
            break

    # 保存最终模型和训练历史
    torch.save(model.state_dict(), out_dir / "final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"训练完成! 最佳 loss={best_loss:.4f}, 权重: {best_path}")
    return best_path


def _compute_miou(preds: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """计算 mean IoU (忽略 background)."""
    ious = []
    for c in range(1, num_classes):  # 跳过 background
        pred_c = (preds == c)
        gt_c = (gt == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _evaluate_loader_miou(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """在给定 loader 上计算平均 mIoU."""
    ious: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(dev))
            preds = logits.argmax(dim=1).cpu().numpy()
            gt = batch["mask"].numpy()
            ious.append(_compute_miou(preds, gt, NUM_CLASSES))
    return float(np.mean(ious)) if ious else 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练多类缺陷分割模型")
    parser.add_argument("--images", required=True, help="图像目录")
    parser.add_argument("--masks", required=True, help="掩码目录")
    parser.add_argument("--output", default="checkpoints", help="输出目录")
    parser.add_argument("--candidates", default=None, help="候选图目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    train(
        image_dir=args.images,
        mask_dir=args.masks,
        output_dir=args.output,
        candidate_dir=args.candidates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
