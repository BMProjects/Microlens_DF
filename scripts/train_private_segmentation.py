#!/usr/bin/env python3
"""
Phase 3.3: 私有数据分割模型微调
=================================
从 MSD 预训练的 LightUNet 出发，在私有弱标签数据上微调。

预训练权重: output/experiments/phase3_segmentation/msd_trained/best.pt
弱标签数据: output/experiments/phase3_segmentation/private_weak_masks/

class_weights 侧重 scratch (class 1)，因为这是 scratch AP 提升的关键。

用法:
    python scripts/train_private_segmentation.py
    python scripts/train_private_segmentation.py --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MSD_WEIGHTS = PROJECT_ROOT / "output/experiments/phase3_segmentation/msd_trained/best.pt"
WEAK_MASKS  = PROJECT_ROOT / "output/experiments/phase3_segmentation/private_weak_masks"
OUTPUT_DIR  = PROJECT_ROOT / "output/experiments/phase3_segmentation/private_finetuned"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.3: 私有数据分割微调",
    )
    parser.add_argument("--pretrained", type=Path, default=MSD_WEIGHTS,
                        help="MSD 预训练权重")
    parser.add_argument("--data-dir", type=Path, default=WEAK_MASKS,
                        help="弱标签数据目录")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 3.3: 私有数据分割微调                          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 检查预训练权重
    if not args.pretrained.exists():
        print(f"  ✗ MSD 预训练权重不存在: {args.pretrained}")
        print("    请先运行: python scripts/train_msd_segmentation.py")
        sys.exit(1)

    # 检查数据
    img_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        print(f"  ✗ 弱标签数据不存在: {args.data_dir}")
        print("    请先运行: python scripts/generate_weak_masks.py")
        sys.exit(1)

    n_images = len(list(img_dir.glob("*.png")))
    print(f"  MSD 预训练权重: {args.pretrained}")
    print(f"  私有弱标签: {n_images} 张图")
    print(f"  LR: {args.lr} (低于 MSD 训练的 1e-3)")
    print(f"  Epochs: {args.epochs}")
    print()

    import torch
    from darkfield_defects.data.dataset import DefectPatchDataset
    from darkfield_defects.data.dataset import NUM_CLASSES
    from darkfield_defects.ml.models import CombinedLoss
    from darkfield_defects.ml.segmentation_factory import build_segmentation_model, spec_from_checkpoint
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 加载 MSD 预训练权重
    print(f"  加载预训练模型 ...", end="", flush=True)
    ckpt = torch.load(str(args.pretrained), map_location="cpu", weights_only=False)
    model_spec = spec_from_checkpoint(ckpt)
    model = build_segmentation_model(model_spec)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(" ✓")

    # 数据集
    dataset = DefectPatchDataset(
        image_dir=str(img_dir),
        mask_dir=str(mask_dir),
        patch_size=args.patch_size,
        patches_per_image=8,
        augment=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"  数据: {len(dataset)} patches ({len(dataset.pairs)} images)")

    # Loss — scratch 加权
    criterion = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        class_weights=[0.1, 2.0, 1.0, 1.0],  # 强调 scratch
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = args.output_dir / "best.pt"
    history = []
    num_batches = len(loader)
    log_interval = max(1, num_batches // 5)

    print(f"\n  开始训练 ({args.epochs} epochs) ...")
    t0 = time.time()

    try:
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(loader, start=1):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
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
                    overall_elapsed = time.time() - t0
                    completed_epochs = epoch + step / num_batches
                    projected_total = (
                        overall_elapsed / max(completed_epochs, 1e-6) * args.epochs
                        if args.max_minutes is None
                        else args.max_minutes * 60
                    )
                    overall_eta = max(0.0, projected_total - overall_elapsed)
                    print(
                        f"    Epoch {epoch+1}/{args.epochs} | step {step}/{num_batches} "
                        f"({step / num_batches:.0%}) | loss={running_loss / step:.4f} | "
                        f"epoch_eta={fmt_eta(epoch_eta)} | overall_eta={fmt_eta(overall_eta)}"
                    )

            scheduler.step()
            avg_loss = running_loss / len(loader)

            history.append({
                "epoch": epoch + 1,
                "loss": round(avg_loss, 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
            })

            print(f"    Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "model_name": model_spec.model_name,
                    "in_channels": 1,  # 私有数据始终为灰度单通道
                    "num_classes": NUM_CLASSES,
                    "base_features": model_spec.base_features,
                    "encoder_name": model_spec.encoder_name,
                    "encoder_weights": model_spec.encoder_weights,
                }, best_path)

            if args.max_minutes is not None and (time.time() - t0) >= args.max_minutes * 60:
                print(f"    达到固定训练预算 {args.max_minutes} 分钟，提前停止")
                break

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  训练被中断，已运行 {fmt_time(elapsed)}")
        sys.exit(0)

    elapsed = time.time() - t0

    # 保存历史
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print()
    print("═" * 56)
    print(f"  Phase 3.3: 私有分割微调完成")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  权重: {best_path}")
    print("═" * 56)
    print()
    print("  下一步: 可视化检查分割质量")
    print("    用 infer_full_pipeline.py --use-segmentation 测试")
    print()


if __name__ == "__main__":
    main()
