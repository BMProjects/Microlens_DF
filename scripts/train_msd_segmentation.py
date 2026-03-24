#!/usr/bin/env python3
"""
Phase 3.1: MSD 数据集分割模型预训练
=====================================
在 MSD 公开数据集上训练 LightUNet 分割模型，学习玻璃缺陷的像素级特征。
MSD 数据已通过 prepare_msd_segmentation.py 转换为灰度+class-ID 格式。

类别对齐:
  0 = background
  1 = scratch (MSD Scratch)
  2 = spot (MSD Stain + Oil)
  3 = damage (MSD 无此类，weight=0)

用法:
    python scripts/train_msd_segmentation.py
    python scripts/train_msd_segmentation.py --epochs 80 --batch-size 8

预计训练时间 (RTX 4090D): ~20 分钟 (50 epochs, 960 images)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MSD_DATA     = PROJECT_ROOT / "output/experiments/phase3_segmentation/msd_prepared"
OUTPUT_DIR   = PROJECT_ROOT / "output/experiments/phase3_segmentation/msd_trained"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.1: MSD 分割预训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patch-size",  type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--base-features", type=int, default=64)
    parser.add_argument("--model-name", default="light_unet")
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--encoder-weights", default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-metric", choices=["loss", "miou"], default="miou")
    args = parser.parse_args()

    output_dir = args.output_dir or (OUTPUT_DIR / args.model_name.lower())

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 3.1: MSD 分割模型预训练                        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 检查数据
    train_images = MSD_DATA / "images" / "train"
    train_masks  = MSD_DATA / "masks" / "train"

    if not train_images.exists() or not train_masks.exists():
        print(f"  ✗ MSD 训练数据不存在: {MSD_DATA}")
        print("    请先运行: python scripts/prepare_msd_segmentation.py")
        sys.exit(1)

    n_images = len(list(train_images.glob("*.png")))
    n_masks  = len(list(train_masks.glob("*.png")))
    print(f"  训练数据: {n_images} images, {n_masks} masks")

    # 加载 split 信息
    split_info_path = MSD_DATA / "split_info.json"
    if split_info_path.exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        print(f"  Train: {split_info['n_train']}, Val: {split_info['n_val']}")
        print(f"  类别: {split_info['class_mapping']}")

    print(f"\n  配置:")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch:  {args.batch_size}")
    print(f"    LR:     {args.lr}")
    print(f"    Patch:  {args.patch_size}×{args.patch_size}")
    print(f"    模型:   {args.model_name}")
    if args.encoder_name:
        print(f"    编码器: {args.encoder_name} ({args.encoder_weights or 'random'})")
    if args.max_minutes is not None:
        print(f"    预算:   {args.max_minutes} 分钟")
    print(f"    输出:   {output_dir}")
    print()

    # 训练
    from darkfield_defects.ml.train import train

    t0 = time.time()
    try:
        best_path = train(
            image_dir=str(train_images),
            mask_dir=str(train_masks),
            output_dir=str(output_dir),
            val_image_dir=str(MSD_DATA / "images" / "val"),
            val_mask_dir=str(MSD_DATA / "masks" / "val"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            base_features=args.base_features,
            device=args.device,
            model_name=args.model_name,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            max_minutes=args.max_minutes,
            num_workers=args.num_workers,
            save_metric=args.save_metric,
            # class_weights: damage(3) 在 MSD 中不存在，设为 0
            class_weights=[0.1, 1.0, 1.0, 0.0],
        )
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  训练被中断，已运行 {fmt_time(elapsed)}")
        sys.exit(0)

    elapsed = time.time() - t0

    print()
    print("═" * 56)
    print("  Phase 3.1: MSD 分割预训练完成")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳权重: {best_path}")
    print("═" * 56)
    print()

    # 验证: 在 val 集上快速评估
    val_images = MSD_DATA / "images" / "val"
    val_masks  = MSD_DATA / "masks" / "val"
    if val_images.exists() and val_masks.exists():
        print("  在验证集上评估 ...")
        _quick_val(best_path, val_images, val_masks, args)

    print()
    print("  下一步:")
    print("    1. 可视化检查: 抽查 5 张 val 图 mask overlay")
    print("    2. 私有数据微调: python scripts/train_private_segmentation.py")
    print()


def _quick_val(weights_path, val_images, val_masks, args):
    """在验证集上快速计算 mIoU。"""
    import numpy as np
    import torch
    from darkfield_defects.data.dataset import DefectPatchDataset, NUM_CLASSES
    from darkfield_defects.ml.segmentation_factory import build_segmentation_model, spec_from_checkpoint
    from torch.utils.data import DataLoader

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 加载模型
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model = build_segmentation_model(spec_from_checkpoint(ckpt))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 数据
    dataset = DefectPatchDataset(
        image_dir=str(val_images),
        mask_dir=str(val_masks),
        patch_size=args.patch_size,
        patches_per_image=4,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_ious = {c: [] for c in range(1, NUM_CLASSES)}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            for c in range(1, NUM_CLASSES):
                pred_c = (preds == c)
                gt_c = (masks == c)
                inter = np.logical_and(pred_c, gt_c).sum()
                union = np.logical_or(pred_c, gt_c).sum()
                if union > 0:
                    all_ious[c].append(inter / union)

    from darkfield_defects.data.dataset import CLASS_NAMES
    print(f"  验证集 IoU:")
    valid_ious = []
    for c in range(1, NUM_CLASSES):
        ious = all_ious[c]
        if ious:
            mean_iou = np.mean(ious)
            valid_ious.append(mean_iou)
            print(f"    {CLASS_NAMES[c]:<10}: {mean_iou:.4f} ({len(ious)} patches)")
        else:
            print(f"    {CLASS_NAMES[c]:<10}: N/A (无该类样本)")

    if valid_ious:
        print(f"    mIoU:       {np.mean(valid_ious):.4f}")


if __name__ == "__main__":
    main()
