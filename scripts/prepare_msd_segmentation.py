#!/usr/bin/env python3
"""
MSD 数据集 → 分割训练格式准备
================================
将 MSD 的彩色图像转灰度，重编码 mask（binary→class ID），创建 80/20 分层 split。

MSD mask 实际编码: 3通道 PNG, 0=background, 128=defect (binary per-class)
缺陷类型由文件前缀决定: Scr_=scratch, Sta_=stain, Oil_=oil

重编码映射（对齐 LightUNet 4 类: bg/scratch/spot/damage）:
    background(0)  → 0 (bg)
    Scratch(128)   → 1 (scratch)
    Stain(128)     → 2 (spot)
    Oil(128)       → 2 (spot)

输出:
    output/experiments/phase3_segmentation/msd_prepared/
        images/train/  images/val/   (灰度 PNG)
        masks/train/   masks/val/    (单通道 class-ID PNG)
        split_info.json

用法:
    python scripts/prepare_msd_segmentation.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MSD_ROOT     = Path("/home/bm/Data/MSD")
OUTPUT_DIR   = PROJECT_ROOT / "output/experiments/phase3_segmentation/msd_prepared"

# 类别映射: 文件前缀 → (源mask目录, 目标class_id)
CLASS_MAP = {
    "Scr": ("ground_truth_1", 1),   # scratch → class 1
    "Sta": ("ground_truth_1", 2),   # stain → class 2 (spot)
    "Oil": ("ground_truth_2", 2),   # oil → class 2 (spot)
}

# 对应图像目录
IMG_DIRS = {
    "Scr": "scratch",
    "Sta": "stain",
    "Oil": "oil",
}

SPLIT_RATIO = 0.8  # 80% train


def process_class(
    prefix: str,
    gt_dir_name: str,
    class_id: int,
    img_dir_name: str,
) -> list[dict]:
    """处理单个类别，返回样本信息列表。"""
    gt_dir  = MSD_ROOT / gt_dir_name
    img_dir = MSD_ROOT / img_dir_name

    masks = sorted(gt_dir.glob(f"{prefix}_*.png"))
    samples = []

    for mask_path in masks:
        stem = mask_path.stem
        # 查找对应图像 (JPG)
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.JPG"
        if not img_path.exists():
            continue

        samples.append({
            "stem": stem,
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "class_id": class_id,
            "prefix": prefix,
        })

    return samples


def convert_and_save(
    sample: dict,
    split: str,
) -> None:
    """转换单个样本：图像转灰度，mask 重编码为 class ID。"""
    img_out_dir  = OUTPUT_DIR / "images" / split
    mask_out_dir = OUTPUT_DIR / "masks" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    stem = sample["stem"]

    # 图像 → 灰度
    img = cv2.imread(sample["img_path"])
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(img_out_dir / f"{stem}.png"), gray)

    # Mask: 3通道 RGB, 每通道 0 或 128 → 单通道 class ID
    # 不同通道可能编码不同子区域，需检测任意通道非零
    raw_mask = cv2.imread(sample["mask_path"], cv2.IMREAD_COLOR)
    if raw_mask is None:
        return

    # 任意通道 > 0 即为缺陷区域
    defect_mask = np.any(raw_mask > 0, axis=2)
    class_mask = np.zeros(raw_mask.shape[:2], dtype=np.uint8)
    class_mask[defect_mask] = sample["class_id"]
    cv2.imwrite(str(mask_out_dir / f"{stem}.png"), class_mask)


def main():
    print("=" * 60)
    print("  MSD → 分割训练格式准备")
    print("=" * 60)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    all_samples = []
    for prefix, (gt_dir, class_id) in CLASS_MAP.items():
        samples = process_class(prefix, gt_dir, class_id, IMG_DIRS[prefix])
        all_samples.extend(samples)
        print(f"  {prefix} ({IMG_DIRS[prefix]}): {len(samples)} 样本 → class {class_id}")

    print(f"\n  总样本: {len(all_samples)}")

    # 分层 split: 每个前缀内部 80/20 划分
    rng = np.random.default_rng(42)
    split_info = {"train": [], "val": []}

    for prefix in ["Scr", "Sta", "Oil"]:
        cls_samples = [s for s in all_samples if s["prefix"] == prefix]
        indices = rng.permutation(len(cls_samples))
        n_train = int(len(cls_samples) * SPLIT_RATIO)

        for i, idx in enumerate(indices):
            split = "train" if i < n_train else "val"
            sample = cls_samples[idx]
            convert_and_save(sample, split)
            split_info[split].append(sample["stem"])

    # 保存 split 信息
    info = {
        "n_train": len(split_info["train"]),
        "n_val": len(split_info["val"]),
        "class_mapping": {
            "0": "background",
            "1": "scratch",
            "2": "spot (stain+oil)",
        },
        "source": str(MSD_ROOT),
        "splits": split_info,
    }
    info_path = OUTPUT_DIR / "split_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n  Train: {info['n_train']} 样本")
    print(f"  Val:   {info['n_val']} 样本")
    print(f"\n[完成] 输出目录: {OUTPUT_DIR}")
    print(f"       Split 信息: {info_path}")


if __name__ == "__main__":
    main()
