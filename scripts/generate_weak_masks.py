#!/usr/bin/env python3
"""
Phase 3.2: 为私有数据生成弱分割标签
======================================
利用 YOLO GT 标注 + Frangi 经典检测器生成像素级弱标签 mask，
用于后续分割模型微调。

算法:
  1. 对每张私有全图运行 ClassicalDetector，获取 Frangi 候选图 (连续响应)
  2. 从 YOLO 切片标注反向映射回全图坐标 (tile → full image)
  3. 在每个 GT bbox 区域内，用 Frangi 响应 > 阈值的像素作为 mask
  4. GT bbox 外的 Frangi 响应丢弃（可能是 FP）
  5. 输出: 多类 mask (0=bg, 1=scratch, 2=spot, 3=damage)

约束:
  - 仅使用 train split 的图像（val/test GT 不可修改）
  - mask 质量为 "弱标签" — 后续可用 SAM 精修

用法:
    python scripts/generate_weak_masks.py
    python scripts/generate_weak_masks.py --output-dir output/experiments/phase3_segmentation/private_weak_masks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TILE_DIR     = PROJECT_ROOT / "output/tile_dataset"
IMAGE_DIR    = PROJECT_ROOT / "output/dataset_v2/images"
OUTPUT_DIR   = PROJECT_ROOT / "output/experiments/phase3_segmentation/private_weak_masks"

# YOLO class_id → mask class_id
# YOLO: 0=scratch, 1=spot, 2=critical → mask: 1=scratch, 2=spot, 3=damage
YOLO_TO_MASK = {0: 1, 1: 2, 2: 3}

# 切片参数 (与 build_dataset 一致)
TILE_SIZE = 640
TILE_OVERLAP = 0.2


def load_train_stems() -> list[str]:
    """从 defects.yaml 的 train split 中获取全图 stem 列表。"""
    # 从 tile 文件名反推全图 stem: {stem}_{row}_{col}.jpg
    train_label_dir = TILE_DIR / "labels" / "train"
    if not train_label_dir.exists():
        return []

    stems = set()
    for txt_path in train_label_dir.glob("*.txt"):
        # tile stem: e.g., 53r_0_2 → full image stem: 53r
        parts = txt_path.stem.rsplit("_", 2)
        if len(parts) >= 3:
            stems.add(parts[0])
    return sorted(stems)


def load_tile_annotations(full_stem: str, split: str = "train") -> list[dict]:
    """加载某张全图的所有 tile YOLO 标注，映射回全图坐标。"""
    label_dir = TILE_DIR / "labels" / split
    annotations = []

    for txt_path in sorted(label_dir.glob(f"{full_stem}_y*_x*.txt")):
        # 解析 tile 位置: 文件名格式 {stem}_y{Y:04d}_x{X:04d}，Y/X 为像素偏移
        parts = txt_path.stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        try:
            tile_y = int(parts[1].lstrip("y"))   # 'y0560' → 560
            tile_x = int(parts[2].lstrip("x"))   # 'x1120' → 1120
        except ValueError:
            continue

        # 读取 YOLO 标注
        for line in txt_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts_ann = line.split()
            cls_id = int(parts_ann[0])
            cx_norm, cy_norm, w_norm, h_norm = (float(v) for v in parts_ann[1:5])

            # 还原全图坐标 (pixel)
            cx_px = cx_norm * TILE_SIZE + tile_x
            cy_px = cy_norm * TILE_SIZE + tile_y
            w_px = w_norm * TILE_SIZE
            h_px = h_norm * TILE_SIZE

            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2

            annotations.append({
                "cls_id": cls_id,
                "bbox": [x1, y1, x2, y2],  # xyxy in full image
                "tile_stem": txt_path.stem,
            })

    return annotations


def generate_mask_for_image(
    image_path: Path,
    annotations: list[dict],
    frangi_threshold: float = 0.15,
) -> np.ndarray | None:
    """为单张全图生成弱分割 mask。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    H, W = image.shape

    # 生成 Frangi 候选图
    from darkfield_defects.detection.features import compute_candidate_map
    from darkfield_defects.detection.params import DetectionParams

    params = DetectionParams()
    candidate_map = compute_candidate_map(image, params, roi_mask=None)

    # 归一化候选图到 [0, 1]
    if candidate_map.max() > 0:
        candidate_norm = candidate_map.astype(np.float32) / candidate_map.max()
    else:
        candidate_norm = candidate_map.astype(np.float32)

    # 初始化多类 mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # 在每个 GT bbox 区域内，用 Frangi 响应生成 mask
    for ann in annotations:
        cls_id = ann["cls_id"]
        mask_cls = YOLO_TO_MASK.get(cls_id)
        if mask_cls is None:
            continue

        x1, y1, x2, y2 = ann["bbox"]
        # 安全裁剪到图像边界
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(W, int(x2)), min(H, int(y2))

        if x2i <= x1i or y2i <= y1i:
            continue

        # 在 bbox 内提取 Frangi 响应
        roi = candidate_norm[y1i:y2i, x1i:x2i]

        # 自适应阈值: Otsu + Frangi threshold 取较低者
        if roi.size > 0 and roi.max() > 0:
            roi_uint8 = (roi * 255).astype(np.uint8)
            thresh_val, _ = cv2.threshold(roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_thresh = min(frangi_threshold, thresh_val / 255.0)
        else:
            adaptive_thresh = frangi_threshold

        # 生成局部 mask (优先级: 高 class_id 覆盖低 class_id)
        local_mask = (roi > adaptive_thresh)

        # 如果 Frangi 响应太弱，用 bbox 填充
        if local_mask.sum() < 10:
            local_mask = np.ones_like(local_mask, dtype=bool)

        # 写入全局 mask (高优先级覆盖)
        current_region = mask[y1i:y2i, x1i:x2i]
        current_region[local_mask & (current_region < mask_cls)] = mask_cls

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.2: 生成弱分割标签",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--frangi-threshold", type=float, default=0.15)
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 3.2: 生成私有数据弱分割标签                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 获取 train split 的全图 stem
    stems = load_train_stems()
    if not stems:
        print("  ✗ 未找到训练切片")
        sys.exit(1)

    print(f"  全图数量: {len(stems)}")

    # 输出目录
    img_out = args.output_dir / "images"
    mask_out = args.output_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "with_defects": 0, "empty": 0, "failed": 0}
    class_pixels = {1: 0, 2: 0, 3: 0}

    for i, stem in enumerate(stems):
        # 查找对应全图
        img_path = None
        for ext in [".png", ".jpg", ".JPG", ".PNG"]:
            candidate = IMAGE_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["failed"] += 1
            continue

        # 加载标注
        annotations = load_tile_annotations(stem)
        if not annotations:
            stats["empty"] += 1
            continue

        # 生成 mask
        mask = generate_mask_for_image(img_path, annotations, args.frangi_threshold)
        if mask is None:
            stats["failed"] += 1
            continue

        # 保存灰度图 (复制或软链接)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(img_out / f"{stem}.png"), image)
        cv2.imwrite(str(mask_out / f"{stem}.png"), mask)

        # 统计
        stats["total"] += 1
        has_defect = mask.max() > 0
        if has_defect:
            stats["with_defects"] += 1
        for cls in [1, 2, 3]:
            class_pixels[cls] += int((mask == cls).sum())

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(stems)}] 已处理 {stats['total']} 张")

    # 打印统计
    print()
    print("═" * 56)
    print(f"  弱标签生成完成")
    print(f"  总图像: {stats['total']}")
    print(f"  含缺陷: {stats['with_defects']}")
    print(f"  空白:   {stats['empty']}")
    print(f"  失败:   {stats['failed']}")
    print(f"  缺陷像素:")
    for cls, px in class_pixels.items():
        cls_name = ["", "scratch", "spot", "damage"][cls]
        print(f"    {cls_name}: {px:,} px")
    print(f"  输出: {args.output_dir}")
    print("═" * 56)

    # 保存统计
    info = {**stats, "class_pixels": class_pixels}
    with open(args.output_dir / "generation_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
