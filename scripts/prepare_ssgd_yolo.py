#!/usr/bin/env python3
"""
SSGD 数据集 → YOLO 检测格式转换
=================================
将 SSGD 的 COCO JSON 标注转换为 YOLO txt 格式，并将 7 类映射到 3 类
（scratch / spot / critical），与私有数据集对齐。

使用 lb201 的 fold1 划分 (train1.json + val1.json) 作为 train/val。

输出:
    output/experiments/phase1_ssgd_pretrain/ssgd_yolo/
        images/train/  images/val/
        labels/train/  labels/val/
        ssgd_defects.yaml

用法:
    python scripts/prepare_ssgd_yolo.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SSGD_ROOT    = Path("/home/bm/Data/SSGD")
OUTPUT_DIR   = PROJECT_ROOT / "output/experiments/phase1_ssgd_pretrain/ssgd_yolo"

# SSGD 7 类 → 私有 3 类映射（按名称，避免 lb101/lb201 category_id 差异）
SSGD_TO_YOLO = {
    "crack":           0,  # scratch
    "scratch":         0,  # scratch
    "spot":            1,  # spot
    "blot":            1,  # spot
    "broken":          2,  # critical
    "light-leakage":   2,  # critical
    "broken-membrane": 2,  # critical
}

YOLO_NAMES = ["scratch", "spot", "critical"]


def convert_coco_split(
    coco_json_path: Path,
    image_source_dir: Path,
    split: str,
) -> dict:
    """将单个 COCO JSON split 转换为 YOLO 格式。

    Returns:
        stats dict with conversion counts.
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    # 构建 category_id → name 映射
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    # 构建 image_id → image_info 映射
    img_map = {img["id"]: img for img in data["images"]}

    # 按 image_id 分组标注
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("ignore", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    # 输出目录
    img_out = OUTPUT_DIR / "images" / split
    lbl_out = OUTPUT_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    stats["images"] = 0
    stats["skipped_images"] = 0

    for img_id, img_info in img_map.items():
        fname = img_info["file_name"]
        src_path = image_source_dir / fname

        if not src_path.exists():
            stats["skipped_images"] += 1
            continue

        W = img_info["width"]
        H = img_info["height"]
        anns = anns_by_image.get(img_id, [])

        # 转换标注
        yolo_lines = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name is None or cat_name not in SSGD_TO_YOLO:
                stats["skipped_anns"] += 1
                continue

            yolo_cls = SSGD_TO_YOLO[cat_name]

            # COCO bbox: [x, y, w, h] → YOLO: cx, cy, w, h (normalized)
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / W
            cy = (by + bh / 2) / H
            nw = bw / W
            nh = bh / H

            # 跳过异常框
            if nw <= 0 or nh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
                stats["skipped_anns"] += 1
                continue

            yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[f"cls_{YOLO_NAMES[yolo_cls]}"] += 1
            stats["total_anns"] += 1

        # 写标注文件（即使无标注也写空文件，作为负样本）
        stem = Path(fname).stem
        lbl_path = lbl_out / f"{stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        # 复制/软链接图像
        dst_path = img_out / fname
        if not dst_path.exists():
            os.symlink(src_path.resolve(), dst_path)

        stats["images"] += 1

    return dict(stats)


def write_dataset_yaml() -> Path:
    yaml_path = OUTPUT_DIR / "ssgd_defects.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 3\n"
        f"names: {YOLO_NAMES}\n"
    )
    return yaml_path


def main():
    print("=" * 60)
    print("  SSGD → YOLO 格式转换")
    print("=" * 60)

    # 清理旧输出
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # 使用 lb201 fold1 划分（更多标注：2257 vs 1657）
    ann_dir = SSGD_ROOT / "annotations_lb201"
    img_dir = SSGD_ROOT / "lb201"

    for split, json_name in [("train", "train1.json"), ("val", "val1.json")]:
        json_path = ann_dir / json_name
        print(f"\n[{split}] 处理 {json_path.name} ...")
        stats = convert_coco_split(json_path, img_dir, split)

        print(f"  图像: {stats.get('images', 0)} 张")
        print(f"  标注: {stats.get('total_anns', 0)} 个")
        for cls_name in YOLO_NAMES:
            print(f"    {cls_name}: {stats.get(f'cls_{cls_name}', 0)}")
        if stats.get("skipped_images", 0):
            print(f"  跳过图像: {stats['skipped_images']}")
        if stats.get("skipped_anns", 0):
            print(f"  跳过标注: {stats['skipped_anns']}")

    yaml_path = write_dataset_yaml()
    print(f"\n[完成] YOLO 数据集配置: {yaml_path}")
    print(f"       图像目录: {OUTPUT_DIR / 'images'}")
    print(f"       标签目录: {OUTPUT_DIR / 'labels'}")


if __name__ == "__main__":
    main()
