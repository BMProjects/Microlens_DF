#!/usr/bin/env python3
"""批量缺陷检测标注 — 对预处理后的数据集运行检测并输出可视化结果.

用法:
    python scripts/run_detection_batch.py [--dataset-dir DIR] [--output-dir DIR] [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 确保项目路径可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.params import DetectionParams, load_params
from darkfield_defects.detection.rendering import (
    export_coco,
    export_metadata_csv,
    export_metadata_jsonl,
    render_overlay,
    render_summary_panel,
    DEFECT_COLORS,
    DEFECT_SHORT,
)
from darkfield_defects.detection.base import DefectType


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量缺陷检测标注")
    p.add_argument(
        "--dataset-dir",
        default="output/dataset_v2",
        help="预处理数据集目录（含 images/ 和 roi_mask.png）",
    )
    p.add_argument(
        "--output-dir",
        default="output/detection_results",
        help="检测结果输出目录",
    )
    p.add_argument(
        "--config",
        default="configs/detect_default.yaml",
        help="检测参数配置文件",
    )
    p.add_argument(
        "--save-mask",
        action="store_true",
        help="同时保存二值缺陷掩码",
    )
    return p.parse_args()


def make_legend(h: int = 120, w: int = 400) -> np.ndarray:
    """生成缺陷类型图例."""
    legend = np.zeros((h, w, 3), dtype=np.uint8)
    legend[:] = (30, 30, 30)
    y = 25
    for dtype in DefectType:
        color = DEFECT_COLORS.get(dtype, (255, 255, 255))
        short = DEFECT_SHORT.get(dtype, "?")
        cv2.rectangle(legend, (15, y - 12), (35, y + 4), color, -1)
        label = f"{short} = {dtype.value}"
        cv2.putText(legend, label, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
    return legend


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    params = load_params(args.config)
    det_params = params.detection
    scoring_params = params.scoring

    # 加载 ROI 掩码
    roi_path = dataset_dir / "roi_mask.png"
    if not roi_path.exists():
        print(f"错误: 未找到 ROI 掩码: {roi_path}")
        sys.exit(1)
    roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) > 0

    # 收集图像
    img_dir = dataset_dir / "images"
    image_files = sorted(img_dir.glob("*.png"))
    if not image_files:
        print(f"错误: 未找到图像: {img_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  暗场镜片缺陷检测标注")
    print("=" * 70)
    print(f"  数据集    : {dataset_dir}")
    print(f"  图像数量  : {len(image_files)}")
    print(f"  输出目录  : {output_dir}")
    print(f"  阈值方法  : {det_params.threshold_method}")
    print(f"  合并方法  : {det_params.merge_method}")
    print(f"  Prominence: {det_params.prominence_min_value}")
    print(f"  增强      : gamma={det_params.enhance_gamma} CLAHE={'ON' if det_params.clahe_enabled else 'OFF'} (clip={det_params.clahe_clip_limit})")
    print(f"  增强开关  : {'ON' if det_params.enhance_enabled else 'OFF'}")
    print()

    # 创建子目录
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    # 创建检测器
    detector = ClassicalDetector(det_params, scoring_params=scoring_params)

    # 批量处理
    all_results: list[tuple[str, any]] = []
    stats = {
        "total": len(image_files),
        "total_defects": 0,
        "by_type": {t.value: 0 for t in DefectType},
        "times": [],
    }

    t_start = time.perf_counter()

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [跳过] {img_path.name}: 无法读取")
            continue

        t0 = time.perf_counter()
        result = detector.detect(
            image=img,
            roi_mask=roi_mask,
            preprocessed_image=img,
        )
        dt = time.perf_counter() - t0
        stats["times"].append(dt)

        # 统计
        n_total = result.num_defects
        stats["total_defects"] += n_total
        for inst in result.instances:
            stats["by_type"][inst.defect_type.value] += 1

        # 保存叠加图
        overlay = render_overlay(img, result, alpha=0.5, show_skeleton=True, show_zones=False)
        cv2.imwrite(str(overlay_dir / f"{stem}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 保存掩码
        if args.save_mask:
            cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), result.mask)

        all_results.append((img_path.name, result))

        # 进度
        type_str = " ".join(
            f"{DEFECT_SHORT.get(t, '?')}={sum(1 for i in result.instances if i.defect_type == t)}"
            for t in DefectType
            if sum(1 for i in result.instances if i.defect_type == t) > 0
        )
        if not type_str:
            type_str = "无缺陷"
        pct = idx * 100 // len(image_files)
        print(f"  [{idx:3d}/{len(image_files)}] {stem:20s}  {n_total:3d} defects  {type_str:30s}  {dt*1000:.0f}ms  ({pct}%)")

    t_total = time.perf_counter() - t_start

    # 导出标注
    print()
    print("[导出] COCO JSON ...")
    export_coco(all_results, output_dir / "annotations_coco.json")

    print("[导出] CSV 元数据 ...")
    export_metadata_csv(all_results, output_dir / "defects_metadata.csv")

    print("[导出] JSONL 元数据 ...")
    export_metadata_jsonl(all_results, output_dir / "defects_metadata.jsonl")

    # 生成汇总统计
    summary = {
        "total_images": stats["total"],
        "total_defects": stats["total_defects"],
        "defects_per_image": round(stats["total_defects"] / max(stats["total"], 1), 1),
        "by_type": stats["by_type"],
        "avg_time_ms": round(np.mean(stats["times"]) * 1000, 0) if stats["times"] else 0,
        "total_time_s": round(t_total, 1),
        "config": {
            "threshold_method": det_params.threshold_method,
            "merge_method": det_params.merge_method,
            "prominence_min_value": det_params.prominence_min_value,
            "clahe_enabled": det_params.clahe_enabled,
            "min_area": det_params.min_area,
        },
    }

    with open(output_dir / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成缩略图汇总网格 — 取前24张
    print("[汇总] 生成缩略图网格 ...")
    thumb_w, thumb_h = 640, 468
    cols, rows = 4, 6
    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    grid[:] = 20

    for i, (fname, res) in enumerate(all_results[:rows * cols]):
        r, c = divmod(i, cols)
        img = cv2.imread(str(img_dir / fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ov = render_overlay(img, res, alpha=0.5, show_skeleton=False, show_zones=False)
        thumb = cv2.resize(ov, (thumb_w, thumb_h))

        # 标注文件名和缺陷数
        label = f"{Path(fname).stem}  ({res.num_defects})"
        cv2.putText(thumb, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y0 = r * thumb_h
        x0 = c * thumb_w
        grid[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

    cv2.imwrite(str(output_dir / "summary_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 打印汇总
    print()
    print("=" * 70)
    print(f"  完成: {stats['total']} 张图像   总耗时 {t_total:.1f}s")
    print()
    print(f"  总缺陷数       : {stats['total_defects']}")
    print(f"  平均每张       : {summary['defects_per_image']}")
    for t in DefectType:
        cnt = stats["by_type"][t.value]
        if cnt > 0:
            print(f"    {t.value:10s}   : {cnt}")
    print()
    print(f"  平均耗时       : {summary['avg_time_ms']:.0f} ms/张")
    print()
    print(f"  输出目录       : {output_dir}")
    print(f"    overlays/    : {len(image_files)} 张标注叠加图")
    print(f"    annotations_coco.json")
    print(f"    defects_metadata.csv")
    print(f"    defects_metadata.jsonl")
    print(f"    detection_summary.json")
    print(f"    summary_grid.jpg")
    print("=" * 70)


if __name__ == "__main__":
    main()
