#!/usr/bin/env python3
"""
Scratch 假阴性 (FN) 案例分析与难例挖掘
=========================================
分析基线模型在验证集上漏检的 scratch 案例，识别特征规律，
生成难例 tile 列表供后续标注质量改善或 hard example mining 使用。

用法:
    python scripts/analyze_scratch_fn.py
    python scripts/analyze_scratch_fn.py --conf 0.20 --iou-match 0.3

输出:
    output/experiments/phase3b_fn_analysis/
        fn_report.json          — 汇总统计
        fn_tiles.csv            — 每张 tile 的 TP/FP/FN 计数
        hard_negatives.txt      — 漏检率最高的全图 stem 列表
        fn_bbox_stats.csv       — FN bbox 的几何特征分布
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TILE_LABEL_DIR = PROJECT_ROOT / "output/tile_dataset/labels/val"
TILE_IMAGE_DIR = PROJECT_ROOT / "output/tile_dataset/images/val"
BASELINE_WEIGHTS = PROJECT_ROOT / "output/training/stage2_cleaned/weights/best.pt"
OUTPUT_DIR = PROJECT_ROOT / "output/experiments/phase3b_fn_analysis"

TILE_SIZE = 640
SCRATCH_CLS = 0


def parse_yolo_labels(txt_path: Path) -> list[tuple]:
    """解析 YOLO txt 标注，返回 [(cls, cx, cy, w, h)] (归一化坐标)。"""
    if not txt_path.exists():
        return []
    labels = []
    for line in txt_path.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        labels.append((cls, cx, cy, w, h))
    return labels


def iou_xyxy(b1: np.ndarray, b2: np.ndarray) -> float:
    """计算两个 [x1,y1,x2,y2] bbox 的 IoU。"""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy(cx, cy, w, h, img_size=TILE_SIZE):
    """归一化 cxcywh → 像素 xyxy。"""
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return np.array([x1, y1, x2, y2])


def match_predictions(gt_boxes: list, pred_boxes: list, iou_thresh: float = 0.3):
    """将预测框与 GT 框匹配，返回 (tp_gt_mask, fp_pred_mask, fn_gt_mask)。"""
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0:
        return np.array([], dtype=bool), np.ones(n_pred, dtype=bool), np.array([], dtype=bool)
    if n_pred == 0:
        return np.zeros(n_gt, dtype=bool), np.array([], dtype=bool), np.ones(n_gt, dtype=bool)

    matched_gt = np.zeros(n_gt, dtype=bool)
    matched_pred = np.zeros(n_pred, dtype=bool)

    # 贪心匹配（按 IoU 降序）
    ious = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ious[i, j] = iou_xyxy(g, p)

    while True:
        best = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[best] < iou_thresh:
            break
        matched_gt[best[0]] = True
        matched_pred[best[1]] = True
        ious[best[0], :] = -1
        ious[:, best[1]] = -1

    tp_gt_mask = matched_gt
    fn_gt_mask = ~matched_gt
    fp_pred_mask = ~matched_pred
    return tp_gt_mask, fp_pred_mask, fn_gt_mask


def run_inference_on_val(weights: Path, conf: float, save_dir: Path) -> Path:
    """使用 ultralytics val 生成 per-tile 预测 txt，返回 pred_label_dir。"""
    from ultralytics import YOLO

    val_yaml = PROJECT_ROOT / "output/tile_dataset/defects.yaml"

    print(f"  运行验证推理 (conf={conf}) ...")
    model = YOLO(str(weights))
    results = model.val(
        data=str(val_yaml),
        split="val",
        conf=conf,
        iou=0.6,
        save_txt=True,
        save_conf=True,
        project=str(save_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    pred_label_dir = save_dir / "predictions" / "labels"
    if not pred_label_dir.exists():
        raise RuntimeError(f"预测标注目录不存在: {pred_label_dir}")
    print(f"  ✓ 预测完成: {len(list(pred_label_dir.glob('*.txt')))} 张 tile")
    return pred_label_dir


def analyze(gt_label_dir: Path, pred_label_dir: Path, iou_thresh: float) -> dict:
    """对比 GT 与预测，计算 FN 统计，返回分析结果。"""
    tile_results = []
    fn_bbox_stats = []

    gt_tiles = set(p.stem for p in gt_label_dir.glob("*.txt"))
    print(f"  GT tiles: {len(gt_tiles)}")

    n_tp = n_fp = n_fn = 0

    for stem in sorted(gt_tiles):
        gt_labels = parse_yolo_labels(gt_label_dir / f"{stem}.txt")
        pred_labels = parse_yolo_labels(pred_label_dir / f"{stem}.txt") if (pred_label_dir / f"{stem}.txt").exists() else []

        # 仅分析 scratch (cls=0)
        gt_scratch  = [xywhn_to_xyxy(cx, cy, w, h) for cls, cx, cy, w, h in gt_labels  if cls == SCRATCH_CLS]
        pred_scratch = [xywhn_to_xyxy(cx, cy, w, h) for cls, cx, cy, w, h in pred_labels if cls == SCRATCH_CLS]

        if not gt_scratch:
            continue

        tp_mask, fp_mask, fn_mask = match_predictions(gt_scratch, pred_scratch, iou_thresh)

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        n_tp += tp; n_fp += fp; n_fn += fn

        fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        tile_results.append({
            "stem": stem,
            "full_stem": "_".join(stem.split("_")[:-2]),  # 去掉 y/x 坐标
            "n_gt": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "fn_rate": round(fn_rate, 3),
        })

        # 收集 FN bbox 特征
        for i, is_fn in enumerate(fn_mask):
            if not is_fn:
                continue
            b = gt_scratch[i]
            w_px = b[2] - b[0]
            h_px = b[3] - b[1]
            area = w_px * h_px
            aspect = max(w_px, h_px) / (min(w_px, h_px) + 1e-6)
            fn_bbox_stats.append({
                "stem": stem,
                "w": round(w_px, 1), "h": round(h_px, 1),
                "area": round(area, 1),
                "aspect_ratio": round(aspect, 2),
                "cx": round((b[0] + b[2]) / 2, 1),
                "cy": round((b[1] + b[3]) / 2, 1),
            })

    # 全图级别汇总
    full_stem_stats = {}
    for t in tile_results:
        fs = t["full_stem"]
        if fs not in full_stem_stats:
            full_stem_stats[fs] = {"n_gt": 0, "fn": 0, "n_tiles": 0}
        full_stem_stats[fs]["n_gt"]    += t["n_gt"]
        full_stem_stats[fs]["fn"]      += t["fn"]
        full_stem_stats[fs]["n_tiles"] += 1

    hard_negatives = sorted(
        [(fs, s["fn"] / s["n_gt"] if s["n_gt"] > 0 else 0.0)
         for fs, s in full_stem_stats.items()],
        key=lambda x: -x[1]
    )

    # FN bbox 几何统计
    if fn_bbox_stats:
        areas    = [s["area"]         for s in fn_bbox_stats]
        aspects  = [s["aspect_ratio"] for s in fn_bbox_stats]
        fn_geo = {
            "count":           len(fn_bbox_stats),
            "area_mean":       round(float(np.mean(areas)), 1),
            "area_median":     round(float(np.median(areas)), 1),
            "area_p5":         round(float(np.percentile(areas, 5)), 1),
            "area_p95":        round(float(np.percentile(areas, 95)), 1),
            "aspect_mean":     round(float(np.mean(aspects)), 2),
            "aspect_median":   round(float(np.median(aspects)), 2),
            "aspect_p95":      round(float(np.percentile(aspects, 95)), 2),
            "tiny_lt300px2":   int(sum(1 for a in areas if a < 300)),
            "ultrathin_ar_gt8": int(sum(1 for a in aspects if a > 8)),
        }
    else:
        fn_geo = {}

    return {
        "summary": {
            "total_gt_scratch": n_tp + n_fn,
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "recall": round(n_tp / (n_tp + n_fn), 4) if (n_tp + n_fn) > 0 else 0.0,
            "precision": round(n_tp / (n_tp + n_fp), 4) if (n_tp + n_fp) > 0 else 0.0,
        },
        "fn_geometry": fn_geo,
        "hard_negatives_top20": hard_negatives[:20],
        "tile_results": tile_results,
        "fn_bbox_stats": fn_bbox_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Scratch FN 分析")
    parser.add_argument("--weights", type=str, default=str(BASELINE_WEIGHTS))
    parser.add_argument("--conf",    type=float, default=0.20,
                        help="推理置信度（较低以减少FN）")
    parser.add_argument("--iou-match", type=float, default=0.30,
                        help="GT 与预测匹配的 IoU 阈值")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理（复用已有预测结果）")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Scratch FN 分析 & 难例挖掘                          ║")
    print("╚══════════════════════════════════════════════════════╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)

    # Step 1: 生成预测
    pred_label_dir = OUTPUT_DIR / "predictions" / "labels"
    if args.skip_inference and pred_label_dir.exists():
        print(f"  复用已有预测: {pred_label_dir}")
    else:
        pred_label_dir = run_inference_on_val(weights, args.conf, OUTPUT_DIR)

    # Step 2: FN 分析
    print(f"  分析 FN (iou_match={args.iou_match}) ...")
    results = analyze(TILE_LABEL_DIR, pred_label_dir, args.iou_match)
    s = results["summary"]
    print(f"\n  === 汇总 (scratch only) ===")
    print(f"  GT boxes:  {s['total_gt_scratch']}")
    print(f"  TP: {s['tp']}  FP: {s['fp']}  FN: {s['fn']}")
    print(f"  Recall:    {s['recall']:.4f}  (漏检率: {1-s['recall']:.2%})")
    print(f"  Precision: {s['precision']:.4f}")

    if results["fn_geometry"]:
        g = results["fn_geometry"]
        print(f"\n  === 漏检 bbox 几何特征 ===")
        print(f"  FN 数量:           {g['count']}")
        print(f"  面积 (px²): mean={g['area_mean']:.0f}  median={g['area_median']:.0f}  p5={g['area_p5']:.0f}  p95={g['area_p95']:.0f}")
        print(f"  长宽比:     mean={g['aspect_mean']:.1f}  median={g['aspect_median']:.1f}  p95={g['aspect_p95']:.1f}")
        print(f"  tiny (<300px²):   {g['tiny_lt300px2']}  ({100*g['tiny_lt300px2']//max(g['count'],1)}%)")
        print(f"  ultra-thin (>8):  {g['ultrathin_ar_gt8']}  ({100*g['ultrathin_ar_gt8']//max(g['count'],1)}%)")

    print(f"\n  === 漏检率最高的全图 (Top 10) ===")
    for stem, fn_rate in results["hard_negatives_top20"][:10]:
        print(f"    {stem:<15}  FN率 {fn_rate:.1%}")

    # Step 3: 保存输出
    # fn_report.json
    report = {
        "weights": str(weights),
        "conf": args.conf,
        "iou_match": args.iou_match,
        "summary": results["summary"],
        "fn_geometry": results["fn_geometry"],
        "hard_negatives_top20": results["hard_negatives_top20"],
    }
    (OUTPUT_DIR / "fn_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    # fn_tiles.csv
    with open(OUTPUT_DIR / "fn_tiles.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "full_stem", "n_gt", "tp", "fp", "fn", "fn_rate"])
        writer.writeheader()
        writer.writerows(sorted(results["tile_results"], key=lambda x: -x["fn_rate"]))

    # fn_bbox_stats.csv
    if results["fn_bbox_stats"]:
        with open(OUTPUT_DIR / "fn_bbox_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "cx", "cy", "w", "h", "area", "aspect_ratio"])
            writer.writeheader()
            writer.writerows(results["fn_bbox_stats"])

    # hard_negatives.txt
    hn_stems = [stem for stem, _ in results["hard_negatives_top20"]]
    (OUTPUT_DIR / "hard_negatives.txt").write_text("\n".join(hn_stems))

    print(f"\n  输出已保存: {OUTPUT_DIR}/")
    print(f"    fn_report.json     — 汇总统计")
    print(f"    fn_tiles.csv       — 每张 tile 的 TP/FP/FN")
    print(f"    fn_bbox_stats.csv  — FN bbox 几何特征")
    print(f"    hard_negatives.txt — 高漏检率全图 stem 列表")
    print()
    print("  下一步建议:")
    print("    1. 查看 fn_tiles.csv 高 FN 率 tile，手工检查标注质量")
    print("    2. 用 hard_negatives.txt 做 hard example re-weighting 或额外增强")
    print("    3. 对比 FN bbox 几何 vs 全量 scratch 几何，找分布差异")


if __name__ == "__main__":
    main()
