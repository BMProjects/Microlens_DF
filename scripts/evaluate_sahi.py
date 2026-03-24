#!/usr/bin/env python3
"""
SAHI 推理结果评估
==========================================
将 infer_sahi.py 生成的全图级检测结果与 GT（切片级聚合到全图）对比，
计算 per-class mAP@0.5、Precision、Recall，输出对比报告。

使用方式:
    # 评估 val 集的 SAHI 结果（需先运行 infer_sahi.py --split val）
    python scripts/evaluate_sahi.py --split val

    # 评估指定目录的标注
    python scripts/evaluate_sahi.py --pred output/sahi_results/labels \
                                    --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fullimage_utils import (
    tile_boxes_to_fullimage,
    nms_ios,
    TILE_SIZE,
    CLASS_NAMES,
)

TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
SAHI_DIR     = PROJECT_ROOT / "output" / "sahi_results"
AUDIT_DIR    = PROJECT_ROOT / "output" / "audit"


# ─── IoU 工具 ──────────────────────────────────────────

def compute_iou(b1, b2) -> float:
    """b = (x1, y1, x2, y2)"""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-8)


# ─── 数据加载 ──────────────────────────────────────────

def load_tile_index(split: str) -> dict[str, dict]:
    idx = {}
    with open(TILE_DATASET / "tile_index.csv") as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            row = dict(zip(header, vals))
            if row["split"] != split:
                continue
            idx[row["tile_id"]] = {
                "source": Path(row["source_image"]).stem,
                "x0": int(row["x0"]), "y0": int(row["y0"]),
            }
    return idx


def load_gt_fullimage(stem: str, tile_index: dict) -> list[tuple]:
    """将某张原图所有切片的 GT 聚合到全图坐标，去重后返回。"""
    all_gt = []
    for tid, info in tile_index.items():
        if info["source"] != stem:
            continue
        label_path = TILE_DATASET / "labels" / "val" / f"{tid}.txt"
        if not label_path.exists():
            label_path = TILE_DATASET / "labels" / "train" / f"{tid}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                box_yolo = (int(p[0]), float(p[1]), float(p[2]),
                            float(p[3]), float(p[4]), 1.0)
                full = tile_boxes_to_fullimage([box_yolo], info["x0"], info["y0"])
                all_gt.extend(full)
    # 去重（高 IOS 阈值）
    deduped = nms_ios(all_gt, ios_thresh=0.75)
    return deduped  # [(cls, x1, y1, x2, y2, conf)]


def load_pred_fullimage(pred_txt: Path) -> list[tuple]:
    """加载全图级 YOLO 预测文件（由 infer_sahi.py 生成）。"""
    # 格式: cls cx cy w h conf （归一化坐标，全图级）
    # 我们需要知道图像分辨率才能反归一化 → 从 sahi_report.json 获取
    return []  # 将在 main 中直接读取


# ─── mAP 计算 ──────────────────────────────────────────

def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """11-点插值 AP。"""
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        ps = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(ps) if ps else 0.0
    return ap / 11.0


def evaluate_class(
    gt_boxes: list[tuple],    # [(x1, y1, x2, y2), ...]
    pred_boxes: list[tuple],  # [(x1, y1, x2, y2, conf), ...]  sorted by conf desc
    iou_thresh: float = 0.5,
) -> dict:
    if not gt_boxes:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0,
                "n_gt": 0, "n_pred": len(pred_boxes), "tp": 0, "fp": len(pred_boxes)}

    pred_boxes = sorted(pred_boxes, key=lambda b: b[4], reverse=True)
    matched_gt = set()
    tp_list = []
    fp_list = []

    for pred in pred_boxes:
        px1, py1, px2, py2, pc = pred
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            v = compute_iou((px1, py1, px2, py2), gt)
            if v > best_iou:
                best_iou = v
                best_idx = i

        if best_iou >= iou_thresh and best_idx >= 0:
            matched_gt.add(best_idx)
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # Cumulative precision/recall
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls    = []
    for tp, fp in zip(tp_list, fp_list):
        tp_cum += tp
        fp_cum += fp
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1))
        recalls.append(tp_cum / max(len(gt_boxes), 1))

    ap = compute_ap(recalls, precisions)
    final_p = precisions[-1] if precisions else 0
    final_r = recalls[-1]    if recalls    else 0

    return {
        "ap": ap,
        "precision": final_p,
        "recall": final_r,
        "n_gt": len(gt_boxes),
        "n_pred": len(pred_boxes),
        "tp": tp_cum,
        "fp": fp_cum,
    }


# ─── 主函数 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SAHI 推理结果评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", default="val",
                        choices=["train", "val"],
                        help="评估哪个 split 的 GT（默认 val）")
    parser.add_argument("--pred", type=str, default=None,
                        help="SAHI 预测标注目录（默认 output/sahi_results/labels）")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU 匹配阈值（默认 0.5）")
    args = parser.parse_args()

    pred_dir = Path(args.pred) if args.pred else SAHI_DIR / "labels"
    if not pred_dir.exists():
        print(f"  ✗ 预测目录不存在: {pred_dir}")
        print("    请先运行: python scripts/infer_sahi.py --split val")
        sys.exit(1)

    # 加载 sahi_report 以获取图像尺寸
    report_path = pred_dir.parent / "sahi_report.json"
    img_sizes: dict[str, tuple[int, int]] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        for info in report.get("per_image", []):
            img_sizes[info["image"]] = (info["image_w"], info["image_h"])

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   SAHI 推理结果评估                                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  split:    {args.split}")
    print(f"  pred_dir: {pred_dir}")
    print(f"  IoU:      {args.iou}")
    print()

    # 加载切片索引
    tile_index = load_tile_index(args.split)
    stems = sorted({info["source"] for info in tile_index.values()})
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}

    # 按类别汇总
    class_gt_boxes: dict[int, list] = defaultdict(list)
    class_pred_boxes: dict[int, list] = defaultdict(list)

    n_images = 0
    for stem in stems:
        if stem not in pred_files:
            continue

        gt_full = load_gt_fullimage(stem, tile_index)
        img_w, img_h = img_sizes.get(stem, (4096, 3000))

        # 读取预测（归一化全图坐标）
        pred_txt = pred_files[stem]
        preds = []
        for line in pred_txt.read_text().splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls  = int(p[0])
            cx_n, cy_n, w_n, h_n = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            conf = float(p[5]) if len(p) > 5 else 0.5
            x1 = (cx_n - w_n / 2) * img_w
            y1 = (cy_n - h_n / 2) * img_h
            x2 = (cx_n + w_n / 2) * img_w
            y2 = (cy_n + h_n / 2) * img_h
            preds.append((cls, x1, y1, x2, y2, conf))

        for cls, x1, y1, x2, y2, conf in gt_full:
            class_gt_boxes[cls].append((x1, y1, x2, y2))
        for cls, x1, y1, x2, y2, conf in preds:
            class_pred_boxes[cls].append((x1, y1, x2, y2, conf))

        n_images += 1

    print(f"  评估 {n_images} 张图像")
    print()

    # 计算各类 AP
    results = {}
    all_aps = []

    print(f"  {'类别':10s} {'AP@0.5':>8} {'P':>7} {'R':>7} "
          f"{'GT数':>7} {'Pred数':>7} {'TP':>6} {'FP':>6}")
    print(f"  {'─'*66}")

    for cls_id in range(3):
        cls_name = CLASS_NAMES[cls_id]
        ev = evaluate_class(
            class_gt_boxes[cls_id],
            class_pred_boxes[cls_id],
            iou_thresh=args.iou,
        )
        results[cls_name] = ev
        all_aps.append(ev["ap"])
        print(f"  {cls_name:10s} {ev['ap']:8.4f} {ev['precision']:7.4f} "
              f"{ev['recall']:7.4f} {ev['n_gt']:7d} {ev['n_pred']:7d} "
              f"{ev['tp']:6d} {ev['fp']:6d}")

    mean_ap = sum(all_aps) / max(len(all_aps), 1)
    print(f"  {'─'*66}")
    print(f"  {'mAP@0.5':10s} {mean_ap:8.4f}")
    print()

    # 对比说明
    print("  对比参考（tile 级评估结果）:")
    print("    Stage3 mAP@0.5 = 0.6543")
    print("    scratch: 0.414  spot: 0.807  critical: 0.742")
    print()
    print("  SAHI 推理优势: 全图级 NMS + scratch 连接 → 减少碎片化误检")
    print("  注意: 全图级 IoU 匹配比切片级更严格（GT 坐标去重可能影响分母）")
    print()

    # 保存结果
    eval_result = {
        "pipeline": "sahi_evaluate",
        "split": args.split,
        "iou_thresh": args.iou,
        "n_images": n_images,
        "mean_ap50": round(mean_ap, 4),
        "per_class": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in results.items()},
    }
    out_path = AUDIT_DIR / "sahi_evaluation.json"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2))
    print(f"  结果保存: {out_path}")
    print()


if __name__ == "__main__":
    main()
