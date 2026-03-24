#!/usr/bin/env python3
"""
Stage 2: 标注质量审核脚本（模型辅助）
==========================================
暗场显微镜镜片缺陷检测 — 标注修正辅助工具

功能:
  1. 用 Stage 1 训练模型推理验证集
  2. 与 YOLO GT 标注对比，计算每个 tile 的"可疑度"
  3. 生成优先审核列表 CSV
  4. 为 TOP-N 可疑 tile 生成对比可视化（左=GT，右=预测）
  5. 输出审核统计报告

使用方式:
    python scripts/audit_stage2_labels.py \\
        --model output/training/stage1/weights/best.pt

    python scripts/audit_stage2_labels.py \\
        --model output/training/stage1/weights/best.pt \\
        --top-n 500 --conf 0.3

输出:
    output/audit/
    ├── suspect_tiles.csv        可疑 tile 列表（按可疑度排序）
    ├── overlays/                TOP-N 对比可视化图
    │   └── <tile_name>_audit.jpg
    └── audit_report.json        统计摘要
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TILE_DATASET   = PROJECT_ROOT / "output" / "tile_dataset"
AUDIT_DIR      = PROJECT_ROOT / "output" / "audit"
CLASS_NAMES    = ["scratch", "spot", "critical"]   # 3-class: damage+crash → critical
# 类别颜色 BGR
CLASS_COLORS   = {
    "scratch":  (0, 255, 0),    # 绿
    "spot":     (0, 255, 255),  # 黄
    "critical": (180, 0, 255),  # 紫红（原 damage+crash 合并）
}
CLS_ID_TO_NAME = {i: n for i, n in enumerate(CLASS_NAMES)}


# ─── 辅助函数 ──────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Microlens_DF  Stage 2  标注质量审核                 ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def parse_yolo_label(txt_path: Path, img_w: int = 640, img_h: int = 640) -> np.ndarray:
    """解析 YOLO txt 标注，返回 (N, 5) [cls, x1, y1, x2, y2]。"""
    boxes: list[list[float]] = []
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            x1 = max(0, (cx - bw / 2) * img_w)
            y1 = max(0, (cy - bh / 2) * img_h)
            x2 = min(img_w, (cx + bw / 2) * img_w)
            y2 = min(img_h, (cy + bh / 2) * img_h)
            boxes.append([cls, x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


def parse_yolo_pred(txt_path: Path, img_w: int = 640, img_h: int = 640) -> np.ndarray:
    """解析 YOLO 预测 txt，返回 (N, 6) [cls, x1, y1, x2, y2, conf]。"""
    boxes: list[list[float]] = []
    if not txt_path.exists():
        return np.zeros((0, 6), dtype=np.float32)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                # 没有 conf 列（GT 格式），补 conf=1.0
                if len(parts) == 5:
                    parts.append("1.0")
                else:
                    continue
            cls, cx, cy, bw, bh, conf = map(float, parts[:6])
            x1 = max(0, (cx - bw / 2) * img_w)
            y1 = max(0, (cy - bh / 2) * img_h)
            x2 = min(img_w, (cx + bw / 2) * img_w)
            y2 = min(img_h, (cy + bh / 2) * img_h)
            boxes.append([cls, x1, y1, x2, y2, conf])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 6), dtype=np.float32)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算两组框的 IoU 矩阵，输入 [..., 4] [x1,y1,x2,y2]。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_suspicion(gt: np.ndarray, pred: np.ndarray, iou_thr: float) -> dict:
    """
    计算 tile 的可疑度分数及各项细节。

    suspicion = FP数 + FN数 + 2×类别错误数

    Returns dict with keys: fp, fn, cls_err, suspicion, matched_gt, matched_pred
    """
    if len(gt) == 0 and len(pred) == 0:
        return {"fp": 0, "fn": 0, "cls_err": 0, "suspicion": 0.0,
                "n_gt": 0, "n_pred": 0}

    n_gt   = len(gt)
    n_pred = len(pred)

    if n_gt == 0:
        return {"fp": n_pred, "fn": 0, "cls_err": 0,
                "suspicion": float(n_pred), "n_gt": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"fp": 0, "fn": n_gt, "cls_err": 0,
                "suspicion": float(n_gt), "n_gt": n_gt, "n_pred": 0}

    iou_mat = box_iou_matrix(pred[:, 1:5], gt[:, 1:5])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    cls_err = 0

    for pi in range(n_pred):
        best_iou, best_gi = 0.0, -1
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            if iou_mat[pi, gi] > best_iou:
                best_iou, best_gi = iou_mat[pi, gi], gi
        if best_iou >= iou_thr and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if int(pred[pi, 0]) != int(gt[best_gi, 0]):
                cls_err += 1

    fp = n_pred - len(matched_pred)
    fn = n_gt  - len(matched_gt)
    suspicion = float(fp + fn + 2 * cls_err)

    return {
        "fp": fp, "fn": fn, "cls_err": cls_err,
        "suspicion": suspicion,
        "n_gt": n_gt, "n_pred": n_pred,
        "matched_gt": matched_gt, "matched_pred": matched_pred,
    }


def draw_audit_overlay(
    img: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thr: float,
    matched_gt: set[int],
    matched_pred: set[int],
) -> np.ndarray:
    """生成左右对比图（左=GT标注，右=模型预测）。"""
    h, w = img.shape[:2]
    # 确保 BGR
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()

    left  = img_bgr.copy()
    right = img_bgr.copy()

    # ── 左图：GT 标注 ──────────────────────────────────────
    for gi, box in enumerate(gt_boxes):
        cls_name = CLS_ID_TO_NAME.get(int(box[0]), "?")
        color = CLASS_COLORS.get(cls_name, (128, 128, 128))
        x1, y1, x2, y2 = map(int, box[1:5])
        if gi in matched_gt:
            # 正确匹配 → 实线
            cv2.rectangle(left, (x1, y1), (x2, y2), color, 2)
        else:
            # 未匹配（FN）→ 蓝色虚框
            cv2.rectangle(left, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(left, "FN", (x1, max(y1 - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.putText(left, cls_name[0].upper(), (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── 右图：预测结果 ─────────────────────────────────────
    for pi, box in enumerate(pred_boxes):
        cls_name = CLS_ID_TO_NAME.get(int(box[0]), "?")
        conf = float(box[5]) if len(box) > 5 else 1.0
        x1, y1, x2, y2 = map(int, box[1:5])

        if pi in matched_pred:
            color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            thickness = 2
        else:
            # FP → 橙色
            color = (0, 128, 255) if conf >= 0.5 else (0, 80, 200)
            thickness = 1
            cv2.putText(right, "FP", (x1, max(y1 - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.rectangle(right, (x1, y1), (x2, y2), color, thickness)
        label = f"{cls_name[0].upper()}{conf:.2f}"
        cv2.putText(right, label, (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # ── 添加标题 ───────────────────────────────────────────
    def add_title(panel, text, color=(200, 200, 200)):
        cv2.rectangle(panel, (0, 0), (w, 18), (30, 30, 30), -1)
        cv2.putText(panel, text, (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    n_gt_boxes = len(gt_boxes)
    n_pred_boxes = len(pred_boxes)
    add_title(left,  f"GT ({n_gt_boxes} boxes)   绿=匹配 蓝=FN")
    add_title(right, f"预测 ({n_pred_boxes} boxes)  绿=TP 橙=FP")

    # ── 分隔线 ─────────────────────────────────────────────
    divider = np.full((h, 3, 3), 80, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def run_inference(model, val_img_dir: Path, pred_output_dir: Path, conf: float, device: str):
    """运行 YOLO 推理，将预测结果保存为 txt。"""
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  推理验证集: {val_img_dir}")
    print(f"  置信度阈值: {conf}")
    print(f"  预测输出:   {pred_output_dir}")
    print()

    results = model.predict(
        source=str(val_img_dir),
        conf=conf,
        iou=0.45,
        imgsz=640,
        device=device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(pred_output_dir.parent),
        name=pred_output_dir.name,
        exist_ok=True,
        verbose=False,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: 标注质量审核（模型辅助）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   required=True,
                        help="Stage 1 最佳权重路径，如 output/training/stage1/weights/best.pt")
    parser.add_argument("--top-n",   type=int, default=300,
                        help="生成可视化的 TOP-N 可疑 tile 数量（默认 300）")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="推理置信度阈值（默认 0.25）")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="GT 匹配 IoU 阈值（默认 0.5）")
    parser.add_argument("--device",  default="0",
                        help="GPU id（默认 0）")
    parser.add_argument("--use-existing-preds", action="store_true",
                        help="跳过推理，使用已有预测结果（加快重复运行）")
    args = parser.parse_args()

    print_header()

    # ── 检查路径 ──────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  ✗ 模型权重不存在: {model_path}")
        print("  请先运行 Stage 1 训练:")
        print("      python scripts/train_stage1_baseline.py")
        sys.exit(1)

    val_img_dir  = TILE_DATASET / "images" / "val"
    val_lbl_dir  = TILE_DATASET / "labels" / "val"
    pred_dir     = AUDIT_DIR / "predictions" / "labels"
    overlay_dir  = AUDIT_DIR / "overlays"

    if not val_img_dir.exists():
        print(f"  ✗ 验证集图像目录不存在: {val_img_dir}")
        sys.exit(1)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # ── 推理 ──────────────────────────────────────────────
    if args.use_existing_preds and pred_dir.exists():
        print("  使用已有预测结果 (--use-existing-preds)")
    else:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  ✗ 请安装 ultralytics: pip install ultralytics>=8.3")
            sys.exit(1)

        print(f"  加载模型: {model_path.name}")
        model = YOLO(str(model_path))
        print()

        t0 = time.time()
        run_inference(model, val_img_dir, pred_dir.parent, args.conf, args.device)
        elapsed = time.time() - t0
        print(f"  推理完成，耗时 {fmt_time(elapsed)}")
        print()

    # ── 对比分析 ──────────────────────────────────────────
    val_images = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    if not val_images:
        print(f"  ✗ 验证集目录为空: {val_img_dir}")
        sys.exit(1)

    print(f"  分析 {len(val_images)} 张验证集 tile ...")

    records: list[dict] = []
    class_stats = {
        name: {"n_gt": 0, "n_pred": 0, "n_fp": 0, "n_fn": 0}
        for name in CLASS_NAMES
    }

    for img_path in val_images:
        stem = img_path.stem
        gt_path   = val_lbl_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"

        gt_boxes   = parse_yolo_label(gt_path)
        pred_boxes = parse_yolo_pred(pred_path)

        # 过滤置信度
        if len(pred_boxes) > 0 and pred_boxes.shape[1] >= 6:
            pred_boxes = pred_boxes[pred_boxes[:, 5] >= args.conf]

        info = compute_suspicion(gt_boxes, pred_boxes, args.iou_thr)

        # 更新类别统计
        for c, name in enumerate(CLASS_NAMES):
            class_stats[name]["n_gt"]   += int(np.sum(gt_boxes[:, 0] == c))   if len(gt_boxes) else 0
            class_stats[name]["n_pred"] += int(np.sum(pred_boxes[:, 0] == c)) if len(pred_boxes) else 0

        records.append({
            "tile":       img_path.name,
            "stem":       stem,
            "suspicion":  info["suspicion"],
            "fp":         info["fp"],
            "fn":         info["fn"],
            "cls_err":    info["cls_err"],
            "n_gt":       info["n_gt"],
            "n_pred":     info["n_pred"],
            "_matched_gt":   info.get("matched_gt", set()),
            "_matched_pred": info.get("matched_pred", set()),
            "_gt_boxes":     gt_boxes,
            "_pred_boxes":   pred_boxes,
            "_img_path":     img_path,
        })

    # ── 排序 + 分级 ───────────────────────────────────────
    records.sort(key=lambda r: -r["suspicion"])

    def priority(s: float) -> str:
        if s >= 5:  return "high"
        if s >= 2:  return "medium"
        return "low"

    n_high   = sum(1 for r in records if r["suspicion"] >= 5)
    n_medium = sum(1 for r in records if 2 <= r["suspicion"] < 5)
    n_low    = sum(1 for r in records if r["suspicion"] < 2)

    print(f"  可疑度分布:")
    print(f"    高优先级 (≥5):  {n_high:4d} tiles  → 必须审核")
    print(f"    中优先级 (2-4): {n_medium:4d} tiles  → 抽样审核")
    print(f"    低优先级 (<2):  {n_low:4d} tiles  → 可跳过")
    print()

    # ── 写 CSV ────────────────────────────────────────────
    csv_path = AUDIT_DIR / "suspect_tiles.csv"
    csv_fields = ["rank", "tile", "priority", "suspicion", "fp", "fn", "cls_err", "n_gt", "n_pred"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            writer.writerow({
                "rank":      rank,
                "tile":      r["tile"],
                "priority":  priority(r["suspicion"]),
                "suspicion": round(r["suspicion"], 1),
                "fp":        r["fp"],
                "fn":        r["fn"],
                "cls_err":   r["cls_err"],
                "n_gt":      r["n_gt"],
                "n_pred":    r["n_pred"],
            })
    print(f"  审核列表已保存: {csv_path}")

    # ── 生成 TOP-N 可视化 ─────────────────────────────────
    top_n = min(args.top_n, len(records))
    print(f"  生成 TOP-{top_n} 对比可视化 ...")

    done = 0
    for r in records[:top_n]:
        img = cv2.imread(str(r["_img_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        overlay = draw_audit_overlay(
            img,
            r["_gt_boxes"],
            r["_pred_boxes"],
            args.iou_thr,
            r["_matched_gt"],
            r["_matched_pred"],
        )

        pri = priority(r["suspicion"])
        out_name = f"{pri}_{r['suspicion']:.1f}_{r['stem']}_audit.jpg"
        out_path = overlay_dir / out_name
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        done += 1

        if done % 50 == 0:
            print(f"\r  可视化进度: {done}/{top_n}", end="", flush=True)

    print(f"\r  可视化完成: {done} 张 → {overlay_dir}")
    print()

    # ── 统计报告 ──────────────────────────────────────────
    total_fp  = sum(r["fp"]      for r in records)
    total_fn  = sum(r["fn"]      for r in records)
    total_cls = sum(r["cls_err"] for r in records)
    total_gt  = sum(r["n_gt"]    for r in records)
    total_pred= sum(r["n_pred"]  for r in records)

    precision_est = (total_pred - total_fp) / max(total_pred, 1)
    recall_est    = (total_gt   - total_fn) / max(total_gt, 1)

    report = {
        "generated_at":   __import__("datetime").datetime.now().isoformat(),
        "model":          str(model_path),
        "conf_threshold": args.conf,
        "iou_threshold":  args.iou_thr,
        "n_val_tiles":    len(val_images),
        "suspect_tiles":  {
            "high_priority":   n_high,
            "medium_priority": n_medium,
            "low_priority":    n_low,
        },
        "aggregate": {
            "total_gt_boxes":   total_gt,
            "total_pred_boxes": total_pred,
            "total_fp":         total_fp,
            "total_fn":         total_fn,
            "total_cls_err":    total_cls,
            "precision_est":    round(precision_est, 4),
            "recall_est":       round(recall_est, 4),
        },
        "action_guide": {
            "step1": f"打开 {overlay_dir}，按 priority_score 排序浏览",
            "step2": "对 high 优先级 tile：确认 FP（蓝框）是否为误标，FN（橙框）是否为漏标",
            "step3": "修正标注：直接编辑 output/tile_dataset/labels/val/ 中对应 txt 文件",
            "step4": f"修正完成后重新训练：python scripts/train_stage1_baseline.py --epochs 50 --name stage2_refined",
        },
    }

    report_path = AUDIT_DIR / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 打印汇总 ──────────────────────────────────────────
    print("═" * 56)
    print("  Stage 2 审核分析完成")
    print("─" * 56)
    print(f"  验证集: {len(val_images)} tiles")
    print(f"  GT 框:  {total_gt:,}   预测框: {total_pred:,}")
    print(f"  FP:     {total_fp:,}   FN: {total_fn:,}   类别错误: {total_cls}")
    print(f"  估算 Precision: {precision_est:.3f}   Recall: {recall_est:.3f}")
    print("─" * 56)
    print(f"  高优先级审核 tile: {n_high} 张  (suspicion ≥ 5)")
    print("─" * 56)
    print(f"  审核列表: {csv_path}")
    print(f"  对比图:   {overlay_dir}/")
    print(f"  报告:     {report_path}")
    print("═" * 56)
    print()
    print("  审核操作指南:")
    print("  1. 用图片查看器打开 output/audit/overlays/")
    print("     按文件名排序（high_ 开头的优先审核）")
    print("  2. 左图：绿框=正确匹配  蓝框=FN（可能是误标）")
    print("     右图：绿框=TP  橙框=FP（可能是漏标）")
    print("  3. 修正对应的 labels/val/*.txt 文件")
    print("  4. 修正后重训:")
    print("     python scripts/train_stage1_baseline.py --epochs 50 --name stage2_refined")
    print()


if __name__ == "__main__":
    main()
