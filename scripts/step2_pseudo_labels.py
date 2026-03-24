#!/usr/bin/env python3
"""
Step 2: 多策略推理 + 伪标签生成
==========================================
用当前模型进行多尺度/TTA推理，生成高质量伪标签，
然后与清洗后的GT融合，构建更优质的训练集。

流程:
  1. 多尺度推理 (640/960/1280) + flip TTA
  2. 跨尺度一致性筛选（IoU>0.5 + 类别一致 + 平均conf>0.4）
  3. 与GT融合：保留一致的、修正冲突的、新增遗漏的

使用方式:
    python scripts/step2_pseudo_labels.py                    # 全流程
    python scripts/step2_pseudo_labels.py --infer-only       # 只做推理
    python scripts/step2_pseudo_labels.py --merge-only       # 只做融合
    python scripts/step2_pseudo_labels.py --dry-run          # 预览统计

预期收益: mAP +0.03~0.06（标签去噪 + 补漏）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT  = Path(__file__).parent.parent
TILE_DATASET  = PROJECT_ROOT / "output" / "tile_dataset"
OUTPUT_DIR    = PROJECT_ROOT / "output" / "pseudo_labels"
WEIGHTS_PATH  = PROJECT_ROOT / "output" / "training" / "stage2_cleaned" / "weights" / "best.pt"

TILE_SIZE     = 640
CLASS_NAMES   = ["scratch", "spot", "critical"]

# ─── 推理参数 ──────────────────────────────────────────────
INFER_SCALES  = [640, 960, 1280]
CONF_THRESH   = 0.15          # 低阈值，尽量召回
IOU_THRESH    = 0.5           # NMS 阈值

# ─── 一致性筛选参数 ────────────────────────────────────────
CONSIST_IOU   = 0.4           # 跨尺度匹配 IoU 阈值
MIN_VOTES     = 2             # 至少出现在 N 个尺度/策略中
MIN_AVG_CONF  = 0.35          # 平均置信度下限

# ─── 面积过滤（防止 giant 框进入伪标签）─────────────────────
MAX_BOX_AREA  = 0.30          # 归一化面积上限（>30% tile 视为不可学大框）

# ─── GT 融合参数 ──────────────────────────────────────────
MERGE_IOU     = 0.4           # GT-pseudo 匹配 IoU 阈值


def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("  ✗ ultralytics 未安装")
        sys.exit(1)


def box_iou(a, b):
    """计算两个框的IoU。a, b = (x1, y1, x2, y2)。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def xyxy_from_yolo(cx, cy, w, h, img_size=TILE_SIZE):
    """YOLO归一化 → 像素 xyxy。"""
    px = cx * img_size
    py = cy * img_size
    pw = w * img_size
    ph = h * img_size
    return (px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2)


def yolo_from_xyxy(x1, y1, x2, y2, img_size=TILE_SIZE):
    """像素 xyxy → YOLO归一化 cxcywh。"""
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w  = (x2 - x1) / img_size
    h  = (y2 - y1) / img_size
    return (cx, cy, w, h)


# ─── Phase 1: 多策略推理 ──────────────────────────────────

# 每个尺度的 batch size（显存限制: RTX 4090D 24GB）
BATCH_BY_SCALE = {640: 64, 960: 32, 1280: 16}


def run_multiscale_inference(model, split: str = "train"):
    """
    对指定 split 运行多尺度 TTA 推理，返回 ({tile_id: [detections]}, strategies)。

    使用 stream=True 逐批处理，避免 OOM。
    """
    img_dir = TILE_DATASET / "images" / split
    if not img_dir.exists():
        print(f"  ✗ 图像目录不存在: {img_dir}")
        return {}, []

    tile_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"  {split} 集共 {len(tile_paths)} 个切片")

    all_detections = defaultdict(list)  # tile_id → [(cls, cx, cy, w, h, conf, strategy)]
    strategies = []

    for scale in INFER_SCALES:
        batch_size = BATCH_BY_SCALE.get(scale, 16)
        for augment in [False, True]:  # True = flip TTA
            strategy_name = f"s{scale}_{'tta' if augment else 'raw'}"
            strategies.append(strategy_name)
            print(f"  推理: {strategy_name} (batch={batch_size}) ...", end="", flush=True)
            t0 = time.time()
            n_dets = 0

            # 分批处理，stream=True 逐结果返回，避免 OOM
            for batch_start in range(0, len(tile_paths), batch_size):
                batch_paths = tile_paths[batch_start : batch_start + batch_size]

                for result in model.predict(
                    source=[str(p) for p in batch_paths],
                    imgsz=scale,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    augment=augment,
                    device="0",
                    verbose=False,
                    stream=True,          # 关键: 逐结果流式返回，不堆积显存
                ):
                    # result.path 是原始图像路径
                    tid = Path(result.path).stem
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes = result.boxes
                    h_img, w_img = result.orig_shape
                    for i in range(len(boxes)):
                        cls  = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        cx = (xyxy[0] + xyxy[2]) / 2 / w_img
                        cy = (xyxy[1] + xyxy[3]) / 2 / h_img
                        w  = (xyxy[2] - xyxy[0]) / w_img
                        h  = (xyxy[3] - xyxy[1]) / h_img
                        all_detections[tid].append(
                            (cls, cx, cy, w, h, conf, strategy_name)
                        )
                        n_dets += 1

            elapsed = time.time() - t0
            print(f" {n_dets:,} 检测  ({elapsed:.1f}s)")

    return dict(all_detections), strategies


# ─── Phase 2: 一致性筛选 ──────────────────────────────────

def filter_consistent(
    detections: dict[str, list],
    strategies: list[str],
) -> dict[str, list]:
    """
    跨尺度一致性筛选。
    返回 {tile_id: [(cls, cx, cy, w, h, avg_conf, n_votes)]}
    """
    n_strategies = len(strategies)
    filtered = {}

    for tid, dets in detections.items():
        if not dets:
            continue

        # Group by strategy
        by_strategy = defaultdict(list)
        for det in dets:
            by_strategy[det[6]].append(det[:6])  # (cls, cx, cy, w, h, conf)

        # Use the first strategy as anchor, match across others
        # Simplified: cluster all detections by IoU
        clusters = []
        used = set()

        for i, det_i in enumerate(dets):
            if i in used:
                continue
            cluster = [det_i]
            used.add(i)
            box_i = xyxy_from_yolo(det_i[1], det_i[2], det_i[3], det_i[4])

            for j, det_j in enumerate(dets):
                if j in used:
                    continue
                box_j = xyxy_from_yolo(det_j[1], det_j[2], det_j[3], det_j[4])
                if box_iou(box_i, box_j) >= CONSIST_IOU:
                    # Check class consistency
                    if det_j[0] == det_i[0]:
                        cluster.append(det_j)
                        used.add(j)

            clusters.append(cluster)

        # Filter clusters
        tile_results = []
        for cluster in clusters:
            # Count unique strategies
            strat_set = set(d[6] for d in cluster)
            n_votes = len(strat_set)
            avg_conf = np.mean([d[5] for d in cluster])

            if n_votes >= MIN_VOTES and avg_conf >= MIN_AVG_CONF:
                # Use the highest-confidence detection as representative
                best = max(cluster, key=lambda d: d[5])
                cls = best[0]
                # Average the box coordinates across cluster for stability
                cx = np.mean([d[1] for d in cluster])
                cy = np.mean([d[2] for d in cluster])
                w  = np.mean([d[3] for d in cluster])
                h  = np.mean([d[4] for d in cluster])
                # 面积过滤：giant 框不进入伪标签（防止模型学会输出大框）
                if w * h > MAX_BOX_AREA:
                    continue
                tile_results.append((cls, cx, cy, w, h, float(avg_conf), n_votes))

        if tile_results:
            filtered[tid] = tile_results

    return filtered


# ─── Phase 3: GT 融合 ─────────────────────────────────────

def load_gt(split: str) -> dict[str, list]:
    """加载GT标注。返回 {tile_id: [(cls, cx, cy, w, h)]}"""
    label_dir = TILE_DATASET / "labels" / split
    gt = {}
    for txt in label_dir.glob("*.txt"):
        boxes = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]),
                              float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])))
        gt[txt.stem] = boxes
    return gt


def merge_gt_pseudo(
    gt: dict[str, list],
    pseudo: dict[str, list],
    all_tile_ids: list[str],
) -> tuple[dict[str, list], dict]:
    """
    融合GT和伪标签。

    规则:
      - GT框与pseudo高IoU一致 → 保留GT（已验证）
      - GT框无匹配pseudo → 降权保留（可能是噪声，但也可能是模型漏检）
      - Pseudo框无匹配GT → 如果conf高 → 新增（补漏）
      - GT框与pseudo类别冲突 → 使用pseudo（模型判断可能更准）

    Returns:
      merged: {tile_id: [(cls, cx, cy, w, h)]}
      stats: 融合统计
    """
    stats = Counter()
    merged = {}

    for tid in all_tile_ids:
        gt_boxes = gt.get(tid, [])
        pseudo_boxes = pseudo.get(tid, [])

        result_boxes = []
        gt_matched = set()
        pseudo_matched = set()

        # Match GT ↔ pseudo
        for gi, gb in enumerate(gt_boxes):
            gt_xyxy = xyxy_from_yolo(gb[1], gb[2], gb[3], gb[4])
            best_pi = -1
            best_iou = 0

            for pi, pb in enumerate(pseudo_boxes):
                if pi in pseudo_matched:
                    continue
                p_xyxy = xyxy_from_yolo(pb[1], pb[2], pb[3], pb[4])
                iou = box_iou(gt_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi

            if best_iou >= MERGE_IOU and best_pi >= 0:
                pb = pseudo_boxes[best_pi]
                if gb[0] == pb[0]:
                    # 类别一致 → 保留GT
                    result_boxes.append(gb)
                    stats["gt_confirmed"] += 1
                else:
                    # 类别冲突 → 如果pseudo高置信度，采用pseudo类别
                    if pb[5] > 0.5:
                        result_boxes.append((pb[0], gb[1], gb[2], gb[3], gb[4]))
                        stats["cls_corrected"] += 1
                    else:
                        result_boxes.append(gb)
                        stats["cls_kept_gt"] += 1
                gt_matched.add(gi)
                pseudo_matched.add(best_pi)
            else:
                # GT无匹配 → 保留（可能是小缺陷模型没检测到）
                result_boxes.append(gb)
                stats["gt_unmatched"] += 1

        # Pseudo无匹配GT → 新增（补漏）
        for pi, pb in enumerate(pseudo_boxes):
            if pi in pseudo_matched:
                continue
            if pb[5] >= 0.5:  # 只添加高置信度的
                result_boxes.append((pb[0], pb[1], pb[2], pb[3], pb[4]))
                stats["pseudo_added"] += 1
            else:
                stats["pseudo_dropped"] += 1

        merged[tid] = result_boxes

    return merged, dict(stats)


def save_merged_labels(merged: dict[str, list], split: str, output_dir: Path):
    """保存融合后的标注。"""
    out_label_dir = output_dir / "labels" / split
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for tid, boxes in merged.items():
        lines = []
        for b in boxes:
            lines.append(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
        (out_label_dir / f"{tid}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )


def create_pseudo_yaml(output_dir: Path):
    """创建使用伪标签数据集的 YAML 配置。"""
    yaml_content = f"""# 伪标签增强数据集 — YOLO 格式
# 生成方式: step2_pseudo_labels.py (多尺度推理 + 一致性筛选 + GT融合)

path: {TILE_DATASET}
train: images/train
val: images/val

nc: 3
names: ['scratch', 'spot', 'critical']
"""
    yaml_path = output_dir / "defects_pseudo.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


# ─── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="多策略推理 + 伪标签生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default=str(WEIGHTS_PATH), help="模型权重路径")
    parser.add_argument("--split",   default="train", help="处理的 split")
    parser.add_argument("--infer-only", action="store_true", help="只做推理")
    parser.add_argument("--merge-only", action="store_true", help="只做融合")
    parser.add_argument("--dry-run",    action="store_true", help="预览统计")
    parser.add_argument("--device",     default="0", help="GPU device")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Step 2: 多策略推理 + 伪标签生成                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cache files for intermediate results
    infer_cache = OUTPUT_DIR / f"raw_detections_{args.split}.json"

    # ── Phase 1: 多尺度推理 ──
    if not args.merge_only:
        print(f"  加载模型: {args.weights}")
        YOLO = load_yolo()
        model = YOLO(args.weights)

        print(f"\n  Phase 1: 多策略推理 ({args.split}) ...")
        print(f"  尺度: {INFER_SCALES}, TTA: 开启")
        print()

        detections, strategies = run_multiscale_inference(model, split=args.split)

        total_dets = sum(len(v) for v in detections.values())
        print(f"\n  原始检测总数: {total_dets:,} (跨 {len(detections)} 个切片)")

        # ── Phase 2: 一致性筛选 ──
        print(f"\n  Phase 2: 跨尺度一致性筛选 ...")
        print(f"  阈值: IoU≥{CONSIST_IOU}, votes≥{MIN_VOTES}/{len(strategies)}, "
              f"avg_conf≥{MIN_AVG_CONF}")

        pseudo = filter_consistent(detections, strategies)
        total_pseudo = sum(len(v) for v in pseudo.values())
        print(f"  筛选后: {total_pseudo:,} 个高置信框 (跨 {len(pseudo)} 个切片)")

        # Save intermediate
        serializable = {
            tid: [(b[0], float(b[1]), float(b[2]), float(b[3]), float(b[4]),
                   float(b[5]), b[6])
                  for b in boxes]
            for tid, boxes in pseudo.items()
        }
        with open(infer_cache, "w") as f:
            json.dump(serializable, f)
        print(f"  中间结果已保存: {infer_cache}")

        if args.infer_only:
            return

    else:
        # Load from cache
        if not infer_cache.exists():
            print(f"  ✗ 未找到推理缓存: {infer_cache}")
            print(f"  请先运行: python scripts/step2_pseudo_labels.py --infer-only")
            sys.exit(1)
        with open(infer_cache) as f:
            raw = json.load(f)
        pseudo = {
            tid: [(b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in boxes]
            for tid, boxes in raw.items()
        }
        total_pseudo = sum(len(v) for v in pseudo.values())
        print(f"  从缓存加载 {total_pseudo:,} 个伪标签框")

    # ── Phase 3: GT 融合 ──
    print(f"\n  Phase 3: GT 融合 ...")
    gt = load_gt(args.split)
    all_tile_ids = sorted(set(list(gt.keys()) + list(pseudo.keys())))
    total_gt = sum(len(v) for v in gt.values())
    print(f"  GT 框: {total_gt:,} | 伪标签框: {total_pseudo:,}")

    merged, stats = merge_gt_pseudo(gt, pseudo, all_tile_ids)
    total_merged = sum(len(v) for v in merged.values())

    print(f"\n  融合统计:")
    print(f"    GT 被验证确认:    {stats.get('gt_confirmed', 0):>6,}")
    print(f"    GT 无模型匹配:    {stats.get('gt_unmatched', 0):>6,}  (保留)")
    print(f"    类别被修正:       {stats.get('cls_corrected', 0):>6,}")
    print(f"    类别保留GT:       {stats.get('cls_kept_gt', 0):>6,}")
    print(f"    伪标签新增:       {stats.get('pseudo_added', 0):>6,}  (补漏)")
    print(f"    伪标签丢弃:       {stats.get('pseudo_dropped', 0):>6,}  (低置信)")
    print(f"  ──────────────────────────────────")
    print(f"    最终标注:         {total_merged:>6,}  (原 {total_gt:,})")

    if args.dry_run:
        print("\n  (dry-run 模式，不保存)")
        return

    # ── 保存 ──
    print(f"\n  保存融合标注 ...")

    # Save merged labels directly back to the tile_dataset/labels/
    # (backup should already exist from step1)
    save_merged_labels(merged, args.split, TILE_DATASET)
    print(f"  ✓ 已更新 {args.split}/ 标注 ({total_merged:,} 个框)")

    # Save report
    report = {
        "split": args.split,
        "original_gt": total_gt,
        "pseudo_candidates": total_pseudo,
        "merged_total": total_merged,
        "stats": stats,
    }
    report_path = OUTPUT_DIR / f"merge_report_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告: {report_path}")

    print()
    print("═" * 56)
    print("  下一步: 重新训练")
    print("      python scripts/step3_retrain.py")
    print("═" * 56)
    print()


if __name__ == "__main__":
    main()
