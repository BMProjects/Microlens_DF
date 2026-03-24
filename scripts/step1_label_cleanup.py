#!/usr/bin/env python3
"""
Step 1: 自动标注清洗 — 去除噪声标注
======================================
核心策略：移除不可学习的标注噪声，让 mAP 评估对齐模型真实能力。

清洗规则：
  A. 噪声小框: 移除面积过小 / min_side 过小 / 极端长宽比的框
  B. 边缘截断残留: 紧贴边缘的小框（切片分割产物）
  C. 满切片大框: 贴边且面积 > 70% 切片的 scratch 框
     → 这些是密集划痕区域被经典检测器整体框住的产物
     → 模型无法从 "整个切片 = 一个scratch" 学到有用特征
     → 转为 critical 类（语义 = "严重缺陷区"）或直接移除
  D. 重叠去重: 同一切片中 IoU > 0.8 的同类框只保留一个

使用方式:
    python scripts/step1_label_cleanup.py --dry-run    # 预览，不修改
    python scripts/step1_label_cleanup.py              # 执行清洗
    python scripts/step1_label_cleanup.py --restore    # 从备份恢复
    python scripts/step1_label_cleanup.py --large-to-critical  # 大框→critical(默认移除)

预期收益: mAP +0.05~0.10（消除噪声GT提高recall计算基数的合理性）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
BACKUP_DIR   = TILE_DATASET / "labels_backup_preclean"

CLASS_NAMES = ["scratch", "spot", "critical"]
TILE_SIZE   = 640

# ─── 清洗阈值（像素坐标，基于640×640切片）───────────────────

# A. 小框阈值
THRESHOLDS = {
    # class_id: (min_area_px, min_side_px, max_aspect_ratio)
    0: (150, 6, 50),    # scratch: 允许极细（6px宽），但面积不能太小
    1: (120, 8, 6),     # spot:    近圆形，最小8×8px
    2: (200, 10, 10),   # critical: 大面积缺陷，至少10×10
}
DEFAULT_THRESHOLD = (150, 8, 30)

# C. 大框阈值
LARGE_AREA_THRESH  = 0.50   # 归一化面积 > 50% 切片 → 大框
LARGE_SIDE_THRESH  = 0.85   # 任一边 > 85% 切片宽/高 → 贴边大框
EDGE_MARGIN        = 0.02   # 框边缘距切片边缘 < 2% → 视为贴边
# 注：giant 框一律删除（不再转 critical），原因：
#   "to_critical" 策略让模型学会对密集区域输出大框，造成推理时 giant 预测框大量增生。
#   这类标注无法让模型学到真正的划痕定位，删除后让模型专注于可检测的小/中尺度缺陷。

# D. 去重阈值
DEDUP_IOU_THRESH   = 0.80   # 同类框 IoU > 80% 视为重复


def box_iou(b1, b2):
    """计算两个 YOLO 归一化框的 IoU。b = (cx, cy, w, h)。"""
    x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    inter_w = max(0, min(x2a, x2b) - max(x1a, x1b))
    inter_h = max(0, min(y2a, y2b) - max(y1a, y1b))
    inter = inter_w * inter_h
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / max(union, 1e-8)


def is_edge_hugging(cx: float, cy: float, w: float, h: float) -> int:
    """检查框有几条边贴着切片边缘。"""
    m = EDGE_MARGIN
    n = 0
    if cx - w / 2 < m:       n += 1  # 贴左
    if cx + w / 2 > 1 - m:   n += 1  # 贴右
    if cy - h / 2 < m:       n += 1  # 贴上
    if cy + h / 2 > 1 - m:   n += 1  # 贴下
    return n


def classify_box(cls: int, cx: float, cy: float, w: float, h: float):
    """
    分析一个框的类型。

    返回: (action, reason)
      action: "keep" | "remove" | "to_critical"
    """
    pw = w * TILE_SIZE
    ph = h * TILE_SIZE
    area_px = pw * ph
    area_norm = w * h
    min_side = min(pw, ph)
    max_side = max(pw, ph)
    aspect = max_side / max(min_side, 0.1)
    n_edges = is_edge_hugging(cx, cy, w, h)

    # ── C: 满切片大框检测 (优先检查) ──────────────────────
    # 条件：scratch/critical + 面积>50% + 至少一边>85% + 贴>=2条边
    if cls in (0, 2):
        is_giant = (
            area_norm > LARGE_AREA_THRESH
            and max(w, h) > LARGE_SIDE_THRESH
            and n_edges >= 2
        )
        if is_giant:
            return "remove", f"giant_{n_edges}edges_area={area_norm:.2f}"

    # 面积 > 70% 且贴边 → 也是大框（即使不满足 side>0.85）
    if cls == 0 and area_norm > 0.70 and n_edges >= 2:
        return "remove", f"large_scratch_area={area_norm:.2f}"

    # ── A: 噪声小框 ──────────────────────────────────────
    min_area, min_s, max_asp = THRESHOLDS.get(cls, DEFAULT_THRESHOLD)

    if area_px < min_area:
        return "remove", f"area={area_px:.0f}<{min_area}"
    if min_side < min_s:
        return "remove", f"min_side={min_side:.1f}<{min_s}"
    if aspect > max_asp:
        return "remove", f"aspect={aspect:.1f}>{max_asp}"

    # ── B: 边缘截断小残留 ────────────────────────────────
    margin = 3.0 / TILE_SIZE
    if (cx - w / 2 < margin and w < 0.05) or (cx + w / 2 > 1 - margin and w < 0.05):
        return "remove", "edge_truncated_x"
    if (cy - h / 2 < margin and h < 0.05) or (cy + h / 2 > 1 - margin and h < 0.05):
        return "remove", "edge_truncated_y"

    return "keep", "ok"


def dedup_boxes(boxes: list[tuple]) -> tuple[list[tuple], int]:
    """
    对同一切片中的框进行去重（IoU > 阈值的同类框只保留一个）。
    boxes: [(cls, cx, cy, w, h, original_line), ...]
    返回: (kept_boxes, n_removed)
    """
    if len(boxes) <= 1:
        return boxes, 0

    kept = []
    removed = 0

    for i, box_i in enumerate(boxes):
        is_dup = False
        for box_j in kept:
            if box_i[0] != box_j[0]:  # 不同类不算重复
                continue
            iou = box_iou(
                (box_i[1], box_i[2], box_i[3], box_i[4]),
                (box_j[1], box_j[2], box_j[3], box_j[4]),
            )
            if iou > DEDUP_IOU_THRESH:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(box_i)

    return kept, removed


def clean_label_file(txt_path: Path, dry_run: bool = False,
                     large_to_critical: bool = False):
    """
    清洗单个标注文件。

    返回 (original_count, kept_count, removed_reasons)
    """
    lines = txt_path.read_text().splitlines()
    kept_boxes = []     # (cls, cx, cy, w, h, line_str)
    removed_reasons = Counter()
    original_count = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        original_count += 1
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        action, reason = classify_box(cls, cx, cy, w, h)

        if action == "keep":
            kept_boxes.append((cls, cx, cy, w, h, line))
        elif action == "to_critical":
            if large_to_critical:
                # 转为 critical 类，保留坐标，限制最大面积
                new_w = min(w, 0.85)
                new_h = min(h, 0.85)
                new_line = f"2 {cx:.6f} {cy:.6f} {new_w:.6f} {new_h:.6f}"
                kept_boxes.append((2, cx, cy, new_w, new_h, new_line))
                removed_reasons[f"scratch→critical:{reason}"] += 1
            else:
                # 默认直接移除
                removed_reasons[f"remove_giant:{reason}"] += 1
        else:  # remove
            cname = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            removed_reasons[f"{cname}:{reason}"] += 1

    # D: 去重
    kept_boxes, n_dedup = dedup_boxes(kept_boxes)
    if n_dedup > 0:
        removed_reasons["dedup_iou>0.8"] += n_dedup

    # 写回
    if not dry_run and (len(kept_boxes) != original_count or n_dedup > 0):
        txt_path.write_text(
            "\n".join(b[5] for b in kept_boxes)
            + ("\n" if kept_boxes else "")
        )

    return original_count, len(kept_boxes), removed_reasons


# ─── E: 跨切片重叠区去重 ──────────────────────────────────

STRIDE = 560
OVERLAP = TILE_SIZE - STRIDE  # 80px
OVERLAP_NORM = OVERLAP / TILE_SIZE  # 0.125


def load_tile_index():
    """加载切片索引，返回 {tile_id: {src, y0, x0, split}}。"""
    idx_path = TILE_DATASET / "tile_index.csv"
    tiles = {}
    if not idx_path.exists():
        return tiles
    with open(idx_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            row = dict(zip(header, vals))
            tiles[row["tile_id"]] = {
                "src": Path(row["source_image"]).stem,
                "y0": int(row["y0"]),
                "x0": int(row["x0"]),
                "split": row["split"],
            }
    return tiles


def find_adjacent_pairs(tile_index: dict, split: str):
    """找出所有水平/垂直相邻的切片对。"""
    from collections import defaultdict
    by_source = defaultdict(list)
    for tid, info in tile_index.items():
        if info["split"] == split:
            by_source[info["src"]].append(tid)

    pairs = []
    for src, tids in by_source.items():
        for i, ta in enumerate(tids):
            for tb in tids[i + 1:]:
                ia, ib = tile_index[ta], tile_index[tb]
                dx = ib["x0"] - ia["x0"]
                dy = ib["y0"] - ia["y0"]
                if dx == STRIDE and dy == 0:
                    pairs.append((ta, tb, "h"))  # horizontal: b is to the right of a
                elif dx == -STRIDE and dy == 0:
                    pairs.append((tb, ta, "h"))
                elif dy == STRIDE and dx == 0:
                    pairs.append((ta, tb, "v"))  # vertical: b is below a
                elif dy == -STRIDE and dx == 0:
                    pairs.append((tb, ta, "v"))
    return pairs


def cross_tile_dedup(split: str, tile_index: dict, dry_run: bool = False):
    """
    跨切片重叠区去重。

    策略: 对于出现在两个相邻切片重叠区的同类框（IoU>0.3 after 坐标映射），
    保留框中心更远离切片边缘的那个，移除另一个。

    Returns: n_removed
    """
    label_dir = TILE_DATASET / "labels" / split
    pairs = find_adjacent_pairs(tile_index, split)

    # Load all labels into memory
    all_labels = {}
    for txt in label_dir.glob("*.txt"):
        boxes = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append({
                    "cls": int(parts[0]),
                    "cx": float(parts[1]), "cy": float(parts[2]),
                    "w": float(parts[3]), "h": float(parts[4]),
                    "line": line.strip(),
                    "remove": False,
                })
        all_labels[txt.stem] = boxes

    n_removed = 0

    for ta, tb, direction in pairs:
        boxes_a = all_labels.get(ta, [])
        boxes_b = all_labels.get(tb, [])
        if not boxes_a or not boxes_b:
            continue

        info_a = tile_index[ta]
        info_b = tile_index[tb]

        # Find boxes in overlap zone
        if direction == "h":
            # b is right of a. Overlap zone:
            #   In tile_a: cx in [STRIDE/TILE, 1.0] → right edge
            #   In tile_b: cx in [0, OVERLAP/TILE]  → left edge
            overlap_a = [
                (i, b) for i, b in enumerate(boxes_a)
                if not b["remove"] and b["cx"] > (STRIDE / TILE_SIZE - 0.02)
            ]
            overlap_b = [
                (i, b) for i, b in enumerate(boxes_b)
                if not b["remove"] and b["cx"] < (OVERLAP_NORM + 0.02)
            ]
        else:  # vertical
            overlap_a = [
                (i, b) for i, b in enumerate(boxes_a)
                if not b["remove"] and b["cy"] > (STRIDE / TILE_SIZE - 0.02)
            ]
            overlap_b = [
                (i, b) for i, b in enumerate(boxes_b)
                if not b["remove"] and b["cy"] < (OVERLAP_NORM + 0.02)
            ]

        # Match boxes: convert to full-image coords and compute IoU
        for ia, ba in overlap_a:
            for ib, bb in overlap_b:
                if ba["remove"] or bb["remove"]:
                    continue
                if ba["cls"] != bb["cls"]:
                    continue

                # Convert to full-image pixel coords
                ax = info_a["x0"] + ba["cx"] * TILE_SIZE
                ay = info_a["y0"] + ba["cy"] * TILE_SIZE
                aw = ba["w"] * TILE_SIZE
                ah = ba["h"] * TILE_SIZE

                bx = info_b["x0"] + bb["cx"] * TILE_SIZE
                by = info_b["y0"] + bb["cy"] * TILE_SIZE
                bw = bb["w"] * TILE_SIZE
                bh = bb["h"] * TILE_SIZE

                # IoU in full-image coords
                a_xyxy = (ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2)
                b_xyxy = (bx - bw/2, by - bh/2, bx + bw/2, by + bh/2)

                ix1 = max(a_xyxy[0], b_xyxy[0])
                iy1 = max(a_xyxy[1], b_xyxy[1])
                ix2 = min(a_xyxy[2], b_xyxy[2])
                iy2 = min(a_xyxy[3], b_xyxy[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = aw * ah + bw * bh - inter
                iou = inter / max(union, 1e-8)

                if iou > 0.3:
                    # Keep the box whose center is further from its tile edge
                    if direction == "h":
                        dist_a = 1.0 - ba["cx"]  # distance from right edge
                        dist_b = bb["cx"]          # distance from left edge
                    else:
                        dist_a = 1.0 - ba["cy"]
                        dist_b = bb["cy"]

                    if dist_a >= dist_b:
                        bb["remove"] = True
                    else:
                        ba["remove"] = True
                    n_removed += 1

    # Write back modified files
    if not dry_run:
        for tid, boxes in all_labels.items():
            kept = [b for b in boxes if not b["remove"]]
            if len(kept) != len(boxes):
                txt_path = label_dir / f"{tid}.txt"
                txt_path.write_text(
                    "\n".join(b["line"] for b in kept)
                    + ("\n" if kept else "")
                )

    return n_removed


def main():
    parser = argparse.ArgumentParser(
        description="自动标注清洗 — 去除噪声标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不修改")
    parser.add_argument("--restore", action="store_true", help="从备份恢复原始标注")
    parser.add_argument("--large-to-critical", action="store_true",
                        help="将满切片大框转为 critical 类（默认直接移除）")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="处理的 split (默认 train val)")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Step 1: 自动标注清洗 — 去除噪声标注               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── 恢复模式 ──
    if args.restore:
        if not BACKUP_DIR.exists():
            print("  ✗ 备份目录不存在，无法恢复")
            sys.exit(1)
        for split in args.splits:
            src = BACKUP_DIR / split
            dst = TILE_DATASET / "labels" / split
            if src.exists():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                print(f"  ✓ 已恢复 {split}/ 标注")
        print("  恢复完成。")
        return

    # ── 备份 ──
    if not args.dry_run and not BACKUP_DIR.exists():
        print("  备份原始标注 ...", end="", flush=True)
        BACKUP_DIR.mkdir(parents=True)
        for split in args.splits:
            src = TILE_DATASET / "labels" / split
            if src.exists():
                shutil.copytree(src, BACKUP_DIR / split)
        print(" ✓")
        print()

    # ── 清洗阈值说明 ──
    print("  清洗规则:")
    print("  A. 小框阈值:")
    for cls_id, (min_a, min_s, max_asp) in THRESHOLDS.items():
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        print(f"      {name}:  min_area={min_a}px²  min_side={min_s}px  max_aspect={max_asp}:1")
    print(f"  B. 边缘截断残留:  紧贴边缘的小框 (w<0.05)")
    print(f"  C. 满切片大框:    area>{LARGE_AREA_THRESH} + side>{LARGE_SIDE_THRESH} + 贴>=2边")
    large_action = "→ 转为 critical" if args.large_to_critical else "→ 直接移除"
    print(f"     处理方式:      {large_action}")
    print(f"  D. 切片内去重:    同类 IoU>{DEDUP_IOU_THRESH}")
    print(f"  E. 跨切片去重:    重叠区 (80px) 同类框 IoU>0.3 → 保留中心更远离边缘的")
    print()

    # ── 执行清洗 ──
    total_original = 0
    total_kept = 0
    all_reasons = Counter()
    per_class_original = Counter()
    per_class_kept = Counter()

    for split in args.splits:
        label_dir = TILE_DATASET / "labels" / split
        if not label_dir.exists():
            print(f"  ⚠ {split}/ 不存在，跳过")
            continue

        split_orig = 0
        split_kept = 0
        txt_files = sorted(label_dir.glob("*.txt"))
        print(f"  处理 {split}/: {len(txt_files)} 文件 ...", end="", flush=True)

        for txt in txt_files:
            # Count per-class before cleaning
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    per_class_original[int(parts[0])] += 1

            orig, kept, reasons = clean_label_file(
                txt, dry_run=args.dry_run,
                large_to_critical=args.large_to_critical,
            )
            split_orig += orig
            split_kept += kept
            all_reasons.update(reasons)

            # Count per-class after cleaning (re-read if not dry-run)
            if not args.dry_run:
                for line in txt.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        per_class_kept[int(parts[0])] += 1

        removed = split_orig - split_kept
        pct = (removed / max(split_orig, 1)) * 100
        print(f" {split_orig:,} → {split_kept:,} (移除 {removed:,}, {pct:.1f}%)")
        total_original += split_orig
        total_kept += split_kept

    # ── E: 跨切片重叠区去重 ──
    print()
    print("  E. 跨切片重叠区去重 ...", end="", flush=True)
    tile_index = load_tile_index()
    total_overlap_removed = 0
    if tile_index:
        for split in args.splits:
            n = cross_tile_dedup(split, tile_index, dry_run=args.dry_run)
            total_overlap_removed += n
    print(f" 移除 {total_overlap_removed:,} 个重叠区重复框")
    total_kept -= total_overlap_removed
    all_reasons["E_cross_tile_dedup"] = total_overlap_removed

    # ── 统计汇总 ──
    total_removed = total_original - total_kept
    pct = (total_removed / max(total_original, 1)) * 100

    print()
    print("═" * 56)
    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"  {tag}清洗汇总:")
    print(f"    原始标注: {total_original:,}")
    print(f"    保留标注: {total_kept:,}")
    print(f"    移除标注: {total_removed:,}  ({pct:.1f}%)")
    print()

    if not args.dry_run and per_class_kept:
        print("  清洗后各类分布:")
        for cls_id in sorted(per_class_kept):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            orig = per_class_original.get(cls_id, 0)
            kept = per_class_kept.get(cls_id, 0)
            removed = orig - kept
            print(f"    {name}:  {orig:>6,} → {kept:>6,}  (移除 {removed:,})")
        print()

    # Top removal reasons
    print("  移除原因 Top-15:")
    for reason, count in all_reasons.most_common(15):
        print(f"    {reason:40s}  {count:>5,}")
    # Summarize by category
    n_small = sum(v for k, v in all_reasons.items()
                  if "area=" in k or "min_side" in k or "aspect" in k)
    n_edge  = sum(v for k, v in all_reasons.items() if "edge_truncated" in k)
    n_giant = sum(v for k, v in all_reasons.items()
                  if "giant" in k or "large_scratch" in k or "→critical" in k)
    n_dedup = sum(v for k, v in all_reasons.items()
                  if "dedup" in k and "cross" not in k)
    n_cross = all_reasons.get("E_cross_tile_dedup", 0)
    print()
    print("  按类别汇总:")
    print(f"    A. 噪声小框:       {n_small:>6,}")
    print(f"    B. 边缘截断残留:   {n_edge:>6,}")
    print(f"    C. 满切片大框:     {n_giant:>6,}")
    print(f"    D. 切片内去重:     {n_dedup:>6,}")
    print(f"    E. 跨切片去重:     {n_cross:>6,}")
    print(f"    ─────────────────────────")
    print(f"    合计:              {total_removed:>6,}  ({pct:.1f}%)")

    print("═" * 56)

    if args.dry_run:
        print()
        print("  确认后运行:")
        print("      python scripts/step1_label_cleanup.py")
        print("  恢复原始标注:")
        print("      python scripts/step1_label_cleanup.py --restore")
    else:
        # Save cleanup report
        report = {
            "original_total": total_original,
            "kept_total": total_kept,
            "removed_total": total_removed,
            "removed_pct": round(pct, 1),
            "cross_tile_dedup": total_overlap_removed,
            "reasons": dict(all_reasons.most_common()),
        }
        report_path = TILE_DATASET / "cleanup_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print()
        print(f"  报告已保存: {report_path}")
        print()
        print("  下一步: 运行伪标签生成")
        print("      python scripts/step2_pseudo_labels.py")

    print()


if __name__ == "__main__":
    main()
