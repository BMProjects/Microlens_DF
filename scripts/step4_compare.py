#!/usr/bin/env python3
"""
Step4 标注前后对比可视化
========================
随机抽取 N 张源图，生成 step4 半监督重标注前后的全图对比图，
供人工粗略审核改进效果。

输出: output/audit/step4_compare/
  - {stem}_compare.jpg  : 左=Before / 右=After 并排对比
  - summary.txt         : 每张图的统计数据

使用方式:
    python scripts/step4_compare.py              # 随机10张
    python scripts/step4_compare.py -n 20        # 随机20张
    python scripts/step4_compare.py --seed 42    # 固定随机种子
    python scripts/step4_compare.py --stem 13r   # 指定具体图像
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
IMAGES_DIR   = PROJECT_ROOT / "output" / "dataset_v2" / "images"
AUDIT_DIR    = PROJECT_ROOT / "output" / "audit"
COMPARE_DIR  = AUDIT_DIR / "step4_compare"
TILE_SIZE    = 640
SCALE        = 0.35   # 全图缩放比例

CLASS_NAMES = ["scratch", "spot", "critical"]
# BGR: scratch=金黄  spot=青蓝  critical=紫
COLORS = [(255, 220, 0), (0, 230, 255), (80, 0, 255)]

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.fullimage_utils import nms_ios

# 差异标注颜色 (BGR)
COL_UNCHANGED = (80, 80, 80)    # 灰色 = 未改变
COL_EXTENDED  = (0, 165, 255)   # 橙色 = 位置扩展
COL_ADDED     = (0, 255, 60)    # 亮绿 = 新增
COL_REMOVED   = (0, 0, 220)     # 红色 = 删除


# ─── 数据加载 ──────────────────────────────────────────

def load_tile_index(split: str = "train") -> dict[str, dict]:
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
                "x0": int(row["x0"]),
                "y0": int(row["y0"]),
            }
    return idx


def load_labels(label_dir: Path) -> dict[str, list[tuple]]:
    labels: dict[str, list[tuple]] = {}
    for txt in label_dir.glob("*.txt"):
        boxes = []
        for line in txt.read_text().splitlines():
            p = line.strip().split()
            if len(p) >= 5:
                boxes.append((int(p[0]),
                               float(p[1]), float(p[2]),
                               float(p[3]), float(p[4])))
        labels[txt.stem] = boxes
    return labels


def group_by_source(tile_index: dict) -> dict[str, list[tuple]]:
    by_src: dict[str, list] = defaultdict(list)
    for tid, info in tile_index.items():
        by_src[info["source"]].append((tid, info))
    return by_src


# ─── 渲染 ─────────────────────────────────────────────

def load_source_img(stem: str) -> np.ndarray:
    for ext in (".png", ".jpg"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    # placeholder
    return np.zeros((3000, 4096, 3), dtype=np.uint8) + 30


def _aggregate_fullimg_boxes(tile_list, labels):
    """将所有切片标注聚合到全图坐标，用 IOS NMS 去除跨切片重复。"""
    raw = []
    for tid, info in tile_list:
        x0, y0 = info["x0"], info["y0"]
        for cls, cx, cy, w, h in labels.get(tid, []):
            px = x0 + cx * TILE_SIZE
            py = y0 + cy * TILE_SIZE
            pw = w * TILE_SIZE
            ph = h * TILE_SIZE
            x1 = px - pw / 2
            y1 = py - ph / 2
            x2 = px + pw / 2
            y2 = py + ph / 2
            raw.append((cls, x1, y1, x2, y2, 1.0))
    # IOS NMS 去除重叠切片产生的重复框
    return nms_ios(raw, ios_thresh=0.5)


def draw_boxes_on_img(img: np.ndarray,
                      stem: str,
                      tile_list: list[tuple],
                      labels: dict[str, list[tuple]],
                      box_color_fn=None,
                      thickness: int = 2) -> tuple[np.ndarray, int]:
    """在全图 img 上绘制去重后的 GT 框。"""
    unique_boxes = _aggregate_fullimg_boxes(tile_list, labels)
    n = 0
    for cls, x1, y1, x2, y2, conf in unique_boxes:
        color = COLORS[cls] if cls < len(COLORS) else (200, 200, 200)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)
        n += 1
    return img, n


def render_panel(stem: str,
                 base_img: np.ndarray,
                 tile_list: list[tuple],
                 labels: dict[str, list[tuple]],
                 title: str,
                 scale: float = SCALE) -> np.ndarray:
    """渲染一个面板（带标题和统计信息）。"""
    img = base_img.copy()
    img, n_total = draw_boxes_on_img(img, stem, tile_list, labels)

    # 各类统计（基于去重后的框）
    unique_boxes = _aggregate_fullimg_boxes(tile_list, labels)
    cls_cnt = [0, 0, 0]
    for box in unique_boxes:
        if box[0] < 3:
            cls_cnt[box[0]] += 1

    H, W = img.shape[:2]
    img_small = cv2.resize(img, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA)
    sh, sw = img_small.shape[:2]

    # 标题栏
    bar_h = 36
    bar = np.zeros((bar_h, sw, 3), dtype=np.uint8) + 25
    cv2.putText(bar, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    stats_txt = (f"total={n_total}  "
                 f"scratch={cls_cnt[0]}  spot={cls_cnt[1]}  crit={cls_cnt[2]}")
    cv2.putText(bar, stats_txt, (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    panel = np.vstack([bar, img_small])
    return panel, {"total": n_total, "scratch": cls_cnt[0],
                   "spot": cls_cnt[1], "critical": cls_cnt[2]}


def render_diff_panel(stem: str,
                      base_img: np.ndarray,
                      tile_list: list[tuple],
                      before_labels: dict[str, list[tuple]],
                      after_labels:  dict[str, list[tuple]],
                      scale: float = SCALE) -> np.ndarray:
    """
    差异面板：灰色=不变, 橙=扩展, 绿=新增, 红=删除。
    简化匹配：按中心点距离配对（容差 = 一个 TILE_SIZE/8）
    """
    img = base_img.copy()

    # 使用去重后的全图级框进行匹配
    before_nms = _aggregate_fullimg_boxes(tile_list, before_labels)
    after_nms  = _aggregate_fullimg_boxes(tile_list, after_labels)

    # 转换为 (cls, cx, cy, w, h) 格式便于中心点匹配
    def to_cxcywh(boxes):
        result = []
        for cls, x1, y1, x2, y2, conf in boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            result.append((cls, cx, cy, w, h))
        return result

    before_boxes = to_cxcywh(before_nms)
    after_boxes  = to_cxcywh(after_nms)

    MATCH_DIST = TILE_SIZE / 8   # 80px 容差

    matched_after = set()
    n_unchanged = n_extended = n_removed = n_added = 0

    # 对每个 before 框找最近的 after 框
    for bi, (bcls, bpx, bpy, bpw, bph) in enumerate(before_boxes):
        best_d = float("inf")
        best_j = -1
        for j, (acls, apx, apy, apw, aph) in enumerate(after_boxes):
            if j in matched_after:
                continue
            if acls != bcls:
                continue
            d = ((apx - bpx) ** 2 + (apy - bpy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_j = j

        bx1 = int(bpx - bpw / 2)
        by1 = int(bpy - bph / 2)
        bx2 = int(bpx + bpw / 2)
        by2 = int(bpy + bph / 2)

        if best_j >= 0 and best_d < MATCH_DIST:
            matched_after.add(best_j)
            acls, apx, apy, apw, aph = after_boxes[best_j]
            area_b = bpw * bph
            area_a = apw * aph
            ax1 = int(apx - apw / 2)
            ay1 = int(apy - aph / 2)
            ax2 = int(apx + apw / 2)
            ay2 = int(apy + aph / 2)

            if area_a > area_b * 1.15:
                # 扩展（橙色）
                cv2.rectangle(img, (bx1, by1), (bx2, by2), COL_REMOVED, 1)   # before dim
                cv2.rectangle(img, (ax1, ay1), (ax2, ay2), COL_EXTENDED, 2)  # after orange
                n_extended += 1
            else:
                # 不变（灰色细线）
                cv2.rectangle(img, (bx1, by1), (bx2, by2), COL_UNCHANGED, 1)
                n_unchanged += 1
        else:
            # 删除（红色）
            cv2.rectangle(img, (bx1, by1), (bx2, by2), COL_REMOVED, 2)
            n_removed += 1

    # 未匹配的 after 框 = 新增（亮绿）
    for j, (acls, apx, apy, apw, aph) in enumerate(after_boxes):
        if j not in matched_after:
            ax1 = int(apx - apw / 2)
            ay1 = int(apy - aph / 2)
            ax2 = int(apx + apw / 2)
            ay2 = int(apy + aph / 2)
            cv2.rectangle(img, (ax1, ay1), (ax2, ay2), COL_ADDED, 2)
            n_added += 1

    H, W = img.shape[:2]
    img_small = cv2.resize(img, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA)
    sh, sw = img_small.shape[:2]

    bar_h = 36
    bar = np.zeros((bar_h, sw, 3), dtype=np.uint8) + 25
    cv2.putText(bar, "DIFF (橙=扩展 绿=新增 红=删除 灰=不变)", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    diff_txt = (f"不变={n_unchanged}  扩展={n_extended}  "
                f"新增={n_added}  删除={n_removed}")
    cv2.putText(bar, diff_txt, (8, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    panel = np.vstack([bar, img_small])
    return panel, {
        "unchanged": n_unchanged, "extended": n_extended,
        "added": n_added, "removed": n_removed,
    }


# ─── 主函数 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step4 标注前后对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-n", type=int, default=10,
                        help="随机抽取张数（默认 10）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（默认随机）")
    parser.add_argument("--stem", type=str, default=None,
                        help="指定具体图像名（如 13r），可多个逗号分隔")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"],
                        help="数据集 split（默认 train）")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help=f"全图缩放比例（默认 {SCALE}）")
    args = parser.parse_args()

    backup_dir = TILE_DATASET / "labels" / f"{args.split}_pre_step4_backup"
    current_dir = TILE_DATASET / "labels" / args.split

    if not backup_dir.exists():
        print(f"  ✗ 备份目录不存在: {backup_dir}")
        print("    请先运行 step4_scratch_relabel.py")
        sys.exit(1)

    print()
    print("  加载标注数据 ...", end="", flush=True)
    tile_index  = load_tile_index(args.split)
    by_source   = group_by_source(tile_index)
    before_lbl  = load_labels(backup_dir)
    after_lbl   = load_labels(current_dir)
    print(f" {len(by_source)} 张原图")

    # 选取目标图像
    if args.stem:
        stems = [s.strip() for s in args.stem.split(",")]
        # 验证存在
        stems = [s for s in stems if s in by_source]
        if not stems:
            print(f"  ✗ 指定的图像不在 {args.split} split 中")
            sys.exit(1)
    else:
        all_stems = sorted(by_source.keys())
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.n, len(all_stems))
        stems = random.sample(all_stems, n)

    stems = sorted(stems)
    print(f"  选取 {len(stems)} 张图像: {', '.join(stems)}")

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"Step4 比对报告 — split: {args.split}",
        f"{'─'*70}",
        f"{'图像':12s} {'Before':>8} {'After':>8} "
        f"{'不变':>6} {'扩展':>6} {'新增':>6} {'删除':>6}",
        f"{'─'*70}",
    ]

    for stem in stems:
        print(f"  渲染 {stem} ...", end="", flush=True)
        tile_list = by_source[stem]
        base_img  = load_source_img(stem)

        # 面板1：Before
        pan_before, stats_b = render_panel(
            stem, base_img, tile_list, before_lbl,
            f"BEFORE  [{stem}]", scale=args.scale)

        # 面板2：After
        pan_after, stats_a = render_panel(
            stem, base_img, tile_list, after_lbl,
            f"AFTER  [{stem}]", scale=args.scale)

        # 面板3：Diff
        pan_diff, diff_stats = render_diff_panel(
            stem, base_img, tile_list, before_lbl, after_lbl, scale=args.scale)

        # 拼接 Before | Diff | After
        # 确保高度一致
        max_h = max(pan_before.shape[0], pan_after.shape[0], pan_diff.shape[0])

        def pad_h(panel, target_h):
            h, w = panel.shape[:2]
            if h < target_h:
                pad = np.zeros((target_h - h, w, 3), dtype=np.uint8) + 15
                return np.vstack([panel, pad])
            return panel

        pan_before = pad_h(pan_before, max_h)
        pan_after  = pad_h(pan_after,  max_h)
        pan_diff   = pad_h(pan_diff,   max_h)

        # 分隔线
        sep = np.zeros((max_h, 3, 3), dtype=np.uint8) + 60
        combined = np.hstack([pan_before, sep, pan_diff, sep, pan_after])

        # 顶部标题条
        title_bar = np.zeros((28, combined.shape[1], 3), dtype=np.uint8) + 15
        cv2.putText(title_bar,
                    f"Step4 前后对比 — {stem}  "
                    f"(Before: total={stats_b['total']}  "
                    f"After: total={stats_a['total']}  "
                    f"Delta={stats_a['total']-stats_b['total']:+d})",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        final = np.vstack([title_bar, combined])
        out_path = COMPARE_DIR / f"{stem}_compare.jpg"
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 88])

        print(f" before={stats_b['total']} after={stats_a['total']} "
              f"(+{diff_stats['added']}新增 "
              f"+{diff_stats['extended']}扩展 "
              f"-{diff_stats['removed']}删除)")

        summary_lines.append(
            f"{stem:12s} {stats_b['total']:>8} {stats_a['total']:>8} "
            f"{diff_stats['unchanged']:>6} {diff_stats['extended']:>6} "
            f"{diff_stats['added']:>6} {diff_stats['removed']:>6}"
        )

    summary_lines.append(f"{'─'*70}")
    summary_txt = "\n".join(summary_lines)
    (COMPARE_DIR / "summary.txt").write_text(summary_txt + "\n")

    print()
    print(f"  输出目录: {COMPARE_DIR}/")
    print(f"  对比图:   {len(stems)} 张 (*_compare.jpg)")
    print()
    print(summary_txt)
    print()


if __name__ == "__main__":
    main()
