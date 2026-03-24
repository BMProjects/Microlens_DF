#!/usr/bin/env python3
"""
Step 1b: Frangi 响应反向验证 GT scratch 标注质量
================================================
用传统算法的 Frangi 线状响应图作为"验证器"，对 GT scratch 框进行双重验证：

  - 若框内 Frangi 最大响应 < NOISE_THRESH，且模型置信度 < MODEL_CONF_THRESH
    → 自动删除（传统算法和模型均认为该处无划痕）

  - 若框内 Frangi 响应为中等（NOISE_THRESH ~ REVIEW_THRESH）
    → 写入人工审核队列（suspect_tiles.csv 高优先级）

  - Frangi 响应 > REVIEW_THRESH
    → 保留（有明确线状结构证据）

核心思想：
  传统算法（ClassicalDetector）产生了 GT，其 Frangi 响应图是该标注的
  "物理依据"。若一个 GT scratch 框内 Frangi 完全无响应，说明该框是
  算法噪声而非真实划痕，模型无法从中学习，应当删除。

使用方式：
    python scripts/step1b_frangi_verify.py --dry-run      # 预览，不修改
    python scripts/step1b_frangi_verify.py                # 执行（只处理 train+val）
    python scripts/step1b_frangi_verify.py --split train  # 只处理 train

配合 step1.py 使用：先跑 step1（规则清洗），再跑 step1b（Frangi 验证）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── 项目路径 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from darkfield_defects.detection.features import frangi_filter

TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
AUDIT_DIR    = PROJECT_ROOT / "output" / "audit"

# ── 阈值配置 ─────────────────────────────────────────────────

# 双重验证阈值
NOISE_FRANGI_THRESH  = 0.05   # 框内 Frangi_max < 此值 → Frangi认为无划痕
REVIEW_FRANGI_THRESH = 0.30   # 框内 Frangi_max < 此值 → 弱响应，进入审核队列
MODEL_CONF_THRESH    = 0.15   # 模型对该 GT 框位置的最高预测置信度 < 此值 → 模型也不认

# Frangi 参数（与原始标注保持一致）
FRANGI_SIGMAS = [1.0, 2.0, 3.0, 5.0]
FRANGI_ALPHA  = 0.5
FRANGI_BETA   = 0.5
FRANGI_GAMMA  = 15.0

TILE_SIZE = 640


def compute_frangi(img_gray: np.ndarray) -> np.ndarray:
    """计算切片的 Frangi 线状响应图，返回 [0,1] 归一化结果。"""
    img_norm = img_gray.astype(np.float32) / 255.0
    resp = frangi_filter(img_norm, sigmas=FRANGI_SIGMAS,
                         alpha=FRANGI_ALPHA, beta=FRANGI_BETA, gamma=FRANGI_GAMMA)
    return resp


def frangi_response_in_box(frangi_map: np.ndarray,
                           cx: float, cy: float, w: float, h: float) -> tuple[float, float]:
    """
    返回 GT 框内的 Frangi (max, mean)。
    框坐标为 YOLO 归一化格式。
    """
    H, W = frangi_map.shape
    x1 = max(0, int((cx - w / 2) * W))
    y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W - 1, int((cx + w / 2) * W))
    y2 = min(H - 1, int((cy + h / 2) * H))
    roi = frangi_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0
    return float(roi.max()), float(roi.mean())


def load_pred_conf_at_box(pred_file: Path,
                          cx: float, cy: float, w: float, h: float) -> float:
    """
    在预测文件中查找与 GT 框重叠最大的预测框的置信度。
    若无重叠预测，返回 0.0。
    """
    if not pred_file.exists():
        return 0.0

    def iou(b1, b2):
        x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
        x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
        x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
        x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
        iw = max(0, min(x2a, x2b) - max(x1a, x1b))
        ih = max(0, min(y2a, y2b) - max(y1a, y1b))
        inter = iw * ih
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / max(union, 1e-8)

    best_conf = 0.0
    for line in pred_file.read_text().strip().split("\n"):
        if not line.strip():
            continue
        p = line.strip().split()
        if int(p[0]) != 0:   # 只考虑 scratch 类
            continue
        pcx, pcy, pw, ph = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        conf = float(p[5]) if len(p) > 5 else 1.0
        v = iou((cx, cy, w, h), (pcx, pcy, pw, ph))
        if v > 0.1 and conf > best_conf:  # 10% IoU 即可（松散匹配）
            best_conf = conf

    return best_conf


def process_split(split: str, dry_run: bool = False) -> dict:
    """处理一个 split（train/val）的所有切片标注。"""
    label_dir = TILE_DATASET / "labels" / split
    img_dir   = TILE_DATASET / "images" / split
    pred_dir  = AUDIT_DIR / "predictions" / "labels"

    label_files = sorted(label_dir.glob("*.txt"))
    print(f"\n  处理 {split}: {len(label_files)} 个切片")

    stats = {
        "total_scratch": 0,
        "auto_removed": 0,
        "flagged_review": 0,
        "kept": 0,
        "tiles_modified": 0,
        "review_queue": [],   # [(tile_id, frangi_max, model_conf, box_area)]
    }

    for label_path in label_files:
        img_path = img_dir / (label_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (label_path.stem + ".png")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        lines = label_path.read_text().strip().split("\n")
        new_lines = []
        tile_modified = False
        frangi_map = None  # 懒计算

        for line in lines:
            if not line.strip():
                continue
            p = line.strip().split()
            cls = int(p[0])

            # 非 scratch 框不处理
            if cls != 0:
                new_lines.append(line)
                continue

            cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            stats["total_scratch"] += 1

            # 懒计算 Frangi
            if frangi_map is None:
                frangi_map = compute_frangi(img)

            f_max, f_mean = frangi_response_in_box(frangi_map, cx, cy, w, h)

            # === 自动删除 ===
            if f_max < NOISE_FRANGI_THRESH:
                # 进一步用模型置信度确认
                pred_file = pred_dir / (label_path.stem + ".txt")
                model_conf = load_pred_conf_at_box(pred_file, cx, cy, w, h)

                if model_conf < MODEL_CONF_THRESH:
                    stats["auto_removed"] += 1
                    tile_modified = True
                    continue   # 不加入 new_lines → 等效删除
                else:
                    # 模型认为有，Frangi 不认为有 → 可能是细划痕，标记审核
                    stats["flagged_review"] += 1
                    stats["review_queue"].append({
                        "tile": label_path.stem,
                        "split": split,
                        "reason": "low_frangi_but_model_conf",
                        "frangi_max": round(f_max, 4),
                        "model_conf": round(model_conf, 3),
                        "box": [round(cx,4), round(cy,4), round(w,4), round(h,4)],
                    })
                    new_lines.append(line)

            # === 标记弱响应审核 ===
            elif f_max < REVIEW_FRANGI_THRESH:
                stats["flagged_review"] += 1
                stats["review_queue"].append({
                    "tile": label_path.stem,
                    "split": split,
                    "reason": "weak_frangi",
                    "frangi_max": round(f_max, 4),
                    "model_conf": -1.0,
                    "box": [round(cx,4), round(cy,4), round(w,4), round(h,4)],
                })
                new_lines.append(line)

            # === 保留 ===
            else:
                stats["kept"] += 1
                new_lines.append(line)

        if tile_modified:
            stats["tiles_modified"] += 1
            if not dry_run:
                label_path.write_text("\n".join(new_lines) + "\n" if new_lines else "")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Frangi 验证 GT scratch 标注")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不修改文件")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = parser.parse_args()

    dry_run = args.dry_run
    splits  = ["train", "val"] if args.split == "both" else [args.split]

    print(f"{'[DRY RUN] ' if dry_run else ''}Frangi GT 验证")
    print(f"  阈值: Frangi_max < {NOISE_FRANGI_THRESH} + ModelConf < {MODEL_CONF_THRESH} → 自动删除")
    print(f"         Frangi_max < {REVIEW_FRANGI_THRESH} → 标记审核")

    all_stats = defaultdict(int)
    review_queue = []

    for split in splits:
        st = process_split(split, dry_run=dry_run)
        for k in ["total_scratch", "auto_removed", "flagged_review", "kept", "tiles_modified"]:
            all_stats[k] += st[k]
        review_queue.extend(st["review_queue"])

    # 输出汇总
    total = all_stats["total_scratch"]
    removed = all_stats["auto_removed"]
    flagged = all_stats["flagged_review"]
    kept    = all_stats["kept"]

    print(f"\n=== 结果汇总 {'(DRY RUN)' if dry_run else ''} ===")
    print(f"  总 scratch GT 框:    {total:>7}")
    print(f"  自动删除（噪声）:    {removed:>7} ({removed/max(total,1)*100:.1f}%)")
    print(f"  标记人工审核（弱）:  {flagged:>7} ({flagged/max(total,1)*100:.1f}%)")
    print(f"  保留（强 Frangi）:   {kept:>7} ({kept/max(total,1)*100:.1f}%)")
    print(f"  修改的切片文件:      {all_stats['tiles_modified']:>7}")

    # 保存审核队列（按 tile 聚合）
    if review_queue:
        # 按 tile 分组，每个 tile 取最低 frangi_max 的作为代表
        by_tile: dict[str, list] = defaultdict(list)
        for item in review_queue:
            by_tile[item["tile"]].append(item)

        queue_path = AUDIT_DIR / "frangi_review_queue.json"
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

        # 每个 tile 取平均 Frangi 分作为优先级分
        tile_entries = []
        for tile, items in by_tile.items():
            avg_frangi = sum(i["frangi_max"] for i in items) / len(items)
            tile_entries.append({
                "tile": tile,
                "split": items[0]["split"],
                "n_weak_boxes": len(items),
                "avg_frangi_max": round(avg_frangi, 4),
                "boxes": items,
            })

        # 按 avg_frangi 升序（最可疑的排前面）
        tile_entries.sort(key=lambda x: x["avg_frangi_max"])

        if not dry_run:
            queue_path.write_text(json.dumps(tile_entries, ensure_ascii=False, indent=2))
            print(f"\n  审核队列保存至: {queue_path}")
            print(f"  共 {len(tile_entries)} 个切片需要人工审核")
        else:
            print(f"\n  [DRY RUN] 审核队列 {len(tile_entries)} 个切片（最可疑 top5）:")
            for entry in tile_entries[:5]:
                print(f"    {entry['tile']}: {entry['n_weak_boxes']} 个弱框, "
                      f"avg_frangi={entry['avg_frangi_max']:.4f}")

    print("\n完成。")
    if dry_run:
        print("→ 确认无误后去掉 --dry-run 参数执行实际修改。")
    else:
        print("→ 下一步：运行 step2_pseudo_labels.py 重新生成伪标签，然后 step3_retrain.py 重训练。")


if __name__ == "__main__":
    main()
