#!/usr/bin/env python3
"""
Step 4: 半监督 Scratch 重标注
==========================================
利用训练好的模型 + Frangi 传统算法，对 GT 标注进行三方面改进：

  A. **碎片合并**：将同一条划痕被切片截断后形成的多个 GT 碎片框
     在全图坐标系下重新连接为一个完整框，再分配回各切片。
  B. **漏标补全**：模型高置信度预测 + Frangi 线状响应验证 → 添加新 GT 框。
  C. **不完整扩展**：GT 框只覆盖了划痕的一小段，模型预测出更长的范围
     → 将 GT 扩展到模型预测的覆盖范围。

核心思想：
  模型学习了 "什么是划痕"，传统算法提供 "物理证据"，
  两者交叉验证产生的标注比任一单方更可靠。

使用方式:
    python scripts/step4_scratch_relabel.py --dry-run     # 预览不修改
    python scripts/step4_scratch_relabel.py               # 执行
    python scripts/step4_scratch_relabel.py --conf 0.25   # 调整置信度阈值

前置条件:
    - 已完成 step3（有训练好的模型权重）
    - output/tile_dataset/ 中有完整的切片数据

后续步骤:
    - step5_retrain.py 用改进后的 GT 重训练（Stage4）
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fullimage_utils import (
    tile_boxes_to_fullimage,
    fullimage_to_tile_boxes,
    nms_ios,
    connect_scratches,
    merge_gt_with_predictions,
    TILE_SIZE,
)
from darkfield_defects.detection.features import frangi_filter

TILE_DATASET  = PROJECT_ROOT / "output" / "tile_dataset"
TILE_IMG_DIR  = TILE_DATASET / "images"
TILE_LBL_DIR  = TILE_DATASET / "labels"
IMAGES_DIR    = PROJECT_ROOT / "output" / "dataset_v2" / "images"
AUDIT_DIR     = PROJECT_ROOT / "output" / "audit"
WEIGHTS_PATH  = (PROJECT_ROOT / "output" / "training" /
                 "stage2_cleaned" / "weights" / "best.pt")

# ── 参数 ──────────────────────────────────────────────────
CONF_THRESH       = 0.15   # 模型推理置信度（低阈值 → 多召回）
BATCH_SIZE        = 16     # 模型推理 batch size
CHUNK_SIZE        = 200    # 切片数/chunk（控制显存）

SCRATCH_GAP       = 100    # scratch 连接最大端点间距（像素）
SCRATCH_ANGLE     = 30     # scratch 连接最大角度差（度）
NMS_IOS_THRESH    = 0.35   # 全图 NMS IOS 阈值

ADD_CONF_THRESH   = 0.35   # 新增 GT 的最低模型置信度
EXTEND_RATIO      = 1.5    # 预测面积 > GT × 此值 → 扩展 GT

FRANGI_SIGMAS     = [1.0, 2.0, 3.0, 5.0]
FRANGI_ALPHA      = 0.5
FRANGI_BETA       = 0.5
FRANGI_GAMMA      = 15.0
FRANGI_ADD_THRESH  = 0.08  # 新增框的 Frangi 验证阈值

MAX_BOX_AREA_NORM  = 0.30  # 全图级：面积 > 30% 切片 → 排除


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def load_tile_index(split: str) -> dict[str, dict]:
    """加载 tile_index.csv，按 source 分组。"""
    idx_path = TILE_DATASET / "tile_index.csv"
    tiles: dict[str, dict] = {}
    with open(idx_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            row = dict(zip(header, vals))
            if row["split"] != split:
                continue
            tid = row["tile_id"]
            tiles[tid] = {
                "source": Path(row["source_image"]).stem,
                "split":  row["split"],
                "y0":     int(row["y0"]),
                "x0":     int(row["x0"]),
            }
    return tiles


def group_tiles_by_source(tiles: dict[str, dict]) -> dict[str, list]:
    """按 source image 分组。"""
    by_src: dict[str, list] = defaultdict(list)
    for tid, info in tiles.items():
        by_src[info["source"]].append((tid, info))
    return by_src


def load_gt_for_tile(tid: str, split: str) -> list[tuple]:
    """加载一个切片的 GT 标注。"""
    txt = TILE_LBL_DIR / split / f"{tid}.txt"
    if not txt.exists():
        return []
    boxes = []
    for line in txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]),
                          float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def compute_frangi_for_tile(tid: str, split: str) -> np.ndarray | None:
    """计算切片的 Frangi 响应图（懒计算）。"""
    for ext in (".jpg", ".png"):
        p = TILE_IMG_DIR / split / f"{tid}{ext}"
        if p.exists():
            break
    else:
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_norm = img.astype(np.float32) / 255.0
    return frangi_filter(img_norm, sigmas=FRANGI_SIGMAS,
                         alpha=FRANGI_ALPHA, beta=FRANGI_BETA,
                         gamma=FRANGI_GAMMA)


def frangi_max_in_box(frangi_map: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int) -> float:
    """返回像素坐标区域内的 Frangi 最大值。"""
    H, W = frangi_map.shape
    x1c = max(0, min(x1, W - 1))
    y1c = max(0, min(y1, H - 1))
    x2c = max(0, min(x2, W))
    y2c = max(0, min(y2, H))
    roi = frangi_map[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return 0.0
    return float(roi.max())


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}m{s:02d}s"


# ═══════════════════════════════════════════════════════
#  核心处理
# ═══════════════════════════════════════════════════════

def run_model_inference(model, tile_ids: list[str], split: str,
                        conf_thresh: float = CONF_THRESH) -> dict[str, list]:
    """
    对一组切片运行模型推理，返回 {tile_id: [(cls, cx, cy, w, h, conf)]}.
    """
    tile_preds: dict[str, list] = {}
    paths = []
    ids   = []

    for tid in tile_ids:
        for ext in (".jpg", ".png"):
            p = TILE_IMG_DIR / split / f"{tid}{ext}"
            if p.exists():
                paths.append(str(p))
                ids.append(tid)
                break

    if not paths:
        return tile_preds

    # 分 chunk 处理（控制显存）
    for ci in range(0, len(paths), CHUNK_SIZE):
        chunk_paths = paths[ci:ci + CHUNK_SIZE]
        chunk_ids   = ids[ci:ci + CHUNK_SIZE]

        results = model(
            chunk_paths, conf=conf_thresh, imgsz=640,
            batch=BATCH_SIZE, stream=True, verbose=False, half=True,
        )

        for idx, r in enumerate(results):
            tid = chunk_ids[idx]
            boxes = []
            if r.boxes is not None:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    xywh = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    # 面积过滤
                    if xywh[2] * xywh[3] > MAX_BOX_AREA_NORM:
                        continue
                    boxes.append((cls, xywh[0], xywh[1], xywh[2], xywh[3], conf))
            tile_preds[tid] = boxes

    return tile_preds


def process_source_image(
    stem: str,
    tile_list: list[tuple[str, dict]],
    tile_preds: dict[str, list],
    split: str,
    dry_run: bool = False,
    frangi_validate: bool = True,
) -> dict:
    """处理一张原图：全图级 NMS + 连接 + GT 合并。"""

    stats = {
        "n_tiles": len(tile_list),
        "gt_before": 0, "gt_after": 0,
        "extended": 0, "added": 0, "confirmed": 0,
        "scratch_chains": 0, "gt_chains": 0,
        "frangi_rejected": 0,
        "tiles_modified": 0,
    }

    # 1. 映射现有 GT 到全图坐标
    all_gt_full = []
    for tid, info in tile_list:
        gt_boxes = load_gt_for_tile(tid, split)
        stats["gt_before"] += len(gt_boxes)
        gt_with_conf = [(cls, cx, cy, w, h, 1.0) for cls, cx, cy, w, h in gt_boxes]
        full = tile_boxes_to_fullimage(gt_with_conf, info["x0"], info["y0"])
        all_gt_full.extend(full)

    # 2. 映射模型预测到全图坐标
    all_pred_full = []
    for tid, info in tile_list:
        preds = tile_preds.get(tid, [])
        full = tile_boxes_to_fullimage(preds, info["x0"], info["y0"])
        all_pred_full.extend(full)

    # 3. 全图 NMS
    #    GT 去重：高阈值（0.75），仅合并切片重叠区的完全重复框
    #    模型预测：低阈值（0.35），激进合并相邻检测
    gt_nms   = nms_ios(all_gt_full,   ios_thresh=0.75)
    pred_nms = nms_ios(all_pred_full, ios_thresh=NMS_IOS_THRESH)

    # 4. Scratch 连接
    #    注意：只连接模型预测，不连接 GT
    #    GT 保留碎片级标注用于训练（模型应该学习检测碎片，连接在推理时做）
    #    模型预测连接后用于发现 GT 中的漏标和不完整标注
    gt_connected = gt_nms  # GT 不做连接，保留原始碎片
    n_gt_chains = 0
    pred_connected, n_pred_chains = connect_scratches(pred_nms, SCRATCH_GAP, SCRATCH_ANGLE)
    stats["gt_chains"]      = n_gt_chains
    stats["scratch_chains"] = n_pred_chains

    # 5. GT + Prediction 合并
    merged, merge_stats = merge_gt_with_predictions(
        gt_connected, pred_connected,
        min_conf_add=ADD_CONF_THRESH,
        min_iou_match=0.15,
        extend_ratio=EXTEND_RATIO,
    )
    stats["extended"]  = merge_stats["extended"]
    stats["added"]     = merge_stats["added"]
    stats["confirmed"] = merge_stats["confirmed"]

    # 6. Frangi 验证新增框（只验证 added 的，不验证 extended）
    if frangi_validate and merge_stats["added"] > 0:
        # 找出新增框（在 merged 末尾）
        n_base = len(gt_connected)
        new_boxes = merged[n_base:]
        verified = merged[:n_base]

        # 需要 Frangi 验证的新增框：按切片位置分组
        for box in new_boxes:
            cls, x1, y1, x2, y2, conf = box
            if cls != 0:
                verified.append(box)
                continue
            # 找包含此框的切片，计算 Frangi
            frangi_ok = False
            for tid, info in tile_list:
                tx0, ty0 = info["x0"], info["y0"]
                # 检查框是否在此切片内
                if x2 < tx0 or x1 > tx0 + TILE_SIZE:
                    continue
                if y2 < ty0 or y1 > ty0 + TILE_SIZE:
                    continue
                # 计算切片内的 Frangi
                fmap = compute_frangi_for_tile(tid, split)
                if fmap is None:
                    continue
                # 转换到切片内坐标
                lx1 = max(0, int(x1 - tx0))
                ly1 = max(0, int(y1 - ty0))
                lx2 = min(TILE_SIZE, int(x2 - tx0))
                ly2 = min(TILE_SIZE, int(y2 - ty0))
                fval = frangi_max_in_box(fmap, lx1, ly1, lx2, ly2)
                if fval >= FRANGI_ADD_THRESH:
                    frangi_ok = True
                    break
            if frangi_ok:
                verified.append(box)
            else:
                stats["frangi_rejected"] += 1
                stats["added"] -= 1

        merged = verified

    # 6b. 最终全图 NMS：消除 connect_scratches + merge 引入的近重复框
    #     阈值 0.5（比预测 NMS 0.35 保守，保留真实相邻缺陷，消除边界重合框）
    merged = nms_ios(merged, ios_thresh=0.5)

    stats["gt_after_fullimg"] = len(merged)

    # 7. 全图框 → 分配回各切片的 YOLO 格式
    tile_total = 0
    for tid, info in tile_list:
        new_tile_boxes = fullimage_to_tile_boxes(
            merged, info["x0"], info["y0"],
            tile_size=TILE_SIZE,
            min_visible_frac=0.15,
        )
        # 7b. 切片级 NMS：消除全图投射后在切片内产生的包含关系
        if len(new_tile_boxes) > 1:
            tile_full = [(cls, cx, cy, w, h, 1.0)
                         for cls, cx, cy, w, h in new_tile_boxes]
            # 转换为像素坐标做 IOS NMS
            tile_px = [(cls, (cx - w/2) * TILE_SIZE, (cy - h/2) * TILE_SIZE,
                        (cx + w/2) * TILE_SIZE, (cy + h/2) * TILE_SIZE, 1.0)
                       for cls, cx, cy, w, h in new_tile_boxes]
            kept = nms_ios(tile_px, ios_thresh=0.5)
            # 转回 YOLO 归一化
            new_tile_boxes = [
                (cls,
                 ((x1 + x2) / 2) / TILE_SIZE,
                 ((y1 + y2) / 2) / TILE_SIZE,
                 (x2 - x1) / TILE_SIZE,
                 (y2 - y1) / TILE_SIZE)
                for cls, x1, y1, x2, y2, conf in kept
            ]
        tile_total += len(new_tile_boxes)

        old_gt = load_gt_for_tile(tid, split)
        if _boxes_differ(old_gt, new_tile_boxes):
            stats["tiles_modified"] += 1
            if not dry_run:
                label_path = TILE_LBL_DIR / split / f"{tid}.txt"
                lines = []
                for cls, cx, cy, w, h in new_tile_boxes:
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                label_path.write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )

    stats["gt_after_tile"] = tile_total
    return stats


def _boxes_differ(old: list[tuple], new: list[tuple]) -> bool:
    """检查新旧标注是否有差异。"""
    if len(old) != len(new):
        return True
    for o, n in zip(sorted(old), sorted(new)):
        for a, b in zip(o, n):
            if abs(a - b) > 0.002:
                return True
    return False


# ═══════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: 半监督 Scratch 重标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计，不修改文件")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "both"],
                        help="处理哪个 split（默认 train，建议只处理 train 以保持 val 评估基准不变）")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help=f"模型推理置信度阈值（默认 {CONF_THRESH}）")
    parser.add_argument("--no-frangi", action="store_true",
                        help="跳过 Frangi 验证（更快，但可能引入噪声）")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径（默认 Stage3 best.pt）")
    args = parser.parse_args()

    conf_thresh = args.conf

    weights = Path(args.weights) if args.weights else WEIGHTS_PATH
    if not weights.exists():
        print(f"  ✗ 模型权重不存在: {weights}")
        sys.exit(1)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Step 4: 半监督 Scratch 重标注                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    if args.dry_run:
        print("  [DRY RUN] 仅统计，不修改文件")
    print(f"  split:    {args.split}")
    print(f"  模型:     {weights.name}")
    print(f"  置信度:   {conf_thresh}")
    print(f"  连接参数: gap={SCRATCH_GAP}px, angle={SCRATCH_ANGLE}°")
    print(f"  Frangi:   {'启用' if not args.no_frangi else '禁用'}")
    print()

    # 备份标注
    if not args.dry_run:
        for sp in splits:
            bak_dir = TILE_LBL_DIR / f"{sp}_pre_step4_backup"
            if not bak_dir.exists():
                print(f"  备份 {sp} 标注 → {bak_dir.name}/")
                shutil.copytree(TILE_LBL_DIR / sp, bak_dir)
            else:
                print(f"  备份已存在: {bak_dir.name}/")

    # 加载模型
    print("  加载模型 ...", end="", flush=True)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(" OK")

    all_stats = defaultdict(int)
    t_start = time.time()

    for sp in splits:
        print(f"\n  ══ 处理 {sp} ══")

        # 加载切片索引
        tiles = load_tile_index(sp)
        by_source = group_tiles_by_source(tiles)
        n_images = len(by_source)
        print(f"  {n_images} 张原图, {len(tiles)} 个切片")

        # ── 模型推理（全部切片一次性）──
        print("  模型推理中 ...", end="", flush=True)
        t0 = time.time()
        all_tile_ids = list(tiles.keys())
        tile_preds = run_model_inference(model, all_tile_ids, sp,
                                         conf_thresh=conf_thresh)
        n_preds = sum(len(v) for v in tile_preds.values())
        print(f" {n_preds} 个预测框 ({fmt_time(time.time()-t0)})")

        # ── 逐图处理 ──
        print("  全图连接+合并中:")
        for idx, (stem, tile_list) in enumerate(sorted(by_source.items())):
            st = process_source_image(
                stem, tile_list, tile_preds, sp,
                dry_run=args.dry_run,
                frangi_validate=not args.no_frangi,
            )
            for k, v in st.items():
                all_stats[k] += v

            if (idx + 1) % 20 == 0 or idx == n_images - 1:
                elapsed = time.time() - t_start
                eta = elapsed / (idx + 1) * (n_images - idx - 1)
                print(f"    [{idx+1}/{n_images}] "
                      f"连接={all_stats['scratch_chains']}+{all_stats['gt_chains']} "
                      f"扩展={all_stats['extended']} "
                      f"新增={all_stats['added']} "
                      f"Frangi拒绝={all_stats['frangi_rejected']} "
                      f"({fmt_time(elapsed)}, ETA {fmt_time(eta)})")

    # ── 汇总 ──
    elapsed = time.time() - t_start
    print(f"\n{'='*56}")
    print(f"  Step 4 完成 {'(DRY RUN)' if args.dry_run else ''} ({fmt_time(elapsed)})")
    print(f"{'='*56}")
    print(f"  原切片 GT 框总数:     {all_stats['gt_before']:>8}")
    print(f"  唯一缺陷数(全图去重): {all_stats['gt_after_fullimg']:>8}")
    print(f"  新切片 GT 框总数:     {all_stats['gt_after_tile']:>8}")
    print(f"  ─────────────────────────────")
    print(f"  scratch 连接链:       {all_stats['scratch_chains']:>8}  (模型预测)")
    print(f"  GT 连接链:            {all_stats['gt_chains']:>8}  (现有 GT)")
    print(f"  GT 框扩展:            {all_stats['extended']:>8}  (不完整→完整)")
    print(f"  新增 GT 框:           {all_stats['added']:>8}  (模型+Frangi 确认)")
    print(f"  Frangi 拒绝新增:      {all_stats['frangi_rejected']:>8}")
    print(f"  模型确认 GT:          {all_stats['confirmed']:>8}")
    print(f"  修改的切片:           {all_stats['tiles_modified']:>8}")

    # 保存审计日志
    audit = {
        "stage": "step4_scratch_relabel",
        "splits": splits,
        "weights": str(weights),
        "conf_thresh": conf_thresh,
        "scratch_gap": SCRATCH_GAP,
        "scratch_angle": SCRATCH_ANGLE,
        "frangi_validate": not args.no_frangi,
        "dry_run": args.dry_run,
        "stats": dict(all_stats),
        "elapsed_sec": round(elapsed, 1),
    }
    audit_path = AUDIT_DIR / "step4_audit.json"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\n  审计日志: {audit_path}")

    if args.dry_run:
        print("\n  → 去掉 --dry-run 参数执行实际修改。")
    else:
        print("\n  → 下一步: 用改进后的 GT 重训练 (step5_retrain.py)")
    print()


if __name__ == "__main__":
    main()
