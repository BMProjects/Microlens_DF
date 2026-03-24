#!/usr/bin/env python3
"""切图 + 半自动标注数据集构建脚本.

功能：
  - 将预处理图像切割为 640×640 tile（带 overlap）
  - 用经典检测器（gamma+CLAHE 增强辅助）对每个 tile 进行检测
  - 输出 YOLO 格式标注（坐标基于原始 tile，非增强图）
  - 同时保存标注叠加图 + 增强参考图，供人工审核
  - 生成 defects.yaml 数据集配置文件

增强策略：
  增强图仅用于驱动检测器找到更多缺陷，标注坐标映射回原始 tile。
  训练集只使用原始 tile 图像，不存在域差距问题。

用法：
  python scripts/build_tile_dataset.py [--n-sample N] [--seed SEED] [--all]

示例：
  python scripts/build_tile_dataset.py --n-sample 10  # 测试10张
  python scripts/build_tile_dataset.py --all          # 全量247张
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.params import load_params
from darkfield_defects.detection.base import DefectType

# ── 常量 ─────────────────────────────────────────────────────────────────────

TILE_SIZE = 640
OVERLAP = 80
STRIDE = TILE_SIZE - OVERLAP
ROI_COVERAGE_MIN = 0.20       # tile 内 ROI 覆盖率低于此值则跳过
TRAIN_RATIO = 0.80

DEFECT_COLORS = {
    DefectType.SCRATCH: (0, 255, 0),      # 绿色
    DefectType.SPOT:    (0, 128, 255),    # 橙色
    DefectType.DAMAGE:  (180, 0, 255),    # 紫红（与 CRASH 同色系，已合并）
    DefectType.CRASH:   (255, 0, 200),    # 品红（与 DAMAGE 同色系，已合并）
}
# YOLO 3-class 映射：DAMAGE 和 CRASH 合并为 critical(2)
# 原因：DAMAGE 仅 46 实例（0.045%）无法独立学习；两者视觉上均为"大片高亮区域"
# 内部评分逻辑仍使用 DefectType 完整 4 类区分
CLASS_MAP = {
    DefectType.SCRATCH: 0,
    DefectType.SPOT:    1,
    DefectType.DAMAGE:  2,   # → critical
    DefectType.CRASH:   2,   # → critical（与 DAMAGE 合并）
}
CLASS_NAMES = ["scratch", "spot", "critical"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def iter_tiles(img_h: int, img_w: int):
    """生成所有 tile 的 (y0, x0) 起始坐标."""
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            yield y, x
            x = min(x + STRIDE, img_w - TILE_SIZE) if x + TILE_SIZE < img_w else img_w
            if x >= img_w:
                break
        y = min(y + STRIDE, img_h - TILE_SIZE) if y + TILE_SIZE < img_h else img_h
        if y >= img_h:
            break


def iter_tiles_v2(img_h: int, img_w: int):
    """生成覆盖全图的 tile 起始坐标（末尾 tile 向前对齐）."""
    import math
    nx = max(1, math.ceil((img_w - OVERLAP) / STRIDE))
    ny = max(1, math.ceil((img_h - OVERLAP) / STRIDE))
    for iy in range(ny):
        y0 = min(iy * STRIDE, img_h - TILE_SIZE)
        for ix in range(nx):
            x0 = min(ix * STRIDE, img_w - TILE_SIZE)
            yield y0, x0


def instances_to_yolo(instances, tile_h: int = TILE_SIZE, tile_w: int = TILE_SIZE):
    """将 DefectInstance 列表转为 YOLO 标注行列表."""
    lines = []
    for inst in instances:
        cls_id = CLASS_MAP.get(inst.defect_type, 1)
        x, y, w, h = inst.bbox
        # 确保 bbox 在 tile 范围内
        x = max(0, x); y = max(0, y)
        w = min(w, tile_w - x); h = min(h, tile_h - y)
        if w < 2 or h < 2:
            continue
        cx = (x + w / 2) / tile_w
        cy = (y + h / 2) / tile_h
        nw = w / tile_w
        nh = h / tile_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def draw_overlay(tile_orig: np.ndarray, instances) -> np.ndarray:
    """在原始 tile 上绘制检测框和类型标签."""
    vis = cv2.cvtColor(tile_orig, cv2.COLOR_GRAY2BGR)
    for inst in instances:
        color = DEFECT_COLORS.get(inst.defect_type, (255, 255, 255))
        x, y, w, h = inst.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        label = f"{inst.defect_type.value[0].upper()}"
        cv2.putText(vis, label, (x, max(y - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return vis


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def build_dataset(
    img_dir: Path,
    roi_path: Path,
    out_dir: Path,
    config_path: Path,
    n_sample: int | None = None,
    seed: int = 42,
) -> None:
    t_start = time.perf_counter()

    # ── 初始化 ──
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    params = load_params(str(config_path))
    detector = ClassicalDetector(params.detection, scoring_params=params.scoring)

    roi_full = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi_full is None:
        raise FileNotFoundError(f"ROI mask not found: {roi_path}")

    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        raise FileNotFoundError(f"No PNG images in {img_dir}")

    # 随机采样
    if n_sample and n_sample < len(img_files):
        random.seed(seed)
        img_files = sorted(random.sample(img_files, n_sample))

    print("=" * 70)
    print("  暗场镜片缺陷 — 切图标注数据集构建")
    print("=" * 70)
    print(f"  图像目录 : {img_dir}")
    print(f"  输出目录 : {out_dir}")
    print(f"  处理图像 : {len(img_files)} 张")
    print(f"  tile 尺寸: {TILE_SIZE}×{TILE_SIZE}, overlap={OVERLAP}px, stride={STRIDE}px")
    print(f"  ROI 覆盖门槛: {ROI_COVERAGE_MIN*100:.0f}%")
    print(f"  配置文件 : {config_path}")
    print(f"  增强策略 : gamma={params.detection.enhance_gamma}, "
          f"CLAHE={'ON' if params.detection.clahe_enabled else 'OFF'}")
    print(f"  otsu_floor: {params.detection.otsu_floor}")
    print()

    # ── 统计 ──
    tile_index = []       # CSV 行
    all_counts: Counter = Counter()
    n_tiles_total = 0
    n_tiles_skipped = 0
    n_tiles_empty = 0

    img_h, img_w = roi_full.shape[:2]
    tile_coords = list(iter_tiles_v2(img_h, img_w))

    # 按图像序号决定 train/val 划分（整图为单位，避免同图 tile 跨集）
    random.seed(seed)
    shuffled = list(range(len(img_files)))
    random.shuffle(shuffled)
    n_train_imgs = int(len(shuffled) * TRAIN_RATIO)
    train_img_ids = set(shuffled[:n_train_imgs])

    for img_idx, fpath in enumerate(img_files):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [跳过] 无法读取: {fpath.name}")
            continue

        split = "train" if img_idx in train_img_ids else "val"
        img_name = fpath.stem
        img_tile_count = 0
        img_defect_count = 0

        for y0, x0 in tile_coords:
            y1 = y0 + TILE_SIZE
            x1 = x0 + TILE_SIZE

            # ROI 覆盖率检查
            roi_tile = roi_full[y0:y1, x0:x1]
            coverage = np.count_nonzero(roi_tile > 0) / (TILE_SIZE * TILE_SIZE)
            if coverage < ROI_COVERAGE_MIN:
                n_tiles_skipped += 1
                continue

            tile_orig = img[y0:y1, x0:x1].copy()
            roi_tile_bool = roi_tile > 0

            # 检测（增强图驱动，坐标在 tile 坐标系内）
            try:
                result = detector.detect(
                    image=tile_orig,
                    roi_mask=roi_tile_bool,
                    preprocessed_image=tile_orig,
                )
            except Exception as e:
                print(f"  [警告] {img_name} tile({y0},{x0}) 检测失败: {e}")
                n_tiles_skipped += 1
                continue

            n_tiles_total += 1

            # YOLO 标注
            yolo_lines = instances_to_yolo(result.instances, TILE_SIZE, TILE_SIZE)
            tile_id = f"{img_name}_y{y0:04d}_x{x0:04d}"

            if not yolo_lines:
                n_tiles_empty += 1
                # 保留空标注文件（作为负样本，比例不超过50%则保留全部，否则按比例抽样）
                # 此处全部保留，后续可过滤

            # 保存原始 tile 图像（训练用，非增强）
            img_out = out_dir / "images" / split / f"{tile_id}.jpg"
            cv2.imwrite(str(img_out), tile_orig,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 保存 YOLO 标注
            lbl_out = out_dir / "labels" / split / f"{tile_id}.txt"
            lbl_out.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            # 保存标注叠加图（供人工审核）
            if result.instances:
                overlay = draw_overlay(tile_orig, result.instances)
                # 右侧拼接增强图
                enhanced = detector._enhance_darkfield(tile_orig) \
                    if params.detection.enhance_enabled else tile_orig
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                side_by_side = np.hstack([overlay, enhanced_bgr])
                cv2.imwrite(
                    str(out_dir / "overlays" / f"{tile_id}_overlay.jpg"),
                    side_by_side,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )

            # 更新统计
            counts = Counter(x.defect_type.value for x in result.instances)
            all_counts.update(counts)
            img_tile_count += 1
            img_defect_count += len(result.instances)

            tile_index.append({
                "tile_id": tile_id,
                "source_image": fpath.name,
                "split": split,
                "y0": y0, "x0": x0,
                "roi_coverage": f"{coverage:.3f}",
                "n_defects": len(result.instances),
                "n_scratch": counts.get("scratch", 0),
                "n_spot": counts.get("spot", 0),
                "n_damage": counts.get("damage", 0),
                "n_crash": counts.get("crash", 0),
            })

        elapsed = time.perf_counter() - t_start
        print(f"[{img_idx+1:3d}/{len(img_files)}] {fpath.name:20s}  "
              f"tiles={img_tile_count:3d}  defects={img_defect_count:4d}  "
              f"split={split}  elapsed={elapsed:.0f}s")

    # ── 写 CSV tile 索引 ──
    if tile_index:
        csv_path = out_dir / "tile_index.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tile_index[0].keys())
            writer.writeheader()
            writer.writerows(tile_index)

    # ── 写 YOLO defects.yaml ──
    n_train = sum(1 for r in tile_index if r["split"] == "train")
    n_val   = sum(1 for r in tile_index if r["split"] == "val")
    yaml_content = f"""# 暗场镜片缺陷检测数据集 — YOLO 格式
# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 图像数量: {len(img_files)} 张原图 → {n_tiles_total} 有效 tiles
# train: {n_train}  val: {n_val}

path: {out_dir.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# 类别说明（3-class YOLO 方案）
# 0: scratch  — 划痕（细长线状散射）
# 1: spot     — 斑点（近圆形点状散射）
# 2: critical — 严重缺陷区（大面积缺损 + 密集缺陷聚集 合并）
#
# 注: 内部检测器仍区分 damage / crash 用于磨损评分，
#     YOLO 标签层合并以解决 damage 极度稀缺问题（原 46 实例）
"""
    (out_dir / "defects.yaml").write_text(yaml_content)

    # ── 写 JSON 报告 ──
    elapsed_total = time.perf_counter() - t_start
    report = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_images": len(img_files),
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "tiles_total": n_tiles_total,
        "tiles_skipped_low_roi": n_tiles_skipped,
        "tiles_empty": n_tiles_empty,
        "tiles_with_defects": n_tiles_total - n_tiles_empty,
        "split": {"train": n_train, "val": n_val},
        "defect_counts": dict(all_counts),
        "elapsed_sec": round(elapsed_total, 1),
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    # ── 打印汇总 ──
    print()
    print("=" * 70)
    print(f"  构建完成  耗时 {elapsed_total/60:.1f} min")
    print(f"  原始图像  : {len(img_files)} 张")
    print(f"  有效 tile : {n_tiles_total}  "
          f"(跳过低ROI: {n_tiles_skipped})")
    print(f"  含缺陷 tile: {n_tiles_total - n_tiles_empty}  "
          f"空白 tile: {n_tiles_empty}")
    print(f"  划分      : train={n_train}  val={n_val}")
    print(f"  缺陷统计  : {dict(all_counts)}")
    print(f"  输出目录  : {out_dir}")
    print(f"  数据集配置: {out_dir / 'defects.yaml'}")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="切图 + 半自动标注数据集构建")
    p.add_argument("--img-dir",
                   default="output/dataset_v2/images",
                   help="预处理图像目录")
    p.add_argument("--roi",
                   default="output/dataset_v2/roi_mask.png",
                   help="ROI 掩膜路径")
    p.add_argument("--out-dir",
                   default="output/tile_dataset",
                   help="输出目录")
    p.add_argument("--config",
                   default="configs/detect_tile.yaml",
                   help="检测参数配置文件")
    p.add_argument("--n-sample", type=int, default=None,
                   help="随机采样 N 张图像（调试用）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true",
                   help="处理全部图像（等效于不指定 --n-sample）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        img_dir=Path(args.img_dir),
        roi_path=Path(args.roi),
        out_dir=Path(args.out_dir),
        config_path=Path(args.config),
        n_sample=None if args.all else args.n_sample,
        seed=args.seed,
    )
