#!/usr/bin/env python3
"""
DAMAGE 类过采样与增强脚本
==============================
暗场显微镜镜片缺陷检测 — 解决 damage 类严重不足问题

问题: 数据集中 damage 仅 46 个实例（0.045%），导致模型无法学习
解决: 将含 damage 标注的 tile 过采样 ×15，生成 ~690 个增强副本

增强操作（针对暗场灰度图像）:
  - 翻转: 水平/垂直/对角（3 种）
  - 旋转: 90°/180°/270°（3 种）
  - 亮度扰动: ±10%, ±20%（2 种）
  - 高斯噪声: σ=3, σ=6（2 种）
  - 对比度微调: CLAHE clip=1.5 vs 2.5（2 种）

使用方式:
    python scripts/damage_augment.py
    python scripts/damage_augment.py --multiplier 20
    python scripts/damage_augment.py --dry-run   # 只统计，不生成

输出:
    output/tile_dataset/images/train_damage_aug/    增强 tile
    output/tile_dataset/labels/train_damage_aug/    对应标注
    output/tile_dataset/defects_augmented.yaml      合并后训练集配置
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
TRAIN_IMG_DIR = TILE_DATASET / "images" / "train"
TRAIN_LBL_DIR = TILE_DATASET / "labels" / "train"
AUG_IMG_DIR   = TILE_DATASET / "images" / "train_damage_aug"
AUG_LBL_DIR   = TILE_DATASET / "labels" / "train_damage_aug"

# 3-class 方案下：critical(2) = 原 damage(2) + crash(3)
# 脚本目标：过采样含 critical 标注的 tile（原 damage 比 crash 仍然更稀少）
CRITICAL_CLS_ID = 2   # critical 在 3-class 方案中的 ID
CLASS_NAMES = ["scratch", "spot", "critical"]


# ─── 辅助函数 ──────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Microlens_DF  DAMAGE 类过采样增强                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def has_critical(label_path: Path) -> bool:
    """检查标注文件中是否含有 critical 类（原 damage 或 crash，现均为 id=2）。"""
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == CRITICAL_CLS_ID:
                return True
    return False


def transform_boxes_flip(boxes: list[str], flip_code: int) -> list[str]:
    """翻转操作后修正 YOLO 格式坐标。

    flip_code: 0=上下翻转, 1=左右翻转, -1=对角翻转
    """
    new_boxes = []
    for line in boxes:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if flip_code == 0:    # 上下翻转: y → 1-y
            cy = 1.0 - cy
        elif flip_code == 1:  # 左右翻转: x → 1-x
            cx = 1.0 - cx
        else:                 # 对角翻转: x → 1-x, y → 1-y
            cx = 1.0 - cx
            cy = 1.0 - cy
        new_boxes.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return new_boxes


def transform_boxes_rotate(boxes: list[str], k: int) -> list[str]:
    """旋转 k×90° 后修正 YOLO 格式坐标（图像尺寸假设为正方形）。

    k=1: 逆时针 90°
    k=2: 180°
    k=3: 顺时针 90°
    """
    new_boxes = []
    for line in boxes:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        for _ in range(k % 4):
            # 逆时针 90°: (cx, cy) → (cy, 1-cx), (bw, bh) → (bh, bw)
            cx, cy = cy, 1.0 - cx
            bw, bh = bh, bw
        new_boxes.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return new_boxes


# ─── 增强操作列表 ──────────────────────────────────────────

def aug_fliph(img, boxes):
    """左右翻转."""
    return cv2.flip(img, 1), transform_boxes_flip(boxes, 1)

def aug_flipv(img, boxes):
    """上下翻转."""
    return cv2.flip(img, 0), transform_boxes_flip(boxes, 0)

def aug_flipd(img, boxes):
    """对角翻转."""
    return cv2.flip(img, -1), transform_boxes_flip(boxes, -1)

def aug_rot90(img, boxes):
    """逆时针 90°."""
    return np.rot90(img, 1).copy(), transform_boxes_rotate(boxes, 1)

def aug_rot180(img, boxes):
    """180°."""
    return np.rot90(img, 2).copy(), transform_boxes_rotate(boxes, 2)

def aug_rot270(img, boxes):
    """顺时针 90°."""
    return np.rot90(img, 3).copy(), transform_boxes_rotate(boxes, 3)

def aug_bright_up(img, boxes):
    """亮度提升 20%."""
    return np.clip(img.astype(np.float32) * 1.20, 0, 255).astype(np.uint8), boxes[:]

def aug_bright_down(img, boxes):
    """亮度降低 10%."""
    return np.clip(img.astype(np.float32) * 0.90, 0, 255).astype(np.uint8), boxes[:]

def aug_bright_up2(img, boxes):
    """亮度提升 10%."""
    return np.clip(img.astype(np.float32) * 1.10, 0, 255).astype(np.uint8), boxes[:]

def aug_bright_down2(img, boxes):
    """亮度降低 20%."""
    return np.clip(img.astype(np.float32) * 0.80, 0, 255).astype(np.uint8), boxes[:]

def aug_noise_light(img, boxes):
    """轻微高斯噪声 σ=3."""
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), boxes[:]

def aug_noise_medium(img, boxes):
    """中等高斯噪声 σ=6."""
    noise = np.random.normal(0, 6, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8), boxes[:]

def aug_clahe_low(img, boxes):
    """CLAHE clip=1.5（轻度对比度增强）."""
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(img), boxes[:]

def aug_clahe_high(img, boxes):
    """CLAHE clip=2.5（强度对比度增强）."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img), boxes[:]

# 所有增强操作（12 种）
ALL_AUGMENTATIONS = [
    ("fliph",        aug_fliph),
    ("flipv",        aug_flipv),
    ("flipd",        aug_flipd),
    ("rot90",        aug_rot90),
    ("rot180",       aug_rot180),
    ("rot270",       aug_rot270),
    ("bright_up",    aug_bright_up),
    ("bright_down",  aug_bright_down),
    ("bright_up2",   aug_bright_up2),
    ("bright_down2", aug_bright_down2),
    ("noise_light",  aug_noise_light),
    ("noise_medium", aug_noise_medium),
    # 组合增强（增加多样性）
    ("rot90_fliph",  lambda img, boxes: aug_fliph(*aug_rot90(img, boxes))),
    ("rot180_flipv", lambda img, boxes: aug_flipv(*aug_rot180(img, boxes))),
]


def generate_augmentations(
    img: np.ndarray,
    boxes: list[str],
    multiplier: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, list[str], str]]:
    """为一张 tile 生成 multiplier 个增强副本.

    Returns:
        list of (aug_img, aug_boxes, aug_suffix)
    """
    random.seed(seed)
    np.random.seed(seed)

    results = []
    augs = ALL_AUGMENTATIONS[:]

    # 如果 multiplier > 基础操作数，添加随机组合
    while len(augs) < multiplier:
        # 随机组合两种基础操作
        a1 = random.choice(ALL_AUGMENTATIONS[:6])  # 几何变换
        a2 = random.choice(ALL_AUGMENTATIONS[6:])  # 亮度/噪声
        name = f"{a1[0]}_{a2[0]}"
        fn1, fn2 = a1[1], a2[1]
        def combined(img, boxes, _fn1=fn1, _fn2=fn2):
            return _fn2(*_fn1(img, boxes))
        augs.append((name, combined))

    # 选取前 multiplier 个（确保多样性）
    selected = augs[:multiplier]

    for suffix, aug_fn in selected:
        try:
            aug_img, aug_boxes = aug_fn(img, boxes[:])
            results.append((aug_img, aug_boxes, suffix))
        except Exception as e:
            pass  # 跳过失败的增强

    return results


def write_augmented_yaml(
    original_yaml: Path,
    aug_img_dir: Path,
    aug_lbl_dir: Path,
    output_yaml: Path,
):
    """生成合并了过采样数据的新 YAML 配置。"""
    with open(original_yaml) as f:
        content = f.read()

    # 在 train 路径后添加过采样目录
    aug_img_relative = aug_img_dir.relative_to(TILE_DATASET)

    new_content = content.replace(
        "train: images/train",
        "train:\n  - images/train\n  - images/train_damage_aug",
    )
    # 更新注释标题
    new_content = "# Dark-field defect dataset — 含 DAMAGE 过采样增强版本\n" + new_content.lstrip("#").lstrip()

    with open(output_yaml, "w") as f:
        f.write(new_content)


def main():
    parser = argparse.ArgumentParser(
        description="DAMAGE 类过采样增强脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--multiplier", type=int, default=15,
                        help="每个 damage tile 生成的增强副本数（默认 15）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计，不实际生成文件")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    args = parser.parse_args()

    print_header()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 检查目录 ──────────────────────────────────────────
    if not TRAIN_IMG_DIR.exists():
        print(f"  ✗ 训练集不存在: {TRAIN_IMG_DIR}")
        print("  请先运行: bash scripts/run_build_dataset.sh")
        sys.exit(1)

    # ── 找出含 damage 的 tile ─────────────────────────────
    print("  扫描含 critical 标注的 tile ...")
    critical_tiles: list[tuple[Path, Path]] = []
    label_files = sorted(TRAIN_LBL_DIR.glob("*.txt"))

    for lbl_path in label_files:
        if has_critical(lbl_path):
            img_stem = lbl_path.stem
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = TRAIN_IMG_DIR / f"{img_stem}{ext}"
                if img_path.exists():
                    critical_tiles.append((img_path, lbl_path))
                    break

    n_critical = len(critical_tiles)
    print(f"  找到 {n_critical} 个含 critical 的 tile")
    print(f"  过采样倍率: ×{args.multiplier}")
    print(f"  预计生成: {n_critical * args.multiplier} 个增强副本")
    print()
    if n_critical > 2000:
        print(f"  提示: critical 已有 {n_critical} 个 tile（合并后数量充足），")
        print(f"        过采样倍率 ×{args.multiplier} 可能不必要。")
        print(f"        建议改为 --multiplier 3 或直接跳过此步骤。")
        print()

    if n_critical == 0:
        print("  ✗ 未找到含 critical 标注的 tile，请先运行标注迁移脚本:")
        print("      python scripts/relabel_merge_critical.py")
        sys.exit(1)

    # 向后兼容变量名
    damage_tiles = critical_tiles
    n_damage = n_critical

    if args.dry_run:
        print("  (dry-run 模式，不生成实际文件)")
        return

    # ── 创建输出目录 ──────────────────────────────────────
    AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    AUG_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 生成增强副本 ──────────────────────────────────────
    t0 = time.time()
    total_generated = 0
    per_class_after = {name: 0 for name in CLASS_NAMES}

    for tile_idx, (img_path, lbl_path) in enumerate(damage_tiles):
        # 读取图像
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ⚠ 无法读取: {img_path.name}")
            continue

        # 读取标注
        with open(lbl_path) as f:
            boxes = [line.strip() for line in f if line.strip()]

        # 统计类别
        for box in boxes:
            parts = box.split()
            if parts:
                cls_id = int(parts[0])
                if cls_id < len(CLASS_NAMES):
                    per_class_after[CLASS_NAMES[cls_id]] += 1

        # 生成增强
        aug_list = generate_augmentations(
            img, boxes, args.multiplier, seed=args.seed + tile_idx
        )

        for aug_img, aug_boxes, suffix in aug_list:
            stem = img_path.stem
            aug_name = f"{stem}_daug_{suffix}"
            out_img = AUG_IMG_DIR / f"{aug_name}.jpg"
            out_lbl = AUG_LBL_DIR / f"{aug_name}.txt"

            cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out_lbl, "w") as f:
                f.write("\n".join(aug_boxes) + "\n")

            # 统计增强后类别
            for box in aug_boxes:
                parts = box.split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(CLASS_NAMES):
                        per_class_after[CLASS_NAMES[cls_id]] += 1

            total_generated += 1

        # 进度显示
        progress = (tile_idx + 1) / n_damage * 100
        elapsed = time.time() - t0
        eta = elapsed / max(tile_idx + 1, 1) * (n_damage - tile_idx - 1)
        print(f"\r  进度: {tile_idx+1}/{n_damage} ({progress:.0f}%)  "
              f"已生成: {total_generated}  剩余: {fmt_time(eta)}  ",
              end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r  增强完成！生成 {total_generated} 张，耗时 {fmt_time(elapsed)}  ")
    print()

    # ── 生成合并 YAML ─────────────────────────────────────
    original_yaml = TILE_DATASET / "defects.yaml"
    aug_yaml      = TILE_DATASET / "defects_augmented.yaml"

    if original_yaml.exists():
        write_augmented_yaml(original_yaml, AUG_IMG_DIR, AUG_LBL_DIR, aug_yaml)
        print(f"  合并 YAML 已生成: {aug_yaml}")
    else:
        print(f"  ⚠ 原始 defects.yaml 不存在，请手动创建 defects_augmented.yaml")

    # ── 保存统计报告 ──────────────────────────────────────
    report = {
        "generated_at":     __import__("datetime").datetime.now().isoformat(),
        "source_tiles":     n_damage,
        "multiplier":       args.multiplier,
        "total_generated":  total_generated,
        "elapsed_sec":      round(elapsed),
        "aug_img_dir":      str(AUG_IMG_DIR),
        "aug_lbl_dir":      str(AUG_LBL_DIR),
        "augmented_yaml":   str(aug_yaml),
        "class_stats_aug":  per_class_after,
    }
    report_path = TILE_DATASET / "damage_augment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 打印汇总 ──────────────────────────────────────────
    print("═" * 56)
    print("  DAMAGE 过采样完成")
    print("─" * 56)
    print(f"  源 tile:    {n_damage} 张")
    print(f"  增强副本:   {total_generated} 张 (×{args.multiplier})")
    print(f"  输出目录:   {AUG_IMG_DIR}")
    print()
    print(f"  增强后类别分布 (仅过采样部分):")
    for name, cnt in per_class_after.items():
        marker = "⚠→" if name == "damage" else "  "
        print(f"  {marker} {name:<8}: {cnt:,}")
    print("─" * 56)
    print(f"  合并配置:   {aug_yaml}")
    print(f"  统计报告:   {report_path}")
    print("═" * 56)
    print()
    print("  下一步: 使用增强数据集训练")
    print(f"      python scripts/train_stage1_baseline.py \\")
    print(f"          --data {aug_yaml}")
    print()


if __name__ == "__main__":
    main()
