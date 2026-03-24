#!/usr/bin/env python3
"""
标注迁移脚本：将 crash(3) 与 damage(2) 合并为 critical(2)
==============================================================
适用于已生成的 tile_dataset/labels/ 目录（10,621 个 txt 文件）。

变更规则:
    旧标注 (4类): scratch=0, spot=1, damage=2, crash=3
    新标注 (3类): scratch=0, spot=1, critical=2
    → 将所有 "3 " 开头的行替换为 "2 "
    → damage(2) 行保持不变，语义自然融入 critical

同时更新:
    - output/tile_dataset/defects.yaml
    - output/tile_dataset/defects_augmented.yaml（如存在）

使用方式:
    python scripts/relabel_merge_critical.py             # 先预览，再确认
    python scripts/relabel_merge_critical.py --yes       # 直接执行不询问
    python scripts/relabel_merge_critical.py --dry-run   # 只统计，不修改
    python scripts/relabel_merge_critical.py --backup    # 修改前备份标注目录
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ─── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TILE_DATASET = PROJECT_ROOT / "output" / "tile_dataset"
LABEL_DIRS = [
    TILE_DATASET / "labels" / "train",
    TILE_DATASET / "labels" / "val",
    TILE_DATASET / "labels" / "train_damage_aug",  # 若存在
]


def count_classes(label_dir: Path) -> dict[str, int]:
    """统计标注目录中各类别的实例数。"""
    counts: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0, "other": 0}
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = parts[0]
                counts[cls] = counts.get(cls, 0) + 1
    return counts


def migrate_label_dir(
    label_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """迁移单个标注目录中的所有 txt 文件。

    Returns:
        (files_changed, lines_changed)
    """
    if not label_dir.exists():
        return 0, 0

    files_changed = 0
    lines_changed = 0

    for txt_path in label_dir.glob("*.txt"):
        original = txt_path.read_text()
        lines = original.splitlines()
        new_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == "3":
                # crash(3) → critical(2)
                new_line = "2 " + " ".join(parts[1:])
                new_lines.append(new_line)
                lines_changed += 1
                file_changed = True
            else:
                new_lines.append(line)

        if file_changed:
            files_changed += 1
            if not dry_run:
                txt_path.write_text("\n".join(new_lines) + "\n")

    return files_changed, lines_changed


def update_yaml(yaml_path: Path, dry_run: bool = False):
    """更新 YOLO 数据集 YAML 文件，将 4 类改为 3 类。"""
    if not yaml_path.exists():
        return

    content = yaml_path.read_text()
    new_content = content

    # 替换类别数量
    new_content = new_content.replace("nc: 4", "nc: 3")

    # 替换类别名称（4类 → 3类）
    for old_names in [
        "['scratch', 'spot', 'damage', 'crash']",
        '["scratch", "spot", "damage", "crash"]',
        "names: ['scratch', 'spot', 'damage', 'crash']",
        'names: ["scratch", "spot", "damage", "crash"]',
    ]:
        new_content = new_content.replace(old_names, "['scratch', 'spot', 'critical']")

    # 替换注释说明
    old_comment = "# 2: damage   — 缺损（大面积不规则高亮）\n# 3: crash    — 密集区（高密度缺陷聚集）"
    new_comment = "# 2: critical — 严重缺陷区（大面积缺损 + 密集缺陷聚集 合并）"
    new_content = new_content.replace(old_comment, new_comment)

    if new_content != content:
        print(f"  {'[dry-run] ' if dry_run else ''}更新 YAML: {yaml_path.name}")
        if not dry_run:
            yaml_path.write_text(new_content)


def main():
    parser = argparse.ArgumentParser(
        description="将 crash(3) 合并为 critical(2)，迁移已有 YOLO 标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes",      action="store_true", help="跳过确认直接执行")
    parser.add_argument("--dry-run",  action="store_true", help="只统计，不修改文件")
    parser.add_argument("--backup",   action="store_true", help="修改前备份 labels/ 目录")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   标注迁移: damage+crash → critical (3-class)         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── 统计当前状态 ──────────────────────────────────────
    existing_dirs = [d for d in LABEL_DIRS if d.exists()]
    if not existing_dirs:
        print(f"  ✗ 未找到标注目录，请确认 tile_dataset 已构建")
        sys.exit(1)

    print("  当前标注统计:")
    total_crash = 0
    total_damage = 0
    total_scratch = 0
    total_spot = 0
    for ldir in existing_dirs:
        counts = count_classes(ldir)
        n_files = len(list(ldir.glob("*.txt")))
        print(f"  {ldir.parent.name}/{ldir.name}/: {n_files} 文件  "
              f"scratch={counts.get('0',0):,} "
              f"spot={counts.get('1',0):,} "
              f"damage={counts.get('2',0):,} "
              f"crash={counts.get('3',0):,}")
        total_scratch += counts.get("0", 0)
        total_spot    += counts.get("1", 0)
        total_damage  += counts.get("2", 0)
        total_crash   += counts.get("3", 0)

    print()
    print(f"  变更预览:")
    print(f"    scratch (0):  {total_scratch:,}  → 不变")
    print(f"    spot    (1):  {total_spot:,}  → 不变")
    print(f"    damage  (2):  {total_damage:,}  ┐ 合并为")
    print(f"    crash   (3):  {total_crash:,}  ┘ critical(2) = {total_damage + total_crash:,}")
    print()

    if total_crash == 0:
        print("  ✓ 没有 crash(3) 标注需要迁移（可能已经迁移过了）")
        update_yaml(TILE_DATASET / "defects.yaml", dry_run=args.dry_run)
        update_yaml(TILE_DATASET / "defects_augmented.yaml", dry_run=args.dry_run)
        return

    if args.dry_run:
        print("  (dry-run 模式，不修改任何文件)")
        return

    # ── 确认 ──────────────────────────────────────────────
    if not args.yes:
        print("  此操作将直接修改标注文件（不可逆，除非使用 --backup）。")
        ans = input("  确认执行迁移？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("  已取消。")
            return
        print()

    # ── 备份 ──────────────────────────────────────────────
    if args.backup:
        backup_dir = TILE_DATASET / "labels_backup_4class"
        if backup_dir.exists():
            print(f"  ⚠ 备份目录已存在: {backup_dir}，跳过备份")
        else:
            shutil.copytree(TILE_DATASET / "labels", backup_dir)
            print(f"  ✓ 已备份标注目录: {backup_dir}")
        print()

    # ── 执行迁移 ──────────────────────────────────────────
    total_files = 0
    total_lines = 0
    for ldir in existing_dirs:
        fc, lc = migrate_label_dir(ldir, dry_run=False)
        total_files += fc
        total_lines += lc
        if fc > 0:
            print(f"  ✓ {ldir.parent.name}/{ldir.name}/: {fc} 个文件，{lc:,} 行已迁移")

    # ── 更新 YAML ─────────────────────────────────────────
    update_yaml(TILE_DATASET / "defects.yaml")
    update_yaml(TILE_DATASET / "defects_augmented.yaml")

    # ── 汇总 ──────────────────────────────────────────────
    print()
    print("═" * 52)
    print("  迁移完成")
    print(f"  修改文件: {total_files} 个")
    print(f"  修改标注: {total_lines:,} 行 (crash:3 → critical:2)")
    print()
    print("  新类别映射:")
    print("    0: scratch   — 划痕")
    print("    1: spot      — 斑点")
    print("    2: critical  — 严重缺陷区 (原 damage + crash)")
    print("═" * 52)
    print()
    print("  下一步: 开始训练 (3-class 方案)")
    print("      python scripts/damage_augment.py   # 可选，现在 critical 已够多")
    print("      python scripts/train_stage1_baseline.py")
    print()


if __name__ == "__main__":
    main()
