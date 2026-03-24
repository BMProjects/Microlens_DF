#!/usr/bin/env python3
"""
实验对比评估
============
在 CNAS 固定测试集上统一评估多个实验权重，生成对比报告。
所有实验使用相同参数 (conf=0.001, iou=0.6, 860 tiles) 确保可比性。

用法:
    # 自动发现所有实验权重并评估
    python scripts/compare_experiments.py

    # 指定权重列表
    python scripts/compare_experiments.py \\
        --weights output/training/stage2_cleaned/weights/best.pt \\
        --weights output/experiments/phase1_ssgd_pretrain/private_finetuned/weights/best.pt

    # 添加自定义标签
    python scripts/compare_experiments.py \\
        --weights output/training/stage2_cleaned/weights/best.pt:baseline \\
        --weights output/experiments/.../best.pt:ssgd_finetune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output/experiments/comparison"


# 已知实验权重及其标签
KNOWN_EXPERIMENTS = [
    ("output/training/stage2_cleaned/weights/best.pt",                                "A0_baseline"),
    ("output/experiments/phase1_ssgd_pretrain/private_finetuned/weights/best.pt",      "C1_ssgd_finetune"),
    ("output/experiments/phase1_ssgd_pretrain/private_finetuned_freeze10/weights/best.pt", "C1_ssgd_freeze10"),
    ("output/experiments/phase2_gsdnet/sfe_only/weights/best.pt",                      "B1_sfe_only"),
    ("output/experiments/phase2_gsdnet/nwd_only/weights/best.pt",                      "B2_nwd_only"),
    ("output/experiments/phase2_gsdnet/sfe_nwd/weights/best.pt",                       "B3_sfe_nwd"),
]


def discover_experiments() -> list[tuple[Path, str]]:
    """自动发现所有已完成训练的实验权重。"""
    found = []
    for rel_path, tag in KNOWN_EXPERIMENTS:
        weights_path = PROJECT_ROOT / rel_path
        if weights_path.exists():
            found.append((weights_path, tag))
    return found


def print_comparison_table(results: list[dict]) -> None:
    """打印对比表格。"""
    if not results:
        print("  无评估结果")
        return

    # 基线 (第一个结果)
    baseline = results[0]

    sep = "═" * 80
    thin = "─" * 80

    print(f"\n{sep}")
    print("  CNAS 测试集对比评估 (conf=0.001, iou=0.6, 860 tiles)")
    print(sep)

    # 表头
    header = f"  {'实验':<25} {'mAP@0.5':>8} {'scratch':>8} {'spot':>8} {'critical':>8} {'Δ mAP':>7}"
    print(header)
    print(thin)

    for r in results:
        ap = r.get("per_class_AP50", {})
        delta = r["mAP50"] - baseline["mAP50"]
        delta_str = f"{delta:+.4f}" if r != baseline else "  base"

        # 高亮 scratch AP 改善
        scratch_ap = ap.get("scratch", 0)
        scratch_marker = " ↑" if scratch_ap > baseline.get("per_class_AP50", {}).get("scratch", 0) and r != baseline else ""

        print(f"  {r['tag']:<25} {r['mAP50']:>8.4f} "
              f"{scratch_ap:>7.4f}{scratch_marker} "
              f"{ap.get('spot', 0):>7.4f} "
              f"{ap.get('critical', 0):>7.4f} "
              f"{delta_str:>7}")

    print(thin)

    # 判断
    best = max(results, key=lambda r: r["mAP50"])
    best_scratch = max(results, key=lambda r: r.get("per_class_AP50", {}).get("scratch", 0))

    print(f"\n  最高 mAP@0.5:      {best['tag']} ({best['mAP50']:.4f})")
    print(f"  最高 scratch AP:   {best_scratch['tag']} "
          f"({best_scratch.get('per_class_AP50', {}).get('scratch', 0):.4f})")

    if best["tag"] != baseline["tag"]:
        improvement = best["mAP50"] - baseline["mAP50"]
        print(f"  mAP 改善:          {improvement:+.4f} ({improvement/baseline['mAP50']*100:+.1f}%)")

    scratch_bl = baseline.get("per_class_AP50", {}).get("scratch", 0)
    scratch_best = best_scratch.get("per_class_AP50", {}).get("scratch", 0)
    if scratch_best > scratch_bl:
        print(f"  scratch AP 改善:   {scratch_best - scratch_bl:+.4f} "
              f"({(scratch_best - scratch_bl)/max(scratch_bl, 1e-6)*100:+.1f}%)")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="CNAS 测试集多实验对比评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", action="append", default=None,
                        help="权重路径[:标签] (可多次指定)")
    parser.add_argument("--save-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   多实验对比评估 (CNAS 固定测试集)                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 确定要评估的实验
    if args.weights:
        experiments = []
        for spec in args.weights:
            if ":" in spec and not spec[1] == ":":  # 避免 Windows 路径误判
                path_str, tag = spec.rsplit(":", 1)
            else:
                path_str = spec
                tag = Path(spec).parent.parent.name  # 用目录名作标签
            weights_path = Path(path_str)
            if not weights_path.is_absolute():
                weights_path = PROJECT_ROOT / weights_path
            if weights_path.exists():
                experiments.append((weights_path, tag))
            else:
                print(f"  ⚠ 权重不存在，跳过: {weights_path}")
    else:
        print("  自动发现已完成的实验 ...")
        experiments = discover_experiments()

    if not experiments:
        print("  ✗ 未找到任何可评估的实验权重")
        print("    请先完成训练，或用 --weights 手动指定")
        sys.exit(1)

    print(f"  找到 {len(experiments)} 个实验:")
    for path, tag in experiments:
        print(f"    {tag}: {path}")
    print()

    # 逐个评估
    from cnas_test.runner import run_cnas_eval

    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, (weights_path, tag) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] 评估: {tag}")
        print(f"  权重: {weights_path}")

        try:
            result = run_cnas_eval(
                weights_path=weights_path,
                save_dir=args.save_dir / tag,
                tag=tag,
                verbose=True,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ✗ 评估失败: {e}")
            continue

    # 打印对比表
    print_comparison_table(all_results)

    # 保存完整对比结果
    comparison_path = args.save_dir / "comparison_report.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiments": [r["tag"] for r in all_results],
            "results": all_results,
            "baseline": all_results[0]["tag"] if all_results else None,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  对比报告: {comparison_path}")


if __name__ == "__main__":
    main()
