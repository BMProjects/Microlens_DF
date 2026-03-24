#!/usr/bin/env python3
"""
Stage 1: YOLOv12m 基线训练脚本
=====================================
暗场显微镜镜片缺陷检测 — 第一阶段基线训练

使用方式:
    python scripts/train_stage1_baseline.py
    python scripts/train_stage1_baseline.py --epochs 50 --batch 8
    python scripts/train_stage1_baseline.py --model yolo11m.pt  # 若 yolo12m 不可用

预计训练时间 (RTX 4090D, batch=16):
    50 epochs  → ~45 分钟
    100 epochs → ~88 分钟

注意: 请先运行 damage 过采样脚本（可选但强烈建议）:
    python scripts/damage_augment.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET_YAML   = PROJECT_ROOT / "output" / "tile_dataset" / "defects.yaml"
DATASET_AUG    = PROJECT_ROOT / "output" / "tile_dataset" / "defects_augmented.yaml"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "training" / "stage1"
CLASS_NAMES    = ["scratch", "spot", "critical"]   # 3-class: damage+crash → critical


# ─── 辅助函数 ──────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Microlens_DF  Stage 1  YOLOv12m 基线训练           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def countdown_start(estimated_minutes: int, delay: int = 8):
    """倒计时提示，让用户有时间安排实验。"""
    start_time = datetime.now()
    finish_time = start_time + timedelta(minutes=estimated_minutes)

    print(f"  预计训练时长:  ~{estimated_minutes} 分钟")
    print(f"  训练开始时间: {start_time.strftime('%H:%M:%S')}")
    print(f"  预计完成时间: {finish_time.strftime('%H:%M:%S')}")
    print()
    print(f"  如需修改参数请在 {delay} 秒内按 Ctrl+C 中断 ...")
    print()

    for i in range(delay, 0, -1):
        print(f"\r  倒计时: {i:2d} 秒  ", end="", flush=True)
        time.sleep(1)
    print("\r  开始训练 ...         ")
    print()


def estimate_minutes(epochs: int, batch: int, n_tiles: int = 8471, cache_ram: bool = True) -> int:
    """粗略估算训练时间（基于 RTX 4090D 实测）。

    基准: batch=32, cache=ram → ~0.055s/iter (约 35 分钟/100 epochs)
    """
    # 基准: batch=32 时约 0.055s/iter (cache=ram, AMP, AdamW, 4090D)
    # batch=16 时 I/O 瓶颈更重，约 0.10s/iter
    if cache_ram:
        # RAM 缓存消除 I/O 瓶颈，吞吐量线性扩展
        sec_per_iter = 0.055 * (32 / max(batch, 1)) ** 0.8
    else:
        sec_per_iter = 0.10 * (16 / max(batch, 1)) ** 0.7
    iters_per_epoch = n_tiles / max(batch, 1)
    total_sec = epochs * iters_per_epoch * sec_per_iter
    return max(5, int(total_sec / 60))


def check_dataset(yaml_path: Path):
    """检查数据集是否存在，返回实际使用的 yaml 路径。"""
    if not yaml_path.exists():
        print(f"  ✗ 数据集配置文件不存在: {yaml_path}")
        print()
        print("  请先运行切图数据集构建脚本:")
        print("      bash scripts/run_build_dataset.sh")
        sys.exit(1)

    # 优先使用含 damage 过采样的版本
    if DATASET_AUG.exists():
        print(f"  ✓ 使用增强数据集 (含 damage 过采样): {DATASET_AUG.name}")
        return DATASET_AUG
    else:
        print(f"  ✓ 使用标准数据集: {yaml_path.name}")
        print("  → 提示: 运行 python scripts/damage_augment.py 可提升 damage 类召回率")
        return yaml_path


def print_dataset_stats():
    """打印数据集统计信息（3-class 方案）。"""
    report_path = PROJECT_ROOT / "output" / "tile_dataset" / "build_report.json"
    if not report_path.exists():
        return
    with open(report_path) as f:
        r = json.load(f)
    print(f"  数据集: {r['tiles_total']} tiles  "
          f"(train={r['split']['train']}, val={r['split']['val']})")
    dc = r.get("defect_counts", {})
    # 3-class 方案：critical = damage + crash
    n_scratch  = dc.get("scratch", 0)
    n_spot     = dc.get("spot", 0)
    n_critical = dc.get("crash", 0) + dc.get("damage", 0)
    print(f"  标注量 (3-class): scratch={n_scratch:,}  "
          f"spot={n_spot:,}  critical={n_critical:,}")
    print()


def load_yolo():
    """导入 ultralytics，失败时给出安装提示。"""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("  ✗ 未安装 ultralytics")
        print()
        print("  请安装:")
        print("      pip install ultralytics>=8.3")
        print("  或:")
        print("      uv add ultralytics")
        sys.exit(1)


def try_load_model(YOLO, model_name: str):
    """尝试加载模型，提供 fallback。"""
    candidates = [model_name]
    if "12" in model_name:
        # yolo12m → fallback to yolo11m
        fallback = model_name.replace("12", "11")
        candidates.append(fallback)
    candidates.append("yolov8m.pt")  # 最终 fallback

    for name in candidates:
        try:
            print(f"  加载模型: {name} ...", end="", flush=True)
            model = YOLO(name)
            print(" ✓")
            return model, name
        except Exception as e:
            print(f" ✗ ({e})")

    print("  ✗ 无法加载任何 YOLO 模型，请检查网络连接或手动下载权重文件")
    sys.exit(1)


def print_train_summary(results, elapsed: float, output_dir: Path):
    """训练完成后打印汇总。"""
    best_weights = output_dir / "weights" / "best.pt"

    print()
    print("═" * 56)
    print("  Stage 1 训练完成")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳权重: {best_weights}")
    print("─" * 56)

    # 尝试从 results 对象读取指标
    try:
        box = results.results_dict
        map50     = box.get("metrics/mAP50(B)",    box.get("metrics/mAP_0.5", 0))
        map50_95  = box.get("metrics/mAP50-95(B)", box.get("metrics/mAP_0.5:0.95", 0))
        prec      = box.get("metrics/precision(B)", box.get("metrics/precision", 0))
        recall    = box.get("metrics/recall(B)",    box.get("metrics/recall", 0))

        print(f"  mAP@0.5:        {map50:.3f}  "
              f"{'✓ 达标' if map50 >= 0.60 else f'✗ 距目标 {0.60 - map50:.3f}'}")
        print(f"  mAP@0.5:0.95:   {map50_95:.3f}")
        print(f"  Precision:      {prec:.3f}")
        print(f"  Recall:         {recall:.3f}")
        print("─" * 56)

        # 保存到 JSON
        summary = {
            "stage": "stage1_baseline",
            "completed_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed),
            "best_weights": str(best_weights),
            "metrics": {
                "map50":     round(map50, 4),
                "map50_95":  round(map50_95, 4),
                "precision": round(prec, 4),
                "recall":    round(recall, 4),
            },
        }
        summary_path = output_dir / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  汇总已保存: {summary_path}")
    except Exception:
        pass

    print("═" * 56)
    print()
    print("  下一步: 运行 Stage 2 标注审核")
    print(f"      python scripts/audit_stage2_labels.py \\")
    print(f"          --model {best_weights}")
    print()


# ─── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: YOLOv12m 基线训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   default="yolo12m.pt", help="YOLO 权重文件")
    parser.add_argument("--epochs",  type=int, default=100, help="训练 epoch 数")
    parser.add_argument("--batch",   type=int, default=32,  help="batch size (默认 32, 4090D 24GB 约用 10-12GB)")
    parser.add_argument("--device",  default="0",           help="GPU id (0/1/cpu)")
    parser.add_argument("--name",    default="stage1",      help="实验名称")
    parser.add_argument("--data",    default=None,          help="覆盖数据集 yaml 路径")
    parser.add_argument("--workers", type=int, default=8,   help="数据加载线程数 (默认 8)")
    parser.add_argument("--no-cache-ram", action="store_true",
                        help="禁用 RAM 缓存（内存不足时使用）")
    parser.add_argument("--resume",  action="store_true",   help="从上次检查点继续训练")
    parser.add_argument("--no-countdown", action="store_true", help="跳过倒计时")
    args = parser.parse_args()

    print_header()

    # ── 检查数据集 ──
    yaml_path = Path(args.data) if args.data else DATASET_YAML
    actual_yaml = check_dataset(yaml_path)
    print_dataset_stats()

    # ── 估算时间 ──
    use_cache_ram = not args.no_cache_ram
    est_min = estimate_minutes(args.epochs, args.batch, cache_ram=use_cache_ram)

    print(f"  配置预览:")
    print(f"    模型:      {args.model}")
    print(f"    轮次:      {args.epochs} epochs")
    print(f"    批大小:    {args.batch}  (4090D 24GB, ~{args.batch * 0.35:.0f}GB 显存)")
    print(f"    workers:   {args.workers}  (12核 CPU 并行加载)")
    print(f"    RAM缓存:   {'开启' if use_cache_ram else '关闭'}  (56GB可用，数据集~13GB)")
    print(f"    设备:      {args.device}")
    print(f"    输出:      output/training/{args.name}/")
    print()

    if not args.no_countdown:
        countdown_start(est_min)

    # ── 加载模型 ──
    YOLO = load_yolo()
    if args.resume:
        last_pt = OUTPUT_DIR / "weights" / "last.pt"
        if last_pt.exists():
            model, used_name = try_load_model(YOLO, str(last_pt))
        else:
            print(f"  ✗ 找不到检查点: {last_pt}")
            sys.exit(1)
    else:
        model, used_name = try_load_model(YOLO, args.model)

    print()

    # ── 训练 ──
    t0 = time.time()
    try:
        results = model.train(
            data=str(actual_yaml),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            project=str(PROJECT_ROOT / "output" / "training"),
            name=args.name,
            exist_ok=True,

            # ── 硬件加速 ─────────────────────────────────
            workers=args.workers,          # 并行数据加载 (12核 → 8 workers)
            cache='ram' if use_cache_ram else False,  # RAM 缓存消除 I/O 瓶颈
            amp=True,                      # 自动混合精度 (FP16)，节省显存约 40%
            optimizer='AdamW',             # AdamW 微调收敛更快（预训练权重）

            # ── 学习率 ────────────────────────────────────
            lr0=0.01,
            lrf=0.01,                      # 终止 LR = lr0 × lrf = 1e-4
            warmup_epochs=3,
            weight_decay=0.0005,

            # ── 类别不均衡处理 ────────────────────────────
            # ultralytics 8.4.x 已移除 fl_gamma，Focal Loss 内置于 Varifocal Loss
            # 提高 cls 权重强化分类损失信号（默认 0.5）
            cls=1.0,

            # ── 早停与检查点 ──────────────────────────────
            patience=20,
            save=True,
            save_period=10,
            close_mosaic=15,               # 最后 15 epoch 关闭 Mosaic 提升收敛稳定性

            # ── 暗场专用增强 ──────────────────────────────
            # 禁用色彩增强（单通道灰度图，色相/饱和度无意义）
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.4,                     # 亮度抖动 ±40%（模拟散射强度变化）

            # 几何增强（缺陷无方向偏好，允许全角度旋转）
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            scale=0.3,
            translate=0.2,
            mosaic=1.0,
            copy_paste=0.1,

            # 禁用随机擦除（会破坏稀少的 critical 类缺陷区域）
            erasing=0.0,

            # 显示设置
            verbose=True,
            plots=True,
        )
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  训练被中断，已运行 {fmt_time(elapsed)}")
        last_pt = PROJECT_ROOT / "output" / "training" / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"  中断检查点已保存: {last_pt}")
            print(f"  继续训练: python scripts/train_stage1_baseline.py --resume")
        sys.exit(0)

    elapsed = time.time() - t0
    out_dir = PROJECT_ROOT / "output" / "training" / args.name
    print_train_summary(results, elapsed, out_dir)


if __name__ == "__main__":
    main()
