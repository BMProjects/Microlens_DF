#!/usr/bin/env python3
"""
Phase 1.2: 用 SSGD 预训练权重微调私有暗场数据
===============================================
从 Phase 1.1 的 SSGD 预训练权重出发，在私有暗场镜片切片数据集上微调。
目标: 验证 "任务相关公开数据预训练 → 私有微调" 能否改善 scratch AP。

对比基线: output/training/stage2_cleaned/weights/best.pt (mAP@0.5=0.6765)

关键差异（与 step3_retrain.py 相比）:
  - 基础权重: SSGD 预训练（而非 COCO 预训练后 stage1 微调）
  - 更低学习率: lr0=0.002（保留 SSGD 学到的玻璃缺陷特征）
  - 暗场增强: hsv_h=0, hsv_s=0（灰度图，禁用色彩增强）

变体 (可选):
  --freeze 10   只微调 detection head，冻结 backbone 前 10 层

用法:
    python scripts/train_ssgd_finetune.py
    python scripts/train_ssgd_finetune.py --freeze 10
    python scripts/train_ssgd_finetune.py --weights path/to/ssgd_best.pt

预计训练时间 (RTX 4090D, batch=32): ~35 分钟 (80 epochs, ~8500 tiles)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT   = Path(__file__).parent.parent
DATASET_YAML   = PROJECT_ROOT / "output/tile_dataset/defects.yaml"
DATASET_AUG    = PROJECT_ROOT / "output/tile_dataset/defects_augmented.yaml"
SSGD_WEIGHTS   = PROJECT_ROOT / "output/experiments/phase1_ssgd_pretrain/ssgd_trained/weights/best.pt"
OUTPUT_DIR     = PROJECT_ROOT / "output/experiments/phase1_ssgd_pretrain"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def load_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("  ✗ ultralytics 未安装")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.2: SSGD 预训练 → 私有数据微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, default=SSGD_WEIGHTS,
                        help="SSGD 预训练权重路径")
    parser.add_argument("--epochs",  type=int, default=80,  help="训练轮数")
    parser.add_argument("--batch",   type=int, default=32,  help="batch size")
    parser.add_argument("--device",  default="0",           help="GPU id")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0",     type=float, default=0.002, help="初始学习率 (低于 SSGD 训练)")
    parser.add_argument("--freeze",  type=int, default=0,
                        help="冻结前 N 层 (10=只微调 head)")
    parser.add_argument("--name",    default=None,
                        help="实验名称 (默认: private_finetuned 或 private_finetuned_freeze{N})")
    parser.add_argument("--no-countdown", action="store_true")
    args = parser.parse_args()

    # 自动命名
    if args.name is None:
        args.name = f"private_finetuned_freeze{args.freeze}" if args.freeze > 0 else "private_finetuned"

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 1.2: SSGD 预训练 → 私有数据微调               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 检查权重
    if not args.weights.exists():
        print(f"  ✗ SSGD 预训练权重不存在: {args.weights}")
        print("    请先运行: python scripts/train_ssgd_pretrain.py")
        sys.exit(1)

    # 检查数据集 — 优先用增强版
    if not DATASET_YAML.exists():
        print(f"  ✗ 私有数据集不存在: {DATASET_YAML}")
        sys.exit(1)
    actual_yaml = DATASET_AUG if DATASET_AUG.exists() else DATASET_YAML

    print(f"  SSGD 权重: {args.weights}")
    print(f"  私有数据集: {actual_yaml.name}")
    print(f"  学习率: {args.lr0} (SSGD 训练: 0.01)")
    print(f"  Freeze: {args.freeze} 层" if args.freeze > 0 else "  Freeze: 无 (全参微调)")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"  实验名: {args.name}")
    print()

    # 倒计时
    if not args.no_countdown:
        est_min = max(5, int(args.epochs * 0.45))
        finish_time = datetime.now() + timedelta(minutes=est_min)
        print(f"  预计训练时长: ~{est_min} 分钟")
        print(f"  预计完成时间: {finish_time.strftime('%H:%M:%S')}")
        print()
        for i in range(5, 0, -1):
            print(f"\r  倒计时: {i:2d} 秒  ", end="", flush=True)
            time.sleep(1)
        print("\r  开始训练 ...         ")
        print()

    # 加载模型
    YOLO = load_yolo()
    print(f"  加载 SSGD 预训练模型 ...", end="", flush=True)
    model = YOLO(str(args.weights))
    print(" ✓")
    print()

    # 训练
    t0 = time.time()
    try:
        results = model.train(
            data=str(actual_yaml),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            project=str(OUTPUT_DIR),
            name=args.name,
            exist_ok=True,

            # 硬件
            workers=args.workers,
            cache="ram",
            amp=True,
            optimizer="AdamW",

            # 学习率 — 保守微调，保留 SSGD 特征
            lr0=args.lr0,
            lrf=0.01,
            warmup_epochs=5,        # 更长 warmup 保护预训练权重
            weight_decay=0.0005,

            # Loss — 加强定位 (scratch bbox 精度关键)
            cls=1.0,
            box=10.0,               # 提高 box loss 权重

            # 冻结 backbone (可选)
            freeze=args.freeze if args.freeze > 0 else None,

            # 早停
            patience=25,
            save=True,
            save_period=10,
            close_mosaic=10,

            # 暗场专用增强 — 灰度图，禁用色彩
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.3,              # 亮度抖动 ±30%
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            scale=0.3,
            translate=0.2,
            mosaic=1.0,
            copy_paste=0.15,
            erasing=0.0,

            verbose=True,
            plots=True,
        )
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  训练被中断，已运行 {fmt_time(elapsed)}")
        sys.exit(0)

    elapsed = time.time() - t0
    out_dir = OUTPUT_DIR / args.name
    best_weights = out_dir / "weights" / "best.pt"

    # 打印结果
    print()
    print("═" * 56)
    print(f"  Phase 1.2: 私有微调完成 ({args.name})")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳权重: {best_weights}")
    print("─" * 56)

    try:
        box = results.results_dict
        map50    = box.get("metrics/mAP50(B)",    box.get("metrics/mAP_0.5", 0))
        map50_95 = box.get("metrics/mAP50-95(B)", box.get("metrics/mAP_0.5:0.95", 0))
        prec     = box.get("metrics/precision(B)", box.get("metrics/precision", 0))
        recall   = box.get("metrics/recall(B)",    box.get("metrics/recall", 0))

        print(f"  val mAP@0.5:      {map50:.4f}")
        print(f"  val mAP@0.5:0.95: {map50_95:.4f}")
        print(f"  Precision:        {prec:.4f}")
        print(f"  Recall:           {recall:.4f}")

        summary = {
            "stage": "phase1_ssgd_finetune",
            "experiment": args.name,
            "completed_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed),
            "best_weights": str(best_weights),
            "base_weights": str(args.weights),
            "dataset": str(actual_yaml),
            "config": {
                "epochs": args.epochs,
                "batch": args.batch,
                "lr0": args.lr0,
                "freeze": args.freeze,
            },
            "metrics": {
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
                "precision": round(prec, 4),
                "recall": round(recall, 4),
            },
        }
        summary_path = out_dir / "train_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  汇总: {summary_path}")
    except Exception:
        pass

    print("═" * 56)
    print()
    print("  下一步: 在 CNAS 测试集上评估")
    print(f"      python scripts/compare_experiments.py")
    print()
    print("  如果 mAP 回退但 scratch AP 改善，尝试冻结 backbone:")
    if args.freeze == 0:
        print(f"      python scripts/train_ssgd_finetune.py --freeze 10")
    print()


if __name__ == "__main__":
    main()
