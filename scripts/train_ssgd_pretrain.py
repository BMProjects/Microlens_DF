#!/usr/bin/env python3
"""
Phase 1.1: SSGD 公开数据集预训练 YOLOv12m
==========================================
在 SSGD 玻璃表面缺陷数据集上预训练 YOLOv12m，学习玻璃缺陷的通用特征表示，
为后续私有暗场镜片数据微调提供更好的初始化权重。

SSGD 数据集: 1246 张彩色图 (1500×1000), 7 类 → 映射为 3 类 (scratch/spot/critical)
已通过 prepare_ssgd_yolo.py 转换为 YOLO 格式。

用法:
    python scripts/train_ssgd_pretrain.py
    python scripts/train_ssgd_pretrain.py --epochs 80 --batch 16
    python scripts/train_ssgd_pretrain.py --no-countdown

预计训练时间 (RTX 4090D, batch=32): ~30-45 分钟 (100 epochs, ~1000 张图)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SSGD_YAML    = PROJECT_ROOT / "output/experiments/phase1_ssgd_pretrain/ssgd_yolo/ssgd_defects.yaml"
OUTPUT_DIR   = PROJECT_ROOT / "output/experiments/phase1_ssgd_pretrain"


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
        print("    pip install ultralytics>=8.3")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.1: SSGD 预训练 YOLOv12m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   default="yolo12m.pt", help="基础模型 (COCO 预训练)")
    parser.add_argument("--epochs",  type=int, default=100, help="训练轮数")
    parser.add_argument("--batch",   type=int, default=32,  help="batch size")
    parser.add_argument("--device",  default="0",           help="GPU id")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0",     type=float, default=0.01, help="初始学习率")
    parser.add_argument("--no-countdown", action="store_true")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 1.1: SSGD 公开数据预训练 YOLOv12m            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 检查数据集
    if not SSGD_YAML.exists():
        print(f"  ✗ SSGD YOLO 数据集不存在: {SSGD_YAML}")
        print("    请先运行: python scripts/prepare_ssgd_yolo.py")
        sys.exit(1)

    print(f"  数据集: {SSGD_YAML}")
    print(f"  基础模型: {args.model} (COCO 预训练)")
    print(f"  学习率: {args.lr0}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}")
    print()

    # 倒计时
    if not args.no_countdown:
        est_min = max(5, int(args.epochs * 0.35))
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
    print(f"  加载模型: {args.model} ...", end="", flush=True)
    try:
        model = YOLO(args.model)
        print(" ✓")
    except Exception as e:
        print(f" ✗ ({e})")
        # Fallback
        fallback = args.model.replace("12", "11")
        print(f"  尝试 fallback: {fallback} ...", end="", flush=True)
        model = YOLO(fallback)
        print(" ✓")

    print()

    # 训练
    t0 = time.time()
    try:
        results = model.train(
            data=str(SSGD_YAML),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            project=str(OUTPUT_DIR),
            name="ssgd_trained",
            exist_ok=True,

            # 硬件
            workers=args.workers,
            cache="ram",
            amp=True,
            optimizer="AdamW",

            # 学习率
            lr0=args.lr0,
            lrf=0.01,
            warmup_epochs=3,
            weight_decay=0.0005,

            # Loss
            cls=1.0,

            # 早停
            patience=20,
            save=True,
            save_period=10,
            close_mosaic=15,

            # 增强 — SSGD 是彩色图，保留色彩增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=90,       # 玻璃缺陷有方向性，但允许旋转
            fliplr=0.5,
            flipud=0.5,
            scale=0.3,
            translate=0.2,
            mosaic=1.0,
            copy_paste=0.1,
            erasing=0.0,

            verbose=True,
            plots=True,
        )
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  训练被中断，已运行 {fmt_time(elapsed)}")
        sys.exit(0)

    elapsed = time.time() - t0
    out_dir = OUTPUT_DIR / "ssgd_trained"
    best_weights = out_dir / "weights" / "best.pt"

    # 打印结果
    print()
    print("═" * 56)
    print("  Phase 1.1: SSGD 预训练完成")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳权重: {best_weights}")
    print("─" * 56)

    try:
        box = results.results_dict
        map50    = box.get("metrics/mAP50(B)",    box.get("metrics/mAP_0.5", 0))
        map50_95 = box.get("metrics/mAP50-95(B)", box.get("metrics/mAP_0.5:0.95", 0))
        prec     = box.get("metrics/precision(B)", box.get("metrics/precision", 0))
        recall   = box.get("metrics/recall(B)",    box.get("metrics/recall", 0))

        print(f"  SSGD val mAP@0.5:      {map50:.4f}")
        print(f"  SSGD val mAP@0.5:0.95: {map50_95:.4f}")
        print(f"  Precision:             {prec:.4f}")
        print(f"  Recall:                {recall:.4f}")

        summary = {
            "stage": "phase1_ssgd_pretrain",
            "completed_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed),
            "best_weights": str(best_weights),
            "dataset": str(SSGD_YAML),
            "config": {
                "model": args.model,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr0": args.lr0,
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
    print("  下一步: 用 SSGD 预训练权重微调私有数据")
    print(f"      python scripts/train_ssgd_finetune.py \\")
    print(f"          --weights {best_weights}")
    print()


if __name__ == "__main__":
    main()
