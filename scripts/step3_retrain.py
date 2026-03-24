#!/usr/bin/env python3
"""
Step 3: 基于清洗+伪标签的重新训练
==========================================
使用 Step1 清洗 + Step2 伪标签融合后的标注进行重训练。

改进:
  1. 从 Stage1 best.pt 继续微调（不从头训练）
  2. 更低学习率 (lr0=0.002)，更长 warmup
  3. 更强的 box loss 权重（精化定位）
  4. SAHI 推理配合（可选）

使用方式:
    python scripts/step3_retrain.py
    python scripts/step3_retrain.py --epochs 60 --batch 32
    python scripts/step3_retrain.py --from-scratch    # 不用预训练权重

预计时间 (RTX 4090D, batch=32): ~25-40 分钟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YAML = PROJECT_ROOT / "output" / "tile_dataset" / "defects.yaml"
STAGE1_BEST  = PROJECT_ROOT / "output" / "training" / "stage1" / "weights" / "best.pt"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "training"


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
        description="Step 3: 清洗标注重训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",  type=int, default=80,  help="训练 epoch 数")
    parser.add_argument("--batch",   type=int, default=32,  help="batch size")
    parser.add_argument("--device",  default="0",           help="GPU id")
    parser.add_argument("--name",    default="stage2_cleaned", help="实验名称")
    parser.add_argument("--from-scratch", action="store_true",
                        help="从预训练权重开始（不用 stage1 best）")
    parser.add_argument("--lr0",     type=float, default=0.002, help="初始学习率")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-countdown", action="store_true")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Step 3: 清洗标注重训练                              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Check dataset
    if not DATASET_YAML.exists():
        print(f"  ✗ 数据集不存在: {DATASET_YAML}")
        sys.exit(1)

    # Check cleanup report
    cleanup_report = PROJECT_ROOT / "output" / "tile_dataset" / "cleanup_report.json"
    if cleanup_report.exists():
        with open(cleanup_report) as f:
            cr = json.load(f)
        print(f"  标注清洗: {cr['original_total']:,} → {cr['kept_total']:,} "
              f"(移除 {cr['removed_pct']}%)")

    # Check pseudo label report
    pseudo_report = PROJECT_ROOT / "output" / "pseudo_labels" / "merge_report_train.json"
    if pseudo_report.exists():
        with open(pseudo_report) as f:
            pr = json.load(f)
        print(f"  伪标签融合: GT {pr['original_gt']:,} + 新增 {pr['stats'].get('pseudo_added',0):,} "
              f"→ {pr['merged_total']:,}")

    # Select base model
    if args.from_scratch or not STAGE1_BEST.exists():
        base_model = "yolo12m.pt"
        print(f"  基础模型: {base_model} (预训练)")
    else:
        base_model = str(STAGE1_BEST)
        print(f"  基础模型: stage1/best.pt (微调)")

    print(f"  学习率: {args.lr0} (Stage1: 0.01)")
    print(f"  Epochs: {args.epochs}")
    print()

    if not args.no_countdown:
        est_min = max(5, int(args.epochs * 0.4))  # rough estimate
        finish_time = datetime.now() + timedelta(minutes=est_min)
        print(f"  预计训练时长: ~{est_min} 分钟")
        print(f"  预计完成时间: {finish_time.strftime('%H:%M:%S')}")
        print()
        for i in range(5, 0, -1):
            print(f"\r  倒计时: {i:2d} 秒  ", end="", flush=True)
            time.sleep(1)
        print("\r  开始训练 ...         ")
        print()

    # Load and train
    YOLO = load_yolo()
    model = YOLO(base_model)

    t0 = time.time()
    try:
        results = model.train(
            data=str(DATASET_YAML),
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

            # 学习率 — 比 Stage1 更保守
            lr0=args.lr0,
            lrf=0.01,
            warmup_epochs=5,            # 更长 warmup
            weight_decay=0.0005,

            # Loss 权重 — 加强定位精度
            cls=1.0,
            box=10.0,                    # 提高 box loss 权重（默认 7.5）

            # 早停
            patience=25,
            save=True,
            save_period=10,
            close_mosaic=10,

            # 增强 — 同 Stage1
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.3,                   # 略减亮度抖动
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            scale=0.3,
            translate=0.2,
            mosaic=1.0,
            copy_paste=0.15,             # 略增 copy-paste
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

    print()
    print("═" * 56)
    print(f"  Step 3 训练完成")
    print(f"  训练时长: {fmt_time(elapsed)}")
    print(f"  最佳权重: {best_weights}")

    try:
        box = results.results_dict
        map50    = box.get("metrics/mAP50(B)",    box.get("metrics/mAP_0.5", 0))
        map50_95 = box.get("metrics/mAP50-95(B)", box.get("metrics/mAP_0.5:0.95", 0))
        prec     = box.get("metrics/precision(B)", box.get("metrics/precision", 0))
        recall   = box.get("metrics/recall(B)",    box.get("metrics/recall", 0))

        status = "✓ 达标" if map50 >= 0.60 else f"差 {0.60 - map50:.3f}"
        print(f"  mAP@0.5:        {map50:.3f}  ({status})")
        print(f"  mAP@0.5:0.95:   {map50_95:.3f}")
        print(f"  Precision:      {prec:.3f}")
        print(f"  Recall:         {recall:.3f}")

        summary = {
            "stage": "step3_retrain",
            "completed_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed),
            "best_weights": str(best_weights),
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
        print(f"  汇总: {summary_path}")

        if map50 >= 0.60:
            print()
            print("  ✓✓✓ 已达到 60% mAP 目标！")
            print("  下一步: 导出推理模型")
            print(f"      python -c \"from ultralytics import YOLO; YOLO('{best_weights}').export(format='onnx')\"")
        else:
            print()
            print("  下一步: 如果 mAP 仍不足，可以:")
            print("      1. 再次运行伪标签: python scripts/step2_pseudo_labels.py --weights", best_weights)
            print("      2. 增加推理策略: SAHI 滑窗推理")
            print("      3. 调整阈值: python scripts/step1_label_cleanup.py 修改面积阈值")
    except Exception:
        pass

    print("═" * 56)
    print()


if __name__ == "__main__":
    main()
