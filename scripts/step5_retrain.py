#!/usr/bin/env python3
"""
Step 5: Stage4 重训练
==========================================
在 Step4 半监督重标注（scratch 碎片连接 + 漏标补全 + 不完整扩展）
改进后的 GT 数据上进行重训练。

训练策略:
  - 从 Stage3 best.pt 继续微调（保留已学到的特征）
  - 学习率进一步降低（lr0=0.001），专注定位质量
  - 更强的 box loss 权重（12.0），因为 GT 几何质量提升了
  - 关闭 mosaic 更早（close_mosaic=5），让模型适应更真实的划痕形状

使用方式:
    python scripts/step5_retrain.py
    python scripts/step5_retrain.py --epochs 60
    python scripts/step5_retrain.py --batch 32 --device 0

预计时间 (RTX 4090D, batch=32): ~30-45 分钟
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
STAGE3_BEST  = (PROJECT_ROOT / "output" / "training" /
                "stage2_cleaned" / "weights" / "best.pt")
OUTPUT_DIR   = PROJECT_ROOT / "output" / "training"
AUDIT_DIR    = PROJECT_ROOT / "output" / "audit"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Stage4 重训练（半监督标注优化后）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",  type=int, default=60,
                        help="训练 epoch 数（默认 60，比 Stage3 少因为已接近收敛）")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--name",    default="stage4_relabel",
                        help="实验名（默认 stage4_relabel）")
    parser.add_argument("--lr0",     type=float, default=0.001,
                        help="初始学习率（默认 0.001，Stage3 的一半）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-countdown", action="store_true")
    parser.add_argument("--weights", type=str, default=None,
                        help="基础权重路径（默认 Stage3 best.pt）")
    args = parser.parse_args()

    base_weights = Path(args.weights) if args.weights else STAGE3_BEST

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Step 5: Stage4 重训练（半监督标注优化）            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    if not DATASET_YAML.exists():
        print(f"  ✗ 数据集 yaml 不存在: {DATASET_YAML}")
        sys.exit(1)

    if not base_weights.exists():
        print(f"  ✗ 基础权重不存在: {base_weights}")
        print("    请先完成 step3_retrain.py")
        sys.exit(1)

    # 显示 step4 改进统计
    step4_audit = AUDIT_DIR / "step4_audit.json"
    if step4_audit.exists():
        d = json.loads(step4_audit.read_text())
        st = d.get("stats", {})
        print(f"  Step4 改进摘要:")
        print(f"    GT 碎片连接:  {st.get('gt_chains', 0):,} 条链")
        print(f"    框扩展:       {st.get('extended', 0):,}")
        print(f"    新增框:       {st.get('added', 0):,}")
        print(f"    切片修改:     {st.get('tiles_modified', 0):,}")
        print()

    print(f"  基础模型: {base_weights.parent.parent.name}/best.pt")
    print(f"  学习率:   {args.lr0}  (Stage3: 0.002)")
    print(f"  Epochs:   {args.epochs}")
    print(f"  输出目录: {OUTPUT_DIR}/{args.name}/")
    print()

    if not args.no_countdown:
        est_min = max(5, int(args.epochs * 0.5))
        finish_time = datetime.now() + timedelta(minutes=est_min)
        print(f"  预计时长: ~{est_min} 分钟，完成约 {finish_time.strftime('%H:%M')}")
        print()
        for i in range(5, 0, -1):
            print(f"\r  {i} 秒后开始 ...", end="", flush=True)
            time.sleep(1)
        print("\r  开始训练 ...              ")
        print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ✗ ultralytics 未安装: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(base_weights))
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

            # 学习率 — 更保守（模型已趋于收敛，专注定位质量）
            lr0=args.lr0,
            lrf=0.01,
            warmup_epochs=3,
            weight_decay=0.0005,

            # Loss 权重 — 增强 box loss（GT 几何质量提升后有意义）
            cls=1.0,
            box=12.0,          # 比 Stage3(10.0) 更高
            dfl=1.5,

            # 更早关闭 mosaic，让模型适应连续划痕形态
            close_mosaic=5,

            patience=20,
            save=True,
            save_period=10,

            # 增强（暗场图像专用）
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.3,
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            scale=0.3,
            translate=0.2,
            mosaic=1.0,
            copy_paste=0.15,
            erasing=0.3,
        )
    except KeyboardInterrupt:
        print("\n  ⚠ 训练被中断")
        return
    except Exception as e:
        print(f"\n  ✗ 训练出错: {e}")
        raise

    elapsed = time.time() - t0
    print()
    print(f"  ✓ 训练完成，用时 {fmt_time(elapsed)}")

    # 保存摘要
    out_dir = OUTPUT_DIR / args.name
    try:
        metrics = {
            "map50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
        }
    except Exception:
        metrics = {}

    summary = {
        "stage": "step5_stage4_retrain",
        "base_weights": str(base_weights),
        "epochs": args.epochs,
        "lr0": args.lr0,
        "batch": args.batch,
        "elapsed_sec": round(elapsed, 1),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "final_metrics": metrics,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print()
    print(f"  最终指标:")
    print(f"    mAP@0.5:      {metrics.get('map50', 0):.4f}")
    print(f"    mAP@0.5:0.95: {metrics.get('map50_95', 0):.4f}")
    print(f"    Precision:    {metrics.get('precision', 0):.4f}")
    print(f"    Recall:       {metrics.get('recall', 0):.4f}")
    print()
    print(f"  权重保存: {out_dir}/weights/best.pt")
    print()
    print("  → 下一步: python scripts/infer_sahi.py --split val "
          f"--weights {out_dir}/weights/best.pt")
    print()


if __name__ == "__main__":
    main()
