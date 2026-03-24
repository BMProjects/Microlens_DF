#!/usr/bin/env python3
"""
Phase 2: GSDNet 模块消融实验
=============================
逐步集成 SFE (方向特征) 和 NWD (Wasserstein 框回归) 模块，
通过消融实验验证各模块对 scratch AP 的贡献。

消融矩阵:
  B1: YOLOv12m + SFE only    → 验证方向性特征提取
  B2: YOLOv12m + NWD only    → 验证框回归改善
  B3: YOLOv12m + SFE + NWD   → 组合效果

预期 (基于 GSDNet 论文消融):
  SFE: +3.5% mAP50 (最大增益)
  NWD (Hybrid Loss): +1.6%

用法:
    python scripts/train_phase2_sfe_nwd.py --exp B1
    python scripts/train_phase2_sfe_nwd.py --exp B2
    python scripts/train_phase2_sfe_nwd.py --exp B3
    python scripts/train_phase2_sfe_nwd.py --exp all  # 全部运行

预计训练时间 (RTX 4090D, batch=32): 每个实验 ~40 分钟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET_YAML   = PROJECT_ROOT / "output/tile_dataset/defects.yaml"
DATASET_AUG    = PROJECT_ROOT / "output/tile_dataset/defects_augmented.yaml"
SFE_MODEL_YAML = PROJECT_ROOT / "configs/yolo12m_sfe.yaml"
BASELINE_WEIGHTS = PROJECT_ROOT / "output/training/stage2_cleaned/weights/best.pt"
OUTPUT_DIR     = PROJECT_ROOT / "output/experiments/phase2_gsdnet"

# 实验配置
EXPERIMENTS = {
    "B1": {
        "name": "sfe_only",
        "desc": "YOLOv12m + SFE (方向特征)",
        "use_sfe": True,
        "use_nwd": False,
        "nwd_alpha": 0.0,
    },
    "B2": {
        "name": "nwd_only",
        "desc": "YOLOv12m + NWD (Wasserstein 框回归)",
        "use_sfe": False,
        "use_nwd": True,
        "nwd_alpha": 0.5,
        "lr0": 0.001,  # NWD 梯度组合使 0.002 不稳定，降低 LR 稳定训练
    },
    "B3": {
        "name": "sfe_nwd",
        "desc": "YOLOv12m + SFE + NWD (组合)",
        "use_sfe": True,
        "use_nwd": True,
        "nwd_alpha": 0.5,
        "lr0": 0.001,  # 同 B2 原因
    },
}


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


class _NWDBboxLoss:
    """占位符 — 实际类在 _make_nwd_bbox_loss_class() 中按需构建（模块级，可 pickle）。"""


def _make_nwd_bbox_loss_class(nwd_alpha: float):
    """返回可 pickle 的模块级 NWD BboxLoss 替代类。

    类必须在模块顶层（非函数内部）才能被 pickle 序列化（torch.save 要求）。
    通过将构建好的类赋给模块全局变量 `_NWDBboxLoss` 来满足此条件。
    """
    import sys
    from darkfield_defects.ml.nwd_loss import HybridBboxLoss
    import ultralytics.utils.loss as loss_module
    import torch.nn as nn

    _OrigBboxLoss = loss_module.BboxLoss

    # 构建模块级类（非 closure），nwd_alpha 通过类属性传入
    class NWDBboxLossImpl(_OrigBboxLoss):
        """注入 NWD 的 BboxLoss（模块级，可 pickle）。

        签名与 ultralytics 8.4.24 BboxLoss 完全一致:
          __init__(reg_max) — DFL 由内部 dfl_loss 模块控制
          forward 接受 9 个位置参数 (含 imgsz, stride)
        """
        _nwd_alpha: float = nwd_alpha  # 类属性，pickle 安全

        def __init__(self, reg_max):
            nn.Module.__init__(self)
            self._hybrid = HybridBboxLoss(reg_max, nwd_alpha=self._nwd_alpha)
            self.dfl_loss = self._hybrid.dfl_loss  # 镜像原版属性

        def forward(self, pred_dist, pred_bboxes, anchor_points,
                    target_bboxes, target_scores, target_scores_sum, fg_mask,
                    imgsz=None, stride_tensor=None):
            return self._hybrid(
                pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask,
                imgsz, stride_tensor,
            )

    # 提升为当前模块顶层名称，使 pickle 可通过 'train_phase2_sfe_nwd.NWDBboxLossImpl' 找到它
    current_module = sys.modules[__name__]
    setattr(current_module, "NWDBboxLossImpl", NWDBboxLossImpl)
    NWDBboxLossImpl.__module__ = __name__
    NWDBboxLossImpl.__qualname__ = "NWDBboxLossImpl"

    return NWDBboxLossImpl, _OrigBboxLoss


def patch_nwd_loss(nwd_alpha: float):
    """Monkey-patch ultralytics 的 BboxLoss，注入 NWD hybrid loss。"""
    import ultralytics.utils.loss as loss_module

    NWDBboxLossImpl, orig = _make_nwd_bbox_loss_class(nwd_alpha)
    loss_module.BboxLoss = NWDBboxLossImpl
    print(f"  ✓ NWD loss 已注入 (alpha={nwd_alpha})")
    return orig  # 返回原始类用于恢复


def restore_bbox_loss(original_class):
    """恢复原始 BboxLoss。"""
    import ultralytics.utils.loss as loss_module
    loss_module.BboxLoss = original_class


def run_experiment(
    exp_id: str,
    config: dict,
    args: argparse.Namespace,
    YOLO,
) -> dict | None:
    """运行单个消融实验。"""
    exp_name = config["name"]
    print()
    print("═" * 60)
    print(f"  实验 {exp_id}: {config['desc']}")
    print("═" * 60)

    # 确定数据集
    actual_yaml = DATASET_AUG if DATASET_AUG.exists() else DATASET_YAML
    if not actual_yaml.exists():
        print(f"  ✗ 数据集不存在: {actual_yaml}")
        return None

    # 注册 SFE 模块 (如需)
    if config["use_sfe"]:
        import register_custom_modules  # noqa: F401
        print("  ✓ SFE 模块已注册")

    # 注入 NWD loss (如需)
    orig_bbox_loss = None
    if config["use_nwd"]:
        orig_bbox_loss = patch_nwd_loss(config["nwd_alpha"])

    try:
        # 加载模型
        if config["use_sfe"]:
            # 从自定义 YAML 构建模型
            print(f"  加载自定义架构: {SFE_MODEL_YAML.name}")
            model = YOLO(str(SFE_MODEL_YAML))

            # 如果有基线权重，尝试部分加载 (迁移兼容层的权重)
            if BASELINE_WEIGHTS.exists() and args.transfer_weights:
                print(f"  部分迁移基线权重 ...")
                import torch
                ckpt = torch.load(str(BASELINE_WEIGHTS), map_location="cpu", weights_only=False)
                state = ckpt.get("model", ckpt).float().state_dict() if hasattr(ckpt.get("model", ckpt), "state_dict") else {}
                if state:
                    model_state = model.model.state_dict()
                    transferred = 0
                    for k, v in state.items():
                        if k in model_state and v.shape == model_state[k].shape:
                            model_state[k] = v
                            transferred += 1
                    model.model.load_state_dict(model_state, strict=False)
                    print(f"  迁移 {transferred}/{len(model_state)} 层权重")
        else:
            # 标准 YOLOv12m (只改 loss，不改架构)
            if BASELINE_WEIGHTS.exists():
                print(f"  加载基线权重: {BASELINE_WEIGHTS.name}")
                model = YOLO(str(BASELINE_WEIGHTS))
            else:
                print("  加载 COCO 预训练: yolo12m.pt")
                model = YOLO("yolo12m.pt")

        # 训练
        t0 = time.time()
        results = model.train(
            data=str(actual_yaml),
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            project=str(OUTPUT_DIR),
            name=exp_name,
            exist_ok=True,

            workers=args.workers,
            cache="ram",
            amp=True,
            optimizer="AdamW",

            lr0=config.get("lr0", args.lr0),  # 实验级 lr 优先于全局 lr0
            lrf=0.01,
            warmup_epochs=5,
            weight_decay=0.0005,

            cls=1.0,
            box=10.0,

            patience=25,
            save=True,
            save_period=10,
            close_mosaic=10,

            # 暗场增强
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
            erasing=0.0,

            verbose=True,
            plots=True,
        )
        elapsed = time.time() - t0

    finally:
        # 恢复原始 loss
        if orig_bbox_loss is not None:
            restore_bbox_loss(orig_bbox_loss)

    # 提取结果
    out_dir = OUTPUT_DIR / exp_name
    best_weights = out_dir / "weights" / "best.pt"

    try:
        box = results.results_dict
        map50    = box.get("metrics/mAP50(B)",    box.get("metrics/mAP_0.5", 0))
        map50_95 = box.get("metrics/mAP50-95(B)", box.get("metrics/mAP_0.5:0.95", 0))
        prec     = box.get("metrics/precision(B)", box.get("metrics/precision", 0))
        recall   = box.get("metrics/recall(B)",    box.get("metrics/recall", 0))
    except Exception:
        map50 = map50_95 = prec = recall = 0.0

    summary = {
        "experiment": exp_id,
        "name": exp_name,
        "description": config["desc"],
        "completed_at": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed),
        "best_weights": str(best_weights),
        "config": {
            "use_sfe": config["use_sfe"],
            "use_nwd": config["use_nwd"],
            "nwd_alpha": config["nwd_alpha"],
            "epochs": args.epochs,
            "batch": args.batch,
            "lr0": config.get("lr0", args.lr0),
        },
        "metrics": {
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "precision": round(prec, 4),
            "recall": round(recall, 4),
        },
    }

    summary_path = out_dir / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  {exp_id} 完成: mAP@0.5={map50:.4f}, 耗时 {fmt_time(elapsed)}")
    print(f"  权重: {best_weights}")
    print(f"  汇总: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: GSDNet 模块消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp", default="all",
                        help="实验 ID (B1/B2/B3/all)")
    parser.add_argument("--epochs",  type=int, default=80)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0",     type=float, default=0.002)
    parser.add_argument("--transfer-weights", action="store_true", default=True,
                        help="SFE 模型部分迁移基线权重")
    parser.add_argument("--no-countdown", action="store_true")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Phase 2: GSDNet 模块消融实验                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # 确定要运行的实验
    if args.exp.upper() == "ALL":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = [e.strip().upper() for e in args.exp.split(",")]

    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f"  ✗ 未知实验 ID: {eid}  (可选: {', '.join(EXPERIMENTS.keys())})")
            sys.exit(1)

    print(f"  计划运行 {len(exp_ids)} 个实验: {', '.join(exp_ids)}")
    print()

    if not args.no_countdown:
        est_total = len(exp_ids) * max(5, int(args.epochs * 0.5))
        print(f"  预计总时长: ~{est_total} 分钟")
        for i in range(5, 0, -1):
            print(f"\r  倒计时: {i:2d} 秒  ", end="", flush=True)
            time.sleep(1)
        print("\r  开始实验 ...         ")

    YOLO = load_yolo()
    all_results = []

    for eid in exp_ids:
        config = EXPERIMENTS[eid]
        result = run_experiment(eid, config, args, YOLO)
        if result:
            all_results.append(result)

    # 打印汇总
    if all_results:
        print("\n")
        print("═" * 60)
        print("  消融实验汇总")
        print("─" * 60)
        print(f"  {'实验':<15} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'耗时':>10}")
        print("─" * 60)
        for r in all_results:
            m = r["metrics"]
            print(f"  {r['experiment']:<15} {m['map50']:>10.4f} {m['map50_95']:>14.4f} "
                  f"{fmt_time(r['elapsed_sec']):>10}")
        print("═" * 60)

        # 保存汇总
        summary_path = OUTPUT_DIR / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  汇总: {summary_path}")

    print()
    print("  下一步: 用 CNAS 测试集评估")
    print("      python scripts/compare_experiments.py")
    print()


if __name__ == "__main__":
    main()
