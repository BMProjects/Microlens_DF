#!/usr/bin/env python3
"""生成分割模型综合分析研究日志与图表."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "doc"
ASSET_DIR = DOC_DIR / "assets" / "generated"
ANALYSIS_BASE = ROOT / "output" / "experiments" / "phase3_segmentation" / "analysis"
BRIDGE_BASE = ROOT / "output" / "experiments" / "phase3_segmentation" / "bridge_review_20260325_12h"

PHASE3C = ANALYSIS_BASE / "segmentation_review_20260325"
PHASE3C_12H = ANALYSIS_BASE / "segmentation_review_20260325_12h"


def load_review_tables(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(base / "private_review_aggregate.csv")
    l3_path = base / "private_review_aggregate_l3_reference.csv"
    if l3_path.exists():
        l3 = pd.read_csv(l3_path)
    else:
        l3 = pd.DataFrame(columns=main.columns)
    return main, l3


def md_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("-")
            elif isinstance(value, float):
                values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_source_ranking(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    colors = ["#7c3aed", "#2563eb", "#059669", "#ea580c"]

    axes[0].bar(df["model"], df["best_val_miou"], color=colors)
    axes[0].set_title("Source-Domain Ranking (best val_mIoU)")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(df["model"], df["scratch_iou"], color=colors)
    axes[1].set_title("Source-Domain Scratch IoU")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_private_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, out_path: Path) -> None:
    merged = df_before.merge(df_after, on="model", suffixes=("_40ep", "_60ep"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    x = range(len(merged))

    axes[0].bar([i - 0.18 for i in x], merged["best_loss_40ep"], width=0.36, label="40-epoch stage", color="#93c5fd")
    axes[0].bar([i + 0.18 for i in x], merged["best_loss_60ep"], width=0.36, label="12h stage", color="#2563eb")
    axes[0].set_xticks(list(x), merged["model"])
    axes[0].set_title("Private Fine-tuning: best loss")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].bar([i - 0.18 for i in x], merged["final_loss_40ep"], width=0.36, label="40-epoch stage", color="#bbf7d0")
    axes[1].bar([i + 0.18 for i in x], merged["final_loss_60ep"], width=0.36, label="12h stage", color="#16a34a")
    axes[1].set_xticks(list(x), merged["model"])
    axes[1].set_title("Private Fine-tuning: final loss")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_review_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, out_path: Path) -> None:
    merged = df_before.merge(df_after, on="model", suffixes=("_40ep", "_12h"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)
    x = range(len(merged))

    metrics = [
        ("scratch_iou_mean", "Scratch IoU (higher is better)"),
        ("scratch_dice_mean", "Scratch Dice (higher is better)"),
        ("scratch_length_error_mean", "Scratch length error (lower is better)"),
        ("scratch_area_error_mean", "Scratch area error (lower is better)"),
    ]
    colors = ("#c4b5fd", "#7c3aed")

    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar([i - 0.18 for i in x], merged[f"{col}_40ep"], width=0.36, label="40-epoch stage", color=colors[0])
        ax.bar([i + 0.18 for i in x], merged[f"{col}_12h"], width=0.36, label="12h stage", color=colors[1])
        ax.set_xticks(list(x), merged["model"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_status(out_path: Path) -> tuple[int, int]:
    rows = []
    success_total = 0
    total_images = 0
    for model_name in ("unetplusplus_r34", "fpn_r34"):
        path = BRIDGE_BASE / model_name / "summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        success = len(data.get("images", []))
        total = int(data.get("n_images", 0))
        success_total += success
        total_images = max(total_images, total)
        rows.append((model_name, success, total - success))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = [r[0] for r in rows]
    success_vals = [r[1] for r in rows]
    fail_vals = [r[2] for r in rows]
    ax.bar(labels, success_vals, color="#16a34a", label="completed")
    ax.bar(labels, fail_vals, bottom=success_vals, color="#dc2626", label="unfinished / failed")
    ax.set_title("Bridge Review Status")
    ax.set_ylabel("Image count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return success_total, total_images


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(PHASE3C / "source_summary.csv")
    private_40_df = pd.read_csv(PHASE3C / "private_finetune_summary.csv")
    review_40_df, review_40_l3_df = load_review_tables(PHASE3C)

    private_12h_df = pd.read_csv(PHASE3C_12H / "private_finetune_summary.csv")
    review_12h_df, review_12h_l3_df = load_review_tables(PHASE3C_12H)

    source_chart = ASSET_DIR / "segmentation_source_ranking_20260326.png"
    private_chart = ASSET_DIR / "segmentation_private_finetune_compare_20260326.png"
    review_chart = ASSET_DIR / "segmentation_private_review_compare_20260326.png"
    bridge_chart = ASSET_DIR / "segmentation_bridge_status_20260326.png"

    plot_source_ranking(source_df, source_chart)
    plot_private_comparison(private_40_df, private_12h_df, private_chart)
    plot_review_comparison(review_40_df, review_12h_df, review_chart)
    bridge_success, bridge_total = plot_bridge_status(bridge_chart)

    conclusions = [
        "源域 clean-mask 排名仍然明确：FPN 与 DeepLabV3+ 在 `MSD` 上最强，说明多尺度语义融合对暗场缺陷分割非常有效。",
        "目标域弱标签微调中，Unet++ 的训练损失最低，说明其在不完美标签监督下仍然具备较强拟合能力。",
        "私有数据分割评估口径现已调整为 `L1+L2` 主指标、`L3` 复杂样本参考指标。这更符合“先在较简单与中等复杂图像上稳定标签迭代，再逐步攻克复杂图像”的研究安排。",
        "若把物理意义和标注更新作为优先目标，FPN 目前更稳：它在 `L1+L2` 主指标上取得最低的 `scratch_length_error_mean` 与最低的 `scratch_area_error_mean`，更适合作为长度、面积和连通性量化的基础模型。",
        "DeepLabV3+ 在长训后 loss 有所改善，但面积扩张问题仍然明显，当前更适合作为对照模型，而不是优先接入正式系统。",
        f"桥接评测当前仍未形成有效全图结论：最新桥接汇总里成功完成的图像数为 {bridge_success}/{bridge_total}，说明系统集成验证还处在修复与复测阶段，暂不宜据此宣布分割模型已可直接替换正式检测主线。",
    ]

    md = f"""# 研究实验日志 — Phase 3D：分割模型综合分析与当前研究结论

**日期**：2026-03-26  
**阶段目标**：对当前全部分割模型实验进行统一汇总，以图表、对比图和量化指标的方式明确当前研究结论，并给出正式系统接入建议。  
**数据来源**：

- `output/experiments/phase3_segmentation/analysis/segmentation_review_20260325/`
- `output/experiments/phase3_segmentation/analysis/segmentation_review_20260325_12h/`
- `output/experiments/phase3_segmentation/bridge_review_20260325_12h/`

---

## 一、实验范围

本轮综合分析覆盖以下分割实验：

1. `LightUNet / Unet++ / DeepLabV3+ / FPN` 在 `MSD` 源域上的结构对照。
2. `LightUNet / Unet++ / DeepLabV3+ / FPN` 在私有弱标签数据上的第一阶段微调。
3. `Unet++ / FPN / DeepLabV3+` 在 `12 小时窗口` 下的强化微调与复核。
4. `Unet++ / FPN` 在完整推理流水线中的桥接验证尝试。

---

## 二、源域结构对照结果

源域 `MSD` 结果表明，当前分割分支的“结构学习能力”排序已经很清楚，多尺度特征融合结构显著优于轻量基线。

{md_table(source_df[["model", "epochs_ran", "best_epoch", "best_val_miou", "scratch_iou", "spot_iou", "miou_quick"]])}

![Source ranking 对比图](assets/generated/segmentation_source_ranking_20260326.png)

![源域训练曲线](assets/generated/segmentation_source_miou_curves.png)

### 结论

- `FPN` 和 `DeepLabV3+` 在源域 clean-mask 条件下最强，说明暗场缺陷分割对多尺度上下文非常敏感。
- `Unet++` 虽然源域不是第一，但其 `scratch_iou=0.6632` 已明显高于 `LightUNet`，说明它对细长结构的恢复能力更强。

---

## 三、私有弱标签微调结果：40 epoch 阶段 vs 12 小时阶段

### 3.1 训练收敛结果

{md_table(private_40_df, float_digits=4)}

{md_table(private_12h_df, float_digits=4)}

![私有微调 loss 对比图](assets/generated/segmentation_private_finetune_compare_20260326.png)

![私有微调曲线](assets/generated/segmentation_private_loss_curves.png)

### 3.2 24 张私有图像量化复核结果

当前分割评估采用新的复杂度分层协议：

- `L1 + L2`：主指标，作为私有数据分割实验的正式对比口径
- `L3`：复杂样本参考指标，用于观察模型在高难场景下的退化方式，但暂不计入主排名

#### 第一阶段（40 epoch）

{md_table(review_40_df, float_digits=4)}

#### 第一阶段复杂样本参考（L3）

{md_table(review_40_l3_df, float_digits=4) if not review_40_l3_df.empty else '当前目录下尚无 L3 参考汇总。'}

#### 12 小时窗口复核

{md_table(review_12h_df, float_digits=4)}

#### 12 小时窗口复杂样本参考（L3）

{md_table(review_12h_l3_df, float_digits=4) if not review_12h_l3_df.empty else '当前目录下尚无 L3 参考汇总。'}

![私有复核指标对比图](assets/generated/segmentation_private_review_compare_20260326.png)

![多模型私有复核统计图](assets/generated/segmentation_private_review_metrics.png)

![24 张代表性样本并列核查图](assets/generated/segmentation_review_contact_sheet.png)

### 3.3 阶段分析

- `Unet++` 在 12 小时阶段取得最低 `best_loss=0.2756`，说明它仍是目标域弱标签拟合能力最强的模型。
- `FPN` 的 `best_loss` 并没有继续下降，但其几何量化稳定性最强，尤其是 `scratch_length_error_mean=0.2266`、`scratch_area_error_mean=0.5529`，明显优于 `Unet++` 和 `DeepLabV3+`。
- `DeepLabV3+` 在长训后 loss 有改善，但面积和长度误差仍然偏大，说明其在当前弱标签条件下仍存在区域外扩倾向。

---

## 四、桥接验证状态

桥接验证的目标，是把分割结果真正放回“完整推理流水线 (SAHI → WearMetrics → WearScore)”中，检验其对正式系统的可集成性。

![桥接验证状态图](assets/generated/segmentation_bridge_status_20260326.png)

当前桥接汇总显示：

- `Unet++ bridge review`：`0/{bridge_total}` 张形成有效汇总结果
- `FPN bridge review`：`0/{bridge_total}` 张形成有效汇总结果

这说明桥接链路虽然已经进入真实运行阶段，但目前仍处在接口兼容修复与复测过程中，尚不能把“模型本身表现”直接等同于“系统集成已完成”。因此，桥接评测当前只能作为进展状态，不宜作为最终模型选型的唯一依据。

---

## 五、当前明确的研究结论

"""
    for idx, item in enumerate(conclusions, start=1):
        md += f"{idx}. {item}\n"

    md += """

---

## 六、当前推荐策略

1. **若目标是继续提升目标域弱标签收敛能力**：优先保留 `Unet++` 作为研究主线。
2. **若目标是尽快形成可解释、可量化、可接入评分体系的分割结果**：优先保留 `FPN` 作为当前最稳的工程候选。
3. **若目标是正式系统短期集成**：建议采取“双模型策略”。

   - `Unet++`：继续做 scratch 敏感性研究和结构改进
   - `FPN`：优先进入长度/面积/骨架量化与评分验证

4. **DeepLabV3+** 暂时保留作对照组，不作为正式系统优先集成对象。
5. **桥接验证** 需要在脚本兼容修复后重新执行，待形成有效全图结果后，再判断是否可以让分割分支直接影响正式系统决策。

---

## 七、下一步工作建议

1. 继续完成 `Unet++ / FPN` 的桥接验证复测，确认全图级输出是否稳定。
2. 在 `FPN` 上优先补充 `scratch_length_mm / scratch_area_mm2 / scratch_components` 的稳定性分析。
3. 在 `Unet++` 上继续尝试更针对 scratch 的损失权重和弱标签精修策略。
4. 后续若桥接链路稳定，再补做一次 `mask->bbox` 对照评测，以明确分割模型对正式检测主线的替代潜力。
"""

    md_path = DOC_DIR / "研究实验日志_20260326_Phase3D_分割模型综合分析与当前研究结论.md"
    md_path.write_text(md, encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
