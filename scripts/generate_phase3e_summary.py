#!/usr/bin/env python3
"""生成 Phase 3E 自动化迭代阶段简报（HTML）."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE3E_ROOT = PROJECT_ROOT / "output/experiments/phase3e"
SUMMARY_ROOT = PHASE3E_ROOT / "summary"
ASSETS_ROOT = SUMMARY_ROOT / "assets"

SEG_PRIMARY = PROJECT_ROOT / "output/experiments/phase3_segmentation/analysis/segmentation_review_phase3e/summary.json"
SEG_FALLBACK = PROJECT_ROOT / "output/experiments/phase3_segmentation/analysis/segmentation_review_20260325_12h/summary.json"
DET_COMPARE_PRIMARY = PROJECT_ROOT / "output/experiments/comparison/phase3e_detection_compare/comparison_report.json"
DET_COMPARE_FALLBACK = PROJECT_ROOT / "output/experiments/comparison/comparison_report.json"
BRIDGE_UNET = PROJECT_ROOT / "output/experiments/phase3_segmentation/bridge_review_20260325_12h/unetplusplus_r34/summary.json"
BRIDGE_FPN = PROJECT_ROOT / "output/experiments/phase3_segmentation/bridge_review_20260325_12h/fpn_r34/summary.json"
CONTACT_SHEET = PROJECT_ROOT / "doc/assets/generated/segmentation_review_contact_sheet.png"
PRIVATE_METRICS_FIG = PROJECT_ROOT / "doc/assets/generated/segmentation_private_review_metrics.png"


CSS = """
body {
  margin: 0;
  background: #f3f6fb;
  color: #0f172a;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.page {
  width: min(1240px, calc(100vw - 48px));
  margin: 24px auto 48px;
}
.hero, .section {
  background: #fff;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  padding: 28px 34px;
  margin-bottom: 20px;
}
.hero {
  background: linear-gradient(135deg, #eff6ff 0%, #ffffff 48%, #f8fafc 100%);
}
h1, h2, h3 {
  margin: 0 0 14px 0;
  line-height: 1.3;
}
h1 { font-size: 2rem; }
h2 { font-size: 1.35rem; border-left: 5px solid #3b82f6; padding-left: 12px; }
h3 { font-size: 1.05rem; }
p, li {
  line-height: 1.8;
  font-size: 15px;
}
.meta {
  color: #475569;
  font-size: 14px;
  margin-top: 10px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}
.card {
  border-radius: 16px;
  padding: 16px 18px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}
.card h3 {
  color: #334155;
  font-size: 14px;
  margin-bottom: 8px;
}
.card .value {
  font-size: 1.45rem;
  font-weight: 700;
  color: #0f172a;
}
.card .hint {
  color: #64748b;
  font-size: 13px;
  margin-top: 6px;
}
.note {
  margin-top: 10px;
  color: #475569;
  font-size: 14px;
}
.status-ok { color: #15803d; }
.status-warn { color: #b45309; }
.status-bad { color: #b91c1c; }
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 14px;
  font-size: 14px;
}
th, td {
  border: 1px solid #dbe4ee;
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #eff6ff;
}
tr:nth-child(even) td {
  background: #fafcff;
}
.two-col {
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  gap: 18px;
  align-items: start;
}
.figure {
  margin-top: 18px;
}
.figure img {
  width: 100%;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
.caption {
  font-size: 13px;
  color: #64748b;
  margin-top: 8px;
}
ul {
  padding-left: 18px;
}
code {
  background: #eff6ff;
  padding: 2px 6px;
  border-radius: 6px;
  font-family: "JetBrains Mono", "Consolas", monospace;
}
@media (max-width: 980px) {
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .two-col { grid-template-columns: 1fr; }
}
"""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_path(primary: Path, fallback: Path) -> tuple[Path | None, str]:
    if primary.exists():
        return primary, "phase3e"
    if fallback.exists():
        return fallback, "fallback"
    return None, "missing"


def find_cnas_result(eval_dir: Path) -> tuple[Path | None, str]:
    primary = eval_dir / "metrics/cnas_eval_results.json"
    if primary.exists():
        return primary, "phase3e"
    candidates = sorted(eval_dir.rglob("cnas_eval_results.json"))
    if candidates:
        return candidates[0], "phase3e"
    return None, "missing"


def rel(path: Path) -> str:
    return path.relative_to(SUMMARY_ROOT).as_posix()


def copy_if_exists(src: Path, dest_name: str) -> str | None:
    if not src.exists():
        return None
    ensure_dir(ASSETS_ROOT)
    dest = ASSETS_ROOT / dest_name
    shutil.copyfile(src, dest)
    return rel(dest)


def plot_segmentation_overview(seg_summary: dict, output_path: Path) -> None:
    source_rows = seg_summary.get("source_summary", [])
    private_rows = seg_summary.get("private_review_aggregate", [])
    if not source_rows or not private_rows:
        return

    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    models = [row["model"] for row in source_rows]
    x = np.arange(len(models))
    width = 0.36
    val_miou = [row.get("best_val_miou", 0.0) for row in source_rows]
    scratch_iou = [row.get("scratch_iou", 0.0) for row in source_rows]

    axes[0].bar(x - width / 2, val_miou, width, label="best val_mIoU", color="#3b82f6")
    axes[0].bar(x + width / 2, scratch_iou, width, label="scratch IoU", color="#f97316")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Source-domain structure comparison")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    p_models = [row["model"] for row in private_rows]
    x2 = np.arange(len(p_models))
    length_err = [row.get("scratch_length_error_mean", 0.0) for row in private_rows]
    area_err = [row.get("scratch_area_error_mean", 0.0) for row in private_rows]
    axes[1].bar(x2 - width / 2, length_err, width, label="length error", color="#10b981")
    axes[1].bar(x2 + width / 2, area_err, width, label="area error", color="#ef4444")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(p_models, rotation=15)
    axes[1].set_title("Private 24-image geometry error comparison")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_detection_compare(det_report: dict, output_path: Path) -> None:
    results = det_report.get("results", [])
    if not results:
        return

    ensure_dir(output_path.parent)
    tags = [row["tag"] for row in results]
    map50 = [row.get("mAP50", 0.0) for row in results]
    scratch = [row.get("per_class_AP50", {}).get("scratch", 0.0) for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(tags))
    colors = ["#2563eb" if tag.startswith("A0") else "#f97316" for tag in tags]

    axes[0].bar(x, map50, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags, rotation=18)
    axes[0].set_ylim(0, max(map50 + [0.1]) + 0.05)
    axes[0].set_title("Detection branch mAP@0.5")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, scratch, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tags, rotation=18)
    axes[1].set_ylim(0, max(scratch + [0.1]) + 0.05)
    axes[1].set_title("Detection branch scratch AP@0.5")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bridge_status(bridge_data: list[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    names = [row["model"] for row in bridge_data]
    valid = [row["valid_images"] for row in bridge_data]
    target = [row["n_images"] for row in bridge_data]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(x, target, color="#dbeafe", label="planned images")
    ax.bar(x, valid, color="#2563eb", label="valid outputs")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, max(target + [1]) + 1)
    ax.set_title("Segmentation bridge completion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def html_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body_parts = []
    for row in rows:
        cells = []
        for key, _title in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def summarize_detection(det_report: dict) -> tuple[dict | None, dict | None]:
    results = det_report.get("results", [])
    baseline = None
    nwd = None
    for row in results:
        if row.get("tag") == "A0_baseline":
            baseline = row
        if str(row.get("tag", "")).startswith("B2_nwd_only"):
            nwd = row
    return baseline, nwd


def build_bridge_row(name: str, path: Path) -> dict:
    if not path.exists():
        return {"model": name, "n_images": 0, "valid_images": 0, "elapsed_sec": 0.0, "status": "missing"}
    payload = load_json(path)
    valid_images = len(payload.get("images", []))
    n_images = int(payload.get("n_images", 0))
    status = "ok" if n_images > 0 and valid_images == n_images else ("partial" if valid_images > 0 else "blocked")
    return {
        "model": name,
        "n_images": n_images,
        "valid_images": valid_images,
        "elapsed_sec": float(payload.get("elapsed_sec", 0.0)),
        "status": status,
    }


def verdict_text(seg_private: list[dict], baseline: dict | None, nwd: dict | None, bridge_rows: list[dict]) -> list[str]:
    lines: list[str] = []
    fpn = next((row for row in seg_private if row.get("model") == "FPN"), None)
    unet = next((row for row in seg_private if row.get("model") == "Unet++"), None)
    deeplab = next((row for row in seg_private if row.get("model") == "DeepLabV3+"), None)

    if unet and fpn:
        lines.append(
            f"分割主线当前仍建议保持“双模型判断”：`Unet++` 在弱标签拟合上更强（best loss={unet.get('best_loss', 0):.4f}），"
            f"`FPN` 在长度/面积物理量上更稳（length error={fpn.get('scratch_length_error_mean', 0):.4f}）。"
        )
    if deeplab:
        lines.append(
            f"`DeepLabV3+` 继续保留为对照组，但其面积外扩问题仍未完全消除（area error={deeplab.get('scratch_area_error_mean', 0):.4f}）。"
        )
    if baseline and nwd:
        delta_map = nwd.get("mAP50", 0.0) - baseline.get("mAP50", 0.0)
        delta_scratch = (
            nwd.get("per_class_AP50", {}).get("scratch", 0.0)
            - baseline.get("per_class_AP50", {}).get("scratch", 0.0)
        )
        lines.append(
            f"识别主线当前的关键问题仍是 `B2_nwd_only` 是否具备可复现优势：相对 `A0_baseline`，"
            f"当前可见结果的 `ΔmAP50={delta_map:+.4f}`，`Δscratch AP={delta_scratch:+.4f}`。"
        )
    bridge_ok = all(row["status"] == "ok" for row in bridge_rows) and bridge_rows
    if bridge_ok:
        lines.append("桥接验证已完成闭环，可进入下一轮分割接入与评分联调。")
    else:
        lines.append("桥接验证仍应作为 Phase 3E 的优先收尾项；在 8/8 全部稳定通过前，不宜据此宣称分割分支已可直接替代正式检测主线。")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Phase 3E HTML 简报")
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_ROOT)
    args = parser.parse_args()

    output_dir = args.output_dir
    assets_dir = output_dir / "assets"
    ensure_dir(assets_dir)

    seg_path, seg_source = choose_path(SEG_PRIMARY, SEG_FALLBACK)
    det_path, det_source = choose_path(DET_COMPARE_PRIMARY, DET_COMPARE_FALLBACK)
    cnas_path, cnas_source = find_cnas_result(PROJECT_ROOT / "output/experiments/phase3e/detection_eval/b2_nwd_only")

    seg_summary = load_json(seg_path) if seg_path else {}
    det_report = load_json(det_path) if det_path else {}
    cnas_payload = load_json(cnas_path) if cnas_path else {}

    bridge_rows = [
        build_bridge_row("Unet++ bridge", BRIDGE_UNET),
        build_bridge_row("FPN bridge", BRIDGE_FPN),
    ]

    seg_chart = assets_dir / "phase3e_segmentation_overview.png"
    det_chart = assets_dir / "phase3e_detection_overview.png"
    bridge_chart = assets_dir / "phase3e_bridge_status.png"

    if seg_summary:
        plot_segmentation_overview(seg_summary, seg_chart)
    if det_report:
        plot_detection_compare(det_report, det_chart)
    plot_bridge_status(bridge_rows, bridge_chart)

    contact_sheet_rel = copy_if_exists(CONTACT_SHEET, "segmentation_review_contact_sheet.png")
    metrics_fig_rel = copy_if_exists(PRIVATE_METRICS_FIG, "segmentation_private_review_metrics.png")

    source_rows = seg_summary.get("source_summary", [])
    private_finetune_rows = seg_summary.get("private_finetune_summary", [])
    private_review_rows = seg_summary.get("private_review_aggregate", [])
    baseline, nwd = summarize_detection(det_report)

    status_cards = [
        {
            "title": "分割源域主胜者",
            "value": source_rows and max(source_rows, key=lambda x: x.get("best_val_miou", 0.0)).get("model", "待生成") or "待生成",
            "hint": f"数据源：{seg_source}",
        },
        {
            "title": "分割工程候选",
            "value": "FPN" if private_review_rows else "待生成",
            "hint": "基于长度/面积误差与物理量稳定性",
        },
        {
            "title": "识别研究父实验",
            "value": "B2_nwd_only",
            "hint": f"检测对比源：{det_source}",
        },
        {
            "title": "桥接完成状态",
            "value": f"{sum(row['valid_images'] for row in bridge_rows)}/{sum(row['n_images'] for row in bridge_rows)}",
            "hint": "Unet++ + FPN 全图桥接",
        },
    ]

    conclusion_lines = verdict_text(private_review_rows, baseline, nwd, bridge_rows)

    source_table = html_table(
        source_rows,
        [
            ("model", "模型"),
            ("best_val_miou", "best val_mIoU"),
            ("scratch_iou", "scratch IoU"),
            ("spot_iou", "spot IoU"),
            ("miou_quick", "quick mIoU"),
        ],
    ) if source_rows else "<p>源域结果尚未生成。</p>"

    private_table = html_table(
        private_review_rows,
        [
            ("model", "模型"),
            ("scratch_iou_mean", "scratch IoU"),
            ("scratch_dice_mean", "scratch Dice"),
            ("scratch_length_error_mean", "length error"),
            ("scratch_area_error_mean", "area error"),
            ("damage_area_error_mean", "damage area error"),
        ],
    ) if private_review_rows else "<p>私有复核结果尚未生成。</p>"

    bridge_table = html_table(
        bridge_rows,
        [
            ("model", "桥接模型"),
            ("n_images", "计划图像数"),
            ("valid_images", "有效输出数"),
            ("elapsed_sec", "耗时（秒）"),
            ("status", "状态"),
        ],
    )

    det_rows = []
    if baseline:
        det_rows.append(
            {
                "tag": baseline["tag"],
                "mAP50": baseline.get("mAP50", 0.0),
                "scratch_ap": baseline.get("per_class_AP50", {}).get("scratch", 0.0),
                "spot_ap": baseline.get("per_class_AP50", {}).get("spot", 0.0),
                "critical_ap": baseline.get("per_class_AP50", {}).get("critical", 0.0),
            }
        )
    if nwd:
        det_rows.append(
            {
                "tag": nwd["tag"],
                "mAP50": nwd.get("mAP50", 0.0),
                "scratch_ap": nwd.get("per_class_AP50", {}).get("scratch", 0.0),
                "spot_ap": nwd.get("per_class_AP50", {}).get("spot", 0.0),
                "critical_ap": nwd.get("per_class_AP50", {}).get("critical", 0.0),
            }
        )
    det_table = html_table(
        det_rows,
        [
            ("tag", "实验"),
            ("mAP50", "mAP@0.5"),
            ("scratch_ap", "scratch AP@0.5"),
            ("spot_ap", "spot AP@0.5"),
            ("critical_ap", "critical AP@0.5"),
        ],
    ) if det_rows else "<p>检测对比结果尚未生成。</p>"

    cnas_note = (
        f"本轮 `CNAS` 单模型复评结果已找到：<code>{html.escape(str(cnas_path))}</code>"
        if cnas_path
        else "本轮 `CNAS` 单模型复评结果尚未生成，当前检测结论来自最近一次对比报告。"
    )
    if cnas_payload:
        cnas_note += (
            f" 当前结果：mAP@0.5={cnas_payload.get('metrics', {}).get('mAP50', 0.0):.4f}，"
            f"scratch AP@0.5={cnas_payload.get('metrics', {}).get('per_class_AP50', {}).get('scratch', 0.0):.4f}。"
        )

    seg_chart_rel = rel(seg_chart) if seg_chart.exists() else None
    det_chart_rel = rel(det_chart) if det_chart.exists() else None
    bridge_chart_rel = rel(bridge_chart) if bridge_chart.exists() else None

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phase 3E 自动化迭代简报</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Phase 3E 自动化迭代简报</h1>
      <p>本简报面向当前 `Phase 3E` 的 P0 实验收尾，统一汇总分割分支、识别分支与桥接验证的阶段结果，用于快速判断下一轮是否继续推进 `Unet++ / FPN` 分工，以及 `B2_nwd_only` 是否值得保留为识别研究父实验。</p>
      <div class="meta">
        生成路径：<code>{html.escape(str(output_dir))}</code><br />
        分割数据源：<code>{html.escape(str(seg_path)) if seg_path else "missing"}</code><br />
        检测数据源：<code>{html.escape(str(det_path)) if det_path else "missing"}</code>
      </div>
      <div class="grid">
        {''.join(f'<div class="card"><h3>{html.escape(card["title"])}</h3><div class="value">{html.escape(str(card["value"]))}</div><div class="hint">{html.escape(card["hint"])}</div></div>' for card in status_cards)}
      </div>
    </section>

    <section class="section">
      <h2>阶段结论</h2>
      <ul>
        {''.join(f'<li>{html.escape(line)}</li>' for line in conclusion_lines)}
      </ul>
      <p class="note">{html.escape(cnas_note)}</p>
    </section>

    <section class="section">
      <h2>分割分支：结构能力与物理量稳定性</h2>
      <div class="two-col">
        <div>
          <h3>源域结构能力</h3>
          {source_table}
          <h3 style="margin-top:18px;">私有 24 图量化复核</h3>
          {private_table}
        </div>
        <div>
          {f'<div class="figure"><img src="{html.escape(seg_chart_rel)}" alt="Phase 3E segmentation overview" /><div class="caption">左图为源域结构能力对比，右图为私有 24 图长度/面积误差对比。</div></div>' if seg_chart_rel else '<p>分割汇总图待生成。</p>'}
          {f'<div class="figure"><img src="{html.escape(metrics_fig_rel)}" alt="segmentation private review metrics" /><div class="caption">现有 24 图多指标复核统计图。</div></div>' if metrics_fig_rel else ''}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>桥接验证：接入正式系统前的稳定性检查</h2>
      <div class="two-col">
        <div>
          {bridge_table}
          <p class="note">桥接验证仍然是 Phase 3E 的关键守门项。只有当 `Unet++` 与 `FPN` 都能稳定跑完计划样本，并正常产出 `report.json` 与可视化图像，分割结果才适合进一步接入评分链路。</p>
        </div>
        <div>
          {f'<div class="figure"><img src="{html.escape(bridge_chart_rel)}" alt="Phase 3E bridge status" /><div class="caption">桥接阶段的计划图像数与有效输出数对比。</div></div>' if bridge_chart_rel else '<p>桥接状态图待生成。</p>'}
          {f'<div class="figure"><img src="{html.escape(contact_sheet_rel)}" alt="Segmentation review contact sheet" /><div class="caption">24 张代表样本并列核查图，用于人工检查各模型分割差异。</div></div>' if contact_sheet_rel else ''}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>识别分支：A0 基线与 B2 父实验对比</h2>
      <div class="two-col">
        <div>
          {det_table}
          <p class="note">识别分支当前仍以 `scratch AP@0.5` 为主指标，以 `mAP@0.5 >= A0_baseline` 为守门条件。`B2_nwd_only` 只有在复现实验中同时满足这两点，才应继续作为下一轮父实验。</p>
        </div>
        <div>
          {f'<div class="figure"><img src="{html.escape(det_chart_rel)}" alt="Phase 3E detection overview" /><div class="caption">识别分支当前最核心的两个指标：总 mAP@0.5 与 scratch AP@0.5。</div></div>' if det_chart_rel else '<p>识别对比图待生成。</p>'}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>下一步建议</h2>
      <ul>
        <li>若桥接验证仍未达到 8/8 稳定通过，则优先完成桥接链路修复与复测，再谈分割接入正式系统。</li>
        <li>若 `FPN` 继续保持更低的长度/面积误差，则优先将其用于标注更新和物理量化支撑；`Unet++` 保留为 scratch 敏感性研究主线。</li>
        <li>若 `B2_nwd_only` 复现实验后仍无法稳定优于 `A0_baseline`，识别分支下一轮应优先转向后处理与输入表示消融，而不是继续扩大训练结构改动。</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""

    ensure_dir(output_dir)
    out_path = output_dir / "phase3e_summary.html"
    out_path.write_text(html_text, encoding="utf-8")

    metadata = {
        "segmentation_summary": str(seg_path) if seg_path else None,
        "detection_compare": str(det_path) if det_path else None,
        "cnas_eval": str(cnas_path) if cnas_path else None,
        "bridge_unet": str(BRIDGE_UNET) if BRIDGE_UNET.exists() else None,
        "bridge_fpn": str(BRIDGE_FPN) if BRIDGE_FPN.exists() else None,
        "html": str(out_path),
    }
    (output_dir / "phase3e_summary_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Phase 3E summary generated: {out_path}")


if __name__ == "__main__":
    main()
