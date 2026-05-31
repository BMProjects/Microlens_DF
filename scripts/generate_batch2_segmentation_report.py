#!/usr/bin/env python3
"""生成 batch2 分割实验对比分析报告."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PHASE3 = ROOT / "output/experiments/phase3_segmentation"
OUT_DIR = PHASE3 / "analysis" / "batch2_compare"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def best_source_metrics(path: Path) -> dict[str, float | int]:
    history = read_json(path)
    best = max(history, key=lambda row: row.get("val_miou", -1.0))
    return {
        "status": "completed",
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_miou": float(best["val_miou"]),
        "best_loss": float(best["loss"]),
    }


def best_private_metrics(path: Path) -> dict[str, float | int]:
    history = read_json(path)
    best = min(history, key=lambda row: row.get("loss", 1e9))
    return {
        "status": "completed",
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_loss": float(best["loss"]),
    }


def hrnet_metrics(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    return read_json(path)


def plot_source(rows: list[dict], out_path: Path) -> None:
    available = [r for r in rows if r.get("source_status") == "completed"]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.bar(
        [r["model"] for r in available],
        [r["source_best_val_miou"] for r in available],
        color=["#2563eb", "#ea580c", "#16a34a"][: len(available)],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("best val_mIoU")
    ax.set_title("Batch2 Source-Domain Comparison")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_private(rows: list[dict], out_path: Path) -> None:
    available = [r for r in rows if r.get("private_status") == "completed"]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.bar(
        [r["model"] for r in available],
        [r["private_best_loss"] for r in available],
        color=["#1d4ed8", "#c2410c", "#15803d"][: len(available)],
    )
    ax.set_ylabel("best loss")
    ax.set_title("Batch2 Private Weak-Label Comparison")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_status(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    labels = [r["model"] for r in rows]
    source_vals = [1 if r["source_status"] == "completed" else 0 for r in rows]
    private_vals = [1 if r["private_status"] == "completed" else 0 for r in rows]
    x = list(range(len(rows)))
    ax.bar([i - 0.18 for i in x], source_vals, width=0.36, label="source", color="#60a5fa")
    ax.bar([i + 0.18 for i in x], private_vals, width=0.36, label="private", color="#34d399")
    ax.set_xticks(x, labels)
    ax.set_yticks([0, 1], ["pending", "done"])
    ax.set_title("Batch2 Experiment Completion Status")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        vals = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                vals.append("-")
            elif isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def html_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{title}</th>" for _, title in columns)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": "FPN",
            "framework": "SMP",
            "source_status": "completed",
            **best_source_metrics(PHASE3 / "batch2_fpn_msd" / "history.json"),
            "private_status": "completed",
            **{f"private_{k}": v for k, v in best_private_metrics(PHASE3 / "batch2_fpn_private" / "history.json").items()},
        },
        {
            "model": "SegFormer-B2",
            "framework": "HF Transformers",
            "source_status": "completed",
            **best_source_metrics(PHASE3 / "batch2_segformer_b2_hf_msd" / "history.json"),
            "private_status": "completed",
            **{f"private_{k}": v for k, v in best_private_metrics(PHASE3 / "batch2_segformer_b2_hf_private" / "history.json").items()},
        },
    ]

    local_hrnet_source = PHASE3 / "batch2_hrnet_ocr_w18_msd" / "history.json"
    local_hrnet_private = PHASE3 / "batch2_hrnet_ocr_w18_private" / "history.json"
    if local_hrnet_source.exists() and local_hrnet_private.exists():
        rows.append(
            {
                "model": "HRNet-OCR-W18",
                "framework": "Local HRNet-OCR",
                "source_status": "completed",
                **best_source_metrics(local_hrnet_source),
                "private_status": "completed",
                **{f"private_{k}": v for k, v in best_private_metrics(local_hrnet_private).items()},
            }
        )
    else:
        hrnet_source = hrnet_metrics(PHASE3 / "batch2_hrnet_ocr_official_msd" / "summary.json")
        hrnet_private = hrnet_metrics(PHASE3 / "batch2_hrnet_ocr_official_private" / "summary.json")
        rows.append(
            {
                "model": "HRNet-OCR-W18",
                "framework": "HRNet official package",
                "source_status": hrnet_source.get("status", "missing"),
                "source_best_epoch": hrnet_source.get("best_epoch"),
                "source_best_val_miou": hrnet_source.get("best_val_miou"),
                "source_best_loss": hrnet_source.get("best_loss"),
                "private_status": hrnet_private.get("status", "missing"),
                "private_best_epoch": hrnet_private.get("best_epoch"),
                "private_best_loss": hrnet_private.get("best_loss"),
                "private_best_val_miou": hrnet_private.get("best_val_miou"),
            }
        )

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "model": row["model"],
                "framework": row["framework"],
                "source_status": row.get("source_status", "missing"),
                "source_best_epoch": row.get("best_epoch", row.get("source_best_epoch")),
                "source_best_val_miou": row.get("best_val_miou", row.get("source_best_val_miou")),
                "source_best_loss": row.get("best_loss", row.get("source_best_loss")),
                "private_status": row.get("private_status", "missing"),
                "private_best_epoch": row.get("private_best_epoch"),
                "private_best_loss": row.get("private_best_loss"),
                "private_best_val_miou": row.get("private_best_val_miou"),
            }
        )

    source_chart = OUT_DIR / "batch2_source_compare.png"
    private_chart = OUT_DIR / "batch2_private_compare.png"
    status_chart = OUT_DIR / "batch2_status_compare.png"
    plot_source(normalized_rows, source_chart)
    plot_private(normalized_rows, private_chart)
    plot_status(normalized_rows, status_chart)

    summary = {
        "rows": normalized_rows,
        "conclusions": [
            "FPN 当前仍是 batch2 中最稳的 CNN 主线：源域 `best_val_miou` 和私有微调 `best_loss` 都优于 SegFormer-B2。",
            "SegFormer-B2 已跑通并形成有效对照，但当前在本私有弱标签条件下尚未超过 FPN，更适合作为 Transformer 补充线。",
            "HRNet-OCR-W18 现在优先采用本地训练结果；若本地结果尚未生成，才回退显示官方实验包准备状态。",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Batch2 分割实验对比报告</title>
  <style>
    body {{ margin:0; background:#f5f7fb; color:#0f172a; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
    .page {{ max-width:1100px; margin:28px auto; background:#fff; padding:32px 40px 48px; border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.08); }}
    h1, h2 {{ margin-top:0; }}
    p, li {{ line-height:1.8; }}
    img {{ max-width:100%; border:1px solid #e2e8f0; border-radius:12px; margin:14px 0 22px; }}
    table {{ width:100%; border-collapse:collapse; margin:14px 0 22px; }}
    th, td {{ border:1px solid #cbd5e1; padding:10px 12px; text-align:left; }}
    th {{ background:#eff6ff; }}
    code {{ background:#eff6ff; padding:1px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Batch2 分割实验定量对比分析</h1>
    <p>当前报告统一比较 <code>FPN</code>、<code>SegFormer-B2</code> 和 <code>HRNet-OCR-W18</code> 三条 batch2 路线。HRNet 优先读取本地训练结果；若本地结果尚未生成，则回退显示官方实验包准备状态。</p>

    <h2>1. 结构与训练状态</h2>
    {html_table(normalized_rows, [
        ("model", "模型"),
        ("framework", "框架"),
        ("source_status", "源域状态"),
        ("source_best_val_miou", "源域 best val_mIoU"),
        ("source_best_epoch", "源域 best epoch"),
        ("private_status", "私有状态"),
        ("private_best_loss", "私有 best loss"),
        ("private_best_epoch", "私有 best epoch"),
    ])}

    <h2>2. 图表</h2>
    <img src="batch2_source_compare.png" alt="batch2 source compare" />
    <img src="batch2_private_compare.png" alt="batch2 private compare" />
    <img src="batch2_status_compare.png" alt="batch2 status compare" />

    <h2>3. 当前结论</h2>
    <ul>
      <li>{summary["conclusions"][0]}</li>
      <li>{summary["conclusions"][1]}</li>
      <li>{summary["conclusions"][2]}</li>
    </ul>

    <h2>4. 下一步</h2>
    <p>当本地 <code>HRNet-OCR-W18</code> 训练完成后，直接重跑本报告脚本即可自动纳入最终三模型排名；若仍采用官方仓单独训练，则继续使用 <code>scripts/register_hrnet_official_result.py</code> 回填结果。</p>
  </main>
</body>
</html>
"""
    (OUT_DIR / "batch2_report.html").write_text(report, encoding="utf-8")
    print(OUT_DIR / "batch2_report.html")


if __name__ == "__main__":
    main()
