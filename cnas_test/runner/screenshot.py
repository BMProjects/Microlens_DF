"""测试过程截图采集：终端关键节点 + 结果摘要图。

为满足"算法识别准确率测试通行做法"中对测试过程留痕的要求：
- 关键节点：测试启动横幅、测试集与权重确认、评测结果汇总
- 通过将文本以图片形式渲染保存为 PNG，避免依赖 X server，
  在 RustDesk / NoMachine / SSH 等任意会话下均可复现
- 同时生成一张结果摘要图（条形图：各类别 AP50 + mAP50）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _resolve_cjk_font() -> str | None:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Noto Serif CJK JP",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "SimHei",
        "Droid Sans Fallback",
        "UKIJ CJK",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CJK_FONT = _resolve_cjk_font()
if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT]
    plt.rcParams["axes.unicode_minus"] = False
_TEXT_FONT_FAMILY = _CJK_FONT or "monospace"


def render_text_screenshot(
    lines: list[str],
    save_path: Path,
    *,
    title: str = "",
    width_in: float = 12.0,
    line_height_in: float = 0.28,
    margin_in: float = 0.5,
    font_size: int = 11,
) -> Path:
    """把终端文本（含 ANSI 已剥离）渲染为 PNG，保留行序与等宽对齐。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = max(len(lines), 1) + (2 if title else 0)
    height_in = margin_in * 2 + n_lines * line_height_in

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 1 - margin_in / height_in
    if title:
        ax.text(
            0.01,
            y,
            title,
            family=_TEXT_FONT_FAMILY,
            fontsize=font_size + 2,
            fontweight="bold",
            verticalalignment="top",
        )
        y -= 2 * line_height_in / height_in

    for line in lines:
        ax.text(
            0.01,
            y,
            line.rstrip(),
            family=_TEXT_FONT_FAMILY,
            fontsize=font_size,
            verticalalignment="top",
        )
        y -= line_height_in / height_in

    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def render_result_summary(
    metrics: dict[str, Any],
    pass_threshold: float,
    save_path: Path,
) -> Path:
    """各类别 AP50 + mAP50 条形图。pass_threshold 保留为兼容旧调用。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = metrics["per_class_AP50"]
    names = list(per_class.keys()) + ["mAP50"]
    values = list(per_class.values()) + [metrics["mAP50"]]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = ["#4C78A8"] * len(per_class) + ["#F58518"]
    bars = ax.bar(names, values, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP@0.5")
    ax.set_title("CNAS 测试 — 各类别 AP50 与 mAP50")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(save_path, facecolor="white")
    plt.close(fig)
    return save_path


def collect_screenshots(
    save_dir: Path,
    *,
    startup_lines: list[str],
    result_lines: list[str],
    metrics: dict[str, Any],
    pass_threshold: float,
) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    startup_png = render_text_screenshot(
        startup_lines,
        save_dir / "01_startup_banner.png",
        title="CNAS 测试 — 启动确认",
    )
    result_png = render_text_screenshot(
        result_lines,
        save_dir / "02_result_summary.png",
        title="CNAS 测试 — 结果汇总",
    )
    summary_png = render_result_summary(metrics, pass_threshold, save_dir / "03_metrics_chart.png")
    return {
        "startup": startup_png,
        "result_text": result_png,
        "metrics_chart": summary_png,
    }
