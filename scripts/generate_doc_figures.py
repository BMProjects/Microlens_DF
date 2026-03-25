#!/usr/bin/env python3
"""生成项目综合文档使用的示意图."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "doc" / "assets" / "generated"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#f5f7fb", ec="#334155", text_color="#0f172a", size=10):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=text_color)


def arrow(ax, x1, y1, x2, y2, color="#475569", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=1.5,
            color=color,
        )
    )


def build_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.05, 0.66, 0.18, 0.14, "Lens Image Input", fc="#e0f2fe")
    add_box(ax, 0.30, 0.66, 0.18, 0.14, "GUI Workbench\nGradio", fc="#dbeafe")
    add_box(ax, 0.55, 0.66, 0.20, 0.14, "Application Service\ninference_service", fc="#dcfce7")
    add_box(ax, 0.79, 0.66, 0.16, 0.14, "Outputs\nScore / Overlay", fc="#fae8ff")

    add_box(ax, 0.18, 0.38, 0.18, 0.14, "Preprocess & ROI\nregistration / background", fc="#fff7ed")
    add_box(ax, 0.42, 0.38, 0.18, 0.14, "Detection Mainline\nYOLO + SAHI + scratch merge", fc="#fef3c7")
    add_box(ax, 0.66, 0.38, 0.18, 0.14, "Unified Result\nDetectionResult", fc="#fef9c3")

    add_box(ax, 0.12, 0.12, 0.22, 0.14, "Scoring System\nL_center / L_total / N / A / S_scatter", fc="#ede9fe")
    add_box(ax, 0.40, 0.12, 0.22, 0.14, "Independent CNAS Test\nhold-out set & report", fc="#fee2e2")
    add_box(ax, 0.68, 0.12, 0.22, 0.14, "R&D Branches\ndetection improvement / segmentation", fc="#ecfccb")

    arrow(ax, 0.23, 0.73, 0.30, 0.73)
    arrow(ax, 0.48, 0.73, 0.55, 0.73)
    arrow(ax, 0.75, 0.73, 0.79, 0.73)

    arrow(ax, 0.39, 0.66, 0.28, 0.52)
    arrow(ax, 0.58, 0.66, 0.51, 0.52)
    arrow(ax, 0.75, 0.66, 0.75, 0.52)

    arrow(ax, 0.27, 0.45, 0.42, 0.45)
    arrow(ax, 0.60, 0.45, 0.66, 0.45)

    arrow(ax, 0.66, 0.38, 0.23, 0.26)
    arrow(ax, 0.66, 0.38, 0.51, 0.26)
    arrow(ax, 0.66, 0.38, 0.79, 0.26)

    ax.text(0.5, 0.93, "Current Product Architecture", fontsize=18, fontweight="bold", ha="center")
    ax.text(
        0.5,
        0.89,
        "GUI mainline, independent testing, and research branches are decoupled",
        fontsize=11,
        ha="center",
        color="#475569",
    )

    plt.tight_layout()
    ensure_dir(OUT)
    fig.savefig(OUT / "project_system_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_algorithm_branches() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Detection and Segmentation Branches", fontsize=18, fontweight="bold", ha="center")
    ax.text(0.5, 0.90, "Detection supports the current product. Segmentation supports refinement, physical quantification, and relabeling.", fontsize=11, ha="center", color="#475569")

    add_box(ax, 0.07, 0.72, 0.20, 0.13, "Private Lens Data\nfull images / tiles / weak labels", fc="#e0f2fe")
    add_box(ax, 0.40, 0.72, 0.20, 0.13, "Detection Branch\nstage2_cleaned baseline", fc="#fef3c7")
    add_box(ax, 0.73, 0.72, 0.20, 0.13, "Segmentation Branch\nMSD pretrain + private finetune", fc="#dcfce7")

    add_box(ax, 0.36, 0.48, 0.12, 0.12, "NWD", fc="#fde68a")
    add_box(ax, 0.52, 0.48, 0.12, 0.12, "SAHI / IOS\nscratch linking", fc="#fde68a")

    add_box(ax, 0.67, 0.48, 0.10, 0.12, "Unet++", fc="#bbf7d0")
    add_box(ax, 0.80, 0.48, 0.10, 0.12, "FPN", fc="#bbf7d0")
    add_box(ax, 0.67, 0.30, 0.10, 0.12, "DeepLabV3+", fc="#bbf7d0")
    add_box(ax, 0.80, 0.30, 0.10, 0.12, "LightUNet", fc="#bbf7d0")

    add_box(ax, 0.10, 0.15, 0.28, 0.12, "Product Output\nboxes / score / GUI conclusion", fc="#fae8ff")
    add_box(ax, 0.58, 0.15, 0.32, 0.12, "Research Output\nmasks / skeleton length / area / relabeling hints", fc="#ecfccb")

    arrow(ax, 0.27, 0.78, 0.40, 0.78)
    arrow(ax, 0.27, 0.78, 0.73, 0.78)

    arrow(ax, 0.50, 0.72, 0.42, 0.60)
    arrow(ax, 0.50, 0.72, 0.58, 0.60)
    arrow(ax, 0.46, 0.48, 0.24, 0.27)
    arrow(ax, 0.58, 0.48, 0.24, 0.27)

    arrow(ax, 0.83, 0.72, 0.72, 0.60)
    arrow(ax, 0.83, 0.72, 0.85, 0.60)
    arrow(ax, 0.83, 0.72, 0.72, 0.42)
    arrow(ax, 0.83, 0.72, 0.85, 0.42)
    arrow(ax, 0.72, 0.30, 0.72, 0.27)
    arrow(ax, 0.85, 0.30, 0.85, 0.27)
    arrow(ax, 0.77, 0.48, 0.72, 0.27)
    arrow(ax, 0.85, 0.48, 0.85, 0.27)
    arrow(ax, 0.74, 0.21, 0.58, 0.21)

    plt.tight_layout()
    ensure_dir(OUT)
    fig.savefig(OUT / "project_algorithm_branches.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_execution_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(3.1, 0.92, "Project Execution Overview (2025-10 to 2026-03)", fontsize=18, fontweight="bold", ha="center")

    phases = [
        ("2025-10", "Research\nrequirements & route", "#dbeafe"),
        ("2025-11", "Preprocessing\nROI / registration / background", "#e0f2fe"),
        ("2025-12", "Data cleanup\noptical normalization", "#f0f9ff"),
        ("2026-01", "Tiling & labels\ntraining baseline", "#fef3c7"),
        ("2026-02", "Full-image inference\nscore & GUI", "#fde68a"),
        ("2026-03", "Independent testing\nsegmentation branch", "#dcfce7"),
    ]
    x = 0.2
    for month, label, color in phases:
        add_box(ax, x, 0.40, 0.85, 0.22, f"{month}\n{label}", fc=color, size=11)
        x += 0.95
    for i in range(5):
        arrow(ax, 1.05 + i * 0.95, 0.51, 1.15 + i * 0.95, 0.51)

    ax.text(3.1, 0.18, "The project has converged from exploratory code into a GUI mainline, an independent test subsystem, and dual R&D branches.", fontsize=11, ha="center", color="#334155")

    plt.tight_layout()
    ensure_dir(OUT)
    fig.savefig(OUT / "project_execution_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_system_architecture()
    build_algorithm_branches()
    build_execution_overview()


if __name__ == "__main__":
    main()
