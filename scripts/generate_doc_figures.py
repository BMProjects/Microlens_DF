#!/usr/bin/env python3
"""生成项目综合文档使用的示意图."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.font_manager as fm


def _use_cjk_font() -> None:
    """让图中的中文正常渲染。找不到中文字体时静默退回默认字体。"""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [
                fm.FontProperties(fname=path).get_name(),
                "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
            return


_use_cjk_font()


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


# 架构图配色：生产=蓝，数据=琥珀，研发=绿，第三方测试=红
_ARCH_C = dict(
    prod="#2f6fb0", prod_bg="#eaf2fb", prod_fill="#d3e5f7",
    data="#b07d0a", data_bg="#fdf6e6", data_fill="#f8e9c4",
    rnd="#3f8f4f", rnd_bg="#eef7ef", rnd_fill="#d6ecd9",
    test="#b3453c", test_bg="#fcefee", test_fill="#f6d8d5",
    ink="#1c2733", mute="#5d6b7a",
)


def _band(ax, x, y, w, h, title, bg, edge, tag=None):
    """一条横向泳道，代表一个主线/层。"""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                       facecolor=bg, edgecolor=edge, lw=1.6, alpha=0.95, zorder=1)
    )
    ax.text(x + 1.8, y + h - 2.8, title, fontsize=13, fontweight="bold", color=edge, zorder=4)
    if tag:
        ax.text(x + w - 1.8, y + h - 2.8, tag, fontsize=9.5, color=edge,
                ha="right", va="center", zorder=4, style="italic")


def _mod(ax, x, y, w, h, num, name, sub, fill, edge, strong=True):
    """一个模块方框。strong=False 用虚线边框表示研发/可选模块。"""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
                       facecolor=fill, edgecolor=edge, lw=2.0 if strong else 1.5,
                       linestyle="-" if strong else (0, (4, 2)), zorder=3)
    )
    cx = x + w / 2
    if num:
        ax.text(cx, y + h - 3.4, num, fontsize=15, fontweight="bold", color=edge,
                ha="center", va="center", zorder=4)
        ax.text(cx, y + h - 7.4, name, fontsize=11.5, fontweight="bold",
                color=_ARCH_C["ink"], ha="center", va="center", zorder=4)
    else:
        ax.text(cx, y + h - 4.8, name, fontsize=11.5, fontweight="bold",
                color=_ARCH_C["ink"], ha="center", va="center", zorder=4)
    ax.text(cx, y + 2.8, sub, fontsize=9, color=_ARCH_C["mute"],
            ha="center", va="center", zorder=4)


def _arr(ax, p1, p2, color, label=None, dashed=False, rad=0.0, lw=2.0, lp=None, fs=9.2):
    ax.add_patch(
        FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                        zorder=6, linestyle="--" if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}")
    )
    if label:
        mx, my = lp if lp else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(mx, my, label, fontsize=fs, color=color, ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=0.7, alpha=0.95))


def build_system_architecture() -> None:
    """系统总架构：双主线 + 八模块。

    布局约定（改动前请先读）：
      · 三条横带自上而下 = GUI 生产主线 / 数据与研发基础 / 独立第三方测试
      · x 在 72–100 之间保留为空白走线廊道，"待测权重"竖直下行经过此处，
        不穿越任何模块带；新增模块请勿占用该区间
      · 实线 = 生产路径，虚线 = 可选或研发路径
      · 模块编号与 doc/系统架构_模块化梳理_20260531.md 的 ① – ⑧ 一一对应
    """
    c = _ARCH_C
    fig, ax = plt.subplots(figsize=(15.5, 9.4))
    ax.set_xlim(0, 160)
    ax.set_ylim(-4, 100)
    ax.axis("off")

    ax.text(80, 96.6, "暗场镜片缺陷检测系统 · 模块架构", fontsize=19.5,
            fontweight="bold", ha="center", color=c["ink"])
    ax.text(80, 92.4, "双主线：GUI 生产主线（② ④ ⑥ ⑦）　｜　研发与测试分支（① ③ ④ ⑤ ⑥ ⑧）",
            fontsize=11.5, ha="center", color=c["mute"])

    # ── 带一：GUI 生产主线 ──
    _band(ax, 4, 64, 152, 24, "GUI 生产主线", c["prod_bg"], c["prod"],
          "面向非专业操作员 · 单张图像端到端 600–800 ms")
    bw, bh, by = 25, 15, 66.5
    xs = [7, 36, 65, 94, 123]
    _mod(ax, xs[0], by, bw, bh, "", "镜片图像输入",
         "PNG / BMP / TIFF\n4096×3000 灰度", "#ffffff", c["prod"])
    _mod(ax, xs[1], by, bw, bh, "②", "预处理",
         "背景融合 · ROI 提取\n配准 · 亮度校正", c["prod_fill"], c["prod"])
    _mod(ax, xs[2], by, bw, bh, "④", "识别（检测）",
         "YOLOv12m + SAHI 切片\nIOS 合并 · 擦痕接续", c["prod_fill"], c["prod"])
    _mod(ax, xs[3], by, bw, bh, "⑥", "评估与评分",
         "mAP 指标 · WearScore\nA / B / C / D 评级", c["prod_fill"], c["prod"])
    _mod(ax, xs[4], by, bw, bh, "⑦", "前端工作台",
         "Gradio 四步流程\n四联结果 + 结论", c["prod_fill"], c["prod"])
    for i in range(4):
        _arr(ax, (xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2), c["prod"])

    # ── 带二左：数据与标注基础 ──
    dy, dh = 34.5, 15
    _band(ax, 4, 32, 68, 24, "数据与标注基础", c["data_bg"], c["data"], "数据与代码物理分离")
    _mod(ax, 8, dy, 24, dh, "①", "数据集",
         "247 全图 · 10,621 切片\n划分与清单 manifest", c["data_fill"], c["data"])
    _mod(ax, 44, dy, 24, dh, "③", "半监督标注",
         "标签清洗 · 伪标签\n弱掩膜 · 标注审计", c["data_fill"], c["data"])
    _arr(ax, (32, dy + 10.2), (44, dy + 10.2), c["data"], "训练数据", lp=(38, dy + 13.6), fs=8.8)
    _arr(ax, (44, dy + 4.4), (32, dy + 4.4), c["data"], "标签回流", lp=(38, dy + 1.2), fs=8.8)

    # ── 带二右：研发分支 ──
    _band(ax, 100, 32, 56, 24, "研发分支", c["rnd_bg"], c["rnd"], "未进入生产默认路径")
    _mod(ax, 106, dy, 44, dh, "⑤", "分割算法",
         "HRNet-OCR / SegFormer / FPN\n像素级形态 → 结构解析与物理量",
         c["rnd_fill"], c["rnd"], strong=False)

    # ── 带三：独立第三方测试 ──
    ty, th = 6.5, 13.5
    _band(ax, 4, 1, 152, 24, "独立第三方测试（CNAS）", c["test_bg"], c["test"],
          "与生产系统解耦 · 单独运行")
    _mod(ax, 9, ty, 32, th, "", "固定留出集",
         "20 图 / 860 切片\n从不参与训练", "#ffffff", c["test"])
    _mod(ax, 68, ty, 36, th, "⑧", "CNAS 测试子系统",
         "评测 · 报告生成\n过程留痕 · 溯源", c["test_fill"], c["test"])
    _mod(ax, 116, ty, 36, th, "", "测试交付件",
         "指标 JSON · Word / HTML 报告\n过程截图与溯源信息", "#ffffff", c["test"])
    _arr(ax, (41, ty + th / 2), (68, ty + th / 2), c["test"])
    _arr(ax, (104, ty + th / 2), (116, ty + th / 2), c["test"])

    # ── 跨带连接 ──
    _arr(ax, (28, dy + dh), (44, by), c["data"], "批量数据", lp=(31.5, 61.5), fs=9, rad=-0.08)
    _arr(ax, (75, by), (60, dy + dh), c["data"], "模型预测", lp=(72.5, 61.5), fs=9, rad=-0.08)
    _arr(ax, (128, dy + dh), (108, by), c["rnd"], "可选启用\n--use-segmentation",
         lp=(123.5, 61.0), dashed=True, rad=-0.12, fs=8.6)
    # 待测权重：沿 x≈82 的空白廊道竖直下行
    _arr(ax, (82, by), (84, ty + th), c["test"], "待测权重", lp=(82, 44), dashed=True, fs=9)
    _arr(ax, (18, dy), (22, ty + th), c["test"], "固定划分", lp=(17.5, 28.5), dashed=True, fs=8.8)

    ax.text(80, 3.6,
            "硬边界：留出集不参与任何训练、模型选择与超参数调整　|　评测参数固定 conf=0.001, iou=0.6",
            fontsize=9.8, color=c["test"], ha="center", fontweight="bold")
    ax.text(4, -2.6,
            "模块编号对应《系统架构_模块化梳理》① – ⑧　·　实线 = 生产路径，虚线 = 可选 / 研发路径",
            fontsize=9, color=c["mute"], ha="left")

    ensure_dir(OUT)
    fig.savefig(OUT / "project_system_architecture.png", dpi=170,
                bbox_inches="tight", facecolor="white")
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
