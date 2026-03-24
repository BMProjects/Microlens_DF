"""暗场镜片缺陷检测系统 — 操作员工作台（重构版）
================================================
布局：左控制面板（25%）+ 右结果面板（75%  2×2 网格）
流程：上传图像 → 开始检测 → 检测叠加图 / 缺陷地形图 / 评分过程 / 评级卡片
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Noto Sans CJK JP", "Noto Serif CJK JP",
                           "WenQuanYi Micro Hei", "Noto Sans CJK SC",
                           "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":         110,
})

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install darkfield-defects[viz]")

from darkfield_defects.logging import get_logger
from darkfield_defects.measurement import DEFAULT_CALIBRATION
from darkfield_defects.app_services import (
    get_default_weights_path,
    run_full_image_inference,
)

logger = get_logger(__name__)

# ── 路径 ──────────────────────────────────────────────────────────────
_PROJECT_ROOT    = Path(__file__).parent.parent.parent.parent
_DEFAULT_WEIGHTS = get_default_weights_path()

# ── 视觉常量 ──────────────────────────────────────────────────────────
_BG       = "#0d1117"
_CARD     = "#161b27"
_BORDER   = "#2a2f3e"
_ACCENT   = "#2979ff"
_TEXT_DIM = "#666e7a"

CLASS_NAMES      = ["scratch", "spot", "critical"]
CLASS_COLORS_BGR = {0: (30, 200, 60),   1: (0, 170, 255),  2: (200, 80, 255)}
CLASS_COLORS_RGB = {0: (60, 200, 30),   1: (255, 170, 0),  2: (255, 80, 200)}
CLASS_LABELS_CN  = {0: "划痕",          1: "斑点",          2: "缺损"}

GRADE_COLOR  = {"A": "#00c853", "B": "#2979ff", "C": "#ff9100", "D": "#d50000"}
GRADE_BG     = {"A": "#00c85318","B": "#2979ff18","C": "#ff910018","D": "#d5000018"}
GRADE_SHORT  = {"A": "极佳", "B": "良好", "C": "警告", "D": "报废"}

# contributor 键名 → 中文标签、权重
_CONTRIB_META = [
    ("center_length",     "中心区划痕", 0.30),
    ("effective_length",  "区域加权长度", 0.20),
    ("defect_count",      "区域加权缺陷数", 0.15),
    ("defect_area",       "区域加权面积", 0.20),
    ("scatter_intensity", "区域加权散射", 0.15),
]


# ══════════════════════════════════════════════════════════════════════
#  可视化工具函数
# ══════════════════════════════════════════════════════════════════════

def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    """matplotlib Figure → RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    arr = np.array(PILImage.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _placeholder(label: str = "等待检测...",
                 w: int = 640, h: int = 420) -> np.ndarray:
    """深色占位图，带居中提示文字。"""
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.65, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.putText(img, label,
                ((w - tw) // 2, (h + th) // 2),
                font, scale, (55, 65, 80), thick, cv2.LINE_AA)
    return img


# ── 检测叠加图 ────────────────────────────────────────────────────────

def _make_overlay(boxes: list, img_gray: np.ndarray) -> np.ndarray:
    """在灰度图上绘制彩色框 + 图例，返回 RGB。"""
    H, W = img_gray.shape
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    counts = {0: 0, 1: 0, 2: 0}
    for cls, x1, y1, x2, y2, *_ in boxes:
        color  = CLASS_COLORS_BGR[cls]
        thick  = 2 if cls == 0 else 1
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        counts[cls] += 1

    # 图例（左下角）
    leg_y = H - 10
    for cls in (2, 1, 0):
        label = f"{CLASS_LABELS_CN[cls]}: {counts[cls]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (8, leg_y - th - 8),
                      (8 + tw + 22, leg_y + 4), (20, 20, 30), -1)
        cv2.rectangle(vis, (10, leg_y - th//2 - 3),
                      (20, leg_y - th//2 + 5),
                      CLASS_COLORS_BGR[cls], -1)
        cv2.putText(vis, label, (24, leg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        leg_y -= th + 14

    # 缩放到显示宽度 1024px
    scale  = 1024 / W
    vis_s  = cv2.resize(vis, (1024, int(H * scale)),
                        interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(vis_s, cv2.COLOR_BGR2RGB)


# ── 缺陷地形图 ────────────────────────────────────────────────────────

def _make_topo_map(boxes: list, img_h: int, img_w: int) -> np.ndarray:
    """KDE 热力图 + 区域圆环 + 各类型散点，返回 RGB。

    关键：以镜片圆心和实际半径归一化坐标，确保镜片始终呈现为正圆。
    镜片外的缺陷在网格中自然落到圆外，不再显示。
    """
    GRID = 300
    # 镜片几何参数（图像坐标系）
    cx_img    = img_w / 2.0
    cy_img    = img_h / 2.0
    lens_r_px = min(img_h, img_w) * 0.43   # 与 boxes_to_defects 保持一致

    # 网格中心和镜片圆半径（square grid → 正圆）
    cx_g, cy_g = GRID // 2, GRID // 2
    r_g = GRID * 0.44   # 镜片占 88% 网格宽度

    def to_grid(bx: float, by: float):
        """图像像素坐标 → 网格坐标（以镜片中心和半径归一化）。"""
        ux = (bx - cx_img) / lens_r_px   # 归一化：镜片边缘 = ±1
        uy = (by - cy_img) / lens_r_px
        return cx_g + ux * r_g, cy_g + uy * r_g

    # 方形 figure，比例 1:1，确保正圆不变形
    fig, ax = plt.subplots(figsize=(5.0, 5.0), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")

    # ── KDE 密度（只对镜片内的框累加）──
    density = np.zeros((GRID, GRID), dtype=float)
    for cls, x1, y1, x2, y2, conf, *_ in boxes:
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        # 检查是否在镜片内
        r_norm = np.sqrt((bx - cx_img) ** 2 + (by - cy_img) ** 2) / lens_r_px
        if r_norm > 1.0:
            continue
        px, py  = to_grid(bx, by)
        area    = (x2 - x1) * (y2 - y1)
        sigma   = np.clip(np.sqrt(area) / lens_r_px * r_g * 0.8, 2, 12)
        weight  = (1.0 + 0.5 * (cls == 2)) * float(conf)
        xx, yy  = np.meshgrid(np.arange(GRID), np.arange(GRID))
        density += weight * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)
        )

    # 圆形掩膜（镜片圆）
    yy_m, xx_m = np.ogrid[:GRID, :GRID]
    lens_mask   = (xx_m - cx_g) ** 2 + (yy_m - cy_g) ** 2 <= r_g ** 2
    density_masked = np.where(lens_mask, density, np.nan)

    # 颜色映射
    cmap = LinearSegmentedColormap.from_list(
        "defect_heat",
        ["#0d1117", "#0a2a6e", "#8b0000", "#ff4500", "#ffd700", "#ffffff"],
    )
    vmax = np.nanmax(density_masked) if np.nanmax(density_masked) > 0 else 1
    im = ax.imshow(density_masked, cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="gaussian", origin="upper",
                   extent=[0, GRID, GRID, 0])

    # ── 区域圆环（半径与 boxes_to_defects 的区域划分完全一致）──
    for frac, alpha, ls in [
        (0.40, 0.65, "--"),
        (0.75, 1.00, "--"),
    ]:
        ax.add_patch(plt.Circle((cx_g, cy_g), r_g * frac,
                                fill=False, color="white",
                                linestyle=ls, linewidth=0.9, alpha=alpha))

    # 外轮廓
    ax.add_patch(plt.Circle((cx_g, cy_g), r_g,
                             fill=False, color="white",
                             linewidth=1.4, alpha=0.85))

    # 各区中间位置文字标注
    ax.text(cx_g, cy_g, "中心区", ha="center", va="center",
            fontsize=6.5, color="#aaaaaa", alpha=0.6)
    ax.text(cx_g + r_g * 0.57, cy_g, "微结构区",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold", alpha=0.85)
    ax.text(cx_g + r_g * 0.88, cy_g, "边缘区",
            ha="center", va="center",
            fontsize=6, color="#aaaaaa", alpha=0.5)

    # ── 各类型散点（只显示镜片内的框，随机采样最多 200 个）──
    sample_max = 200
    for cls in (0, 1, 2):
        pts = []
        for b in boxes:
            if b[0] != cls:
                continue
            bx, by = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
            r_norm = np.sqrt((bx - cx_img) ** 2 + (by - cy_img) ** 2) / lens_r_px
            if r_norm > 1.0:
                continue
            pts.append(to_grid(bx, by))
        if not pts:
            continue
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), min(len(pts), sample_max), replace=False)
        xs  = [pts[i][0] for i in idx]
        ys  = [pts[i][1] for i in idx]
        c   = tuple(v / 255 for v in CLASS_COLORS_RGB[cls])
        ms  = 18 if cls == 0 else (28 if cls == 1 else 32)
        mk  = "|" if cls == 0 else ("o" if cls == 1 else "s")
        ax.scatter(xs, ys, c=[c], s=ms, marker=mk,
                   alpha=0.75, linewidths=0.8, zorder=3)

    # 图例
    handles = [mpatches.Patch(color=tuple(v / 255 for v in CLASS_COLORS_RGB[c]),
                               label=f"{CLASS_LABELS_CN[c]} "
                                     f"({sum(1 for b in boxes if b[0]==c)})")
               for c in (0, 1, 2)]
    ax.legend(handles=handles, loc="lower right",
              fontsize=7, framealpha=0.35, labelcolor="white",
              facecolor="#0d1117", edgecolor="#2a2f3e")

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("缺陷密度", color="#aaaaaa", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888888")

    ax.set_xlim(0, GRID)
    ax.set_ylim(GRID, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout(pad=0.3)
    return _fig_to_array(fig)


# ── WearScore 计算过程图 ──────────────────────────────────────────────

def _make_score_chart(contributors: dict, score: float) -> np.ndarray:
    """水平条形图（各分量）+ 半圆仪表盘（总分），返回 RGB。"""
    fig, (ax_bar, ax_gauge) = plt.subplots(
        1, 2,
        figsize=(7.5, 3.6),
        gridspec_kw={"width_ratios": [2.6, 1]},
        facecolor=_BG,
    )

    # ── 左：分量条形图 ──
    ax_bar.set_facecolor(_CARD)

    keys    = [m[0] for m in _CONTRIB_META]
    labels  = [f"{m[1]}  ×{m[2]:.0%}" for m in _CONTRIB_META]
    weights = [m[2] for m in _CONTRIB_META]
    raw_f   = [contributors.get(k, 0.0) for k in keys]
    contrib = [f * w for f, w in zip(raw_f, weights)]

    norm     = max(max(contrib), 1e-9)
    bar_clrs = plt.cm.RdYlGn_r(np.array(contrib) / max(max(raw_f), 1))

    bars = ax_bar.barh(labels, contrib,
                       color=bar_clrs, height=0.52, edgecolor="none",
                       zorder=2)
    ax_bar.barh(labels,
                [max(v, 0) for v in
                 [w * 100 - c for w, c in zip(weights, contrib)]],
                left=contrib,
                color="#1e2535", height=0.52, edgecolor="none", zorder=1)

    for bar, raw, w in zip(bars, raw_f, weights):
        x = bar.get_width()
        ax_bar.text(x + norm * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{raw:.1f} → {x:.1f}",
                    va="center", ha="left", fontsize=7.5, color="#cccccc")

    ax_bar.set_xlim(0, max(sum(weights) * 100, 1) * 0.9)
    ax_bar.tick_params(axis="y", labelsize=8.5, colors="#cccccc")
    ax_bar.tick_params(axis="x", colors="#555555", labelsize=7)
    ax_bar.set_xlabel("加权分值", color="#666e7a", fontsize=8)
    ax_bar.set_title("各因素贡献（原始分 → 加权值）",
                     color="white", fontsize=10, pad=7, fontweight="bold")
    ax_bar.set_facecolor(_CARD)
    ax_bar.figure.set_facecolor(_BG)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(_BORDER)
    ax_bar.axvline(score, color="#2979ff", linewidth=1.2,
                   linestyle=":", alpha=0.6, zorder=5)

    # ── 右：半圆仪表盘 ──
    ax_gauge.set_facecolor(_BG)

    # 四段弧：A / B / C / D
    segs = [
        (np.pi,         np.pi * 3/4,  "#00c853"),  # A  0-25
        (np.pi * 3/4,   np.pi / 2,    "#2979ff"),  # B  25-50
        (np.pi / 2,     np.pi / 5,    "#ff9100"),  # C  50-80
        (np.pi / 5,     0,            "#d50000"),  # D  80-100
    ]
    r_in, r_out = 0.58, 0.92
    for t0, t1, color in segs:
        t   = np.linspace(t0, t1, 50)
        xs  = np.concatenate([r_in * np.cos(t), r_out * np.cos(t[::-1])])
        ys  = np.concatenate([r_in * np.sin(t), r_out * np.sin(t[::-1])])
        ax_gauge.fill(xs, ys, color=color, alpha=0.88)

    # 指针
    needle_angle = np.pi * (1 - score / 100)
    ax_gauge.annotate(
        "",
        xy  =(0.72 * np.cos(needle_angle), 0.72 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=2.0, mutation_scale=12),
    )
    ax_gauge.add_patch(plt.Circle((0, 0), 0.09,
                                   color="#cccccc", zorder=6))

    # 刻度标注
    for ang, lbl in [(np.pi, "0"), (np.pi / 2, "50"), (0, "100")]:
        ax_gauge.text(1.08 * np.cos(ang), 1.08 * np.sin(ang),
                      lbl, ha="center", va="center",
                      fontsize=6.5, color="#888888")
    for ang, lbl in [(np.pi * 3/4, "25"), (np.pi / 2, "50"),
                      (np.pi / 5, "80")]:
        ax_gauge.text(0.5 * np.cos(ang), 0.5 * np.sin(ang),
                      lbl, ha="center", va="center",
                      fontsize=5.5, color="#888888")

    # 总分
    grade = ("A" if score < 25 else "B" if score < 50
             else "C" if score < 80 else "D")
    g_color = GRADE_COLOR[grade]
    ax_gauge.text(0, -0.22, f"{score:.0f}",
                  ha="center", va="center",
                  fontsize=22, fontweight="bold", color="white")
    ax_gauge.text(0, -0.46, "/ 100",
                  ha="center", va="center",
                  fontsize=8, color="#888888")
    ax_gauge.text(0, -0.63, f"Grade {grade}",
                  ha="center", va="center",
                  fontsize=9, fontweight="bold", color=g_color)

    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-0.75, 1.15)
    ax_gauge.set_title("综合评分", color="white",
                       fontsize=10, pad=6, fontweight="bold")
    ax_gauge.axis("off")

    fig.tight_layout(pad=0.6)
    return _fig_to_array(fig)


# ── 评级卡片 HTML ──────────────────────────────────────────────────────

def _make_grade_html(assessment: dict) -> str:
    grade     = assessment.get("grade", "?")
    score     = assessment.get("score", 0.0)
    label     = assessment.get("grade_label", "")
    glare     = assessment.get("glare_index", 0.0)
    haze      = assessment.get("haze_index", 0.0)
    dominant  = assessment.get("dominant_factor", "")
    gc        = GRADE_COLOR.get(grade, "#888888")
    gbg       = GRADE_BG.get(grade, "#88888818")
    gs        = GRADE_SHORT.get(grade, grade)

    # 主导因素中文名 + 通俗解释
    _dominant_info = {
        "center_length": (
            "中心区划痕长度",
            "中心视觉区内的划痕会直接干扰清晰度，是佩戴时最敏感的磨损因素。",
        ),
        "effective_length": (
            "区域加权总长度",
            "综合考虑中心区、微结构区和边缘区后，当前划痕长度负担偏高，说明镜片关键区域已有连续损伤。",
        ),
        "defect_count": (
            "区域加权缺陷数量",
            "缺陷数量在关键区域内累积较多，已不只是孤立噪点，而是整体性磨损问题。",
        ),
        "defect_area": (
            "区域加权缺陷面积",
            "缺陷覆盖面积偏大，说明已有较明显的功能性受损或严重缺陷聚集。",
        ),
        "scatter_intensity": (
            "区域加权散射强度",
            "关键区域散射偏强，可能导致眩光、雾化或夜间视觉舒适度下降。",
        ),
    }
    dominant_cn, dominant_desc = _dominant_info.get(
        dominant, (dominant, "")
    )

    def _bar(val: float, cap: float = 100,
              fg: str = "white") -> str:
        pct = min(val / max(cap, 1) * 100, 100)
        return (
            f'<div style="background:#1e2535;border-radius:4px;'
            f'height:5px;margin:3px 0 10px 0;">'
            f'<div style="background:{fg};width:{pct:.0f}%;'
            f'height:100%;border-radius:4px;'
            f'transition:width 0.4s ease;"></div></div>'
        )

    return f"""
<div style="
  background:linear-gradient(160deg,{_CARD} 0%,#0d1117 100%);
  border:1px solid {gc}44;
  border-radius:14px;
  padding:22px 24px;
  font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
  color:#e0e0e0;
  height:100%;
  box-sizing:border-box;
">

  <!-- 等级 + 分数 -->
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">
    <div style="
      width:68px;height:68px;border-radius:14px;
      background:{gbg};border:2px solid {gc};
      display:flex;align-items:center;justify-content:center;
      font-size:40px;font-weight:900;color:{gc};flex-shrink:0;
      box-shadow:0 0 18px {gc}33;
    ">{grade}</div>
    <div>
      <div style="font-size:26px;font-weight:700;color:white;
                  line-height:1.1;">
        {score:.1f}
        <span style="font-size:13px;color:#555;font-weight:400;">&thinsp;/ 100</span>
      </div>
      <div style="font-size:12px;color:{gc};margin-top:3px;
                  font-weight:500;">{gs} &nbsp;·&nbsp; {label.split('（')[0] if '（' in label else label}</div>
      <div style="font-size:11px;color:#555;margin-top:1px;">
        {label.split('（')[1].rstrip('）') if '（' in label else ''}</div>
    </div>
  </div>

  <div style="border-top:1px solid {_BORDER};margin:0 0 14px 0;"></div>

  <!-- 眩光 / 雾化 -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;
              margin-bottom:14px;">
    <div>
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;color:#888;margin-bottom:1px;">
        <span>眩光指数</span>
        <span style="color:#ff9100;font-weight:600;">{glare:.1f}</span>
      </div>
      {_bar(glare, 100, "#ff9100")}
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;color:#888;margin-bottom:1px;">
        <span>雾化指数</span>
        <span style="color:#29b6f6;font-weight:600;">{haze:.2f}</span>
      </div>
      {_bar(haze, 0.5, "#29b6f6")}
    </div>
  </div>

  <!-- 主导因素 -->
  <div style="
    background:#0d1117;
    border-left:3px solid {gc};
    border-radius:0 8px 8px 0;
    padding:11px 14px;
    font-size:12px;line-height:1.7;color:#bbbbbb;
  ">
    <span style="color:#666;font-size:11px;text-transform:uppercase;
                 letter-spacing:0.06em;">主要影响因素</span><br>
    <strong style="color:{gc};font-size:13px;">{dominant_cn}</strong><br>
    <span style="color:#8a95a3;font-size:12px;">{dominant_desc}</span>
  </div>

</div>"""


# ══════════════════════════════════════════════════════════════════════
#  主推理回调
# ══════════════════════════════════════════════════════════════════════

def run_full_detection(
    image_file,
    conf_thresh: float,
    weights_path: str,
) -> tuple:
    """SAHI 推理 → 4 个结果面板 + 状态栏 HTML。"""
    ph = _placeholder
    empty_html = (
        "<div style='padding:24px;color:#444;font-size:13px;"
        "text-align:center;'>等待检测...</div>"
    )

    if image_file is None:
        return (ph("请先上传镜片图像"), ph(),
                ph(), empty_html,
                "<div class='status-bar'>⬆️ 请先上传图像</div>")

    _root = str(_PROJECT_ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    img_path = Path(image_file)
    try:
        result = run_full_image_inference(
            img_path,
            conf_thresh=conf_thresh,
            weights_path=weights_path.strip() if weights_path else None,
        )
        img_gray = result.image_gray
        H, W = img_gray.shape
        final_boxes = result.final_boxes
        infer_time = result.infer_time
        n_chains = result.n_chains
        assessment = result.assessment.to_dict()

        # ── 四个输出 ──
        overlay    = _make_overlay(final_boxes, img_gray)
        topo       = _make_topo_map(final_boxes, H, W)
        chart      = _make_score_chart(
            assessment["contributors"],
            assessment["score"],
        )
        grade_html = _make_grade_html(assessment)

        # ── 状态栏 ──
        n       = len(final_boxes)
        counts  = {cn: sum(1 for b in final_boxes if b[0] == i)
                   for i, cn in enumerate(CLASS_NAMES)}
        grade   = assessment.get("grade", "?")
        score   = assessment.get("score", 0)
        g_color = GRADE_COLOR.get(grade, "#888")
        status  = (
            f"<div class='status-bar' style='color:#ccc;'>"
            f"✅&nbsp; 检测完成 &nbsp;{infer_time:.1f}s &nbsp;|&nbsp; "
            f"共 <strong>{n}</strong> 个缺陷："
            f"&nbsp;划痕 {counts['scratch']}"
            f"&nbsp;·&nbsp;斑点 {counts['spot']}"
            f"&nbsp;·&nbsp;缺损 {counts['critical']}"
            f"&nbsp;·&nbsp;链合并 {n_chains}"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Grade <strong style='color:{g_color};'>{grade}</strong>"
            f"&nbsp;{score:.1f}/100"
            f"</div>"
        )
        return overlay, topo, chart, grade_html, status

    except Exception as exc:
        logger.error("推理失败: %s", exc, exc_info=True)
        err = ph(f"推理失败: {str(exc)[:50]}")
        err_html = (
            f"<div style='padding:20px;color:#f44336;"
            f"font-size:13px;'>❌ {exc}</div>"
        )
        status = (
            f"<div class='status-bar' style='color:#f44336;'>"
            f"❌ 推理失败: {exc}</div>"
        )
        return err, err, err, err_html, status


# ── 图像上传时更新信息 ─────────────────────────────────────────────────

def _on_image_upload(file_path) -> str:
    if file_path is None:
        return "<p style='font-size:13px;color:#555;margin:4px 0;'>尚未选择图像</p>"
    p = Path(file_path)
    try:
        img  = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape if img is not None else (0, 0)
        kb   = p.stat().st_size / 1024
        width_mm = w * DEFAULT_CALIBRATION.pixel_size_mm
        height_mm = h * DEFAULT_CALIBRATION.pixel_size_mm
        return (
            f"<p style='font-size:13px;color:#c8d0de;margin:4px 0;"
            f"font-weight:600;'>📄 {p.name}</p>"
            f"<p style='font-size:12px;color:#7a8599;margin:2px 0;'>"
            f"{width_mm:.2f} × {height_mm:.2f} mm"
            f" &nbsp;·&nbsp; {w} × {h} px"
            f" &nbsp;·&nbsp; {kb:.0f} KB</p>"
        )
    except Exception:
        return (
            f"<p style='font-size:13px;color:#c8d0de;margin:4px 0;'>"
            f"📄 {p.name}</p>"
        )


# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════

_CSS = """
/* ── 全局 ── */
body, .gradio-container {
  background-color: #0d1117 !important;
  font-family: -apple-system, 'Noto Sans CJK JP', 'PingFang SC',
               'Microsoft YaHei', 'Inter', sans-serif !important;
  font-size: 14px !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ── 顶栏 ── */
.app-header {
  padding: 16px 28px;
  background: linear-gradient(90deg, #0d1117 0%, #161b27 60%, #0d1117 100%);
  border-bottom: 1px solid #1e2535;
  display: flex; align-items: center; gap: 14px;
}
.app-header-title { font-size: 20px; font-weight: 700; color: #e8eaf0; margin: 0; }
.app-header-sub   { font-size: 13px; color: #8a95a3; margin: 3px 0 0 0; }
.app-header-badge {
  margin-left: auto;
  font-size: 11px; color: #c0cadd;
  border: 1px solid #3a4558;
  border-radius: 20px; padding: 4px 12px;
  background: #1a2030;
}

/* ── 左控制面板 ── */
.left-panel {
  background: #10151f !important;
  border-right: 1px solid #1e2535 !important;
  min-height: 100vh; padding: 18px 16px !important;
}

/* 步骤标题：大号白色，带序号色块 */
.section-title {
  font-size: 13px; font-weight: 700; color: #dde3ee;
  letter-spacing: 0.02em;
  margin: 20px 0 10px 0; padding: 5px 10px;
  display: block;
  background: #1a2030;
  border-left: 3px solid #5c7aaa;
  border-radius: 0 6px 6px 0;
}
.section-title:first-child { margin-top: 4px; }

/* ── 状态栏 ── */
.status-bar {
  font-size: 13px; color: #b0bac6;
  background: #0d1117; border: 1px solid #1e2535;
  border-radius: 8px; padding: 10px 14px; margin-top: 14px;
  line-height: 1.6;
}

/* ── 开始检测按钮 ── */
.detect-btn > button, button.detect-btn {
  background: linear-gradient(135deg, #1565c0, #1976d2) !important;
  border: none !important; border-radius: 10px !important;
  font-size: 16px !important; font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  padding: 14px 0 !important; width: 100% !important;
  color: white !important;
  box-shadow: 0 4px 14px #1976d244 !important;
  transition: all 0.25s !important;
}
.detect-btn > button:hover, button.detect-btn:hover {
  background: linear-gradient(135deg, #1976d2, #2196f3) !important;
  box-shadow: 0 6px 20px #1976d255 !important;
  transform: translateY(-1px) !important;
}

/* ── 结果区标签 ── */
.result-label {
  font-size: 13px; color: #c8d0de; font-weight: 700;
  letter-spacing: 0.04em;
  margin: 0 0 6px 0; padding: 0;
}

/* ── 图像显示：保持原始宽高比，禁止拉伸 ── */
.result-img img, .result-img canvas {
  border-radius: 10px !important;
  object-fit: contain !important;
}

/* ── Accordion ── */
.gradio-accordion {
  background: #0d1117 !important;
  border: 1px solid #1e2535 !important;
  border-radius: 8px !important;
  margin-top: 6px !important;
}

/* ── 滑块 ── */
input[type=range] { accent-color: #2979ff !important; }

/* ── Gradio 默认 label 颜色 ── */
label.svelte-1b6s6s, .block > label > span {
  color: #8a95a3 !important; font-size: 13px !important;
}

/* ── 输入框 ── */
.gradio-textbox textarea, .gradio-textbox input {
  background: #0d1117 !important;
  border: 1px solid #1e2535 !important;
  color: #cccccc !important;
  border-radius: 7px !important;
  font-size: 13px !important;
}

/* ── 分割线 ── */
.divider {
  border: none; border-top: 1px solid #1e2535;
  margin: 14px 0;
}
"""


# ══════════════════════════════════════════════════════════════════════
#  应用构建
# ══════════════════════════════════════════════════════════════════════

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0d1117",
    block_background_fill="#161b27",
    block_border_color="#1e2535",
    block_label_text_color="#4a5568",
    input_background_fill="#0d1117",
    button_primary_background_fill="linear-gradient(135deg,#1565c0,#1976d2)",
    button_primary_text_color="#ffffff",
    slider_color="#2979ff",
)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="离焦微结构镜片缺陷检测及评价系统") as app:

        # ── 顶栏 ─────────────────────────────────────────────────────
        gr.HTML("""
<div class="app-header">
  <span style="font-size:24px;line-height:1;">🔬</span>
  <div>
    <p class="app-header-title">离焦微结构镜片缺陷检测及评价系统</p>
    <p class="app-header-sub">全图切片推理 &nbsp;·&nbsp; IOS 跨片合并 &nbsp;·&nbsp; 微结构区优先评分</p>
  </div>
  <span class="app-header-badge">操作员工作台 v2.1</span>
</div>""")

        with gr.Row(equal_height=False, variant="panel"):

            # ════════════════════════════════════════════════════════
            # 左：控制面板
            # ════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=270,
                           elem_classes="left-panel"):

                gr.HTML('<span class="section-title">① 导入图像</span>')

                image_input = gr.Image(
                    label="选择镜片图像（PNG / BMP / TIFF）",
                    type="filepath",
                    height=195,
                    show_label=True,
                    buttons=["download"],
                )
                image_info = gr.HTML(
                    "<p style='font-size:13px;color:#555;margin:4px 0;'>"
                    "尚未选择图像</p>"
                )

                gr.HTML('<hr class="divider">'
                        '<span class="section-title">② 检测设置</span>')

                conf_slider = gr.Slider(
                    label="置信度阈值",
                    minimum=0.05, maximum=0.50,
                    value=0.20, step=0.05,
                    info="↓ 低阈值：召回更多 &nbsp;|&nbsp; ↑ 高阈值：精度更高",
                )

                with gr.Accordion("高级设置", open=False):
                    weights_input = gr.Textbox(
                        label="模型权重路径（留空使用默认）",
                        placeholder=str(_DEFAULT_WEIGHTS),
                        lines=1,
                    )

                gr.HTML('<hr class="divider">'
                        '<span class="section-title">③ 执行检测</span>')

                detect_btn = gr.Button(
                    "🚀  开始检测",
                    variant="primary",
                    elem_classes="detect-btn",
                    size="lg",
                )

                status_out = gr.HTML(
                    "<div class='status-bar'>"
                    "⬆️ 上传图像后点击「开始检测」"
                    "</div>"
                )

            # ════════════════════════════════════════════════════════
            # 右：结果面板  2 × 2
            # ════════════════════════════════════════════════════════
            with gr.Column(scale=3):

                # ── 上行：叠加图 + 地形图 ──────────────────────────
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.HTML('<p class="result-label">检测结果</p>')
                        overlay_out = gr.Image(
                            label="缺陷标注叠加图",
                            show_label=False,
                            elem_classes="result-img",
                            height=390,
                            value=_placeholder("上传图像后开始检测"),
                            buttons=["download"],
                        )
                    with gr.Column():
                        gr.HTML('<p class="result-label">缺陷地形图</p>')
                        topo_out = gr.Image(
                            label="缺陷密度分布",
                            show_label=False,
                            elem_classes="result-img",
                            height=390,
                            value=_placeholder("缺陷密度热力图"),
                            buttons=["download"],
                        )

                # ── 下行：评分过程 + 评级卡片 ──────────────────────
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.HTML('<p class="result-label">WearScore 计算过程</p>')
                        chart_out = gr.Image(
                            label="各因素贡献",
                            show_label=False,
                            elem_classes="result-img",
                            height=310,
                            value=_placeholder("检测后显示评分分解"),
                            buttons=["download"],
                        )
                    with gr.Column():
                        gr.HTML('<p class="result-label">磨损评级</p>')
                        grade_out = gr.HTML(
                            "<div style='padding:28px 24px;color:#333;"
                            "font-size:13px;text-align:center;'>"
                            "等待检测...</div>"
                        )

        # ── 事件绑定 ──────────────────────────────────────────────
        image_input.upload(
            fn=_on_image_upload,
            inputs=[image_input],
            outputs=[image_info],
        )
        image_input.change(
            fn=_on_image_upload,
            inputs=[image_input],
            outputs=[image_info],
        )
        detect_btn.click(
            fn=run_full_detection,
            inputs=[image_input, conf_slider, weights_input],
            outputs=[overlay_out, topo_out, chart_out, grade_out, status_out],
        )

    return app


# ── 启动入口 ──────────────────────────────────────────────────────────

def launch(
    share: bool = False,
    port: int = 7860,
    server_name: str = "0.0.0.0",
) -> None:
    """启动 Gradio 应用."""
    app = create_app()
    app.launch(
        share=share,
        server_port=port,
        server_name=server_name,
        theme=_THEME,
        css=_CSS,
    )


if __name__ == "__main__":
    launch()
