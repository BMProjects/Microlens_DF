"""预处理标准化：在配准后的暗场图像上得到“标准输入图像”。

步骤（第二阶段计划 v3.0 第三章）：
1. 背景补偿：减去设备背景模板，并逐图拟合残余偏置，使背景统一为 0。
   只做加性补偿，不做增益或噪声归一化，缺陷的散射光强保持原始计数单位。
2. 弥散层分离：形态学开运算提取宽于约 2·opening_radius 的低频结构；其中幅度低、
   边缘柔和的部分（离焦晕斑、宽模糊条纹、雾状底光）归入弥散层，亮或边缘锐利的宽结构
   视为真实缺陷予以保留。补偿后图像 = 标准输入 + 弥散层，弥散层单独保存。
3. 边界宽亮带剔除：沿 ROI 边界延伸的宽亮带（镜片边缘反光、高光环漏光）从该图有效区域中剔除。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StandardizeParams:
    bg_fit_percentile: float = 60.0      # 拟合偏置时只用低于该分位的像素（排除缺陷）
    opening_radius: int = 20             # 弥散层尺度（像素）；宽于约 2r 的结构进入低频分量
    diffuse_cap: float = 20.0            # 低频分量高于该值视为真实宽缺陷，不归入弥散层
    sharp_edge_ratio: float = 0.08       # 低频分量“梯度/幅度”高于该值视为边缘锐利的真实缺陷
    edge_band_px: int = 200              # 检测边界宽亮带的 ROI 外缘带宽度
    edge_wide_px: int = 31               # 宽亮带的最小宽度（开运算核尺寸）
    edge_level: float = 6.0              # 宽亮带的最低亮度（灰度）
    edge_min_contact_px: int = 150       # 与 ROI 边界的最小接触长度，区分沿边界的亮带与穿过边界的划痕
    edge_max_depth_px: int = 120         # 亮带 95% 像素距边界不超过该值；从边界斜向深入的真实缺陷不剔除
    edge_margin_px: int = 15             # 剔除区外扩


@dataclass
class StandardizeResult:
    standard: np.ndarray                 # float32，背景≈0，原始计数单位
    diffuse: np.ndarray                  # float32，弥散层
    valid_mask: np.ndarray               # bool，本图有效区域（ROI 减去边界宽亮带）
    stats: dict = field(default_factory=dict)


def fit_background_offset(aligned: np.ndarray, template: np.ndarray, roi: np.ndarray,
                          ring: np.ndarray, percentile: float = 60.0) -> float:
    """模板形状固定，稳健拟合残余偏置 b，使 aligned − template − b 的背景为 0。"""
    m = roi & ~ring
    r = (aligned[m] - template[m]).astype(np.float64)
    keep = r <= np.percentile(r, percentile)
    for _ in range(3):  # 迭代剔除偏亮像素（缺陷）
        b = float(np.median(r[keep]))
        s = 1.4826 * float(np.median(np.abs(r[keep] - b))) or 1.0
        keep = r < b + 2.5 * s
    return float(np.median(r[keep]))


def separate_diffuse(comp: np.ndarray, roi: np.ndarray,
                     p: StandardizeParams) -> tuple[np.ndarray, np.ndarray]:
    """返回 (弥散层, 被保留的宽缺陷掩膜)。在半分辨率上计算以提速。"""
    h, w = comp.shape
    half = cv2.resize(comp, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    half = cv2.GaussianBlur(half, (0, 0), 1.5)
    r = max(1, p.opening_radius // 2)
    base = cv2.morphologyEx(half, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy) / 8.0
    hard = (base > p.diffuse_cap) | (((grad / np.maximum(base, 1.0)) > p.sharp_edge_ratio) & (base > 8))
    hard = cv2.dilate(hard.astype(np.uint8), np.ones((9, 9), np.uint8))
    soft = base.copy()
    soft[hard > 0] = 0
    soft8 = np.clip(soft, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(soft8, hard, 15, cv2.INPAINT_TELEA).astype(np.float32)
    filled = cv2.GaussianBlur(filled, (0, 0), 4)
    diffuse = cv2.resize(np.maximum(filled, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    diffuse[~roi] = 0
    kept = cv2.resize(hard, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return diffuse, kept & roi


def detect_edge_bands(comp: np.ndarray, roi: np.ndarray, p: StandardizeParams) -> np.ndarray:
    """沿 ROI 边界延伸的宽亮带掩膜（镜片边缘反光、高光环漏光）。"""
    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    band = (dist > 0) & (dist <= p.edge_band_px)
    img = np.clip(comp, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.edge_wide_px, p.edge_wide_px))
    wide = (cv2.morphologyEx(img, cv2.MORPH_OPEN, k) > p.edge_level) & band
    n, lab, st, _ = cv2.connectedComponentsWithStats(wide.astype(np.uint8), connectivity=8)
    contact = (dist > 0) & (dist <= 10)
    out = np.zeros_like(roi)
    for i in range(1, n):
        comp_i = lab == i
        if int((comp_i & contact).sum()) < p.edge_min_contact_px * 10:  # 接触带宽 10px
            continue
        if float(np.percentile(dist[comp_i], 95)) <= p.edge_max_depth_px:
            out |= comp_i
    if out.any():
        out = cv2.dilate(out.astype(np.uint8), np.ones((2 * p.edge_margin_px + 1,) * 2, np.uint8)) > 0
    return out & roi


def standardize(aligned: np.ndarray, template: np.ndarray, roi: np.ndarray, ring: np.ndarray,
                params: StandardizeParams | None = None) -> StandardizeResult:
    p = params or StandardizeParams()
    aligned = aligned.astype(np.float32)
    template = template.astype(np.float32)
    offset = fit_background_offset(aligned, template, roi, ring, p.bg_fit_percentile)
    comp = aligned - template - offset
    comp[~roi] = 0
    edge = detect_edge_bands(comp, roi, p)
    valid = roi & ~edge
    diffuse, kept_wide = separate_diffuse(comp, roi, p)
    standard = comp - diffuse
    standard[~valid] = 0
    diffuse[~valid] = 0
    v = standard[valid]
    low = v[v <= np.median(v)]
    noise = 1.4826 * float(np.median(np.abs(low - np.median(low)))) if low.size else 0.0
    bgpix = v[v < max(3 * noise, 1.0)]
    stats = {
        "bg_offset": round(offset, 3),
        "noise_sigma": round(noise, 3),
        "standard_bg_median": round(float(np.median(bgpix)), 3) if bgpix.size else None,
        "diffuse_mean": round(float(diffuse[valid].mean()), 3),
        "diffuse_p99": round(float(np.percentile(diffuse[valid], 99)), 3),
        "kept_wide_px": int(kept_wide.sum()),
        "edge_excluded_px": int(edge.sum()),
        "valid_px": int(valid.sum()),
    }
    return StandardizeResult(standard=standard, diffuse=diffuse, valid_mask=valid, stats=stats)
