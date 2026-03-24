"""ROI 掩膜构建 — 基于高光ring提取.

核心思路:
  1) 以B_blur中心区域亮度为基准（中心均一灰度），
     threshold = center_level × center_ratio，自适应分离中心灰与ring。
  2) 形态学膨胀桥接ring不连续区域，形成完整屏障。
  3) 取中心连通域作为ROI，向内缩进留安全边距。
"""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def _estimate_center_level(
    image: np.ndarray,
    center_sample_ratio: float = 0.40,
) -> tuple[float, float]:
    """估计图像中心区域的亮度基准（中位数和MAD）.

    Args:
        image: 输入图像，float64.
        center_sample_ratio: 取中心区域的边长比例（如0.4 → 中心40%×40%区域）.

    Returns:
        (center_median, center_mad)
    """
    h, w = image.shape[:2]
    r = float(np.clip(center_sample_ratio, 0.1, 0.8))
    y0 = int(h * (1 - r) / 2)
    y1 = int(h * (1 + r) / 2)
    x0 = int(w * (1 - r) / 2)
    x1 = int(w * (1 + r) / 2)
    patch = image[y0:y1, x0:x1].ravel().astype(np.float64)
    # 排除极黑像素（边角溢出等）
    patch = patch[patch > 5.0]
    if len(patch) < 100:
        patch = image.ravel().astype(np.float64)
        patch = patch[patch > 5.0]
    median = float(np.median(patch))
    mad    = float(np.median(np.abs(patch - median)))
    return median, mad


def build_highlight_structure_mask(
    bg_template: np.ndarray,
    center_ratio: float = 1.25,
    center_sample_ratio: float = 0.40,
    edge_expand_px: int = 25,
    pre_blur_sigma: float = 0.0,
    return_steps: bool = False,
) -> tuple[np.ndarray, dict[str, float]] | tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """从背景失焦模板提取高光ring区域.

    阈值策略（中心相对法）:
      - 在B_blur中心区域采样，得到中心亮度基准 center_level
      - threshold = center_level × center_ratio
      - 保证阈值始终高于中心均一灰度，低于ring的高亮区域
      - 适应不同镜头、不同曝光的背景图像

    Args:
        bg_template: 离焦背景模板 B_blur，float64.
        center_ratio: 阈值倍率，threshold = center_level × ratio.
                      1.25 意味着比中心亮度高 25% 才算ring.
        center_sample_ratio: 用于估计中心亮度的内区比例（边长占比）.
        edge_expand_px: ring掩膜膨胀像素数，用于桥接ring过渡区断缝.
        pre_blur_sigma: 预模糊 sigma（减少亮点噪声干扰），通常不需要.
        return_steps: 是否返回中间步骤图像.

    Returns:
        (ring_mask, stats_dict) 或含 steps 的三元组.
    """
    img = bg_template.astype(np.float64)
    h, w = img.shape[:2]

    if pre_blur_sigma > 0.0:
        img = cv2.GaussianBlur(img, (0, 0), float(pre_blur_sigma))

    # ① 中心亮度基准
    center_level, center_mad = _estimate_center_level(img, center_sample_ratio)

    # ② 中心相对阈值：center_level × ratio，且至少比中心高 3σ（MAD×4.5）
    ratio   = float(max(1.05, center_ratio))
    th_abs  = center_level * ratio
    th_min  = center_level + max(center_mad * 4.5, center_level * 0.10)   # 至少高10%
    th_val  = max(th_abs, th_min)

    highlight = img > th_val

    logger.info(
        "高亮ring阈值: center_level=%.1f  MAD=%.2f  ratio=%.2f  threshold=%.1f",
        center_level, center_mad, ratio, th_val,
    )

    # ③ 去除零星噪点（小连通域）
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    highlight = cv2.morphologyEx(
        highlight.astype(np.uint8), cv2.MORPH_OPEN, kernel3,
    ) > 0

    # ④ 膨胀：桥接ring过渡区断缝，形成连续屏障
    expand = max(1, int(edge_expand_px))
    k = 2 * expand + 1
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(highlight.astype(np.uint8), kernel_expand, iterations=1) > 0

    # ⑤ 保留面积达标的连通域（过滤孤立亮点）
    min_area = max(200, int(round(h * w * 0.00003)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated.astype(np.uint8))
    out   = np.zeros_like(dilated, dtype=bool)
    kept  = 0
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lid] = True
            kept += 1

    stats_out = {
        "center_level": center_level,
        "center_mad":   center_mad,
        "center_ratio": ratio,
        "bright_thresh": th_val,
        "edge_expand_px": float(expand),
        "min_area":      float(min_area),
        "mask_pixels":   float(np.count_nonzero(out)),
        "components_kept": float(kept),
    }
    logger.info(
        "高亮ring提取: th=%.1f expand=%dpx kept=%d px=%d",
        th_val, expand, kept, int(np.count_nonzero(out)),
    )

    if not return_steps:
        return out, stats_out
    steps = {
        "threshold_highlight": highlight.astype(bool),
        "dilated_highlight":   dilated.astype(bool),
        "edge_barrier":        out.astype(bool),
    }
    return out, stats_out, steps


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _largest_non_border_component(mask: np.ndarray) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best_id   = -1
    best_area = -1
    for lid in range(1, n):
        comp = labels == lid
        if _touches_border(comp):
            continue
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_id   = lid
    if best_id <= 0:
        return None
    return labels == best_id


def build_roi_from_highlight_mask(
    highlight_mask: np.ndarray,
    erode_diameter_ratio: float = 0.002,
    return_steps: bool = False,
) -> tuple[np.ndarray, dict[str, float]] | tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """由高亮ring掩膜构建中心ROI区域.

    思路:
      1) 高亮ring视作"屏障"，取其反相区域的中心连通域。
      2) 若中心域泄漏到图像边界，逐步增加闭运算修补缺口。
      3) 距离变换向内收缩，留安全边距。
    """
    # 对ring做闭运算，修补小缺口
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    barrier = cv2.morphologyEx(highlight_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    h, w    = barrier.shape[:2]
    cy, cx  = h // 2, w // 2

    short_edge       = float(min(h, w))
    close_candidates = [0,
                        int(round(short_edge * 0.01)),
                        int(round(short_edge * 0.02))]

    roi       = None
    used_close = 0
    for k in close_candidates:
        kk = int(max(0, k))
        if kk > 0 and kk % 2 == 0:
            kk += 1
        b = barrier
        if kk > 1:
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
            b = cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE, kernel2) > 0

        inv = ~b
        n, labels, _, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8))
        if n <= 1:
            continue
        center_id = int(labels[cy, cx])
        if center_id <= 0:
            continue
        cand = labels == center_id
        if not _touches_border(cand):
            roi        = cand
            used_close = kk
            break

    if roi is None:
        inv = ~barrier
        roi = _largest_non_border_component(inv)
        used_close = 0
        if roi is None:
            n, labels, _, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8))
            if n > 1 and int(labels[cy, cx]) > 0:
                roi = labels == int(labels[cy, cx])
            else:
                roi = np.zeros_like(barrier, dtype=bool)

    roi_area     = float(np.count_nonzero(roi))
    roi_diameter = 2.0 * np.sqrt(max(roi_area, 1.0) / np.pi)
    erode_px     = int(round(max(0.0, float(erode_diameter_ratio)) * roi_diameter))
    center_roi   = roi.copy()
    refined      = roi
    if erode_px > 0 and np.any(refined):
        dist    = cv2.distanceTransform(refined.astype(np.uint8), cv2.DIST_L2, 3)
        refined = dist > float(erode_px)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8))
    if n > 1:
        lid     = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        refined = labels == lid

    stats_out = {
        "roi_pixels":    float(np.count_nonzero(refined)),
        "roi_diameter":  roi_diameter,
        "erode_ratio":   float(erode_diameter_ratio),
        "erode_px":      float(erode_px),
        "close_used":    float(used_close),
    }
    logger.info(
        "ROI(高亮模板): close=%d erode=%.2f%%(%dpx) roi_px=%d",
        int(stats_out["close_used"]),
        stats_out["erode_ratio"] * 100.0,
        int(stats_out["erode_px"]),
        int(stats_out["roi_pixels"]),
    )
    if not return_steps:
        return refined, stats_out
    steps = {
        "center_roi_raw": center_roi.astype(bool),
        "roi_final":      refined.astype(bool),
    }
    return refined, stats_out, steps
