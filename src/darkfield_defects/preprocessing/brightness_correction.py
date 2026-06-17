"""背景修正.

包含两层背景模型：
- 模板背景驱动的线性修正（全局 B_blur，标定阶段得到，去高光ring/整体明暗）；
- 单图背景面残差（形态学开运算估计的逐图缓变背景面，去雾霾/亮缺陷晕染/ring残留），
  供缺陷分割的背景锚定阈值复用——把"预处理背景修正"与"阈值分割"统一到同一背景模型。
"""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def estimate_background_surface(image: np.ndarray, open_ksize: int = 41) -> np.ndarray:
    """形态学开运算估计逐图缓变背景面.

    开运算（先腐蚀后膨胀）抹掉比结构元小的亮结构（细划痕、斑点），保留缓变的
    背景/雾霾/高光ring与亮缺陷晕染等大尺度成分 → 即"背景面"。

    Args:
        image: 灰度图 (H, W) uint8.
        open_ksize: 椭圆结构元直径，应大于最大划痕宽度（默认 41）。
    Returns:
        背景面 (H, W) float32。
    """
    img8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    return cv2.morphologyEx(img8, cv2.MORPH_OPEN, se).astype(np.float32)


def background_surface_residual(
    image: np.ndarray,
    roi_mask: np.ndarray,
    open_ksize: int = 41,
) -> np.ndarray:
    """背景面残差 R = I − 背景面（白顶帽），ROI 外置 0.

    残差里缓变雾霾、亮缺陷晕染、高光ring残留被减除，仅留相对局部背景显著凸起的
    尖锐结构（细划痕、斑点），且背景近 0，便于固定阈值分割。
    """
    img_f = image.astype(np.float32)
    bg = estimate_background_surface(image, open_ksize)
    residual = img_f - bg
    residual[~roi_mask.astype(bool)] = 0.0
    return residual


def estimate_linear_params(
    target: np.ndarray,
    bg_template: np.ndarray,
    roi_mask: np.ndarray,
    scratch_mask: np.ndarray | None = None,
    ring_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """估计目标图与背景模板之间的线性修正参数."""
    del scratch_mask

    omega = roi_mask.astype(bool).copy()
    if ring_mask is not None:
        omega &= ~ring_mask.astype(bool)

    if np.count_nonzero(omega) < 100:
        logger.warning("背景采样区域太小，回退到 a=1, b=0")
        return 1.0, 0.0

    target_vals = target[omega].astype(np.float64)
    bg_vals = bg_template[omega].astype(np.float64)

    target_med = float(np.median(target_vals))
    bg_med = float(np.median(bg_vals))
    target_mad = float(np.median(np.abs(target_vals - target_med)))
    bg_mad = float(np.median(np.abs(bg_vals - bg_med)))

    gain = 1.0 if bg_mad < 1e-6 else target_mad / bg_mad
    bias = target_med - gain * bg_med
    return gain, bias


def apply_linear_correction(
    target: np.ndarray,
    bg_template: np.ndarray,
    a: float,
    b: float,
    mode: str = "subtract",
) -> np.ndarray:
    """应用背景模板修正."""
    target_f = target.astype(np.float64)
    bg_f = bg_template.astype(np.float64)

    if mode == "subtract":
        return target_f - (a * bg_f + b)
    if mode == "normalize":
        return (target_f - b) / max(a, 1e-6)
    raise ValueError(f"不支持的校正模式: {mode}")


def finalize_image(
    corrected: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """ROI 外置黑并转换到 uint8."""
    result = np.where(roi_mask.astype(bool), corrected, 0.0)
    return np.clip(result, 0, 255).astype(np.uint8)
