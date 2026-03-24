"""背景修正.

仅保留模板背景驱动的线性修正逻辑。
"""

from __future__ import annotations

import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


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
