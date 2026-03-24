"""背景融合 — 多帧平均 + 离焦模板生成."""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def generate_defocused_template(
    B_avg: np.ndarray,
    sigma: float,
    per_pass_sigma: float = 3.0,
    brightness_gain: float = 1.08,
) -> np.ndarray:
    """生成散焦版背景模板.

    Args:
        B_avg: 平均背景 (H, W), float64.
        sigma: 散焦高斯标准差.
        per_pass_sigma: 单次高斯模糊 sigma（控制核尺寸较小）.
        brightness_gain: 模糊后整体亮度增益.

    Returns:
        B_blur (H, W), float64.
    """
    target_sigma = float(max(0.1, sigma))
    pass_sigma = float(max(0.3, per_pass_sigma))
    # 多次小 sigma 模糊，累计到目标散焦强度
    n_pass = max(1, int(np.ceil((target_sigma / pass_sigma) ** 2)))
    sigma_each = target_sigma / np.sqrt(n_pass)

    B_blur = B_avg.astype(np.float64, copy=True)
    for _ in range(n_pass):
        B_blur = cv2.GaussianBlur(B_blur, (0, 0), sigma_each)

    gain = float(max(0.1, brightness_gain))
    B_blur = np.clip(B_blur * gain, 0.0, 255.0)
    logger.info(
        "散焦模板生成: sigma=%.2f, pass_sigma=%.2f, n_pass=%d, gain=%.3f",
        target_sigma,
        sigma_each,
        n_pass,
        gain,
    )
    return B_blur
