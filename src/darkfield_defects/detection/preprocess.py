"""图像预处理模块 — 背景校正 + ROI 提取 + 降噪 + 增强."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from darkfield_defects.detection.params import PreprocessParams
from darkfield_defects.exceptions import PreprocessError
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessResult:
    """预处理结果."""
    corrected: np.ndarray           # 校正后图像 (H, W), uint8
    roi_mask: np.ndarray            # ROI 二值掩码 (H, W), bool
    optical_center: tuple[int, int] # 光学中心 (row, col)
    lens_radius: float              # 镜片半径（像素）
    background_used: bool = False   # 是否执行了背景校正


def flat_field_correct(
    image: np.ndarray,
    background: np.ndarray,
    eps: float = 1.0,
) -> np.ndarray:
    """除法平场校正: I' = I / (B + eps) × mean(B).

    Args:
        image: 原始灰度图 (H, W), uint8.
        background: 背景图 (H, W), float64.
        eps: 防除零常数.

    Returns:
        校正后图像 (H, W), uint8.
    """
    img_f = image.astype(np.float64)
    bg_mean = np.mean(background)
    corrected = img_f / (background + eps) * bg_mean

    # 裁剪到 [0, 255]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    logger.debug("平场校正完成 (division)")
    return corrected


def flat_field_subtract(
    image: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """减法平场校正: I' = I - k·B, 其中 k = mean(I)/mean(B).

    适用于暗场图像中信号叠加在背景之上的场景。

    Args:
        image: 原始灰度图 (H, W), uint8.
        background: 背景图 (H, W), float64.

    Returns:
        校正后图像 (H, W), uint8.
    """
    img_f = image.astype(np.float64)
    bg_mean = np.mean(background)
    img_mean = np.mean(img_f)

    k = img_mean / max(bg_mean, 1e-6)
    corrected = img_f - k * background

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    logger.debug(f"平场校正完成 (subtraction, k={k:.4f})")
    return corrected


def local_contrast_enhance(
    image: np.ndarray,
    sigma: float = 80.0,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """局部对比度增强 — 消除残余背景渐变.

    通过大核高斯估计局部背景，除法校正使暗区/边缘的
    微弱划痕获得与中心区同等的对比度。

    Args:
        image: 灰度图 (H, W), uint8.
        sigma: 高斯核标准差 (越大 → 估计越平滑).
        roi_mask: ROI 掩码，非 ROI 区域不处理.

    Returns:
        局部校正后图像 (H, W), uint8.
    """
    img_f = image.astype(np.float64)

    # 大核高斯估计局部背景
    local_bg = cv2.GaussianBlur(img_f, (0, 0), sigma)

    # 避免除零
    local_bg = np.maximum(local_bg, 1.0)

    # 局部除法校正: I_out = I / local_bg × global_mean
    global_mean = np.mean(img_f[roi_mask]) if roi_mask is not None and np.any(roi_mask) else np.mean(img_f)
    enhanced = img_f / local_bg * global_mean

    # 裁剪到有效范围
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    if roi_mask is not None:
        enhanced = np.where(roi_mask, enhanced, image)

    logger.debug(f"局部对比度增强完成, sigma={sigma:.0f}")
    return enhanced


def extract_roi(
    image: np.ndarray,
    method: str = "hough",
    min_radius_ratio: float = 0.3,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """提取镜片圆形 ROI.

    Args:
        image: 灰度图 (H, W).
        method: "hough" 或 "threshold".
        min_radius_ratio: 最小半径占图像短边比例.

    Returns:
        (roi_mask, center, radius)
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    min_radius = int(min_dim * min_radius_ratio)
    max_radius = int(min_dim * 0.5)

    if method == "hough":
        return _extract_roi_hough(image, min_radius, max_radius)
    elif method == "threshold":
        return _extract_roi_threshold(image, min_radius)
    else:
        raise PreprocessError(f"未知 ROI 方法: {method}")


def _extract_roi_hough(
    image: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Hough 圆检测提取 ROI."""
    # 轻度模糊减噪
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max_radius,
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        logger.warning("Hough 圆检测失败，回退到阈值方法")
        return _extract_roi_threshold(image, min_radius)

    # 选择最大的圆
    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    logger.info(f"ROI (Hough): center=({cy},{cx}), radius={r}")
    return mask.astype(bool), (cy, cx), float(r)


def _extract_roi_threshold(
    image: np.ndarray,
    min_radius: int,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """阈值+轮廓方法提取 ROI."""
    # 增强对比度后二值化
    blurred = cv2.GaussianBlur(image, (15, 15), 5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise PreprocessError("未找到镜片轮廓")

    # 选面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(largest)
    cx, cy, r = int(cx), int(cy), int(r)

    if r < min_radius:
        raise PreprocessError(f"检测到的圆半径 {r} < 最小要求 {min_radius}")

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    logger.info(f"ROI (Threshold): center=({cy},{cx}), radius={r}")
    return mask.astype(bool), (cy, cx), float(r)


def shrink_roi(
    roi_mask: np.ndarray,
    center: tuple[int, int],
    radius: float,
    shrink_ratio: float = 0.05,
) -> np.ndarray:
    """内缩 ROI 去除边缘光环.

    Args:
        roi_mask: 原始 ROI 掩码.
        center: 中心坐标 (row, col).
        radius: 原始半径.
        shrink_ratio: 内缩比例.

    Returns:
        内缩后的 ROI 掩码.
    """
    new_radius = int(radius * (1.0 - shrink_ratio))
    h, w = roi_mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = center
    cv2.circle(mask, (cx, cy), new_radius, 255, -1)
    logger.debug(f"ROI 内缩: {radius:.0f} → {new_radius}")
    return mask.astype(bool)


def denoise(
    image: np.ndarray,
    method: str = "bilateral",
    params: Optional[PreprocessParams] = None,
) -> np.ndarray:
    """降噪处理.

    Args:
        image: 输入图像 (H, W), uint8.
        method: "bilateral" 或 "nlm".
        params: 预处理参数.

    Returns:
        降噪后图像.
    """
    if params is None:
        params = PreprocessParams()

    if method == "bilateral":
        result = cv2.bilateralFilter(
            image,
            d=params.bilateral_d,
            sigmaColor=params.bilateral_sigma_color,
            sigmaSpace=params.bilateral_sigma_space,
        )
    elif method == "nlm":
        result = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        logger.warning(f"未知降噪方法: {method}, 跳过")
        return image

    logger.debug(f"降噪完成: {method}")
    return result


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """CLAHE 自适应直方图均衡化."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    result = clahe.apply(image)
    logger.debug("CLAHE 增强完成")
    return result


def preprocess_image(
    image: np.ndarray,
    background: Optional[np.ndarray] = None,
    params: Optional[PreprocessParams] = None,
) -> PreprocessResult:
    """完整预处理流水线.

    Args:
        image: 原始灰度图 (H, W), uint8.
        background: 可选背景图用于平场校正.
        params: 预处理参数.

    Returns:
        PreprocessResult.
    """
    if params is None:
        params = PreprocessParams()

    # 1. 背景校正
    bg_used = False
    if background is not None:
        if params.bg_correction_method == "subtraction":
            image = flat_field_subtract(image, background)
        else:
            image = flat_field_correct(image, background, eps=params.bg_epsilon)
        bg_used = True

    # 2. ROI 提取
    roi_mask, center, radius = extract_roi(
        image,
        method=params.roi_method,
        min_radius_ratio=params.roi_min_radius_ratio,
    )

    # 3. 内缩去除边缘光环
    roi_mask = shrink_roi(roi_mask, center, radius, params.roi_shrink_ratio)

    # 4. 局部亮度补偿 (消除残余渐变)
    if params.local_enhance_enabled:
        image = local_contrast_enhance(image, params.local_enhance_sigma, roi_mask)

    # 5. 降噪
    image = denoise(image, method=params.denoise_method, params=params)

    # 6. 可选 CLAHE 增强
    if params.clahe_enabled:
        image = enhance_contrast(image, params.clahe_clip_limit, params.clahe_tile_size)

    # 7. ROI 蒙版应用
    image = np.where(roi_mask, image, 0).astype(np.uint8)

    return PreprocessResult(
        corrected=image,
        roi_mask=roi_mask,
        optical_center=center,
        lens_radius=radius,
        background_used=bg_used,
    )
