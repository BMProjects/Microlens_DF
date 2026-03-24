"""特征提取模块 — Frangi/Hessian 线结构增强 + Gabor 纹理 + 顶帽变换."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from darkfield_defects.detection.params import DetectionParams
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def frangi_filter(
    image: np.ndarray,
    sigmas: list[float] | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    dark_on_bright: bool = False,
) -> np.ndarray:
    """多尺度 Frangi 滤波器增强管状/线状结构.

    暗场成像下划痕为亮线结构 (bright on dark)，使用 Hessian 矩阵特征值分析.

    Args:
        image: 灰度图 (H, W), uint8 或 float.
        sigmas: Gaussian 尺度列表.
        alpha: 控制 blob-like vs plate-like 响应.
        beta: 控制背景噪声抑制.
        gamma: 结构强度阈值.
        dark_on_bright: 若 True 则检测暗线（暗场下不需要）.

    Returns:
        Frangi 响应图 (H, W), float64, 值域 [0, 1].
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 3.0, 5.0]

    img_f = image.astype(np.float64)
    h, w = img_f.shape
    max_response = np.zeros((h, w), dtype=np.float64)

    for sigma in sigmas:
        # 计算 Hessian 矩阵元素
        Ixx = gaussian_filter(img_f, sigma, order=[0, 2])
        Ixy = gaussian_filter(img_f, sigma, order=[1, 1])
        Iyy = gaussian_filter(img_f, sigma, order=[2, 0])

        # 特征值分析（2D Hessian）
        # λ1, λ2 为特征值，|λ1| <= |λ2|
        tmp = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
        lambda1 = 0.5 * (Ixx + Iyy + tmp)
        lambda2 = 0.5 * (Ixx + Iyy - tmp)

        # 确保 |lambda1| <= |lambda2|
        abs1 = np.abs(lambda1)
        abs2 = np.abs(lambda2)
        swap = abs1 > abs2
        lambda1[swap], lambda2[swap] = lambda2[swap].copy(), lambda1[swap].copy()

        # 对于亮线（bright tubular structures），lambda2 应为大负值
        if not dark_on_bright:
            # 只保留 lambda2 < 0 的区域
            valid = lambda2 < 0
        else:
            valid = lambda2 > 0

        lambda2_safe = np.where(np.abs(lambda2) < 1e-10, 1e-10, lambda2)

        # Blobness ratio Rb = |λ1| / |λ2|
        Rb = np.abs(lambda1) / np.abs(lambda2_safe)

        # Structureness S = sqrt(λ1² + λ2²)
        S = np.sqrt(lambda1**2 + lambda2**2)

        # Frangi vesselness
        vesselness = np.exp(-Rb**2 / (2 * alpha**2)) * (1 - np.exp(-S**2 / (2 * gamma**2)))

        # 尺度归一化
        vesselness *= sigma**2

        vesselness[~valid] = 0
        max_response = np.maximum(max_response, vesselness)

    # 归一化到 [0, 1]
    vmax = max_response.max()
    if vmax > 0:
        max_response /= vmax

    logger.debug(f"Frangi 滤波完成, sigmas={sigmas}, max_resp={vmax:.4f}")
    return max_response


def tophat_filter(
    image: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """白顶帽变换 — 提取亮于背景的小结构.

    Args:
        image: 灰度图 (H, W), uint8.
        kernel_size: 结构元素大小.

    Returns:
        顶帽变换结果 (H, W), uint8.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    logger.debug(f"顶帽变换完成, kernel_size={kernel_size}")
    return result


def gabor_response(
    image: np.ndarray,
    n_orientations: int = 8,
    frequency: float = 0.1,
    sigma: float = 3.0,
) -> np.ndarray:
    """多方向 Gabor 纹理响应.

    Args:
        image: 灰度图 (H, W), uint8 或 float.
        n_orientations: 方向数.
        frequency: 空间频率.
        sigma: Gaussian 包络宽度.

    Returns:
        最大 Gabor 响应图 (H, W), float64.
    """
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    max_resp = np.zeros((h, w), dtype=np.float64)

    ksize = int(6 * sigma + 1) | 1  # 确保奇数
    wavelength = 1.0 / frequency

    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=wavelength,
            gamma=0.5,
            psi=0,
        )
        response = cv2.filter2D(img_f, cv2.CV_64F, kernel)
        max_resp = np.maximum(max_resp, np.abs(response))

    # 归一化
    rmax = max_resp.max()
    if rmax > 0:
        max_resp /= rmax

    logger.debug(f"Gabor 滤波完成, n_orientations={n_orientations}")
    return max_resp


def brightness_channel(
    image: np.ndarray,
    roi_mask: np.ndarray | None = None,
    percentile: float = 95.0,
) -> np.ndarray:
    """直接亮度检测通道 — 捕获 Frangi/TopHat 遗漏的大面积高亮缺陷.

    Frangi 仅对管状结构有响应，TopHat 受核大小限制。
    本通道直接用亮度阈值检出实心高亮区域，与线状滤波器互补。

    Args:
        image: 灰度图 (H, W), uint8.
        roi_mask: ROI 掩码.
        percentile: ROI 内亮度百分位数，作为归一化上限.

    Returns:
        亮度响应图 (H, W), float64, 值域 [0, 1].
    """
    img_f = image.astype(np.float64)

    # 以 ROI 内像素分布确定背景上限
    if roi_mask is not None:
        roi_pixels = img_f[roi_mask]
    else:
        roi_pixels = img_f.ravel()
    if len(roi_pixels) == 0:
        return np.zeros_like(img_f)

    # 暗场预处理图: 背景接近全黑 (median≈0)，不能用 median 作基准
    # 用 p90 作为"背景上限"—— 只有超过 90% 像素亮度的才算高亮缺陷起点
    bg_floor = float(np.percentile(roi_pixels, 90))
    bg_floor = max(bg_floor, 3.0)  # 至少灰度 3（避免全黑图除零）

    # 归一化上限用高百分位数
    above = roi_pixels[roi_pixels > bg_floor]
    if len(above) == 0:
        return np.zeros_like(img_f)
    p_hi = float(np.percentile(above, percentile))
    if p_hi <= bg_floor + 1:
        return np.zeros_like(img_f)

    resp = (img_f - bg_floor) / (p_hi - bg_floor)
    resp = np.clip(resp, 0.0, 1.0)

    if roi_mask is not None:
        resp[~roi_mask] = 0

    return resp


def compute_candidate_map(
    image: np.ndarray,
    params: DetectionParams | None = None,
    roi_mask: np.ndarray | None = None,
    original_image: np.ndarray | None = None,
) -> np.ndarray:
    """融合多特征生成高召回候选概率图.

    三通道融合策略：
      - Frangi: 管状/线状结构检测（细划痕）— 在增强图上运行
      - TopHat: 小尺度亮结构检测（斑点、小颗粒）— 在增强图上运行
      - Brightness: 直接亮度检测（大面积高亮缺陷）— 在原始图上运行

    Brightness 通道必须在原始图上运行，否则增强后大量像素被抬高，
    导致密度检测误将正常区域标记为 crash。

    最终候选图 = max(线状通道, 亮度通道)，确保两类缺陷都不遗漏。

    Args:
        image: 预处理后灰度图 (H, W), uint8（可能已增强）.
        params: 检测参数.
        roi_mask: ROI 掩码.
        original_image: 原始未增强图像，用于 brightness_channel.
                        若为 None 则使用 image.

    Returns:
        候选概率图 (H, W), float64, 值域 [0, 1].
    """
    if params is None:
        params = DetectionParams()

    # 1. Frangi 线结构增强 — 管状/线状缺陷（在增强图上运行效果更好）
    frangi_resp = frangi_filter(
        image,
        sigmas=params.frangi_sigmas,
        alpha=params.frangi_alpha,
        beta=params.frangi_beta,
        gamma=params.frangi_gamma,
    )

    # 2. 顶帽变换 — 小尺度亮结构（在增强图上运行）
    tophat_resp = tophat_filter(image, params.tophat_kernel_size)
    tophat_norm = tophat_resp.astype(np.float64)
    tmax = tophat_norm.max()
    if tmax > 0:
        tophat_norm /= tmax

    # 3. 线状通道融合 (Frangi + TopHat)
    line_channel = 0.7 * frangi_resp + 0.3 * tophat_norm

    # 4. 直接亮度通道 — 在原始图上运行，避免增强噪声放大
    bright_img = original_image if original_image is not None else image
    bright_resp = brightness_channel(bright_img, roi_mask)

    # 5. 取逐像素最大值合并（确保两类缺陷都不遗漏）
    candidate = np.maximum(line_channel, bright_resp)

    # 6. ROI 蒙版
    if roi_mask is not None:
        candidate[~roi_mask] = 0

    n_line = int(np.count_nonzero(line_channel > 0.1))
    n_bright = int(np.count_nonzero(bright_resp > 0.1))
    logger.info(
        f"候选图生成完成: line_ch={n_line}, bright_ch={n_bright}, "
        f"max={candidate.max():.3f}"
    )
    return candidate
