"""图像配准 — 高光ring模板匹配.

使用离焦背景的高光ring区域作为模板，通过相位相关粗定位 + ECC仿射细配
实现目标图像到参考坐标系的配准，允许平移、旋转和有限缩放。
"""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def apply_warp(
    image: np.ndarray,
    warp_matrix: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    border_value: int = 0,
) -> np.ndarray:
    """应用仿射变换.

    Args:
        image: 输入图像.
        warp_matrix: 2×3 变换矩阵.
        output_shape: (H, W) 输出尺寸; None 则保持原尺寸.
        border_value: 边界填充值.

    Returns:
        变换后图像.
    """
    if output_shape is None:
        output_shape = image.shape[:2]

    h, w = output_shape
    warp_f32 = warp_matrix.astype(np.float32)

    if warp_f32.shape != (2, 3):
        raise ValueError(f"不支持的变换矩阵形状: {warp_f32.shape}")

    return cv2.warpAffine(
        image, warp_f32, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _normalize_downsample(downsample: float) -> float:
    """规范化降采样比例到 (0, 1]."""
    try:
        s = float(downsample)
    except (TypeError, ValueError):
        return 1.0
    if s <= 0.0 or s > 1.0:
        return 1.0
    return s


def _downsample_inputs(
    src: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray | None,
    downsample: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """按比例降采样配准输入."""
    scale = _normalize_downsample(downsample)
    if scale >= 0.999:
        return src, template, mask, 1.0

    h, w = template.shape[:2]
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    size = (new_w, new_h)

    src_ds = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    tpl_ds = cv2.resize(template, size, interpolation=cv2.INTER_AREA)

    mask_ds = None
    if mask is not None:
        mask_ds = cv2.resize(
            mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST,
        )
        mask_ds = (mask_ds > 0).astype(np.uint8)

    return src_ds, tpl_ds, mask_ds, scale


def _rescale_warp(warp: np.ndarray, scale: float) -> np.ndarray:
    """将降采样坐标系的变换矩阵恢复到原图坐标系."""
    out = warp.astype(np.float64).copy()
    if scale < 0.999:
        out[0, 2] /= scale
        out[1, 2] /= scale
    return out


def _invert_affine_2x3(warp: np.ndarray) -> np.ndarray:
    """求 2x3 仿射矩阵的逆."""
    m = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(m)
    return inv[:2, :]


def _warp_geometry(warp: np.ndarray) -> tuple[float, float, float, float]:
    """从仿射矩阵提取 (dx, dy, rotation_deg, scale).

    Returns:
        (dx, dy, angle_degrees, scale)
    """
    a, b = float(warp[0, 0]), float(warp[0, 1])
    c, d = float(warp[1, 0]), float(warp[1, 1])
    dx = float(warp[0, 2])
    dy = float(warp[1, 2])
    angle = float(np.degrees(np.arctan2(c, a)))
    sx = float(np.hypot(a, c))
    sy = float(np.hypot(b, d))
    scale = 0.5 * (sx + sy)
    return dx, dy, angle, scale


def _estimate_translation(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """相位相关估计平移量.

    Returns:
        (dx, dy, response)
    """
    s = src.astype(np.float64)
    d = dst.astype(np.float64)

    window = None
    if mask is not None:
        m = mask.astype(bool)
        if np.any(m):
            s_mean, s_std = float(np.mean(s[m])), float(np.std(s[m]))
            d_mean, d_std = float(np.mean(d[m])), float(np.std(d[m]))
            s_std = s_std if s_std > 1e-6 else 1.0
            d_std = d_std if d_std > 1e-6 else 1.0
            s = (s - s_mean) / s_std
            d = (d - d_mean) / d_std
        m_f = m.astype(np.float64)
        s = s * m_f
        d = d * m_f
        window = m_f

    if window is not None:
        shift, response = cv2.phaseCorrelate(s, d, window)
    else:
        shift, response = cv2.phaseCorrelate(s, d)

    dx, dy = shift
    return float(dx), float(dy), float(response)


def _refine_ecc(
    src: np.ndarray,
    dst: np.ndarray,
    warp_init: np.ndarray,
    mask: np.ndarray | None = None,
    max_iter: int = 200,
    epsilon: float = 1e-6,
    motion_model: str = "affine",
) -> tuple[np.ndarray, float]:
    """ECC 细配准.

    Returns:
        (warp_matrix, ecc_value)
    """
    model_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    motion = model_map.get(motion_model, cv2.MOTION_AFFINE)

    warp = warp_init.astype(np.float32)
    warp_safe = warp.copy()

    s = src.astype(np.float32)
    d = dst.astype(np.float32)

    # 归一化到 [0, 1]
    s_max = s.max()
    d_max = d.max()
    if s_max > 0:
        s = s / s_max
    if d_max > 0:
        d = d / d_max

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)

    input_mask = None
    if mask is not None:
        input_mask = mask.astype(np.uint8)
        if input_mask.max() == 1:
            input_mask = input_mask * 255

    try:
        ecc_val, warp_out = cv2.findTransformECC(
            s, d, warp, motion, criteria, input_mask, 5,
        )
    except cv2.error as e:
        logger.warning("ECC 配准失败: %s, 返回初始矩阵", e)
        return warp_safe.astype(np.float64), 0.0

    return warp_out.astype(np.float64), float(ecc_val)


def register_to_template(
    src: np.ndarray,
    ring_template: np.ndarray,
    ring_mask: np.ndarray,
    max_scale_deviation: float = 0.05,
    ecc_max_iter: int = 200,
    ecc_epsilon: float = 1e-6,
    downsample: float = 1.0,
    pre_blur_sigma: float = 15.0,
    ecc_score_threshold: float = 0.75,
) -> tuple[np.ndarray, float]:
    """高光ring模板匹配配准.

    流程:
      1) 对 src/template 预模糊（σ=pre_blur_sigma）以平滑细节干扰
      2) 扩张 ring_mask 构建 wide_band（±100px），作为相位相关和ECC的引导区域
      3) 相位相关粗估平移初值
      4) ECC仿射细配（wide_band + 预模糊）
      5) 若 ECC 分数 < ecc_score_threshold，级联 no_mask 从 identity 初始化再试
      6) 缩放幅度检查（超过 max_scale_deviation 则回退到 euclidean）
      7) 返回 src→ref 的仿射矩阵

    Args:
        src: 待配准目标图像 (H, W), uint8.
        ring_template: 离焦背景模板 B_blur (H, W).
        ring_mask: 高光ring区域掩码 (H, W), bool/uint8.
        max_scale_deviation: 最大允许缩放偏差, 默认0.05即±5%.
        ecc_max_iter: ECC最大迭代次数.
        ecc_epsilon: ECC收敛阈值.
        downsample: 降采样比例 (0, 1].
        pre_blur_sigma: ECC前对src/template预模糊的σ; 0禁用.
        ecc_score_threshold: ECC分数低于此值时触发 no_mask 级联回退.

    Returns:
        (warp_matrix, score): warp_matrix 是 src→ref 的 2×3 仿射矩阵,
        score 是配准质量分数.
    """
    # ① 降采样
    src_ds, tpl_ds, mask_ds, scale = _downsample_inputs(
        src, ring_template, ring_mask, downsample,
    )

    # ② 预模糊（平滑缺陷/划痕干扰，提升ECC收敛稳定性）
    if pre_blur_sigma > 0.0:
        ks = int(pre_blur_sigma * 6 + 1) | 1  # 保证奇数
        s_blur = cv2.GaussianBlur(
            src_ds.astype(np.float32), (ks, ks), pre_blur_sigma,
        )
        t_blur = cv2.GaussianBlur(
            tpl_ds.astype(np.float32), (ks, ks), pre_blur_sigma,
        )
    else:
        s_blur = src_ds.astype(np.float32)
        t_blur = tpl_ds.astype(np.float32)

    # ③ 构建 wide_band：ring_mask 扩张 ~100px（全分辨率等效）
    if mask_ds is not None:
        dil_px = max(5, int(round(100.0 * scale)))
        ks_dil = 2 * dil_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_dil, ks_dil))
        wide_band = cv2.dilate(mask_ds.astype(np.uint8), kernel).astype(np.uint8)
    else:
        wide_band = None

    # ④ 相位相关粗估平移（使用 wide_band + 预模糊图）
    dx_coarse, dy_coarse, phase_response = _estimate_translation(
        t_blur.astype(np.float64),
        s_blur.astype(np.float64),
        wide_band,
    )
    logger.debug("相位相关粗估: dx=%.2f dy=%.2f response=%.4f", dx_coarse, dy_coarse, phase_response)

    warp_init = np.eye(2, 3, dtype=np.float64)
    warp_init[0, 2] = dx_coarse
    warp_init[1, 2] = dy_coarse

    # ⑤ ECC仿射细配 — wide_band + 预模糊
    warp1, ecc1 = _refine_ecc(
        t_blur, s_blur,
        warp_init=warp_init,
        mask=wide_band,
        max_iter=ecc_max_iter,
        epsilon=ecc_epsilon,
        motion_model="affine",
    )
    best_warp, best_score, best_label = warp1, ecc1, "wide_band"
    logger.debug("ECC wide_band: score=%.4f", ecc1)

    # ⑥ 级联回退 — no_mask + 预模糊（从 identity 初始化，避免错误先验传播）
    if ecc1 < ecc_score_threshold:
        warp2, ecc2 = _refine_ecc(
            t_blur, s_blur,
            warp_init=np.eye(2, 3, dtype=np.float64),
            mask=None,
            max_iter=ecc_max_iter,
            epsilon=ecc_epsilon,
            motion_model="affine",
        )
        logger.debug("ECC no_mask: score=%.4f", ecc2)
        if ecc2 > best_score:
            best_warp, best_score, best_label = warp2, ecc2, "no_mask"

    warp_bg2src_ds = best_warp
    ecc_affine = best_score

    # ⑦ 缩放幅度检查，超限则回退到 euclidean
    _, _, angle_affine, scale_affine = _warp_geometry(warp_bg2src_ds)
    scale_dev = abs(scale_affine - 1.0)

    if scale_dev > max_scale_deviation:
        logger.warning(
            "仿射缩放 %.4f 超限(±%.1f%%), 回退到 euclidean",
            scale_affine, max_scale_deviation * 100,
        )
        warp_eu, ecc_eu = _refine_ecc(
            t_blur, s_blur,
            warp_init=warp_init,
            mask=wide_band,
            max_iter=ecc_max_iter,
            epsilon=ecc_epsilon,
            motion_model="euclidean",
        )
        score = ecc_eu if ecc_eu > 0 else phase_response
        _, _, angle_final, scale_final = _warp_geometry(warp_eu)
        logger.info(
            "euclidean回退: angle=%.4f° scale=%.6f score=%.6f",
            angle_final, scale_final, score,
        )
        warp_bg2src_ds = warp_eu
    else:
        score = ecc_affine if ecc_affine > 0 else phase_response
        logger.info(
            "仿射配准[%s]: angle=%.4f° scale=%.6f score=%.6f",
            best_label, angle_affine, scale_affine, score,
        )

    # ⑧ 恢复到全分辨率，转换为 src→ref（取逆）
    warp_bg2src_full = _rescale_warp(warp_bg2src_ds, scale)
    warp_src2ref = _invert_affine_2x3(warp_bg2src_full)

    dx, dy, angle, s = _warp_geometry(warp_src2ref)
    logger.info(
        "最终 src→ref: dx=%.2f dy=%.2f angle=%.4f° scale=%.6f",
        dx, dy, angle, s,
    )

    return warp_src2ref, score
