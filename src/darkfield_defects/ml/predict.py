"""推理模块 — 全图滑窗分割 + DetectionResult 输出."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required: pip install darkfield-defects[ml]")

from darkfield_defects.detection.base import DefectInstance, DefectType, DetectionResult
from darkfield_defects.logging import get_logger
from darkfield_defects.measurement import DEFAULT_CALIBRATION
from darkfield_defects.ml.segmentation_factory import build_segmentation_model, spec_from_checkpoint

logger = get_logger(__name__)

# 像素值 → DefectType 映射
PIXEL_TO_DEFECT: dict[int, DefectType] = {
    1: DefectType.SCRATCH,
    2: DefectType.SPOT,
    3: DefectType.DAMAGE,
}


def load_model(
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """加载训练好的模型.

    Args:
        checkpoint_path: 权重文件路径 (.pt).
        device: 推理设备.

    Returns:
        (model, device) 元组.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)

    model_spec = spec_from_checkpoint(ckpt)
    model = build_segmentation_model(model_spec)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    logger.info(
        f"模型加载完成: model={model_spec.model_name}, in_ch={model_spec.in_channels}, "
        f"classes={model_spec.num_classes}, device={dev}"
    )
    return model, dev


def predict_full_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 512,
    overlap: int = 64,
    candidate_map: np.ndarray | None = None,
) -> np.ndarray:
    """全图滑窗推理, 输出多类分割掩码.

    使用 overlap stitching 避免块边界伪影.

    Args:
        model: 训练好的模型.
        image: 灰度图 (H, W), uint8.
        device: 推理设备.
        patch_size: 推理 patch 大小.
        overlap: 块间重叠像素.
        candidate_map: 可选候选概率图 (H, W), float [0,1].

    Returns:
        分割掩码 (H, W), 像素值=类别 ID (0-3).
    """
    h, w = image.shape[:2]
    stride = patch_size - overlap

    # 预分配输出
    pred_count = np.zeros((h, w), dtype=np.float32)
    pred_sum = np.zeros((4, h, w), dtype=np.float32)  # 4 类

    # 归一化输入
    img_f = image.astype(np.float32) / 255.0

    # 滑窗
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y, h - patch_size)
            x1 = min(x, w - patch_size)
            y2 = y1 + patch_size
            x2 = x1 + patch_size

            # 边界检查
            if y2 > h or x2 > w:
                continue

            patch = img_f[y1:y2, x1:x2]
            channels = [patch]

            if candidate_map is not None:
                cand_patch = candidate_map[y1:y2, x1:x2].astype(np.float32)
                channels.append(cand_patch)

            input_tensor = np.stack(channels, axis=0)  # (C, H, W)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)  # (1, C, H, W)

            with torch.no_grad():
                logits = model(input_tensor)  # (1, 4, H, W)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4, H, W)

            pred_sum[:, y1:y2, x1:x2] += probs
            pred_count[y1:y2, x1:x2] += 1

    # 平均重叠区域
    pred_count = np.maximum(pred_count, 1)
    for c in range(4):
        pred_sum[c] /= pred_count

    # 取 argmax
    result = pred_sum.argmax(axis=0).astype(np.uint8)

    logger.info(
        f"全图推理完成: {h}x{w}, "
        f"scratch={np.count_nonzero(result == 1)}, "
        f"spot={np.count_nonzero(result == 2)}, "
        f"damage={np.count_nonzero(result == 3)}"
    )
    return result


def mask_to_detection_result(
    segmentation_mask: np.ndarray,
    image: np.ndarray,
    optical_center: tuple[int, int] = (0, 0),
    lens_radius: float = 0.0,
    min_area: int = 30,
) -> DetectionResult:
    """将多类分割掩码转换为 DetectionResult.

    Args:
        segmentation_mask: (H, W) 像素值=类别 ID.
        image: 原始灰度图 (用于散射强度计算).
        optical_center: 光学中心.
        lens_radius: 镜片半径.
        min_area: 最小面积阈值.

    Returns:
        DetectionResult 包含多类 DefectInstance.
    """
    instances: list[DefectInstance] = []
    full_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)

    for pixel_val, defect_type in PIXEL_TO_DEFECT.items():
        class_mask = (segmentation_mask == pixel_val).astype(np.uint8)

        if not np.any(class_mask):
            continue

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            inst_mask = (labels == label_id)

            # Bounding box
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            # 散射强度
            scatter = float(np.mean(image[inst_mask])) if np.any(inst_mask) else 0.0

            # 骨架化 (仅 scratch)
            skel_coords = np.empty((0, 2))
            length = 0.0
            endpoints = None

            if defect_type == DefectType.SCRATCH:
                skel = cv2.ximgproc.thinning(inst_mask.astype(np.uint8) * 255) \
                    if hasattr(cv2, 'ximgproc') else _thin_fallback(inst_mask)
                skel_coords = np.argwhere(skel > 0)
                length = float(len(skel_coords))

                if len(skel_coords) >= 2:
                    dists = np.linalg.norm(skel_coords - skel_coords[0], axis=1)
                    far_idx = int(np.argmax(dists))
                    endpoints = (
                        tuple(skel_coords[0].tolist()),
                        tuple(skel_coords[far_idx].tolist()),
                    )

            # 圆度
            contours, _ = cv2.findContours(
                inst_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            circularity = 0.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                perim = cv2.arcLength(largest, True)
                if perim > 0:
                    circularity = 4 * np.pi * area / (perim * perim)

            avg_width = area / max(length, 1.0) if length > 0 else float(np.sqrt(area))
            aspect = length / max(avg_width, 1.0) if length > 0 else 1.0

            # 视区
            zone = _classify_zone(
                skel_coords if len(skel_coords) > 0 else np.argwhere(inst_mask),
                optical_center, lens_radius,
            )

            inst = DefectInstance(
                instance_id=len(instances),
                defect_type=defect_type,
                skeleton_coords=skel_coords,
                mask=inst_mask,
                length_px=length,
                area_px=area,
                avg_width_px=avg_width,
                bbox=(x, y, w, h),
                scatter_intensity=scatter,
                zone=zone,
                endpoints=endpoints,
                circularity=circularity,
                aspect_ratio=aspect,
                pixel_size_mm=DEFAULT_CALIBRATION.pixel_size_mm,
            )
            instances.append(inst)

            full_mask[inst_mask] = 255

    return DetectionResult(
        mask=full_mask,
        instances=instances,
        metadata={
            "detector": "ml",
            "optical_center": optical_center,
            "lens_radius": lens_radius,
            "pixel_size_mm": DEFAULT_CALIBRATION.pixel_size_mm,
        },
    )


def _thin_fallback(binary: np.ndarray) -> np.ndarray:
    """骨架化回退."""
    from skimage.morphology import skeletonize
    skel = skeletonize(binary > 0)
    return (skel.astype(np.uint8) * 255)


def _classify_zone(
    coords: np.ndarray,
    center: tuple[int, int],
    radius: float,
    center_ratio: float = 0.30,
    transition_ratio: float = 0.60,
) -> str:
    """判断缺陷所属视区."""
    if len(coords) == 0 or radius <= 0:
        return "edge"
    cy, cx = center
    dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
    mean_dist = float(np.mean(dists))
    if mean_dist <= radius * center_ratio:
        return "center"
    elif mean_dist <= radius * transition_ratio:
        return "microstructure"
    else:
        return "edge"
