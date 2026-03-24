"""磨损量化模块 — 统一评分指标计算."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from darkfield_defects.detection.base import DefectType, DetectionResult
from darkfield_defects.logging import get_logger
from darkfield_defects.measurement import get_calibration

logger = get_logger(__name__)


@dataclass
class WearMetrics:
    """磨损量化指标.

    统一使用三分区视角：
    - center: 主视觉中心区
    - microstructure: 离焦微结构关键功能区
    - edge: 镜片边缘区

    为保持向后兼容，`L_transition` 作为 `L_microstructure` 的别名保留。
    """

    L_total: float = 0.0
    L_center: float = 0.0
    L_microstructure: float = 0.0
    L_edge: float = 0.0

    N_total: int = 0
    N_center: int = 0
    N_microstructure: int = 0
    N_edge: int = 0

    A_total: int = 0
    A_center: int = 0
    A_microstructure: int = 0
    A_edge: int = 0

    N_critical: int = 0
    A_critical: int = 0

    D_density: float = 0.0
    D_micro_density: float = 0.0

    S_scatter: float = 0.0
    S_scatter_center: float = 0.0
    S_scatter_micro: float = 0.0
    S_scatter_edge: float = 0.0

    n_crossings: int = 0
    roi_area: int = 0
    microstructure_area: int = 0
    pixel_size_mm: float = 0.0

    @property
    def L_transition(self) -> float:
        return self.L_microstructure

    @property
    def A_transition(self) -> int:
        return self.A_microstructure

    def to_dict(self) -> dict[str, Any]:
        length_scale = self.pixel_size_mm
        area_scale = self.pixel_size_mm ** 2
        return {
            "unit_length": "mm",
            "unit_area": "mm^2",
            "pixel_size_mm": round(self.pixel_size_mm, 6),
            "L_total": round(self.L_total * length_scale, 4),
            "L_center": round(self.L_center * length_scale, 4),
            "L_microstructure": round(self.L_microstructure * length_scale, 4),
            "L_transition": round(self.L_microstructure * length_scale, 4),
            "L_edge": round(self.L_edge * length_scale, 4),
            "N_total": self.N_total,
            "N_center": self.N_center,
            "N_microstructure": self.N_microstructure,
            "N_edge": self.N_edge,
            "A_total": round(self.A_total * area_scale, 6),
            "A_center": round(self.A_center * area_scale, 6),
            "A_microstructure": round(self.A_microstructure * area_scale, 6),
            "A_transition": round(self.A_microstructure * area_scale, 6),
            "A_edge": round(self.A_edge * area_scale, 6),
            "N_critical": self.N_critical,
            "A_critical": round(self.A_critical * area_scale, 6),
            "D_density": round(self.D_density / max(length_scale, 1e-12), 6),
            "D_micro_density": round(self.D_micro_density / max(length_scale, 1e-12), 6),
            "S_scatter": round(self.S_scatter, 2),
            "S_scatter_center": round(self.S_scatter_center, 2),
            "S_scatter_micro": round(self.S_scatter_micro, 2),
            "S_scatter_edge": round(self.S_scatter_edge, 2),
            "n_crossings": self.n_crossings,
            "roi_area": round(self.roi_area * area_scale, 6),
            "microstructure_area": round(self.microstructure_area * area_scale, 6),
            "raw_px": {
                "L_total_px": round(self.L_total, 1),
                "L_center_px": round(self.L_center, 1),
                "L_microstructure_px": round(self.L_microstructure, 1),
                "L_edge_px": round(self.L_edge, 1),
                "A_total_px2": self.A_total,
                "A_center_px2": self.A_center,
                "A_microstructure_px2": self.A_microstructure,
                "A_edge_px2": self.A_edge,
                "roi_area_px2": self.roi_area,
                "microstructure_area_px2": self.microstructure_area,
            },
        }


def compute_wear_metrics(
    result: DetectionResult,
    roi_mask: np.ndarray | None = None,
    *,
    center_ratio: float = 0.40,
    microstructure_ratio: float = 0.75,
) -> WearMetrics:
    """从检测结果计算统一磨损指标."""
    metrics = WearMetrics()
    calibration = get_calibration(result.metadata)
    metrics.pixel_size_mm = calibration.pixel_size_mm

    if roi_mask is not None:
        metrics.roi_area = int(np.count_nonzero(roi_mask))
        annulus_ratio = max(microstructure_ratio**2 - center_ratio**2, 0.0)
        metrics.microstructure_area = int(metrics.roi_area * annulus_ratio)

    weighted_scatter_total = 0.0
    weighted_scatter_center = 0.0
    weighted_scatter_micro = 0.0
    weighted_scatter_edge = 0.0

    for inst in result.instances:
        zone = _normalize_zone(inst.zone)

        metrics.N_total += 1
        metrics.A_total += inst.area_px
        weighted_scatter_total += inst.scatter_intensity * inst.area_px

        if zone == "center":
            metrics.N_center += 1
            metrics.A_center += inst.area_px
            weighted_scatter_center += inst.scatter_intensity * inst.area_px
        elif zone == "microstructure":
            metrics.N_microstructure += 1
            metrics.A_microstructure += inst.area_px
            weighted_scatter_micro += inst.scatter_intensity * inst.area_px
        else:
            metrics.N_edge += 1
            metrics.A_edge += inst.area_px
            weighted_scatter_edge += inst.scatter_intensity * inst.area_px

        if inst.defect_type in (DefectType.DAMAGE, DefectType.CRASH):
            metrics.N_critical += 1
            metrics.A_critical += inst.area_px

        if inst.defect_type == DefectType.SCRATCH:
            metrics.L_total += inst.length_px
            if zone == "center":
                metrics.L_center += inst.length_px
            elif zone == "microstructure":
                metrics.L_microstructure += inst.length_px
            else:
                metrics.L_edge += inst.length_px

    if metrics.A_total > 0:
        metrics.S_scatter = weighted_scatter_total / metrics.A_total
    if metrics.A_center > 0:
        metrics.S_scatter_center = weighted_scatter_center / metrics.A_center
    if metrics.A_microstructure > 0:
        metrics.S_scatter_micro = weighted_scatter_micro / metrics.A_microstructure
    if metrics.A_edge > 0:
        metrics.S_scatter_edge = weighted_scatter_edge / metrics.A_edge

    if metrics.roi_area > 0:
        metrics.D_density = metrics.L_total / metrics.roi_area
    if metrics.microstructure_area > 0:
        metrics.D_micro_density = metrics.L_microstructure / metrics.microstructure_area

    metrics.n_crossings = _count_crossings(result)

    logger.info(
        "磨损指标: L_center=%.0f, L_micro=%.0f, N=%d, critical=%d, scatter=%.1f",
        metrics.L_center,
        metrics.L_microstructure,
        metrics.N_total,
        metrics.N_critical,
        metrics.S_scatter,
    )
    return metrics


def _normalize_zone(zone: str) -> str:
    if zone == "transition":
        return "microstructure"
    if zone in {"center", "microstructure", "edge"}:
        return zone
    return "edge"


def _count_crossings(result: DetectionResult) -> int:
    """估算骨架交叉点数量."""
    scratch_instances = [
        inst for inst in result.instances if inst.defect_type == DefectType.SCRATCH
    ]
    if len(scratch_instances) < 2:
        return 0

    from scipy.ndimage import convolve

    mask = result.mask > 0
    h, w = mask.shape
    skel_mask = np.zeros((h, w), dtype=np.uint8)
    for inst in scratch_instances:
        for r, c in inst.skeleton_coords:
            if 0 <= r < h and 0 <= c < w:
                skel_mask[r, c] = 1

    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbor_count = convolve(skel_mask.astype(int), kernel, mode="constant")
    return int(np.count_nonzero((skel_mask > 0) & (neighbor_count >= 3)))
