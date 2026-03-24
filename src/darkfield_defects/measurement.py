"""物理量标定与单位换算工具."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_PIXEL_SIZE_UM = 6.8
DEFAULT_PIXEL_SIZE_MM = DEFAULT_PIXEL_SIZE_UM / 1000.0


@dataclass(frozen=True)
class SpatialCalibration:
    """空间标定参数.

    默认使用当前项目确认的图像比例尺:
    6.8 um / pixel = 0.0068 mm / pixel
    """

    pixel_size_mm: float = DEFAULT_PIXEL_SIZE_MM
    unit: str = "mm"

    @property
    def pixel_size_um(self) -> float:
        return self.pixel_size_mm * 1000.0

    @property
    def px_per_mm(self) -> float:
        return 1.0 / max(self.pixel_size_mm, 1e-12)

    def length_px_to_mm(self, value_px: float) -> float:
        return float(value_px) * self.pixel_size_mm

    def area_px_to_mm2(self, value_px2: float) -> float:
        return float(value_px2) * (self.pixel_size_mm ** 2)

    def bbox_px_to_mm(self, bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x, y, w, h = bbox
        s = self.pixel_size_mm
        return x * s, y * s, w * s, h * s

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_size_mm": self.pixel_size_mm,
            "pixel_size_um": self.pixel_size_um,
            "px_per_mm": self.px_per_mm,
            "unit": self.unit,
        }


DEFAULT_CALIBRATION = SpatialCalibration()


def get_calibration(metadata: dict[str, Any] | None = None) -> SpatialCalibration:
    """从结果元数据中解析标定信息，缺省为系统默认值."""
    if not metadata:
        return DEFAULT_CALIBRATION
    value = metadata.get("pixel_size_mm")
    if value is None:
        return DEFAULT_CALIBRATION
    return SpatialCalibration(pixel_size_mm=float(value))
