"""检测器抽象接口与数据结构."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from darkfield_defects.measurement import DEFAULT_CALIBRATION, get_calibration


class DefectType(str, Enum):
    """缺陷类别枚举.

    内部分类（4类）保持完整语义，用于磨损评分的物理量化。
    YOLO 训练标签（3类）将 DAMAGE 与 CRASH 合并为 "critical"，
    通过 yolo_class_id() / yolo_class_name() 方法获取。
    """
    SCRATCH = "scratch"      # 划痕 — 细长线状散射
    SPOT = "spot"            # 斑点/凹坑 — 近圆形点状散射
    DAMAGE = "damage"        # 大面积缺损 — 不规则连片高亮区域（孤立大面积）
    CRASH = "crash"          # 密集缺陷区 — 高密度缺陷聚集（多缺陷堆叠）

    # YOLO 3-class 映射
    # scratch → 0, spot → 1, damage/crash → 2 (critical)
    _YOLO_CLASS_MAP: dict = {}  # 延迟初始化

    def yolo_class_id(self) -> int:
        """返回 YOLO 训练使用的 3 类别索引.

        DAMAGE 和 CRASH 合并为同一类 (2, "critical")。
        原因：两者在 tile 级别视觉上均表现为"大片高亮区域"，
        且 DAMAGE 仅 46 实例（0.045%）无法独立学习。
        内部评分逻辑仍保留完整 4 类区分。
        """
        return {
            DefectType.SCRATCH: 0,
            DefectType.SPOT:    1,
            DefectType.DAMAGE:  2,
            DefectType.CRASH:   2,
        }[self]

    def yolo_class_name(self) -> str:
        """返回 YOLO 3 类别名称."""
        return ["scratch", "spot", "critical"][self.yolo_class_id()]


@dataclass
class DefectInstance:
    """单个缺陷实例（泛化版本）."""
    instance_id: int
    defect_type: DefectType = DefectType.SCRATCH
    skeleton_coords: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=bool))
    length_px: float = 0.0             # 骨架长度（像素）
    area_px: int = 0                   # 掩码面积（像素）
    avg_width_px: float = 0.0          # 平均宽度（像素）
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    scatter_intensity: float = 0.0     # 散射强度代理值
    prominence: float = 0.0           # 显著性（中心与角落灰度差）
    zone: str = "edge"                 # 所属视区: "center", "microstructure", "edge"
    endpoints: tuple[tuple[int, int], tuple[int, int]] | None = None
    circularity: float = 0.0          # 圆度 (4π·area/perimeter²)
    aspect_ratio: float = 0.0         # 细长比 (length/width)
    pixel_size_mm: float = DEFAULT_CALIBRATION.pixel_size_mm

    @property
    def length_mm(self) -> float:
        return self.length_px * self.pixel_size_mm

    @property
    def area_mm2(self) -> float:
        return self.area_px * (self.pixel_size_mm ** 2)

    @property
    def avg_width_mm(self) -> float:
        return self.avg_width_px * self.pixel_size_mm

    @property
    def bbox_mm(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        scale = self.pixel_size_mm
        return x * scale, y * scale, w * scale, h * scale


# 向后兼容别名
ScratchInstance = DefectInstance


@dataclass
class DetectionResult:
    """检测结果容器."""
    mask: np.ndarray                                # 全图二值掩码 (H, W)
    instances: list[DefectInstance] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)  # COCO 格式
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_scratches(self) -> int:
        return sum(1 for i in self.instances if i.defect_type == DefectType.SCRATCH)

    @property
    def num_spots(self) -> int:
        return sum(1 for i in self.instances if i.defect_type == DefectType.SPOT)

    @property
    def num_damages(self) -> int:
        return sum(1 for i in self.instances if i.defect_type == DefectType.DAMAGE)

    @property
    def num_crashes(self) -> int:
        return sum(1 for i in self.instances if i.defect_type == DefectType.CRASH)

    @property
    def num_defects(self) -> int:
        return len(self.instances)

    @property
    def total_length(self) -> float:
        return sum(s.length_px for s in self.instances)

    @property
    def total_area(self) -> int:
        return sum(s.area_px for s in self.instances)

    @property
    def calibration(self):
        return get_calibration(self.metadata)

    @property
    def total_length_mm(self) -> float:
        return self.calibration.length_px_to_mm(self.total_length)

    @property
    def total_area_mm2(self) -> float:
        return self.calibration.area_px_to_mm2(self.total_area)

    def get_by_type(self, defect_type: DefectType) -> list[DefectInstance]:
        """按类型过滤缺陷实例."""
        return [i for i in self.instances if i.defect_type == defect_type]


class BaseDetector(ABC):
    """检测器抽象基类.

    所有检测器（经典/ML）必须实现 detect() 方法，接受单帧图像，返回 DetectionResult.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        background: np.ndarray | None = None,
        roi_mask: np.ndarray | None = None,
    ) -> DetectionResult:
        """执行缺陷检测.

        Args:
            image: 输入图像 (H, W) 灰度图.
            background: 可选背景图用于平场校正.
            roi_mask: 可选 ROI 掩码.

        Returns:
            DetectionResult 包含掩码、实例列表和标注.
        """
        ...
