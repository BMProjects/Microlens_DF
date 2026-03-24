"""验证 roi_mask 从检测到评分的完整数据流传递."""

from __future__ import annotations

import numpy as np
import pytest

from darkfield_defects.detection.base import DefectInstance, DefectType, DetectionResult
from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.params import DetectionParams, PreprocessParams, ScoringParams
from darkfield_defects.scoring.quantify import compute_wear_metrics


class TestRoiMaskInMetadata:
    """Bug 1: ClassicalDetector 应将 roi_mask 存入 metadata."""

    @pytest.fixture()
    def synthetic_dark_image(self):
        """合成暗场图像：暗背景 + 圆形亮区 + 亮线."""
        rng = np.random.default_rng(42)
        img = rng.integers(5, 20, size=(256, 256), dtype=np.uint8)

        # 添加镜片圆形区域（让 ROI 提取能找到圆）
        cy, cx, r = 128, 128, 100
        for y in range(256):
            for x in range(256):
                if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                    img[y, x] = 40

        # 水平亮线（模拟划痕）
        img[120:124, 50:200] = 200
        return img

    @pytest.fixture()
    def low_ratio_params(self):
        """降低 min_radius_ratio 避免小图 ROI 提取失败."""
        return PreprocessParams(roi_min_radius_ratio=0.15)

    def test_metadata_contains_roi_mask(self, synthetic_dark_image, low_ratio_params):
        """检测结果的 metadata 应包含 roi_mask."""
        detector = ClassicalDetector(DetectionParams(), preprocess_params=low_ratio_params)
        result = detector.detect(synthetic_dark_image)

        assert "roi_mask" in result.metadata
        roi_mask = result.metadata["roi_mask"]
        assert isinstance(roi_mask, np.ndarray)
        assert roi_mask.shape == synthetic_dark_image.shape[:2]

    def test_metadata_roi_mask_not_none(self, synthetic_dark_image, low_ratio_params):
        """roi_mask 不应为 None."""
        detector = ClassicalDetector(DetectionParams(), preprocess_params=low_ratio_params)
        result = detector.detect(synthetic_dark_image)
        assert result.metadata["roi_mask"] is not None


class TestWearMetricsDensity:
    """验证有 roi_mask 时 D_density 非零."""

    def test_density_nonzero_with_roi(self):
        """当有划痕且有 roi_mask 时，D_density 应 > 0."""
        inst = DefectInstance(
            instance_id=0,
            defect_type=DefectType.SCRATCH,
            length_px=100.0,
            area_px=200,
            scatter_intensity=50.0,
            zone="center",
        )
        result = DetectionResult(
            mask=np.zeros((128, 128), dtype=np.uint8),
            instances=[inst],
        )

        roi_mask = np.zeros((128, 128), dtype=bool)
        roi_mask[10:120, 10:120] = True

        metrics = compute_wear_metrics(result, roi_mask)
        assert metrics.roi_area > 0
        assert metrics.D_density > 0

    def test_density_zero_without_roi(self):
        """当 roi_mask 为 None 时，D_density 应为 0."""
        inst = DefectInstance(
            instance_id=0,
            defect_type=DefectType.SCRATCH,
            length_px=100.0,
            area_px=200,
            scatter_intensity=50.0,
            zone="edge",
        )
        result = DetectionResult(
            mask=np.zeros((128, 128), dtype=np.uint8),
            instances=[inst],
        )

        metrics = compute_wear_metrics(result, roi_mask=None)
        assert metrics.roi_area == 0
        assert metrics.D_density == 0.0


class TestZoneClassification:
    """Bug 3: 视区比例应来自 ScoringParams."""

    def test_custom_zone_ratios_accepted(self):
        """自定义比例参数应被接受且不报错."""
        # 较大的合成镜片图像，有明确的圆形亮区
        img = np.zeros((256, 256), dtype=np.uint8)
        cy, cx, r = 128, 128, 100
        for y in range(256):
            for x in range(256):
                if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                    img[y, x] = 40
        img[128, 80:176] = 200  # 中心水平亮线

        custom_scoring = ScoringParams(
            zone_center_ratio=0.50,
            zone_transition_ratio=0.80,
        )
        low_ratio_params = PreprocessParams(roi_min_radius_ratio=0.15)
        detector = ClassicalDetector(
            DetectionParams(),
            preprocess_params=low_ratio_params,
            scoring_params=custom_scoring,
        )
        result = detector.detect(img)
        assert result is not None
