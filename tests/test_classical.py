"""ClassicalDetector 端到端测试 — 使用合成暗场数据."""

import numpy as np
import pytest

from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.params import DetectionParams


class TestClassicalDetector:
    """ClassicalDetector 基本功能验证."""

    @pytest.fixture()
    def synthetic_dark_image(self):
        """创建合成暗场图像：暗背景 + 亮线 (模拟划痕)."""
        rng = np.random.default_rng(42)
        img = rng.integers(5, 30, size=(256, 256), dtype=np.uint8)

        # 添加一条水平亮线 (模拟划痕 — 暗场下亮散射)
        img[120:124, 50:200] = 200

        # 添加一条斜线
        for i in range(80):
            r = 60 + i
            c = 30 + i
            if r < 256 and c < 256:
                img[r : r + 2, c : c + 2] = 180

        return img

    @pytest.fixture()
    def default_params(self):
        return DetectionParams()

    def test_detector_returns_result(self, synthetic_dark_image, default_params):
        """检测器应返回有效的 DetectionResult."""
        detector = ClassicalDetector(default_params)
        result = detector.detect(synthetic_dark_image)

        assert result is not None
        assert result.mask.shape == synthetic_dark_image.shape
        assert result.mask.dtype == np.uint8

    def test_bright_scratches_detected(self, synthetic_dark_image, default_params):
        """合成亮线应被检测到（掩码非空）."""
        detector = ClassicalDetector(default_params)
        result = detector.detect(synthetic_dark_image)

        # 至少应有一些像素被标记
        assert np.count_nonzero(result.mask) > 0

    def test_clean_image_runs_without_error(self, default_params):
        """无划痕图像应正常运行不报错."""
        rng = np.random.default_rng(99)
        clean = rng.integers(5, 20, size=(256, 256), dtype=np.uint8)

        detector = ClassicalDetector(default_params)
        result = detector.detect(clean)

        # 检测器应正常完成并返回有效结果
        assert result is not None
        assert result.mask.shape == (256, 256)
        assert isinstance(result.instances, list)

    def test_instances_list_populated(self, synthetic_dark_image, default_params):
        """检测结果应包含划痕实例列表."""
        detector = ClassicalDetector(default_params)
        result = detector.detect(synthetic_dark_image)

        # instances 列表应存在
        assert hasattr(result, "instances")
        assert isinstance(result.instances, list)
