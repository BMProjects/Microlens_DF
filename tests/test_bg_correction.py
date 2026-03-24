"""验证 bg_correction_method 配置生效."""

from __future__ import annotations

import numpy as np

from darkfield_defects.detection.params import PreprocessParams
from darkfield_defects.detection.preprocess import (
    flat_field_correct,
    flat_field_subtract,
    preprocess_image,
)


class TestFlatFieldSubtract:
    """新增的减法校正函数."""

    def test_basic_subtraction(self):
        """减法校正应返回有效的 uint8 图像."""
        img = np.full((64, 64), 100, dtype=np.uint8)
        bg = np.full((64, 64), 80.0, dtype=np.float64)
        result = flat_field_subtract(img, bg)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_subtraction_reduces_background(self):
        """减法校正后，均匀背景上的信号应被保留."""
        # 构造：背景=50，信号区=150 (信号比背景高100)
        bg = np.full((64, 64), 50.0, dtype=np.float64)
        img = np.full((64, 64), 50, dtype=np.uint8)
        img[20:40, 20:40] = 150  # 信号区

        result = flat_field_subtract(img, bg)
        # 信号区应显著高于背景区
        signal_mean = float(np.mean(result[20:40, 20:40]))
        bg_mean = float(np.mean(result[0:10, 0:10]))
        assert signal_mean > bg_mean


class TestBgCorrectionMethodConfig:
    """bg_correction_method 配置应影响 preprocess_image 行为."""

    def test_division_mode(self):
        """默认 division 模式应正常工作."""
        img = np.random.randint(10, 100, (64, 64), dtype=np.uint8)
        bg = np.full((64, 64), 50.0, dtype=np.float64)
        params = PreprocessParams(
            bg_correction_method="division",
            roi_method="threshold",
            roi_min_radius_ratio=0.1,
        )
        result = preprocess_image(img, bg, params)
        assert result.corrected.shape == img.shape
        assert result.background_used is True

    def test_subtraction_mode(self):
        """subtraction 模式应正常工作且 background_used=True."""
        img = np.random.randint(10, 100, (64, 64), dtype=np.uint8)
        bg = np.full((64, 64), 50.0, dtype=np.float64)
        params = PreprocessParams(
            bg_correction_method="subtraction",
            roi_method="threshold",
            roi_min_radius_ratio=0.1,
        )
        result = preprocess_image(img, bg, params)
        assert result.corrected.shape == img.shape
        assert result.background_used is True

    def test_no_background_skips_correction(self):
        """无背景图时不执行校正."""
        img = np.random.randint(10, 100, (64, 64), dtype=np.uint8)
        params = PreprocessParams()
        result = preprocess_image(img, background=None, params=params)
        assert result.background_used is False
