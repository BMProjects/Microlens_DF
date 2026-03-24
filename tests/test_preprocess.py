"""预处理模块单元测试."""

from __future__ import annotations

import numpy as np
import pytest

from darkfield_defects.detection.preprocess import (
    denoise,
    enhance_contrast,
    flat_field_correct,
)


class TestFlatFieldCorrect:
    def test_uniform_background(self) -> None:
        """均匀背景下校正后应接近原图."""
        img = np.random.randint(10, 100, (64, 64), dtype=np.uint8)
        bg = np.full((64, 64), 50.0, dtype=np.float64)
        result = flat_field_correct(img, bg)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_gradient_correction(self) -> None:
        """渐变背景校正后应更均匀."""
        # 构造一个从左到右渐变的背景
        bg = np.tile(np.linspace(20, 80, 64), (64, 1)).astype(np.float64)
        # 信号叠加在背景上
        img = (bg * 0.8 + 20).astype(np.uint8)
        result = flat_field_correct(img, bg)
        # 校正后左右标准差应减小
        std_before = float(np.std(img.astype(float), axis=1).mean())
        std_after = float(np.std(result.astype(float), axis=1).mean())
        assert std_after < std_before

    def test_output_range(self) -> None:
        """输出应在 [0, 255] 范围内."""
        img = np.full((32, 32), 200, dtype=np.uint8)
        bg = np.full((32, 32), 10.0, dtype=np.float64)
        result = flat_field_correct(img, bg)
        assert result.min() >= 0
        assert result.max() <= 255


class TestDenoise:
    def test_bilateral(self) -> None:
        """双边滤波应返回同样 shape."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = denoise(img, method="bilateral")
        assert result.shape == img.shape

    def test_nlm(self) -> None:
        """非局部均值应返回同样 shape."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = denoise(img, method="nlm")
        assert result.shape == img.shape


class TestEnhanceContrast:
    def test_clahe(self) -> None:
        """CLAHE 应扩展动态范围."""
        img = np.random.randint(100, 150, (64, 64), dtype=np.uint8)
        result = enhance_contrast(img)
        assert result.shape == img.shape
        # 增强后动态范围应更大
        assert float(np.std(result)) >= float(np.std(img)) * 0.8
