"""特征提取模块单元测试."""

from __future__ import annotations

import numpy as np

from darkfield_defects.detection.features import (
    compute_candidate_map,
    frangi_filter,
    gabor_response,
    tophat_filter,
)


class TestFrangiFilter:
    def test_bright_line_response(self) -> None:
        """亮线区域应产生较强的 Frangi 响应."""
        img = np.zeros((64, 64), dtype=np.uint8)
        img[30, 10:50] = 200  # 水平亮线
        result = frangi_filter(img, sigmas=[1.0, 2.0])
        assert result.shape == img.shape
        # 线上的响应应大于背景
        line_resp = float(result[30, 20:40].mean())
        bg_resp = float(result[10, 20:40].mean())
        assert line_resp > bg_resp

    def test_output_range(self) -> None:
        """输出应归一化到 [0, 1]."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = frangi_filter(img, sigmas=[1.0])
        assert result.min() >= 0
        assert result.max() <= 1.0 + 1e-6


class TestTophatFilter:
    def test_bright_detail_extraction(self) -> None:
        """顶帽变换应提取亮于背景的小结构."""
        img = np.full((64, 64), 50, dtype=np.uint8)
        img[30, 20:40] = 200  # 亮线
        result = tophat_filter(img, kernel_size=15)
        assert float(result[30, 30]) > float(result[10, 30])

    def test_shape_preserved(self) -> None:
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = tophat_filter(img)
        assert result.shape == img.shape


class TestGaborResponse:
    def test_directional_response(self) -> None:
        """应对线状结构产生响应."""
        img = np.zeros((64, 64), dtype=np.uint8)
        img[30:33, 10:50] = 200  # 水平线
        result = gabor_response(img, n_orientations=4)
        assert result.shape == img.shape
        assert result.max() > 0

    def test_output_range(self) -> None:
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = gabor_response(img, n_orientations=4)
        assert result.min() >= 0
        assert result.max() <= 1.0 + 1e-6


class TestCandidateMap:
    def test_candidate_from_scratched_image(self, synthetic_lens_image) -> None:
        """有划痕的图像应产生候选区."""
        candidate = compute_candidate_map(synthetic_lens_image)
        assert candidate.shape == synthetic_lens_image.shape
        assert candidate.max() > 0
