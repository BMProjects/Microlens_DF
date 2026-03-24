"""评估指标与评分模块单元测试."""

from __future__ import annotations

import numpy as np

from darkfield_defects.eval import compute_segmentation_metrics
from darkfield_defects.scoring.quantify import WearMetrics
from darkfield_defects.scoring.wear_score import WearAssessment, compute_wear_score


class TestSegmentationMetrics:
    def test_perfect_match(self) -> None:
        """完全匹配应得到 P=R=F1=IoU=1."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        m = compute_segmentation_metrics(mask, mask)
        assert abs(m.precision - 1.0) < 1e-6
        assert abs(m.recall - 1.0) < 1e-6
        assert abs(m.f1 - 1.0) < 1e-6
        assert abs(m.iou - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        """无重叠应得到 P=R=F1=IoU=0."""
        pred = np.zeros((32, 32), dtype=np.uint8)
        gt = np.zeros((32, 32), dtype=np.uint8)
        pred[0:5, 0:5] = 255
        gt[20:25, 20:25] = 255
        m = compute_segmentation_metrics(pred, gt)
        assert m.precision == 0
        assert m.recall == 0

    def test_partial_overlap(self, binary_mask_pair) -> None:
        """部分重叠应产生合理的中间值."""
        pred, gt = binary_mask_pair
        m = compute_segmentation_metrics(pred, gt)
        assert 0 < m.precision < 1
        assert 0 < m.recall < 1
        assert 0 < m.f1 < 1


class TestWearScore:
    def test_clean_lens(self) -> None:
        """无划痕应得到 Grade A."""
        metrics = WearMetrics()
        result = compute_wear_score(metrics)
        assert result.grade == "A"
        assert result.score < 5

    def test_heavily_worn(self) -> None:
        """大量划痕应得到 Grade D."""
        metrics = WearMetrics(
            L_total=5000,
            L_center=2000,
            N_total=50,
            N_center=20,
            A_total=10000,
            S_scatter=150,
            S_scatter_center=180,
        )
        result = compute_wear_score(metrics)
        assert result.grade in ("C", "D")
        assert result.score > 40

    def test_score_has_contributors(self) -> None:
        """评分结果应包含贡献因素."""
        metrics = WearMetrics(L_total=100, N_total=3)
        result = compute_wear_score(metrics)
        assert result.contributors is not None
        assert len(result.contributors) > 0

    def test_conclusion_nonempty(self) -> None:
        """评估结论不应为空."""
        metrics = WearMetrics(L_total=500, L_center=200, N_total=10, N_center=5)
        result = compute_wear_score(metrics)
        assert len(result.conclusion) > 0


class TestWearMetricsUnits:
    def test_to_dict_uses_mm_units(self) -> None:
        metrics = WearMetrics(
            L_total=100,
            L_center=40,
            A_total=250,
            A_center=100,
            roi_area=1000,
            pixel_size_mm=0.0068,
        )
        result = metrics.to_dict()
        assert result["unit_length"] == "mm"
        assert result["unit_area"] == "mm^2"
        assert abs(result["L_total"] - 0.68) < 1e-6
        assert abs(result["A_total"] - (250 * 0.0068 * 0.0068)) < 1e-6
        assert result["raw_px"]["L_total_px"] == 100
