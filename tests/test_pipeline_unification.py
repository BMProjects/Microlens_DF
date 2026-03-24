"""验证检测管线与模板背景预处理管线的集成."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from darkfield_defects.cli.app import app
from darkfield_defects.detection.base import DetectionResult
from darkfield_defects.detection.classical import ClassicalDetector, _estimate_center_from_roi
from darkfield_defects.detection.params import DetectionParams

runner = CliRunner()


class TestEstimateCenterFromRoi:
    def test_estimate_center_from_roi_empty(self):
        mask = np.zeros((100, 100), dtype=bool)
        center, radius = _estimate_center_from_roi(mask)
        assert center == (50, 50)
        assert radius == 0.0

    def test_estimate_center_from_roi_circle(self):
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (60, 40), 25, 255, -1)
        center, radius = _estimate_center_from_roi(mask.astype(bool))
        assert center == (40, 60)
        assert abs(radius - 25.0) < 1.0


class TestDetectorWithPreprocessedImage:
    def test_skip_internal_preprocess_if_provided(self):
        img_raw = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        preprocessed = np.zeros((100, 100), dtype=np.uint8)
        preprocessed[48:52, 20:80] = 200

        roi_mask = np.zeros((100, 100), dtype=bool)
        roi_mask[10:90, 10:90] = True

        detector = ClassicalDetector(DetectionParams())
        result = detector.detect(
            img_raw,
            roi_mask=roi_mask,
            preprocessed_image=preprocessed,
        )

        assert isinstance(result, DetectionResult)
        assert result.metadata["roi_mask"] is roi_mask
        assert result.metadata["background_used"] is True
        assert result.num_scratches > 0


class TestCliTemplatePipeline:
    def test_cli_detect_help_has_calibration(self):
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "--calibration" in result.output
        assert "--pipeline" not in result.output

    def test_cli_missing_calibration(self):
        result = runner.invoke(app, ["detect", "dummy_input"])
        assert result.exit_code != 0
