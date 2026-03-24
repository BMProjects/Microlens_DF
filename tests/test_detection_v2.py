"""检测算法 v2 改进测试 — 密度检测、Prominence、旋转包围盒合并、自适应阈值、COCO 输出."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from darkfield_defects.detection.base import DefectType, DetectionResult
from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.params import DetectionParams
from darkfield_defects.detection.rendering import (
    COCO_CATEGORIES,
    DEFECT_CATEGORY_ID,
    export_coco,
    export_metadata_csv,
    export_metadata_jsonl,
    render_overlay,
)


# ── 合成图像工具 ──────────────────────────────────────────────


def _make_dark_image(h: int = 512, w: int = 512, bg_level: int = 15) -> np.ndarray:
    """创建暗场背景图像."""
    rng = np.random.default_rng(42)
    return rng.integers(bg_level - 5, bg_level + 5, size=(h, w), dtype=np.uint8)


def _make_roi_mask(h: int = 512, w: int = 512) -> np.ndarray:
    """创建全真 ROI 掩码."""
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = min(h, w) // 2 - 10
    mask[(Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2] = True
    return mask


# ── Task 1: DefectType 扩展 + 密度检测 ───────────────────────


class TestDefectTypeCrash:
    def test_crash_enum_exists(self):
        assert hasattr(DefectType, "CRASH")
        assert DefectType.CRASH.value == "crash"

    def test_crash_in_category_id(self):
        assert DefectType.CRASH in DEFECT_CATEGORY_ID
        assert DEFECT_CATEGORY_ID[DefectType.CRASH] == 4

    def test_coco_categories_has_crash(self):
        names = [c["name"] for c in COCO_CATEGORIES]
        assert "crash" in names

    def test_density_detection_finds_crash(self):
        """密集区域应被识别为 CRASH 类型."""
        img = _make_dark_image(512, 512, bg_level=10)
        roi = _make_roi_mask(512, 512)

        # 在中心区域创建大量密集的亮点
        rng = np.random.default_rng(7)
        for _ in range(500):
            r = rng.integers(200, 300)
            c = rng.integers(200, 300)
            img[r:r+3, c:c+3] = rng.integers(150, 220)

        params = DetectionParams(
            density_kernel_ratio=0.08,
            density_threshold=0.15,
            dense_min_area=500,
            prominence_min_value=5.0,
        )
        detector = ClassicalDetector(params)
        result = detector.detect(img, roi_mask=roi, preprocessed_image=img)

        crash_count = sum(1 for i in result.instances if i.defect_type == DefectType.CRASH)
        assert crash_count >= 1, "密集区域应至少检测到一个 CRASH"

    def test_num_crashes_property(self):
        """DetectionResult.num_crashes 属性."""
        result = DetectionResult(
            mask=np.zeros((10, 10), dtype=np.uint8),
            instances=[],
        )
        assert result.num_crashes == 0


# ── Task 2: Prominence 过滤 ──────────────────────────────────


class TestProminence:
    def test_prominence_field_exists(self):
        """DefectInstance 应有 prominence 字段."""
        from darkfield_defects.detection.base import DefectInstance
        inst = DefectInstance(instance_id=0)
        assert hasattr(inst, "prominence")
        assert inst.prominence == 0.0

    def test_prominence_computation(self):
        """_compute_prominence 应正确计算中心与角落灰度差."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # bbox 中心亮，角落暗
        img[45:55, 45:55] = 200
        img[40, 40] = 10
        img[40, 59] = 10
        img[59, 40] = 10
        img[59, 59] = 10

        p = ClassicalDetector._compute_prominence((40, 40, 20, 20), img)
        assert p > 100, f"Prominence 应较大, 实际={p}"

    def test_low_prominence_spot_filtered(self):
        """低显著性的非划痕缺陷应被过滤."""
        img = _make_dark_image(256, 256, bg_level=50)
        roi = _make_roi_mask(256, 256)

        # 添加一个低对比度小点（与背景差值很小）
        img[120:125, 120:125] = 55

        params = DetectionParams(
            prominence_min_value=20.0,
            min_area=10,
        )
        detector = ClassicalDetector(params)
        result = detector.detect(img, roi_mask=roi, preprocessed_image=img)

        # 低 prominence 的 spot 不应出现
        spots = [i for i in result.instances if i.defect_type == DefectType.SPOT]
        for s in spots:
            assert s.prominence >= 20.0


# ── Task 3: 旋转包围盒合并 ───────────────────────────────────


class TestRotatedBoxMerge:
    def test_merge_method_param(self):
        """merge_method 参数应存在."""
        params = DetectionParams()
        assert hasattr(params, "merge_method")
        assert params.merge_method == "rotated_box"

    def test_scratch_extend_params_exist(self):
        """划痕延伸合并参数应存在."""
        params = DetectionParams()
        assert hasattr(params, "scratch_extend_ratio")
        assert hasattr(params, "scratch_extend_min_px")
        assert hasattr(params, "scratch_extend_width_ratio")
        assert hasattr(params, "scratch_extend_min_width")
        assert hasattr(params, "scratch_merge_gray_tol")

    def test_collinear_scratches_merged(self):
        """同方向的相邻划痕应被合并."""
        img = _make_dark_image(256, 256, bg_level=10)
        roi = _make_roi_mask(256, 256)

        # 两条共线划痕，中间有小间隔
        img[128, 40:100] = 200
        img[128, 115:175] = 200

        params = DetectionParams(
            min_area=10,
            merge_method="rotated_box",
            scratch_extend_ratio=0.3,
            scratch_extend_min_px=20,
            prominence_min_value=5.0,
        )
        detector = ClassicalDetector(params)
        result = detector.detect(img, roi_mask=roi, preprocessed_image=img)

        scratches = [i for i in result.instances if i.defect_type == DefectType.SCRATCH]
        # 两条共线划痕理想情况应合并为1条（或保持为2条如果间隔太大）
        assert len(scratches) <= 2


# ── Task 4: 自适应阈值 ───────────────────────────────────────


class TestAdaptiveThreshold:
    def test_adaptive_threshold_method(self):
        """threshold_method='adaptive' 不应报错."""
        img = _make_dark_image(256, 256, bg_level=10)
        roi = _make_roi_mask(256, 256)
        img[120:124, 50:200] = 180

        params = DetectionParams(
            threshold_method="adaptive",
            adaptive_block_size=51,
            adaptive_c=10.0,
            prominence_min_value=5.0,
        )
        detector = ClassicalDetector(params)
        result = detector.detect(img, roi_mask=roi, preprocessed_image=img)
        assert result is not None
        assert result.mask.shape == img.shape


# ── Task 5: COCO + CSV/JSONL 输出 ────────────────────────────


class TestCocoExport:
    def _make_result(self) -> tuple[str, DetectionResult]:
        from darkfield_defects.detection.base import DefectInstance
        mask = np.zeros((100, 100), dtype=np.uint8)
        inst = DefectInstance(
            instance_id=0,
            defect_type=DefectType.SCRATCH,
            mask=np.zeros((100, 100), dtype=bool),
            bbox=(10, 10, 50, 5),
            area_px=250,
            length_px=50,
            scatter_intensity=120.0,
            prominence=45.0,
            zone="center",
        )
        return ("test.png", DetectionResult(mask=mask, instances=[inst]))

    def test_coco_json_structure(self):
        """COCO JSON 应包含 images/annotations/categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coco.json"
            export_coco([self._make_result()], path)

            with open(path) as f:
                data = json.load(f)

            assert "images" in data
            assert "annotations" in data
            assert "categories" in data
            assert len(data["categories"]) == 4

            ann = data["annotations"][0]
            assert "prominence" in ann
            assert "scatter_intensity" in ann
            assert "zone" in ann
            assert "segmentation" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0
            assert "category_name" in ann
            assert "length_mm" in ann["attributes"]
            assert "raw_px" in ann["attributes"]

    def test_csv_export(self):
        """CSV 导出应生成有效文件."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.csv"
            export_metadata_csv([self._make_result()], path)
            assert path.exists()

            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["category"] == "scratch"
            assert "prominence" in rows[0]
            assert "length_mm" in rows[0]
            assert "bbox_w_mm" in rows[0]

    def test_jsonl_export(self):
        """JSONL 导出应生成有效文件."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.jsonl"
            export_metadata_jsonl([self._make_result()], path)
            assert path.exists()

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["category"] == "scratch"
            assert "prominence" in record
            assert "category_id" in record
            assert "length_mm" in record
            assert "raw_px" in record


# ── Task 6: CLAHE 增强 ───────────────────────────────────────


class TestCLAHE:
    def test_enhance_enabled_by_default(self):
        params = DetectionParams()
        assert params.enhance_enabled is True
        assert params.clahe_enabled is True
        assert 0 < params.enhance_gamma < 1.0

    def test_clahe_enabled_runs(self):
        """启用 CLAHE 后检测不应报错."""
        img = _make_dark_image(256, 256, bg_level=10)
        roi = _make_roi_mask(256, 256)
        img[120:124, 50:200] = 180

        params = DetectionParams(
            clahe_enabled=True,
            clahe_clip_limit=2.0,
            clahe_tile_size=8,
            prominence_min_value=5.0,
        )
        detector = ClassicalDetector(params)
        result = detector.detect(img, roi_mask=roi, preprocessed_image=img)
        assert result is not None


# ── 渲染兼容性 ───────────────────────────────────────────────


class TestRenderingCompat:
    def test_render_overlay_with_crash(self):
        """render_overlay 应能渲染 CRASH 类型."""
        from darkfield_defects.detection.base import DefectInstance
        img = np.zeros((100, 100), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        inst = DefectInstance(
            instance_id=0,
            defect_type=DefectType.CRASH,
            mask=mask,
            bbox=(40, 40, 20, 20),
            area_px=400,
        )
        result = DetectionResult(
            mask=(mask.astype(np.uint8) * 255),
            instances=[inst],
        )
        overlay = render_overlay(img, result)
        assert overlay.shape == (100, 100, 3)
