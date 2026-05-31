"""CNAS 独立测试子系统验证."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cnas_test.runner.config import DEFAULT_TEST_SET, make_timestamped_save_dir
from cnas_test.runner.dataset_loader import (
    build_val_dataset_yaml,
    collect_tile_paths,
    load_test_manifest,
    load_test_stems,
)
from cnas_test.runner.report import prepare_output_dirs, save_markdown_report


def test_manifest_exists_and_has_full_dataset() -> None:
    manifest = load_test_manifest(DEFAULT_TEST_SET)
    assert "images" in manifest
    assert len(manifest["images"]) == 247
    assert manifest["summary"]["tiles"] == 10621


def test_collect_tile_paths_matches_expected_full_dataset_tiles() -> None:
    stems = load_test_stems(DEFAULT_TEST_SET)
    tiles = collect_tile_paths(stems)
    assert len(stems) == 247
    assert len(tiles) == 10621


def test_build_val_dataset_yaml(tmp_path: Path) -> None:
    stems = load_test_stems(DEFAULT_TEST_SET)
    tiles = collect_tile_paths(stems[:2])
    yaml_path = build_val_dataset_yaml(tiles, tmp_path)
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    assert "names: ['scratch', 'spot', 'critical']" in text
    assert "train:" in text
    assert "val:" in text


def test_prepare_output_dirs_creates_delivery_layout(tmp_path: Path) -> None:
    output_dirs = prepare_output_dirs(tmp_path / "latest")
    assert output_dirs["dataset"].exists()
    assert output_dirs["metrics"].exists()
    assert output_dirs["plots"].exists()
    assert output_dirs["reports"].exists()


def test_make_timestamped_save_dir_uses_clear_run_name() -> None:
    out_dir = make_timestamped_save_dir(datetime(2026, 5, 20, 9, 30, 45))
    assert out_dir.name == "20260520_093045"
    assert "latest" not in out_dir.parts


def test_save_markdown_report_renders_template(tmp_path: Path) -> None:
    payload = {
        "test_timestamp": "2026-03-24T12:00:00",
        "n_tiles": 10621,
        "eval_conf": 0.001,
        "eval_iou": 0.6,
        "passed": True,
        "elapsed_seconds": 9.2,
        "metrics": {
            "per_class_AP50": {"scratch": 0.4525, "spot": 0.8180, "critical": 0.7589},
            "mAP50": 0.6765,
            "mAP50_95": 0.4541,
            "precision": 0.5997,
            "recall": 0.6735,
        },
        "dataset": {
            "images": 247,
            "tiles": 10621,
            "boxes": 90325,
            "background_tiles": 368,
            "missing_labels": [],
            "class_counts": {"scratch": 58032, "spot": 17475, "critical": 14818},
        },
    }
    report_path = save_markdown_report(
        payload,
        tmp_path,
        weights_path=Path(
            "output/experiments/phase3e/detection_training/b2_nwd_only_phase3e/weights/best.pt"
        ),
        test_set_path=DEFAULT_TEST_SET,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "mAP@0.5 | 0.6765" in text
    assert "第三方测试结果记录" in text
    assert "离焦微结构镜片磨损识别数据集" in text
    assert "符合性判定以委托测试文件" in text
    assert "测试结论：`通过`" not in text
