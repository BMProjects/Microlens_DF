"""CNAS 独立测试子系统验证."""

from __future__ import annotations

from pathlib import Path

from cnas_test.runner.config import DEFAULT_TEST_SET
from cnas_test.runner.dataset_loader import (
    build_val_dataset_yaml,
    collect_tile_paths,
    load_test_manifest,
    load_test_stems,
)
from cnas_test.runner.report import prepare_output_dirs, save_markdown_report


def test_manifest_exists_and_has_20_images() -> None:
    manifest = load_test_manifest(DEFAULT_TEST_SET)
    assert "images" in manifest
    assert len(manifest["images"]) == 20


def test_collect_tile_paths_matches_expected_holdout_tiles() -> None:
    stems = load_test_stems(DEFAULT_TEST_SET)
    tiles = collect_tile_paths(stems)
    assert len(stems) == 20
    assert len(tiles) == 860


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


def test_save_markdown_report_renders_template(tmp_path: Path) -> None:
    payload = {
        "test_timestamp": "2026-03-24T12:00:00",
        "n_tiles": 860,
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
    }
    report_path = save_markdown_report(
        payload,
        tmp_path,
        weights_path=Path("output/training/stage2_cleaned/weights/best.pt"),
        test_set_path=DEFAULT_TEST_SET,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "mAP@0.5 | 0.6765" in text
    assert "测试结论：`通过`" in text
