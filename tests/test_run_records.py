"""检测记录服务测试（保存 → 列表 → 读取 → 批量导出）."""

from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest

from darkfield_defects.app_services import run_records
from darkfield_defects.app_services.inference_service import (
    FullImageInferenceResult,
    boxes_to_detection_result,
    resolve_weights_path,
)
from darkfield_defects.scoring.quantify import compute_wear_metrics
from darkfield_defects.scoring.wear_score import compute_wear_score


def _fake_result() -> FullImageInferenceResult:
    img = np.full((800, 800), 10, dtype=np.uint8)
    img[390:410, 300:500] = 180
    boxes = [(0, 300.0, 390.0, 500.0, 410.0, 0.9), (1, 600.0, 600.0, 620.0, 620.0, 0.8)]
    det, roi = boxes_to_detection_result(boxes, img)
    metrics = compute_wear_metrics(det, roi)
    return FullImageInferenceResult(
        image_gray=img, roi_mask=roi, raw_boxes=boxes, nms_boxes=boxes, final_boxes=boxes,
        n_chains=0, infer_time=0.5, detection_result=det, metrics=metrics,
        assessment=compute_wear_score(metrics),
    )


def test_save_list_load_export(tmp_path, monkeypatch):
    monkeypatch.setenv("DARKFIELD_RUNS_DIR", str(tmp_path / "runs"))
    result = _fake_result()
    overlay = np.zeros((64, 64, 3), dtype=np.uint8)
    run_dir = run_records.save_run(result, "lens_01.png", overlay, conf=0.2, weights="w/best.pt")

    rows = run_records.list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["image"] == "lens_01.png" and row["scratch"] == 1 and row["spot"] == 1
    assert row["grade"] == result.assessment.grade

    report, overlay_path, html_path = run_records.load_run(run_dir.name)
    assert report["extra"]["conf"] == 0.2
    assert overlay_path.is_file() and "data:image/png;base64," in html_path.read_text()

    csv_path, zip_path = run_records.export_runs([run_dir.name], tmp_path / "exports")
    with open(csv_path, encoding="utf-8-sig") as f:
        assert list(csv.DictReader(f))[0]["image"] == "lens_01.png"
    names = zipfile.ZipFile(zip_path).namelist()
    assert f"{run_dir.name}/report.html" in names and csv_path.name in names


def test_load_run_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("DARKFIELD_RUNS_DIR", str(tmp_path / "runs"))
    with pytest.raises(FileNotFoundError):
        run_records.load_run("../etc")


def test_resolve_weights_path_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="DARKFIELD_WEIGHTS"):
        resolve_weights_path(tmp_path / "missing.pt")
