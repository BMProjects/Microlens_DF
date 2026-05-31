"""CNAS 测试子系统：溯源 / 截图 / HTML / DOCX 报告模块测试."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnas_test.runner.config import PASS_THRESHOLD
from cnas_test.runner.provenance import (
    collect_provenance,
    finalize_provenance,
    save_provenance,
)
from cnas_test.runner.report import format_print_lines
from cnas_test.runner.report_html import build_html_report
from cnas_test.runner.screenshot import (
    collect_screenshots,
    render_result_summary,
    render_text_screenshot,
)


@pytest.fixture()
def fake_metrics() -> dict:
    return {
        "per_class_AP50": {"scratch": 0.50, "spot": 0.80, "critical": 0.70},
        "mAP50": 0.6667,
        "mAP50_95": 0.45,
        "precision": 0.60,
        "recall": 0.67,
    }


@pytest.fixture()
def fake_payload(fake_metrics: dict) -> dict:
    return {
        "test_timestamp": "2026-05-19T10:00:00",
        "model_weights": "/tmp/fake.pt",
        "n_tiles": 10621,
        "eval_conf": 0.001,
        "eval_iou": 0.6,
        "pass_threshold": PASS_THRESHOLD,
        "passed": True,
        "metrics": fake_metrics,
        "dataset": {
            "images": 247,
            "tiles": 10621,
            "boxes": 90325,
            "background_tiles": 368,
            "missing_labels": [],
            "class_counts": {"scratch": 58032, "spot": 17475, "critical": 14818},
        },
        "elapsed_seconds": 9.0,
    }


def test_format_print_lines_records_objective_result(fake_payload: dict) -> None:
    lines = format_print_lines(fake_payload)
    joined = "\n".join(lines)
    assert "第三方测试结果记录" in joined
    assert "判定依据以委托测试文件" in joined
    assert "通过" not in joined
    assert "mAP@0.5" in joined
    assert "scratch" in joined


def test_collect_provenance_records_sha256(tmp_path: Path) -> None:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"\x00" * 32)
    test_set = tmp_path / "manifest.json"
    test_set.write_text("{}", encoding="utf-8")

    prov = collect_provenance(
        repo_root=tmp_path,
        weights_path=weights,
        test_set_path=test_set,
        started_at=0.0,
    )
    assert prov["weights"]["sha256"]
    assert len(prov["weights"]["sha256"]) == 64
    assert prov["test_set"]["sha256"]
    assert prov["environment"]["python_version"]


def test_finalize_and_save_provenance(tmp_path: Path) -> None:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    test_set = tmp_path / "m.json"
    test_set.write_text("{}", encoding="utf-8")
    prov = collect_provenance(
        repo_root=tmp_path, weights_path=weights, test_set_path=test_set, started_at=10.0
    )
    finalized = finalize_provenance(prov, finished_at=15.5, artifacts={"x": "y"})
    assert finalized["duration_seconds"] == 5.5
    assert finalized["artifacts"] == {"x": "y"}
    out_path = save_provenance(finalized, tmp_path)
    assert out_path.exists()
    assert out_path.name == "provenance.json"


def test_render_text_screenshot_writes_png(tmp_path: Path) -> None:
    out = render_text_screenshot(["line one", "line two"], tmp_path / "txt.png", title="标题")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_render_result_summary_writes_png(tmp_path: Path, fake_metrics: dict) -> None:
    out = render_result_summary(fake_metrics, PASS_THRESHOLD, tmp_path / "chart.png")
    assert out.exists()


def test_collect_screenshots_creates_three_pngs(tmp_path: Path, fake_metrics: dict) -> None:
    shots = collect_screenshots(
        tmp_path,
        startup_lines=["a", "b"],
        result_lines=["c", "d"],
        metrics=fake_metrics,
        pass_threshold=PASS_THRESHOLD,
    )
    assert set(shots.keys()) == {"startup", "result_text", "metrics_chart"}
    for p in shots.values():
        assert p.exists()


def test_build_html_report_is_self_contained(tmp_path: Path, fake_payload: dict) -> None:
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    test_set = tmp_path / "m.json"
    test_set.write_text("{}", encoding="utf-8")
    prov = finalize_provenance(
        collect_provenance(
            repo_root=tmp_path,
            weights_path=weights,
            test_set_path=test_set,
            started_at=0.0,
        ),
        finished_at=1.0,
        artifacts={},
    )
    shots = collect_screenshots(
        tmp_path / "shots",
        startup_lines=["x"],
        result_lines=["y"],
        metrics=fake_payload["metrics"],
        pass_threshold=PASS_THRESHOLD,
    )
    out = build_html_report(
        payload=fake_payload,
        provenance=prov,
        screenshots=shots,
        plots_dir=tmp_path / "plots",
        test_set_path=test_set,
        weights_path=weights,
        save_path=tmp_path / "report.html",
    )
    text = out.read_text(encoding="utf-8")
    assert "<title>镜片磨损智能识别算法 CNAS 测试报告" in text
    assert "data:image/png;base64," in text
    assert "mAP@0.5" in text


def test_build_docx_report_writes_openable_docx(tmp_path: Path, fake_payload: dict) -> None:
    import docx

    from cnas_test.runner.report_docx import build_docx_report

    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    test_set = tmp_path / "m.json"
    test_set.write_text("{}", encoding="utf-8")
    prov = finalize_provenance(
        collect_provenance(
            repo_root=tmp_path,
            weights_path=weights,
            test_set_path=test_set,
            started_at=0.0,
        ),
        finished_at=1.0,
        artifacts={},
    )
    shots = collect_screenshots(
        tmp_path / "shots",
        startup_lines=["x"],
        result_lines=["y"],
        metrics=fake_payload["metrics"],
        pass_threshold=PASS_THRESHOLD,
    )
    out = build_docx_report(
        payload=fake_payload,
        provenance=prov,
        screenshots=shots,
        plots_dir=tmp_path / "plots",
        test_set_path=test_set,
        weights_path=weights,
        save_path=tmp_path / "report.docx",
    )
    assert out.exists()
    assert docx.Document(str(out))  # 可被正常打开
