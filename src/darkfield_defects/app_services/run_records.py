"""检测记录服务 — 每次检测落盘为一个记录目录，供报告下载、批量导出与历史查询。

目录结构（默认 output/app_runs/，可用环境变量 DARKFIELD_RUNS_DIR 覆盖）::

    <YYYYmmdd_HHMMSS_ffffff>_<图像名>/
        report.json   指标、评级与运行参数
        report.html   自包含 HTML 报告（叠加图内嵌）
        overlay.png   检测叠加图
"""

from __future__ import annotations

import base64
import csv
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.app_services.inference_service import PROJECT_ROOT, FullImageInferenceResult
from darkfield_defects.scoring.report import generate_html_report, generate_json_report

CLASS_NAMES = ("scratch", "spot", "critical")
SUMMARY_FIELDS = ["run_id", "image", "time", "grade", "score", "n_defects",
                  "scratch", "spot", "critical", "infer_s", "conf", "weights"]


def runs_dir() -> Path:
    return Path(os.environ.get("DARKFIELD_RUNS_DIR", PROJECT_ROOT / "output" / "app_runs"))


def summarize(result: FullImageInferenceResult) -> dict:
    counts = {name: sum(1 for b in result.final_boxes if b[0] == i)
              for i, name in enumerate(CLASS_NAMES)}
    return {
        "grade": result.assessment.grade,
        "score": round(float(result.assessment.score), 1),
        "n_defects": len(result.final_boxes),
        **counts,
        "n_chains": result.n_chains,
        "infer_s": round(float(result.infer_time), 2),
    }


def save_run(
    result: FullImageInferenceResult,
    image_path: str | Path,
    overlay_rgb: np.ndarray,
    *,
    conf: float,
    weights: str | Path,
) -> Path:
    """保存一次检测：叠加图 + JSON + HTML 报告，返回记录目录。"""
    image_path = Path(image_path)
    now = datetime.now()
    run_dir = runs_dir() / f"{now:%Y%m%d_%H%M%S_%f}_{image_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=False)

    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("叠加图编码失败")
    (run_dir / "overlay.png").write_bytes(png.tobytes())

    extra = {
        "run_id": run_dir.name,
        "time": now.isoformat(timespec="seconds"),
        "conf": conf,
        "weights": str(weights),
        "summary": summarize(result),
    }
    generate_json_report(image_path.name, result.metrics, result.assessment,
                         run_dir / "report.json", extra=extra)
    data_uri = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    generate_html_report(image_path.name, result.metrics, result.assessment,
                         run_dir / "report.html", overlay_path=data_uri)
    return run_dir


def _row(run_dir: Path) -> dict | None:
    try:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    extra = report.get("extra", {})
    s = extra.get("summary", {})
    return {
        "run_id": run_dir.name, "image": report.get("source_image", ""),
        "time": extra.get("time", ""), "grade": s.get("grade", ""), "score": s.get("score", ""),
        "n_defects": s.get("n_defects", ""), "scratch": s.get("scratch", ""),
        "spot": s.get("spot", ""), "critical": s.get("critical", ""),
        "infer_s": s.get("infer_s", ""), "conf": extra.get("conf", ""),
        "weights": Path(str(extra.get("weights", ""))).name,
    }


def list_runs(limit: int = 200) -> list[dict]:
    """按时间倒序列出检测记录；损坏的记录目录跳过。"""
    root = runs_dir()
    if not root.is_dir():
        return []
    rows = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        row = _row(d)
        if row is not None:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def load_run(run_id: str) -> tuple[dict, Path, Path]:
    """返回 (report.json 内容, overlay.png 路径, report.html 路径)。"""
    run_dir = runs_dir() / run_id
    if run_dir.parent != runs_dir() or not (run_dir / "report.json").is_file():
        raise FileNotFoundError(f"检测记录不存在: {run_id}")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return report, run_dir / "overlay.png", run_dir / "report.html"


def export_runs(run_ids: list[str], out_dir: Path) -> tuple[Path, Path]:
    """把一组记录导出为汇总 CSV 与 ZIP（含各自报告与叠加图）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    rows = [r for r in (_row(runs_dir() / rid) for rid in run_ids) if r is not None]
    csv_path = out_dir / f"batch_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM 便于 Excel 打开中文
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    zip_path = out_dir / f"batch_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_path.name)
        for rid in run_ids:
            for name in ("report.json", "report.html", "overlay.png"):
                p = runs_dir() / rid / name
                if p.exists():
                    zf.write(p, f"{rid}/{name}")
    return csv_path, zip_path
