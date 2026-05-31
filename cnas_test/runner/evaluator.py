"""CNAS 模型评测执行器."""

from __future__ import annotations

import time
from pathlib import Path

from cnas_test.runner.config import (
    CLASS_NAMES,
    DATASET_NAME,
    DEFAULT_TEST_SET,
    EVAL_CONF,
    EVAL_IOU,
    PASS_THRESHOLD,
    PROJECT_ROOT,
    SOFTWARE_NAME,
    STANDARD_SAMPLE_UNIT,
    make_timestamped_save_dir,
)
from cnas_test.runner.dataset_loader import (
    build_val_dataset_yaml,
    collect_tile_paths,
    load_test_stems,
    summarize_tiles,
)
from cnas_test.runner.provenance import (
    collect_provenance,
    finalize_provenance,
    save_provenance,
)
from cnas_test.runner.report import (
    build_result_payload,
    format_print_lines,
    prepare_output_dirs,
    print_report,
    save_delivery_manifest,
    save_markdown_report,
    save_result_json,
)
from cnas_test.runner.report_docx import build_docx_report
from cnas_test.runner.report_html import build_html_report
from cnas_test.runner.screenshot import collect_screenshots


def evaluate_model(weights_path: Path, yaml_path: Path, plots_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(yaml_path),
        conf=EVAL_CONF,
        iou=EVAL_IOU,
        imgsz=640,
        batch=16,
        workers=4,
        plots=True,
        save_dir=str(plots_dir / "ultralytics_output"),
        verbose=False,
    )

    per_class_ap = {}
    ap50_array = metrics.box.ap50
    for idx, name in CLASS_NAMES.items():
        per_class_ap[name] = float(ap50_array[idx]) if idx < len(ap50_array) else 0.0

    return {
        "per_class_AP50": per_class_ap,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _startup_lines(
    *,
    test_set_path: Path,
    n_stems: int,
    n_tiles: int,
    n_boxes: int,
    save_dir: Path,
    yaml_path: Path,
    weights_path: Path,
) -> list[str]:
    return [
        "=" * 60,
        f"  CNAS 第三方测试 — {SOFTWARE_NAME}",
        "=" * 60,
        f"[测试集] {DATASET_NAME}（清单：{test_set_path.name}）",
        f"[类型]  {STANDARD_SAMPLE_UNIT}数据集",
        f"[样本]  共找到 {n_tiles} 个 640×640 {STANDARD_SAMPLE_UNIT}",
        f"[编号]  覆盖 {n_stems} 个镜片图像编号",
        f"[标注]  共统计 {n_boxes} 个缺陷框",
        f"[输出]  交付目录 → {save_dir}",
        f"[配置]  数据集 YAML → {yaml_path}",
        f"[权重]  {weights_path}",
        "",
        "[评测中] 正在运行标准评测…",
    ]


def run_cnas_eval(
    weights_path: Path,
    *,
    test_set_path: Path = DEFAULT_TEST_SET,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    resolved_save_dir = save_dir or make_timestamped_save_dir()
    output_dirs = prepare_output_dirs(resolved_save_dir)
    stems = load_test_stems(test_set_path)
    tile_paths = collect_tile_paths(stems)
    dataset_summary = summarize_tiles(tile_paths)
    yaml_path = build_val_dataset_yaml(tile_paths, output_dirs["dataset"])

    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    t0 = time.time()
    provenance = collect_provenance(
        repo_root=PROJECT_ROOT,
        weights_path=weights_path,
        test_set_path=test_set_path,
        started_at=t0,
    )

    startup_lines = _startup_lines(
        test_set_path=test_set_path,
        n_stems=len(stems),
        n_tiles=len(tile_paths),
        n_boxes=dataset_summary["boxes"],
        save_dir=resolved_save_dir,
        yaml_path=yaml_path,
        weights_path=weights_path,
    )
    if verbose:
        print("\n" + "\n".join(startup_lines))

    metrics = evaluate_model(weights_path, yaml_path, output_dirs["plots"])
    elapsed = time.time() - t0

    payload = build_result_payload(
        weights_path=weights_path,
        n_tiles=len(tile_paths),
        eval_conf=EVAL_CONF,
        eval_iou=EVAL_IOU,
        pass_threshold=PASS_THRESHOLD,
        metrics=metrics,
        dataset_summary=dataset_summary,
        elapsed_seconds=elapsed,
    )
    out_path = save_result_json(payload, output_dirs["metrics"])
    report_path = save_markdown_report(
        payload,
        output_dirs["reports"],
        weights_path=weights_path,
        test_set_path=test_set_path,
    )

    result_lines = format_print_lines(payload)
    if verbose:
        print_report(payload)

    screenshots = collect_screenshots(
        output_dirs["screenshots"],
        startup_lines=startup_lines,
        result_lines=result_lines,
        metrics=metrics,
        pass_threshold=PASS_THRESHOLD,
    )

    html_report_path = output_dirs["reports"] / "cnas_test_report.html"
    docx_report_path = output_dirs["reports"] / "cnas_test_report.docx"
    finalized_provenance = finalize_provenance(
        provenance,
        finished_at=time.time(),
        artifacts={
            "result_json": str(out_path),
            "markdown_report": str(report_path),
            "html_report": str(html_report_path),
            "docx_report": str(docx_report_path),
            "plots_dir": str(output_dirs["plots"]),
            "screenshots_dir": str(output_dirs["screenshots"]),
        },
    )

    html_path = build_html_report(
        payload=payload,
        provenance=finalized_provenance,
        screenshots=screenshots,
        plots_dir=output_dirs["plots"],
        test_set_path=test_set_path,
        weights_path=weights_path,
        save_path=html_report_path,
    )

    try:
        docx_path: Path | None = build_docx_report(
            payload=payload,
            provenance=finalized_provenance,
            screenshots=screenshots,
            plots_dir=output_dirs["plots"],
            test_set_path=test_set_path,
            weights_path=weights_path,
            save_path=docx_report_path,
        )
    except RuntimeError as exc:
        docx_path = None
        if verbose:
            print(f"[警告] 跳过 .docx 生成：{exc}")
        finalized_provenance["artifacts"]["docx_report"] = ""
    provenance_path = save_provenance(finalized_provenance, output_dirs["provenance"])

    manifest_path = save_delivery_manifest(
        resolved_save_dir,
        test_set_path=test_set_path,
        weights_path=weights_path,
        dataset_yaml_path=yaml_path,
        result_json_path=out_path,
        report_path=report_path,
        plots_dir=output_dirs["plots"],
        html_report_path=html_path,
        docx_report_path=docx_path,
        provenance_path=provenance_path,
        screenshots_dir=output_dirs["screenshots"],
    )

    if verbose:
        print(f"\n[已保存] {out_path}")
        print(f"[Markdown 报告] {report_path}")
        print(f"[HTML 报告]     {html_path}")
        if docx_path:
            print(f"[DOCX 报告]     {docx_path}")
        print(f"[溯源]   {provenance_path}")
        print(f"[清单]   {manifest_path}")

    return payload
