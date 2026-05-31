"""CNAS 结果输出."""

from __future__ import annotations

import json
import time
from pathlib import Path

from cnas_test.runner.config import (
    COMPAT_COMMAND,
    DATASET_CONSTRUCTION_METHOD,
    DATASET_NAME,
    DATASET_SUMMARY,
    MODEL_DESCRIPTION,
    OUTPUT_SUBDIRS,
    RECOMMENDED_COMMAND,
    SOFTWARE_NAME,
    SOFTWARE_VERSION,
    STANDARD_SAMPLE_UNIT,
    TEMPLATES_DIR,
    VERSION_NOTE,
)


def format_print_lines(result: dict) -> list[str]:
    """返回与 print_report 一致的文本行（供截图/日志复用，不含 ANSI）。"""
    sep = "=" * 60
    lines = [
        sep,
        f"  {SOFTWARE_NAME} — 第三方测试结果记录",
        sep,
        f"  测试版本    : {SOFTWARE_VERSION}（{VERSION_NOTE}）",
        f"  测试日期    : {result['test_timestamp']}",
        f"  模型权重    : {result['model_weights']}",
        f"  评测样本数  : {result['n_tiles']} 个",
        f"  评测参数    : conf={result['eval_conf']}, IoU={result['eval_iou']}",
        "",
        "  --- 各类别 AP@0.5 ---",
    ]
    for cls_name, ap in result["metrics"]["per_class_AP50"].items():
        lines.append(f"  {cls_name:<12}: {ap:.4f}")
    lines += [
        "",
        f"  mAP@0.5     : {result['metrics']['mAP50']:.4f}   <- 测试指标",
        f"  mAP@0.5:0.95: {result['metrics']['mAP50_95']:.4f}",
        f"  Precision   : {result['metrics']['precision']:.4f}",
        f"  Recall      : {result['metrics']['recall']:.4f}",
        "",
        "  结果说明    : 本页记录第三方测试实测值；判定依据以委托测试文件为准",
        f"  耗时        : {result['elapsed_seconds']:.1f} 秒",
        sep,
    ]
    return lines


def print_report(result: dict) -> None:
    print()
    for line in format_print_lines(result):
        print(line)


def build_result_payload(
    *,
    weights_path: Path,
    n_tiles: int,
    eval_conf: float,
    eval_iou: float,
    pass_threshold: float,
    metrics: dict,
    dataset_summary: dict,
    elapsed_seconds: float,
) -> dict:
    return {
        "test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_weights": str(weights_path),
        "n_tiles": n_tiles,
        "eval_conf": eval_conf,
        "eval_iou": eval_iou,
        "pass_threshold": pass_threshold,
        "passed": metrics["mAP50"] >= pass_threshold,
        "metrics": metrics,
        "dataset": dataset_summary,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def prepare_output_dirs(save_root: Path) -> dict[str, Path]:
    save_root.mkdir(parents=True, exist_ok=True)
    output_dirs = {"root": save_root}
    for key, dirname in OUTPUT_SUBDIRS.items():
        path = save_root / dirname
        path.mkdir(parents=True, exist_ok=True)
        output_dirs[key] = path
    return output_dirs


def save_result_json(
    payload: dict, save_dir: Path, filename: str = "cnas_eval_results.json"
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _render_template(template_path: Path, replacements: dict[str, str]) -> str:
    text = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def save_markdown_report(
    payload: dict,
    save_dir: Path,
    *,
    weights_path: Path,
    test_set_path: Path,
    command: str = RECOMMENDED_COMMAND,
    compat_command: str = COMPAT_COMMAND,
    filename: str = "cnas_test_report.md",
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    template_path = TEMPLATES_DIR / "cnas_test_report_template.md"
    dataset = payload["dataset"]
    class_counts = dataset["class_counts"]
    metrics = payload["metrics"]
    content = _render_template(
        template_path,
        {
            "TEST_TIMESTAMP": payload["test_timestamp"],
            "SOFTWARE_NAME": SOFTWARE_NAME,
            "SOFTWARE_VERSION": SOFTWARE_VERSION,
            "VERSION_NOTE": VERSION_NOTE,
            "MODEL_DESCRIPTION": MODEL_DESCRIPTION,
            "DATASET_NAME": DATASET_NAME,
            "DATASET_CONSTRUCTION_METHOD": DATASET_CONSTRUCTION_METHOD,
            "STANDARD_SAMPLE_UNIT": STANDARD_SAMPLE_UNIT,
            "TEST_SET_PATH": str(test_set_path),
            "MODEL_WEIGHTS": str(weights_path),
            "COMMAND": command,
            "COMPAT_COMMAND": compat_command,
            "N_TILES": str(payload["n_tiles"]),
            "DATASET_IMAGES": str(dataset["images"]),
            "DATASET_TILES": str(dataset["tiles"]),
            "DATASET_BOXES": str(dataset["boxes"]),
            "BACKGROUND_TILES": str(dataset["background_tiles"]),
            "TRAIN_IMAGES": str(DATASET_SUMMARY["train"]["images"]),
            "TRAIN_TILES": str(DATASET_SUMMARY["train"]["tiles"]),
            "TRAIN_BOXES": str(DATASET_SUMMARY["train"]["boxes"]),
            "VAL_IMAGES": str(DATASET_SUMMARY["val"]["images"]),
            "VAL_TILES": str(DATASET_SUMMARY["val"]["tiles"]),
            "VAL_BOXES": str(DATASET_SUMMARY["val"]["boxes"]),
            "SCRATCH_BOXES": str(class_counts["scratch"]),
            "SPOT_BOXES": str(class_counts["spot"]),
            "CRITICAL_BOXES": str(class_counts["critical"]),
            "EVAL_CONF": f"{payload['eval_conf']}",
            "EVAL_IOU": f"{payload['eval_iou']}",
            "SCRATCH_AP50": f"{metrics['per_class_AP50']['scratch']:.4f}",
            "SPOT_AP50": f"{metrics['per_class_AP50']['spot']:.4f}",
            "CRITICAL_AP50": f"{metrics['per_class_AP50']['critical']:.4f}",
            "MAP50": f"{metrics['mAP50']:.4f}",
            "MAP50_95": f"{metrics['mAP50_95']:.4f}",
            "PRECISION": f"{metrics['precision']:.4f}",
            "RECALL": f"{metrics['recall']:.4f}",
            "ELAPSED_SECONDS": f"{payload['elapsed_seconds']:.1f}",
        },
    )
    out_path = save_dir / filename
    out_path.write_text(content, encoding="utf-8")
    return out_path


def save_delivery_manifest(
    save_root: Path,
    *,
    test_set_path: Path,
    weights_path: Path,
    dataset_yaml_path: Path,
    result_json_path: Path,
    report_path: Path,
    plots_dir: Path,
    html_report_path: Path | None = None,
    docx_report_path: Path | None = None,
    provenance_path: Path | None = None,
    screenshots_dir: Path | None = None,
) -> Path:
    artifacts = {
        "dataset_yaml": str(dataset_yaml_path),
        "dataset_list": str(dataset_yaml_path.parent / "cnas_val_list.txt"),
        "result_json": str(result_json_path),
        "markdown_report": str(report_path),
        "plots_dir": str(plots_dir),
    }
    if html_report_path is not None:
        artifacts["html_report"] = str(html_report_path)
    if docx_report_path is not None:
        artifacts["docx_report"] = str(docx_report_path)
    if provenance_path is not None:
        artifacts["provenance"] = str(provenance_path)
    if screenshots_dir is not None:
        artifacts["screenshots_dir"] = str(screenshots_dir)

    payload = {
        "delivery_root": str(save_root),
        "recommended_command": RECOMMENDED_COMMAND,
        "compat_command": COMPAT_COMMAND,
        "test_set_manifest": str(test_set_path),
        "weights_path": str(weights_path),
        "artifacts": artifacts,
    }
    out_path = save_root / "delivery_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
