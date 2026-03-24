"""CNAS 结果输出."""

from __future__ import annotations

import json
import time
from pathlib import Path

from cnas_test.runner.config import (
    CLASS_DISTRIBUTION,
    COMPAT_COMMAND,
    DATASET_SUMMARY,
    OUTPUT_SUBDIRS,
    PASS_THRESHOLD,
    RECOMMENDED_COMMAND,
    TEMPLATES_DIR,
)


def print_report(result: dict) -> None:
    sep = "=" * 60
    verdict = "✅ 通过" if result["passed"] else "❌ 未通过"

    print()
    print(sep)
    print("  暗场镜片缺陷检测系统 — CNAS 第三方测试结果")
    print(sep)
    print(f"  测试日期    : {result['test_timestamp']}")
    print(f"  模型权重    : {result['model_weights']}")
    print(f"  测试切片数  : {result['n_tiles']} 张")
    print(f"  评测参数    : conf={result['eval_conf']}, IoU={result['eval_iou']}")
    print()
    print("  --- 各类别 AP@0.5 ---")
    for cls_name, ap in result["metrics"]["per_class_AP50"].items():
        print(f"  {cls_name:<12}: {ap:.4f}")
    print()
    print(f"  mAP@0.5     : {result['metrics']['mAP50']:.4f}   ← 测试指标")
    print(f"  mAP@0.5:0.95: {result['metrics']['mAP50_95']:.4f}")
    print(f"  Precision   : {result['metrics']['precision']:.4f}")
    print(f"  Recall      : {result['metrics']['recall']:.4f}")
    print()
    print(f"  通过标准    : mAP@0.5 ≥ {result['pass_threshold']:.0%}")
    print(f"  测试结论    : {verdict}")
    print(f"  耗时        : {result['elapsed_seconds']:.1f} 秒")
    print(sep)


def build_result_payload(
    *,
    weights_path: Path,
    n_tiles: int,
    eval_conf: float,
    eval_iou: float,
    pass_threshold: float,
    metrics: dict,
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


def save_result_json(payload: dict, save_dir: Path, filename: str = "cnas_eval_results.json") -> Path:
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
    holdout = DATASET_SUMMARY["holdout"]
    holdout_dist = CLASS_DISTRIBUTION["holdout"]
    metrics = payload["metrics"]
    content = _render_template(
        template_path,
        {
            "TEST_TIMESTAMP": payload["test_timestamp"],
            "TEST_SET_PATH": str(test_set_path),
            "MODEL_WEIGHTS": str(weights_path),
            "COMMAND": command,
            "COMPAT_COMMAND": compat_command,
            "N_TILES": str(payload["n_tiles"]),
            "HOLDOUT_IMAGES": str(holdout["images"]),
            "HOLDOUT_TILES": str(holdout["tiles"]),
            "HOLDOUT_BOXES": str(holdout["boxes"]),
            "SCRATCH_BOXES": str(holdout_dist["scratch"]),
            "SPOT_BOXES": str(holdout_dist["spot"]),
            "CRITICAL_BOXES": str(holdout_dist["critical"]),
            "EVAL_CONF": f"{payload['eval_conf']}",
            "EVAL_IOU": f"{payload['eval_iou']}",
            "SCRATCH_AP50": f"{metrics['per_class_AP50']['scratch']:.4f}",
            "SPOT_AP50": f"{metrics['per_class_AP50']['spot']:.4f}",
            "CRITICAL_AP50": f"{metrics['per_class_AP50']['critical']:.4f}",
            "MAP50": f"{metrics['mAP50']:.4f}",
            "MAP50_95": f"{metrics['mAP50_95']:.4f}",
            "PRECISION": f"{metrics['precision']:.4f}",
            "RECALL": f"{metrics['recall']:.4f}",
            "PASS_THRESHOLD": f"{PASS_THRESHOLD:.2f}",
            "VERDICT": "通过" if payload["passed"] else "未通过",
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
) -> Path:
    payload = {
        "delivery_root": str(save_root),
        "recommended_command": RECOMMENDED_COMMAND,
        "compat_command": COMPAT_COMMAND,
        "test_set_manifest": str(test_set_path),
        "weights_path": str(weights_path),
        "artifacts": {
            "dataset_yaml": str(dataset_yaml_path),
            "dataset_list": str(dataset_yaml_path.parent / "cnas_val_list.txt"),
            "result_json": str(result_json_path),
            "markdown_report": str(report_path),
            "plots_dir": str(plots_dir),
        },
    }
    out_path = save_root / "delivery_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
