"""CNAS 模型评测执行器."""

from __future__ import annotations

import time
from pathlib import Path

from cnas_test.runner.config import (
    CLASS_NAMES,
    DEFAULT_SAVE_DIR,
    DEFAULT_TEST_SET,
    EVAL_CONF,
    EVAL_IOU,
    PASS_THRESHOLD,
)
from cnas_test.runner.dataset_loader import (
    build_val_dataset_yaml,
    collect_tile_paths,
    load_test_stems,
)
from cnas_test.runner.report import build_result_payload, print_report, save_result_json
from cnas_test.runner.report import (
    prepare_output_dirs,
    save_delivery_manifest,
    save_markdown_report,
)


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


def run_cnas_eval(
    weights_path: Path,
    *,
    test_set_path: Path = DEFAULT_TEST_SET,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    resolved_save_dir = save_dir or DEFAULT_SAVE_DIR
    output_dirs = prepare_output_dirs(resolved_save_dir)
    stems = load_test_stems(test_set_path)
    tile_paths = collect_tile_paths(stems)
    yaml_path = build_val_dataset_yaml(tile_paths, output_dirs["dataset"])

    if verbose:
        print("\n" + "=" * 60)
        print("  CNAS 第三方测试 — 缺陷检测准确率")
        print("=" * 60)
        print(f"[测试集] 共 {len(stems)} 张图像（来源：{test_set_path.name}）")
        print(f"[切片]  共找到 {len(tile_paths)} 张切片（来自 {len(stems)} 张图）")
        print(f"[输出]  交付目录 → {resolved_save_dir}")
        print(f"[配置]  数据集 YAML → {yaml_path}")
        print(f"[权重]  {weights_path}")
        print("\n[评测中] 正在运行标准评测…")

    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    t0 = time.time()
    metrics = evaluate_model(weights_path, yaml_path, output_dirs["plots"])
    elapsed = time.time() - t0

    payload = build_result_payload(
        weights_path=weights_path,
        n_tiles=len(tile_paths),
        eval_conf=EVAL_CONF,
        eval_iou=EVAL_IOU,
        pass_threshold=PASS_THRESHOLD,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )
    out_path = save_result_json(payload, output_dirs["metrics"])
    report_path = save_markdown_report(
        payload,
        output_dirs["reports"],
        weights_path=weights_path,
        test_set_path=test_set_path,
    )
    manifest_path = save_delivery_manifest(
        resolved_save_dir,
        test_set_path=test_set_path,
        weights_path=weights_path,
        dataset_yaml_path=yaml_path,
        result_json_path=out_path,
        report_path=report_path,
        plots_dir=output_dirs["plots"],
    )

    if verbose:
        print_report(payload)
        print(f"\n[已保存] {out_path}")
        print(f"[报告]  {report_path}")
        print(f"[清单]  {manifest_path}")

    return payload
