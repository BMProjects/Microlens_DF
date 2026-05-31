"""基于 python-docx 的 CNAS 测试报告 .docx 生成。

设计要点：
- 不依赖 .docx 模板文件（保持仓库干净），通过代码直接构造文档
- 中文字体回退到 SimSun / DejaVu Sans
- 包含与 HTML 报告一致的章节结构、指标表、截图、溯源
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cnas_test.runner.config import (
    DATASET_CONSTRUCTION_METHOD,
    DATASET_NAME,
    DATASET_SUMMARY,
    MODEL_DESCRIPTION,
    RECOMMENDED_COMMAND,
    SOFTWARE_NAME,
    SOFTWARE_VERSION,
    STANDARD_SAMPLE_UNIT,
    VERSION_NOTE,
)


def _set_cell(cell, text: str) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.font.name = "Microsoft YaHei"


def _add_kv_table(doc, rows: list[tuple[str, str]]) -> None:
    table = doc.add_table(rows=len(rows), cols=2)
    table.style = "Light Grid"
    for idx, (k, v) in enumerate(rows):
        _set_cell(table.rows[idx].cells[0], k)
        _set_cell(table.rows[idx].cells[1], v)


def _add_image_safe(doc, path: Path, caption: str, width_in: float = 6.0) -> None:
    from docx.shared import Inches

    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
    else:
        doc.add_paragraph(f"[缺图] {caption} — 期望路径：{path}")
    p = doc.add_paragraph(caption)
    p.runs[0].italic = True


def build_docx_report(
    *,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    screenshots: dict[str, Path],
    plots_dir: Path,
    test_set_path: Path,
    weights_path: Path,
    save_path: Path,
) -> Path:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx 未安装。请执行：uv sync --extra ml --extra dev") from exc

    doc = Document()
    metrics = payload["metrics"]
    dataset = payload["dataset"]
    dist = dataset["class_counts"]
    git = provenance.get("git", {})
    weights_info = provenance.get("weights", {})
    test_set_info = provenance.get("test_set", {})
    env = provenance.get("environment", {})

    doc.add_heading(f"{SOFTWARE_NAME} CNAS 测试报告", level=0)
    doc.add_paragraph(
        f"第三方测试结果记录。测试时间：{payload['test_timestamp']}　|　"
        f"mAP50 = {metrics['mAP50']:.4f}"
    )
    doc.add_paragraph(
        "本报告用于客观呈现测试环境、测试数据、测试过程、复现条件和实测指标。"
        "符合性判定依据以委托测试文件或测试机构正式规则为准。"
    )

    doc.add_heading("1. 测试概况", level=1)
    _add_kv_table(
        doc,
        [
            ("被测软件名称", SOFTWARE_NAME),
            ("测试版本", SOFTWARE_VERSION),
            ("版本备注", VERSION_NOTE),
            ("模型描述", MODEL_DESCRIPTION),
            ("标准命令", RECOMMENDED_COMMAND),
            ("测试集清单", str(test_set_path)),
            ("测试集 SHA256", test_set_info.get("sha256") or "-"),
            ("模型权重", str(weights_path)),
            ("权重 SHA256", weights_info.get("sha256") or "-"),
            ("权重大小（字节）", str(weights_info.get("size_bytes", "-"))),
        ],
    )

    doc.add_heading("2. 数据集说明", level=1)
    _add_kv_table(
        doc,
        [
            ("数据集名称", DATASET_NAME),
            ("数据集类型", f"{STANDARD_SAMPLE_UNIT}数据集"),
            ("样本构建方法", DATASET_CONSTRUCTION_METHOD),
            ("样本图像尺寸", "640 × 640 像素"),
            ("样本图像数量", str(dataset["tiles"])),
            ("实际参与评测样本数", str(payload["n_tiles"])),
            ("标注缺陷框数量", str(dataset["boxes"])),
            ("空背景样本数", str(dataset["background_tiles"])),
            ("类别分布 scratch", str(dist["scratch"])),
            ("类别分布 spot", str(dist["spot"])),
            ("类别分布 critical", str(dist["critical"])),
        ],
    )

    doc.add_heading("3. 数据划分与执行过程", level=1)
    doc.add_paragraph(
        f"单次测试使用固定模型权重和经确认的测试集清单，纳入全部 {payload['n_tiles']} "
        "个标准化图像样本进行评测。测试程序自动生成数据清单、评测日志、过程截图、"
        "指标结果、溯源文件和本报告。"
    )
    doc.add_paragraph(
        "训练后测试先按既定训练配置重新训练模型，再使用训练输出的 weights/best.pt "
        "按单次测试流程评测。训练阶段使用下表固定划分，最终报告指标以测试命令对"
        "全量确认样本重新计算得到的结果为准。"
    )
    split_table = doc.add_table(rows=3, cols=5)
    split_table.style = "Light Grid"
    split_rows = [
        ("子集", "图像编号数", "标准化图像样本数", "标注缺陷框数", "用途"),
        (
            "训练",
            str(DATASET_SUMMARY["train"]["images"]),
            str(DATASET_SUMMARY["train"]["tiles"]),
            str(DATASET_SUMMARY["train"]["boxes"]),
            "参数学习",
        ),
        (
            "验证",
            str(DATASET_SUMMARY["val"]["images"]),
            str(DATASET_SUMMARY["val"]["tiles"]),
            str(DATASET_SUMMARY["val"]["boxes"]),
            "训练过程监控与模型选择",
        ),
    ]
    for row_idx, row_values in enumerate(split_rows):
        for col_idx, value in enumerate(row_values):
            _set_cell(split_table.rows[row_idx].cells[col_idx], value)

    doc.add_heading("4. 评测参数", level=1)
    _add_kv_table(
        doc,
        [
            ("置信度阈值", str(payload["eval_conf"])),
            ("NMS IoU 阈值", str(payload["eval_iou"])),
            ("AP 匹配 IoU", "0.50"),
        ],
    )

    doc.add_heading("5. 结果计算方法", level=1)
    doc.add_paragraph(
        "对每个类别 c，预测框按置信度从高到低排序，并在 IoU = 0.5 "
        "条件下与同类别标注框进行一对一匹配。"
    )
    for formula in [
        "Precision_c(k) = TP_c(k) / [TP_c(k) + FP_c(k)]",
        "Recall_c(k) = TP_c(k) / N_gt,c",
        "AP50_c = integral_0^1 P_c(R)dR, IoU = 0.5",
        "mAP@0.5 = (1 / C) * sum_{c=1..C} AP50_c, C = 3",
    ]:
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(formula)
        run.font.name = "Cambria Math"

    doc.add_heading("6. 测试结果", level=1)
    table = doc.add_table(rows=8, cols=2)
    table.style = "Light Grid"
    rows = [
        ("scratch AP@0.5", f"{metrics['per_class_AP50']['scratch']:.4f}"),
        ("spot AP@0.5", f"{metrics['per_class_AP50']['spot']:.4f}"),
        ("critical AP@0.5", f"{metrics['per_class_AP50']['critical']:.4f}"),
        ("mAP@0.5（主指标）", f"{metrics['mAP50']:.4f}"),
        ("mAP@0.5:0.95", f"{metrics['mAP50_95']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("耗时（秒）", f"{payload['elapsed_seconds']:.1f}"),
    ]
    for idx, (k, v) in enumerate(rows):
        _set_cell(table.rows[idx].cells[0], k)
        _set_cell(table.rows[idx].cells[1], v)

    doc.add_heading("7. 测试过程截图", level=1)
    _add_image_safe(doc, screenshots.get("startup", Path("-")), "截图 1：测试启动确认横幅")
    _add_image_safe(doc, screenshots.get("result_text", Path("-")), "截图 2：测试结果终端输出")
    _add_image_safe(
        doc, screenshots.get("metrics_chart", Path("-")), "截图 3：AP50 与 mAP50 条形图"
    )

    doc.add_heading("8. 评测产物图", level=1)
    ult = plots_dir / "ultralytics_output"
    _add_image_safe(doc, ult / "BoxPR_curve.png", "PR 曲线（BoxPR_curve）")
    _add_image_safe(doc, ult / "confusion_matrix_normalized.png", "归一化混淆矩阵")
    for i in range(3):
        _add_image_safe(doc, ult / f"val_batch{i}_pred.jpg", f"典型预测样例 val_batch{i}_pred")

    doc.add_heading("9. 测试执行溯源", level=1)
    _add_kv_table(
        doc,
        [
            ("git commit", git.get("commit", "-")),
            ("git 分支", git.get("branch", "-")),
            ("工作区状态", "有未提交修改" if git.get("dirty") == "true" else "干净"),
            ("主机名", env.get("hostname", "-")),
            ("操作系统", env.get("platform", "-")),
            ("Python", env.get("python_version", "-")),
            ("GPU", provenance.get("gpu", "-")),
            ("开始时间", provenance.get("started_at_iso", "-")),
            ("结束时间", provenance.get("finished_at_iso", "-")),
            ("总耗时（秒）", str(provenance.get("duration_seconds", "-"))),
        ],
    )

    doc.add_heading("10. 结论", level=1)
    doc.add_paragraph(
        f"本次第三方测试完成，主指标实测值为 mAP@0.5 = {metrics['mAP50']:.4f}。"
        "本报告仅记录实测结果；符合性判定以委托测试文件或测试机构正式规则为准。"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(save_path))
    return save_path
