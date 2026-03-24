"""Typer CLI — 暗场镜片缺陷检测与磨损评估工具."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer(
    name="darkfield-defects",
    help="暗场显微镜离焦微结构镜片缺陷检测与磨损评估系统",
    add_completion=False,
)


@app.command()
def detect(
    input_path: str = typer.Argument(..., help="输入图像或目录路径"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="输出目录"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML 配置文件路径"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="最大处理图像数"),
    no_report: bool = typer.Option(False, "--no-report", help="不生成 HTML 报告"),
    calibration: str = typer.Option(..., "--calibration", "--cal", help="标定目录"),
) -> None:
    """检测镜片划痕并生成评估报告."""
    from darkfield_defects.data.loader import load_image, scan_directory
    from darkfield_defects.detection.classical import ClassicalDetector
    from darkfield_defects.detection.params import load_params
    from darkfield_defects.detection.rendering import export_coco, save_detection_output
    from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
    from darkfield_defects.scoring.quantify import compute_wear_metrics
    from darkfield_defects.scoring.report import generate_html_report, generate_json_report
    from darkfield_defects.scoring.wear_score import compute_wear_score

    params = load_params(config)
    detector = ClassicalDetector(params.detection, params.preprocess, params.scoring)
    adv_pipeline = PreprocessPipeline.from_roi_pipeline_params(params.roi_pipeline)
    adv_pipeline.load_calibration(calibration)

    # 确定输入文件
    inp = Path(input_path)
    if inp.is_file():
        image_files = [(inp.name, inp)]
    elif inp.is_dir():
        infos = scan_directory(inp)
        image_files = [(info.filename, info.path) for info in infos if info.image_type.value != "background"]
    else:
        typer.echo(f"❌ 路径无效: {input_path}", err=True)
        raise typer.Exit(1)

    if limit:
        image_files = image_files[:limit]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco_results = []

    typer.echo(f"📷 处理 {len(image_files)} 张图像...")

    for i, (fname, fpath) in enumerate(image_files, 1):
        typer.echo(f"[{i}/{len(image_files)}] {fname}")

        try:
            img = load_image(fpath)
            
            res = adv_pipeline.process(img, frame_index=i)
            result = detector.detect(
                img,
                roi_mask=res.roi_mask,
                preprocessed_image=res.image_final,
            )

            # 保存检测结果
            stem = Path(fname).stem
            saved = save_detection_output(img, result, out_dir, stem)

            # 磨损评分
            roi_mask = result.metadata.get("roi_mask")
            metrics = compute_wear_metrics(result, roi_mask)
            assessment = compute_wear_score(metrics, params.scoring)

            # 报告
            generate_json_report(fname, metrics, assessment, out_dir / f"{stem}_report.json")

            if not no_report:
                overlay_rel = f"{stem}_overlay.png" if "overlay" in saved else None
                generate_html_report(fname, metrics, assessment, out_dir / f"{stem}_report.html", overlay_rel)

            coco_results.append((fname, result))

            typer.echo(f"  → Score={assessment.score:.1f} Grade={assessment.grade} ({result.num_scratches} scratches)")

        except Exception as e:
            logger.error(f"处理失败 {fname}: {e}")
            typer.echo(f"  ⚠️ 失败: {e}", err=True)

    # 导出 COCO 标注
    if coco_results:
        export_coco(coco_results, out_dir / "annotations.json")

    typer.echo(f"\n✅ 完成！结果保存在: {out_dir}")


@app.command()
def preprocess(
    input_path: str = typer.Argument(..., help="输入图像路径"),
    output_dir: str = typer.Option("./output/preprocess", "--output", "-o"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    calibration: str = typer.Option(..., "--calibration", "--cal", help="标定结果目录"),
) -> None:
    """执行模板背景预处理并导出结果."""
    import cv2

    from darkfield_defects.data.loader import load_image
    from darkfield_defects.detection.params import load_params
    from darkfield_defects.preprocessing.pipeline import PreprocessPipeline

    params = load_params(config)
    img = load_image(input_path)
    pipeline = PreprocessPipeline.from_roi_pipeline_params(params.roi_pipeline)
    pipeline.load_calibration(calibration)
    prep = pipeline.process(img)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    cv2.imwrite(str(out / f"{stem}_original.png"), img)
    cv2.imwrite(str(out / f"{stem}_corrected.png"), prep.image_final)
    cv2.imwrite(str(out / f"{stem}_roi.png"), (prep.roi_mask.astype("uint8") * 255))

    typer.echo(f"✅ 预处理结果保存在: {out}")
    typer.echo(f"  配准分数: {prep.quality.reg_ecc:.4f}")
    typer.echo(f"  修正参数: gain={prep.quality.correction_gain:.4f}, bias={prep.quality.correction_bias:.2f}")


@app.command()
def info(
    input_path: str = typer.Argument(..., help="数据目录路径"),
) -> None:
    """显示数据集统计信息."""
    from darkfield_defects.data.loader import scan_directory

    images = scan_directory(input_path)

    type_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}

    for img in images:
        t = img.image_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
        side_counts[img.lens_side.value] = side_counts.get(img.lens_side.value, 0) + 1
        b = img.batch or "(root)"
        batch_counts[b] = batch_counts.get(b, 0) + 1

    typer.echo(f"\n📁 数据集: {input_path}")
    typer.echo(f"   总图像数: {len(images)}")
    typer.echo(f"\n   类型分布:")
    for t, n in sorted(type_counts.items()):
        typer.echo(f"     {t}: {n}")
    typer.echo(f"\n   左右分布:")
    for s, n in sorted(side_counts.items()):
        typer.echo(f"     {s}: {n}")
    typer.echo(f"\n   批次分布:")
    for b, n in sorted(batch_counts.items()):
        typer.echo(f"     {b}: {n}")


@app.command(name="eval")
def evaluate(
    pred_dir: str = typer.Argument(..., help="预测掩码目录（PNG 二值图）"),
    gt_dir: str = typer.Argument(..., help="Ground Truth 掩码目录（PNG 二值图）"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出 JSON 路径"),
    iou_threshold: float = typer.Option(0.3, "--iou-threshold", help="实例匹配 IoU 阈值"),
) -> None:
    """评估检测结果：像素级 + 实例级指标."""
    import json

    import cv2

    from darkfield_defects.eval import (
        compute_instance_metrics,
        compute_segmentation_metrics,
    )

    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    if not pred_path.is_dir():
        typer.echo(f"❌ 预测目录不存在: {pred_dir}", err=True)
        raise typer.Exit(1)
    if not gt_path.is_dir():
        typer.echo(f"❌ GT 目录不存在: {gt_dir}", err=True)
        raise typer.Exit(1)

    # 查找匹配的文件对
    pred_files = {p.stem: p for p in pred_path.glob("*.png")}
    gt_files = {p.stem: p for p in gt_path.glob("*.png")}
    common = sorted(set(pred_files) & set(gt_files))

    if not common:
        typer.echo("⚠️ 未找到匹配的预测-GT 文件对", err=True)
        raise typer.Exit(1)

    typer.echo(f"📊 评估 {len(common)} 对文件 (IoU 阈值={iou_threshold})\n")

    all_seg = []
    all_inst = []

    for stem in common:
        pred_img = cv2.imread(str(pred_files[stem]), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(str(gt_files[stem]), cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            typer.echo(f"  ⚠️ 跳过 {stem}: 读取失败", err=True)
            continue

        seg = compute_segmentation_metrics(pred_img, gt_img)
        inst = compute_instance_metrics(pred_img, gt_img, iou_threshold)

        all_seg.append(seg)
        all_inst.append(inst)

        typer.echo(
            f"  {stem}: P={seg.precision:.3f} R={seg.recall:.3f} "
            f"F1={seg.f1:.3f} IoU={seg.iou:.3f} | "
            f"Hit={inst.hit_rate:.2f} Miss={inst.miss_rate:.2f} FP={inst.false_positives_per_image:.0f}"
        )

    # 汇总平均
    n = len(all_seg)
    if n > 0:
        avg_seg = {
            "precision": sum(s.precision for s in all_seg) / n,
            "recall": sum(s.recall for s in all_seg) / n,
            "f1": sum(s.f1 for s in all_seg) / n,
            "iou": sum(s.iou for s in all_seg) / n,
        }
        avg_inst = {
            "hit_rate": sum(i.hit_rate for i in all_inst) / n,
            "miss_rate": sum(i.miss_rate for i in all_inst) / n,
            "fp_per_image": sum(i.false_positives_per_image for i in all_inst) / n,
        }

        typer.echo(f"\n📊 平均指标 ({n} images):")
        typer.echo(
            f"  像素级: P={avg_seg['precision']:.3f} R={avg_seg['recall']:.3f} "
            f"F1={avg_seg['f1']:.3f} IoU={avg_seg['iou']:.3f}"
        )
        typer.echo(
            f"  实例级: Hit={avg_inst['hit_rate']:.3f} Miss={avg_inst['miss_rate']:.3f} "
            f"FP/img={avg_inst['fp_per_image']:.2f}"
        )

        results = {
            "num_images": n,
            "iou_threshold": iou_threshold,
            "segmentation": {k: round(v, 4) for k, v in avg_seg.items()},
            "instance": {k: round(v, 4) for k, v in avg_inst.items()},
        }

        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            typer.echo(f"\n💾 结果保存: {out_path}")


@app.command("roi-calibrate")
def roi_calibrate(
    bg_dir: str = typer.Argument(..., help="背景图目录 (含 bg-*.png)"),
    save_dir: str = typer.Option("./output/calibration", "--save", "-s", help="标定结果保存目录"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML 配置文件"),
) -> None:
    """从背景模板生成高光 ring、ROI 和修正参考参数."""
    from darkfield_defects.detection.params import load_params
    from darkfield_defects.preprocessing.pipeline import PreprocessPipeline

    params = load_params(config)
    rp = params.roi_pipeline

    pipeline = PreprocessPipeline.from_roi_pipeline_params(rp)

    cal = pipeline.calibrate(bg_dir, save_dir)
    typer.echo(f"✅ 标定完成, 保存于 {save_dir}")
    typer.echo(f"  ROI像素数: {int(cal.roi_mask.sum())}")
    typer.echo(f"  Ring像素数: {int(cal.ring_mask.sum())}")
    typer.echo(f"  模板模糊σ: {cal.template_sigma:.1f}")
    typer.echo(f"  修正参考: median={cal.correction_ref_median:.2f}, mad={cal.correction_ref_mad:.2f}")


@app.command("roi-process")
def roi_process(
    target_dir: str = typer.Argument(..., help="目标图像目录"),
    output_dir: str = typer.Option("./output/processed", "--output", "-o", help="输出目录"),
    calibration: str = typer.Option("./output/calibration", "--calibration", "--cal", help="标定结果目录"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML 配置文件"),
) -> None:
    """批量执行模板 ring 匹配、ROI 投影和背景修正."""
    from darkfield_defects.detection.params import load_params
    from darkfield_defects.preprocessing.pipeline import PreprocessPipeline

    params = load_params(config)
    pipeline = PreprocessPipeline.from_roi_pipeline_params(params.roi_pipeline)

    pipeline.load_calibration(calibration)
    report = pipeline.process_batch(target_dir, output_dir)

    n_valid = sum(1 for r in report if r.get("valid", False))
    n_total = len(report)
    typer.echo(f"✅ 完成 {n_total} 张, 通过 {n_valid}, 异常 {n_total - n_valid}")
    typer.echo(f"  报告: {Path(output_dir) / 'quality_report.json'}")



def main() -> None:
    app()


if __name__ == "__main__":
    main()
