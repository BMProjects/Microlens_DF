#!/usr/bin/env python3
"""分割实验结果分析与私有图像可视化核查.

输出内容:
1. MSD 源域预训练曲线图与结果汇总
2. 私有弱标签微调 loss 曲线与结果汇总
3. 20~30 张私有图像的三模型并列可视化
4. 面向标注更新和物理量化的详细指标 CSV / JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.measurement import DEFAULT_CALIBRATION
from darkfield_defects.ml.predict import load_model, predict_full_image


PHASE3_ROOT = PROJECT_ROOT / "output/experiments/phase3_segmentation"
ANALYSIS_ROOT = PHASE3_ROOT / "analysis"
DOC_ASSETS_ROOT = PROJECT_ROOT / "doc/assets/generated"
PIXEL_SIZE_MM = DEFAULT_CALIBRATION.pixel_size_mm
CLASS_NAMES = ["background", "scratch", "spot", "damage"]
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 80, 80),
    2: (90, 210, 255),
    3: (255, 210, 60),
}


@dataclass(slots=True)
class ModelRun:
    key: str
    display_name: str
    checkpoint: Path
    history: Path
    log_path: Path | None = None


SOURCE_MODELS = [
    ModelRun(
        "light_unet",
        "LightUNet",
        PHASE3_ROOT / "model_zoo/light_unet_baseline/best.pt",
        PHASE3_ROOT / "model_zoo/light_unet_baseline/history.json",
        PHASE3_ROOT / "logs/light_unet_baseline.log",
    ),
    ModelRun(
        "unetplusplus",
        "Unet++",
        PHASE3_ROOT / "model_zoo/unetplusplus_r34/best.pt",
        PHASE3_ROOT / "model_zoo/unetplusplus_r34/history.json",
        PHASE3_ROOT / "logs/unetplusplus_r34.log",
    ),
    ModelRun(
        "deeplabv3plus",
        "DeepLabV3+",
        PHASE3_ROOT / "model_zoo/deeplabv3plus_r34/best.pt",
        PHASE3_ROOT / "model_zoo/deeplabv3plus_r34/history.json",
        PHASE3_ROOT / "logs/deeplabv3plus_r34.log",
    ),
    ModelRun(
        "fpn",
        "FPN",
        PHASE3_ROOT / "model_zoo/fpn_r34/best.pt",
        PHASE3_ROOT / "model_zoo/fpn_r34/history.json",
        PHASE3_ROOT / "logs/fpn_r34.log",
    ),
]


PRIVATE_MODELS = [
    ModelRun(
        "unetplusplus",
        "Unet++",
        PHASE3_ROOT / "private_finetuned_unetplusplus_r34/best.pt",
        PHASE3_ROOT / "private_finetuned_unetplusplus_r34/history.json",
        PHASE3_ROOT / "logs/private_finetuned_unetplusplus_r34.log",
    ),
    ModelRun(
        "fpn",
        "FPN",
        PHASE3_ROOT / "private_finetuned_fpn_r34/best.pt",
        PHASE3_ROOT / "private_finetuned_fpn_r34/history.json",
        PHASE3_ROOT / "logs/private_finetuned_fpn_r34.log",
    ),
    ModelRun(
        "deeplabv3plus",
        "DeepLabV3+",
        PHASE3_ROOT / "private_finetuned_deeplabv3plus_r34/best.pt",
        PHASE3_ROOT / "private_finetuned_deeplabv3plus_r34/history.json",
        PHASE3_ROOT / "logs/private_finetuned_deeplabv3plus_r34.log",
    ),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_quick_val_metrics(log_path: Path | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if log_path is None or not log_path.exists():
        return metrics

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pattern = re.compile(r"^\s*(scratch|spot|damage)\s*:\s*([0-9.]+)")
    miou_pattern = re.compile(r"^\s*mIoU:\s*([0-9.]+)")
    for line in lines[-40:]:
        m = pattern.search(line)
        if m:
            metrics[f"{m.group(1)}_iou"] = float(m.group(2))
        m2 = miou_pattern.search(line)
        if m2:
            metrics["miou_quick"] = float(m2.group(1))
    return metrics


def summarize_source_history(model: ModelRun) -> dict[str, float | int | str]:
    history = read_json(model.history)
    best = max(history, key=lambda x: x.get("val_miou", -1.0))
    final = history[-1]
    summary = {
        "model": model.display_name,
        "epochs_ran": len(history),
        "best_epoch": best["epoch"],
        "best_val_miou": float(best["val_miou"]),
        "best_loss": float(best["loss"]),
        "final_train_miou": float(final["train_miou"]),
        "final_val_miou": float(final["val_miou"]),
    }
    summary.update(parse_quick_val_metrics(model.log_path))
    return summary


def summarize_private_history(model: ModelRun) -> dict[str, float | int | str]:
    history = read_json(model.history)
    best = min(history, key=lambda x: x.get("loss", 1e9))
    final = history[-1]
    return {
        "model": model.display_name,
        "epochs_ran": len(history),
        "best_epoch": best["epoch"],
        "best_loss": float(best["loss"]),
        "final_loss": float(final["loss"]),
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_source_curves(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 6))
    for model in SOURCE_MODELS:
        history = read_json(model.history)
        epochs = [x["epoch"] for x in history]
        vals = [x["val_miou"] for x in history]
        plt.plot(epochs, vals, label=model.display_name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.title("MSD Source-Domain Segmentation Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_private_curves(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 6))
    for model in [
        ModelRun(
            "light_unet",
            "LightUNet",
            PHASE3_ROOT / "private_finetuned_light_unet/best.pt",
            PHASE3_ROOT / "private_finetuned_light_unet/history.json",
            PHASE3_ROOT / "logs/private_finetuned_light_unet.log",
        ),
        *PRIVATE_MODELS,
    ]:
        history = read_json(model.history)
        epochs = [x["epoch"] for x in history]
        vals = [x["loss"] for x in history]
        plt.plot(epochs, vals, label=model.display_name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Private Weak-Label Finetuning Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.stack([image] * 3, axis=-1)
    color = colorize_mask(mask)
    overlay = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)
    overlay[mask == 0] = base[mask == 0]
    return overlay


def compute_iou(true_mask: np.ndarray, pred_mask: np.ndarray, cls_id: int) -> float:
    true_bin = true_mask == cls_id
    pred_bin = pred_mask == cls_id
    union = np.logical_or(true_bin, pred_bin).sum()
    if union == 0:
        return float("nan")
    inter = np.logical_and(true_bin, pred_bin).sum()
    return float(inter / union)


def compute_dice(true_mask: np.ndarray, pred_mask: np.ndarray, cls_id: int) -> float:
    true_bin = true_mask == cls_id
    pred_bin = pred_mask == cls_id
    denom = true_bin.sum() + pred_bin.sum()
    if denom == 0:
        return float("nan")
    inter = np.logical_and(true_bin, pred_bin).sum()
    return float(2.0 * inter / denom)


def skeleton_length_px(binary_mask: np.ndarray) -> float:
    if not np.any(binary_mask):
        return 0.0
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skel = cv2.ximgproc.thinning(binary_mask.astype(np.uint8) * 255)
        return float(np.count_nonzero(skel))
    from skimage.morphology import skeletonize

    skel = skeletonize(binary_mask > 0)
    return float(np.count_nonzero(skel))


def connected_components(binary_mask: np.ndarray) -> int:
    if not np.any(binary_mask):
        return 0
    n, _, _, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8))
    return int(max(n - 1, 0))


def physical_stats(mask: np.ndarray) -> dict[str, float]:
    scratch = mask == 1
    spot = mask == 2
    damage = mask == 3
    scratch_area_px = float(np.count_nonzero(scratch))
    spot_area_px = float(np.count_nonzero(spot))
    damage_area_px = float(np.count_nonzero(damage))
    scratch_length_px = skeleton_length_px(scratch.astype(np.uint8))
    scratch_width_px = scratch_area_px / max(scratch_length_px, 1.0) if scratch_area_px > 0 else 0.0
    return {
        "scratch_area_mm2": scratch_area_px * (PIXEL_SIZE_MM ** 2),
        "spot_area_mm2": spot_area_px * (PIXEL_SIZE_MM ** 2),
        "damage_area_mm2": damage_area_px * (PIXEL_SIZE_MM ** 2),
        "scratch_length_mm": scratch_length_px * PIXEL_SIZE_MM,
        "scratch_avg_width_mm": scratch_width_px * PIXEL_SIZE_MM,
        "scratch_components": float(connected_components(scratch.astype(np.uint8))),
        "spot_components": float(connected_components(spot.astype(np.uint8))),
        "damage_components": float(connected_components(damage.astype(np.uint8))),
    }


def disagreement_map(pred_masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(pred_masks, axis=0)
    agreement = np.all(stack == stack[:1], axis=0)
    out = np.zeros((*agreement.shape, 3), dtype=np.uint8)
    out[~agreement] = (255, 0, 255)
    return out


def pairwise_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def score_for_selection(mask_path: Path) -> tuple[float, float, float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    scratch = float(np.count_nonzero(mask == 1))
    spot = float(np.count_nonzero(mask == 2))
    damage = float(np.count_nonzero(mask == 3))
    total = scratch + spot + damage
    mixed = sum(v > 0 for v in [scratch, spot, damage])
    return mixed, total, scratch, spot + damage


def select_representative_images(image_dir: Path, mask_dir: Path, sample_count: int) -> list[str]:
    names = [p.name for p in sorted(image_dir.glob("*.png")) if (mask_dir / p.name).exists()]
    scratch_rank = sorted(names, key=lambda n: score_for_selection(mask_dir / n)[2], reverse=True)
    mix_rank = sorted(names, key=lambda n: score_for_selection(mask_dir / n), reverse=True)
    spot_damage_rank = sorted(names, key=lambda n: score_for_selection(mask_dir / n)[3], reverse=True)

    ordered: list[str] = []
    for candidate_list, quota in [
        (mix_rank, max(sample_count // 3, 8)),
        (scratch_rank, max(sample_count // 3, 8)),
        (spot_damage_rank, max(sample_count // 3, 8)),
    ]:
        for name in candidate_list:
            if name not in ordered:
                ordered.append(name)
            if len(ordered) >= quota and len(ordered) >= sample_count:
                break

    if len(ordered) < sample_count:
        for name in names:
            if name not in ordered:
                ordered.append(name)
            if len(ordered) >= sample_count:
                break
    return ordered[:sample_count]


def aggregate_metric(rows: list[dict], key: str) -> float:
    vals = [row[key] for row in rows if not math.isnan(float(row[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def save_panel(
    image_name: str,
    image: np.ndarray,
    weak_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    sample_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    panels = [
        ("Input", np.stack([image] * 3, axis=-1)),
        ("Weak Label", overlay_mask(image, weak_mask)),
        (
            f"Unet++\nIoU(s)={sample_metrics['unetplusplus']['scratch_iou']:.3f}",
            overlay_mask(image, predictions["unetplusplus"]),
        ),
        (
            f"FPN\nIoU(s)={sample_metrics['fpn']['scratch_iou']:.3f}",
            overlay_mask(image, predictions["fpn"]),
        ),
        (
            f"DeepLabV3+\nIoU(s)={sample_metrics['deeplabv3plus']['scratch_iou']:.3f}",
            overlay_mask(image, predictions["deeplabv3plus"]),
        ),
        ("Disagreement", disagreement_map(list(predictions.values()))),
    ]
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(image_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    source_rows: list[dict],
    private_rows: list[dict],
    aggregate_rows: list[dict],
    sample_count: int,
) -> None:
    ensure_dir(output_path.parent)
    lines = [
        "# 分割实验分析输出",
        "",
        "## 源域预训练概览",
        "",
        "| 模型 | 运行轮数 | 最佳 epoch | 最佳 val_mIoU | scratch IoU | spot IoU | quick mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['model']} | {row['epochs_ran']} | {row['best_epoch']} | "
            f"{row['best_val_miou']:.4f} | {row.get('scratch_iou', float('nan')):.4f} | "
            f"{row.get('spot_iou', float('nan')):.4f} | {row.get('miou_quick', float('nan')):.4f} |"
        )

    lines += [
        "",
        "## 私有弱标签微调概览",
        "",
        "| 模型 | 运行轮数 | 最佳 epoch | 最佳 loss | 最终 loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in private_rows:
        lines.append(
            f"| {row['model']} | {row['epochs_ran']} | {row['best_epoch']} | "
            f"{row['best_loss']:.4f} | {row['final_loss']:.4f} |"
        )

    lines += [
        "",
        f"## {sample_count} 张私有图像人工核查汇总",
        "",
        "| 模型 | scratch IoU | scratch Dice | spot IoU | damage IoU | consensus ratio | scratch length error | scratch area error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['model']} | {row['scratch_iou_mean']:.4f} | {row['scratch_dice_mean']:.4f} | "
            f"{row['spot_iou_mean']:.4f} | {row['damage_iou_mean']:.4f} | {row['consensus_ratio_mean']:.4f} | "
            f"{row['scratch_length_error_mean']:.4f} | {row['scratch_area_error_mean']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_error(pred: float, ref: float) -> float:
    if ref <= 1e-9:
        return 0.0 if pred <= 1e-9 else 1.0
    return abs(pred - ref) / ref


def main() -> None:
    parser = argparse.ArgumentParser(description="分析分割实验结果并生成可视化核查产物")
    parser.add_argument("--sample-count", type=int, default=24, help="人工核查样本数量")
    parser.add_argument("--device", default=None, help="推理设备，默认自动选择")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_ROOT / "segmentation_review_20260325",
        help="输出目录",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    panels_dir = analysis_dir / "panels"
    ensure_dir(analysis_dir)
    ensure_dir(DOC_ASSETS_ROOT)

    source_rows = [summarize_source_history(model) for model in SOURCE_MODELS]
    private_rows = [
        summarize_private_history(
            ModelRun(
                "light_unet",
                "LightUNet",
                PHASE3_ROOT / "private_finetuned_light_unet/best.pt",
                PHASE3_ROOT / "private_finetuned_light_unet/history.json",
                PHASE3_ROOT / "logs/private_finetuned_light_unet.log",
            )
        ),
        *[summarize_private_history(model) for model in PRIVATE_MODELS],
    ]

    plot_source_curves(DOC_ASSETS_ROOT / "segmentation_source_miou_curves.png")
    plot_private_curves(DOC_ASSETS_ROOT / "segmentation_private_loss_curves.png")

    image_dir = PHASE3_ROOT / "private_weak_masks/images"
    mask_dir = PHASE3_ROOT / "private_weak_masks/masks"
    selected_names = select_representative_images(image_dir, mask_dir, args.sample_count)
    (analysis_dir / "selected_samples.json").write_text(
        json.dumps(selected_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded_models: dict[str, tuple] = {}
    for model in PRIVATE_MODELS:
        loaded_models[model.key] = load_model(model.checkpoint, device=args.device)

    all_sample_rows: list[dict] = []
    aggregate_by_model: dict[str, list[dict]] = {model.key: [] for model in PRIVATE_MODELS}

    panel_paths: list[Path] = []
    for image_name in selected_names:
        image = cv2.imread(str(image_dir / image_name), cv2.IMREAD_GRAYSCALE)
        weak_mask = cv2.imread(str(mask_dir / image_name), cv2.IMREAD_GRAYSCALE)
        if image is None or weak_mask is None:
            continue

        predictions: dict[str, np.ndarray] = {}
        for model in PRIVATE_MODELS:
            net, device = loaded_models[model.key]
            predictions[model.key] = predict_full_image(net, image, device, patch_size=512, overlap=64)

        sample_metrics: dict[str, dict[str, float]] = {}
        weak_physical = physical_stats(weak_mask)
        pair_disagreements = {
            "unet_vs_fpn": pairwise_disagreement(predictions["unetplusplus"], predictions["fpn"]),
            "unet_vs_deeplab": pairwise_disagreement(predictions["unetplusplus"], predictions["deeplabv3plus"]),
            "fpn_vs_deeplab": pairwise_disagreement(predictions["fpn"], predictions["deeplabv3plus"]),
        }
        consensus_ratio = 1.0 - float(np.mean(disagreement_map(list(predictions.values())).sum(axis=2) > 0))

        for model in PRIVATE_MODELS:
            pred = predictions[model.key]
            pred_physical = physical_stats(pred)
            row = {
                "image": image_name,
                "model": model.display_name,
                "scratch_iou": compute_iou(weak_mask, pred, 1),
                "scratch_dice": compute_dice(weak_mask, pred, 1),
                "spot_iou": compute_iou(weak_mask, pred, 2),
                "damage_iou": compute_iou(weak_mask, pred, 3),
                "pixel_accuracy": float(np.mean(weak_mask == pred)),
                "consensus_ratio": consensus_ratio,
                "unet_vs_fpn": pair_disagreements["unet_vs_fpn"],
                "unet_vs_deeplab": pair_disagreements["unet_vs_deeplab"],
                "fpn_vs_deeplab": pair_disagreements["fpn_vs_deeplab"],
                "weak_scratch_length_mm": weak_physical["scratch_length_mm"],
                "pred_scratch_length_mm": pred_physical["scratch_length_mm"],
                "weak_scratch_area_mm2": weak_physical["scratch_area_mm2"],
                "pred_scratch_area_mm2": pred_physical["scratch_area_mm2"],
                "weak_spot_area_mm2": weak_physical["spot_area_mm2"],
                "pred_spot_area_mm2": pred_physical["spot_area_mm2"],
                "weak_damage_area_mm2": weak_physical["damage_area_mm2"],
                "pred_damage_area_mm2": pred_physical["damage_area_mm2"],
                "weak_scratch_components": weak_physical["scratch_components"],
                "pred_scratch_components": pred_physical["scratch_components"],
                "scratch_length_error": relative_error(
                    pred_physical["scratch_length_mm"], weak_physical["scratch_length_mm"]
                ),
                "scratch_area_error": relative_error(
                    pred_physical["scratch_area_mm2"], weak_physical["scratch_area_mm2"]
                ),
                "spot_area_error": relative_error(
                    pred_physical["spot_area_mm2"], weak_physical["spot_area_mm2"]
                ),
                "damage_area_error": relative_error(
                    pred_physical["damage_area_mm2"], weak_physical["damage_area_mm2"]
                ),
            }
            all_sample_rows.append(row)
            aggregate_by_model[model.key].append(row)
            sample_metrics[model.key] = row

        panel_path = panels_dir / f"{Path(image_name).stem}_compare.png"
        save_panel(image_name, image, weak_mask, predictions, sample_metrics, panel_path)
        panel_paths.append(panel_path)

    summary_rows: list[dict] = []
    for model in PRIVATE_MODELS:
        rows = aggregate_by_model[model.key]
        summary_rows.append(
            {
                "model": model.display_name,
                "scratch_iou_mean": aggregate_metric(rows, "scratch_iou"),
                "scratch_dice_mean": aggregate_metric(rows, "scratch_dice"),
                "spot_iou_mean": aggregate_metric(rows, "spot_iou"),
                "damage_iou_mean": aggregate_metric(rows, "damage_iou"),
                "pixel_accuracy_mean": aggregate_metric(rows, "pixel_accuracy"),
                "consensus_ratio_mean": aggregate_metric(rows, "consensus_ratio"),
                "scratch_length_error_mean": aggregate_metric(rows, "scratch_length_error"),
                "scratch_area_error_mean": aggregate_metric(rows, "scratch_area_error"),
                "spot_area_error_mean": aggregate_metric(rows, "spot_area_error"),
                "damage_area_error_mean": aggregate_metric(rows, "damage_area_error"),
            }
        )

    save_csv(
        analysis_dir / "source_summary.csv",
        source_rows,
        [
            "model",
            "epochs_ran",
            "best_epoch",
            "best_val_miou",
            "best_loss",
            "final_train_miou",
            "final_val_miou",
            "scratch_iou",
            "spot_iou",
            "damage_iou",
            "miou_quick",
        ],
    )
    save_csv(
        analysis_dir / "private_finetune_summary.csv",
        private_rows,
        ["model", "epochs_ran", "best_epoch", "best_loss", "final_loss"],
    )
    save_csv(
        analysis_dir / "private_review_metrics.csv",
        all_sample_rows,
        list(all_sample_rows[0].keys()) if all_sample_rows else ["image", "model"],
    )
    save_csv(
        analysis_dir / "private_review_aggregate.csv",
        summary_rows,
        list(summary_rows[0].keys()) if summary_rows else ["model"],
    )
    (analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "source_summary": source_rows,
                "private_finetune_summary": private_rows,
                "private_review_aggregate": summary_rows,
                "selected_samples": selected_names,
                "panel_count": len(panel_paths),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_markdown_report(
        analysis_dir / "README.md",
        source_rows,
        private_rows,
        summary_rows,
        len(selected_names),
    )


if __name__ == "__main__":
    main()
