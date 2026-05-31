#!/usr/bin/env python3
"""统计全图弱标签复杂度并输出 L1/L2/L3 数据分层清单。"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "output/dataset_v2/images"
DEFAULT_MASK_DIR = PROJECT_ROOT / "output/experiments/phase3_segmentation/private_weak_masks/masks"
DEFAULT_ROI_MASK = PROJECT_ROOT / "output/dataset_v2/calibration/roi_mask.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments/phase3_segmentation/data_stratification"
DEFAULT_PIXEL_SIZE_MM = 0.0068


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_roi_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        roi = np.load(path)
    else:
        roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return None
    return roi > 0


def thin_mask(binary: np.ndarray) -> np.ndarray:
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if callable(thinning):
        return thinning(binary.astype(np.uint8) * 255) > 0
    from skimage.morphology import skeletonize

    return skeletonize(binary > 0)


def skeleton_branch_stats(skel: np.ndarray) -> tuple[int, int]:
    if not np.any(skel):
        return 0, 0
    img = skel.astype(np.uint8)
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    neigh = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # center=10, + n_neighbors
    n_neighbors = neigh[skel] - 10
    branch_points = int(np.count_nonzero(n_neighbors >= 3))
    intersection_points = int(np.count_nonzero(n_neighbors >= 4))
    return branch_points, intersection_points


def component_stats(binary_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    components: list[dict] = []
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        if area <= 0:
            continue
        components.append(
            {
                "label": idx,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area_px": int(area),
                "cx": float(centroids[idx][0]),
                "cy": float(centroids[idx][1]),
            }
        )
    return components, labels


def nearest_neighbor_distance_mm(components: list[dict], pixel_size_mm: float) -> float:
    if len(components) < 2:
        return 0.0
    pts = np.array([(c["cx"], c["cy"]) for c in components], dtype=np.float32)
    dists = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(np.min(dists, axis=1)) * pixel_size_mm)


def component_dispersion(components: list[dict], roi_shape: tuple[int, int], pixel_size_mm: float) -> float:
    if len(components) < 2:
        return 0.0
    pts = np.array([(c["cx"], c["cy"]) for c in components], dtype=np.float32)
    spread_px = float(np.sqrt(np.var(pts[:, 0]) + np.var(pts[:, 1])))
    h, w = roi_shape
    diag_px = max(math.hypot(w, h), 1.0)
    return (spread_px / diag_px) * (diag_px * pixel_size_mm)


def local_cluster_density(components: list[dict], pixel_size_mm: float, radius_mm: float = 0.6) -> float:
    if len(components) < 2:
        return 0.0
    pts = np.array([(c["cx"], c["cy"]) for c in components], dtype=np.float32)
    radius_px = radius_mm / max(pixel_size_mm, 1e-9)
    dists = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    neighbor_counts = np.sum((dists > 0) & (dists <= radius_px), axis=1)
    return float(np.mean(neighbor_counts))


def summarize_mask(mask: np.ndarray, roi_mask: np.ndarray | None, pixel_size_mm: float) -> dict[str, float | int]:
    defect_mask = mask > 0
    if roi_mask is not None and roi_mask.shape == defect_mask.shape:
        roi_bool = roi_mask
        defect_mask = defect_mask & roi_bool
    else:
        roi_bool = np.ones_like(defect_mask, dtype=bool)

    roi_area_px = int(np.count_nonzero(roi_bool))
    bright_area_px = int(np.count_nonzero(defect_mask))
    components, labels = component_stats(defect_mask)
    component_areas = [c["area_px"] for c in components]

    skel = thin_mask(defect_mask)
    branch_points, intersections = skeleton_branch_stats(skel)
    skeleton_length_px = int(np.count_nonzero(skel))

    if component_areas:
        largest_component_px = max(component_areas)
        mean_component_px = float(np.mean(component_areas))
        max_component_px = float(max(component_areas))
    else:
        largest_component_px = 0
        mean_component_px = 0.0
        max_component_px = 0.0

    return {
        "roi_area_px": roi_area_px,
        "bright_area_px": bright_area_px,
        "bright_area_ratio": bright_area_px / max(roi_area_px, 1),
        "n_components": len(components),
        "largest_component_ratio": largest_component_px / max(bright_area_px, 1),
        "mean_component_area_mm2": mean_component_px * (pixel_size_mm ** 2),
        "max_component_area_mm2": max_component_px * (pixel_size_mm ** 2),
        "centroid_spread_mm": component_dispersion(components, defect_mask.shape, pixel_size_mm),
        "nearest_component_distance_mm_mean": nearest_neighbor_distance_mm(components, pixel_size_mm),
        "skeleton_length_total_mm": skeleton_length_px * pixel_size_mm,
        "branch_points": branch_points,
        "intersection_count": intersections,
        "intersection_density": intersections / max(skeleton_length_px, 1),
        "local_cluster_density": local_cluster_density(components, pixel_size_mm),
        "scratch_area_mm2": float(np.count_nonzero(mask == 1) * (pixel_size_mm ** 2)),
        "spot_area_mm2": float(np.count_nonzero(mask == 2) * (pixel_size_mm ** 2)),
        "damage_area_mm2": float(np.count_nonzero(mask == 3) * (pixel_size_mm ** 2)),
    }


def quantile_threshold(rows: list[dict], key: str, q: float) -> float:
    vals = np.array([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.quantile(vals, q)) if len(vals) else 0.0


def percentile_rank(rows: list[dict], key: str, value: float) -> float:
    vals = np.array([float(row[key]) for row in rows], dtype=np.float64)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals <= value))


def compute_level_scores(row: dict, rows: list[dict]) -> tuple[float, float]:
    complexity_score = float(
        np.mean(
            [
                percentile_rank(rows, "bright_area_ratio", row["bright_area_ratio"]),
                percentile_rank(rows, "n_components", row["n_components"]),
                percentile_rank(rows, "skeleton_length_total_mm", row["skeleton_length_total_mm"]),
                percentile_rank(rows, "branch_points", row["branch_points"]),
            ]
        )
    )
    cluster_score = float(
        np.mean(
            [
                percentile_rank(rows, "largest_component_ratio", row["largest_component_ratio"]),
                percentile_rank(rows, "local_cluster_density", row["local_cluster_density"]),
            ]
        )
        - 0.35 * percentile_rank(rows, "n_components", row["n_components"])
    )
    return complexity_score, cluster_score


def assign_level(row: dict, rows: list[dict], thresholds: dict[str, dict[str, float]]) -> tuple[str, str]:
    low = thresholds["low"]
    high = thresholds["high"]
    complexity_score, cluster_score = compute_level_scores(row, rows)
    row["complexity_score"] = round(complexity_score, 4)
    row["cluster_score"] = round(cluster_score, 4)

    if complexity_score <= 0.25 and row["local_cluster_density"] <= high["local_cluster_density"]:
        return "L1", "缺陷少且分散"
    if cluster_score >= 0.40 and complexity_score <= 0.78:
        return "L2", "缺陷数量不多但聚集明显"
    if (
        row["bright_area_ratio"] >= high["bright_area_ratio"]
        or row["n_components"] >= high["n_components"]
        or row["skeleton_length_total_mm"] >= high["skeleton_length_total_mm"]
        or row["branch_points"] >= high["branch_points"]
        or complexity_score >= 0.60
    ):
        return "L3", "缺陷多或结构复杂"
    if row["bright_area_ratio"] <= low["bright_area_ratio"] and row["n_components"] <= low["n_components"]:
        return "L1", "缺陷少且分散"
    return "L2", "缺陷数量不多但聚集明显"


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(path: Path, rows: list[dict], thresholds: dict[str, dict[str, float]]) -> None:
    counts: dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
    for row in rows:
        counts[row["complexity_level"]] += 1

    lines = [
        "# 分割数据复杂度分层摘要",
        "",
        "## 分层数量",
        "",
        f"- L1: {counts['L1']} 张",
        f"- L2: {counts['L2']} 张",
        f"- L3: {counts['L3']} 张",
        "",
        "## 分位阈值",
        "",
        f"- bright_area_ratio low/high: {thresholds['low']['bright_area_ratio']:.6f} / {thresholds['high']['bright_area_ratio']:.6f}",
        f"- n_components low/high: {thresholds['low']['n_components']:.2f} / {thresholds['high']['n_components']:.2f}",
        f"- local_cluster_density high: {thresholds['high']['local_cluster_density']:.4f}",
        f"- skeleton_length_total_mm high: {thresholds['high']['skeleton_length_total_mm']:.4f}",
        "",
        "## 说明",
        "",
        "- `L1` 更适合作为高质量种子标签集。",
        "- `L2` 更适合作为 critical/crash 规则验证集。",
        "- `L3` 作为困难集和最终挑战集。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def process_one_mask(
    stem: str,
    weak_mask_dir: Path,
    roi_mask: np.ndarray | None,
    pixel_size_mm: float,
) -> dict | None:
    mask = cv2.imread(str(weak_mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    stats = summarize_mask(mask, roi_mask, pixel_size_mm)
    return {"stem": stem, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建分割复杂度分层清单")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--weak-mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_MASK)
    parser.add_argument("--pixel-size-mm", type=float, default=DEFAULT_PIXEL_SIZE_MM)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    roi_mask = load_roi_mask(args.roi_mask)

    names = [p.stem for p in sorted(args.image_dir.glob("*.png")) if (args.weak_mask_dir / f"{p.stem}.png").exists()]
    if args.max_images is not None:
        names = names[: args.max_images]
    rows: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        future_map = {
            ex.submit(process_one_mask, stem, args.weak_mask_dir, roi_mask, args.pixel_size_mm): stem
            for stem in names
        }
        for idx, fut in enumerate(futures.as_completed(future_map), start=1):
            row = fut.result()
            if row is not None:
                rows.append(row)
            if idx % 25 == 0 or idx == len(names):
                print(f"[{idx}/{len(names)}] 已完成复杂度统计")
    rows.sort(key=lambda item: item["stem"])

    if not rows:
        raise SystemExit("未找到可用图像与弱标签掩码对。")

    thresholds = {
        "low": {
            "bright_area_ratio": quantile_threshold(rows, "bright_area_ratio", 0.33),
            "n_components": quantile_threshold(rows, "n_components", 0.33),
            "intersection_density": quantile_threshold(rows, "intersection_density", 0.50),
        },
        "high": {
            "bright_area_ratio": quantile_threshold(rows, "bright_area_ratio", 0.67),
            "n_components": quantile_threshold(rows, "n_components", 0.67),
            "largest_component_ratio": quantile_threshold(rows, "largest_component_ratio", 0.67),
            "local_cluster_density": quantile_threshold(rows, "local_cluster_density", 0.67),
            "skeleton_length_total_mm": quantile_threshold(rows, "skeleton_length_total_mm", 0.67),
            "branch_points": quantile_threshold(rows, "branch_points", 0.67),
        },
    }

    levels: dict[str, list[str]] = {"L1": [], "L2": [], "L3": []}
    for row in rows:
        level, reason = assign_level(row, rows, thresholds)
        row["complexity_level"] = level
        row["reason"] = reason
        levels[level].append(row["stem"])

    write_csv(args.output_dir / "image_stats.csv", rows)
    (args.output_dir / "image_stats.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "complexity_manifest.json").write_text(
        json.dumps(
            {
                "pixel_size_mm": args.pixel_size_mm,
                "n_images": len(rows),
                "levels": levels,
                "thresholds": thresholds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for level, stems in levels.items():
        (args.output_dir / f"strata_{level.lower()}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    write_summary_report(args.output_dir / "summary_report.md", rows, thresholds)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   分割复杂度分层清单构建完成                           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  图像数: {len(rows)}")
    print(f"  输出:   {args.output_dir}")
    print(f"  L1/L2/L3: {len(levels['L1'])}/{len(levels['L2'])}/{len(levels['L3'])}")
    print()


if __name__ == "__main__":
    main()
