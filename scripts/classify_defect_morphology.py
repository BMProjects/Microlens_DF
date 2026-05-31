#!/usr/bin/env python3
"""基于分割结果的形态规则分类器。"""

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
DEFAULT_MASK_DIR = PROJECT_ROOT / "output/experiments/phase3_segmentation/private_weak_masks/masks"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "output/dataset_v2/images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments/phase3_segmentation/morphology_rules"
DEFAULT_ROI_MASK = PROJECT_ROOT / "output/dataset_v2/calibration/roi_mask.npy"
DEFAULT_PIXEL_SIZE_MM = 0.0068
DEFAULT_COMPLEXITY_MANIFEST = PROJECT_ROOT / "output/experiments/phase3_segmentation/data_stratification/complexity_manifest.json"


CLASS_NAME_MAP = {
    0: "background",
    1: "scratch",
    2: "spot",
    3: "damage",
}
RULE_COLORS = {
    "spot": (80, 220, 255),
    "scratch": (0, 220, 80),
    "damage": (0, 128, 255),
    "critical_crash": (255, 80, 80),
    "uncertain": (255, 0, 255),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_roi_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        roi = np.load(path)
    else:
        roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (roi > 0) if roi is not None else None


def thin_mask(binary: np.ndarray) -> np.ndarray:
    thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
    if callable(thinning):
        return thinning(binary.astype(np.uint8) * 255) > 0
    from skimage.morphology import skeletonize

    return skeletonize(binary > 0)


def branch_stats(skel: np.ndarray) -> tuple[int, int]:
    if not np.any(skel):
        return 0, 0
    img = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    n_neighbors = neigh[skel] - 10
    branch_points = int(np.count_nonzero(n_neighbors >= 3))
    intersections = int(np.count_nonzero(n_neighbors >= 4))
    return branch_points, intersections


def classify_zone(coords: np.ndarray, center: tuple[int, int], radius: float, center_ratio: float = 0.30, micro_ratio: float = 0.60) -> str:
    if len(coords) == 0 or radius <= 0:
        return "edge"
    cy, cx = center
    dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
    mean_dist = float(np.mean(dists))
    if mean_dist <= radius * center_ratio:
        return "center"
    if mean_dist <= radius * micro_ratio:
        return "microstructure"
    return "edge"


def circularity(binary: np.ndarray) -> float:
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = float(np.count_nonzero(binary))
    perim = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    if perim <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def majority_class(mask_values: np.ndarray) -> tuple[int, float]:
    vals = mask_values[mask_values > 0]
    if vals.size == 0:
        return 0, 0.0
    counts = np.bincount(vals.astype(np.int64), minlength=4)
    cls_id = int(np.argmax(counts))
    return cls_id, float(counts[cls_id] / max(np.sum(counts), 1))


def component_graph_features(centroids: np.ndarray, pixel_size_mm: float, radius_mm: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    if len(centroids) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    radius_px = radius_mm / max(pixel_size_mm, 1e-9)
    dists = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    neighbors = ((dists > 0) & (dists <= radius_px)).astype(np.float32)
    neighbor_count = neighbors.sum(axis=1)
    cluster_score = neighbor_count / max(len(centroids) - 1, 1)
    return neighbor_count, cluster_score


def rule_confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def classify_component(features: dict) -> tuple[str, float]:
    conditions = {
        "spot": [
            features["area_mm2"] <= 0.02,
            features["circularity"] >= 0.45,
            features["skeleton_length_mm"] <= 0.20,
            features["aspect_ratio"] <= 3.0,
            features["branch_points"] == 0,
        ],
        "scratch": [
            features["aspect_ratio"] >= 6.0,
            features["avg_width_mm"] <= 0.12,
            features["skeleton_length_mm"] >= 0.15,
            features["branch_points"] <= 4,
            features["longest_span_mm"] >= 0.20,
        ],
        "damage": [
            features["area_mm2"] >= 0.05,
            features["avg_width_mm"] >= 0.08,
            features["circularity"] <= 0.45,
        ],
        "critical_crash": [
            features["cluster_score"] >= 0.20,
            features["intersection_count"] >= 2 or features["branch_points"] >= 3,
            features["area_mm2"] >= 0.08,
            features["neighbor_count"] >= 1,
        ],
    }

    scored = {label: float(sum(vals) / len(vals)) for label, vals in conditions.items()}
    # 优先级：critical -> spot -> scratch -> damage
    if scored["critical_crash"] >= 0.75:
        return "critical_crash", scored["critical_crash"]
    if scored["spot"] >= 0.80:
        return "spot", scored["spot"]
    if scored["scratch"] >= 0.80:
        return "scratch", scored["scratch"]
    if scored["damage"] >= 0.67:
        return "damage", scored["damage"]
    best_label = max(scored, key=scored.get)
    return ("uncertain", scored[best_label]) if scored[best_label] < 0.67 else (best_label, scored[best_label])


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_allowed_stems(
    manifest_path: Path | None,
    levels: list[str] | None,
) -> set[str] | None:
    if manifest_path is None or levels is None or not levels:
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed: set[str] = set()
    for level in levels:
        allowed.update(data.get("levels", {}).get(level, []))
    return allowed


def process_single_image(
    stem: str,
    mask_dir: Path,
    image_dir: Path,
    output_dir: Path,
    roi_mask: np.ndarray | None,
    pixel_size_mm: float,
    min_area_px: int,
    save_panels: bool,
) -> dict:
    mask = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"stem": stem, "rows": [], "uncertain": False, "counts": {}, "n_components": 0}

    binary = mask > 0
    if roi_mask is not None and roi_mask.shape == binary.shape:
        binary = binary & roi_mask
        mask = mask.copy()
        mask[~roi_mask] = 0

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    comps = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        comps.append((idx, area, float(centroids[idx, 0]), float(centroids[idx, 1])))

    centroid_arr = np.array([(c[2], c[3]) for c in comps], dtype=np.float32) if comps else np.zeros((0, 2), dtype=np.float32)
    neighbor_count, cluster_scores = component_graph_features(centroid_arr, pixel_size_mm)

    if roi_mask is not None:
        coords = np.argwhere(roi_mask)
        center = tuple(np.mean(coords, axis=0).astype(int))
        radius = float(np.sqrt(np.count_nonzero(roi_mask) / math.pi))
    else:
        h, w = mask.shape
        center = (h // 2, w // 2)
        radius = min(h, w) * 0.45

    panel = None
    if save_panels:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image is not None else cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    rows: list[dict] = []
    summary_counts: dict[str, int] = {}
    has_uncertain = False

    for comp_idx, (label_id, area_px, cx, cy) in enumerate(comps):
        comp_mask = labels == label_id
        ys, xs = np.where(comp_mask)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1

        skel = thin_mask(comp_mask)
        skel_len_px = int(np.count_nonzero(skel))
        avg_width_px = area_px / max(skel_len_px, 1.0)
        aspect_ratio = skel_len_px / max(avg_width_px, 1.0) if skel_len_px > 0 else 1.0
        branch_points, intersections = branch_stats(skel)
        coords = np.argwhere(comp_mask)
        span_px = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))) if len(coords) else 0.0
        maj_cls, maj_ratio = majority_class(mask[comp_mask])
        zone = classify_zone(coords, center, radius)

        features = {
            "area_mm2": area_px * (pixel_size_mm ** 2),
            "skeleton_length_mm": skel_len_px * pixel_size_mm,
            "avg_width_mm": avg_width_px * pixel_size_mm,
            "aspect_ratio": float(aspect_ratio),
            "circularity": circularity(comp_mask),
            "branch_points": int(branch_points),
            "intersection_count": int(intersections),
            "neighbor_count": float(neighbor_count[comp_idx]) if len(neighbor_count) else 0.0,
            "cluster_score": float(cluster_scores[comp_idx]) if len(cluster_scores) else 0.0,
            "longest_span_mm": span_px * pixel_size_mm,
        }
        rule_label, score = classify_component(features)
        confidence = rule_confidence(score)
        summary_counts[rule_label] = summary_counts.get(rule_label, 0) + 1
        if rule_label == "uncertain":
            has_uncertain = True

        rows.append(
            {
                "stem": stem,
                "component_id": int(label_id),
                "mask_majority_class": CLASS_NAME_MAP.get(maj_cls, "unknown"),
                "mask_majority_ratio": round(maj_ratio, 4),
                "rule_label": rule_label,
                "rule_score": round(score, 4),
                "rule_confidence": confidence,
                "zone": zone,
                "area_mm2": round(features["area_mm2"], 6),
                "skeleton_length_mm": round(features["skeleton_length_mm"], 6),
                "avg_width_mm": round(features["avg_width_mm"], 6),
                "aspect_ratio": round(features["aspect_ratio"], 4),
                "circularity": round(features["circularity"], 4),
                "branch_points": features["branch_points"],
                "intersection_count": features["intersection_count"],
                "neighbor_count": round(features["neighbor_count"], 4),
                "cluster_score": round(features["cluster_score"], 4),
                "longest_span_mm": round(features["longest_span_mm"], 6),
                "bbox_x": x0,
                "bbox_y": y0,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            }
        )

        if save_panels and panel is not None:
            color = RULE_COLORS[rule_label]
            cv2.rectangle(panel, (x0, y0), (x1, y1), color, 2)
            cv2.putText(panel, rule_label, (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if save_panels and panel is not None:
        cv2.imwrite(str(output_dir / "rule_panels" / f"{stem}.png"), panel)
    return {
        "stem": stem,
        "rows": rows,
        "uncertain": has_uncertain,
        "counts": summary_counts,
        "n_components": len(comps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按形态规则对分割结果进行缺陷分类")
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_MASK)
    parser.add_argument("--pixel-size-mm", type=float, default=DEFAULT_PIXEL_SIZE_MM)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-area-px", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--complexity-manifest", type=Path, default=DEFAULT_COMPLEXITY_MANIFEST)
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated complexity levels, e.g. L2,L3")
    parser.add_argument("--complex-only", action="store_true", help="Equivalent to --levels L2,L3")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if not args.no_panels:
        ensure_dir(args.output_dir / "rule_panels")
    roi_mask = load_roi_mask(args.roi_mask)

    level_list: list[str] | None = None
    if args.complex_only:
        level_list = ["L2", "L3"]
    elif args.levels:
        level_list = [item.strip() for item in args.levels.split(",") if item.strip()]

    allowed_stems = load_allowed_stems(args.complexity_manifest, level_list)

    names = sorted(p.stem for p in args.mask_dir.glob("*.png"))
    if allowed_stems is not None:
        names = [stem for stem in names if stem in allowed_stems]
    if args.max_images is not None:
        names = names[: args.max_images]

    rows: list[dict] = []
    uncertain_stems: set[str] = set()
    summary_counts: dict[str, int] = {}

    with futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        future_map = {
            ex.submit(
                process_single_image,
                stem,
                args.mask_dir,
                args.image_dir,
                args.output_dir,
                roi_mask,
                args.pixel_size_mm,
                args.min_area_px,
                not args.no_panels,
            ): stem
            for stem in names
        }
        for image_idx, fut in enumerate(futures.as_completed(future_map), start=1):
            result = fut.result()
            rows.extend(result["rows"])
            summary_counts.update(
                {
                    key: summary_counts.get(key, 0) + result["counts"].get(key, 0)
                    for key in set(summary_counts) | set(result["counts"])
                }
            )
            if result["uncertain"]:
                uncertain_stems.add(result["stem"])
            print(f"[{image_idx}/{len(names)}] {result['stem']}: {result['n_components']} 个连通域已分类")

    if not rows:
        raise SystemExit("未生成任何形态分类结果。")

    write_csv(args.output_dir / "component_labels.csv", rows)
    (args.output_dir / "component_labels.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "uncertain_samples.txt").write_text("\n".join(sorted(uncertain_stems)) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_images": len(names),
                "n_components": len(rows),
                "class_counts": summary_counts,
                "uncertain_images": sorted(uncertain_stems),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   形态规则分类完成                                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  图像数:     {len(names)}")
    print(f"  组件数:     {len(rows)}")
    print(f"  输出:       {args.output_dir}")
    print(f"  uncertain:  {len(uncertain_stems)} 张图")
    if level_list:
        print(f"  levels:     {','.join(level_list)}")
    print()


if __name__ == "__main__":
    main()
