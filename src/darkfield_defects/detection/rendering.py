"""可视化与渲染模块 — 多类缺陷叠加 + COCO 标注输出 + CSV/JSONL 元数据."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darkfield_defects.measurement import get_calibration

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from darkfield_defects.detection.base import DefectType, DetectionResult, DefectInstance
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)

# ── 缺陷类型颜色映射 (BGR) ──────────────────────────────
DEFECT_COLORS: dict[DefectType, tuple[int, int, int]] = {
    DefectType.SCRATCH: (0, 255, 0),    # 绿色 — 划痕
    DefectType.SPOT: (0, 255, 255),     # 黄色 — 斑点/凹坑
    DefectType.DAMAGE: (0, 0, 255),     # 红色 — 大面积缺损
    DefectType.CRASH: (255, 0, 255),    # 品红 — 密集缺陷区
}

# ── COCO category 定义 ───────────────────────────────────
COCO_CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "scratch", "supercategory": "defect"},
    {"id": 2, "name": "spot", "supercategory": "defect"},
    {"id": 3, "name": "damage", "supercategory": "defect"},
    {"id": 4, "name": "crash", "supercategory": "defect"},
]

DEFECT_CATEGORY_ID: dict[DefectType, int] = {
    DefectType.SCRATCH: 1,
    DefectType.SPOT: 2,
    DefectType.DAMAGE: 3,
    DefectType.CRASH: 4,
}

# ── 类型缩写 ─────────────────────────────────────────────
DEFECT_SHORT: dict[DefectType, str] = {
    DefectType.SCRATCH: "S",
    DefectType.SPOT: "P",
    DefectType.DAMAGE: "D",
    DefectType.CRASH: "C",
}


def render_overlay(
    image: np.ndarray,
    result: DetectionResult,
    alpha: float = 0.5,
    show_skeleton: bool = True,
    show_zones: bool = True,
) -> np.ndarray:
    """在原图上叠加检测结果（按缺陷类型着色）."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    overlay = canvas.copy()

    for inst in result.instances:
        color = DEFECT_COLORS.get(inst.defect_type, (255, 255, 255))
        inst_mask = inst.mask if inst.mask.dtype == bool else inst.mask > 0
        overlay[inst_mask] = color

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    if show_skeleton:
        for inst in result.instances:
            if inst.defect_type == DefectType.SCRATCH:
                for r, c in inst.skeleton_coords:
                    if 0 <= r < canvas.shape[0] and 0 <= c < canvas.shape[1]:
                        canvas[r, c] = (0, 0, 255)

    if show_zones and "optical_center" in result.metadata:
        center = result.metadata["optical_center"]
        radius = result.metadata.get("lens_radius", 0)
        if radius > 0:
            cy, cx = center
            r_center = int(radius * 0.40)
            r_trans = int(radius * 0.75)
            cv2.circle(canvas, (cx, cy), r_center, (255, 255, 0), 1)
            cv2.circle(canvas, (cx, cy), r_trans, (255, 128, 0), 1)
            cv2.circle(canvas, (cx, cy), int(radius), (128, 128, 128), 1)

    for inst in result.instances:
        x, y, w, h = inst.bbox
        color = DEFECT_COLORS.get(inst.defect_type, (255, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)

        type_label = DEFECT_SHORT.get(inst.defect_type, "?")
        label = f"#{inst.instance_id} {type_label} L={inst.length_mm:.3f} mm"
        cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return canvas


def render_summary_panel(
    image: np.ndarray,
    result: DetectionResult,
    wear_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """生成汇总面板，包含原图、检测结果和统计信息."""
    overlay = render_overlay(image, result)
    h, w = overlay.shape[:2]

    panel_w = 400
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    lines = [
        f"Total Defects: {result.num_defects}",
        f"  Scratches: {result.num_scratches}",
        f"  Spots: {result.num_spots}",
        f"  Damage: {result.num_damages}",
        f"  Crash: {result.num_crashes}",
        "",
        f"Total Length: {result.total_length_mm:.3f} mm",
        f"Total Area: {result.total_area_mm2:.4f} mm^2",
        "",
    ]

    zone_stats: dict[str, list[DefectInstance]] = {"center": [], "microstructure": [], "edge": []}
    for inst in result.instances:
        zone_key = "microstructure" if inst.zone == "transition" else inst.zone
        zone_stats.get(zone_key, []).append(inst)

    for zone, insts in zone_stats.items():
        if insts:
            total_l = sum(i.length_mm for i in insts)
            lines.append(f"  [{zone}] {len(insts)} defects, L={total_l:.3f} mm")

    if wear_info:
        lines.append("")
        lines.append(f"WearScore: {wear_info.get('score', 'N/A')}")
        lines.append(f"Grade: {wear_info.get('grade', 'N/A')}")

    y = 30
    for line in lines:
        cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22

    return np.hstack([overlay, panel])


def _instance_segmentation(inst: DefectInstance) -> list[list[float]]:
    """获取实例的分割多边形.

    划痕类用旋转矩形多边形，其他用轮廓。
    """
    mask_u8 = inst.mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if inst.defect_type == DefectType.SCRATCH:
        cnt_all = np.vstack(contours)
        rect = cv2.minAreaRect(cnt_all)
        if rect[1][0] > 0 and rect[1][1] > 0:
            box = cv2.boxPoints(rect)
            polygon = [float(v) for point in box for v in point]
            return [polygon]

    # 其他类型：用轮廓多边形
    result = []
    for cnt in contours:
        if len(cnt) >= 3:
            polygon = [float(v) for point in cnt.reshape(-1, 2) for v in point]
            result.append(polygon)
    return result


def export_coco(
    results: list[tuple[str, DetectionResult]],
    output_path: str | Path,
) -> None:
    """导出 COCO 格式标注 JSON（完整版）."""
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
    }

    ann_id = 1
    for img_id, (filename, result) in enumerate(results, 1):
        h, w = result.mask.shape[:2]
        calibration = get_calibration(result.metadata)
        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h,
            "pixel_size_mm": calibration.pixel_size_mm,
        })

        for inst in result.instances:
            x, y, bw, bh = inst.bbox
            cat_id = DEFECT_CATEGORY_ID.get(inst.defect_type, 1)
            segmentation = _instance_segmentation(inst)

            ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "category_name": inst.defect_type.value,
                "bbox": [x, y, bw, bh],
                "area": inst.area_px,
                "segmentation": segmentation,
                "iscrowd": 0,
                "prominence": round(inst.prominence, 2),
                "scatter_intensity": round(inst.scatter_intensity, 2),
                "zone": inst.zone,
                "attributes": {
                    "length_mm": round(inst.length_mm, 4),
                    "avg_width_mm": round(inst.avg_width_mm, 4),
                    "area_mm2": round(inst.area_mm2, 6),
                    "aspect_ratio": round(inst.aspect_ratio, 2),
                    "circularity": round(inst.circularity, 3),
                    "raw_px": {
                        "length_px": round(inst.length_px, 1),
                        "avg_width_px": round(inst.avg_width_px, 1),
                        "area_px": inst.area_px,
                    },
                },
            }
            coco["annotations"].append(ann)
            ann_id += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info(f"COCO 标注导出: {output_path} ({ann_id - 1} annotations)")


def export_metadata_csv(
    results: list[tuple[str, DetectionResult]],
    output_path: str | Path,
) -> None:
    """导出 CSV 元数据汇总（每行一个缺陷实例）."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image", "defect_id", "category", "zone",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "bbox_x_mm", "bbox_y_mm", "bbox_w_mm", "bbox_h_mm",
        "area_mm2", "length_mm", "avg_width_mm",
        "aspect_ratio", "circularity",
        "scatter_intensity", "prominence",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename, result in results:
            for inst in result.instances:
                x, y, w, h = inst.bbox
                x_mm, y_mm, w_mm, h_mm = inst.bbox_mm
                writer.writerow({
                    "image": filename,
                    "defect_id": inst.instance_id,
                    "category": inst.defect_type.value,
                    "zone": inst.zone,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "bbox_x_mm": round(x_mm, 4),
                    "bbox_y_mm": round(y_mm, 4),
                    "bbox_w_mm": round(w_mm, 4),
                    "bbox_h_mm": round(h_mm, 4),
                    "area_mm2": round(inst.area_mm2, 6),
                    "length_mm": round(inst.length_mm, 4),
                    "avg_width_mm": round(inst.avg_width_mm, 4),
                    "aspect_ratio": round(inst.aspect_ratio, 2),
                    "circularity": round(inst.circularity, 3),
                    "scatter_intensity": round(inst.scatter_intensity, 2),
                    "prominence": round(inst.prominence, 2),
                })

    logger.info(f"CSV 元数据导出: {output_path}")


def export_metadata_jsonl(
    results: list[tuple[str, DetectionResult]],
    output_path: str | Path,
) -> None:
    """导出 JSONL 元数据汇总（每行一条 JSON 记录）."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for filename, result in results:
            for inst in result.instances:
                x, y, w, h = inst.bbox
                x_mm, y_mm, w_mm, h_mm = inst.bbox_mm
                record = {
                    "image": filename,
                    "defect_id": inst.instance_id,
                    "category": inst.defect_type.value,
                    "category_id": DEFECT_CATEGORY_ID.get(inst.defect_type, 0),
                    "zone": inst.zone,
                    "bbox": [x, y, w, h],
                    "bbox_mm": [round(x_mm, 4), round(y_mm, 4), round(w_mm, 4), round(h_mm, 4)],
                    "area_mm2": round(inst.area_mm2, 6),
                    "length_mm": round(inst.length_mm, 4),
                    "avg_width_mm": round(inst.avg_width_mm, 4),
                    "aspect_ratio": round(inst.aspect_ratio, 2),
                    "circularity": round(inst.circularity, 3),
                    "scatter_intensity": round(inst.scatter_intensity, 2),
                    "prominence": round(inst.prominence, 2),
                    "raw_px": {
                        "area_px": inst.area_px,
                        "length_px": round(inst.length_px, 1),
                        "avg_width_px": round(inst.avg_width_px, 1),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False, cls=_NumpyEncoder) + "\n")

    logger.info(f"JSONL 元数据导出: {output_path}")


def save_detection_output(
    image: np.ndarray,
    result: DetectionResult,
    output_dir: str | Path,
    filename_stem: str,
    save_overlay: bool = True,
    save_mask: bool = True,
) -> dict[str, Path]:
    """保存检测结果文件."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    if save_overlay:
        overlay = render_overlay(image, result)
        path = out_dir / f"{filename_stem}_overlay.png"
        cv2.imwrite(str(path), overlay)
        saved["overlay"] = path

    if save_mask:
        path = out_dir / f"{filename_stem}_mask.png"
        cv2.imwrite(str(path), result.mask)
        saved["mask"] = path

    logger.info(f"检测结果已保存: {out_dir}/{filename_stem}_*")
    return saved
