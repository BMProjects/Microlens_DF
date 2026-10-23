"""单图像判定 δᵢ 评估 — 阶段二“磨损识别准确率” A = Σδᵢ / N.

两条规则（2026-10-19 确定）:

- 正式口径（镜片级）: 由切片框拼回整镜 → IOS-NMS → 划痕链接 → WearScore,
  预测等级与基准标注推得的等级一致则 δᵢ = 1。
- 内部跟踪（切片级）: 类别感知、贪心匹配的框 F1 ≥ ``f1_min``（IoU ≥ ``iou_thr``）则 δᵢ = 1。

输入均为 YOLO 归一化 txt（预测可带第 6 列置信度），不依赖 torch，
因此检测模型或分割模型（mask→bbox）的预测都可以用同一协议评估。
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (cls, cx, cy, w, h, conf)，坐标为切片内归一化值
YoloBox = tuple[int, float, float, float, float, float]

TILE_NAME_RE = re.compile(r"^(?P<lens>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")


def read_yolo_txt(path: Path, conf_min: float = 0.0) -> list[YoloBox]:
    """读取 YOLO txt；文件不存在视为无框（Ultralytics 对空预测不写文件）。"""
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_no}: 需要至少 5 列，实际 {len(parts)}")
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        if conf >= conf_min:
            boxes.append((int(parts[0]), *map(float, parts[1:5]), conf))
    return boxes


def parse_tile_name(stem: str) -> tuple[str, int, int]:
    """``100r_y0000_x0560`` → (``100r``, y0=0, x0=560)。"""
    m = TILE_NAME_RE.match(stem)
    if m is None:
        raise ValueError(f"无法解析切片名: {stem}")
    return m["lens"], int(m["y"]), int(m["x"])


def _box_iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def tile_box_f1(gt: list[YoloBox], pred: list[YoloBox], iou_thr: float = 0.3) -> float:
    """类别感知的贪心匹配 F1；两者皆空记 1.0。"""
    if not gt and not pred:
        return 1.0
    used: set[int] = set()
    tp = 0
    for p in sorted(pred, key=lambda b: -b[5]):
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gt):
            if j in used or g[0] != p[0]:
                continue
            v = _box_iou(p, g)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used.add(best_j)
            tp += 1
    return 2 * tp / (len(gt) + len(pred))


def assess_lens(
    tiles: dict[str, list[YoloBox]],
    image_gray: np.ndarray,
):
    """把一张镜片的全部切片框拼回整图，按 GUI 主线同一后处理计算 WearAssessment。"""
    from darkfield_defects.app_services.inference_service import (
        NMS_IOS,
        SCRATCH_ANGLE,
        SCRATCH_GAP,
        boxes_to_detection_result,
        connect_scratches,
        nms_ios,
        tile_boxes_to_fullimage,
    )
    from darkfield_defects.scoring.quantify import compute_wear_metrics
    from darkfield_defects.scoring.wear_score import compute_wear_score

    full: list[tuple] = []
    for stem, boxes in tiles.items():
        _, y0, x0 = parse_tile_name(stem)
        full += tile_boxes_to_fullimage(boxes, x0, y0)
    full = nms_ios(full, ios_thresh=NMS_IOS)
    full, _ = connect_scratches(full, max_gap=SCRATCH_GAP, max_angle_diff=SCRATCH_ANGLE)
    det, roi = boxes_to_detection_result(full, image_gray)
    return compute_wear_score(compute_wear_metrics(det, roi))


@dataclass
class ImageLevelReport:
    conf: float
    iou_thr: float
    f1_min: float
    tile_rows: list[dict]
    lens_rows: list[dict]

    def summary(self) -> dict:
        lens_ok = [r["delta"] for r in self.lens_rows]
        tile_ok = [r["delta"] for r in self.tile_rows]
        gt_grades = [r["gt_grade"] for r in self.lens_rows]
        return {
            "conf": self.conf,
            "lens_grade_accuracy": float(np.mean(lens_ok)) if lens_ok else None,
            "n_lenses": len(lens_ok),
            "mean_abs_score_diff": (
                float(np.mean([abs(r["pred_score"] - r["gt_score"]) for r in self.lens_rows]))
                if self.lens_rows else None
            ),
            "gt_grade_distribution": {g: gt_grades.count(g) for g in "ABCD"},
            f"tile_f1@iou{self.iou_thr}>={self.f1_min}_accuracy": (
                float(np.mean(tile_ok)) if tile_ok else None
            ),
            "n_tiles": len(tile_ok),
        }


def evaluate_image_level(
    tile_stems: list[str],
    gt_dir: Path,
    pred_dir: Path,
    image_dir: Path,
    *,
    conf: float = 0.2,
    iou_thr: float = 0.3,
    f1_min: float = 0.5,
    image_ext: str = ".png",
) -> ImageLevelReport:
    """对一组切片计算切片级与镜片级 δᵢ。镜片集合由切片名推出。"""
    import cv2

    tile_rows: list[dict] = []
    by_lens: dict[str, dict[str, tuple[list[YoloBox], list[YoloBox]]]] = defaultdict(dict)
    for stem in tile_stems:
        gt = read_yolo_txt(gt_dir / f"{stem}.txt")
        pred = read_yolo_txt(pred_dir / f"{stem}.txt", conf_min=conf)
        f1 = tile_box_f1(gt, pred, iou_thr)
        tile_rows.append({"tile": stem, "n_gt": len(gt), "n_pred": len(pred),
                          "f1": round(f1, 4), "delta": int(f1 >= f1_min)})
        by_lens[parse_tile_name(stem)[0]][stem] = (gt, pred)

    lens_rows: list[dict] = []
    for lens, tiles in sorted(by_lens.items()):
        img_path = image_dir / f"{lens}{image_ext}"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"读取整镜图像失败: {img_path}")
        gt_a = assess_lens({s: v[0] for s, v in tiles.items()}, img)
        pr_a = assess_lens({s: v[1] for s, v in tiles.items()}, img)
        lens_rows.append({
            "lens": lens, "n_tiles": len(tiles),
            "gt_score": round(gt_a.score, 2), "gt_grade": gt_a.grade,
            "pred_score": round(pr_a.score, 2), "pred_grade": pr_a.grade,
            "delta": int(gt_a.grade == pr_a.grade),
        })
    return ImageLevelReport(conf, iou_thr, f1_min, tile_rows, lens_rows)
