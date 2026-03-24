"""GUI 预发布主线使用的全图推理服务."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.detection.base import DefectInstance, DefectType, DetectionResult
from darkfield_defects.logging import get_logger
from darkfield_defects.measurement import DEFAULT_CALIBRATION
from darkfield_defects.scoring.quantify import compute_wear_metrics
from darkfield_defects.scoring.wear_score import compute_wear_score

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "output" / "training" / "stage2_cleaned" / "weights" / "best.pt"
)

TILE_SIZE = 640
STRIDE = 480
NMS_IOS = 0.35
SCRATCH_GAP = 100
SCRATCH_ANGLE = 30
BATCH_SIZE = 16


@dataclass
class FullImageInferenceResult:
    image_gray: np.ndarray
    roi_mask: np.ndarray
    raw_boxes: list[tuple]
    nms_boxes: list[tuple]
    final_boxes: list[tuple]
    n_chains: int
    infer_time: float
    detection_result: DetectionResult
    metrics: object
    assessment: object


def get_default_weights_path() -> Path:
    return DEFAULT_WEIGHTS


@lru_cache(maxsize=4)
def _load_model(weights_path: str):
    from ultralytics import YOLO

    logger.info("加载全图推理模型: %s", weights_path)
    return YOLO(weights_path)


def run_full_image_inference(
    image_path: str | Path,
    *,
    conf_thresh: float = 0.20,
    weights_path: str | Path | None = None,
) -> FullImageInferenceResult:
    img_path = Path(image_path)
    resolved_weights = str(Path(weights_path) if weights_path else DEFAULT_WEIGHTS)
    model = _load_model(resolved_weights)

    raw_boxes, infer_time, img_gray = _run_sahi(model, img_path, conf=conf_thresh)
    nms_boxes = nms_ios(raw_boxes, ios_thresh=NMS_IOS)
    final_boxes, n_chains = connect_scratches(
        nms_boxes, max_gap=SCRATCH_GAP, max_angle_diff=SCRATCH_ANGLE
    )

    detection_result, roi_mask = boxes_to_detection_result(final_boxes, img_gray)
    metrics = compute_wear_metrics(detection_result, roi_mask)
    assessment = compute_wear_score(metrics)

    return FullImageInferenceResult(
        image_gray=img_gray,
        roi_mask=roi_mask,
        raw_boxes=raw_boxes,
        nms_boxes=nms_boxes,
        final_boxes=final_boxes,
        n_chains=n_chains,
        infer_time=infer_time,
        detection_result=detection_result,
        metrics=metrics,
        assessment=assessment,
    )


def _run_sahi(model, img_path: Path, conf: float) -> tuple[list[tuple], float, np.ndarray]:
    import time

    t0 = time.time()
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像不存在: {img_path}")
    h, w = img.shape
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    positions = generate_tile_positions(h, w, tile_size=TILE_SIZE, stride=STRIDE)
    tiles_rgb: list[np.ndarray] = []
    for y0, x0 in positions:
        tile = img3[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
        if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
            pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        tiles_rgb.append(tile)

    all_full: list[tuple] = []
    for bi in range(0, len(tiles_rgb), BATCH_SIZE):
        batch = tiles_rgb[bi:bi + BATCH_SIZE]
        batch_pos = positions[bi:bi + BATCH_SIZE]
        results = model(
            batch,
            conf=conf,
            imgsz=TILE_SIZE,
            batch=len(batch),
            stream=False,
            verbose=False,
            half=True,
        )
        for result, (y0, x0) in zip(results, batch_pos):
            if result.boxes is None:
                continue
            tile_boxes = []
            for box in result.boxes:
                cls = int(box.cls[0])
                xywh = box.xywhn[0].tolist()
                cf = float(box.conf[0])
                if xywh[2] * xywh[3] > 0.30:
                    continue
                tile_boxes.append((cls, xywh[0], xywh[1], xywh[2], xywh[3], cf))
            all_full.extend(tile_boxes_to_fullimage(tile_boxes, x0, y0))

    return all_full, time.time() - t0, img


def boxes_to_detection_result(
    boxes: list[tuple],
    img: np.ndarray,
    *,
    center_ratio: float = 0.40,
    microstructure_ratio: float = 0.75,
) -> tuple[DetectionResult, np.ndarray]:
    img_h, img_w = img.shape
    center = (img_h // 2, img_w // 2)
    lens_radius = min(img_h, img_w) * 0.43
    roi_mask = _build_circular_roi(img_h, img_w, center, lens_radius)
    bg_val = float(np.percentile(img, 5))

    instances: list[DefectInstance] = []
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx, (cls, x1, y1, x2, y2, conf) in enumerate(boxes):
        x1_i = int(max(0, round(x1)))
        y1_i = int(max(0, round(y1)))
        x2_i = int(min(img_w, round(x2)))
        y2_i = int(min(img_h, round(y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        w = x2_i - x1_i
        h = y2_i - y1_i
        area = w * h

        if cls == 0:
            defect_type = DefectType.SCRATCH
            length = float(max(w, h))
            aspect_ratio = float(max(w, h) / max(min(w, h), 1))
        elif cls == 1:
            defect_type = DefectType.SPOT
            length = float(np.sqrt(area))
            aspect_ratio = 1.0
        else:
            defect_type = DefectType.CRASH
            length = float(np.sqrt(area))
            aspect_ratio = float(max(w, h) / max(min(w, h), 1))

        roi = img[y1_i:y2_i, x1_i:x2_i]
        scatter = max(0.0, float(roi.mean()) - bg_val) if roi.size else 0.0
        zone = _classify_zone(
            x1_i, y1_i, x2_i, y2_i, center, lens_radius, center_ratio, microstructure_ratio
        )

        inst_mask = np.zeros((img_h, img_w), dtype=bool)
        inst_mask[y1_i:y2_i, x1_i:x2_i] = True
        full_mask[inst_mask] = 255

        instances.append(
            DefectInstance(
                instance_id=idx,
                defect_type=defect_type,
                skeleton_coords=np.empty((0, 2), dtype=int),
                mask=inst_mask,
                length_px=length,
                area_px=area,
                avg_width_px=float(min(w, h)),
                bbox=(x1_i, y1_i, w, h),
                scatter_intensity=scatter,
                prominence=float(conf),
                zone=zone,
                circularity=0.0,
                aspect_ratio=aspect_ratio,
                pixel_size_mm=DEFAULT_CALIBRATION.pixel_size_mm,
            )
        )

    return (
        DetectionResult(
            mask=full_mask,
            instances=instances,
            metadata={
                "detector": "fullimage-service",
                "optical_center": center,
                "lens_radius": lens_radius,
                "roi_mask": roi_mask,
                "pixel_size_mm": DEFAULT_CALIBRATION.pixel_size_mm,
            },
        ),
        roi_mask,
    )


def _build_circular_roi(
    img_h: int,
    img_w: int,
    center: tuple[int, int],
    radius: float,
) -> np.ndarray:
    cy, cx = center
    yy, xx = np.ogrid[:img_h, :img_w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)


def _classify_zone(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    center: tuple[int, int],
    radius: float,
    center_ratio: float,
    microstructure_ratio: float,
) -> str:
    cy, cx = center
    bcx = (x1 + x2) / 2.0
    bcy = (y1 + y2) / 2.0
    dist = float(np.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2))
    r = dist / max(radius, 1e-6)
    if r < center_ratio:
        return "center"
    if r < microstructure_ratio:
        return "microstructure"
    return "edge"


def tile_boxes_to_fullimage(
    yolo_boxes: list[tuple],
    tile_x0: int,
    tile_y0: int,
    tile_size: int = TILE_SIZE,
) -> list[tuple]:
    result = []
    for box in yolo_boxes:
        cls = box[0]
        cx_n, cy_n, w_n, h_n = box[1], box[2], box[3], box[4]
        conf = box[5] if len(box) > 5 else 1.0

        cx_px = tile_x0 + cx_n * tile_size
        cy_px = tile_y0 + cy_n * tile_size
        w_px = w_n * tile_size
        h_px = h_n * tile_size

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2
        result.append((cls, x1, y1, x2, y2, conf))
    return result


def generate_tile_positions(
    img_h: int,
    img_w: int,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
) -> list[tuple[int, int]]:
    positions = []
    y = 0
    while y + tile_size <= img_h:
        x = 0
        while x + tile_size <= img_w:
            positions.append((y, x))
            x += stride
        if x < img_w and (img_w - tile_size) != positions[-1][1]:
            positions.append((y, img_w - tile_size))
        y += stride
    if y < img_h:
        x = 0
        while x + tile_size <= img_w:
            positions.append((img_h - tile_size, x))
            x += stride
        if x < img_w:
            positions.append((img_h - tile_size, img_w - tile_size))
    return positions


def nms_ios(boxes: list[tuple], ios_thresh: float = 0.35) -> list[tuple]:
    if len(boxes) <= 1:
        return list(boxes)

    by_cls: dict[int, list[tuple]] = {}
    for box in boxes:
        by_cls.setdefault(int(box[0]), []).append(box)

    result: list[tuple] = []
    for cls_boxes in by_cls.values():
        cls_boxes.sort(key=lambda b: b[5], reverse=True)
        keep: list[tuple] = []
        suppressed: set[int] = set()

        for i, bi in enumerate(cls_boxes):
            if i in suppressed:
                continue
            keep.append(bi)
            for j in range(i + 1, len(cls_boxes)):
                if j in suppressed:
                    continue
                bj = cls_boxes[j]
                ix1 = max(bi[1], bj[1])
                iy1 = max(bi[2], bj[2])
                ix2 = min(bi[3], bj[3])
                iy2 = min(bi[4], bj[4])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_i = max((bi[3] - bi[1]) * (bi[4] - bi[2]), 1e-6)
                area_j = max((bj[3] - bj[1]) * (bj[4] - bj[2]), 1e-6)
                smaller = min(area_i, area_j)
                if inter / smaller > ios_thresh:
                    suppressed.add(j)

        result.extend(keep)
    return result


def _scratch_endpoints(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if w > h * 1.5:
        return (x1, cy), (x2, cy), 0.0
    if h > w * 1.5:
        return (cx, y1), (cx, y2), 90.0
    return (x1, y1), (x2, y2), float(np.degrees(np.arctan2(h, w)))


def connect_scratches(
    boxes: list[tuple],
    max_gap: float = 100.0,
    max_angle_diff: float = 30.0,
) -> tuple[list[tuple], int]:
    scratches = [b for b in boxes if b[0] == 0]
    others = [b for b in boxes if b[0] != 0]
    if len(scratches) <= 1:
        return list(boxes), 0

    infos = []
    for box in scratches:
        _, x1, y1, x2, y2, conf = box
        ep1, ep2, angle = _scratch_endpoints(x1, y1, x2, y2)
        infos.append({"box": box, "ep1": ep1, "ep2": ep2, "angle": angle, "conf": conf})

    parent = list(range(len(infos)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a_root, b_root = find(a), find(b)
        if a_root != b_root:
            parent[a_root] = b_root

    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            da = abs(infos[i]["angle"] - infos[j]["angle"])
            da = min(da, 180 - da)
            if da > max_angle_diff:
                continue

            min_dist = float("inf")
            for ea in (infos[i]["ep1"], infos[i]["ep2"]):
                for eb in (infos[j]["ep1"], infos[j]["ep2"]):
                    dist = ((ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
            if min_dist < max_gap:
                union(i, j)

    components: dict[int, list[int]] = {}
    for idx in range(len(infos)):
        components.setdefault(find(idx), []).append(idx)

    merged: list[tuple] = []
    n_chains = 0
    for indices in components.values():
        x1 = min(infos[i]["box"][1] for i in indices)
        y1 = min(infos[i]["box"][2] for i in indices)
        x2 = max(infos[i]["box"][3] for i in indices)
        y2 = max(infos[i]["box"][4] for i in indices)
        conf = max(infos[i]["conf"] for i in indices)
        merged.append((0, x1, y1, x2, y2, conf))
        if len(indices) > 1:
            n_chains += 1

    return merged + others, n_chains
