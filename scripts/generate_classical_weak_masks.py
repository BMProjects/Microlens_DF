#!/usr/bin/env python3
"""为无标注暗场数据集生成 label-free 弱标签掩码（二值缺陷区域）。

复用项目自带经典检测器的前半段流水线：
  enhance → ``compute_candidate_map``(Frangi 线状 + TopHat 斑点 + Brightness 亮度)
  → Otsu 阈值 → 形态学清理 → 二值缺陷掩码。

刻意跳过实例提取/分类（``_extract_instances``，最耗时的一段），因为复杂度分层与
分割弱监督只需要"哪里是缺陷区域"的二值掩码。

与 ``generate_weak_masks.py`` 的区别：后者依赖 YOLO GT bbox；本脚本完全不依赖
人工标注，适用于新采集、尚未标注的暗场批次。

用法：
    uv run python scripts/generate_classical_weak_masks.py \
        --image-dir output/scrachdf_20260318_preprocessed/images \
        --roi-mask  output/scrachdf_20260318_preprocessed/roi_mask.png \
        --output-dir output/scrachdf_20260318_preprocessed/weak_masks --jobs 8
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.detection.classical import ClassicalDetector
from darkfield_defects.detection.features import compute_candidate_map

_DETECTOR: ClassicalDetector | None = None
_ROI: np.ndarray | None = None
_OUT: Path | None = None


def load_roi(path: Path) -> np.ndarray:
    roi = np.load(path) if path.suffix.lower() == ".npy" else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise FileNotFoundError(f"无法读取 ROI 掩码: {path}")
    return roi > 0


def _binary_defect_mask(detector: ClassicalDetector, img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """复刻 ClassicalDetector.detect 的 ①~④ 段，返回二值缺陷掩码。"""
    original_for_bright = img.copy()
    corrected = img
    if detector.det_params.enhance_enabled:
        corrected = detector._enhance_darkfield(corrected)
    elif detector.det_params.clahe_enabled:
        corrected = detector._apply_clahe(corrected)
    candidate = compute_candidate_map(
        corrected, detector.det_params, roi, original_image=original_for_bright
    )
    binary = detector._threshold(candidate)
    binary = detector._morphological_clean(binary, roi)
    return (binary.astype(np.uint8)) * 255


def _init(roi_path: str, out_dir: str) -> None:
    global _DETECTOR, _ROI, _OUT
    _DETECTOR = ClassicalDetector()
    _ROI = load_roi(Path(roi_path))
    _OUT = Path(out_dir)


def _work(fpath_str: str) -> tuple[str, int]:
    fpath = Path(fpath_str)
    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return fpath.name, -1
    mask = _binary_defect_mask(_DETECTOR, img, _ROI.copy())  # type: ignore[arg-type]
    cv2.imwrite(str(_OUT / fpath.name), mask)  # type: ignore[operator]
    return fpath.name, int((mask > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--roi-mask", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = [str(p) for p in sorted(args.image_dir.glob("*.png"))]
    n = len(images)
    print(f"[mask] {n} 张 → {args.output_dir}  (jobs={args.jobs})")

    t0 = time.perf_counter()
    ok = 0
    with ProcessPoolExecutor(
        max_workers=args.jobs, initializer=_init, initargs=(str(args.roi_mask), str(args.output_dir))
    ) as ex:
        for idx, (name, px) in enumerate(ex.map(_work, images), 1):
            if px >= 0:
                ok += 1
            if idx % 25 == 0 or idx == n:
                print(f"[{idx}/{n}] {name}  ({idx / (time.perf_counter() - t0):.2f} img/s)")

    print(f"[mask] 完成 {ok}/{n}，耗时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
