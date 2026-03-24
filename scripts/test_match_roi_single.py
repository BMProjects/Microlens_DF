#!/usr/bin/env python3
"""单图测试：背景高光ring模板匹配 + ROI投影检查.

验证配准的仿射变换（平移、旋转、缩放）是否正确，
将roi_mask和ring_mask反投影到原图坐标，输出可视化图像。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
from darkfield_defects.preprocessing.registration import (
    _invert_affine_2x3,
    _warp_geometry,
    register_to_template,
)


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float64)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.6) -> np.ndarray:
    base = _to_u8(gray)
    if base.ndim == 2:
        out = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        out = base.copy()
    if np.any(mask):
        for c in range(3):
            ch = out[..., c].astype(np.float64)
            ch[mask] = (1.0 - alpha) * ch[mask] + alpha * float(color[c])
            out[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="单图匹配+ROI投影测试")
    parser.add_argument("--image", required=True, help="目标图像路径")
    parser.add_argument("--calibration", required=True, help="标定目录")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)

    pipeline = PreprocessPipeline()
    pipeline.load_calibration(args.calibration)
    cal = pipeline.calibration
    if cal is None:
        raise RuntimeError("标定加载失败")

    # 执行配准
    warp, score = register_to_template(
        src=img,
        ring_template=cal.B_blur,
        ring_mask=cal.ring_mask,
        max_scale_deviation=pipeline.max_scale_deviation,
        ecc_max_iter=pipeline.ecc_max_iter,
        ecc_epsilon=pipeline.ecc_epsilon,
        downsample=pipeline.registration_downsample,
    )

    # 提取仿射变换幅度
    dx, dy, angle_deg, scale = _warp_geometry(warp)

    raw_u8 = _to_u8(img)
    tpl_u8 = _to_u8(cal.B_blur)

    # 直接使用背景ROI作为目标ROI（参考坐标）
    roi_ref = cal.roi_mask.astype(bool)
    ring_ref = cal.ring_mask.astype(bool)

    # 投影回原图坐标（便于目视）
    inv_warp = _invert_affine_2x3(warp)
    roi_on_raw = cv2.warpAffine(
        roi_ref.astype(np.uint8),
        inv_warp.astype(np.float32),
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    ring_on_raw = cv2.warpAffine(
        ring_ref.astype(np.uint8),
        inv_warp.astype(np.float32),
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    tpl_on_raw = cv2.warpAffine(
        tpl_u8,
        inv_warp.astype(np.float32),
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    cv2.imwrite(str(out_dir / "00_template.png"), tpl_u8)
    cv2.imwrite(str(out_dir / "01_raw.png"), raw_u8)
    cv2.imwrite(str(out_dir / "02_raw_with_ring.png"), _overlay(raw_u8, ring_on_raw, (0, 255, 255), 0.6))
    cv2.imwrite(str(out_dir / "03_raw_with_roi.png"), _overlay(raw_u8, roi_on_raw, (0, 200, 255), 0.55))
    cv2.imwrite(
        str(out_dir / "04_raw_with_ring_roi.png"),
        _overlay(_overlay(raw_u8, roi_on_raw, (0, 200, 255), 0.45), ring_on_raw, (0, 255, 0), 0.65),
    )
    cv2.imwrite(str(out_dir / "05_template_warped_to_raw.png"), tpl_on_raw)
    cv2.imwrite(str(out_dir / "06_template_target_overlay.png"), _overlay(tpl_on_raw, ring_on_raw, (0, 255, 0), 0.55))
    cv2.imwrite(str(out_dir / "07_template_target_roi_overlay.png"), _overlay(_overlay(raw_u8, roi_on_raw, (0, 200, 255), 0.45), ring_on_raw, (0, 255, 0), 0.65))

    meta = {
        "image": str(args.image),
        "registration_score": float(score),
        "warp_src_to_ref": warp.astype(float).tolist(),
        "affine_dx": round(dx, 3),
        "affine_dy": round(dy, 3),
        "affine_angle_deg": round(angle_deg, 4),
        "affine_scale": round(scale, 6),
        "roi_pixels": int(np.count_nonzero(roi_ref)),
        "ring_pixels": int(np.count_nonzero(ring_ref)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 输出目录: {out_dir}")
    print(f"[INFO] reg_score={score:.6f}")
    print(f"[INFO] dx={dx:.3f}, dy={dy:.3f}, angle={angle_deg:.4f}°, scale={scale:.6f}")
    print(f"[INFO] roi_px={int(np.count_nonzero(roi_ref))}")


if __name__ == "__main__":
    main()
