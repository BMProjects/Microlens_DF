"""模板背景预处理流水线.

简化流程:
1. 标定: 多张背景直接平均 → 离焦模糊 → 高光ring检测 → ROI构建 → 亮度修正系数.
2. 处理: 高光ring模板匹配(平移+旋转+缩放≤5%) → ROI投影 → 亮度修正 → 最终化.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darkfield_defects.logging import get_logger
from darkfield_defects.preprocessing.background_fusion import (
    generate_defocused_template,
)
from darkfield_defects.preprocessing.brightness_correction import (
    apply_linear_correction,
    finalize_image,
)
from darkfield_defects.preprocessing.registration import (
    apply_warp,
    register_to_template,
)
from darkfield_defects.preprocessing.roi_builder import (
    build_highlight_structure_mask,
    build_roi_from_highlight_mask,
)

logger = get_logger(__name__)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _preview_u8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float64)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class QualityMetrics:
    """单帧处理质量指标."""

    bg_mean: float = 0.0
    bg_std: float = 0.0
    reg_ecc: float = 0.0
    reg_dx: float = 0.0
    reg_dy: float = 0.0
    reg_angle_deg: float = 0.0
    reg_scale: float = 1.0
    correction_gain: float = 1.0
    correction_bias: float = 0.0
    is_valid: bool = True
    stage_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationResult:
    """背景模板标定结果."""

    B_avg: np.ndarray          # 多帧平均背景
    B_blur: np.ndarray         # 离焦模板
    roi_mask: np.ndarray       # ROI区域
    ring_mask: np.ndarray      # 高光ring区域
    correction_ref_median: float
    correction_ref_mad: float
    template_sigma: float = 0.0
    step_threshold_mask: np.ndarray | None = None
    step_dilated_mask: np.ndarray | None = None
    step_center_roi_raw: np.ndarray | None = None
    ref_shape: tuple[int, int] = (0, 0)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "B_avg.npy", self.B_avg)
        np.save(out / "B_blur.npy", self.B_blur)
        np.save(out / "roi_mask.npy", self.roi_mask)
        np.save(out / "ring_mask.npy", self.ring_mask)
        if self.step_threshold_mask is not None:
            np.save(out / "step_threshold_mask.npy", self.step_threshold_mask)
        if self.step_dilated_mask is not None:
            np.save(out / "step_dilated_mask.npy", self.step_dilated_mask)
        if self.step_center_roi_raw is not None:
            np.save(out / "step_center_roi_raw.npy", self.step_center_roi_raw)

        b_avg_u8 = _preview_u8(self.B_avg)
        b_blur_u8 = _preview_u8(self.B_blur)
        cv2.imwrite(str(out / "B_avg_preview.png"), b_avg_u8)
        cv2.imwrite(str(out / "B_blur_preview.png"), b_blur_u8)
        cv2.imwrite(str(out / "highlight_mask.png"), self.ring_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out / "roi_mask.png"), self.roi_mask.astype(np.uint8) * 255)
        if self.step_threshold_mask is not None:
            cv2.imwrite(
                str(out / "step1_threshold_highlight.png"),
                self.step_threshold_mask.astype(np.uint8) * 255,
            )
        if self.step_dilated_mask is not None:
            cv2.imwrite(
                str(out / "step2_dilated_highlight.png"),
                self.step_dilated_mask.astype(np.uint8) * 255,
            )
        if self.step_center_roi_raw is not None:
            cv2.imwrite(
                str(out / "step3_center_roi_raw.png"),
                self.step_center_roi_raw.astype(np.uint8) * 255,
            )

        meta = {
            "correction_ref_median": self.correction_ref_median,
            "correction_ref_mad": self.correction_ref_mad,
            "template_sigma": self.template_sigma,
            "ref_shape": list(self.ref_shape),
        }
        with open(out / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(meta), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        p = Path(path)
        with open(p / "calibration.json", encoding="utf-8") as f:
            meta = json.load(f)

        # 兼容旧格式：B_med.npy → B_avg.npy
        if (p / "B_avg.npy").exists():
            B_avg = np.load(p / "B_avg.npy")
        elif (p / "B_med.npy").exists():
            B_avg = np.load(p / "B_med.npy")
        else:
            raise FileNotFoundError(f"未找到背景数据: {p}")

        return cls(
            B_avg=B_avg,
            B_blur=np.load(p / "B_blur.npy"),
            roi_mask=np.load(p / "roi_mask.npy"),
            ring_mask=np.load(p / "ring_mask.npy"),
            correction_ref_median=float(meta.get("correction_ref_median", 0.0)),
            correction_ref_mad=float(meta.get("correction_ref_mad", 0.0)),
            template_sigma=float(meta.get("template_sigma", 0.0)),
            step_threshold_mask=np.load(p / "step_threshold_mask.npy")
            if (p / "step_threshold_mask.npy").exists()
            else None,
            step_dilated_mask=np.load(p / "step_dilated_mask.npy")
            if (p / "step_dilated_mask.npy").exists()
            else None,
            step_center_roi_raw=np.load(p / "step_center_roi_raw.npy")
            if (p / "step_center_roi_raw.npy").exists()
            else None,
            ref_shape=tuple(meta["ref_shape"]),
        )


@dataclass
class PipelineResult:
    """单帧模板匹配与背景修正结果."""

    image_final: np.ndarray
    roi_mask: np.ndarray
    warp_matrix: np.ndarray
    quality: QualityMetrics


class PreprocessPipeline:
    """背景模板驱动的预处理流水线（简化版）.

    标定: 背景平均 → 离焦 → ring检测 → ROI → 修正系数
    处理: ring模板匹配(平移+旋转+缩放≤5%) → ROI投影 → 亮度修正
    """

    def __init__(
        self,
        template_blur_sigma: float = 30.0,
        template_blur_per_pass_sigma: float = 6.0,
        template_brightness_gain: float = 1.08,
        template_highlight_center_ratio: float = 1.25,
        template_highlight_center_sample_ratio: float = 0.40,
        template_edge_expand_px: int = 25,
        roi_erode_ratio: float = 0.002,
        roi_extra_inset_px: float = 10.0,
        brightness_mode: str = "subtract",
        quality_bg_std_max: float = 5.0,
        max_scale_deviation: float = 0.05,
        ecc_max_iter: int = 200,
        ecc_epsilon: float = 1e-6,
        registration_downsample: float = 0.75,
        registration_pre_blur_sigma: float = 15.0,
        registration_ecc_score_threshold: float = 0.75,
        timing_log_enabled: bool = True,
    ) -> None:
        self.template_blur_sigma = float(max(0.0, template_blur_sigma))
        self.template_blur_per_pass_sigma = float(max(0.3, template_blur_per_pass_sigma))
        self.template_brightness_gain = float(max(0.1, template_brightness_gain))
        self.template_highlight_center_ratio = float(max(1.05, template_highlight_center_ratio))
        self.template_highlight_center_sample_ratio = float(template_highlight_center_sample_ratio)
        self.template_edge_expand_px = int(max(1, template_edge_expand_px))
        self.roi_erode_ratio = float(max(0.0, roi_erode_ratio))
        self.roi_extra_inset_px = float(max(0.0, roi_extra_inset_px))
        self.brightness_mode = brightness_mode
        self.quality_bg_std_max = float(quality_bg_std_max)
        self.max_scale_deviation = float(max(0.01, min(0.10, max_scale_deviation)))
        self.ecc_max_iter = int(ecc_max_iter)
        self.ecc_epsilon = float(ecc_epsilon)
        self.registration_downsample = float(registration_downsample)
        self.registration_pre_blur_sigma = float(max(0.0, registration_pre_blur_sigma))
        self.registration_ecc_score_threshold = float(registration_ecc_score_threshold)
        self.timing_log_enabled = bool(timing_log_enabled)
        self.calibration: CalibrationResult | None = None

    @classmethod
    def from_roi_pipeline_params(cls, rp: Any, **overrides: Any) -> "PreprocessPipeline":
        kwargs = {
            "template_blur_sigma": rp.template_blur_sigma,
            "template_blur_per_pass_sigma": rp.template_blur_per_pass_sigma,
            "template_brightness_gain": rp.template_brightness_gain,
            "template_highlight_center_ratio": rp.template_highlight_center_ratio,
            "template_highlight_center_sample_ratio": rp.template_highlight_center_sample_ratio,
            "template_edge_expand_px": rp.template_edge_expand_px,
            "roi_erode_ratio": rp.roi_erode_ratio,
            "roi_extra_inset_px": rp.roi_extra_inset_px,
            "brightness_mode": rp.brightness_correction_mode,
            "quality_bg_std_max": rp.quality_bg_std_max,
            "ecc_max_iter": rp.ecc_max_iterations,
            "ecc_epsilon": rp.ecc_epsilon,
            "registration_downsample": rp.registration_downsample,
            "max_scale_deviation": rp.max_scale_deviation,
            "registration_pre_blur_sigma": rp.registration_pre_blur_sigma,
            "registration_ecc_score_threshold": rp.registration_ecc_score_threshold,
            "timing_log_enabled": rp.timing_log_enabled,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def calibrate(
        self,
        bg_dir: str | Path,
        save_dir: str | Path | None = None,
    ) -> CalibrationResult:
        """标定流程: 平均 → 离焦 → ring检测 → ROI → 修正系数."""
        bg_dir = Path(bg_dir)
        bg_files = sorted(bg_dir.glob("bg-*.png"))
        if not bg_files:
            bg_files = sorted(bg_dir.glob("bg*.png"))
        if not bg_files:
            raise FileNotFoundError(f"未找到背景图: {bg_dir}")

        bg_images: list[np.ndarray] = []
        for file in bg_files:
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                bg_images.append(img)
        if not bg_images:
            raise ValueError("无法加载任何背景图像")

        ref_shape = bg_images[0].shape[:2]

        # ① 直接平均（背景图无大偏差，无需配准）
        stack = np.stack([img.astype(np.float64) for img in bg_images], axis=0)
        B_avg = np.mean(stack, axis=0)
        logger.info("背景平均完成: %d 帧 → B_avg mean=%.1f", len(bg_images), B_avg.mean())

        # ② 模拟离焦 → B_blur
        template_sigma = self.template_blur_sigma
        if template_sigma > 0.0:
            B_blur = generate_defocused_template(
                B_avg,
                template_sigma,
                per_pass_sigma=self.template_blur_per_pass_sigma,
                brightness_gain=self.template_brightness_gain,
            )
        else:
            B_blur = np.clip(B_avg * self.template_brightness_gain, 0.0, 255.0)

        # ③ 高光ring边缘检测（中心相对阈值）
        ring_mask, _, ring_steps = build_highlight_structure_mask(
            B_blur,
            center_ratio=self.template_highlight_center_ratio,
            center_sample_ratio=self.template_highlight_center_sample_ratio,
            edge_expand_px=self.template_edge_expand_px,
            return_steps=True,
        )

        # ④ ROI区域构建
        roi_mask, roi_stats, roi_steps = build_roi_from_highlight_mask(
            ring_mask,
            erode_diameter_ratio=self.roi_erode_ratio,
            return_steps=True,
        )
        if np.any(roi_mask) and self.roi_extra_inset_px > 0.0:
            dist = cv2.distanceTransform(roi_mask.astype(np.uint8), cv2.DIST_L2, 3)
            roi_mask = dist > self.roi_extra_inset_px
            n_cc, labels, stats_cc, _ = cv2.connectedComponentsWithStats(roi_mask.astype(np.uint8))
            if n_cc > 1:
                lid = int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]) + 1)
                roi_mask = labels == lid

        # ⑤ ROI亮度修正系数求解
        correction_region = roi_mask & ~ring_mask
        if not np.any(correction_region):
            correction_region = roi_mask
        ref_vals = B_blur[correction_region].astype(np.float64)
        ref_median = float(np.median(ref_vals)) if ref_vals.size else 0.0
        ref_mad = float(np.median(np.abs(ref_vals - ref_median))) if ref_vals.size else 0.0

        self.calibration = CalibrationResult(
            B_avg=B_avg,
            B_blur=B_blur,
            roi_mask=roi_mask,
            ring_mask=ring_mask,
            correction_ref_median=ref_median,
            correction_ref_mad=ref_mad,
            template_sigma=template_sigma,
            step_threshold_mask=ring_steps.get("threshold_highlight"),
            step_dilated_mask=ring_steps.get("dilated_highlight"),
            step_center_roi_raw=roi_steps.get("center_roi_raw"),
            ref_shape=ref_shape,
        )
        if save_dir is not None:
            self.calibration.save(save_dir)
        return self.calibration

    def load_calibration(self, path: str | Path) -> None:
        self.calibration = CalibrationResult.load(path)

    def process(
        self,
        image: np.ndarray,
        frame_index: int = -1,
    ) -> PipelineResult:
        """处理流程: ring模板匹配 → ROI投影 → 亮度修正 → 最终化."""
        if self.calibration is None:
            raise RuntimeError("请先运行 calibrate() 或 load_calibration()")

        del frame_index
        cal = self.calibration
        stage_ms: dict[str, float] = {}
        t_total = time.perf_counter()

        # ① 高光ring区域模板匹配（允许平移+旋转+缩放≤5%）
        t0 = time.perf_counter()
        warp, score = register_to_template(
            src=image,
            ring_template=cal.B_blur,
            ring_mask=cal.ring_mask,
            max_scale_deviation=self.max_scale_deviation,
            ecc_max_iter=self.ecc_max_iter,
            ecc_epsilon=self.ecc_epsilon,
            downsample=self.registration_downsample,
            pre_blur_sigma=self.registration_pre_blur_sigma,
            ecc_score_threshold=self.registration_ecc_score_threshold,
        )
        stage_ms["register"] = (time.perf_counter() - t0) * 1000.0

        # 提取仿射变换幅度
        a00 = float(warp[0, 0])
        a01 = float(warp[0, 1])
        a10 = float(warp[1, 0])
        a11 = float(warp[1, 1])
        reg_dx = float(warp[0, 2])
        reg_dy = float(warp[1, 2])
        reg_angle_deg = float(np.degrees(np.arctan2(a10, a00)))
        sx = float(np.hypot(a00, a10))
        sy = float(np.hypot(a01, a11))
        reg_scale = 0.5 * (sx + sy)

        logger.info(
            "仿射变换幅度: dx=%.2f dy=%.2f angle=%.4f° scale=%.6f (sx=%.6f sy=%.6f) score=%.6f",
            reg_dx,
            reg_dy,
            reg_angle_deg,
            reg_scale,
            sx,
            sy,
            score,
        )

        # ② 对齐目标图像到参考坐标系
        t0 = time.perf_counter()
        aligned = apply_warp(image.astype(np.float64), warp, cal.ref_shape)
        stage_ms["warp"] = (time.perf_counter() - t0) * 1000.0

        # ③ ROI投影（直接使用标定的roi_mask，已在参考坐标系）
        roi_mask = cal.roi_mask.astype(bool)
        ring_mask = cal.ring_mask.astype(bool)

        # ④ 亮度修正 — 使用固定参数（gain=1, bias=0），直接减去背景模板
        # 所有目标图像使用相同的调整幅度，避免per-image估计受划痕污染
        t0 = time.perf_counter()
        gain, bias = 1.0, 0.0
        corrected = apply_linear_correction(
            aligned,
            cal.B_blur,
            gain,
            bias,
            mode=self.brightness_mode,
        )
        stage_ms["correction"] = (time.perf_counter() - t0) * 1000.0

        # ⑤ 最终化
        t0 = time.perf_counter()
        final = finalize_image(corrected, roi_mask)
        stage_ms["finalize"] = (time.perf_counter() - t0) * 1000.0
        stage_ms["total"] = (time.perf_counter() - t_total) * 1000.0

        # 质量指标
        bg_region = roi_mask & ~ring_mask
        if not np.any(bg_region):
            bg_region = roi_mask
        bg_vals = corrected[bg_region] if np.any(bg_region) else np.array([0.0])
        quality = QualityMetrics(
            bg_mean=float(np.mean(bg_vals)),
            bg_std=float(np.std(bg_vals)),
            reg_ecc=float(score),
            reg_dx=reg_dx,
            reg_dy=reg_dy,
            reg_angle_deg=reg_angle_deg,
            reg_scale=reg_scale,
            correction_gain=float(gain),
            correction_bias=float(bias),
            is_valid=float(np.std(bg_vals)) < self.quality_bg_std_max,
            stage_ms=stage_ms,
        )

        if self.timing_log_enabled:
            logger.info(
                "模板预处理: reg=%.1fms warp=%.1fms corr=%.1fms final=%.1fms total=%.1fms",
                stage_ms["register"],
                stage_ms["warp"],
                stage_ms["correction"],
                stage_ms["finalize"],
                stage_ms["total"],
            )

        return PipelineResult(
            image_final=final,
            roi_mask=roi_mask,
            warp_matrix=warp,
            quality=quality,
        )

    def process_batch(
        self,
        target_dir: str | Path,
        output_dir: str | Path,
        extensions: tuple[str, ...] = (".png", ".bmp"),
        exclude_patterns: tuple[str, ...] = ("bg",),
    ) -> list[dict[str, Any]]:
        target_dir = Path(target_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            file
            for file in target_dir.iterdir()
            if file.suffix.lower() in extensions
            and not any(pat in file.stem.lower() for pat in exclude_patterns)
        )
        report: list[dict[str, Any]] = []
        for idx, file in enumerate(files):
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                report.append({"file": file.name, "error": "read_failed"})
                continue
            try:
                result = self.process(img, frame_index=idx)
            except Exception as exc:
                report.append({"file": file.name, "error": str(exc)})
                continue

            cv2.imwrite(str(output_dir / f"{file.stem}_processed.png"), result.image_final)
            if idx == 0:
                cv2.imwrite(
                    str(output_dir / "roi_mask.png"),
                    result.roi_mask.astype(np.uint8) * 255,
                )
            q = result.quality
            report.append(
                {
                    "file": file.name,
                    "bg_mean": round(q.bg_mean, 2),
                    "bg_std": round(q.bg_std, 2),
                    "ecc": round(q.reg_ecc, 6),
                    "reg_dx": round(q.reg_dx, 3),
                    "reg_dy": round(q.reg_dy, 3),
                    "reg_angle_deg": round(q.reg_angle_deg, 4),
                    "reg_scale": round(q.reg_scale, 6),
                    "correction_gain": round(q.correction_gain, 6),
                    "correction_bias": round(q.correction_bias, 4),
                    "valid": q.is_valid,
                    "stage_ms": {k: round(v, 2) for k, v in q.stage_ms.items()},
                }
            )

        with open(output_dir / "quality_report.json", "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        return report
