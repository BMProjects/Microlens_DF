"""检测参数管理 — dataclass + YAML 加载 + 校验."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from darkfield_defects.exceptions import ConfigurationError
from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessParams:
    """预处理参数."""
    bg_correction_method: str = "division"
    bg_epsilon: float = 1.0
    roi_method: str = "hough"
    roi_shrink_ratio: float = 0.05
    roi_min_radius_ratio: float = 0.3
    denoise_method: str = "bilateral"
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    local_enhance_enabled: bool = True
    local_enhance_sigma: float = 80.0


@dataclass
class DetectionParams:
    """检测参数."""
    # Frangi 滤波
    frangi_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 5.0])
    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0
    # 顶帽变换
    tophat_kernel_size: int = 15
    # 阈值
    threshold_method: str = "otsu"
    adaptive_block_size: int = 51
    adaptive_c: float = 10.0
    # tile 模式下 Otsu 防噪底限：若 Otsu 结果低于此值则提升至此
    # 防止纯暗 tile 因无双峰分布而设出极低阈值导致全图误检
    otsu_floor: float = 0.0        # 默认0=不限制；tile模式建议0.06
    # 形态学
    morph_open_kernel: int = 3
    min_area: int = 50
    min_length: float = 20.0
    min_aspect_ratio: float = 3.0
    # 密度检测（crash）
    density_kernel_ratio: float = 0.08
    density_threshold: float = 0.2
    dense_min_area: int = 2000
    # Prominence 过滤
    prominence_min_value: float = 20.0
    # 断裂合并（端点距离 + 方向）
    merge_max_gap: float = 30.0
    merge_max_angle: float = 20.0
    # 断裂合并（旋转包围盒延伸）
    scratch_extend_ratio: float = 0.25
    scratch_extend_min_px: int = 12
    scratch_extend_width_ratio: float = 1.0
    scratch_extend_min_width: int = 4
    scratch_merge_gray_tol: float = 20.0
    merge_method: str = "rotated_box"   # "endpoint" | "rotated_box" | "both"
    # 暗场图像增强（gamma + CLAHE 两阶段）
    # gamma<1 提亮暗区细节, CLAHE 局部对比度均衡 — 均为逐像素操作,不扩大缺陷面积
    enhance_enabled: bool = True
    enhance_gamma: float = 0.4            # gamma校正指数 (0.3~0.5, 越小越亮)
    clahe_enabled: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 8

    def ensure_valid(self) -> None:
        if self.min_area < 1:
            raise ConfigurationError("min_area must be >= 1")
        if self.min_length < 1:
            raise ConfigurationError("min_length must be >= 1")
        if self.morph_open_kernel < 1 or self.morph_open_kernel % 2 == 0:
            raise ConfigurationError("morph_open_kernel must be odd and >= 1")


@dataclass
class ScoringParams:
    """磨损评分参数."""
    zone_center_ratio: float = 0.40
    zone_transition_ratio: float = 0.75
    weight_L_center: float = 0.30
    weight_L_total: float = 0.20
    weight_N: float = 0.15
    weight_A: float = 0.20
    weight_S_scatter: float = 0.15
    grade_A_max: float = 20.0
    grade_B_max: float = 45.0
    grade_C_max: float = 70.0


@dataclass
class OutputParams:
    """输出参数."""
    save_overlay: bool = True
    save_mask: bool = True
    save_coco_json: bool = True
    save_report_html: bool = True
    overlay_alpha: float = 0.5
    scratch_color: list[int] = field(default_factory=lambda: [0, 255, 0])


@dataclass
class ROIPipelineParams:
    """背景模板预处理流水线参数."""
    # 离焦模板
    template_blur_sigma: float = 30.0
    template_blur_per_pass_sigma: float = 6.0
    template_brightness_gain: float = 1.08
    # 高光ring检测（中心相对阈值）
    template_highlight_center_ratio: float = 1.25    # threshold = center_level × ratio
    template_highlight_center_sample_ratio: float = 0.40  # 中心采样区比例
    template_edge_expand_px: int = 25
    # ROI构建
    roi_erode_ratio: float = 0.002
    roi_extra_inset_px: float = 10.0
    # 配准
    ecc_max_iterations: int = 200
    ecc_epsilon: float = 1e-6
    registration_downsample: float = 0.75
    max_scale_deviation: float = 0.05       # 最大缩放偏差±5%
    registration_pre_blur_sigma: float = 15.0   # ECC前预模糊σ（平滑缺陷干扰）
    registration_ecc_score_threshold: float = 0.75  # 低于此分数触发no_mask级联回退
    # 亮度修正
    brightness_correction_mode: str = "subtract"
    # 质量控制
    quality_bg_std_max: float = 5.0
    timing_log_enabled: bool = True


@dataclass
class PipelineParams:
    """全流水线参数集合."""
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    detection: DetectionParams = field(default_factory=DetectionParams)
    scoring: ScoringParams = field(default_factory=ScoringParams)
    output: OutputParams = field(default_factory=OutputParams)
    roi_pipeline: ROIPipelineParams = field(default_factory=ROIPipelineParams)

    def to_dict(self) -> dict:
        """导出为字典（用于参数快照）."""
        import dataclasses
        return dataclasses.asdict(self)


def load_params(config_path: Optional[str | Path] = None) -> PipelineParams:
    """从 YAML 文件加载参数.

    Args:
        config_path: YAML 配置路径；None 则使用默认值.

    Returns:
        PipelineParams 实例.
    """
    params = PipelineParams()

    if config_path is None:
        logger.info("使用默认参数")
        return params

    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return params

    # 递归更新 dataclass 字段
    def _update_dc(dc: object, data: dict) -> None:
        for key, val in data.items():
            if hasattr(dc, key):
                setattr(dc, key, val)

    if "preprocess" in cfg:
        _update_dc(params.preprocess, cfg["preprocess"])
    if "detection" in cfg:
        _update_dc(params.detection, cfg["detection"])
    if "scoring" in cfg:
        _update_dc(params.scoring, cfg["scoring"])
    if "output" in cfg:
        _update_dc(params.output, cfg["output"])
    if "roi_pipeline" in cfg:
        _update_dc(params.roi_pipeline, cfg["roi_pipeline"])

    params.detection.ensure_valid()
    logger.info(f"从 {path} 加载配置")
    return params
