"""统一磨损评分模块."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from darkfield_defects.detection.params import ScoringParams
from darkfield_defects.logging import get_logger
from darkfield_defects.scoring.quantify import WearMetrics

logger = get_logger(__name__)


@dataclass
class WearAssessment:
    """磨损评估结果."""

    score: float
    grade: str
    grade_label: str
    glare_index: float = 0.0
    haze_index: float = 0.0
    contributors: dict[str, float] | None = None
    conclusion: str = ""
    dominant_factor: str = ""
    effective_metrics: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.contributors is None:
            self.contributors = {}
        if self.effective_metrics is None:
            self.effective_metrics = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "grade_label": self.grade_label,
            "glare_index": round(self.glare_index, 2),
            "haze_index": round(self.haze_index, 2),
            "contributors": {k: round(v, 2) for k, v in self.contributors.items()},
            "dominant_factor": self.dominant_factor,
            "effective_metrics": {
                k: round(v, 4) for k, v in self.effective_metrics.items()
            },
            "conclusion": self.conclusion,
        }


def _sat_log(x: float, *, norm: float, scale: float, cap: float = 100.0) -> float:
    x = max(x, 0.0) / max(norm, 1e-6)
    return min(scale * math.log1p(x), cap)


def compute_wear_score(
    metrics: WearMetrics,
    params: ScoringParams | None = None,
) -> WearAssessment:
    """统一评分公式.

    评分理论：
    - 保留 `L_center / L_total / N / A / S_scatter` 五项主框架
    - 通过区域加权把微结构区的重要性纳入 `L_total/N/A/S_scatter`
    - 中心区独立保留最高优先级
    """
    if params is None:
        params = ScoringParams()

    n_edge = max(metrics.N_edge, metrics.N_total - metrics.N_center - metrics.N_microstructure)
    a_edge = max(metrics.A_edge, metrics.A_total - metrics.A_center - metrics.A_microstructure)
    s_edge = metrics.S_scatter_edge if metrics.S_scatter_edge > 0 else metrics.S_scatter

    effective_length = (
        metrics.L_center
        + 0.8 * metrics.L_microstructure
        + 0.3 * metrics.L_edge
    )
    effective_count = (
        1.5 * metrics.N_center
        + 1.2 * metrics.N_microstructure
        + 0.5 * n_edge
        + 1.5 * metrics.N_critical
    )
    effective_area = (
        1.5 * metrics.A_center
        + 1.3 * metrics.A_microstructure
        + 0.5 * a_edge
        + 1.5 * metrics.A_critical
    )
    effective_scatter = (
        1.4 * metrics.S_scatter_center
        + 1.2 * metrics.S_scatter_micro
        + 0.5 * s_edge
    )

    f_center_length = _sat_log(metrics.L_center, norm=120.0, scale=22.0)
    f_effective_length = _sat_log(effective_length, norm=260.0, scale=18.0)
    f_defect_count = _sat_log(effective_count, norm=4.0, scale=18.0)
    f_defect_area = _sat_log(effective_area, norm=900.0, scale=18.0)
    f_scatter = _sat_log(effective_scatter, norm=12.0, scale=20.0)

    score = (
        params.weight_L_center * f_center_length
        + params.weight_L_total * f_effective_length
        + params.weight_N * f_defect_count
        + params.weight_A * f_defect_area
        + params.weight_S_scatter * f_scatter
    )
    score = min(score, 100.0)

    contributors = {
        "center_length": f_center_length,
        "effective_length": f_effective_length,
        "defect_count": f_defect_count,
        "defect_area": f_defect_area,
        "scatter_intensity": f_scatter,
    }

    weighted = {
        "center_length": params.weight_L_center * f_center_length,
        "effective_length": params.weight_L_total * f_effective_length,
        "defect_count": params.weight_N * f_defect_count,
        "defect_area": params.weight_A * f_defect_area,
        "scatter_intensity": params.weight_S_scatter * f_scatter,
    }
    dominant = max(weighted, key=weighted.get) if weighted else ""

    if score <= params.grade_A_max:
        grade, label = "A", "优秀（中心区与微结构区基本完好）"
    elif score <= params.grade_B_max:
        grade, label = "B", "良好（轻微磨损，功能基本稳定）"
    elif score <= params.grade_C_max:
        grade, label = "C", "警告（关键区域受损，建议更换）"
    else:
        grade, label = "D", "报废（关键功能区严重受损）"

    glare = _sat_log(
        metrics.L_center * max(metrics.S_scatter_center, 1.0)
        + 0.6 * metrics.L_microstructure * max(metrics.S_scatter_micro, 1.0),
        norm=1000.0,
        scale=14.0,
    )
    haze = _sat_log(
        metrics.D_micro_density * 1e6 + metrics.N_critical * 2 + metrics.n_crossings,
        norm=12.0,
        scale=12.0,
    )

    assessment = WearAssessment(
        score=score,
        grade=grade,
        grade_label=label,
        glare_index=glare,
        haze_index=haze,
        contributors=contributors,
        dominant_factor=dominant,
        effective_metrics={
            "effective_length": effective_length,
            "effective_count": effective_count,
            "effective_area": effective_area,
            "effective_scatter": effective_scatter,
        },
        conclusion=_generate_conclusion(metrics, grade, dominant, glare, haze),
    )

    logger.info("磨损评分: %.1f/100 [%s] %s", score, grade, label)
    return assessment


def _generate_conclusion(
    metrics: WearMetrics,
    grade: str,
    dominant: str,
    glare: float,
    haze: float,
) -> str:
    parts: list[str] = []

    if grade == "A":
        parts.append("镜片整体状态优秀，中心区和微结构区未见明显功能性磨损。")
    elif grade == "B":
        parts.append("镜片存在轻微磨损，当前仍可继续使用。")
    elif grade == "C":
        parts.append("镜片关键区域已出现可感知磨损，建议尽快评估更换。")
    else:
        parts.append("镜片中心区或微结构功能区严重受损，建议立即更换。")

    factor_names = {
        "center_length": "中心区划痕长度",
        "effective_length": "区域加权总长度",
        "defect_count": "区域加权缺陷数量",
        "defect_area": "区域加权缺陷面积",
        "scatter_intensity": "区域加权散射强度",
    }
    if dominant:
        parts.append(f"当前主要影响因素为：{factor_names.get(dominant, dominant)}。")

    if metrics.N_center > 0:
        parts.append(
            f"中心区共检出 {metrics.N_center} 处缺陷，相关划痕长度约 {metrics.L_center * metrics.pixel_size_mm:.3f} mm。"
        )
    if metrics.N_microstructure > 0:
        parts.append(
            f"微结构区共检出 {metrics.N_microstructure} 处缺陷，是当前功能性磨损的重要来源。"
        )
    if metrics.N_critical > 0:
        parts.append(f"其中严重缺陷 {metrics.N_critical} 处，应优先关注。")

    if glare > 30:
        parts.append("夜间强光场景下存在较高眩光风险。")
    elif glare > 15:
        parts.append("特定角度下可能产生轻至中度眩光。")

    if haze > 30:
        parts.append("微结构区散射偏高，整体清晰度可能下降。")

    return " ".join(parts)
