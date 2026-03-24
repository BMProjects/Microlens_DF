"""模板背景预处理模块.

公共 API:
  PreprocessPipeline  — 主流水线（标定 + 处理）
  CalibrationResult   — 标定产物（B_blur, roi_mask, ring_mask ...）
  PipelineResult      — 单帧处理结果（image_final, roi_mask, quality）
  QualityMetrics      — 单帧质量指标（ECC, dx/dy/angle/scale, bg_std ...）

子模块（内部使用，可按需直接导入）:
  background_fusion       — 多帧平均 + 离焦模拟
  brightness_correction   — 线性亮度修正 + 最终化
  registration            — ring模板配准（相位相关 + ECC仿射）
  roi_builder             — 高光ring检测 + ROI提取
"""

from darkfield_defects.preprocessing.pipeline import (
    CalibrationResult,
    PipelineResult,
    PreprocessPipeline,
    QualityMetrics,
)

__all__ = [
    "PreprocessPipeline",
    "CalibrationResult",
    "PipelineResult",
    "QualityMetrics",
]
