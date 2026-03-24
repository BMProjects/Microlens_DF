"""异常层次 — 按类型分类的自定义异常."""


class DarkfieldError(Exception):
    """暗场缺陷检测系统基础异常."""


class ImageLoadError(DarkfieldError):
    """图像加载失败."""


class DetectionError(DarkfieldError):
    """检测过程异常."""


class ConfigurationError(DarkfieldError):
    """配置参数错误."""


class PreprocessError(DarkfieldError):
    """预处理异常（背景校正/ROI 等）."""


class ScoringError(DarkfieldError):
    """磨损评分异常."""
