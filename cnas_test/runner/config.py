"""CNAS 测试固定配置."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CNAS_ROOT = PROJECT_ROOT / "cnas_test"
TEMPLATES_DIR = CNAS_ROOT / "templates"
DOCS_DIR = CNAS_ROOT / "docs"

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "output"
    / "experiments"
    / "phase3e"
    / "detection_training"
    / "b2_nwd_only_phase3e"
    / "weights"
    / "best.pt"
)
DEFAULT_TEST_SET = CNAS_ROOT / "manifests" / "full_dataset_v1.json"
DEFAULT_OUTPUTS_ROOT = CNAS_ROOT / "outputs"


def make_timestamped_save_dir(now: datetime | None = None) -> Path:
    """Return a per-run output directory named by local test time."""
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUTS_ROOT / ts


DEFAULT_SAVE_DIR = make_timestamped_save_dir()

SOFTWARE_NAME = "镜片磨损智能识别算法"
SOFTWARE_VERSION = "LWIA-Det v1.0.0"
VERSION_NOTE = "内部训练注释：B2_nwd_only_phase3e"
MODEL_DESCRIPTION = "多尺度深度目标检测网络，结合小目标定位优化策略"
DATASET_NAME = "离焦微结构镜片磨损识别数据集"
STANDARD_SAMPLE_UNIT = "标准化图像样本"
DATASET_CONSTRUCTION_METHOD = (
    "数据集由离焦微结构镜片磨损显微图像及人工复核的目标框标注组成；"
    "按固定尺度和标注坐标映射规则生成 640 × 640 像素的标准化图像样本，"
    "并同步生成三类缺陷目标框标签。无缺陷目标的背景样本保留在评测范围内。"
)

TILE_DATASET_ROOT = PROJECT_ROOT / "output" / "tile_dataset"
TILES_DIRS = [
    TILE_DATASET_ROOT / "images" / "train",
    TILE_DATASET_ROOT / "images" / "val",
]

OUTPUT_SUBDIRS = {
    "dataset": "dataset",
    "metrics": "metrics",
    "plots": "plots",
    "reports": "reports",
    "provenance": "provenance",
    "screenshots": "screenshots",
}

# EVAL_CONF：评测低置信度阈值，用于完整画 PR 曲线（不影响 AP 计算）。
# EVAL_IOU：传入 ultralytics.val 的 NMS IoU 阈值，用于去重叠预测框。
# EVAL_MATCH_IOU：AP50 定义中预测-真值匹配的 IoU 阈值，固定为 0.5（与 mAP50 命名一致）。
EVAL_CONF = 0.001
EVAL_IOU = 0.6
EVAL_MATCH_IOU = 0.5
PASS_THRESHOLD = 0.60

CLASS_NAMES = {0: "scratch", 1: "spot", 2: "critical"}

RECOMMENDED_COMMAND = "uv run python -m cnas_test.runner.run_eval"
COMPAT_COMMAND = "python scripts/run_cnas_eval.py"

DATASET_SUMMARY = {
    "overall": {"images": 247, "tiles": 10621, "boxes": 90325},
    "train": {"images": 197, "tiles": 8471, "boxes": 68339},
    "val": {"images": 50, "tiles": 2150, "boxes": 21986},
    "holdout": {"images": 20, "tiles": 860, "boxes": 8321},
}

CLASS_DISTRIBUTION = {
    "overall": {"scratch": 58032, "spot": 17475, "critical": 14818},
    "train": {"scratch": 43200, "spot": 13766, "critical": 11373},
    "val": {"scratch": 14832, "spot": 3709, "critical": 3445},
    "holdout": {"scratch": 5571, "spot": 1491, "critical": 1259},
}
