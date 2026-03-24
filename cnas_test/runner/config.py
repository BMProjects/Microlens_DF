"""CNAS 测试固定配置."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CNAS_ROOT = PROJECT_ROOT / "cnas_test"
TEMPLATES_DIR = CNAS_ROOT / "templates"
DOCS_DIR = CNAS_ROOT / "docs"

DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "output" / "training" / "stage2_cleaned" / "weights" / "best.pt"
)
DEFAULT_TEST_SET = CNAS_ROOT / "manifests" / "test_set_v1.json"
DEFAULT_SAVE_DIR = CNAS_ROOT / "outputs" / "latest"

TILES_DIR = PROJECT_ROOT / "output" / "tile_dataset" / "images" / "val"

OUTPUT_SUBDIRS = {
    "dataset": "dataset",
    "metrics": "metrics",
    "plots": "plots",
    "reports": "reports",
}

EVAL_CONF = 0.001
EVAL_IOU = 0.6
PASS_THRESHOLD = 0.60

CLASS_NAMES = {0: "scratch", 1: "spot", 2: "critical"}

RECOMMENDED_COMMAND = "uv run python -m cnas_test.runner.run_eval"
COMPAT_COMMAND = "python scripts/run_cnas_eval.py"

DATASET_SUMMARY = {
    "overall": {"images": 247, "tiles": 10621, "boxes": 112929},
    "train_val": {"images": 227, "tiles": 9761, "boxes": 104608},
    "holdout": {"images": 20, "tiles": 860, "boxes": 8321},
}

CLASS_DISTRIBUTION = {
    "overall": {"scratch": 75995, "spot": 19997, "critical": 16937},
    "train_val": {"scratch": 70424, "spot": 18506, "critical": 15678},
    "holdout": {"scratch": 5571, "spot": 1491, "critical": 1259},
}
