#!/usr/bin/env python3
"""兼容入口：对外保留 run_cnas_eval API，内部转发到独立 CNAS 子系统."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnas_test.runner.config import DEFAULT_TEST_SET, DEFAULT_WEIGHTS
from cnas_test.runner.evaluator import run_cnas_eval as _run_cnas_eval


def run_cnas_eval(
    weights_path: Path = DEFAULT_WEIGHTS,
    save_dir: Path | None = None,
    tag: str = "",
    test_set_path: Path = DEFAULT_TEST_SET,
    verbose: bool = True,
) -> dict:
    """兼容旧实验脚本的评测调用.

    `tag` 参数保留，以免旧脚本签名失配；当前独立子系统不再使用它控制核心逻辑。
    """
    result = _run_cnas_eval(
        weights_path,
        test_set_path=test_set_path,
        save_dir=save_dir,
        verbose=verbose,
    )
    if tag:
        result["tag"] = tag
    return result


if __name__ == "__main__":
    _run_cnas_eval(DEFAULT_WEIGHTS, verbose=True)
