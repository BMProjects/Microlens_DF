"""CNAS 测试命令行入口."""

from __future__ import annotations

import argparse
from pathlib import Path

from cnas_test.runner.config import DEFAULT_SAVE_DIR, DEFAULT_TEST_SET, DEFAULT_WEIGHTS
from cnas_test.runner.evaluator import run_cnas_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNAS 缺陷检测准确率评测（独立子系统）")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cnas_eval(
        args.weights,
        test_set_path=args.test_set,
        save_dir=args.save_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
