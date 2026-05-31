#!/usr/bin/env python3
"""仓库内的 MMSegmentation 训练包装入口。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmseg.registry import RUNNERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model with MMSegmentation config")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--work-dir", help="directory to save logs and checkpoints")
    parser.add_argument("--resume", action="store_true", default=False, help="resume from latest checkpoint")
    parser.add_argument("--amp", action="store_true", default=False, help="force-enable AMP wrapper")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config fields, e.g. load_from=/path/to/latest.pth",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 默认关闭 mmseg 可视化钩子，避免训练被额外的可视化依赖阻塞。
    if "default_hooks" in cfg and "visualization" in cfg.default_hooks:
        cfg.default_hooks.pop("visualization")
    cfg.visualizer = dict(type="Visualizer", vis_backends=[])
    cfg.vis_backends = []

    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log("AMP is already enabled in config.", logger="current", level=logging.WARNING)
        else:
            assert optim_wrapper == "OptimWrapper", f"Unsupported optim wrapper for --amp: {optim_wrapper}"
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    cfg.resume = args.resume

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == "__main__":
    main()
