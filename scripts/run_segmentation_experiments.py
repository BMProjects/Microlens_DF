#!/usr/bin/env python3
"""分割算法实验批次入口."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

PHASE3_ROOT = PROJECT_ROOT / "output/experiments/phase3_segmentation"

EXPERIMENTS = {
    "light_unet": {
        "model_name": "light_unet",
        "encoder_name": None,
        "encoder_weights": None,
    },
    "unetplusplus_r34": {
        "model_name": "unetplusplus",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
    "deeplabv3plus_r34": {
        "model_name": "deeplabv3plus",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
    "fpn_r34": {
        "model_name": "fpn",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
}


def build_command(name: str, preset: dict[str, str | None], args: argparse.Namespace) -> list[str]:
    output_dir = PHASE3_ROOT / "model_zoo" / name
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_msd_segmentation.py"),
        "--model-name",
        str(preset["model_name"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--patch-size",
        str(args.patch_size),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_minutes is not None:
        cmd.extend(["--max-minutes", str(args.max_minutes)])
    if preset["encoder_name"]:
        cmd.extend(["--encoder-name", str(preset["encoder_name"])])
    if preset["encoder_weights"]:
        cmd.extend(["--encoder-weights", str(preset["encoder_weights"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行分割算法对照实验")
    parser.add_argument("--experiment", action="append", default=None, help="实验名，可多次指定")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args()

    selected = args.experiment or list(EXPERIMENTS.keys())
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验名: {unknown}. 可选: {sorted(EXPERIMENTS)}")

    for name in selected:
        cmd = build_command(name, EXPERIMENTS[name], args)
        print()
        print(f"[segmentation] {name}")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
