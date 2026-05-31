#!/usr/bin/env python3
"""将 HRNet-OCR 官方仓实验结果回填为本项目统一 summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE3_ROOT = PROJECT_ROOT / "output/experiments/phase3_segmentation"
OUTPUTS = {
    "msd": PHASE3_ROOT / "batch2_hrnet_ocr_official_msd",
    "private": PHASE3_ROOT / "batch2_hrnet_ocr_official_private",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="登记 HRNet 官方实验结果")
    parser.add_argument("--stage", choices=sorted(OUTPUTS), required=True)
    parser.add_argument("--best-epoch", type=int, required=True)
    parser.add_argument("--best-val-miou", type=float, default=None)
    parser.add_argument("--best-loss", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    out_dir = OUTPUTS[args.stage]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))

    payload = {
        "framework": "hrnet_official",
        "stage": args.stage,
        "status": "completed",
        "model": "HRNet-OCR-W18",
        "best_epoch": args.best_epoch,
        "best_val_miou": args.best_val_miou,
        "best_loss": args.best_loss,
        "checkpoint": str(args.checkpoint) if args.checkpoint else existing.get("checkpoint"),
        "output_dir": str(out_dir),
        "notes": args.notes or existing.get("notes", ""),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
