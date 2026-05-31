#!/usr/bin/env python3
"""稳定版桥接核查入口：按图像逐张启动完整推理并汇总结果。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DET_WEIGHTS = (
    PROJECT_ROOT
    / "output"
    / "experiments"
    / "phase3e"
    / "detection_training"
    / "b2_nwd_only_phase3e"
    / "weights"
    / "best.pt"
)


def run_single(
    stem: str,
    weights: Path,
    seg_weights: Path,
    conf: float,
    out_dir: Path,
    no_vis: bool,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/infer_full_pipeline.py"),
        "--weights",
        str(weights),
        "--stems",
        stem,
        "--conf",
        str(conf),
        "--out-dir",
        str(out_dir),
        "--use-segmentation",
        "--seg-weights",
        str(seg_weights),
    ]
    if no_vis:
        cmd.append("--no-vis")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def main() -> None:
    parser = argparse.ArgumentParser(description="逐张桥接核查并汇总结果")
    parser.add_argument("--stems", nargs="+", required=True, help="要核查的图像 stem 列表")
    parser.add_argument("--seg-weights", type=Path, required=True, help="分割权重路径")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--weights", type=Path, default=DEFAULT_DET_WEIGHTS, help="检测模型权重")
    parser.add_argument("--conf", type=float, default=0.2, help="检测置信度")
    parser.add_argument("--no-vis", action="store_true", help="禁用可视化输出")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   稳定版桥接核查（逐张子进程）                         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  检测模型:  {args.weights.name}")
    print(f"  分割模型:  {args.seg_weights.name}")
    print(f"  图像数:    {len(args.stems)}")
    print(f"  置信度:    {args.conf}")
    print(f"  输出:      {args.out_dir}/")
    print()

    started = time.time()
    summary: list[dict] = []
    failures: list[dict] = []

    for idx, stem in enumerate(args.stems, 1):
        print(f"  [{idx}/{len(args.stems)}] {stem} ...", end=" ", flush=True)
        ok, output = run_single(
            stem=stem,
            weights=args.weights,
            seg_weights=args.seg_weights,
            conf=args.conf,
            out_dir=args.out_dir,
            no_vis=args.no_vis,
        )
        report_path = args.out_dir / stem / "report.json"
        if ok and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assessment = report.get("assessment", {})
            inference = report.get("inference", {})
            detections = report.get("detections", {})
            print(
                f"Grade {assessment.get('grade', '?')} ({assessment.get('score', 0):.0f})  "
                f"defects={inference.get('after_connect', 0)}  "
                f"chains={inference.get('scratch_chains', 0)}  "
                f"{inference.get('time_sec', 0):.1f}s"
            )
            summary.append({
                "stem": stem,
                "grade": assessment.get("grade"),
                "score": assessment.get("score"),
                "n_defects": inference.get("after_connect"),
                "scratch": detections.get("scratch"),
                "spot": detections.get("spot"),
                "critical": detections.get("critical"),
            })
        else:
            tail = "\n".join(output.strip().splitlines()[-20:])
            print("✗ failed")
            failures.append({
                "stem": stem,
                "returncode_ok": ok,
                "output_tail": tail,
            })

    elapsed = time.time() - started
    payload = {
        "model": str(args.weights),
        "seg_weights": str(args.seg_weights),
        "conf": args.conf,
        "n_images": len(args.stems),
        "elapsed_sec": round(elapsed, 1),
        "images": summary,
        "failures": failures,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 56)
    print(f"  完成 {len(summary)}/{len(args.stems)} 张  总耗时 {elapsed:.1f}s")
    if failures:
        print(f"  失败 {len(failures)} 张")
    print(f"  汇总报告: {summary_path}")
    print()


if __name__ == "__main__":
    main()
