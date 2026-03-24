#!/usr/bin/env python3
"""批量预处理数据集 — 标定 + 全量目标图像处理.

输出目录结构:
  <output_dir>/
  ├── calibration/              标定产物（B_blur, roi_mask, ring_mask ...）
  ├── images/                   修正后图像（PNG无损，与原文件名对应）
  │   ├── 14r.png
  │   ├── 106l.png
  │   └── ...
  ├── roi_mask.png              ROI掩膜（公用）
  └── quality_report.json       每张图像的配准质量指标

用法:
  python scripts/run_batch_preprocess_dataset.py \\
      --data-dir /path/to/mingyue20260213 \\
      --output-dir /path/to/output_dataset \\
      [--skip-calib]            # 跳过标定，复用已有标定结果
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.preprocessing import PreprocessPipeline
from darkfield_defects.preprocessing.registration import _warp_geometry

# ── 排除模式：不作为目标图像处理 ──────────────────────────────
EXCLUDE_PREFIXES = ("bg",)     # 背景图
EXCLUDE_PREFIXES_EXTRA = ("2.5d",)   # 景深标注图（非普通镜片）
TARGET_EXTENSIONS = (".png", ".bmp")


def is_target(path: Path) -> bool:
    stem_lower = path.stem.lower()
    if path.suffix.lower() not in TARGET_EXTENSIONS:
        return False
    for pat in EXCLUDE_PREFIXES + EXCLUDE_PREFIXES_EXTRA:
        if stem_lower.startswith(pat):
            return False
    return True


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="全量目标图像背景修正预处理")
    parser.add_argument(
        "--data-dir",
        default="/media/bm/Data/Data/Microlens_df/mingyue20260213",
        help="原始数据目录（含 bg-*.png 和目标图像）",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/bm/Data/Data/Microlens_df/mingyue20260213_preprocessed",
        help="输出数据集目录",
    )
    parser.add_argument(
        "--skip-calib",
        action="store_true",
        help="跳过标定，直接加载 <output-dir>/calibration/ 已有结果",
    )
    parser.add_argument(
        "--png-compress",
        type=int,
        default=6,
        help="PNG压缩级别 0-9（默认6，无损）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    cal_dir  = out_dir / "calibration"
    img_dir  = out_dir / "images"
    cal_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compress]

    print("=" * 70)
    print("  暗场镜片图像 — 背景修正预处理（全量数据集）")
    print("=" * 70)
    print(f"  数据目录  : {data_dir}")
    print(f"  输出目录  : {out_dir}")

    # ── 1. 流水线初始化 ────────────────────────────────────────
    pipeline = PreprocessPipeline(timing_log_enabled=False)

    # ── 2. 标定 ────────────────────────────────────────────────
    if args.skip_calib and (cal_dir / "calibration.json").exists():
        print(f"\n[标定] 加载已有结果: {cal_dir}")
        pipeline.load_calibration(cal_dir)
        cal = pipeline.calibration
    else:
        print(f"\n[标定] 开始 ...")
        t0 = time.perf_counter()
        cal = pipeline.calibrate(data_dir, save_dir=cal_dir)
        elapsed = time.perf_counter() - t0
        print(f"[标定] 完成  耗时 {_fmt_time(elapsed)}")
        print(f"       尺寸          : {cal.ref_shape[1]}×{cal.ref_shape[0]}")
        print(f"       离焦 σ        : {cal.template_sigma:.1f}")
        print(f"       ROI 像素      : {int(cal.roi_mask.sum()):,}")
        print(f"       Ring 像素     : {int(cal.ring_mask.sum()):,}")
        print(f"       修正参考      : median={cal.correction_ref_median:.2f}"
              f"  mad={cal.correction_ref_mad:.2f}")

    # 保存公用 ROI 掩膜
    cv2.imwrite(
        str(out_dir / "roi_mask.png"),
        cal.roi_mask.astype(np.uint8) * 255,
        png_params,
    )

    # ── 3. 枚举目标图像 ────────────────────────────────────────
    all_files = sorted(f for f in data_dir.iterdir() if is_target(f))
    n_total = len(all_files)
    print(f"\n[选图] 共找到 {n_total} 张目标图像（已排除 bg-* 和 2.5d-*）")

    if n_total == 0:
        print("  未找到目标图像，退出。")
        sys.exit(1)

    # ── 4. 批量处理 ────────────────────────────────────────────
    report: list[dict] = []
    t_batch_start = time.perf_counter()
    n_ok = 0
    n_fail = 0
    ecc_warn_thresh = 0.75

    print(f"\n[处理] 开始批量处理 {n_total} 张图像 ...\n")

    for idx, fpath in enumerate(all_files, 1):
        out_path = img_dir / fpath.name   # 保持原文件名（扩展名统一为 .png）
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        # 进度前缀
        pct = idx / n_total * 100
        prefix = f"[{idx:3d}/{n_total}] {fpath.name:<18s}"

        # 读取图像
        raw = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"{prefix} ✗ 读取失败")
            report.append({"idx": idx, "file": fpath.name, "error": "read_failed"})
            n_fail += 1
            continue

        # 处理
        try:
            t0 = time.perf_counter()
            result = pipeline.process(raw)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            q = result.quality
            dx, dy, angle, scale = _warp_geometry(result.warp_matrix)

            # 写出修正图像（PNG无损）
            cv2.imwrite(str(out_path), result.image_final, png_params)

            # 状态字符
            warn = " ⚠" if q.reg_ecc < ecc_warn_thresh else "  "
            print(
                f"{prefix}{warn} "
                f"ECC={q.reg_ecc:.4f}  "
                f"dx={dx:+5.1f}  dy={dy:+5.1f}  "
                f"ang={angle:+.3f}°  scl={scale:.4f}  "
                f"{elapsed_ms:.0f}ms  ({pct:.0f}%)"
            )

            report.append({
                "idx":       idx,
                "file":      fpath.name,
                "output":    out_path.name,
                "ecc":       round(q.reg_ecc, 4),
                "dx":        round(dx, 2),
                "dy":        round(dy, 2),
                "angle_deg": round(angle, 4),
                "scale":     round(scale, 6),
                "bg_mean":   round(q.bg_mean, 2),
                "bg_std":    round(q.bg_std, 2),
                "valid":     q.is_valid,
                "total_ms":  round(elapsed_ms, 1),
                "stage_ms":  {k: round(v, 1) for k, v in q.stage_ms.items()},
            })
            n_ok += 1

        except Exception as exc:
            import traceback
            print(f"{prefix} ✗ 异常: {exc}")
            traceback.print_exc()
            report.append({"idx": idx, "file": fpath.name, "error": str(exc)})
            n_fail += 1

    # ── 5. 写质量报告 ──────────────────────────────────────────
    report_path = out_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 6. 终端汇总 ────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_batch_start
    ok_rows  = [r for r in report if "error" not in r]
    fail_rows = [r for r in report if "error" in r]

    print("\n" + "=" * 70)
    print(f"  完成: {n_ok} 成功  {n_fail} 失败   总耗时 {_fmt_time(total_elapsed)}")

    if ok_rows:
        eccs   = [r["ecc"]   for r in ok_rows]
        scales = [r["scale"] for r in ok_rows]
        ms_l   = [r["total_ms"] for r in ok_rows]
        n_warn = sum(1 for e in eccs if e < ecc_warn_thresh)

        print(f"\n  配准统计 ({len(ok_rows)} 张):")
        print(f"    ECC:   avg={np.mean(eccs):.4f}  min={np.min(eccs):.4f}"
              f"  max={np.max(eccs):.4f}  低质量(<{ecc_warn_thresh}): {n_warn}张")
        print(f"    缩放:  avg={np.mean(scales):.4f}  "
              f"max偏差={np.max(np.abs(np.array(scales)-1)):.4f}")
        print(f"    耗时:  avg={np.mean(ms_l):.0f}ms  "
              f"total={_fmt_time(total_elapsed)}")

    if fail_rows:
        print(f"\n  失败列表:")
        for r in fail_rows:
            print(f"    [{r['idx']}] {r['file']} — {r.get('error','?')}")

    print(f"\n  输出目录 : {out_dir.resolve()}")
    print(f"  质量报告 : {report_path.name}")
    print(f"  图像数量 : {len(list(img_dir.glob('*.png')))} 张 PNG")
    print("=" * 70)


if __name__ == "__main__":
    main()
