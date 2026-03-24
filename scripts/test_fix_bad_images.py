"""测试修复后的配准流水线 — 针对已知问题图像."""

from __future__ import annotations

import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 项目路径 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from darkfield_defects.preprocessing.pipeline import PreprocessPipeline

# ── 已知问题图像列表 ──────────────────────────────────────────────────────────
BAD_IMAGES = [
    "1l.png", "1r.png",
    "13l.png", "13r.png",
    "14l.png",
    "15l.png", "15r.png",
    "19l.png",
    "23l.png",
    "37l.png", "37r.png",
    "44r.png",
    "45l.png", "45r.png",
    "51r.png",
    "53l.png", "53r.png",
    "54l.png", "54r.png",
    "60r.png",
    "69l.png", "69r.png",
    "70r.png",
    "80l.png", "80r.png",
    "84r.png",
    "88r.png",
    "106r.png",
    "111l.png", "111r.png",
    "117l.png", "117r.png",
]

DATA_DIR   = Path("/media/bm/Data/Data/Microlens_df/mingyue20260213")
BG_DIR     = DATA_DIR
OUTPUT_DIR = ROOT / "output" / "preprocess_fix"
CALIB_DIR  = ROOT / "output" / "preprocess_test" / "calibration"  # 复用已有标定


def fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def save_compare(raw: np.ndarray, corrected: np.ndarray,
                 roi_mask: np.ndarray, ring_mask: np.ndarray,
                 out_dir: Path, name: str,
                 meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # A: 原图 + ROI/ring 边界
    raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    ring_edge = cv2.dilate(ring_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) - ring_mask.astype(np.uint8)
    roi_edge  = cv2.dilate(roi_mask.astype(np.uint8),  np.ones((3, 3), np.uint8)) - roi_mask.astype(np.uint8)
    raw_bgr[ring_edge > 0] = (0, 0, 255)
    raw_bgr[roi_edge  > 0] = (0, 255, 0)
    cv2.imwrite(str(out_dir / f"{name}_A_raw.jpg"), raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # B: 修正后图像
    cv2.imwrite(str(out_dir / f"{name}_B_corrected.jpg"), corrected, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # C: 并排对比
    h, w = raw.shape
    scale = 800 / max(h, w)
    rh, rw = int(h * scale), int(w * scale)
    left  = cv2.resize(raw,       (rw, rh))
    right = cv2.resize(corrected, (rw, rh))
    pair  = np.hstack([left, right])
    pair_bgr = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
    label = (f"dx={meta['dx']:+.1f} dy={meta['dy']:+.1f} "
             f"ang={meta['angle']:+.3f}° sc={meta['scale']:.4f} "
             f"ECC={meta['score']:.3f}")
    cv2.putText(pair_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{name}_C_compare.jpg"), pair_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=str(DATA_DIR))
    ap.add_argument("--bg-dir",     default=str(BG_DIR))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--calib-dir",  default=str(CALIB_DIR))
    ap.add_argument("--skip-calib", action="store_true")
    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    bg_dir     = Path(args.bg_dir)
    out_dir    = Path(args.output_dir)
    calib_dir  = Path(args.calib_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  问题图像修复验证测试")
    print("=" * 70)

    pipeline = PreprocessPipeline()

    # 标定
    if args.skip_calib and calib_dir.exists():
        print(f"\n[标定] 加载已有标定: {calib_dir}")
        pipeline.load_calibration(calib_dir)
        cal = pipeline.calibration
        print(f"       ROI px: {cal.roi_mask.sum():,}  Ring px: {cal.ring_mask.sum():,}")
    else:
        print(f"\n[标定] 开始 ...")
        t0 = time.perf_counter()
        pipeline.calibrate(bg_dir, save_dir=calib_dir)
        print(f"[标定] 完成  耗时 {time.perf_counter()-t0:.1f}s")

    cal = pipeline.calibration

    # 筛选存在的图像
    targets = []
    for name in BAD_IMAGES:
        p = data_dir / name
        if p.exists():
            targets.append(p)
        else:
            print(f"  [跳过] {name} (文件不存在)")

    print(f"\n共 {len(targets)} 张问题图像待验证\n")
    print(f"{'#':>3}  {'文件':<15}  {'dx':>7}  {'dy':>7}  {'angle':>8}  {'scale':>8}  {'ECC':>6}  {'ms':>6}")
    print("-" * 75)

    results = []
    ok_count = 0
    fail_count = 0

    for i, fpath in enumerate(targets, 1):
        name = fpath.stem
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{i:>3}  {name:<15}  [读取失败]")
            fail_count += 1
            continue

        t0 = time.perf_counter()
        try:
            result = pipeline.process(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            warp = result.warp_matrix
            a00, a01 = float(warp[0, 0]), float(warp[0, 1])
            a10, a11 = float(warp[1, 0]), float(warp[1, 1])
            dx    = float(warp[0, 2])
            dy    = float(warp[1, 2])
            angle = float(np.degrees(np.arctan2(a10, a00)))
            sx    = float(np.hypot(a00, a10))
            sy    = float(np.hypot(a01, a11))
            scale = 0.5 * (sx + sy)
            score = result.quality.reg_ecc

            flag = "✓" if score >= 0.75 else "⚠"
            print(f"{i:>3}  {name:<15}  {dx:>+7.1f}  {dy:>+7.1f}  {angle:>+8.4f}  {scale:>8.5f}  {score:>6.3f}  {elapsed_ms:>6.0f}  {flag}")

            meta = dict(dx=dx, dy=dy, angle=angle, scale=scale, score=score, ms=elapsed_ms)
            results.append({"file": fpath.name, **meta, "ok": score >= 0.75})
            ok_count += (1 if score >= 0.75 else 0)
            fail_count += (0 if score >= 0.75 else 1)

            # 保存输出图像
            save_compare(
                raw=img,
                corrected=result.image_final,
                roi_mask=cal.roi_mask,
                ring_mask=cal.ring_mask,
                out_dir=out_dir / "results",
                name=f"{i:02d}_{name}",
                meta=meta,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"{i:>3}  {name:<15}  [异常: {e}]  {elapsed_ms:.0f}ms")
            fail_count += 1
            results.append({"file": fpath.name, "error": str(e), "ok": False})

    # 汇总统计
    print("=" * 70)
    ok_all  = [r for r in results if r.get("ok")]
    scores  = [r["score"] for r in ok_all]
    print(f"  通过 (ECC≥0.75): {ok_count}/{len(targets)}")
    if scores:
        print(f"  ECC:  avg={np.mean(scores):.4f}  min={np.min(scores):.4f}  max={np.max(scores):.4f}")
    print(f"  输出目录: {out_dir}")

    # 保存 JSON
    report = {"tested": len(targets), "passed": ok_count, "failed": fail_count, "results": results}
    with open(out_dir / "fix_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 生成汇总缩略图网格（并排对比）
    thumbs = []
    for r in results:
        if not r.get("ok") and "error" in r:
            continue
        cmp_path = out_dir / "results" / f"{results.index(r)+1:02d}_{Path(r['file']).stem}_C_compare.jpg"
        if cmp_path.exists():
            t = cv2.imread(str(cmp_path))
            if t is not None:
                label = f"{Path(r['file']).stem}  ECC={r.get('score',0):.3f}"
                cv2.putText(t, label, (5, t.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                thumbs.append(cv2.resize(t, (800, 300)))

    if thumbs:
        grid = np.vstack(thumbs)
        cv2.imwrite(str(out_dir / "summary_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"  summary_grid.jpg: {grid.shape[1]}×{grid.shape[0]}")

    print("=" * 70)


if __name__ == "__main__":
    main()
