#!/usr/bin/env python3
"""背景修正预处理完整测试：标定 + 随机N张目标图可视化输出.

输出结构:
  output/preprocess_test/
  ├── calibration/          标定产物 (B_avg, B_blur, ring_mask, roi_mask ...)
  ├── results/
  │   ├── 01_14r/
  │   │   ├── A_raw_with_roi.jpg       原图 + ROI/ring叠加
  │   │   ├── B_corrected.jpg          背景修正后图像
  │   │   ├── C_compare.jpg            左右并排对比
  │   │   └── D_histogram.jpg          亮度直方图对比
  │   ├── 02_106l/
  │   │   └── ...
  │   └── ...
  ├── summary_grid.jpg      所有图像并排缩略图一览
  └── quality_report.json   量化指标汇总
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from darkfield_defects.preprocessing.pipeline import PreprocessPipeline
from darkfield_defects.preprocessing.registration import _invert_affine_2x3, _warp_geometry


# ═══════════════════════════════════════════════════════════════
# 可视化工具
# ═══════════════════════════════════════════════════════════════

def _to_u8(img: np.ndarray) -> np.ndarray:
    """归一化到 uint8，保留内容（用于可视化）."""
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


def _clip_u8(img: np.ndarray) -> np.ndarray:
    """直接 clip 到 [0,255]（保留绝对亮度关系）."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_mask_border(bgr: np.ndarray, mask: np.ndarray,
                      color: tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    """在 bgr 图上绘制 mask 的轮廓线."""
    out = bgr.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def _resize_thumb(img: np.ndarray, max_long: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return img
    s = max_long / long_side
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str,
               pos: tuple[int, int] = (20, 55),
               scale: float = 1.4) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 100), 2, cv2.LINE_AA)
    return out


def _save_jpg(path: Path, img: np.ndarray, quality: int = 92) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


# ═══════════════════════════════════════════════════════════════
# 各个可视化图生成
# ═══════════════════════════════════════════════════════════════

def make_roi_overlay(raw: np.ndarray,
                     roi_mask: np.ndarray,
                     ring_mask: np.ndarray,
                     warp: np.ndarray,
                     info_text: str) -> np.ndarray:
    """A: 原图 + ROI(青色边界) + ring(黄色边界) + 仿射参数文字."""
    h, w = raw.shape[:2]
    inv = _invert_affine_2x3(warp)

    def _warp_mask(m: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(
            m.astype(np.uint8), inv.astype(np.float32), (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)

    roi_raw  = _warp_mask(roi_mask)
    ring_raw = _warp_mask(ring_mask)

    bgr = cv2.cvtColor(_to_u8(raw), cv2.COLOR_GRAY2BGR)
    bgr = _draw_mask_border(bgr, roi_raw,  color=(255, 200, 0),   thickness=4)   # 青
    bgr = _draw_mask_border(bgr, ring_raw, color=(0,   200, 255), thickness=4)   # 黄
    bgr = _put_label(bgr, "A  RAW + ROI boundary")
    # 仿射参数右下角
    h2, w2 = bgr.shape[:2]
    lines = info_text.split("\n")
    y0 = h2 - len(lines) * 38 - 10
    for i, line in enumerate(lines):
        y = y0 + i * 38
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (50, 255, 180), 2, cv2.LINE_AA)
    return bgr


def make_corrected_img(corrected: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """B: 修正后图像（ROI外置黑，用归一化方式显示划痕对比）."""
    # 两种显示：绝对值（保留亮度关系）
    arr = corrected.astype(np.float64)
    arr = np.where(roi_mask, arr, 0.0)
    u8 = _to_u8(arr)   # 归一化以便看清划痕
    bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    bgr = _put_label(bgr, "B  CORRECTED (normalized)")
    return bgr


def make_compare(raw: np.ndarray,
                 corrected: np.ndarray,
                 roi_mask: np.ndarray,
                 ring_mask: np.ndarray,
                 warp: np.ndarray) -> np.ndarray:
    """C: 左右并排 — 原图(ROI边界) vs 修正图."""
    h, w = raw.shape[:2]
    inv = _invert_affine_2x3(warp)

    roi_raw = cv2.warpAffine(
        roi_mask.astype(np.uint8), inv.astype(np.float32), (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)

    left_bgr  = cv2.cvtColor(_to_u8(raw), cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(_to_u8(np.where(roi_mask, corrected, 0.0)), cv2.COLOR_GRAY2BGR)

    left_bgr  = _draw_mask_border(left_bgr,  roi_raw,  (0, 255, 200), 3)
    right_bgr = _draw_mask_border(right_bgr, roi_mask, (0, 255, 200), 3)

    left_bgr  = _put_label(left_bgr,  "C  LEFT: RAW")
    right_bgr = _put_label(right_bgr, "C  RIGHT: CORRECTED")

    # 中间分隔线
    sep = np.full((h, 6, 3), 200, dtype=np.uint8)
    return np.hstack([left_bgr, sep, right_bgr])


def make_histogram(raw: np.ndarray,
                   corrected: np.ndarray,
                   roi_mask: np.ndarray,
                   fname: str) -> np.ndarray:
    """D: ROI区域亮度直方图对比图（原图 vs 修正后）."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_vals  = raw[roi_mask].astype(np.float64).ravel()
    corr_vals = corrected[roi_mask].astype(np.float64).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")

    # 原图直方图
    axes[0].hist(raw_vals,  bins=200, color="#4fc3f7", alpha=0.85, range=(0, 255))
    axes[0].set_title(f"RAW — ROI  (median={np.median(raw_vals):.1f})",  color="white", fontsize=11)
    axes[0].set_xlabel("Pixel value", color="#aaa")
    axes[0].set_ylabel("Count",       color="#aaa")

    # 修正后直方图
    vmin = float(corr_vals.min())
    vmax = float(corr_vals.max())
    rng  = (min(vmin, -50), max(vmax, 50))
    axes[1].hist(corr_vals, bins=200, color="#81c784", alpha=0.85, range=rng)
    axes[1].axvline(0, color="#ff7043", linewidth=1.5, linestyle="--", label="zero")
    axes[1].set_title(
        f"CORRECTED — ROI  (median={np.median(corr_vals):.1f}  std={np.std(corr_vals):.1f})",
        color="white", fontsize=11,
    )
    axes[1].set_xlabel("Pixel value", color="#aaa")
    axes[1].legend(facecolor="#333", labelcolor="white", fontsize=9)

    fig.suptitle(f"Histogram comparison — {fname}", color="white", fontsize=12, y=1.02)
    plt.tight_layout()

    # 渲染为 numpy BGR（兼容新旧 matplotlib API）
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    except AttributeError:
        # matplotlib >= 3.8: buffer_rgba
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr


# ═══════════════════════════════════════════════════════════════
# Summary grid
# ═══════════════════════════════════════════════════════════════

def make_summary_grid(thumb_pairs: list[tuple[str, np.ndarray, np.ndarray]],
                      cols: int = 2) -> np.ndarray:
    """每对 (label, raw_thumb, corrected_thumb) 拼成一行，多行排列."""
    cell_w, cell_h = 900, 600

    rows_out = []
    for label, raw_t, corr_t in thumb_pairs:
        def _fit(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            s = min(cell_w / w, cell_h / h)
            resized = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            oh = (cell_h - resized.shape[0]) // 2
            ow = (cell_w - resized.shape[1]) // 2
            canvas[oh:oh+resized.shape[0], ow:ow+resized.shape[1]] = resized
            return canvas

        raw_bgr  = cv2.cvtColor(raw_t,  cv2.COLOR_GRAY2BGR) if raw_t.ndim == 2  else raw_t
        corr_bgr = cv2.cvtColor(corr_t, cv2.COLOR_GRAY2BGR) if corr_t.ndim == 2 else corr_t

        left  = _fit(raw_bgr)
        right = _fit(corr_bgr)
        sep   = np.full((cell_h, 4, 3), 60, dtype=np.uint8)
        pair  = np.hstack([left, sep, right])

        # 顶部标签条
        label_bar = np.zeros((44, pair.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 240, 120), 2, cv2.LINE_AA)
        rows_out.append(np.vstack([label_bar, pair]))

    # 两列排列
    if len(rows_out) % cols != 0:
        rows_out.append(np.zeros_like(rows_out[0]))   # 补空格

    grid_rows = []
    for i in range(0, len(rows_out), cols):
        row_imgs = rows_out[i:i+cols]
        # 高度对齐
        max_h = max(r.shape[0] for r in row_imgs)
        padded = []
        for r in row_imgs:
            if r.shape[0] < max_h:
                pad = np.zeros((max_h - r.shape[0], r.shape[1], 3), dtype=np.uint8)
                r = np.vstack([r, pad])
            padded.append(r)
        vsep = np.full((max_h, 6, 3), 40, dtype=np.uint8)
        row = padded[0]
        for p in padded[1:]:
            row = np.hstack([row, vsep, p])
        grid_rows.append(row)

    hsep = np.full((6, grid_rows[0].shape[1], 3), 40, dtype=np.uint8)
    grid = grid_rows[0]
    for gr in grid_rows[1:]:
        # 宽度对齐
        if gr.shape[1] != grid.shape[1]:
            gr = cv2.resize(gr, (grid.shape[1], gr.shape[0]), interpolation=cv2.INTER_AREA)
        grid = np.vstack([grid, hsep, gr])

    return grid


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="背景修正预处理可视化测试")
    parser.add_argument("--data-dir",      default="/media/bm/Data/Data/Microlens_df/mingyue20260213")
    parser.add_argument("--output-dir",    default="output/preprocess_test")
    parser.add_argument("--n",             type=int,   default=10,   help="随机抽取图像数")
    parser.add_argument("--seed",          type=int,   default=42,   help="随机种子")
    parser.add_argument("--skip-calib",    action="store_true",       help="跳过标定，复用已有结果")
    parser.add_argument("--blur-sigma",       type=float, default=30.0)
    parser.add_argument("--center-ratio",     type=float, default=1.25)
    parser.add_argument("--max-scale",        type=float, default=0.05)
    parser.add_argument("--downsample",       type=float, default=0.75)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    out_root   = Path(args.output_dir)
    cal_dir    = out_root / "calibration"
    result_dir = out_root / "results"
    cal_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  背景修正预处理测试")
    print("=" * 65)

    # ── 1. 流水线初始化 ──────────────────────────────────────
    pipeline = PreprocessPipeline(
        template_blur_sigma=args.blur_sigma,
        template_highlight_center_ratio=args.center_ratio,
        max_scale_deviation=args.max_scale,
        registration_downsample=args.downsample,
        timing_log_enabled=False,   # 静默模式，脚本自己打印
    )

    # ── 2. 标定 ───────────────────────────────────────────────
    if args.skip_calib and (cal_dir / "calibration.json").exists():
        print(f"\n[标定] 加载已有标定: {cal_dir}")
        pipeline.load_calibration(cal_dir)
    else:
        print(f"\n[标定] 开始 (数据目录: {data_dir})")
        t0 = time.perf_counter()
        cal = pipeline.calibrate(data_dir, save_dir=cal_dir)
        elapsed = time.perf_counter() - t0
        print(f"[标定] 完成  耗时 {elapsed:.1f}s")
        print(f"       图像尺寸   : {cal.ref_shape[1]}×{cal.ref_shape[0]}")
        print(f"       ROI 像素   : {int(cal.roi_mask.sum()):,}")
        print(f"       Ring 像素  : {int(cal.ring_mask.sum()):,}")
        print(f"       修正参考   : median={cal.correction_ref_median:.2f}  mad={cal.correction_ref_mad:.2f}")
        print(f"       离焦 σ     : {cal.template_sigma:.1f}")

    cal = pipeline.calibration

    # ── 3. 随机选取目标图 ─────────────────────────────────────
    all_targets = sorted(
        f for f in data_dir.iterdir()
        if f.suffix.lower() in (".png", ".bmp")
        and not f.stem.lower().startswith("bg")
    )
    random.seed(args.seed)
    selected = random.sample(all_targets, min(args.n, len(all_targets)))

    print(f"\n[选图] 共 {len(all_targets)} 张目标图，随机选取 {len(selected)} 张 (seed={args.seed}):")
    for f in selected:
        print(f"       {f.name}")

    # ── 4. 逐张处理并生成可视化 ──────────────────────────────
    report       = []
    thumb_pairs  = []   # 用于 summary grid

    for idx, fpath in enumerate(selected, 1):
        tag    = f"{idx:02d}_{fpath.stem}"
        img_dir = result_dir / tag
        img_dir.mkdir(exist_ok=True)

        print(f"\n[{idx:02d}/{len(selected)}] {fpath.name}")
        raw = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"       ✗ 读取失败，跳过")
            report.append({"idx": idx, "file": fpath.name, "error": "read_failed"})
            continue

        try:
            t0 = time.perf_counter()
            result = pipeline.process(raw)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            q   = result.quality
            dx, dy, angle, scale = _warp_geometry(result.warp_matrix)

            print(f"       平移: dx={dx:+.1f}px  dy={dy:+.1f}px")
            print(f"       旋转: {angle:+.4f}°    缩放: {scale:.6f}")
            print(f"       ECC:  {q.reg_ecc:.4f}          耗时: {elapsed_ms:.0f}ms")

            # 仿射信息文本（写在图像上）
            info = (
                f"dx={dx:+.1f}  dy={dy:+.1f}  angle={angle:+.4f}deg\n"
                f"scale={scale:.5f}  ECC={q.reg_ecc:.4f}  [{elapsed_ms:.0f}ms]"
            )

            # ── A: 原图 + ROI/ring 边界叠加 ──────────────────
            img_a = make_roi_overlay(raw, cal.roi_mask, cal.ring_mask,
                                     result.warp_matrix, info)
            _save_jpg(img_dir / "A_raw_with_roi.jpg",
                      _resize_thumb(img_a, 2400))

            # ── B: 修正后图像 ─────────────────────────────────
            img_b = make_corrected_img(result.image_final.astype(np.float64),
                                       result.roi_mask)
            _save_jpg(img_dir / "B_corrected.jpg",
                      _resize_thumb(img_b, 2400))

            # ── C: 并排对比 ───────────────────────────────────
            img_c = make_compare(raw, result.image_final.astype(np.float64),
                                 cal.roi_mask, cal.ring_mask, result.warp_matrix)
            _save_jpg(img_dir / "C_compare.jpg",
                      _resize_thumb(img_c, 4000))

            # ── D: 直方图 ─────────────────────────────────────
            try:
                img_d = make_histogram(raw, result.image_final.astype(np.float64),
                                       result.roi_mask, fpath.name)
                _save_jpg(img_dir / "D_histogram.jpg", img_d)
            except Exception as hist_exc:
                print(f"       ! 直方图生成跳过: {hist_exc}")

            # ── 缩略图（用于 summary grid）───────────────────
            raw_thumb  = _resize_thumb(_to_u8(raw), 600)
            corr_thumb = _resize_thumb(_to_u8(
                np.where(result.roi_mask, result.image_final.astype(np.float64), 0)), 600)
            thumb_pairs.append((f"{tag}  ECC={q.reg_ecc:.3f}", raw_thumb, corr_thumb))

            report.append({
                "idx":       idx,
                "file":      fpath.name,
                "dx":        round(dx, 2),
                "dy":        round(dy, 2),
                "angle_deg": round(angle, 4),
                "scale":     round(scale, 6),
                "ecc":       round(q.reg_ecc, 4),
                "total_ms":  round(elapsed_ms, 1),
                "stage_ms":  {k: round(v, 1) for k, v in q.stage_ms.items()},
            })

        except Exception as exc:
            import traceback
            print(f"       ✗ 处理异常: {exc}")
            traceback.print_exc()
            report.append({"idx": idx, "file": fpath.name, "error": str(exc)})

    # ── 5. Summary grid ───────────────────────────────────────
    if thumb_pairs:
        print("\n[汇总] 生成 summary_grid.jpg ...")
        grid = make_summary_grid(thumb_pairs, cols=2)
        _save_jpg(out_root / "summary_grid.jpg", grid, quality=88)
        print(f"       尺寸: {grid.shape[1]}×{grid.shape[0]}")

    # ── 6. 质量报告 JSON ──────────────────────────────────────
    with open(out_root / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── 7. 终端汇总 ───────────────────────────────────────────
    ok   = [r for r in report if "error" not in r]
    fail = [r for r in report if "error" in r]

    print("\n" + "=" * 65)
    print(f"  完成: {len(ok)} 成功  {len(fail)} 失败")
    if ok:
        eccs   = [r["ecc"]       for r in ok]
        dxs    = [r["dx"]        for r in ok]
        dys    = [r["dy"]        for r in ok]
        angles = [r["angle_deg"] for r in ok]
        scales = [r["scale"]     for r in ok]
        ms_l   = [r["total_ms"]  for r in ok]
        print(f"  ECC:   avg={np.mean(eccs):.4f}  min={np.min(eccs):.4f}  max={np.max(eccs):.4f}")
        print(f"  dx:    avg={np.mean(dxs):+.1f}px   范围 [{np.min(dxs):+.1f}, {np.max(dxs):+.1f}]")
        print(f"  dy:    avg={np.mean(dys):+.1f}px   范围 [{np.min(dys):+.1f}, {np.max(dys):+.1f}]")
        print(f"  旋转:  avg={np.mean(angles):+.4f}°  max_abs={np.max(np.abs(angles)):.4f}°")
        print(f"  缩放:  avg={np.mean(scales):.6f}  最大偏差={np.max(np.abs(np.array(scales)-1)):.6f}")
        print(f"  耗时:  avg={np.mean(ms_l):.0f}ms  max={np.max(ms_l):.0f}ms")
    print(f"\n  输出目录: {out_root.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
