"""分割研发 P1：传统高召回前景 baseline 横评（research_loops/segmentation_program.md）。

在整镜背景校正图（output/dataset_v2/images）的 ROI 内运行 9 种无学习前景提取方法，
用切片标注拼回的整镜 GT 框做“框命中”评估：只评价召回、噪声、断裂和高光环误检，
不承担类别解释。同时输出逐像素投票图（多少种方法判为前景），供 P2 构造高置信种子 mask。

指标（每图 × 每方法）:
    fg_ratio          前景像素 / ROI 像素
    box_recall[_cls]  GT 框内前景像素 ≥ max(5, 0.5% 框面积) 视为命中
    in_box_precision  前景落在 GT 框（外扩 5px）内的比例；框较粗，只作相对比较
    ring_fp           高光环邻近带（ROI 内、距环 ≤ 60px）上 GT 框外的前景像素 / 带面积
                      （高光环本身已被 ROI 排除，残余眩光误检集中在环内侧）
    scratch_frag      已命中 scratch 框内的连通域数均值（越大越碎）

用法:
    uv run python scripts/seg_p1_classical_baselines.py --per-tier 10 10 6 --workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi, sato, threshold_niblack, threshold_otsu, threshold_sauvola

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from darkfield_defects.app_services.inference_service import (  # noqa: E402
    NMS_IOS,
    nms_ios,
    tile_boxes_to_fullimage,
)
from darkfield_defects.eval.image_level import read_yolo_txt  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "output/dataset_v2/images"
CALIB_DIR = PROJECT_ROOT / "output/dataset_v2/calibration"
TILE_DIR = PROJECT_ROOT / "output/tile_dataset"
STRATA_DIR = PROJECT_ROOT / "output/experiments/phase3_segmentation/data_stratification"
DEFAULT_OUT = PROJECT_ROOT / "output/experiments/seg_p1_classical"

CLASS_NAMES = {0: "scratch", 1: "spot", 2: "critical"}
METHODS = ["otsu", "adaptive_mean", "sauvola", "niblack", "frangi", "sato", "gabor",
           "tophat", "frangi_adaptive"]
MIN_AREA = 8          # 去除小于该像素数的连通域
LOCAL_FLOOR = 10      # 局部阈值法的绝对亮度下限，抑制近零背景上的噪声
WINDOW = 51           # 局部阈值窗口


# ── 前景提取方法：输入 uint8 校正图与 ROI，输出 bool mask ─────────────────────

def _otsu_on(resp: np.ndarray, roi: np.ndarray) -> np.ndarray:
    vals = resp[roi & (resp > 0)]
    if vals.size < 100:
        return np.zeros_like(roi)
    return resp > threshold_otsu(vals)


def _clean(mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
    mask = mask & roi
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    return keep[lab]


def run_method(name: str, img: np.ndarray, roi: np.ndarray, cache: dict) -> np.ndarray:
    f = img.astype(np.float32)
    if name == "otsu":
        m = _otsu_on(f, roi)
    elif name == "adaptive_mean":
        mean = cv2.blur(f, (WINDOW, WINDOW))
        m = (f > mean + 5) & (f > LOCAL_FLOOR)
    elif name == "sauvola":
        m = (f > threshold_sauvola(img, window_size=WINDOW, k=0.2)) & (f > LOCAL_FLOOR)
    elif name == "niblack":
        m = (f > threshold_niblack(img, window_size=WINDOW, k=-0.8)) & (f > LOCAL_FLOOR)
    elif name in ("frangi", "sato"):
        if name not in cache:
            fn = frangi if name == "frangi" else sato
            kw = {"sigmas": (1, 2, 3), "black_ridges": False}
            cache[name] = fn(f / 255.0, **kw).astype(np.float32)
        m = _otsu_on(cache[name], roi)
    elif name == "gabor":
        resp = np.zeros_like(f)
        for k in range(8):
            kern = cv2.getGaborKernel((21, 21), 3.0, np.pi * k / 8, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            kern -= kern.mean()
            np.maximum(resp, cv2.filter2D(f, cv2.CV_32F, kern), out=resp)
        m = _otsu_on(resp, roi)
    elif name == "tophat":
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        m = _otsu_on(th.astype(np.float32), roi)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    elif name == "frangi_adaptive":
        m = run_method("frangi", img, roi, cache) | run_method("adaptive_mean", img, roi, cache)
    else:
        raise ValueError(name)
    return _clean(m, roi)


# ── GT 与评估 ────────────────────────────────────────────────────────────

def load_gt_boxes(lens: str, tile_index: dict[str, list[tuple[str, int, int]]]) -> list[tuple]:
    boxes: list[tuple] = []
    for tile_id, split, y0, x0 in tile_index.get(lens, []):
        label_path = TILE_DIR / "labels" / split / f"{tile_id}.txt"
        boxes += tile_boxes_to_fullimage(read_yolo_txt(label_path),
                                         x0, y0)
    return nms_ios(boxes, ios_thresh=NMS_IOS)


def evaluate(mask: np.ndarray, boxes: list[tuple], roi: np.ndarray, ring: np.ndarray) -> dict:
    h, w = mask.shape
    in_box = np.zeros_like(mask)
    hits: dict[int, list[int]] = defaultdict(list)
    frag: list[int] = []
    for cls, x1, y1, x2, y2, _ in boxes:
        xa, ya, xb, yb = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        if xb <= xa or yb <= ya:
            continue
        in_box[max(0, ya - 5):yb + 5, max(0, xa - 5):xb + 5] = True
        sub = mask[ya:yb, xa:xb]
        hit = int(sub.sum() >= max(5, 0.005 * sub.size))
        hits[cls].append(hit)
        if cls == 0 and hit:
            frag.append(cv2.connectedComponents(sub.astype(np.uint8), connectivity=8)[0] - 1)
    fg = int(mask.sum())
    all_hits = [v for vs in hits.values() for v in vs]
    row = {
        "fg_ratio": fg / max(int(roi.sum()), 1),
        "box_recall": float(np.mean(all_hits)) if all_hits else float("nan"),
        "in_box_precision": float((mask & in_box).sum() / fg) if fg else float("nan"),
        "ring_fp": float((mask & ring & ~in_box).sum() / max(int(ring.sum()), 1)),
        "scratch_frag": float(np.mean(frag)) if frag else float("nan"),
        "n_gt": len(all_hits),
    }
    for cls, name in CLASS_NAMES.items():
        row[f"box_recall_{name}"] = float(np.mean(hits[cls])) if hits[cls] else float("nan")
    return row


def _overlay_strip(img: np.ndarray, masks: dict[str, np.ndarray], boxes: list[tuple],
                   scale: float = 0.25) -> np.ndarray:
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    base = cv2.cvtColor(np.clip(small.astype(np.float32) * 3, 0, 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)
    gt = base.copy()
    for cls, x1, y1, x2, y2, _ in boxes:
        color = {0: (60, 200, 30), 1: (0, 170, 255), 2: (255, 80, 200)}[cls]
        cv2.rectangle(gt, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)),
                      color, 1)
    panels = [("image", base), ("GT boxes", gt)]
    for name, m in masks.items():
        p = base.copy()
        ms = cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
        p[ms] = (0, 0, 255)
        panels.append((name, p))
    for name, p in panels:
        cv2.putText(p, name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cols = 4
    while len(panels) % cols:
        panels.append(("", np.zeros_like(base)))
    rows = [np.hstack([p for _, p in panels[i:i + cols]]) for i in range(0, len(panels), cols)]
    return np.vstack(rows)


def process(lens: str, tier: str, out: Path, tile_index: dict, save_masks: bool) -> list[dict]:
    img = cv2.imread(str(IMAGE_DIR / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(IMAGE_DIR / f"{lens}.png")
    roi = np.load(CALIB_DIR / "roi_mask.npy")
    ring = np.load(CALIB_DIR / "ring_mask.npy").astype(np.uint8)
    ring = cv2.dilate(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) > 0
    ring &= roi
    if roi.shape != img.shape:
        raise ValueError(f"{lens}: ROI {roi.shape} 与图像 {img.shape} 尺寸不一致")
    boxes = load_gt_boxes(lens, tile_index)
    cache: dict = {}
    masks, rows = {}, []
    for name in METHODS:
        t0 = time.perf_counter()
        m = run_method(name, img, roi, cache)
        dt = time.perf_counter() - t0
        masks[name] = m
        rows.append({"lens": lens, "tier": tier, "method": name, "time_s": round(dt, 2),
                     **evaluate(m, boxes, roi, ring)})
        if save_masks:
            cv2.imwrite(str(out / "masks" / name / f"{lens}.png"), m.astype(np.uint8) * 255)
    # 投票图只统计相互独立的基础方法（不含组合法）
    votes = sum(masks[n].astype(np.uint8) for n in METHODS if n != "frangi_adaptive")
    cv2.imwrite(str(out / "votes" / f"{lens}.png"), votes)
    cv2.imwrite(str(out / "overlays" / f"{tier}_{lens}.jpg"), _overlay_strip(img, masks, boxes),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rows


def load_tile_index() -> dict[str, list[tuple[str, str, int, int]]]:
    index: dict[str, list] = defaultdict(list)
    with open(TILE_DIR / "tile_index.csv") as f:
        for r in csv.DictReader(f):
            index[Path(r["source_image"]).stem].append(
                (r["tile_id"], r["split"], int(r["y0"]), int(r["x0"])))
    return index


def summarize(rows: list[dict]) -> dict:
    keys = ["box_recall", "box_recall_scratch", "box_recall_spot", "box_recall_critical",
            "in_box_precision", "ring_fp", "scratch_frag", "fg_ratio", "time_s"]
    out: dict = {}
    for group, tiers in (("L1+L2", {"L1", "L2"}), ("L3", {"L3"})):
        out[group] = {}
        for m in METHODS:
            sel = [r for r in rows if r["method"] == m and r["tier"] in tiers]
            if sel:
                out[group][m] = {k: round(float(np.nanmean([r[k] for r in sel])), 4) for k in keys}
                out[group][m]["n_images"] = len(sel)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--per-tier", type=int, nargs=3, default=[10, 10, 6], metavar=("L1", "L2", "L3"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--no-save-masks", action="store_true")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    logging.disable(logging.INFO)

    rng = random.Random(args.seed)
    samples: list[tuple[str, str]] = []
    for tier, n in zip(("L1", "L2", "L3"), args.per_tier):
        f = STRATA_DIR / f"strata_{tier.lower()}.txt"
        if not f.exists():
            raise SystemExit(
                f"缺少分层清单 {f}；先运行 scripts/build_segmentation_complexity_manifest.py")
        names = sorted(f.read_text().split())
        samples += [(s, tier) for s in sorted(rng.sample(names, min(n, len(names))))]

    for sub in ["votes", "overlays"] + ([] if args.no_save_masks else
                                       [f"masks/{m}" for m in METHODS]):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    tile_index = load_tile_index()

    rows: list[dict] = []
    with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        jobs = {ex.submit(process, s, t, args.out, tile_index, not args.no_save_masks): s
                for s, t in samples}
        for i, job in enumerate(futures.as_completed(jobs), 1):
            rows += job.result()
            print(f"[{i}/{len(jobs)}] {jobs[job]}", flush=True)

    rows.sort(key=lambda r: (r["tier"], r["lens"], METHODS.index(r["method"])))
    with open(args.out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "config": {"per_tier": args.per_tier, "seed": args.seed, "min_area": MIN_AREA,
                   "local_floor": LOCAL_FLOOR, "window": WINDOW,
                   "samples": [{"lens": s, "tier": t} for s, t in samples]},
        "summary": summarize(rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["summary"]["L1+L2"], ensure_ascii=False, indent=1))
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
