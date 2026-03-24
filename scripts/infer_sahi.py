#!/usr/bin/env python3
"""
SAHI 全图推理流水线
==========================================
对完整镜片图像进行 SAHI 风格推理（滑窗切片 → 模型推理 → 全图合并）：

  1. 滑窗切片（640×640, stride=480, 25% 重叠）
  2. 模型推理（低阈值 conf=0.15，高召回）
  3. 全图级 NMS（IOS 度量，避免长短划痕互相抑制）
  4. Scratch 连接算法（共线碎片 → 完整划痕）
  5. 输出:
     - YOLO 格式标注文件（全图坐标或切片坐标）
     - 可视化图像（叠加检测结果）
     - JSON 统计报告

使用方式:
    # 对 val 集推理
    python scripts/infer_sahi.py --split val

    # 对单张图推理
    python scripts/infer_sahi.py --image output/dataset_v2/images/13r.png

    # 对目录中所有图推理
    python scripts/infer_sahi.py --image-dir output/dataset_v2/images/

    # 调整参数
    python scripts/infer_sahi.py --split val --stride 480 --conf 0.15 --gap 120

    # 跳过可视化（加速）
    python scripts/infer_sahi.py --split val --no-vis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fullimage_utils import (
    tile_boxes_to_fullimage,
    fullimage_to_tile_boxes,
    nms_ios,
    connect_scratches,
    generate_tile_positions,
    TILE_SIZE,
    CLASS_NAMES,
)

IMAGES_DIR    = PROJECT_ROOT / "output" / "dataset_v2" / "images"
TILE_DATASET  = PROJECT_ROOT / "output" / "tile_dataset"
OUTPUT_DIR    = PROJECT_ROOT / "output" / "sahi_results"
WEIGHTS_PATH  = (PROJECT_ROOT / "output" / "training" /
                 "stage2_cleaned" / "weights" / "best.pt")

# 颜色 (BGR)
COLORS = [(255, 220, 0), (0, 230, 255), (80, 0, 255)]
COLORS_CONNECTED = (60, 255, 60)  # 连接后的 scratch

# 默认参数
DEFAULT_STRIDE = 480
DEFAULT_CONF   = 0.15
DEFAULT_GAP    = 120
DEFAULT_ANGLE  = 30
DEFAULT_NMS    = 0.35
BATCH_SIZE     = 16


# ═══════════════════════════════════════════════════════
#  推理核心
# ═══════════════════════════════════════════════════════

def infer_single_image(
    img_path: Path,
    model,
    stride: int = DEFAULT_STRIDE,
    conf: float = DEFAULT_CONF,
    nms_thresh: float = DEFAULT_NMS,
    gap: float = DEFAULT_GAP,
    angle: float = DEFAULT_ANGLE,
    max_box_area: float = 0.30,
) -> tuple[list[tuple], dict]:
    """
    对一张全图进行 SAHI 推理。

    返回:
      detections: [(cls, x1, y1, x2, y2, conf)] 全图像素坐标
      info: dict with stats
    """
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return [], {"error": f"无法读取: {img_path}"}

    H, W = img_gray.shape
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # 1. 生成切片位置
    positions = generate_tile_positions(H, W, TILE_SIZE, stride)

    # 2. 切片推理
    tiles = []
    for y0, x0 in positions:
        tile = img_bgr[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
        if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
            pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        tiles.append(tile)

    # 分 batch 运行
    all_raw_boxes = []
    for bi in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[bi:bi + BATCH_SIZE]
        batch_positions = positions[bi:bi + BATCH_SIZE]

        results = model(batch, conf=conf, imgsz=TILE_SIZE,
                        batch=BATCH_SIZE, verbose=False, half=True)

        for ri, r in enumerate(results):
            y0, x0 = batch_positions[ri]
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                xywh = box.xywhn[0].tolist()
                c    = float(box.conf[0])
                # 面积过滤
                if xywh[2] * xywh[3] > max_box_area:
                    continue
                yolo_box = (cls, xywh[0], xywh[1], xywh[2], xywh[3], c)
                full = tile_boxes_to_fullimage([yolo_box], x0, y0)
                all_raw_boxes.extend(full)

    n_raw = len(all_raw_boxes)

    # 3. 全图 NMS
    after_nms = nms_ios(all_raw_boxes, ios_thresh=nms_thresh)
    n_nms = len(after_nms)

    # 4. Scratch 连接
    connected, n_chains = connect_scratches(after_nms, gap, angle)
    n_final = len(connected)

    info = {
        "image": img_path.stem,
        "image_w": W, "image_h": H,
        "n_tiles": len(positions),
        "n_raw_detections": n_raw,
        "n_after_nms": n_nms,
        "n_scratch_chains": n_chains,
        "n_final": n_final,
        "by_class": {},
    }
    for cls_id in range(3):
        info["by_class"][CLASS_NAMES[cls_id]] = sum(
            1 for b in connected if b[0] == cls_id
        )

    return connected, info


# ═══════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════

def visualize(
    img_path: Path,
    detections: list[tuple],
    output_path: Path,
    scale: float = 0.5,
):
    """在全图上绘制检测结果并保存。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    for det in detections:
        cls, x1, y1, x2, y2, conf = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = COLORS[cls] if cls < len(COLORS) else (200, 200, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)),
                      (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_detections_yolo(
    detections: list[tuple],
    img_w: int, img_h: int,
    output_path: Path,
):
    """保存检测结果为 YOLO 归一化格式（全图级）。"""
    lines = []
    for cls, x1, y1, x2, y2, conf in detections:
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

def collect_images(args) -> list[Path]:
    """根据参数收集待推理图像列表。"""
    images = []
    if args.image:
        p = Path(args.image)
        if p.exists():
            images.append(p)
        else:
            print(f"  ✗ 图像不存在: {p}")
    elif args.image_dir:
        d = Path(args.image_dir)
        images = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    elif args.split:
        # 从 tile_index.csv 提取 source images
        idx_path = TILE_DATASET / "tile_index.csv"
        if idx_path.exists():
            stems = set()
            with open(idx_path) as f:
                header = f.readline().strip().split(",")
                for line in f:
                    vals = line.strip().split(",")
                    row = dict(zip(header, vals))
                    if row["split"] == args.split:
                        stems.add(Path(row["source_image"]).stem)
            for stem in sorted(stems):
                for ext in (".png", ".jpg"):
                    p = IMAGES_DIR / f"{stem}{ext}"
                    if p.exists():
                        images.append(p)
                        break
    return images


def main():
    parser = argparse.ArgumentParser(
        description="SAHI 全图推理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--split", choices=["train", "val"],
                     help="对指定 split 的所有原图推理")
    grp.add_argument("--image", type=str,
                     help="单张图像路径")
    grp.add_argument("--image-dir", type=str,
                     help="图像目录")

    parser.add_argument("--weights", type=str, default=None,
                        help=f"模型权重（默认 {WEIGHTS_PATH.name}）")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"切片步长（默认 {DEFAULT_STRIDE}，越小重叠越多）")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"置信度阈值（默认 {DEFAULT_CONF}）")
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP,
                        help=f"scratch 连接最大间距（默认 {DEFAULT_GAP}px）")
    parser.add_argument("--angle", type=float, default=DEFAULT_ANGLE,
                        help=f"scratch 连接最大角度差（默认 {DEFAULT_ANGLE}°）")
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS,
                        help=f"NMS IOS 阈值（默认 {DEFAULT_NMS}）")
    parser.add_argument("--output", type=str, default=None,
                        help=f"输出目录（默认 {OUTPUT_DIR}）")
    parser.add_argument("--no-vis", action="store_true",
                        help="跳过可视化输出（加速）")
    parser.add_argument("--vis-scale", type=float, default=0.5,
                        help="可视化缩放比例（默认 0.5）")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else WEIGHTS_PATH
    if not weights.exists():
        print(f"  ✗ 模型权重不存在: {weights}")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else OUTPUT_DIR
    images  = collect_images(args)

    if not images:
        print("  ✗ 未找到待推理图像")
        parser.print_help()
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   SAHI 全图推理流水线                                ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  图像数:     {len(images)}")
    print(f"  模型:       {weights.name}")
    print(f"  stride:     {args.stride}  (重叠率 {100*(1-args.stride/640):.0f}%)")
    print(f"  conf:       {args.conf}")
    print(f"  NMS IOS:    {args.nms}")
    print(f"  连接参数:   gap={args.gap}px, angle={args.angle}°")
    print(f"  输出目录:   {out_dir}")
    print(f"  可视化:     {'否' if args.no_vis else '是'}")
    print()

    # 加载模型
    print("  加载模型 ...", end="", flush=True)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(" OK")

    # 创建输出目录
    labels_dir = out_dir / "labels"
    vis_dir    = out_dir / "visualizations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    totals = defaultdict(int)
    t_start = time.time()

    for idx, img_path in enumerate(images):
        t0 = time.time()

        detections, info = infer_single_image(
            img_path, model,
            stride=args.stride, conf=args.conf,
            nms_thresh=args.nms, gap=args.gap, angle=args.angle,
        )

        if "error" in info:
            print(f"  [{idx+1}/{len(images)}] {img_path.stem}: {info['error']}")
            continue

        # 保存 YOLO 标注
        save_detections_yolo(
            detections, info["image_w"], info["image_h"],
            labels_dir / f"{img_path.stem}.txt",
        )

        # 可视化
        if not args.no_vis:
            visualize(img_path, detections,
                      vis_dir / f"{img_path.stem}.jpg",
                      scale=args.vis_scale)

        all_reports.append(info)
        for k in ["n_raw_detections", "n_after_nms", "n_scratch_chains", "n_final"]:
            totals[k] += info[k]
        for cls_name, cnt in info["by_class"].items():
            totals[f"cls_{cls_name}"] += cnt

        elapsed = time.time() - t0
        if (idx + 1) % 10 == 0 or idx == len(images) - 1:
            total_elapsed = time.time() - t_start
            eta = total_elapsed / (idx + 1) * (len(images) - idx - 1)
            print(f"  [{idx+1}/{len(images)}] {img_path.stem}: "
                  f"{info['n_final']} 个检测 "
                  f"(raw={info['n_raw_detections']} "
                  f"→NMS={info['n_after_nms']} "
                  f"→连接={info['n_final']}, "
                  f"链={info['n_scratch_chains']}) "
                  f"[{elapsed:.1f}s, ETA {int(eta)}s]")

    # ── 汇总报告 ──
    total_elapsed = time.time() - t_start
    report = {
        "pipeline": "sahi_fullimage",
        "n_images": len(images),
        "weights": str(weights),
        "params": {
            "stride": args.stride, "conf": args.conf,
            "nms_ios": args.nms, "gap": args.gap, "angle": args.angle,
        },
        "totals": dict(totals),
        "per_image": all_reports,
        "elapsed_sec": round(total_elapsed, 1),
    }
    report_path = out_dir / "sahi_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n{'='*56}")
    print(f"  SAHI 推理完成 ({int(total_elapsed)}s)")
    print(f"{'='*56}")
    print(f"  总检测数 (合并前):    {totals['n_raw_detections']:>8}")
    print(f"  NMS 后:               {totals['n_after_nms']:>8}")
    print(f"  Scratch 连接链:       {totals['n_scratch_chains']:>8}")
    print(f"  最终检测数:           {totals['n_final']:>8}")
    print(f"  ─────────────────────────────")
    print(f"  scratch:              {totals.get('cls_scratch',0):>8}")
    print(f"  spot:                 {totals.get('cls_spot',0):>8}")
    print(f"  critical:             {totals.get('cls_critical',0):>8}")
    print(f"\n  标注文件:   {labels_dir}/")
    if not args.no_vis:
        print(f"  可视化:     {vis_dir}/")
    print(f"  报告:       {report_path}")
    print()


if __name__ == "__main__":
    main()
