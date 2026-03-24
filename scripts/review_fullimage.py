#!/usr/bin/env python3
"""
交互式标注审核工具 — 逐框审核视图 v2
========================================
以"逐个可疑框"为核心工作流：

  - 左侧：Frangi 可疑框队列（按 frangi_max 升序，最可疑优先）
  - 中央：当前框所在切片的大图，目标框高亮显示（主操作区）
  - 右侧：全图缩略图（位置参考）+ 本切片所有 GT 框列表

操作：
  - 鼠标点击左侧队列条目 → 跳到该框
  - [保留 →] 或按 K / →   → 标记为保留，跳下一框
  - [✕ 删除] 或按 D       → 从 GT 中删除，跳下一框
  - [保存]   或按 S        → 将本切片修改写入磁盘

使用方式:
    python scripts/review_fullimage.py --split train   # 审核 train（Frangi 队列）
    python scripts/review_fullimage.py --split val
    python scripts/review_fullimage.py --split both    # 两个 split 一起加载
    python scripts/review_fullimage.py --port 7860
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
TILE_DATASET   = PROJECT_ROOT / "output" / "tile_dataset"
IMAGES_DIR     = PROJECT_ROOT / "output" / "dataset_v2" / "images"
AUDIT_DIR      = PROJECT_ROOT / "output" / "audit"
PRED_LABEL_DIR = AUDIT_DIR / "predictions" / "labels"
TILE_SIZE      = 640

CLASS_NAMES      = ["scratch", "spot", "critical"]
# BGR (OpenCV)
CLASS_COLORS     = [(255, 220, 0), (0, 230, 255), (80, 0, 255)]
# RGB (PIL / JS hex)
CLASS_COLORS_PIL = [(0, 220, 255), (255, 230, 0), (255, 0, 80)]

DISPLAY_SCALE      = 0.40    # 全图缩放（full preview）
THUMB_SCALE        = 0.065   # 右侧缩略图
TILE_HIGHLIGHT_ALPHA = 0.15

# 高亮目标框配色 (BGR)
HL_BORDER_COLOR  = (60, 255, 60)    # 亮绿色边框
HL_FILL_COLOR    = (60, 255, 60)    # 填充底色（透明叠加）
HL_FILL_ALPHA    = 0.12

# 审核队列最大数量（避免浏览器卡顿）
QUEUE_LIMIT = 5000


# ─── 数据加载 ──────────────────────────────────────────────

class ReviewData:
    def __init__(self, splits: list[str]):
        self.splits     = splits
        self.split      = splits[0] if len(splits) == 1 else "both"
        self.tile_index: dict[str, dict]  = {}
        self.gt_labels:  dict[str, list]  = {}
        self.pred_labels: dict[str, list] = {}
        self.suspicion:  dict[str, float] = {}
        self.source_images: list[dict]    = []
        self._load()

    def _load(self):
        print("  加载切片索引 ...", end="", flush=True)
        self._load_tile_index()
        print(f" {len(self.tile_index)} tiles")

        print("  加载 GT 标注 ...", end="", flush=True)
        self._load_gt_labels()
        print(f" {sum(len(v) for v in self.gt_labels.values())} 个框")

        print("  加载模型预测 ...", end="", flush=True)
        self._load_pred_labels()
        print(f" {sum(len(v) for v in self.pred_labels.values())} 个框")

        print("  加载可疑评分 ...", end="", flush=True)
        self._load_suspicion()
        print(f" {len(self.suspicion)} tiles")

        print("  整理图像列表 ...", end="", flush=True)
        self._build_image_list()
        print(f" {len(self.source_images)} 张原图")

    def _load_tile_index(self):
        idx_path = TILE_DATASET / "tile_index.csv"
        if not idx_path.exists():
            return
        with open(idx_path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                vals = line.strip().split(",")
                row = dict(zip(header, vals))
                if row.get("split") not in self.splits:
                    continue
                tid = row["tile_id"]
                self.tile_index[tid] = {
                    "source": Path(row["source_image"]).stem,
                    "split":  row["split"],
                    "y0":     int(row["y0"]),
                    "x0":     int(row["x0"]),
                    "n_defects": int(row.get("n_defects", 0)),
                }

    def _load_gt_labels(self):
        for split in self.splits:
            label_dir = TILE_DATASET / "labels" / split
            if not label_dir.exists():
                continue
            for txt in label_dir.glob("*.txt"):
                tid = txt.stem
                boxes = []
                for line in txt.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append((int(parts[0]),
                                      float(parts[1]), float(parts[2]),
                                      float(parts[3]), float(parts[4])))
                self.gt_labels[tid] = boxes

    def _load_pred_labels(self):
        if not PRED_LABEL_DIR.exists():
            return
        for txt in PRED_LABEL_DIR.glob("*.txt"):
            tid = txt.stem
            boxes = []
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    boxes.append((int(parts[0]),
                                  float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]), conf))
            self.pred_labels[tid] = boxes

    def _load_suspicion(self):
        csv_path = AUDIT_DIR / "suspect_tiles.csv"
        if not csv_path.exists():
            return
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                vals = line.strip().split(",")
                row = dict(zip(header, vals))
                tid = Path(row.get("tile", "")).stem
                try:
                    self.suspicion[tid] = float(row.get("suspicion", 0))
                except ValueError:
                    pass

    def _build_image_list(self):
        img_data: dict[str, dict] = {}
        for tid, info in self.tile_index.items():
            stem = info["source"]
            if stem not in img_data:
                img_data[stem] = {"stem": stem, "split": info["split"],
                                  "tiles": [], "max_susp": 0.0,
                                  "total_gt": 0, "total_pred": 0}
            susp = self.suspicion.get(tid, 0)
            img_data[stem]["tiles"].append(tid)
            img_data[stem]["max_susp"] = max(img_data[stem]["max_susp"], susp)
            img_data[stem]["total_gt"]   += len(self.gt_labels.get(tid, []))
            img_data[stem]["total_pred"] += len(self.pred_labels.get(tid, []))
        self.source_images = sorted(
            img_data.values(), key=lambda d: d["max_susp"], reverse=True
        )

    def get_tiles_for_stem(self, stem: str) -> list[tuple[str, dict]]:
        return [(tid, info) for tid, info in self.tile_index.items()
                if info["source"] == stem]


# ─── 图像渲染 ──────────────────────────────────────────────

def _draw_dashed_rect(img, x1, y1, x2, y2, color, dash=12):
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
           (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for sx, sy, ex, ey in pts:
        length = max(abs(ex - sx), abs(ey - sy))
        if length == 0:
            continue
        steps = max(1, length // (dash * 2))
        for i in range(steps):
            t0 = i * 2 * dash / length
            t1 = min((i * 2 + 1) * dash / length, 1.0)
            p0 = (int(sx + t0 * (ex - sx)), int(sy + t0 * (ey - sy)))
            p1 = (int(sx + t1 * (ex - sx)), int(sy + t1 * (ey - sy)))
            cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)


def render_tile_highlighted(
    stem: str,
    tile_id: str,
    data: ReviewData,
    hl_cx: float, hl_cy: float, hl_w: float, hl_h: float,
    show_pred: bool = True,
    tile_scale: int = 2,
) -> str:
    """
    渲染切片，将指定的 GT 框高亮显示（亮绿厚边框 + 轻微底色），
    其余框以较暗颜色显示。返回 base64 JPEG。
    tile_scale=2 表示 640→1280 放大。
    """
    img_path = _find_image(stem)
    info = data.tile_index.get(tile_id)
    if info is None or img_path is None:
        placeholder = np.zeros((TILE_SIZE * tile_scale, TILE_SIZE * tile_scale, 3), dtype=np.uint8) + 30
        _, buf = cv2.imencode(".jpg", placeholder)
        return base64.b64encode(buf.tobytes()).decode()

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    x0, y0 = info["x0"], info["y0"]
    tile = img_gray[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
    if tile.size == 0:
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

    # 目标框像素坐标
    hl_px = int(hl_cx * TILE_SIZE)
    hl_py = int(hl_cy * TILE_SIZE)
    hl_pw = max(4, int(hl_w * TILE_SIZE))
    hl_ph = max(4, int(hl_h * TILE_SIZE))
    hl_x1 = max(0, hl_px - hl_pw // 2)
    hl_y1 = max(0, hl_py - hl_ph // 2)
    hl_x2 = min(TILE_SIZE - 1, hl_px + hl_pw // 2)
    hl_y2 = min(TILE_SIZE - 1, hl_py + hl_ph // 2)

    # 半透明底色
    overlay = tile_rgb.copy()
    cv2.rectangle(overlay, (hl_x1, hl_y1), (hl_x2, hl_y2), HL_FILL_COLOR, -1)
    cv2.addWeighted(overlay, HL_FILL_ALPHA, tile_rgb, 1 - HL_FILL_ALPHA, 0, tile_rgb)

    # 其他 GT 框（暗显示）
    hl_cls = 0
    for cls, cx, cy, w, h in data.gt_labels.get(tile_id, []):
        is_hl = (abs(cx - hl_cx) < 0.002 and abs(cy - hl_cy) < 0.002)
        if is_hl:
            hl_cls = cls
            continue
        px, py = int(cx * TILE_SIZE), int(cy * TILE_SIZE)
        pw, ph = int(w * TILE_SIZE), int(h * TILE_SIZE)
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
        dim = tuple(max(30, c // 3) for c in color)
        cv2.rectangle(tile_rgb, (px - pw // 2, py - ph // 2),
                      (px + pw // 2, py + ph // 2), dim, 1)

    # 预测框（虚线，暗色）
    if show_pred:
        for item in data.pred_labels.get(tile_id, []):
            cls = item[0]
            px = int(item[1] * TILE_SIZE)
            py = int(item[2] * TILE_SIZE)
            pw = int(item[3] * TILE_SIZE)
            ph = int(item[4] * TILE_SIZE)
            base_color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
            light = tuple(min(255, int(c * 0.35) + 80) for c in base_color)
            _draw_dashed_rect(tile_rgb, px - pw // 2, py - ph // 2,
                              px + pw // 2, py + ph // 2, light, dash=8)

    # 目标框：高亮（最后绘制，在最上层）
    cv2.rectangle(tile_rgb, (hl_x1, hl_y1), (hl_x2, hl_y2), HL_BORDER_COLOR, 3)
    cv2.rectangle(tile_rgb, (max(0,hl_x1-1), max(0,hl_y1-1)),
                  (min(TILE_SIZE-1,hl_x2+1), min(TILE_SIZE-1,hl_y2+1)),
                  (0, 180, 0), 1)

    label_str = CLASS_NAMES[hl_cls] if hl_cls < len(CLASS_NAMES) else "?"
    label_y = max(14, hl_y1 - 4)
    cv2.putText(tile_rgb, f"[{label_str}]",
                (max(0, hl_x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, HL_BORDER_COLOR, 1, cv2.LINE_AA)

    # 放大
    out_w = TILE_SIZE * tile_scale
    out_h = TILE_SIZE * tile_scale
    tile_big = cv2.resize(tile_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".jpg", tile_big, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()


def render_box_crop(
    stem: str,
    tile_id: str,
    data: ReviewData,
    hl_cx: float, hl_cy: float, hl_w: float, hl_h: float,
    context: float = 2.5,
    out_size: int = 320,
) -> str:
    """
    在切片内裁剪目标框区域（含 context 倍填充），放大到 out_size，
    用于超小框的局部放大查看。返回 base64 JPEG。
    """
    img_path = _find_image(stem)
    info = data.tile_index.get(tile_id)
    if info is None or img_path is None:
        ph = np.zeros((out_size, out_size, 3), dtype=np.uint8) + 30
        _, buf = cv2.imencode(".jpg", ph)
        return base64.b64encode(buf.tobytes()).decode()

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    x0, y0 = info["x0"], info["y0"]
    tile = img_gray[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

    # 中心像素 + context padding
    px = int(hl_cx * TILE_SIZE)
    py = int(hl_cy * TILE_SIZE)
    pw = max(8, int(hl_w * TILE_SIZE))
    ph_box = max(8, int(hl_h * TILE_SIZE))
    pad_w = int(pw * context)
    pad_h = int(ph_box * context)
    crop_x1 = max(0, px - pad_w)
    crop_y1 = max(0, py - pad_h)
    crop_x2 = min(TILE_SIZE, px + pad_w)
    crop_y2 = min(TILE_SIZE, py + pad_h)

    crop = tile_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        crop = tile_rgb.copy()

    # 在裁剪图上绘制目标框
    rel_x = px - crop_x1
    rel_y = py - crop_y1
    bx1 = max(0, rel_x - pw // 2)
    by1 = max(0, rel_y - ph_box // 2)
    bx2 = min(crop.shape[1] - 1, rel_x + pw // 2)
    by2 = min(crop.shape[0] - 1, rel_y + ph_box // 2)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), HL_BORDER_COLOR, 2)

    crop_resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".jpg", crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def render_fullimage(
    stem: str,
    data: ReviewData,
    show_gt: bool = True,
    show_pred: bool = True,
    highlight_tile: str | None = None,
    highlight_box: tuple | None = None,
    scale: float = DISPLAY_SCALE,
) -> tuple[str, dict]:
    """渲染全图（含标注叠加），返回 (base64_jpeg, boxes_json)。"""
    img_path = _find_image(stem)
    if img_path is None:
        img_gray = np.zeros((3000, 4096), dtype=np.uint8) + 40
    else:
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_gray = np.zeros((3000, 4096), dtype=np.uint8) + 40

    H, W = img_gray.shape
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    tiles_for_stem = data.get_tiles_for_stem(stem)

    # 高亮选中切片
    if highlight_tile:
        info_hl = data.tile_index.get(highlight_tile)
        if info_hl and info_hl["source"] == stem:
            tx0, ty0 = info_hl["x0"], info_hl["y0"]
            tx1 = min(tx0 + TILE_SIZE, W)
            ty1 = min(ty0 + TILE_SIZE, H)
            overlay = img.copy()
            cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (140, 80, 255), -1)
            cv2.addWeighted(overlay, TILE_HIGHLIGHT_ALPHA, img, 1 - TILE_HIGHLIGHT_ALPHA, 0, img)
            cv2.rectangle(img, (tx0, ty0), (tx1, ty1), (200, 140, 255), 2)

    all_gt_boxes, all_pred_boxes = [], []

    if show_gt:
        for tid, info in tiles_for_stem:
            x0_t, y0_t = info["x0"], info["y0"]
            for cls, cx, cy, w, h in data.gt_labels.get(tid, []):
                px = int(x0_t + cx * TILE_SIZE)
                py = int(y0_t + cy * TILE_SIZE)
                pw = int(w * TILE_SIZE)
                ph = int(h * TILE_SIZE)
                x1b, y1b = px - pw // 2, py - ph // 2
                x2b, y2b = px + pw // 2, py + ph // 2
                color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
                # 高亮目标框
                if (highlight_box and tid == highlight_tile and
                        highlight_box[0] is not None and
                        abs(cx - highlight_box[0]) < 0.002 and
                        abs(cy - highlight_box[1]) < 0.002):
                    cv2.rectangle(img, (x1b, y1b), (x2b, y2b), HL_BORDER_COLOR, 3)
                else:
                    cv2.rectangle(img, (x1b, y1b), (x2b, y2b), color, 1)
                all_gt_boxes.append({"type": "gt", "cls": cls,
                                     "cx": px, "cy": py, "w": pw, "h": ph, "tile": tid,
                                     "x1": x1b, "y1": y1b, "x2": x2b, "y2": y2b})

    if show_pred:
        for tid, info in tiles_for_stem:
            x0_t, y0_t = info["x0"], info["y0"]
            for item in data.pred_labels.get(tid, []):
                cls = item[0]
                px = int(x0_t + item[1] * TILE_SIZE)
                py = int(y0_t + item[2] * TILE_SIZE)
                pw = int(item[3] * TILE_SIZE)
                ph = int(item[4] * TILE_SIZE)
                conf = item[5] if len(item) > 5 else 1.0
                base_color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
                light = tuple(min(255, int(c * 0.5) + 120) for c in base_color)
                dash_len = max(8, min(pw, ph) // 3)
                x1b, y1b = px - pw // 2, py - ph // 2
                x2b, y2b = px + pw // 2, py + ph // 2
                _draw_dashed_rect(img, x1b, y1b, x2b, y2b, light, dash_len)
                all_pred_boxes.append({"type": "pred", "cls": cls, "conf": round(conf, 3),
                                       "cx": px, "cy": py, "w": pw, "h": ph, "tile": tid,
                                       "x1": x1b, "y1": y1b, "x2": x2b, "y2": y2b})

    new_w, new_h = int(W * scale), int(H * scale)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def sc(b):
        sb = dict(b)
        for k in ["cx", "cy", "w", "h", "x1", "y1", "x2", "y2"]:
            sb[k] = round(b[k] * scale, 1)
        return sb

    _, buf = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return b64, {"gt": [sc(b) for b in all_gt_boxes],
                 "pred": [sc(b) for b in all_pred_boxes],
                 "image_w": new_w, "image_h": new_h}


def render_thumb(stem: str, data: ReviewData,
                 highlight_tile: str | None = None,
                 scale: float = THUMB_SCALE) -> str:
    """渲染全图缩略图（右侧辅助），可高亮切片位置。返回 base64 JPEG。"""
    img_path = _find_image(stem)
    if img_path is None:
        ph = np.zeros((int(3000 * scale), int(4096 * scale), 3), dtype=np.uint8) + 30
        _, buf = cv2.imencode(".jpg", ph)
        return base64.b64encode(buf.tobytes()).decode()

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        ph = np.zeros((int(3000 * scale), int(4096 * scale), 3), dtype=np.uint8) + 30
        _, buf = cv2.imencode(".jpg", ph)
        return base64.b64encode(buf.tobytes()).decode()

    H, W = img_gray.shape
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if highlight_tile:
        info_hl = data.tile_index.get(highlight_tile)
        if info_hl and info_hl["source"] == stem:
            tx0, ty0 = info_hl["x0"], info_hl["y0"]
            tx1 = min(tx0 + TILE_SIZE, W)
            ty1 = min(ty0 + TILE_SIZE, H)
            overlay = img.copy()
            cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (60, 255, 60), -1)
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
            cv2.rectangle(img, (tx0, ty0), (tx1, ty1), (60, 255, 60), 4)

    nw, nh = int(W * scale), int(H * scale)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


def _find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ─── 标注保存 ──────────────────────────────────────────────

def save_labels(tile_id: str, new_boxes: list[dict], data: ReviewData) -> bool:
    info = data.tile_index.get(tile_id)
    if info is None:
        return False
    split = info["split"]
    label_path = TILE_DATASET / "labels" / split / f"{tile_id}.txt"
    lines = []
    for box in new_boxes:
        cls = int(box["cls"])
        cx, cy = float(box["cx"]), float(box["cy"])
        w,  h  = float(box["w"]),  float(box["h"])
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    data.gt_labels[tile_id] = [
        (int(b["cls"]), float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"]))
        for b in new_boxes
    ]
    return True


# ─── HTTP 服务器 ───────────────────────────────────────────

DATA: ReviewData | None = None


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/images":
            self._api_images()
        elif path == "/api/review_queue":
            self._api_review_queue()
        elif path.startswith("/api/image/"):
            self._api_image(path[len("/api/image/"):], params)
        elif path.startswith("/api/tile/"):
            parts = path[len("/api/tile/"):].split("/", 1)
            self._api_tile(parts[0], parts[1] if len(parts) > 1 else "", params)
        elif path.startswith("/api/thumb/"):
            self._api_thumb(path[len("/api/thumb/"):], params)
        elif path.startswith("/api/box_crop/"):
            parts = path[len("/api/box_crop/"):].split("/", 1)
            self._api_box_crop(parts[0], parts[1] if len(parts) > 1 else "", params)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == "/api/save":
            tile_id   = body.get("tile_id", "")
            new_boxes = body.get("boxes", [])
            ok = save_labels(tile_id, new_boxes, DATA) if DATA else False
            self._json({"ok": ok})
        else:
            self._json({"error": "not found"}, 404)

    # ── API handlers ───────────────────────────────────────

    def _api_stats(self):
        if DATA is None:
            self._json({})
            return
        n_tiles = len(DATA.tile_index)
        self._json({
            "split":    DATA.split,
            "n_images": len(DATA.source_images),
            "n_tiles":  n_tiles,
            "n_gt":     sum(len(v) for v in DATA.gt_labels.values()),
            "n_pred":   sum(len(v) for v in DATA.pred_labels.values()),
        })

    def _api_images(self):
        if DATA is None:
            self._json([])
            return
        self._json([
            {"stem": d["stem"], "split": d["split"],
             "max_susp": round(d["max_susp"], 1),
             "n_tiles": len(d["tiles"]),
             "total_gt": d["total_gt"], "total_pred": d["total_pred"]}
            for d in DATA.source_images
        ])

    def _api_review_queue(self):
        """返回 Frangi 可疑框扁平队列（每框一条），按 frangi_max 升序。"""
        if DATA is None:
            self._json([])
            return
        queue_path = AUDIT_DIR / "frangi_review_queue.json"
        if not queue_path.exists():
            self._json([])
            return

        entries = json.loads(queue_path.read_text())
        flat: list[dict] = []

        for tile_entry in entries:
            tile_id = tile_entry["tile"]
            if tile_id not in DATA.tile_index:
                continue
            stem = DATA.tile_index[tile_id]["source"]

            for box_entry in tile_entry["boxes"]:
                cx, cy, w, h = box_entry["box"]

                # 找到当前 GT 中对应的类别
                cls = 0
                for gt_cls, gt_cx, gt_cy, _, _ in DATA.gt_labels.get(tile_id, []):
                    if abs(gt_cx - cx) < 0.003 and abs(gt_cy - cy) < 0.003:
                        cls = gt_cls
                        break

                flat.append({
                    "tile_id":   tile_id,
                    "stem":      stem,
                    "cls":       cls,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "frangi_max":  round(box_entry["frangi_max"], 4),
                    "reason":    box_entry["reason"],
                    "model_conf": round(box_entry.get("model_conf", -1.0), 3),
                })

        flat.sort(key=lambda x: x["frangi_max"])
        self._json(flat[:QUEUE_LIMIT])

    def _api_image(self, stem: str, params: dict):
        if DATA is None:
            self._json({"error": "not loaded"})
            return
        show_gt   = params.get("gt",    ["1"])[0] != "0"
        show_pred = params.get("pred",  ["1"])[0] != "0"
        hl_tile   = params.get("hl",    [None])[0]
        hl_cx = float(params.get("hcx", ["-1"])[0])
        hl_cy = float(params.get("hcy", ["-1"])[0])
        hl_box = (hl_cx, hl_cy) if hl_cx >= 0 else None

        try:
            b64, boxes = render_fullimage(
                stem, DATA,
                show_gt=show_gt, show_pred=show_pred,
                highlight_tile=hl_tile,
                highlight_box=hl_box,
                scale=DISPLAY_SCALE,
            )
        except Exception as e:
            self._json({"error": str(e)})
            return

        # 切片列表（本图所有切片，按可疑度排序）
        tiles = []
        for tid, info in DATA.tile_index.items():
            if info["source"] != stem:
                continue
            tiles.append({
                "tile_id": tid, "x0": info["x0"], "y0": info["y0"],
                "susp": DATA.suspicion.get(tid, 0),
                "n_gt":   len(DATA.gt_labels.get(tid, [])),
                "n_pred": len(DATA.pred_labels.get(tid, [])),
            })
        tiles.sort(key=lambda t: t["susp"], reverse=True)

        self._json({"stem": stem, "image": b64, "boxes": boxes, "tiles": tiles})

    def _api_tile(self, stem: str, tile_id: str, params: dict):
        """返回切片放大图（可高亮指定框）+ GT/Pred 列表。"""
        if DATA is None:
            self._json({"error": "not loaded"})
            return

        hl_cx = float(params.get("hcx", ["-1"])[0])
        hl_cy = float(params.get("hcy", ["-1"])[0])
        hl_w  = float(params.get("hw",  ["0"])[0])
        hl_h  = float(params.get("hh",  ["0"])[0])
        show_pred = params.get("pred", ["1"])[0] != "0"

        if hl_cx >= 0:
            b64 = render_tile_highlighted(
                stem, tile_id, DATA,
                hl_cx, hl_cy, hl_w, hl_h,
                show_pred=show_pred, tile_scale=2,
            )
        else:
            # 无高亮，返回普通切片视图
            b64 = _render_tile_plain(stem, tile_id, DATA, show_pred=show_pred)

        gt  = [{"cls": c, "cx": cx, "cy": cy, "w": w, "h": h}
               for c, cx, cy, w, h in DATA.gt_labels.get(tile_id, [])]
        pr  = [{"cls": item[0], "cx": item[1], "cy": item[2],
                "w": item[3], "h": item[4],
                "conf": item[5] if len(item) > 5 else 1.0}
               for item in DATA.pred_labels.get(tile_id, [])]

        self._json({"tile_id": tile_id, "image": b64, "gt": gt, "pred": pr,
                    "susp": DATA.suspicion.get(tile_id, 0)})

    def _api_thumb(self, stem: str, params: dict):
        if DATA is None:
            self._json({"error": "not loaded"})
            return
        hl_tile = params.get("hl", [None])[0]
        b64 = render_thumb(stem, DATA, highlight_tile=hl_tile, scale=THUMB_SCALE)
        info = DATA.tile_index.get(hl_tile, {}) if hl_tile else {}
        W_orig = 4096
        H_orig = 3000
        tx0 = round(info.get("x0", 0) * THUMB_SCALE, 1) if info else 0
        ty0 = round(info.get("y0", 0) * THUMB_SCALE, 1) if info else 0
        self._json({
            "image": b64,
            "thumb_w": int(W_orig * THUMB_SCALE),
            "thumb_h": int(H_orig * THUMB_SCALE),
            "tile_x0": tx0, "tile_y0": ty0,
            "tile_size": int(TILE_SIZE * THUMB_SCALE),
        })

    def _api_box_crop(self, stem: str, tile_id: str, params: dict):
        if DATA is None:
            self._json({"error": "not loaded"})
            return
        cx = float(params.get("cx", ["0.5"])[0])
        cy = float(params.get("cy", ["0.5"])[0])
        w  = float(params.get("w",  ["0.1"])[0])
        h  = float(params.get("h",  ["0.1"])[0])
        b64 = render_box_crop(stem, tile_id, DATA, cx, cy, w, h,
                              context=2.5, out_size=320)
        self._json({"image": b64})

    # ── Helpers ───────────────────────────────────────────

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html  = _build_html()
        body  = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _render_tile_plain(stem: str, tile_id: str, data: ReviewData, show_pred: bool = True) -> str:
    """不高亮任何框的普通切片视图，用于从切片列表跳转。"""
    img_path = _find_image(stem)
    info = data.tile_index.get(tile_id)
    if info is None or img_path is None:
        placeholder = np.zeros((TILE_SIZE * 2, TILE_SIZE * 2, 3), dtype=np.uint8) + 30
        _, buf = cv2.imencode(".jpg", placeholder)
        return base64.b64encode(buf.tobytes()).decode()

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    x0, y0 = info["x0"], info["y0"]
    tile = img_gray[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

    for cls, cx, cy, w, h in data.gt_labels.get(tile_id, []):
        px, py = int(cx * TILE_SIZE), int(cy * TILE_SIZE)
        pw, ph = int(w * TILE_SIZE), int(h * TILE_SIZE)
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
        cv2.rectangle(tile_rgb, (px - pw // 2, py - ph // 2),
                      (px + pw // 2, py + ph // 2), color, 2)
        cv2.putText(tile_rgb, CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                    (px - pw // 2 + 2, py - ph // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if show_pred:
        for item in data.pred_labels.get(tile_id, []):
            cls = item[0]
            px = int(item[1] * TILE_SIZE)
            py = int(item[2] * TILE_SIZE)
            pw = int(item[3] * TILE_SIZE)
            ph = int(item[4] * TILE_SIZE)
            base_color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (200, 200, 200)
            light = tuple(min(255, int(c * 0.5) + 128) for c in base_color)
            _draw_dashed_rect(tile_rgb, px - pw // 2, py - ph // 2,
                              px + pw // 2, py + ph // 2, light, dash=10)

    tile_big = cv2.resize(tile_rgb, (TILE_SIZE * 2, TILE_SIZE * 2),
                          interpolation=cv2.INTER_LINEAR)
    _, buf = cv2.imencode(".jpg", tile_big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


# ─── HTML 前端 ────────────────────────────────────────────

def _build_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Microlens_DF 标注审核</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', 'PingFang SC', sans-serif;
  background: #111827; color: #d1d5db;
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── Toolbar ── */
#toolbar {
  background: #1f2937; border-bottom: 1px solid #374151;
  padding: 5px 12px; display: flex; align-items: center; gap: 10px;
  flex-shrink: 0; flex-wrap: wrap; min-height: 40px;
}
#toolbar h1 { font-size: 13px; color: #f87171; white-space: nowrap; font-weight: 600; }
.tbtn {
  padding: 3px 9px; border-radius: 4px; border: 1px solid #4b5563;
  background: #374151; color: #9ca3af; cursor: pointer; font-size: 11px;
  transition: all .15s;
}
.tbtn.active { background: #3b82f6; color: #fff; border-color: #3b82f6; }
.tbtn:hover:not(.active) { background: #4b5563; color: #e5e7eb; }
#progress-bar {
  flex: 1; max-width: 220px; height: 8px; background: #374151;
  border-radius: 4px; overflow: hidden; min-width: 80px;
}
#progress-fill { height: 100%; background: #10b981; border-radius: 4px; transition: width .3s; }
#progress-text { font-size: 11px; color: #6b7280; white-space: nowrap; }
#mode-badge {
  font-size: 10px; padding: 2px 7px; border-radius: 10px;
  background: #065f46; color: #6ee7b7; font-weight: 600; white-space: nowrap;
}

/* ── Main 3-column layout ── */
#main { display: flex; flex: 1; overflow: hidden; }

/* ── Left: queue panel ── */
#queue-panel {
  width: 210px; flex-shrink: 0; background: #1f2937;
  border-right: 1px solid #374151; display: flex; flex-direction: column;
}
#queue-panel h2 {
  font-size: 11px; color: #6b7280; padding: 7px 10px 5px;
  border-bottom: 1px solid #374151; flex-shrink: 0;
  display: flex; justify-content: space-between; align-items: center;
}
#queue-filter {
  width: 100%; padding: 4px 8px; background: #111827; border: 1px solid #374151;
  color: #d1d5db; font-size: 10px; border-radius: 3px; flex-shrink: 0; margin: 0;
}
#queue-list { flex: 1; overflow-y: auto; }
.queue-item {
  padding: 5px 8px; cursor: pointer; border-bottom: 1px solid #1f2937;
  font-size: 10px; transition: background .1s; display: flex; flex-direction: column; gap: 2px;
  border-left: 3px solid transparent;
}
.queue-item:hover { background: #273349; }
.queue-item.selected { background: #1e3a5f; border-left-color: #3b82f6; }
.queue-item.reviewed-keep { opacity: 0.45; border-left-color: #10b981; }
.queue-item.reviewed-delete { opacity: 0.35; border-left-color: #ef4444; text-decoration: line-through; }
.qi-row1 { display: flex; justify-content: space-between; align-items: center; }
.qi-tile { color: #93c5fd; font-size: 9px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 130px; }
.qi-frangi { font-size: 9px; font-weight: 600; }
.qi-row2 { color: #6b7280; font-size: 9px; }
.qi-cls-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }

/* ── Center: tile viewer ── */
#center-panel {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  background: #0d1117;
}
#nav-bar {
  background: #1f2937; border-bottom: 1px solid #374151;
  padding: 5px 12px; display: flex; align-items: center; gap: 8px;
  flex-shrink: 0; min-height: 42px; flex-wrap: wrap;
}
#nav-pos { font-size: 12px; color: #9ca3af; white-space: nowrap; min-width: 80px; text-align: center; }
.nav-btn {
  padding: 4px 12px; border-radius: 4px; border: 1px solid #4b5563;
  background: #374151; color: #d1d5db; cursor: pointer; font-size: 12px;
  transition: all .15s;
}
.nav-btn:hover { background: #4b5563; }
#btn-keep {
  padding: 5px 16px; border-radius: 4px; border: 1px solid #059669;
  background: #065f46; color: #6ee7b7; cursor: pointer; font-size: 12px; font-weight: 600;
  transition: all .15s; margin-left: auto;
}
#btn-keep:hover { background: #047857; }
#btn-delete {
  padding: 5px 16px; border-radius: 4px; border: 1px solid #dc2626;
  background: #7f1d1d; color: #fca5a5; cursor: pointer; font-size: 12px; font-weight: 600;
  transition: all .15s;
}
#btn-delete:hover { background: #991b1b; }
#btn-save-nav {
  padding: 5px 12px; border-radius: 4px; border: 1px solid #ca8a04;
  background: #713f12; color: #fde68a; cursor: pointer; font-size: 11px;
  transition: all .15s;
}
#btn-save-nav:hover { background: #92400e; }
#save-badge {
  font-size: 10px; color: #fbbf24; display: none;
}

/* tile viewer area */
#tile-viewer-wrap {
  flex: 1; overflow: auto; display: flex; align-items: center; justify-content: center;
  padding: 8px;
}
#tile-img {
  max-width: 100%; max-height: 100%;
  display: block; border-radius: 4px;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}
#tile-loading { color: #6b7280; font-size: 14px; display: none; }

/* box info strip */
#box-info-bar {
  background: #1f2937; border-top: 1px solid #374151;
  padding: 4px 12px; font-size: 10px; color: #9ca3af;
  display: flex; gap: 16px; align-items: center; flex-shrink: 0; flex-wrap: wrap;
}
#box-info-bar b { color: #e5e7eb; }
.info-chip {
  padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600;
}

/* ── Right: auxiliary ── */
#aux-panel {
  width: 230px; flex-shrink: 0; background: #1f2937;
  border-left: 1px solid #374151; display: flex; flex-direction: column;
  overflow: hidden;
}
.aux-section {
  border-bottom: 1px solid #374151; flex-shrink: 0; overflow: hidden;
}
.aux-section h3 {
  font-size: 10px; color: #6b7280; padding: 5px 8px;
  background: #1a2332; border-bottom: 1px solid #2d3748;
  display: flex; justify-content: space-between; align-items: center;
}
/* full-image thumbnail */
#thumb-wrap { padding: 6px; display: flex; justify-content: center; }
#thumb-img { max-width: 100%; border-radius: 3px; cursor: pointer; }
#thumb-img:hover { opacity: 0.85; }

/* GT list in current tile */
#tile-gt-section { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
#tile-gt-list { flex: 1; overflow-y: auto; padding: 4px; }
.gt-item {
  display: flex; align-items: center; gap: 4px; padding: 3px 5px;
  border-radius: 3px; margin-bottom: 2px; font-size: 10px; cursor: pointer;
  transition: background .1s;
}
.gt-item:hover { background: #273349; }
.gt-item.hl-gt { background: #1e3a2a; outline: 1px solid #10b981; }
.gt-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
.gt-coords { flex: 1; color: #6b7280; font-size: 9px; }
.gt-del-btn {
  color: #ef4444; cursor: pointer; font-size: 12px; background: none;
  border: none; padding: 0 2px; line-height: 1; transition: color .1s;
}
.gt-del-btn:hover { color: #f87171; }

/* box crop zoom */
#crop-section { flex-shrink: 0; }
#crop-img { max-width: 100%; border-radius: 3px; display: none; }

/* Save section */
#save-section { padding: 8px; flex-shrink: 0; }
#save-btn {
  width: 100%; padding: 6px; background: #1e3a5f; color: #93c5fd;
  border: 1px solid #2563eb; border-radius: 4px; cursor: pointer;
  font-size: 11px; font-weight: 600; transition: background .15s;
}
#save-btn:hover { background: #1d4ed8; color: #fff; }
#save-status { font-size: 10px; color: #10b981; margin-top: 4px; text-align: center; min-height: 14px; }

/* Legend */
#legend { padding: 6px 8px; border-top: 1px solid #374151; flex-shrink: 0; }
#legend-title { font-size: 9px; color: #6b7280; margin-bottom: 4px; }
.legend-row { display: flex; align-items: center; gap: 5px; margin-bottom: 2px; font-size: 9px; }
.legend-box { width: 12px; height: 9px; border-radius: 1px; flex-shrink: 0; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }

/* crop zoom section */
#crop-section .crop-label {
  font-size: 9px; color: #6b7280; text-align: center; padding: 2px 0;
}
</style>
</head>
<body>

<!-- ── Toolbar ── -->
<div id="toolbar">
  <h1>Microlens_DF 标注审核</h1>
  <span id="mode-badge">逐框审核</span>
  <button class="tbtn active" id="btn-show-pred" onclick="togglePred()">模型预测</button>
  <button class="tbtn active" id="btn-show-crop" onclick="toggleCrop()">框放大</button>
  <button class="tbtn" id="btn-fullview" onclick="openFullView()">全图视图</button>
  <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
  <span id="progress-text">加载中...</span>
</div>

<!-- ── Main ── -->
<div id="main">

  <!-- Left: review queue -->
  <div id="queue-panel">
    <h2>
      <span id="queue-title">可疑框队列</span>
      <span id="queue-count" style="color:#3b82f6">0</span>
    </h2>
    <input id="queue-filter" type="text" placeholder="筛选 tile / 原因..." oninput="filterQueue(this.value)"/>
    <div id="queue-list"></div>
  </div>

  <!-- Center: tile viewer -->
  <div id="center-panel">
    <div id="nav-bar">
      <button class="nav-btn" onclick="navigate(-1)" title="上一框 (←)">&#9664; 上一框</button>
      <span id="nav-pos">—</span>
      <button class="nav-btn" onclick="navigate(1)" title="下一框 (→)">下一框 &#9654;</button>
      <button id="btn-keep"   onclick="keepBox()"   title="保留此框，跳到下一个 (K)">✓ 保留</button>
      <button id="btn-delete" onclick="deleteBox()" title="删除此框，跳到下一个 (D)">✕ 删除</button>
      <button id="btn-save-nav" onclick="saveCurrentTile()" title="保存本切片修改 (S)">保存</button>
      <span id="save-badge">● 未保存</span>
    </div>

    <div id="tile-viewer-wrap">
      <span id="tile-loading">加载中...</span>
      <img id="tile-img" src="" alt="" style="display:none"/>
    </div>

    <div id="box-info-bar">
      <span>切片: <b id="info-tile">—</b></span>
      <span>类别: <b id="info-cls">—</b></span>
      <span>Frangi: <b id="info-frangi">—</b></span>
      <span>原因: <b id="info-reason">—</b></span>
      <span id="info-modelconf" style="display:none">模型置信度: <b id="info-conf">—</b></span>
    </div>
  </div>

  <!-- Right: auxiliary panel -->
  <div id="aux-panel">

    <!-- Full image thumbnail -->
    <div class="aux-section" id="thumb-section">
      <h3>全图位置 <span style="color:#4b5563;font-weight:400;cursor:pointer" onclick="openFullView()">[全屏]</span></h3>
      <div id="thumb-wrap">
        <img id="thumb-img" src="" alt="" onclick="openFullView()" title="点击查看全图"/>
      </div>
    </div>

    <!-- Box crop zoom -->
    <div class="aux-section" id="crop-section">
      <h3>局部放大</h3>
      <div id="crop-wrap" style="padding:4px;display:flex;justify-content:center;">
        <img id="crop-img" src="" alt="" style="max-width:100%;border-radius:3px;display:none"/>
        <span id="crop-placeholder" style="font-size:9px;color:#4b5563;padding:8px">选中框后显示</span>
      </div>
    </div>

    <!-- Current tile GT boxes -->
    <div class="aux-section" id="tile-gt-section" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
      <h3>本切片 GT 框 <span id="gt-count" style="color:#3b82f6">0</span></h3>
      <div id="tile-gt-list"></div>
    </div>

    <!-- Save -->
    <div id="save-section">
      <button id="save-btn" onclick="saveCurrentTile()">保存本切片修改</button>
      <div id="save-status"></div>
    </div>

    <!-- Legend -->
    <div id="legend">
      <div id="legend-title">图例</div>
      <div class="legend-row"><div class="legend-box" style="background:#00DCFF"></div>GT scratch</div>
      <div class="legend-row"><div class="legend-box" style="background:#FFE600"></div>GT spot</div>
      <div class="legend-row"><div class="legend-box" style="background:#FF0050"></div>GT critical</div>
      <div class="legend-row"><div class="legend-box" style="border:1px dashed #80EEFF"></div>Pred (虚线)</div>
      <div class="legend-row"><div class="legend-box" style="background:#3cff3c;opacity:.8"></div>当前审核框</div>
    </div>

  </div>
</div>

<!-- Full-image modal -->
<div id="fullimg-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,0.85);z-index:1000;overflow:auto;cursor:zoom-out"
     onclick="closeFullView()">
  <img id="fullimg-modal-img" src="" style="max-width:98%;display:block;margin:10px auto;border-radius:4px"/>
</div>

<script>
// ══════════════════════════════════════════════════════
//  State
// ══════════════════════════════════════════════════════
const S = {
  queue:       [],       // full flat queue from /api/review_queue
  filtered:    [],       // currently displayed queue (after filter)
  queueIdx:    -1,       // index into S.filtered
  reviewed:    {},       // { idx_in_queue: 'keep'|'delete' }
  // current tile state
  currentTileId:  null,
  currentStem:    null,
  currentTileGT:  [],    // [{cls,cx,cy,w,h}, ...]
  pendingDeletes: {},    // { tile_id: Set of box-key "cx_cy" }
  unsaved:        false,
  // UI prefs
  showPred:    true,
  showCrop:    true,
  // stats
  keptCount:    0,
  deletedCount: 0,
};

const CLASS_NAMES  = ['scratch', 'spot', 'critical'];
const GT_COLORS    = ['#00DCFF', '#FFE600', '#FF0050'];
const GT_BG_COLORS = ['#00DCFF28', '#FFE60028', '#FF005028'];

// ══════════════════════════════════════════════════════
//  Init
// ══════════════════════════════════════════════════════
async function init() {
  const stats = await fetch('/api/stats').then(r=>r.json()).catch(()=>({}));
  document.getElementById('progress-text').textContent =
    `split:${stats.split||'?'} | GT:${stats.n_gt||0} | Pred:${stats.n_pred||0}`;

  // Load Frangi review queue
  const queue = await fetch('/api/review_queue').then(r=>r.json()).catch(()=>[]);
  S.queue    = queue;
  S.filtered = [...queue];
  document.getElementById('queue-count').textContent = queue.length;
  renderQueueList();

  if (queue.length > 0) {
    jumpToQueue(0);
  }
}

// ══════════════════════════════════════════════════════
//  Queue list rendering
// ══════════════════════════════════════════════════════
function renderQueueList() {
  const el = document.getElementById('queue-list');
  el.innerHTML = '';
  const fmap = [0.05, 0.15, 0.30];  // thresholds for coloring

  S.filtered.forEach((item, i) => {
    const origIdx = S.queue.indexOf(item);
    const status  = S.reviewed[origIdx] || '';

    const div = document.createElement('div');
    div.className = 'queue-item' +
      (status === 'keep'   ? ' reviewed-keep'   : '') +
      (status === 'delete' ? ' reviewed-delete' : '');
    div.dataset.idx = i;

    const frangi = item.frangi_max;
    const fcolor = frangi < 0.05 ? '#f87171' :
                   frangi < 0.15 ? '#fb923c' :
                   frangi < 0.25 ? '#fbbf24' : '#86efac';
    const clsColor = GT_COLORS[item.cls] || '#9ca3af';
    const tileShort = item.tile_id.replace(/.*_y/, 'y').replace('_x', ',x');

    div.innerHTML = `
      <div class="qi-row1">
        <span class="qi-tile" title="${item.tile_id}">${tileShort}</span>
        <span class="qi-frangi" style="color:${fcolor}">${frangi.toFixed(3)}</span>
      </div>
      <div class="qi-row2">
        <span class="qi-cls-dot" style="background:${clsColor}"></span>
        <span>${CLASS_NAMES[item.cls]||'?'}</span>
        <span style="margin-left:4px;color:#4b5563">${item.reason.replace('_',' ')}</span>
        ${status ? '<span style="margin-left:auto;color:#4b5563">'+
          (status==='keep'?'✓':'✕')+'</span>' : ''}
      </div>
    `;
    div.onclick = () => jumpToQueue(i);
    el.appendChild(div);
  });
}

function filterQueue(text) {
  const t = text.trim().toLowerCase();
  S.filtered = t ? S.queue.filter(item =>
    item.tile_id.toLowerCase().includes(t) ||
    item.stem.toLowerCase().includes(t) ||
    item.reason.toLowerCase().includes(t) ||
    CLASS_NAMES[item.cls].includes(t)
  ) : [...S.queue];
  renderQueueList();
}

// ══════════════════════════════════════════════════════
//  Navigation
// ══════════════════════════════════════════════════════
function navigate(delta) {
  const n = S.filtered.length;
  if (n === 0) return;
  const next = Math.max(0, Math.min(n - 1, S.queueIdx + delta));
  if (next !== S.queueIdx) jumpToQueue(next);
}

async function jumpToQueue(filteredIdx) {
  if (filteredIdx < 0 || filteredIdx >= S.filtered.length) return;

  const item = S.filtered[filteredIdx];
  S.queueIdx = filteredIdx;

  // Highlight in list
  document.querySelectorAll('.queue-item').forEach(el => {
    el.classList.toggle('selected', parseInt(el.dataset.idx) === filteredIdx);
  });
  const selEl = document.querySelector(`.queue-item[data-idx="${filteredIdx}"]`);
  if (selEl) selEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

  // Update nav position
  const total = S.filtered.length;
  const doneCount = Object.keys(S.reviewed).length;
  document.getElementById('nav-pos').textContent = `${filteredIdx + 1} / ${total}`;
  updateProgress();

  // If switching tiles, prompt save
  if (S.unsaved && S.currentTileId && S.currentTileId !== item.tile_id) {
    await saveCurrentTile(true);  // silent save
  }

  // Load tile if changed
  if (S.currentTileId !== item.tile_id || S.currentStem !== item.stem) {
    await loadTile(item.stem, item.tile_id, item.cx, item.cy, item.w, item.h);
  } else {
    // Same tile, just re-render with new highlight
    await loadTileImage(item.stem, item.tile_id, item.cx, item.cy, item.w, item.h);
  }

  // Update box info bar
  updateBoxInfo(item);

  // Load crop zoom
  if (S.showCrop) {
    loadCropZoom(item.stem, item.tile_id, item.cx, item.cy, item.w, item.h);
  }

  // Load thumb (only if stem changed)
  loadThumb(item.stem, item.tile_id);

  // Highlight in GT list
  highlightGTItem(item.cx, item.cy);
}

async function loadTile(stem, tileId, hlCx, hlCy, hlW, hlH) {
  S.currentStem   = stem;
  S.currentTileId = tileId;

  // Fetch GT list for this tile
  const tdata = await fetch(
    `/api/tile/${stem}/${tileId}?hcx=${hlCx}&hcy=${hlCy}&hw=${hlW}&hh=${hlH}&pred=${S.showPred?1:0}`
  ).then(r => r.json()).catch(() => null);

  if (tdata && !tdata.error) {
    // Merge pending deletes into GT list
    S.currentTileGT = (tdata.gt || []).filter(b => !isPendingDelete(tileId, b));
    renderTileImg(tdata.image);
    renderGTList(S.currentTileGT, tileId, hlCx, hlCy);
  }
}

async function loadTileImage(stem, tileId, hlCx, hlCy, hlW, hlH) {
  const tdata = await fetch(
    `/api/tile/${stem}/${tileId}?hcx=${hlCx}&hcy=${hlCy}&hw=${hlW}&hh=${hlH}&pred=${S.showPred?1:0}`
  ).then(r => r.json()).catch(() => null);

  if (tdata && !tdata.error) {
    renderTileImg(tdata.image);
  }
}

function renderTileImg(b64) {
  const img = document.getElementById('tile-img');
  const load = document.getElementById('tile-loading');
  load.style.display = 'none';
  img.style.display = 'block';
  img.src = 'data:image/jpeg;base64,' + b64;
}

async function loadThumb(stem, tileId) {
  const data = await fetch(`/api/thumb/${stem}?hl=${tileId}`).then(r=>r.json()).catch(()=>null);
  if (!data || data.error) return;
  const img = document.getElementById('thumb-img');
  img.src = 'data:image/jpeg;base64,' + data.image;
  img.style.display = 'block';
}

async function loadCropZoom(stem, tileId, cx, cy, w, h) {
  const data = await fetch(`/api/box_crop/${stem}/${tileId}?cx=${cx}&cy=${cy}&w=${w}&h=${h}`)
                .then(r=>r.json()).catch(()=>null);
  const img  = document.getElementById('crop-img');
  const ph   = document.getElementById('crop-placeholder');
  if (data && data.image) {
    img.src = 'data:image/jpeg;base64,' + data.image;
    img.style.display = 'block';
    ph.style.display  = 'none';
  }
}

// ══════════════════════════════════════════════════════
//  GT list (right panel)
// ══════════════════════════════════════════════════════
function renderGTList(gt, tileId, hlCx, hlCy) {
  const el = document.getElementById('tile-gt-list');
  el.innerHTML = '';
  document.getElementById('gt-count').textContent = gt.length;
  if (gt.length === 0) {
    el.innerHTML = '<div style="font-size:9px;color:#4b5563;padding:6px">无 GT 框</div>';
    return;
  }
  gt.forEach((b, i) => {
    const isHL = (Math.abs(b.cx - hlCx) < 0.003 && Math.abs(b.cy - hlCy) < 0.003);
    const div = document.createElement('div');
    div.className = 'gt-item' + (isHL ? ' hl-gt' : '');
    div.dataset.idx = i;
    const color = GT_COLORS[b.cls] || '#9ca3af';
    div.innerHTML = `
      <span class="gt-dot" style="background:${color}"></span>
      <span style="color:${color};font-size:9px">${CLASS_NAMES[b.cls]||b.cls}</span>
      <span class="gt-coords">${b.cx.toFixed(3)},${b.cy.toFixed(3)}</span>
      <button class="gt-del-btn" onclick="deleteGTByIdx(${i}, event)" title="删除">✕</button>
    `;
    // Click to jump to this box in queue
    div.onclick = (e) => {
      if (e.target.classList.contains('gt-del-btn')) return;
      const qIdx = S.filtered.findIndex(q =>
        q.tile_id === tileId && Math.abs(q.cx - b.cx) < 0.003);
      if (qIdx >= 0) jumpToQueue(qIdx);
    };
    el.appendChild(div);
  });
}

function highlightGTItem(cx, cy) {
  document.querySelectorAll('.gt-item').forEach(el => {
    const b = S.currentTileGT[parseInt(el.dataset.idx)];
    if (!b) return;
    const isHL = Math.abs(b.cx - cx) < 0.003 && Math.abs(b.cy - cy) < 0.003;
    el.classList.toggle('hl-gt', isHL);
    if (isHL) el.scrollIntoView({ block: 'nearest' });
  });
}

function deleteGTByIdx(idx, e) {
  e.stopPropagation();
  const box = S.currentTileGT[idx];
  if (!box) return;
  markPendingDelete(S.currentTileId, box);
  S.currentTileGT.splice(idx, 1);
  S.unsaved = true;
  updateSaveBadge(true);

  const item = currentQueueItem();
  renderGTList(S.currentTileGT, S.currentTileId,
    item ? item.cx : -1, item ? item.cy : -1);
}

// ══════════════════════════════════════════════════════
//  Keep / Delete actions
// ══════════════════════════════════════════════════════
function keepBox() {
  const item = currentQueueItem();
  if (!item) return;
  const origIdx = S.queue.indexOf(item);
  S.reviewed[origIdx] = 'keep';
  S.keptCount++;
  updateQueueItemStyle(S.queueIdx, 'keep');
  updateProgress();
  navigate(1);
}

function deleteBox() {
  const item = currentQueueItem();
  if (!item) return;

  // Remove from in-memory GT
  const tileId = item.tile_id;
  const boxKey = boxKeyOf(item);
  const prevLen = S.currentTileGT.length;
  S.currentTileGT = S.currentTileGT.filter(b =>
    !(Math.abs(b.cx - item.cx) < 0.003 && Math.abs(b.cy - item.cy) < 0.003)
  );

  if (S.currentTileGT.length < prevLen) {
    markPendingDelete(tileId, item);
    S.unsaved = true;
    updateSaveBadge(true);
  }

  const origIdx = S.queue.indexOf(item);
  S.reviewed[origIdx] = 'delete';
  S.deletedCount++;
  updateQueueItemStyle(S.queueIdx, 'delete');
  updateProgress();

  // Refresh GT list and tile image
  const next = S.filtered[S.queueIdx + 1];
  const nextSameTile = next && next.tile_id === tileId;

  renderGTList(S.currentTileGT, tileId, next ? next.cx : -1, next ? next.cy : -1);

  // Re-render tile to show deletion (only if staying on same tile)
  if (nextSameTile && next) {
    loadTileImage(item.stem, tileId, next.cx, next.cy, next.w, next.h);
  }

  navigate(1);
}

function markPendingDelete(tileId, box) {
  if (!S.pendingDeletes[tileId]) S.pendingDeletes[tileId] = new Set();
  S.pendingDeletes[tileId].add(boxKeyOf(box));
}

function isPendingDelete(tileId, box) {
  return S.pendingDeletes[tileId] &&
         S.pendingDeletes[tileId].has(boxKeyOf(box));
}

function boxKeyOf(box) {
  return `${box.cx.toFixed(4)}_${box.cy.toFixed(4)}`;
}

function currentQueueItem() {
  return S.filtered[S.queueIdx] || null;
}

// ══════════════════════════════════════════════════════
//  Save
// ══════════════════════════════════════════════════════
async function saveCurrentTile(silent = false) {
  if (!S.currentTileId) return;
  const resp = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tile_id: S.currentTileId, boxes: S.currentTileGT }),
  }).then(r => r.json()).catch(() => ({ ok: false }));

  if (!silent) {
    const st = document.getElementById('save-status');
    st.textContent = resp.ok ? '✓ 已保存' : '✗ 保存失败';
    st.style.color  = resp.ok ? '#10b981' : '#ef4444';
    setTimeout(() => { st.textContent = ''; }, 2000);
  }
  if (resp.ok) {
    S.unsaved = false;
    updateSaveBadge(false);
    // Clear pending deletes for this tile
    delete S.pendingDeletes[S.currentTileId];
  }
}

function updateSaveBadge(show) {
  const b = document.getElementById('save-badge');
  b.style.display = show ? 'inline' : 'none';
}

// ══════════════════════════════════════════════════════
//  UI helpers
// ══════════════════════════════════════════════════════
function updateBoxInfo(item) {
  document.getElementById('info-tile').textContent =
    item.tile_id.replace(/.*_y/, 'y').replace('_x', ',x');
  const clsName = CLASS_NAMES[item.cls] || '?';
  const clsEl = document.getElementById('info-cls');
  clsEl.textContent  = clsName;
  clsEl.style.color  = GT_COLORS[item.cls] || '#9ca3af';
  document.getElementById('info-frangi').textContent = item.frangi_max.toFixed(4);
  document.getElementById('info-reason').textContent = item.reason.replace(/_/g, ' ');
  const confWrap = document.getElementById('info-modelconf');
  if (item.model_conf >= 0) {
    confWrap.style.display = 'inline';
    document.getElementById('info-conf').textContent = item.model_conf.toFixed(3);
  } else {
    confWrap.style.display = 'none';
  }
}

function updateQueueItemStyle(filteredIdx, status) {
  const el = document.querySelector(`.queue-item[data-idx="${filteredIdx}"]`);
  if (!el) return;
  el.classList.remove('reviewed-keep', 'reviewed-delete');
  if (status === 'keep')   el.classList.add('reviewed-keep');
  if (status === 'delete') el.classList.add('reviewed-delete');
}

function updateProgress() {
  const total   = S.queue.length;
  const done    = Object.keys(S.reviewed).length;
  const pct     = total > 0 ? Math.round(done / total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent =
    `已审核 ${done}/${total} | 保留 ${S.keptCount} | 删除 ${S.deletedCount}`;
}

function togglePred() {
  S.showPred = !S.showPred;
  document.getElementById('btn-show-pred').classList.toggle('active', S.showPred);
  const item = currentQueueItem();
  if (item) loadTileImage(item.stem, item.tile_id, item.cx, item.cy, item.w, item.h);
}

function toggleCrop() {
  S.showCrop = !S.showCrop;
  document.getElementById('btn-show-crop').classList.toggle('active', S.showCrop);
  const section = document.getElementById('crop-section');
  section.style.display = S.showCrop ? '' : 'none';
  if (S.showCrop) {
    const item = currentQueueItem();
    if (item) loadCropZoom(item.stem, item.tile_id, item.cx, item.cy, item.w, item.h);
  }
}

// Full image modal
async function openFullView() {
  if (!S.currentStem) return;
  const item = currentQueueItem();
  const hlParam = item ? `&hl=${item.tile_id}&hcx=${item.cx}&hcy=${item.cy}` : '';
  const data = await fetch(`/api/image/${S.currentStem}?gt=1&pred=${S.showPred?1:0}${hlParam}`)
                .then(r=>r.json()).catch(()=>null);
  if (!data || data.error) return;
  document.getElementById('fullimg-modal-img').src = 'data:image/jpeg;base64,' + data.image;
  document.getElementById('fullimg-modal').style.display = 'block';
}
function closeFullView() {
  document.getElementById('fullimg-modal').style.display = 'none';
}

// ══════════════════════════════════════════════════════
//  Keyboard shortcuts
// ══════════════════════════════════════════════════════
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'k' || e.key === 'K') {
    keepBox();
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    navigate(-1);
  } else if (e.key === 'd' || e.key === 'D' || e.key === 'Delete') {
    deleteBox();
  } else if (e.key === 's' || e.key === 'S') {
    saveCurrentTile();
  } else if (e.key === 'Escape') {
    closeFullView();
  }
});

init();
</script>
</body>
</html>"""


# ─── 主函数 ────────────────────────────────────────────────

def main():
    global DATA

    parser = argparse.ArgumentParser(
        description="交互式标注审核工具 — 逐框审核视图 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port",  type=int, default=7860, help="服务端口 (默认 7860)")
    parser.add_argument("--split", default="train",
                        choices=["val", "train", "both"],
                        help="审核 split (默认 train，Frangi 队列在 train 集)")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Microlens_DF  标注审核工具  v2 (逐框审核)         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  split: {args.split}  | port: {args.port}")
    print()

    missing = []
    for p in [TILE_DATASET, IMAGES_DIR]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("  ✗ 缺少必要目录:")
        for m in missing:
            print(f"      {m}")
        sys.exit(1)

    DATA = ReviewData(splits=splits)
    print()

    if not DATA.source_images:
        print(f"  ✗ 未找到 {args.split} 集图像，请检查 tile_index.csv 和标注目录")
        sys.exit(1)

    queue_path = AUDIT_DIR / "frangi_review_queue.json"
    if queue_path.exists():
        q = json.loads(queue_path.read_text())
        n_queue = sum(e["n_weak_boxes"] for e in q
                      if e["split"] in splits)
        print(f"  Frangi 审核队列: {n_queue} 个可疑框")
    else:
        print("  ⚠  未找到 frangi_review_queue.json，请先运行 step1b_frangi_verify.py")
    print()

    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    url = f"http://localhost:{args.port}"
    lan_ip = _get_local_ip()
    print(f"  ✓ 服务已启动: {url}")
    if lan_ip:
        print(f"  ✓ 局域网访问: http://{lan_ip}:{args.port}")
    print()
    print("  快捷键:")
    print("    K / → / ↓  — 保留此框并跳下一个")
    print("    D / Delete  — 删除此框并跳下一个")
    print("    ← / ↑      — 跳上一框")
    print("    S           — 保存本切片修改到磁盘")
    print("    Esc         — 关闭全图视图")
    print()
    print("  按 Ctrl+C 停止服务")
    print()

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止。")


def _get_local_ip() -> str:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return ""


if __name__ == "__main__":
    main()
