#!/usr/bin/env python3
"""
全图级检测工具集
================
SAHI 推理与半监督重标注共用的核心算法：

  - 切片 ↔ 全图坐标映射
  - 全图级 NMS（IOS 度量，适合长短不一的划痕）
  - Scratch 连接算法（共线碎片 → 完整划痕）
  - GT-Prediction 合并逻辑
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from pathlib import Path

TILE_SIZE    = 640
CLASS_NAMES  = ["scratch", "spot", "critical"]


# ═══════════════════════════════════════════════════════
#  坐标映射
# ═══════════════════════════════════════════════════════

def tile_boxes_to_fullimage(
    yolo_boxes: list[tuple],
    tile_x0: int,
    tile_y0: int,
    tile_size: int = TILE_SIZE,
) -> list[tuple]:
    """
    YOLO 归一化 → 全图像素坐标。

    输入:  [(cls, cx, cy, w, h, conf?), ...]   （归一化 0-1）
    输出:  [(cls, x1, y1, x2, y2, conf)]       （全图像素）
    """
    result = []
    for box in yolo_boxes:
        cls = box[0]
        cx_n, cy_n, w_n, h_n = box[1], box[2], box[3], box[4]
        conf = box[5] if len(box) > 5 else 1.0

        cx_px = tile_x0 + cx_n * tile_size
        cy_px = tile_y0 + cy_n * tile_size
        w_px  = w_n * tile_size
        h_px  = h_n * tile_size

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2

        result.append((cls, x1, y1, x2, y2, conf))
    return result


def fullimage_to_tile_boxes(
    full_boxes: list[tuple],
    tile_x0: int,
    tile_y0: int,
    tile_size: int = TILE_SIZE,
    min_visible_frac: float = 0.15,
    min_box_px: int = 4,
) -> list[tuple]:
    """
    全图像素坐标 → YOLO 归一化（裁剪到切片可见区域）。

    输入:  [(cls, x1, y1, x2, y2, conf)]
    输出:  [(cls, cx, cy, w, h)]      （无 conf，用于 GT 标注）

    仅保留在切片中可见比例 > min_visible_frac 的框。
    """
    result = []
    for cls, x1, y1, x2, y2, conf in full_boxes:
        # 裁剪到切片范围
        cx1 = max(x1, tile_x0)
        cy1 = max(y1, tile_y0)
        cx2 = min(x2, tile_x0 + tile_size)
        cy2 = min(y2, tile_y0 + tile_size)

        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < min_box_px or ch < min_box_px:
            continue

        orig_area = max((x2 - x1) * (y2 - y1), 1e-6)
        clip_area = cw * ch
        if clip_area / orig_area < min_visible_frac:
            continue

        # 归一化
        ncx = ((cx1 + cx2) / 2 - tile_x0) / tile_size
        ncy = ((cy1 + cy2) / 2 - tile_y0) / tile_size
        nw  = cw / tile_size
        nh  = ch / tile_size

        if not (0 <= ncx <= 1 and 0 <= ncy <= 1):
            continue
        nw = min(nw, 1.0)
        nh = min(nh, 1.0)

        if nw * nh < 1e-4:
            continue

        result.append((cls, ncx, ncy, nw, nh))
    return result


# ═══════════════════════════════════════════════════════
#  全图级 NMS  (IOS: Intersection over Smaller area)
# ═══════════════════════════════════════════════════════

def nms_ios(
    boxes: list[tuple],
    ios_thresh: float = 0.35,
) -> list[tuple]:
    """
    IOS 度量 NMS —— 更适合长短不一的划痕重叠。
    短框被长框包含时 IOS → 1.0，会被合并。

    boxes: [(cls, x1, y1, x2, y2, conf)]
    """
    if len(boxes) <= 1:
        return list(boxes)

    by_cls: dict[int, list] = defaultdict(list)
    for b in boxes:
        by_cls[b[0]].append(b)

    result = []
    for cls, cls_boxes in by_cls.items():
        cls_boxes.sort(key=lambda b: b[5], reverse=True)

        keep: list[tuple] = []
        suppressed: set[int] = set()

        for i, bi in enumerate(cls_boxes):
            if i in suppressed:
                continue
            keep.append(bi)

            for j in range(i + 1, len(cls_boxes)):
                if j in suppressed:
                    continue
                bj = cls_boxes[j]

                ix1 = max(bi[1], bj[1])
                iy1 = max(bi[2], bj[2])
                ix2 = min(bi[3], bj[3])
                iy2 = min(bi[4], bj[4])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                area_i = (bi[3] - bi[1]) * (bi[4] - bi[2])
                area_j = (bj[3] - bj[1]) * (bj[4] - bj[2])
                smaller = min(area_i, area_j)

                if smaller > 0 and inter / smaller > ios_thresh:
                    suppressed.add(j)

        result.extend(keep)
    return result


# ═══════════════════════════════════════════════════════
#  Scratch 连接算法
# ═══════════════════════════════════════════════════════

def _scratch_endpoints(x1, y1, x2, y2):
    """计算 scratch 框的主轴端点和角度。"""
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    if w > h * 1.5:
        # 水平方向
        angle = 0.0
        ep1, ep2 = (x1, cy), (x2, cy)
    elif h > w * 1.5:
        # 垂直方向
        angle = 90.0
        ep1, ep2 = (cx, y1), (cx, y2)
    else:
        # 近正方形或对角线
        angle = float(np.degrees(np.arctan2(h, w)))
        ep1, ep2 = (x1, y1), (x2, y2)

    return ep1, ep2, angle


def connect_scratches(
    boxes: list[tuple],
    max_gap: float = 100.0,
    max_angle_diff: float = 30.0,
) -> list[tuple]:
    """
    合并共线且相邻的 scratch 检测框（只处理 cls=0）。

    算法:
      1. 计算每个 scratch 框的主轴端点和角度
      2. 端点距离 < max_gap AND 角度差 < max_angle_diff → 建立连接
      3. Union-Find 合并连通分量
      4. 每个分量 → 一个外接矩形

    boxes: [(cls, x1, y1, x2, y2, conf)]
    """
    scratches = [(i, b) for i, b in enumerate(boxes) if b[0] == 0]
    others    = [b for b in boxes if b[0] != 0]

    if len(scratches) <= 1:
        return list(boxes), 0

    infos = []
    for idx, b in scratches:
        cls, x1, y1, x2, y2, conf = b
        ep1, ep2, angle = _scratch_endpoints(x1, y1, x2, y2)
        infos.append({
            "box": b, "ep1": ep1, "ep2": ep2, "angle": angle, "conf": conf,
        })

    n = len(infos)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    for i in range(n):
        for j in range(i + 1, n):
            a, b = infos[i], infos[j]

            # 角度兼容性（近正方形 40~50° 放宽判定）
            da = abs(a["angle"] - b["angle"])
            da = min(da, 180 - da)
            a_sq = 30 < a["angle"] < 60
            b_sq = 30 < b["angle"] < 60
            if a_sq or b_sq:
                da = min(da, max_angle_diff)   # 宽容
            if da > max_angle_diff:
                continue

            # 端点最短距离
            min_d = float("inf")
            for ea in [a["ep1"], a["ep2"]]:
                for eb in [b["ep1"], b["ep2"]]:
                    d = ((ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2) ** 0.5
                    min_d = min(min_d, d)

            if min_d < max_gap:
                union(i, j)

    # 合并各连通分量
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    merged = []
    n_chains = 0
    for indices in components.values():
        x1 = min(infos[i]["box"][1] for i in indices)
        y1 = min(infos[i]["box"][2] for i in indices)
        x2 = max(infos[i]["box"][3] for i in indices)
        y2 = max(infos[i]["box"][4] for i in indices)
        conf = max(infos[i]["conf"] for i in indices)
        merged.append((0, x1, y1, x2, y2, conf))
        if len(indices) > 1:
            n_chains += 1

    return merged + others, n_chains


# ═══════════════════════════════════════════════════════
#  GT + 模型预测合并
# ═══════════════════════════════════════════════════════

def _iou(b1, b2):
    ix1 = max(b1[1], b2[1])
    iy1 = max(b1[2], b2[2])
    ix2 = min(b1[3], b2[3])
    iy2 = min(b1[4], b2[4])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[3] - b1[1]) * (b1[4] - b1[2])
    a2 = (b2[3] - b2[1]) * (b2[4] - b2[2])
    return inter / max(a1 + a2 - inter, 1e-8)


def merge_gt_with_predictions(
    gt_boxes: list[tuple],
    pred_boxes: list[tuple],
    min_conf_add: float = 0.35,
    min_iou_match: float = 0.15,
    extend_ratio: float = 1.5,
) -> tuple[list[tuple], dict]:
    """
    将模型连接后的预测与 GT 合并。

    规则:
      1. GT 框全部保留
      2. 模型预测与 GT 重叠（IoU > min_iou_match）:
         - 预测面积 > GT × extend_ratio → 扩展 GT 到并集
         - 否则 → GT 已被确认，不变
      3. 模型预测无 GT 重叠 且 conf > min_conf_add → 新增

    返回: (merged_boxes, stats_dict)
    """
    merged = list(gt_boxes)      # 全部 GT 作为基础
    matched_gt = set()
    stats = {"extended": 0, "added": 0, "confirmed": 0}

    for pred in pred_boxes:
        cls_p, x1p, y1p, x2p, y2p, conf_p = pred
        pred_area = (x2p - x1p) * (y2p - y1p)

        # 找最佳匹配 GT
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(merged):
            if gt[0] != cls_p:
                continue
            v = _iou(pred, gt)
            if v > best_iou:
                best_iou = v
                best_idx = i

        if best_iou > min_iou_match and best_idx >= 0:
            gt = merged[best_idx]
            gt_area = (gt[3] - gt[1]) * (gt[4] - gt[2])

            if pred_area > gt_area * extend_ratio and conf_p > 0.30:
                # 扩展 GT 到并集
                new_x1 = min(gt[1], x1p)
                new_y1 = min(gt[2], y1p)
                new_x2 = max(gt[3], x2p)
                new_y2 = max(gt[4], y2p)
                merged[best_idx] = (cls_p, new_x1, new_y1, new_x2, new_y2, 1.0)
                stats["extended"] += 1
            else:
                stats["confirmed"] += 1
            matched_gt.add(best_idx)
        else:
            if conf_p >= min_conf_add:
                merged.append((cls_p, x1p, y1p, x2p, y2p, 1.0))
                stats["added"] += 1

    return merged, stats


# ═══════════════════════════════════════════════════════
#  辅助：从全图生成切片位置
# ═══════════════════════════════════════════════════════

def generate_tile_positions(
    img_h: int,
    img_w: int,
    tile_size: int = TILE_SIZE,
    stride: int = 480,
) -> list[tuple[int, int]]:
    """
    生成 (y0, x0) 切片位置列表。
    stride < tile_size → 产生重叠。
    """
    positions = []
    y = 0
    while y + tile_size <= img_h:
        x = 0
        while x + tile_size <= img_w:
            positions.append((y, x))
            x += stride
        # 最右列对齐右边界
        if x < img_w and (img_w - tile_size) != positions[-1][1]:
            positions.append((y, img_w - tile_size))
        y += stride
    # 最底行对齐下边界
    if y < img_h:
        x = 0
        while x + tile_size <= img_w:
            positions.append((img_h - tile_size, x))
            x += stride
        if x < img_w:
            positions.append((img_h - tile_size, img_w - tile_size))
    return positions
