#!/usr/bin/env python3
"""
完整推理流水线 — 单张或批量镜片图像
=================================================
SAHI 滑窗推理 → 全图 NMS → Scratch 连接 → WearMetrics → WearScore
（可选：分割模型补充精确 mask 量化）

使用方式:
    # 单张图（原图 PNG）
    python scripts/infer_full_pipeline.py --image output/dataset_v2/images/13r.png

    # 批量（指定 stem 列表）
    python scripts/infer_full_pipeline.py --stems 123l 84r 51r

    # 20 张测试集
    python scripts/infer_full_pipeline.py --test-set output/audit/test_set_v1.json

    # 自定义权重
    python scripts/infer_full_pipeline.py --stems 13r \\
        --weights output/training/stage2_cleaned/weights/best.pt

    # 启用分割模型（补充 mask 量化指标）
    python scripts/infer_full_pipeline.py --image output/dataset_v2/images/13r.png \\
        --use-segmentation

输出（per image）:
    output/pipeline_results/<stem>/
        detection.png    — 全图检测可视化（含 mask overlay）
        segmentation.png — 分割掩码可视化（--use-segmentation 时）
        report.json      — WearMetrics + WearScore + 检测统计 + mask 量化
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fullimage_utils import (
    generate_tile_positions,
    tile_boxes_to_fullimage,
    nms_ios,
    connect_scratches,
    TILE_SIZE,
)

# ── 路径 ──────────────────────────────────────────────────────────────
IMAGES_DIR   = PROJECT_ROOT / "output" / "dataset_v2" / "images"
WEIGHTS_PATH = (PROJECT_ROOT / "output" / "training" /
                "stage2_cleaned" / "weights" / "best.pt")
SEG_WEIGHTS_PATH = (PROJECT_ROOT / "output" / "experiments" /
                    "phase3_segmentation" / "private_finetuned" / "best.pt")
OUT_DIR      = PROJECT_ROOT / "output" / "pipeline_results"

# ── 推理参数 ───────────────────────────────────────────────────────────
CONF_THRESH  = 0.20   # 推理置信度（比 step4 略高，减少噪声）
STRIDE       = 480    # 滑窗步长（与训练时 overlap 匹配）
NMS_IOS      = 0.35   # 全图 IOS NMS 阈值
SCRATCH_GAP  = 100    # scratch 连接最大端点距离（px）
SCRATCH_ANGLE = 30    # scratch 连接最大角度差（度）
BATCH_SIZE   = 16

# ── 类别 ──────────────────────────────────────────────────────────────
CLASS_NAMES  = ["scratch", "spot", "critical"]
CLASS_COLORS = {
    0: (0, 200, 60),    # scratch — 绿
    1: (255, 180, 0),   # spot    — 橙
    2: (0, 120, 255),   # critical— 蓝
}


# ═══════════════════════════════════════════════════════════════════════
#  WearMetrics 适配层
#  — 将 YOLO 全图 bbox (cls, x1, y1, x2, y2, conf) 转换为物理量化指标
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BboxDefect:
    """YOLO bbox 缺陷（WearMetrics 适配用）."""
    cls: int
    x1: int; y1: int; x2: int; y2: int
    conf: float
    # 计算属性
    length_px: float = 0.0      # 划痕：max(w,h)；spot：等效直径；critical：√(w·h)
    area_px: int = 0            # bbox 面积
    scatter_intensity: float = 0.0  # bbox 区域原图均值（背景校正后）
    zone: str = "edge"          # center / transition / edge


def boxes_to_defects(
    boxes: list[tuple],
    img: np.ndarray,
    img_h: int,
    img_w: int,
    center_xy: tuple[float, float] | None = None,
    lens_radius: float | None = None,
) -> list[BboxDefect]:
    """将全图 bbox 列表转换为 BboxDefect，计算物理量化代理值.

    Args:
        boxes:      [(cls, x1, y1, x2, y2, conf), ...]  全图像素坐标
        img:        原始灰度图（背景校正后），用于采样散射强度
        img_h/w:    图像尺寸
        center_xy:  镜片中心（像素），None 则取图像中心
        lens_radius: 镜片有效半径（像素），None 则取 min(H,W)*0.43
    """
    if center_xy is None:
        center_xy = (img_w / 2, img_h / 2)
    if lens_radius is None:
        lens_radius = min(img_h, img_w) * 0.43

    cx, cy = center_xy
    # 背景基准：取全图 p5（暗场背景接近 0，取低百分位作为噪声底）
    bg_val = float(np.percentile(img, 5))

    defects = []
    for cls, x1, y1, x2, y2, conf in boxes:
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        area = w * h

        # 长度代理
        if cls == 0:    # scratch：线状，取长轴
            length = float(max(w, h))
        elif cls == 1:  # spot：近圆形，取等效直径
            length = float(np.sqrt(area))
        else:           # critical：大面积，取几何均值边长
            length = float(np.sqrt(area))

        # 散射强度：bbox 区域原图均值（减背景）
        roi = img[max(0,int(y1)):min(img_h,int(y2)), max(0,int(x1)):min(img_w,int(x2))]
        scatter = max(0.0, float(roi.mean()) - bg_val) if roi.size > 0 else 0.0

        # Zone：以 bbox 中心到镜片中心的距离判断
        bcx = (x1 + x2) / 2
        bcy = (y1 + y2) / 2
        dist = np.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2)
        r = dist / lens_radius
        if r < 0.40:
            zone = "center"
        elif r < 0.75:
            zone = "microstructure"
        else:
            zone = "edge"

        defects.append(BboxDefect(
            cls=cls, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf,
            length_px=length, area_px=area,
            scatter_intensity=scatter, zone=zone,
        ))
    return defects


def compute_wear_metrics(
    defects: list[BboxDefect],
    img_h: int, img_w: int,
    lens_radius: float | None = None,
) -> dict[str, Any]:
    """从 BboxDefect 列表计算磨损量化指标（bbox 代理版）."""
    if lens_radius is None:
        lens_radius = min(img_h, img_w) * 0.43
    roi_area = int(np.pi * lens_radius ** 2)
    micro_area = max(int(roi_area * 0.4025), 1)  # 微结构环带面积（π·R²·(0.75²−0.40²)）

    metrics = dict(
        N_total=0, N_scratch=0, N_spot=0, N_critical=0,
        N_center=0, N_microstructure=0, N_edge=0,
        L_total=0.0, L_center=0.0, L_microstructure=0.0, L_edge=0.0,
        A_total=0, A_center=0, A_microstructure=0,
        S_scatter=0.0, S_scatter_center=0.0, S_scatter_micro=0.0,
        D_density=0.0, D_micro_density=0.0,
        roi_area=roi_area, micro_area=micro_area,
    )

    weighted_s = 0.0
    weighted_s_center = 0.0
    weighted_s_micro = 0.0

    for d in defects:
        metrics["N_total"] += 1
        metrics["L_total"] += d.length_px
        metrics["A_total"] += d.area_px
        weighted_s += d.scatter_intensity * d.area_px

        if d.cls == 0:
            metrics["N_scratch"] += 1
        elif d.cls == 1:
            metrics["N_spot"] += 1
        else:
            metrics["N_critical"] += 1

        if d.zone == "center":
            metrics["N_center"] += 1
            metrics["L_center"] += d.length_px
            metrics["A_center"] += d.area_px
            weighted_s_center += d.scatter_intensity * d.area_px
        elif d.zone == "microstructure":
            metrics["N_microstructure"] += 1
            metrics["L_microstructure"] += d.length_px
            metrics["A_microstructure"] += d.area_px
            weighted_s_micro += d.scatter_intensity * d.area_px
        else:
            metrics["N_edge"] += 1
            metrics["L_edge"] += d.length_px

    if metrics["A_total"] > 0:
        metrics["S_scatter"] = weighted_s / metrics["A_total"]
    if metrics["A_center"] > 0:
        metrics["S_scatter_center"] = weighted_s_center / metrics["A_center"]
    if metrics["A_microstructure"] > 0:
        metrics["S_scatter_micro"] = weighted_s_micro / metrics["A_microstructure"]
    if roi_area > 0:
        metrics["D_density"] = metrics["L_total"] / roi_area
    metrics["D_micro_density"] = metrics["L_microstructure"] / micro_area

    for k, v in metrics.items():
        if isinstance(v, float):
            metrics[k] = round(v, 4)

    return metrics


def compute_wear_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """基于磨损指标计算 WearScore 和 A/B/C/D 等级.

    权重设计根据离焦微结构镜片光学功能区优先级:
        w_micro_scratch  (0.35) — 微结构区划痕长度，直接损伤功能环带
        w_micro_density  (0.25) — 微结构区缺陷密度，影响整体离焦控制效果
        w_critical       (0.20) — critical 缺陷数，严重缺损优先处理
        w_scatter        (0.10) — 微结构区散射强度，影响夜间眩光和舒适度
        w_total          (0.10) — 总体缺陷数，整体损伤参考
    """
    import math

    def sat_log(x: float, scale: float = 1.0, cap: float = 100.0) -> float:
        return min(scale * math.log1p(x), cap)

    # 各分项（0-100 范围估算，基于数据集统计经验值）
    # L_micro: 10000px为重度（≈3条全图横划痕），300px为轻度基线
    f_micro_scratch = sat_log(metrics["L_microstructure"] / 300.0, scale=22.0)  # 10000px → ~78
    # D_micro_density: roi≈5M px²时 micro_area≈2M px²，严重损伤≈0.005 → ~100
    f_micro_density = sat_log(metrics["D_micro_density"] * 2.5e3,  scale=25.0)  # 0.01 → ~100
    f_critical      = sat_log(metrics["N_critical"] / 2.0,        scale=22.0)  # 20个 → ~100
    f_scatter       = sat_log(metrics["S_scatter_micro"] / 5.0,   scale=20.0)  # 50 → ~100
    f_total         = sat_log(metrics["N_total"] / 10.0,          scale=12.0)  # 100个 → ~100

    score = (0.35 * f_micro_scratch +
             0.25 * f_micro_density +
             0.20 * f_critical +
             0.10 * f_scatter +
             0.10 * f_total)
    score = min(100.0, round(score, 1))

    # 眩光指数（微结构区划痕 × 微结构散射强度）
    glare = round(metrics["L_microstructure"] * metrics.get("S_scatter_micro", 0) / 1000, 2)
    # 雾化指数（微结构密度 × critical 占比）
    n_total = max(metrics["N_total"], 1)
    haze = round(metrics["D_micro_density"] * metrics["N_critical"] / n_total * 1000, 2)

    # 等级划分
    if score < 25:
        grade, label = "A", "优秀（微结构完好，功能正常）"
    elif score < 50:
        grade, label = "B", "良好（轻微损伤，功能基本完好）"
    elif score < 80:
        grade, label = "C", "警告（微结构受损，建议更换）"
    else:
        grade, label = "D", "报废（功能区严重损伤，需立即更换）"

    contributors = {
        "micro_scratch_length": round(f_micro_scratch, 1),
        "micro_defect_density": round(f_micro_density, 1),
        "critical_defects":     round(f_critical, 1),
        "scatter_intensity":    round(f_scatter, 1),
        "total_defects":        round(f_total, 1),
    }
    dominant = max(contributors, key=contributors.get)

    return {
        "score":           score,
        "grade":           grade,
        "grade_label":     label,
        "glare_index":     glare,
        "haze_index":      haze,
        "contributors":    contributors,
        "dominant_factor": dominant,
    }


# ═══════════════════════════════════════════════════════════════════════
#  推理核心
# ═══════════════════════════════════════════════════════════════════════

def run_sahi(model, img_path: Path, conf: float = CONF_THRESH) -> tuple[list, float]:
    """对单张全图运行 SAHI 推理，返回 (全图 boxes, 耗时秒)."""
    t0 = time.time()
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像不存在: {img_path}")
    H, W = img.shape

    # 转 3 通道（模型期望 3ch）
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 滑窗位置
    positions = generate_tile_positions(H, W, tile_size=TILE_SIZE, stride=STRIDE)

    # 切片并推理
    tiles_rgb = []
    for y0, x0 in positions:
        tile = img3[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
        if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
            # pad 到 640×640
            pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        tiles_rgb.append(tile)

    all_full = []
    for bi in range(0, len(tiles_rgb), BATCH_SIZE):
        batch = tiles_rgb[bi:bi+BATCH_SIZE]
        batch_pos = positions[bi:bi+BATCH_SIZE]
        results = model(batch, conf=conf, imgsz=TILE_SIZE,
                        batch=len(batch), stream=False, verbose=False, half=True)
        for r, (y0, x0) in zip(results, batch_pos):
            if r.boxes is None:
                continue
            tile_boxes = []
            for box in r.boxes:
                cls = int(box.cls[0])
                xywh = box.xywhn[0].tolist()
                cf = float(box.conf[0])
                if xywh[2] * xywh[3] > 0.30:
                    continue
                tile_boxes.append((cls, xywh[0], xywh[1], xywh[2], xywh[3], cf))
            full = tile_boxes_to_fullimage(tile_boxes, x0, y0)
            all_full.extend(full)

    return all_full, time.time() - t0


def load_seg_model(seg_weights: Path):
    """加载分割模型 (LightUNet)，返回 (model, device) 元组。"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from darkfield_defects.ml.predict import load_model
    import torch
    # load_model 使用 weights_only=True，但私有检查点需要 False
    import torch
    ckpt = torch.load(str(seg_weights), map_location="cpu", weights_only=False)
    from darkfield_defects.ml.models import LightUNet
    model = LightUNet(
        in_channels=ckpt.get("in_channels", 1),
        num_classes=ckpt.get("num_classes", 4),
        base_features=ckpt.get("base_features", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def _run_segmentation(seg_model, device, img: np.ndarray, out_dir: Path, visualize: bool) -> dict:
    """在全图上运行分割模型，返回 mask_metrics 字典。"""
    import sys, torch
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from darkfield_defects.ml.predict import predict_full_image, mask_to_detection_result

    torch_device = torch.device(device)
    seg_mask = predict_full_image(seg_model, img, torch_device)

    # mask 量化：骨架长度 + 真实像素面积
    mask_metrics = {
        "scratch_px":  int(np.count_nonzero(seg_mask == 1)),
        "spot_px":     int(np.count_nonzero(seg_mask == 2)),
        "damage_px":   int(np.count_nonzero(seg_mask == 3)),
    }

    # 用 mask_to_detection_result 获取实例级测量
    result = mask_to_detection_result(seg_mask, img)
    scratch_instances = [inst for inst in result.instances if inst.defect_type.name == "SCRATCH"]
    mask_metrics["scratch_instances"] = len(scratch_instances)
    mask_metrics["scratch_skeleton_px"] = int(sum(inst.length for inst in scratch_instances))

    if visualize:
        _save_seg_visualization(img, seg_mask, out_dir)

    return mask_metrics


def _save_seg_visualization(img: np.ndarray, seg_mask: np.ndarray, out_dir: Path) -> None:
    """保存分割掩码可视化（overlay）。"""
    H, W = img.shape
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()

    SEG_COLORS = {
        1: (0, 220, 50),    # scratch — 绿
        2: (255, 160, 0),   # spot    — 橙
        3: (0, 80, 255),    # damage  — 蓝
    }
    for cls_id, color in SEG_COLORS.items():
        mask = seg_mask == cls_id
        overlay[mask] = color

    blended = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    scale = 0.25
    blended_s = cv2.resize(blended, (int(W * scale), int(H * scale)))
    cv2.imwrite(str(out_dir / "segmentation.png"), blended_s)


def process_image(
    model,
    img_path: Path,
    out_dir: Path,
    conf: float = CONF_THRESH,
    visualize: bool = True,
    seg_model=None,
    seg_device: str = "cpu",
) -> dict[str, Any]:
    """完整流水线：SAHI → NMS → 连接 → WearMetrics → WearScore（可选分割）。"""
    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. SAHI 推理
    raw_boxes, infer_time = run_sahi(model, img_path, conf)

    # 2. 全图 IOS NMS
    nms_boxes = nms_ios(raw_boxes, ios_thresh=NMS_IOS)

    # 3. Scratch 连接
    final_boxes, n_chains = connect_scratches(nms_boxes, SCRATCH_GAP, SCRATCH_ANGLE)

    # 4. WearMetrics
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    defects = boxes_to_defects(final_boxes, img, H, W)
    metrics = compute_wear_metrics(defects, H, W)

    # 5. WearScore
    assessment = compute_wear_score(metrics)

    # 6. 可视化
    if visualize:
        _save_visualization(img, final_boxes, defects, stem, assessment, out_dir)

    # 7. 分割（可选）
    mask_metrics = None
    if seg_model is not None:
        mask_metrics = _run_segmentation(seg_model, seg_device, img, out_dir, visualize)

    # 8. 报告 JSON
    n_by_cls = {n: 0 for n in CLASS_NAMES}
    for d in defects:
        n_by_cls[CLASS_NAMES[d.cls]] += 1

    report = {
        "image":      stem,
        "inference": {
            "raw_boxes":    len(raw_boxes),
            "after_nms":    len(nms_boxes),
            "after_connect":len(final_boxes),
            "scratch_chains": n_chains,
            "time_sec":     round(infer_time, 2),
        },
        "detections": n_by_cls,
        "metrics":    metrics,
        "assessment": assessment,
    }
    if mask_metrics is not None:
        report["mask_metrics"] = mask_metrics

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report


def _save_visualization(
    img: np.ndarray,
    boxes: list[tuple],
    defects: list[BboxDefect],
    stem: str,
    assessment: dict,
    out_dir: Path,
) -> None:
    """保存检测可视化图（缩放到 0.25 全图 + 标注框 + WearScore）."""
    H, W = img.shape
    scale = 0.25
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for d in defects:
        color = CLASS_COLORS[d.cls]
        thick = 2 if d.cls == 0 else 1
        cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), color, thick)

    # 缩放
    vis_s = cv2.resize(vis, (int(W*scale), int(H*scale)))
    sH, sW = vis_s.shape[:2]

    # 标题
    grade = assessment["grade"]
    score = assessment["score"]
    label = assessment["grade_label"]
    grade_color = {"A":(0,200,50),"B":(0,180,255),"C":(0,120,255),"D":(0,0,220)}.get(grade,(255,255,255))

    cv2.putText(vis_s, f"{stem}  Grade {grade} ({score:.0f})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, grade_color, 2)
    cv2.putText(vis_s, label,
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # 统计行
    stats = f"raw={len(boxes)}  scratch={sum(1 for d in defects if d.cls==0)}  spot={sum(1 for d in defects if d.cls==1)}  critical={sum(1 for d in defects if d.cls==2)}"
    cv2.putText(vis_s, stats,
                (10, sH-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    cv2.imwrite(str(out_dir / "detection.png"), vis_s)


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="完整推理流水线：SAHI → NMS → Scratch连接 → WearMetrics → WearScore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",    type=str, help="单张原图路径")
    grp.add_argument("--stems",    type=str, nargs="+", help="图像 stem 列表（在 dataset_v2/images/ 中查找）")
    grp.add_argument("--test-set", type=str, help="测试集 JSON 文件（含 images 列表）")

    parser.add_argument("--weights", type=str, default=None,
                        help=f"检测模型权重（默认 Stage3 best.pt）")
    parser.add_argument("--conf",  type=float, default=CONF_THRESH,
                        help=f"推理置信度（默认 {CONF_THRESH}）")
    parser.add_argument("--no-vis", action="store_true", help="跳过可视化（加速）")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--use-segmentation", action="store_true",
                        help="启用分割模型（输出 mask_metrics + segmentation.png）")
    parser.add_argument("--seg-weights", type=str, default=None,
                        help=f"分割模型权重（默认 phase3 private_finetuned/best.pt）")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else WEIGHTS_PATH
    if not weights.exists():
        print(f"✗ 权重不存在: {weights}")
        sys.exit(1)

    out_root = Path(args.out_dir)

    # 收集图像路径
    img_paths: list[Path] = []
    if args.image:
        img_paths = [Path(args.image)]
    elif args.stems:
        for stem in args.stems:
            for ext in (".png", ".jpg", ".bmp"):
                p = IMAGES_DIR / f"{stem}{ext}"
                if p.exists():
                    img_paths.append(p)
                    break
            else:
                print(f"  ⚠ 未找到图像: {stem}")
    else:
        ts = json.loads(Path(args.test_set).read_text())
        for entry in ts["images"]:
            stem = entry["stem"]
            for ext in (".png", ".jpg", ".bmp"):
                p = IMAGES_DIR / f"{stem}{ext}"
                if p.exists():
                    img_paths.append(p)
                    break
            else:
                print(f"  ⚠ 未找到图像: {stem}")

    if not img_paths:
        print("✗ 没有找到任何图像，退出。")
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   完整推理流水线 (SAHI → WearMetrics → WearScore)    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  模型:    {weights.name}")
    print(f"  图像数:  {len(img_paths)}")
    print(f"  置信度:  {args.conf}")
    print(f"  输出:    {out_root}/")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # 分割模型（可选）
    seg_model, seg_device = None, "cpu"
    if args.use_segmentation:
        seg_weights = Path(args.seg_weights) if args.seg_weights else SEG_WEIGHTS_PATH
        if not seg_weights.exists():
            print(f"  ✗ 分割权重不存在: {seg_weights}")
            print("    请先运行: python scripts/train_private_segmentation.py")
            sys.exit(1)
        print(f"  分割模型: {seg_weights.name}")
        seg_model, seg_device = load_seg_model(seg_weights)

    t_all = time.time()
    summary = []

    for i, img_path in enumerate(img_paths, 1):
        stem = img_path.stem
        out_dir = out_root / stem
        print(f"  [{i}/{len(img_paths)}] {stem} ...", end=" ", flush=True)
        try:
            report = process_image(
                model, img_path, out_dir,
                conf=args.conf,
                visualize=not args.no_vis,
                seg_model=seg_model,
                seg_device=seg_device,
            )
            a = report["assessment"]
            print(f"Grade {a['grade']} ({a['score']:.0f})  "
                  f"defects={report['inference']['after_connect']}  "
                  f"chains={report['inference']['scratch_chains']}  "
                  f"{report['inference']['time_sec']:.1f}s")
            summary.append({
                "stem": stem,
                "grade": a["grade"],
                "score": a["score"],
                "n_defects": report["inference"]["after_connect"],
                "scratch": report["detections"]["scratch"],
                "spot":    report["detections"]["spot"],
                "critical":report["detections"]["critical"],
            })
        except Exception as e:
            print(f"✗ {e}")

    elapsed = time.time() - t_all

    # 汇总报告
    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "model": str(weights),
        "conf": args.conf,
        "n_images": len(img_paths),
        "elapsed_sec": round(elapsed, 1),
        "images": summary,
    }, ensure_ascii=False, indent=2))

    print()
    print(f"{'='*56}")
    print(f"  完成 {len(summary)}/{len(img_paths)} 张  总耗时 {elapsed:.1f}s")
    if summary:
        grades = [r["grade"] for r in summary]
        for g in "ABCD":
            n = grades.count(g)
            if n:
                print(f"  Grade {g}: {n} 张")
    print(f"  汇总报告: {summary_path}")
    print()


if __name__ == "__main__":
    main()
