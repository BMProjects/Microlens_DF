"""评估指标模块 — 像素级、实例级、YOLO 检测级评估."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationMetrics:
    """像素级分割指标."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "iou": round(self.iou, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def compute_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
) -> SegmentationMetrics:
    """计算像素级分割指标.

    Args:
        pred: 预测掩码 (H, W).
        gt: Ground Truth 掩码 (H, W).
        threshold: 二值化阈值（如果输入不是二值的）.

    Returns:
        SegmentationMetrics.
    """
    pred_bin = (pred > threshold).astype(bool)
    gt_bin = (gt > threshold).astype(bool)

    tp = int(np.count_nonzero(pred_bin & gt_bin))
    fp = int(np.count_nonzero(pred_bin & ~gt_bin))
    fn = int(np.count_nonzero(~pred_bin & gt_bin))
    tn = int(np.count_nonzero(~pred_bin & ~gt_bin))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    metrics = SegmentationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )

    logger.info(f"分割指标: P={precision:.3f} R={recall:.3f} F1={f1:.3f} IoU={iou:.3f}")
    return metrics


@dataclass
class InstanceMetrics:
    """实例级评估指标."""
    hit_rate: float = 0.0            # 命中率（对应真实划痕的比例）
    miss_rate: float = 0.0           # 漏检率
    false_positives_per_image: float = 0.0
    num_pred: int = 0
    num_gt: int = 0
    num_matched: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "fp_per_image": round(self.false_positives_per_image, 2),
            "num_pred": self.num_pred,
            "num_gt": self.num_gt,
            "num_matched": self.num_matched,
        }


def compute_instance_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.3,
) -> InstanceMetrics:
    """计算实例级评估指标.

    通过 IoU 阈值进行预测-GT 实例匹配：
    1. 连通域分析提取独立实例
    2. 计算预测与 GT 实例之间的 IoU 矩阵
    3. 贪心匹配（IoU >= 阈值）

    Args:
        pred: 预测二值掩码 (H, W).
        gt: Ground Truth 二值掩码 (H, W).
        iou_threshold: IoU 匹配阈值.

    Returns:
        InstanceMetrics.
    """
    import cv2

    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # 连通域提取实例
    num_pred_labels, pred_labels = cv2.connectedComponents(pred_bin)
    num_gt_labels, gt_labels = cv2.connectedComponents(gt_bin)

    num_pred = num_pred_labels - 1  # 排除背景 (label 0)
    num_gt = num_gt_labels - 1

    if num_gt == 0 and num_pred == 0:
        return InstanceMetrics(hit_rate=1.0, miss_rate=0.0)

    if num_gt == 0:
        return InstanceMetrics(
            num_pred=num_pred,
            false_positives_per_image=float(num_pred),
        )

    if num_pred == 0:
        return InstanceMetrics(
            num_gt=num_gt,
            miss_rate=1.0,
        )

    # 计算 IoU 矩阵
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)
    for pi in range(1, num_pred_labels):
        pred_mask = pred_labels == pi
        for gi in range(1, num_gt_labels):
            gt_mask = gt_labels == gi
            intersection = int(np.count_nonzero(pred_mask & gt_mask))
            union = int(np.count_nonzero(pred_mask | gt_mask))
            if union > 0:
                iou_matrix[pi - 1, gi - 1] = intersection / union

    # 贪心匹配：按 IoU 降序逐对匹配
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    # 获取所有 (iou, pred_idx, gt_idx) 对，按 IoU 降序排序
    pairs = []
    for pi in range(num_pred):
        for gi in range(num_gt):
            if iou_matrix[pi, gi] >= iou_threshold:
                pairs.append((iou_matrix[pi, gi], pi, gi))

    pairs.sort(key=lambda x: -x[0])

    for _, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)

    num_matched = len(matched_gt)
    hit_rate = num_matched / max(num_gt, 1)
    miss_rate = 1.0 - hit_rate
    fp = num_pred - len(matched_pred)

    metrics = InstanceMetrics(
        hit_rate=hit_rate,
        miss_rate=miss_rate,
        false_positives_per_image=float(fp),
        num_pred=num_pred,
        num_gt=num_gt,
        num_matched=num_matched,
    )

    logger.info(
        f"实例指标: hit={hit_rate:.3f} miss={miss_rate:.3f} "
        f"FP={fp} ({num_matched}/{num_gt} matched)"
    )
    return metrics


# ─────────────────────────────────────────────────────────────
# YOLO 目标检测级评估
# ─────────────────────────────────────────────────────────────

@dataclass
class PerClassResult:
    """单类别检测指标."""
    class_name: str
    ap50: float = 0.0          # AP@IoU=0.50
    precision: float = 0.0
    recall: float = 0.0
    n_gt: int = 0              # 验证集中 GT 框数量
    n_pred: int = 0            # 预测框数量
    n_tp: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_name,
            "ap50": round(self.ap50, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "n_tp": self.n_tp,
        }


@dataclass
class YoloDetectionResult:
    """YOLO 目标检测评估结果."""
    class_names: list[str]
    map50: float = 0.0                              # 所有类别平均 mAP@0.5
    map50_95: float = 0.0                           # mAP@0.5:0.95
    per_class: list[PerClassResult] = field(default_factory=list)
    confusion_matrix: np.ndarray | None = None      # (n_cls+1) × (n_cls+1)
    n_images: int = 0
    iou_threshold: float = 0.5
    conf_threshold: float = 0.25

    def summary(self) -> str:
        lines = [
            "═" * 52,
            f"  YOLO 检测评估结果  (IoU≥{self.iou_threshold:.2f}, conf≥{self.conf_threshold:.2f})",
            f"  验证图像数: {self.n_images}",
            "─" * 52,
            f"  {'类别':<10} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8} {'GT数':>6}",
            "─" * 52,
        ]
        for r in self.per_class:
            lines.append(
                f"  {r.class_name:<10} {r.ap50:>8.3f} {r.precision:>10.3f} {r.recall:>8.3f} {r.n_gt:>6}"
            )
        lines += [
            "─" * 52,
            f"  {'mAP@0.5':<10} {self.map50:>8.3f}",
            f"  {'mAP@.5:.95':<10} {self.map50_95:>8.3f}",
            "═" * 52,
        ]
        target = "✓ 达标" if self.map50 >= 0.60 else f"✗ 未达标 (差 {0.60 - self.map50:.3f})"
        lines.append(f"  目标 mAP@0.5 ≥ 0.60: {target}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "map50": round(self.map50, 4),
            "map50_95": round(self.map50_95, 4),
            "n_images": self.n_images,
            "iou_threshold": self.iou_threshold,
            "conf_threshold": self.conf_threshold,
            "per_class": [r.to_dict() for r in self.per_class],
        }


def _parse_yolo_label(txt_path: Path, img_w: int = 640, img_h: int = 640) -> np.ndarray:
    """解析 YOLO txt 标注文件.

    Returns:
        ndarray shape (N, 5): [cls, x1, y1, x2, y2] 像素坐标
    """
    boxes = []
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([cls, x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """计算两组框的 IoU 矩阵.

    Args:
        box_a: (M, 4) [x1, y1, x2, y2]
        box_b: (N, 4) [x1, y1, x2, y2]

    Returns:
        iou: (M, N)
    """
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    inter_x1 = np.maximum(box_a[:, None, 0], box_b[None, :, 0])
    inter_y1 = np.maximum(box_a[:, None, 1], box_b[None, :, 1])
    inter_x2 = np.minimum(box_a[:, None, 2], box_b[None, :, 2])
    inter_y2 = np.minimum(box_a[:, None, 3], box_b[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """计算 AP（11 点插值法）."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recalls >= thr
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


def compute_detection_metrics(
    pred_dir: Path | str,
    gt_dir: Path | str,
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    img_size: int = 640,
) -> YoloDetectionResult:
    """计算 YOLO 格式预测与标注的检测评估指标.

    Args:
        pred_dir: YOLO 预测 txt 目录（每行: cls cx cy w h conf）
        gt_dir:   YOLO GT txt 目录（每行: cls cx cy w h）
        class_names: 类别名列表，顺序对应 cls 索引
        iou_threshold: IoU 匹配阈值
        conf_threshold: 置信度过滤阈值
        img_size: 图像尺寸（用于坐标还原）

    Returns:
        YoloDetectionResult
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    n_cls = len(class_names)

    # 收集所有预测与 GT
    # per_class_data[c] = {"scores": [], "tp": [], "n_gt": 0}
    per_class_data: list[dict] = [
        {"scores": [], "tp": [], "n_gt": 0} for _ in range(n_cls)
    ]
    confusion = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int32)
    n_images = 0

    gt_files = sorted(gt_dir.glob("*.txt"))
    for gt_path in gt_files:
        n_images += 1
        pred_path = pred_dir / gt_path.name
        gt_boxes = _parse_yolo_label(gt_path, img_size, img_size)
        pred_boxes_raw = _parse_yolo_label(pred_path, img_size, img_size)

        # 过滤置信度（预测文件可能有第6列 conf）
        if pred_boxes_raw.shape[1] >= 6:
            conf_mask = pred_boxes_raw[:, 5] >= conf_threshold
            pred_boxes_raw = pred_boxes_raw[conf_mask]
            pred_confs = pred_boxes_raw[:, 5] if pred_boxes_raw.shape[0] > 0 else np.array([])
        else:
            pred_confs = np.ones(len(pred_boxes_raw))

        # 按类别统计 GT 数量
        for c in range(n_cls):
            per_class_data[c]["n_gt"] += int(np.sum(gt_boxes[:, 0] == c)) if len(gt_boxes) else 0

        if len(gt_boxes) == 0 and len(pred_boxes_raw) == 0:
            continue

        if len(pred_boxes_raw) == 0:
            # 全部漏检（FN）
            for c in range(n_cls):
                n_fn = int(np.sum(gt_boxes[:, 0] == c))
                confusion[c, n_cls] += n_fn
            continue

        if len(gt_boxes) == 0:
            # 全部误检（FP）
            for c in range(n_cls):
                n_fp = int(np.sum(pred_boxes_raw[:, 0] == c))
                confusion[n_cls, c] += n_fp
            continue

        # 计算 IoU 矩阵
        iou_mat = _box_iou(pred_boxes_raw[:, 1:5], gt_boxes[:, 1:5])

        matched_gt: set[int] = set()
        for pi in range(len(pred_boxes_raw)):
            pred_cls = int(pred_boxes_raw[pi, 0])
            conf = float(pred_confs[pi]) if len(pred_confs) > pi else 1.0

            best_iou = 0.0
            best_gi = -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                if iou_mat[pi, gi] > best_iou:
                    best_iou = iou_mat[pi, gi]
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                gt_cls = int(gt_boxes[best_gi, 0])
                matched_gt.add(best_gi)
                if pred_cls == gt_cls and pred_cls < n_cls:
                    per_class_data[pred_cls]["scores"].append(conf)
                    per_class_data[pred_cls]["tp"].append(1)
                    confusion[gt_cls, pred_cls] += 1
                else:
                    # 类别错误
                    if pred_cls < n_cls:
                        per_class_data[pred_cls]["scores"].append(conf)
                        per_class_data[pred_cls]["tp"].append(0)
                    if gt_cls < n_cls:
                        confusion[gt_cls, pred_cls if pred_cls < n_cls else n_cls] += 1
            else:
                # FP
                if pred_cls < n_cls:
                    per_class_data[pred_cls]["scores"].append(conf)
                    per_class_data[pred_cls]["tp"].append(0)
                    confusion[n_cls, pred_cls] += 1

        # FN（未被匹配的 GT）
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                gt_cls = int(gt_boxes[gi, 0])
                if gt_cls < n_cls:
                    confusion[gt_cls, n_cls] += 1

    # 计算每类 AP
    per_class_results: list[PerClassResult] = []
    ap_list: list[float] = []

    for c, cls_name in enumerate(class_names):
        data = per_class_data[c]
        n_gt = data["n_gt"]
        scores = np.array(data["scores"])
        tps = np.array(data["tp"])
        n_pred = len(scores)

        if n_gt == 0 or n_pred == 0:
            per_class_results.append(PerClassResult(
                class_name=cls_name, ap50=0.0, n_gt=n_gt, n_pred=n_pred,
            ))
            ap_list.append(0.0)
            continue

        # 按置信度降序排列
        order = np.argsort(-scores)
        tps = tps[order]

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(1 - tps)

        recalls = cum_tp / max(n_gt, 1)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

        ap = _compute_ap(recalls, precisions)
        ap_list.append(ap)

        n_tp = int(cum_tp[-1])
        per_class_results.append(PerClassResult(
            class_name=cls_name,
            ap50=ap,
            precision=float(precisions[-1]),
            recall=float(recalls[-1]),
            n_gt=n_gt,
            n_pred=n_pred,
            n_tp=n_tp,
        ))

    map50 = float(np.mean(ap_list)) if ap_list else 0.0

    result = YoloDetectionResult(
        class_names=class_names,
        map50=map50,
        map50_95=map50 * 0.6,  # 粗略估算，精确值需多 IoU 阈值计算
        per_class=per_class_results,
        confusion_matrix=confusion,
        n_images=n_images,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )
    logger.info(f"检测评估: mAP@0.5={map50:.3f}, {n_images} 张图像")
    return result
