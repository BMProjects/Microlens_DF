"""语义分割通用指标（FCN 定义）：像素准确率 PA、平均像素准确率 MPA、mIoU、FWIoU，分类别 IoU/Dice。

定义（Long, Shelhamer, Darrell, FCN；n_ij 为基准类别 i、预测为 j 的像素数，t_i = Σ_j n_ij）：
    PA    = Σ_i n_ii / Σ_i t_i
    MPA   = (1/n_cl) Σ_i n_ii / t_i
    mIoU  = (1/n_cl) Σ_i n_ii / (t_i + Σ_j n_ji − n_ii)
    FWIoU = (Σ_k t_k)^−1 Σ_i t_i · n_ii / (t_i + Σ_j n_ji − n_ii)
标签 255 为“不计”，不参与统计。基准中不出现的类别在 MPA/mIoU 中按 NaN 跳过（同时报告各类像素数）。
混淆矩阵可跨图像累加，整个测试集统计一次（而不是逐图平均）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IGNORE = 255


@dataclass
class ConfusionMatrix:
    n_classes: int
    mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mat = np.zeros((self.n_classes, self.n_classes), np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        if gt.shape != pred.shape:
            raise ValueError(f"形状不一致: gt {gt.shape} vs pred {pred.shape}")
        keep = gt != IGNORE
        g = gt[keep].astype(np.int64)
        p = pred[keep].astype(np.int64)
        if g.size and (g.max() >= self.n_classes or p.max() >= self.n_classes or p.min() < 0):
            raise ValueError(f"类别超出范围 [0, {self.n_classes})")
        self.mat += np.bincount(g * self.n_classes + p, minlength=self.n_classes ** 2).reshape(
            self.n_classes, self.n_classes)

    def metrics(self, class_names: list[str] | None = None) -> dict:
        m = self.mat.astype(np.float64)
        tp = np.diag(m)
        t = m.sum(1)                     # 基准中各类像素数
        p = m.sum(0)                     # 预测中各类像素数
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(t > 0, tp / t, np.nan)
            iou = np.where(t > 0, tp / (t + p - tp), np.nan)
            dice = np.where(t > 0, 2 * tp / (t + p), np.nan)
        total = t.sum()
        names = class_names or [str(i) for i in range(self.n_classes)]
        return {
            "PA": float(tp.sum() / total) if total else float("nan"),
            "MPA": float(np.nanmean(acc)),
            "mIoU": float(np.nanmean(iou)),
            "FWIoU": float(np.nansum(t * np.nan_to_num(iou)) / total) if total else float("nan"),
            "mDice_defect": float(np.nanmean(dice[1:])) if self.n_classes > 1 else float("nan"),
            "per_class": {n: {"IoU": _r(iou[i]), "Dice": _r(dice[i]), "acc": _r(acc[i]), "gt_px": int(t[i]),
                              "pred_px": int(p[i])} for i, n in enumerate(names)},
            "pixels": int(total),
        }


def _r(v: float) -> float | None:
    return None if np.isnan(v) else round(float(v), 5)


def all_background_pa(cm: ConfusionMatrix) -> float:
    """全部预测为背景（类别 0）时的 PA，作为类别不平衡的参照。"""
    t = cm.mat.sum(1)
    return float(t[0] / t.sum()) if t.sum() else float("nan")
