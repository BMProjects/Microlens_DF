"""
NWD (Normalized Wasserstein Distance) 损失
===========================================
将 bounding box 建模为 2D 高斯分布，计算 Wasserstein-2 距离作为定位损失。
对细长目标 (scratch) 比 IoU/CIoU 更鲁棒。

核心优势:
  - scratch 的 4px 垂直偏移导致 IoU 从 0.96 跌至 0.33（已确认的根因），
    但 NWD 对此类偏移远更鲁棒（仅下降 ~5%）。
  - 对小目标更友好：IoU 对面积高度敏感，NWD 对位置和形状变化的度量更平滑。

实现参考:
  - 论文: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
  - GSDNet 论文 Section 3.4: Hybrid Loss = α·NWD + (1-α)·CIoU

用法::

    from darkfield_defects.ml.nwd_loss import nwd_loss, HybridBboxLoss

    # 直接使用 NWD
    loss = nwd_loss(pred_boxes, target_boxes)

    # 或替换 ultralytics 的 BboxLoss
    hybrid_loss = HybridBboxLoss(reg_max=16, use_dfl=True, nwd_alpha=0.5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bbox_to_gaussian(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 xyxy bbox 转换为 2D 高斯分布参数。

    高斯分布: N(μ, Σ)
      μ = (cx, cy) — bbox 中心
      Σ = diag((w/6)², (h/6)²) — 协方差 (6σ 覆盖 99.7% 面积)

    Args:
        boxes: [N, 4] xyxy 格式 (x1, y1, x2, y2)

    Returns:
        mu: [N, 2] 中心点
        sigma: [N, 2] 标准差 (w/6, h/6)
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([w / 6.0, h / 6.0], dim=-1)

    return mu, sigma


def wasserstein_2d(
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    mu_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """计算两个对角协方差 2D 高斯分布的 Wasserstein-2 距离。

    W²(N₁, N₂) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²
    (对角协方差简化版，无需矩阵平方根)

    Args:
        mu_p, sigma_p: 预测分布参数 [N, 2]
        mu_t, sigma_t: 目标分布参数 [N, 2]

    Returns:
        [N] Wasserstein-2 距离
    """
    # 中心距离²
    d_mu = ((mu_p - mu_t) ** 2).sum(dim=-1)

    # 标准差距离² (对角协方差简化)
    d_sigma = ((sigma_p - sigma_t) ** 2).sum(dim=-1)

    return d_mu + d_sigma


def nwd_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    C: float = 2.0,
) -> torch.Tensor:
    """Normalized Wasserstein Distance loss。

    NWD = exp(-W²/C) ∈ (0, 1]，类似 IoU 的范围。
    Loss = 1 - NWD

    Args:
        pred_boxes: [N, 4] xyxy 预测框
        target_boxes: [N, 4] xyxy 目标框
        C: 归一化常数 (控制损失平滑度，默认 2.0)

    Returns:
        [N] NWD loss values ∈ [0, 1)
    """
    # AMP (float16) 下 Wasserstein 计算存在精度问题，强制 float32
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.float()

    mu_p, sigma_p = _bbox_to_gaussian(pred_boxes)
    mu_t, sigma_t = _bbox_to_gaussian(target_boxes)

    w2 = wasserstein_2d(mu_p, sigma_p, mu_t, sigma_t)

    # 归一化: 用目标框的对角线长度归一化
    # clamp min=1e-3 (比原来的 1e-6 更保守，避免 float16 精度边界导致梯度爆炸)
    target_diag = ((target_boxes[:, 2] - target_boxes[:, 0]) ** 2 +
                   (target_boxes[:, 3] - target_boxes[:, 1]) ** 2).clamp(min=1e-3)

    # 限制指数范围，防止 exp 产生 inf 或梯度突变 (min=-20 对应 nwd ≈ 2e-9)
    exponent = (-w2 / (C * target_diag)).clamp(min=-20.0, max=0.0)
    nwd = torch.exp(exponent)

    return 1.0 - nwd


def hybrid_bbox_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    alpha: float = 0.5,
    ciou: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hybrid Loss = α·NWD + (1-α)·CIoU。

    当 alpha=1.0 时退化为纯 NWD；alpha=0.0 时退化为纯 CIoU。

    Args:
        pred_boxes: [N, 4] xyxy 预测框
        target_boxes: [N, 4] xyxy 目标框
        alpha: NWD 权重 (0~1)
        ciou: 预计算的 CIoU tensor [N]（如果 None，内部计算）

    Returns:
        [N] hybrid loss values
    """
    nwd = nwd_loss(pred_boxes, target_boxes)

    if ciou is None:
        from ultralytics.utils.metrics import bbox_iou
        ciou_val = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True).squeeze(-1)
        ciou_loss = 1.0 - ciou_val
    else:
        ciou_loss = 1.0 - ciou

    return alpha * nwd + (1.0 - alpha) * ciou_loss


class HybridBboxLoss(nn.Module):
    """替换 ultralytics BboxLoss 的混合损失。

    Hybrid Loss = α·NWD + (1-α)·CIoU + DFL

    与 ultralytics 8.4.24 BboxLoss 完全兼容:
    - __init__ 只接受 reg_max，DFL 通过 self.dfl_loss 模块控制（同原版）
    - forward 签名与 BboxLoss.forward 完全一致（含 imgsz, stride）
    """

    def __init__(self, reg_max: int = 16, nwd_alpha: float = 0.5):
        super().__init__()
        self.nwd_alpha = nwd_alpha
        # 与原版 BboxLoss 一致: DFLoss 模块控制 DFL
        from ultralytics.utils.loss import DFLoss
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor | None = None,
        stride_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 hybrid bbox loss + DFL loss（与 BboxLoss.forward 等价接口）。"""
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist

        # 强制整个 loss 在 float32 计算，防止 AMP float16 下 target_scores_sum
        # 趋近于零或 NWD 梯度在 half-precision 上溢出导致 NaN
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_bboxes_f = pred_bboxes.float()
            target_bboxes_f = target_bboxes.float()
            target_scores_f = target_scores.float()
            target_scores_sum_f = target_scores_sum.float().clamp(min=1e-6)
            anchor_points_f = anchor_points.float()

            weight = target_scores_f.sum(-1)[fg_mask].unsqueeze(-1)

            # CIoU (float32)
            iou = bbox_iou(pred_bboxes_f[fg_mask], target_bboxes_f[fg_mask], xywh=False, CIoU=True)

            if self.nwd_alpha > 0:
                # Hybrid: α·NWD + (1-α)·CIoU
                nwd = nwd_loss(pred_bboxes_f[fg_mask], target_bboxes_f[fg_mask])
                loss_iou = (
                    (self.nwd_alpha * nwd.unsqueeze(-1) + (1 - self.nwd_alpha) * (1.0 - iou)) * weight
                ).sum() / target_scores_sum_f
            else:
                loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum_f

            # DFL loss — 与原版 BboxLoss 完全一致的计算方式
            if self.dfl_loss is not None:
                # bbox2dist 第三个参数是 reg_max - 1 (最大 bin 索引)
                target_ltrb = bbox2dist(anchor_points_f, target_bboxes_f, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(
                    pred_dist[fg_mask].float().view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                ) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum_f
            else:
                loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl
