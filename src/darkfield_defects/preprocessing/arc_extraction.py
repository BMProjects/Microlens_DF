"""圆弧边缘提取 — 带通滤波 + 梯度 + 分位阈值 + 区域约束.

只保留图像左右两侧的竖向带状区域中的强边缘，
自动忽略中心划痕干扰和上下缺失部分。
"""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


def extract_arc_edges(
    image: np.ndarray,
    band_width_ratio: float = 0.15,
    highpass_sigma: float = 80.0,
    percentile: float = 97.0,
) -> np.ndarray:
    """从图像中提取左右圆弧的边缘二值图.

    Steps:
      1. 高通滤波: HP = I - GaussianBlur(I, σ_large)
      2. Scharr 梯度幅值
      3. 分位阈值取强边缘
      4. 只保留左右两侧竖向带状区域

    Args:
        image: 灰度图 (H, W), uint8.
        band_width_ratio: 左右带状区域宽度占图宽的比例 (0.1~0.2).
        highpass_sigma: 高通滤波高斯核标准差.
        percentile: 梯度分位数阈值 (97 = top 3%).

    Returns:
        E_arc: 二值边缘图 (H, W), bool.
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float64)

    # ① 高通 / 带通滤波
    low_freq = cv2.GaussianBlur(img_f, (0, 0), highpass_sigma)
    hp = img_f - low_freq

    # ② Scharr 梯度 (比 Sobel 更稳)
    gx = cv2.Scharr(hp, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(hp, cv2.CV_64F, 0, 1)
    gradient = np.sqrt(gx ** 2 + gy ** 2)

    # ③ 分位阈值：只保留 top (100 - percentile)% 的强边缘
    thresh = np.percentile(gradient, percentile)
    edge_mask = gradient > thresh

    # ④ 构建左右带状区域约束
    band_w = max(1, int(w * band_width_ratio))
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[:, :band_w] = True       # 左侧带
    region_mask[:, w - band_w:] = True   # 右侧带

    # 合并: E_arc = 强边缘 ∩ 左右带
    E_arc = edge_mask & region_mask

    n_pts = np.count_nonzero(E_arc)
    logger.info(
        f"圆弧边缘提取: {n_pts} 个边缘点, "
        f"band_w={band_w}, thresh={thresh:.1f}"
    )
    return E_arc


def extract_arc_points(E_arc: np.ndarray) -> np.ndarray:
    """从二值边缘图提取坐标点集.

    Args:
        E_arc: 二值边缘图 (H, W), bool.

    Returns:
        points: (N, 2) 数组, 每行 [x, y].
    """
    # np.argwhere 返回 [row, col], 转为 [x, y] = [col, row]
    yx = np.argwhere(E_arc)
    if len(yx) == 0:
        return np.empty((0, 2), dtype=np.float64)

    xy = yx[:, ::-1].astype(np.float64)  # [col, row] → [x, y]
    return xy


def split_left_right_arcs(
    E_arc: np.ndarray,
    image_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """将边缘点集分为左弧和右弧.

    Args:
        E_arc: 二值边缘图.
        image_width: 图像宽度（用于分左右）.

    Returns:
        (left_points, right_points): 各为 (N, 2) [x, y].
    """
    if image_width is None:
        image_width = E_arc.shape[1]

    yx = np.argwhere(E_arc)
    if len(yx) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    xy = yx[:, ::-1].astype(np.float64)
    mid = image_width / 2.0

    left_mask = xy[:, 0] < mid
    right_mask = ~left_mask

    return xy[left_mask], xy[right_mask]
