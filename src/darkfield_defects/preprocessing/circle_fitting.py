"""鲁棒圆拟合 — Taubin 代数法 + RANSAC 迭代.

从不完整的左右圆弧点集拟合出完整圆的中心和半径。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darkfield_defects.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CircleGeometry:
    """拟合圆几何参数."""
    cx: float        # 圆心 x (列)
    cy: float        # 圆心 y (行)
    R: float         # 半径
    R_mad: float = 0.0   # 半径 MAD (多次拟合的离散度)
    cx_mad: float = 0.0
    cy_mad: float = 0.0
    n_inliers: int = 0


def taubin_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Taubin 代数圆拟合 — 比 Kåsa 法更稳定.

    Args:
        points: (N, 2) 数组, [x, y].

    Returns:
        (cx, cy, R).

    Raises:
        ValueError: 点数不足.
    """
    if len(points) < 3:
        raise ValueError(f"至少需要 3 个点, 实际 {len(points)}")

    x = points[:, 0]
    y = points[:, 1]

    # 中心化
    mx, my = np.mean(x), np.mean(y)
    u = x - mx
    v = y - my

    # 构建矩阵
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suv = np.sum(u * v)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)
    Suvv = np.sum(u * v ** 2)
    Svuu = np.sum(v * u ** 2)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([
        0.5 * (Suuu + Suvv),
        0.5 * (Svvv + Svuu),
    ])

    try:
        center_local = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 退化情况：用最小二乘
        center_local, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    uc, vc = center_local
    cx = uc + mx
    cy = vc + my
    R = np.sqrt(uc ** 2 + vc ** 2 + (Suu + Svv) / len(points))

    return float(cx), float(cy), float(R)


def ransac_circle_fit(
    points: np.ndarray,
    n_iter: int = 1000,
    inlier_thresh: float = 3.0,
    min_inlier_ratio: float = 0.3,
) -> tuple[float, float, float, np.ndarray]:
    """RANSAC 鲁棒圆拟合.

    Args:
        points: (N, 2) 数组, [x, y].
        n_iter: 最大迭代次数.
        inlier_thresh: 内点距离阈值 (像素).
        min_inlier_ratio: 最少内点比例.

    Returns:
        (cx, cy, R, inlier_mask).
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"至少需要 3 个点, 实际 {n}")

    best_cx, best_cy, best_R = 0.0, 0.0, 0.0
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # 随机选 3 个点
        idx = rng.choice(n, 3, replace=False)
        sample = points[idx]

        try:
            cx, cy, R = taubin_circle_fit(sample)
        except (ValueError, np.linalg.LinAlgError):
            continue

        # 合理性检查
        if R < 10 or R > max(points[:, 0].max(), points[:, 1].max()):
            continue

        # 计算所有点到拟合圆的距离
        dists = np.abs(np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) - R)
        inliers = dists < inlier_thresh
        count = np.count_nonzero(inliers)

        if count > best_count:
            best_count = count
            best_cx, best_cy, best_R = cx, cy, R
            best_inliers = inliers

    # 用所有内点重新拟合
    if best_count >= 3:
        inlier_points = points[best_inliers]
        try:
            best_cx, best_cy, best_R = taubin_circle_fit(inlier_points)
        except (ValueError, np.linalg.LinAlgError):
            pass

    inlier_ratio = best_count / n
    logger.info(
        f"RANSAC 圆拟合: cx={best_cx:.1f}, cy={best_cy:.1f}, R={best_R:.1f}, "
        f"inliers={best_count}/{n} ({inlier_ratio:.1%})"
    )

    if inlier_ratio < min_inlier_ratio:
        logger.warning(
            f"内点比例过低 ({inlier_ratio:.1%} < {min_inlier_ratio:.1%})，"
            f"拟合结果可能不可靠"
        )

    return best_cx, best_cy, best_R, best_inliers


def fit_circle_from_arcs(
    arc_points_list: list[np.ndarray],
    n_iter: int = 1000,
    inlier_thresh: float = 3.0,
) -> CircleGeometry:
    """从多张背景的圆弧点集拟合统一圆几何参数.

    分别对每张图做 RANSAC 拟合，然后取中位数作为最终结果，
    MAD 作为误差估计。

    Args:
        arc_points_list: 每张背景的弧点集列表.
        n_iter: RANSAC 迭代数.
        inlier_thresh: 内点阈值.

    Returns:
        CircleGeometry 包含中位数几何参数.
    """
    cxs, cys, Rs = [], [], []
    total_inliers = 0

    for i, pts in enumerate(arc_points_list):
        if len(pts) < 10:
            logger.warning(f"背景 {i}: 弧点太少 ({len(pts)}), 跳过")
            continue
        cx, cy, R, inlier_mask = ransac_circle_fit(pts, n_iter, inlier_thresh)
        cxs.append(cx)
        cys.append(cy)
        Rs.append(R)
        total_inliers += np.count_nonzero(inlier_mask)

    if not cxs:
        raise ValueError("没有有效的圆弧点集参与拟合")

    cxs_arr = np.array(cxs)
    cys_arr = np.array(cys)
    Rs_arr = np.array(Rs)

    def _mad(arr: np.ndarray) -> float:
        return float(np.median(np.abs(arr - np.median(arr))))

    circle = CircleGeometry(
        cx=float(np.median(cxs_arr)),
        cy=float(np.median(cys_arr)),
        R=float(np.median(Rs_arr)),
        R_mad=_mad(Rs_arr),
        cx_mad=_mad(cxs_arr),
        cy_mad=_mad(cys_arr),
        n_inliers=total_inliers,
    )

    logger.info(
        f"多背景圆拟合: cx={circle.cx:.1f}±{circle.cx_mad:.1f}, "
        f"cy={circle.cy:.1f}±{circle.cy_mad:.1f}, "
        f"R={circle.R:.1f}±{circle.R_mad:.1f} (从 {len(cxs)} 张背景)"
    )
    return circle
