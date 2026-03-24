"""pytest 共享 fixtures — 合成测试数据."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_background() -> np.ndarray:
    """合成均匀背景图 (128x128)."""
    bg = np.full((128, 128), 50, dtype=np.uint8)
    # 添加轻微渐变模拟暗场不均匀
    for r in range(128):
        for c in range(128):
            bg[r, c] = min(255, 50 + int(10 * np.sin(r / 40) + 5 * np.cos(c / 30)))
    return bg


@pytest.fixture
def synthetic_lens_image() -> np.ndarray:
    """合成含划痕的镜片图像 (128x128).

    - 圆形亮区域模拟镜片 ROI
    - 几条亮线模拟划痕
    """
    img = np.zeros((128, 128), dtype=np.uint8)

    # 镜片圆形区域
    cy, cx, r = 64, 64, 50
    for y in range(128):
        for x in range(128):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                img[y, x] = 30  # 暗场背景（镜片区域为暗）

    # 模拟划痕（亮线）
    # 水平划痕
    img[50, 30:90] = 200
    img[51, 30:90] = 180

    # 斜划痕
    for i in range(30):
        y, x = 40 + i, 35 + i
        if 0 <= y < 128 and 0 <= x < 128:
            img[y, x] = 190

    # 短划痕（中心区）
    img[64, 55:75] = 210

    return img


@pytest.fixture
def synthetic_clean_image() -> np.ndarray:
    """合成无划痕的干净镜片图像."""
    img = np.zeros((128, 128), dtype=np.uint8)
    cy, cx, r = 64, 64, 50
    for y in range(128):
        for x in range(128):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                img[y, x] = 30
    return img


@pytest.fixture
def binary_mask_pair():
    """一对预测/GT 二值掩码用于评估指标测试."""
    pred = np.zeros((64, 64), dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=np.uint8)

    # GT: 一条水平线
    gt[30, 10:50] = 255
    gt[31, 10:50] = 255

    # Pred: 部分重叠 + 少量多报
    pred[30, 15:55] = 255  # 部分交叉
    pred[31, 15:55] = 255

    return pred, gt
