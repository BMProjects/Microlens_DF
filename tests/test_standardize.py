"""预处理标准化测试：背景补偿到 0、保留真实缺陷、分离离焦晕斑。"""

from __future__ import annotations

import cv2
import numpy as np

from darkfield_defects.preprocessing.standardize import standardize


def _scene():
    h, w = 600, 800
    yy, xx = np.mgrid[:h, :w]
    roi = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < 280 ** 2
    ring = np.zeros_like(roi)
    template = np.full((h, w), 40.0, np.float32)
    rng = np.random.default_rng(0)
    img = template + 6.0 + rng.normal(0, 1.0, (h, w)).astype(np.float32)   # 残余偏置 6 + 噪声
    blob = 10.0 * np.exp(-(((yy - 300) ** 2 + (xx - 250) ** 2) / (2 * 35.0 ** 2)))  # 柔和离焦晕斑
    img += blob.astype(np.float32)
    cv2.line(img, (450, 150), (520, 450), 120.0, 3)                        # 细划痕
    cv2.rectangle(img, (300, 420), (380, 470), 200.0, -1)                  # 高亮、边缘锐利的宽缺陷
    return img, template, roi, ring


def test_background_to_zero_and_defects_kept():
    img, template, roi, ring = _scene()
    res = standardize(img, template, roi, ring)
    s = res.standard
    assert abs(res.stats["standard_bg_median"]) < 0.5
    assert s[300, 485] > 60            # 细划痕保留原始量级
    assert s[445, 340] > 120           # 宽而亮的真实缺陷保留
    assert res.diffuse[300, 250] > 5   # 晕斑进入弥散层
    assert s[300, 250] < 4             # 标准输入中晕斑基本去除
    np.testing.assert_allclose((s + res.diffuse)[roi].mean(),
                               (img - template - res.stats["bg_offset"])[roi].mean(), atol=0.5)
