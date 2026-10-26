"""trad_segment 合成图回归测试：交叉续接、断点桥接、斑点检测且斑点周围不出现环状伪划痕。"""

import cv2
import numpy as np

from darkfield_defects.annotation.trad_segment import CLASS_ID, segment


def _scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    img = np.zeros((400, 400), np.float32)
    cv2.line(img, (20, 60), (380, 340), 40, 3)        # A：与 B 交叉
    cv2.line(img, (20, 340), (380, 60), 30, 3)        # B
    cv2.line(img, (40, 120), (160, 120), 35, 3)       # C：中间断开 24 像素，与 A、B 各交叉一次
    cv2.line(img, (184, 120), (360, 120), 35, 3)
    cv2.circle(img, (330, 330), 4, 120, -1)           # 孤立亮斑
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 0.8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.ones(img.shape, bool)


def _instance_at(res, pts) -> set:
    return {int(res.inst[y, x]) for x, y in pts}


def test_crossing_gap_and_spot():
    img, valid = _scene()
    res = segment(img, valid, 0.8)
    # A、B 各自穿过交叉点保持为一个实例
    a = _instance_at(res, [(60, 91), (340, 309)])
    b = _instance_at(res, [(60, 309), (340, 91)])
    assert len(a) == 1 and 0 not in a
    assert len(b) == 1 and 0 not in b
    assert a != b
    # C 的两段经断点桥接成为同一实例
    c = _instance_at(res, [(60, 120), (340, 120)])
    assert len(c) == 1 and 0 not in c
    assert any(lk["type"] == "gap" for lk in res.links)
    # 亮斑为斑点，周围 6–14 像素环带内没有划痕像素
    assert res.cls[330, 330] == CLASS_ID["spot"]
    yy, xx = np.ogrid[:400, :400]
    ring = (np.hypot(yy - 330, xx - 330) > 6) & (np.hypot(yy - 330, xx - 330) < 14)
    assert not (res.cls[ring] == CLASS_ID["scratch"]).any()


def test_crossing_overlap_records_dimmer_line():
    img, valid = _scene()
    res = segment(img, valid, 0.8)
    a = _instance_at(res, [(60, 91)]).pop()    # 亮线 A（40）
    b = _instance_at(res, [(60, 309)]).pop()   # 暗线 B（30）
    # 交叉点附近：主所属为亮线，第二所属为暗线
    win = np.s_[195:206, 195:206]
    assert (res.inst[win] == a).any()
    assert (res.inst2[win] == b).any()


def test_arc_continues_through_crossing():
    rng = np.random.default_rng(1)
    img = np.zeros((400, 400), np.float32)
    cv2.ellipse(img, (200, 520), (420, 420), 0, 235, 305, 30, 3)    # 半径 420 的圆弧，顶点约在 y=100
    cv2.line(img, (200, 20), (200, 380), 45, 3)                     # 竖直亮线穿过圆弧
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 0.8, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    res = segment(img, np.ones(img.shape, bool), 0.8)
    yy, xx = np.nonzero(img > 15)
    left = [(x, y) for x, y in zip(xx, yy) if x < 150 and y < 200][:1]
    right = [(x, y) for x, y in zip(xx, yy) if x > 250 and y < 200][:1]
    ids = {int(res.inst[y, x]) for x, y in left + right}
    assert len(ids) == 1 and 0 not in ids
