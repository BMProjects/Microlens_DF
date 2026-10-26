"""高召回传统像素分割与候选片段划分（供智能助手做语义判定）。

流程：
1. 前景：在标准输入（背景≈0、原始计数）上，以本图噪声 σ 为单位做迟滞阈值（k_low·σ / k_high·σ），
   并与多尺度脊线响应（Sato）的迟滞结果取并集，尽量多地分割出目标。
2. 片段：前景骨架化，剪除短毛刺；骨架在交叉点处断开为分支，在每个交叉点按走向把最接近共线的
   分支两两配对，连成“笔画”；前景像素按最近骨架归属到笔画；紧凑的小连通域单独作为斑点候选。
每个片段有全镜唯一编号，供跨窗口一致引用。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize


@dataclass
class HRParams:
    k_low: float = 2.0          # 迟滞低阈值（σ 的倍数）
    k_high: float = 4.0         # 迟滞高阈值
    sigma_floor: float = 0.8    # σ 下限（8 位量化）
    ridge_low: float = 0.35     # Sato 响应归一化后的迟滞阈值
    ridge_high: float = 0.8
    min_area: int = 10          # 去除小于该像素数的连通域
    spur_len: int = 8           # 剪除短于该长度的端部毛刺
    pair_angle: float = 35.0    # 交叉点处分支配对的最大偏折角（度）
    spot_max_area: int = 300    # 斑点候选：面积上限
    spot_max_elong: float = 3.0  # 斑点候选：长宽比上限


def hr_foreground(img: np.ndarray, valid: np.ndarray, sigma: float, p: HRParams) -> np.ndarray:
    f = img.astype(np.float32)
    s = max(sigma, p.sigma_floor)
    fg = apply_hysteresis_threshold(f / s, p.k_low, p.k_high)
    # 脊线响应：在半分辨率上计算以提速，归一化到噪声尺度
    half = cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA)
    rid = sato(half, sigmas=(0.75, 1.5), black_ridges=False).astype(np.float32)
    rid = cv2.resize(rid, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR) / s
    fg |= apply_hysteresis_threshold(rid, p.ridge_low, p.ridge_high)
    fg &= valid
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= p.min_area
    return keep[lab]


_K3 = np.ones((3, 3), np.uint8)


def _neighbors(sk: np.ndarray) -> np.ndarray:
    return cv2.filter2D(sk.astype(np.uint8), -1, _K3, borderType=cv2.BORDER_CONSTANT) - sk.astype(np.uint8)


def _prune_spurs(sk: np.ndarray, spur_len: int) -> np.ndarray:
    sk = sk.copy()
    nb = _neighbors(sk)
    junction = sk & (nb >= 3)
    jd = cv2.dilate(junction.astype(np.uint8), _K3) > 0
    branches = sk & ~jd
    n, lab, st, _ = cv2.connectedComponentsWithStats(branches.astype(np.uint8), connectivity=8)
    endpoints = sk & (nb == 1)
    touch_j = cv2.dilate(jd.astype(np.uint8), _K3) > 0
    has_end = np.bincount(lab[endpoints], minlength=n) > 0
    touches = np.bincount(lab[touch_j & branches], minlength=n) > 0
    short = st[:, cv2.CC_STAT_AREA] < spur_len
    drop = has_end & touches & short
    drop[0] = False
    sk[drop[lab] & branches] = False
    return skeletonize(sk)


def _dir_from(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """分支在交叉点附近、指向远离交叉点方向的单位向量。"""
    d = np.hypot(points[:, 0] - origin[0], points[:, 1] - origin[1])
    near = points[np.argsort(d)[:12]]
    v = near.mean(axis=0) - origin
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([0.0, 0.0])


def segments(fg: np.ndarray, p: HRParams) -> tuple[np.ndarray, dict]:
    """返回 (片段编号图 int32, 片段信息 {id: {"kind": "stroke"|"spot", "len": 骨架像素数, "mid": (x, y)}})。"""
    h, w = fg.shape
    seg = np.zeros((h, w), np.int32)
    info: dict = {}
    # 斑点候选：紧凑小连通域
    n, lab, st, cen = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    spot_comp = np.zeros(n, bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a <= p.spot_max_area:
            ww, hh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if max(ww, hh) / max(min(ww, hh), 1) < p.spot_max_elong:
                spot_comp[i] = True
    spot_mask = spot_comp[lab]
    line_fg = fg & ~spot_mask
    sk = _prune_spurs(skeletonize(line_fg), p.spur_len)
    nb = _neighbors(sk)
    junction = sk & (nb >= 3)
    jd = cv2.dilate(junction.astype(np.uint8), _K3) > 0
    branches = sk & ~jd
    nbr, blab = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    parent = list(range(nbr))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # 交叉点处按走向配对
    nj, jlab = cv2.connectedComponents(jd.astype(np.uint8), connectivity=8)
    jmax = ndi.grey_dilation(jlab, size=3)
    sel = branches & (jmax > 0)
    pairs = np.unique(np.stack([jmax[sel], blab[sel]], 1), axis=0) if sel.any() else np.zeros((0, 2), int)
    touch: dict = {}
    for jl, b in pairs:
        touch.setdefault(int(jl), []).append(int(b))
    jobj = ndi.find_objects(jlab)
    R = 12
    for jl, bs in touch.items():
        bs = [b for b in bs if b > 0]
        if len(bs) < 2:
            continue
        sl = jobj[jl - 1]
        jy, jx = np.nonzero(jlab[sl] == jl)
        origin = np.array([jx.mean() + sl[1].start, jy.mean() + sl[0].start])
        oy, ox = int(origin[1]), int(origin[0])
        ya, xa = max(0, oy - R), max(0, ox - R)
        win = blab[ya:oy + R + 1, xa:ox + R + 1]
        dirs = {}
        for b in bs:
            yy, xx = np.nonzero(win == b)
            pts = np.stack([xx + xa, yy + ya], 1).astype(np.float32)
            dirs[b] = _dir_from(pts, origin) if len(pts) else np.array([0.0, 0.0])
        cands = []
        for a in bs:
            for b in bs:
                if a < b:
                    c = float(np.dot(dirs[a], -dirs[b]))
                    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
                    if ang <= p.pair_angle:
                        cands.append((ang, a, b))
        used = set()
        for ang, a, b in sorted(cands):
            if a in used or b in used:
                continue
            used |= {a, b}
            parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(nbr)])
    stroke_of_branch = roots
    skel_lab = np.where(branches, stroke_of_branch[blab], 0).astype(np.int32)
    # 交叉点像素归属到相邻任一笔画
    jfill = ndi.grey_dilation(skel_lab, size=3)
    skel_lab = np.where(jd & sk & (skel_lab == 0), jfill, skel_lab)
    # 前景像素按最近骨架归属
    if skel_lab.any():
        _, (iy, ix) = ndi.distance_transform_edt(skel_lab == 0, return_indices=True)
        near = skel_lab[iy, ix]
        seg = np.where(line_fg, near, 0).astype(np.int32)
    # 重新编号为连续 id（向量化），并记录信息
    uniq, inv = np.unique(seg, return_inverse=True)
    inv = inv.reshape(seg.shape)
    area = np.bincount(inv.ravel())
    keep = (uniq > 0) & (area >= p.min_area)
    newid = np.zeros(len(uniq), np.int32)
    newid[keep] = np.arange(1, int(keep.sum()) + 1)
    out = newid[inv]
    sk_new = np.where(skel_lab > 0, newid[np.searchsorted(uniq, skel_lab)], 0) if skel_lab.any() else skel_lab
    sk_new = np.where(np.isin(skel_lab, uniq), sk_new, 0)
    objs = ndi.find_objects(sk_new)
    lens_ = np.bincount(sk_new.ravel(), minlength=int(keep.sum()) + 1)
    seg_objs = ndi.find_objects(out)
    for nid in range(1, int(keep.sum()) + 1):
        sl = objs[nid - 1] if nid - 1 < len(objs) else None
        if sl is not None:
            yy, xx = np.nonzero(sk_new[sl] == nid)
            k = len(yy) // 2
            mid = (int(xx[k] + sl[1].start), int(yy[k] + sl[0].start))
        else:
            sl2 = seg_objs[nid - 1]
            yy, xx = np.nonzero(out[sl2] == nid)
            mid = (int(xx[0] + sl2[1].start), int(yy[0] + sl2[0].start))
        info[nid] = {"kind": "stroke", "len": int(lens_[nid]), "mid": mid}
    nid = int(keep.sum())
    spot_ids = np.zeros(n, np.int32)
    for i in np.nonzero(spot_comp)[0]:
        nid += 1
        spot_ids[i] = nid
        info[nid] = {"kind": "spot", "len": 0, "mid": (int(cen[i][0]), int(cen[i][1]))}
    out = np.where(spot_mask, spot_ids[lab], out).astype(np.int32)
    return out, info
