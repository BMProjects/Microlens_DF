"""传统像素级缺陷分割与划痕重连，用于生成可人工修订的基础标注。

输入为预处理标准输入（背景≈0、原始计数）与有效区域。流程：
1. 降噪去光晕：轻度高斯平滑 + 白顶帽去除宽于约 2·tophat_radius 的柔和光晕/阴影。
2. 分割与初步分类：
   - 划痕：多尺度方向线检测器（Nguyen 等 2013 的线检测思路，12 方向 × 2 长度），线上均值取两个
     半线段的较小者，抑制亮斑周围的环状伪响应；以响应噪声 σ 为单位迟滞阈值，同时得到局部走向；
   - 斑点：高斯差分峰值 + 各向同性检验 + 区域生长 + 长宽比检验；
   - 碎裂：按标注定义，“亮而无法逐条追踪”的像素占比高且方向杂乱（方向熵高）的区域。
3. 划痕重连：骨架在交叉点断开为分支（相邻三叉点之间的短段并入交叉区）；沿中心线采样截面灰度剖面
   （峰值幅度、半高宽、剖面形状）；止于交叉区的分支端按“走向连续性 × 截面相似性”配对穿越
   （借鉴方向分数/多方向追踪处理交叉的思想），自由断点按“距离、共线性、间隙脊线证据、截面相似性”
   双向最优配对桥接（借鉴最小路径/感知补全的间隙连接思想）。
4. 实例与置信：输出实例图、类别图、每个实例的几何属性、连接记录与复核标记。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

CLASS_ID = {"scratch": 1, "spot": 2, "damage": 3, "uncertain": 4}
K3 = np.ones((3, 3), np.uint8)


@dataclass
class TradParams:
    smooth_sigma: float = 0.8
    tophat_radius: int = 12          # 去除宽于约 24 像素的光晕/阴影
    n_orient: int = 12
    line_lengths: tuple = (15, 31)
    line_low: float = 1.5            # 线响应迟滞阈值（响应噪声 σ 的倍数）
    line_high: float = 3.0
    line_noise_min: float = 0.25     # 线响应噪声相对图像噪声 σ 的下、上限
    line_noise_max: float = 0.6
    grow_k: float = 2.0              # 受限生长的亮度阈值（图像噪声 σ 倍数）
    grow_radius: int = 5
    width_margin: float = 1.0        # 划痕掩膜半宽 = 半高宽/2 + width_margin（像素）
    width_k: float = 3.0             # 划痕宽度扩展：线掩膜 1 像素邻域内且亮于 width_k·σ 的像素
    spot_peak_k: float = 6.0         # 斑点峰值（DoG 噪声 σ 倍数）
    spot_min_amp_k: float = 5.0      # 斑点峰值最低亮度（图像噪声 σ 倍数）
    spot_max_area: int = 1500
    spot_grow_frac: float = 0.3      # 斑点区域生长到峰值的该比例（含光晕）
    spot_guard_px: int = 7           # 斑点周围该范围内抑制线响应，避免光晕被当作划痕
    spot_max_elong: float = 4.0      # 允许略长的点状亮斑（如 8×20 像素）；短划痕碎片一般 > 6
    spot_max_aniso: float = 0.5      # 峰值处 12 个方向 ±spot_probe 像素均值的 (max-min)/max；线与交叉点接近 1
    spot_probe: int = 5
    spur_len: int = 8
    edge_guard_px: int = 8           # 有效区边界带宽度（不检测划痕）
    damage_win: int = 64
    damage_fill: float = 0.35        # 64 像素窗口内“亮而无法追踪”像素的占比（72l 中 R2 碎裂区中位数约 0.33）
    damage_entropy: float = 0.85     # 窗口内线方向熵（12 个方向归一化）
    damage_min_area: int = 64 * 64 * 2
    damage_keep_len: int = 40        # 密集区内骨架短于该值的实例归碎裂，更长的保留为划痕并标记复核
    junction_max_dev: float = 35.0   # 交叉点穿越的最大偏折角
    junction_merge_len: int = 15     # 两端都接交叉点、短于该值的分支并入交叉区
    junction_radius: float = 30.0    # 交叉穿越时两个分支端点的最大距离
    junction_min_score: float = 0.35
    gap_max_dist: float = 50.0       # 断点桥接最大距离（像素，约 0.34 mm）
    gap_max_dev: float = 30.0
    gap_min_score: float = 0.2
    gap_need_evidence: float = 30.0  # 超过该距离的断点只在间隙上有脊线证据时桥接
    min_scratch_len: int = 25
    min_elong: float = 5.0
    noise_amp_k: float = 2.5         # 幅度低于该倍数且很短的实例视为噪声
    sigma_floor: float = 0.8
    # 迭代对照开关（默认即最终方案；scripts/annot_ablation.py 用它们逐步复现改进过程）
    line_two_sided: bool = True      # False：经典线检测（整条线均值）
    line_noise_mode: str = "bg_bounded"   # "mad_floor"：全图 MAD，下限 σ/4（初版）
    spot_isotropy: bool = True       # False：不做各向同性与孤立性检验（初版）
    junction_mode: str = "ends2"     # "ends"：端点配对 + 全分支剖面（v1）；"cluster"：逐个交叉区配对（初版）
    damage_mode: str = "footprint"   # "untraced"：无法追踪亮像素占比 + 方向熵（v1）；"legacy"：初版
    damage_avg_len: float = 20.0     # 仅 legacy 碎裂判定使用
    # 第二轮改进（v2）
    faint_lengths: tuple = (61,)     # 长尺度弱划痕检测核长度；() 关闭
    faint_low: float = 2.0           # 长尺度线响应迟滞阈值（其噪声 σ 的倍数）
    faint_high: float = 4.0
    field_sigma: float = 32.0        # 划痕走向场的平滑尺度（像素）
    grey_confirm: bool = True        # 待确认片段按局部走向场确认为浅密划痕
    grey_min_elong: float = 2.0
    grey_min_coh: float = 0.5        # 走向场相干度（0 无序 – 1 完全平行）
    grey_max_dev: float = 20.0       # 片段走向与局部走向场的最大夹角（度）
    grey_line_ratio: float = 4.0     # 轴比不低于该值、长度 ≥ grey_line_len、亮度 ≥ 3σ 的片段本身即为线状
    grey_line_len: int = 15
    end_len: int = 60                # 端部局部剖面、曲率的取样长度（像素）
    curv_min: float = 0.004          # 曲率噪声下限（1/像素）：60 像素端段 ±1 像素抖动约对应 0.002–0.004
    curv_scale: float = 0.01         # 曲率相似性尺度（1/像素，对应半径 100 像素）
    crossing_overlap: bool = True    # 重建交叉区二元叠加所属（亮线为主、暗线为次）
    overlap_min_angle: float = 15.0  # 两线夹角小于该值（近乎平行）时不记叠加
    junction_match: str = "optimal"  # 交叉区配对方式："optimal" 平方加权的整体最优；"greedy" 依次成全最高分配对
    pixel_mm: float = 0.0068
    footprint_mm: float = 3.0        # 光束足迹直径 ≈ 明视瞳孔直径
    damage_f_min: float = 0.35       # 足迹内缺陷像素占比下限（损伤区）；随检测灵敏度变化，需与雾度测量一起标定
    damage_traceable_max: float = 0.5   # 区内可追踪划痕像素占比 ≥ 该值为“密集划痕型”，否则“碎裂型”
    damage_trace_len_mm: float = 0.5    # “可追踪”的最短划痕长度（密集区划痕常被交叉切成 0.5–1 mm 的段）
    damage_lens_frac: float = 0.10   # 碎裂区占有效区面积比例下限（大面积碎裂）
    central_zone_mm: float = 9.0     # 中央清晰区直径（用于分区统计）


@dataclass
class TradResult:
    cls: np.ndarray                  # uint8 类别图（0 背景，1 划痕，2 斑点，3 碎裂，4 待确认）
    inst: np.ndarray                 # int32 实例图
    instances: list = field(default_factory=list)
    links: list = field(default_factory=list)      # 交叉穿越与断点桥接记录
    maps: dict = field(default_factory=dict)       # 中间结果（调试/可视化用）
    stats: dict = field(default_factory=dict)
    inst2: np.ndarray | None = None  # 交叉区的第二所属实例（0 表示无叠加）
    regions: list = field(default_factory=list)    # 碎裂区描述


# ── 1. 降噪去光晕 ─────────────────────────────────────────────────────

def flatten(img: np.ndarray, p: TradParams) -> tuple[np.ndarray, np.ndarray]:
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), p.smooth_sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.tophat_radius + 1,) * 2)
    return f, f - cv2.morphologyEx(f, cv2.MORPH_OPEN, k)


def robust_sigma(v: np.ndarray) -> float:
    low = v[v <= np.median(v)]
    return float(1.4826 * np.median(np.abs(low - np.median(low)))) if low.size else 1.0


# ── 2. 响应与掩膜 ─────────────────────────────────────────────────────

def line_response(x: np.ndarray, p: TradParams) -> tuple[np.ndarray, np.ndarray]:
    """多尺度方向线检测：线上均值减窗口均值，取各方向最大值与对应方向。

    线上均值取两个半线段均值中的较小者（线必须同时穿过像素两侧），
    否则亮斑会在半径约 L/2 的环带上产生虚假线响应，划痕端点也会向外延伸 L/2。
    """
    best = np.full(x.shape, -np.inf, np.float32)
    arg = np.zeros(x.shape, np.uint8)
    for k in range(p.n_orient):
        th = math.pi * k / p.n_orient
        rk = np.zeros(x.shape, np.float32)
        for L in p.line_lengths:
            c = L // 2
            dx, dy = math.cos(th) * c, math.sin(th) * c
            if p.line_two_sided:
                half = []
                for sgn in (-1, 1):
                    ker = np.zeros((L, L), np.float32)
                    cv2.line(ker, (c, c), (int(round(c + sgn * dx)), int(round(c + sgn * dy))), 1.0, 1, cv2.LINE_AA)
                    ker /= ker.sum()
                    half.append(cv2.filter2D(x, -1, ker, borderType=cv2.BORDER_REFLECT))
                on_line = np.minimum(half[0], half[1])
            else:
                ker = np.zeros((L, L), np.float32)
                cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))),
                         1.0, 1, cv2.LINE_AA)
                ker /= ker.sum()
                on_line = cv2.filter2D(x, -1, ker, borderType=cv2.BORDER_REFLECT)
            rk += on_line - cv2.blur(x, (L, L))
        rk /= len(p.line_lengths)
        upd = rk > best
        best[upd] = rk[upd]
        arg[upd] = k
    return best, arg


def prune_spurs(sk: np.ndarray, spur_len: int) -> np.ndarray:
    sk = sk.copy()
    for _ in range(2):
        nb = cv2.filter2D(sk.astype(np.uint8), -1, K3, borderType=cv2.BORDER_CONSTANT) - sk
        junction = sk & (nb >= 3)
        jd = cv2.dilate(junction.astype(np.uint8), K3) > 0
        br = sk & ~jd
        n, lab, st, _ = cv2.connectedComponentsWithStats(br.astype(np.uint8), connectivity=8)
        endpoints = sk & (nb == 1)
        touch = cv2.dilate(jd.astype(np.uint8), K3) > 0
        has_end = np.bincount(lab[endpoints], minlength=n) > 0
        touches = np.bincount(lab[touch & br], minlength=n) > 0
        drop = has_end & touches & (st[:, cv2.CC_STAT_AREA] < spur_len)
        drop[0] = False
        sk[drop[lab] & br] = False
        sk = skeletonize(sk)
    return sk


def detect_spots(fl: np.ndarray, valid: np.ndarray, sig_i: float, p: TradParams) -> np.ndarray:
    dog = cv2.GaussianBlur(fl, (0, 0), 1.5) - cv2.GaussianBlur(fl, (0, 0), 4.0)
    sd = max(robust_sigma(dog[valid]), 1e-3)
    peaks = (dog == ndi.maximum_filter(dog, size=7)) & (dog > p.spot_peak_k * sd) & (fl > p.spot_min_amp_k * sig_i) & valid
    ys, xs = np.nonzero(peaks)
    # 各向同性检验：沿 12 个方向取 ±probe 像素的平均亮度，斑点各方向相近，划痕与交叉点在某些方向显著变暗
    th = np.pi * np.arange(p.n_orient) / p.n_orient
    t = np.arange(-p.spot_probe, p.spot_probe + 1)
    oy, ox = np.sin(th)[:, None] * t, np.cos(th)[:, None] * t
    samp = ndi.map_coordinates(fl, [(ys[:, None, None] + oy).ravel(), (xs[:, None, None] + ox).ravel()],
                               order=1, mode="nearest").reshape(len(ys), p.n_orient, len(t)).mean(2)
    aniso = (samp.max(1) - samp.min(1)) / np.maximum(samp.max(1), 1e-6)
    keep = aniso < p.spot_max_aniso if p.spot_isotropy else np.ones(len(ys), bool)
    ys, xs = ys[keep], xs[keep]
    mask = np.zeros(fl.shape, bool)
    for y, x in zip(ys, xs):
        r = 20
        y0, x0 = max(0, y - r), max(0, x - r)
        win = fl[y0:y + r + 1, x0:x + r + 1]
        thr = max(3 * sig_i, p.spot_grow_frac * float(fl[y, x]))
        lab, _ = ndi.label(win > thr)
        li = lab[y - y0, x - x0]
        if li == 0:
            continue
        comp = lab == li
        if p.spot_isotropy and (comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()):
            continue  # 生长到窗口边界：与划痕或大片亮区相连，不是孤立斑点
        a = int(comp.sum())
        if a > p.spot_max_area or a < 4:
            continue
        yy, xx = np.nonzero(comp)
        if a >= 6:
            cov = np.cov(np.stack([xx, yy]))
            ev = np.linalg.eigvalsh(cov)
            if math.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)) >= p.spot_max_elong:
                continue
        mask[y0:y + r + 1, x0:x + r + 1] |= comp
    return mask


def damage_regions(untraced: np.ndarray, line_mask: np.ndarray, arg: np.ndarray, valid: np.ndarray,
                   p: TradParams) -> np.ndarray:
    """碎裂区（按标注定义）：无法逐条追踪的亮像素占比高、且线方向杂乱（方向熵高）的区域。在 1/4 分辨率上计算。

    untraced 为亮于 3σ 但未被追踪为划痕或斑点的像素。五图诊断中，平均碎片长度、长短核响应比、
    片状亮区占比等特征都无法区分 R2 圈出的碎裂区与相邻的密集划痕区（三名评审员对碎裂本身也不一致），
    因此结果偏保守，全部进入人工复核。
    """
    h, w = line_mask.shape
    sh, sw = h // 4, w // 4
    win = max(3, p.damage_win // 4)
    area = lambda m: cv2.resize(m.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)  # noqa: E731
    fill = cv2.blur(area(untraced), (win, win))
    hist = []
    for k in range(p.n_orient):
        hist.append(cv2.blur(area(line_mask & (arg == k)), (win, win)))
    hist = np.stack(hist)
    tot = hist.sum(0) + 1e-6
    q = hist / tot
    ent = -(q * np.log(q + 1e-9)).sum(0) / math.log(p.n_orient)
    dmg = (fill > p.damage_fill) & (ent > p.damage_entropy)
    dmg = cv2.morphologyEx(dmg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((win, win), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dmg, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] * 16 >= p.damage_min_area
    dmg = keep[lab].astype(np.uint8)
    return (cv2.resize(dmg, (w, h), interpolation=cv2.INTER_NEAREST) > 0) & valid


def damage_regions_legacy(line_mask: np.ndarray, arg: np.ndarray, fg: np.ndarray, sk: np.ndarray,
                          endpoints: np.ndarray, valid: np.ndarray, p: TradParams) -> np.ndarray:
    """初版碎裂判定（仅作迭代对照）：前景填充率高、方向熵高、平均碎片短的窗口。"""
    h, w = fg.shape
    sh, sw = h // 4, w // 4
    win = max(3, p.damage_win // 4)
    area = lambda m: cv2.resize(m.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)  # noqa: E731
    fill = cv2.blur(area(fg), (win, win))
    hist = np.stack([cv2.blur(area(line_mask & (arg == k)), (win, win)) for k in range(p.n_orient)])
    q = hist / (hist.sum(0) + 1e-6)
    ent = -(q * np.log(q + 1e-9)).sum(0) / math.log(p.n_orient)
    avg_len = cv2.blur(area(sk), (win, win)) * 16 / np.maximum(cv2.blur(area(endpoints), (win, win)) * 8, 1e-3)
    dmg = ((fill > p.damage_fill) & (ent > p.damage_entropy) & (avg_len < p.damage_avg_len)).astype(np.uint8)
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_CLOSE, np.ones((win, win), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dmg, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] * 16 >= p.damage_min_area
    return (cv2.resize(keep[lab].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0) & valid


# ── 3. 骨架图与重连 ───────────────────────────────────────────────────

def order_branch(coords: np.ndarray) -> np.ndarray:
    """把一条无分叉骨架分支的像素按走向排序（从端点出发逐邻接步进）。"""
    if len(coords) <= 2:
        return coords
    s = {(int(y), int(x)) for y, x in coords}
    nbrs = lambda q: [(q[0] + dy, q[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)  # noqa: E731
                      if (dy or dx) and (q[0] + dy, q[1] + dx) in s]
    start = next((q for q in s if len(nbrs(q)) == 1), next(iter(s)))
    out, seen, cur = [start], {start}, start
    while True:
        nxt = [q for q in nbrs(cur) if q not in seen]
        if not nxt:
            break
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))  # 优先 4 邻接
        cur = nxt[0]
        seen.add(cur)
        out.append(cur)
    return np.array(out)


def end_dir(chain: np.ndarray, at_start: bool, n: int = 15, skip: int = 3) -> np.ndarray:
    """分支端部走向（指向分支外侧）。跳过紧邻端点的几个像素，避开骨架在交叉点附近的弯折。"""
    m = min(n, len(chain) - 1)
    if m < 1:
        return np.array([0.0, 0.0])
    s = min(skip, m - 1) if m > skip + 4 else 0
    a, b = (chain[m], chain[s]) if at_start else (chain[-1 - m], chain[-1 - s])
    v = np.array([b[1] - a[1], b[0] - a[0]], float)        # (x, y)，指向分支外侧
    nv = np.hypot(*v)
    return v / nv if nv > 0 else v


def profiles(chains: list[np.ndarray], img: np.ndarray, half: int = 7, step: int = 3) -> list[dict]:
    """沿中心线每 step 像素取法向截面剖面，返回每条分支的中位剖面、峰值与半高宽。"""
    xs_all, ys_all, owner = [], [], []
    offs = np.arange(-half, half + 1, dtype=np.float32)
    for bi, ch in enumerate(chains):
        if len(ch) < 3:
            continue
        idx = np.arange(0, len(ch), step)
        pts = ch[idx][:, ::-1].astype(np.float32)          # (x, y)
        a = ch[np.clip(idx - 2, 0, len(ch) - 1)][:, ::-1]
        b = ch[np.clip(idx + 2, 0, len(ch) - 1)][:, ::-1]
        t = (b - a).astype(np.float32)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-6)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        xs_all.append(pts[:, 0:1] + nrm[:, 0:1] * offs)
        ys_all.append(pts[:, 1:2] + nrm[:, 1:2] * offs)
        owner += [bi] * len(idx)
    out = [{"amp": 0.0, "width": 0.0, "shape": np.zeros(2 * half + 1, np.float32)} for _ in chains]
    if not xs_all:
        return out
    X = np.vstack(xs_all).astype(np.float32)
    Y = np.vstack(ys_all).astype(np.float32)
    S = ndi.map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="reflect").reshape(X.shape)
    owner = np.array(owner)
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(len(chains) + 1))
    for bi in range(len(chains)):
        rows = order[bounds[bi]:bounds[bi + 1]]
        if len(rows) == 0:
            continue
        prof = np.median(S[rows], axis=0)
        base = float(min(prof[0], prof[-1]))
        pk = float(prof[half - 1:half + 2].max())
        amp = max(pk - base, 0.0)
        width = float(((prof - base) >= 0.5 * amp).sum()) if amp > 0 else 0.0
        shape = (prof - base) / amp if amp > 0 else np.zeros_like(prof)
        out[bi] = {"amp": amp, "width": width, "shape": shape.astype(np.float32)}
    return out


def profile_sim(a: dict, b: dict) -> float:
    if a["amp"] <= 0 or b["amp"] <= 0:
        return 0.0
    ra = min(a["amp"], b["amp"]) / max(a["amp"], b["amp"])
    rw = min(a["width"], b["width"]) / max(a["width"], b["width"], 1e-6)
    c = float(np.corrcoef(a["shape"], b["shape"])[0, 1]) if a["shape"].std() > 0 and b["shape"].std() > 0 else 0.0
    return float(math.sqrt(ra) * math.sqrt(max(rw, 0)) * max(c, 0.0))


def dev_deg(u: np.ndarray, v: np.ndarray) -> float:
    """两段在连接处的偏折角：u、v 为各自指向外侧的方向，笔直穿越时 u ≈ −v。"""
    return math.degrees(math.acos(float(np.clip(-np.dot(u, v), -1, 1))))


def pair_by_cluster(jd: np.ndarray, branches: np.ndarray, blab: np.ndarray, chains: list, prof: list,
                    uf: "UF", p: TradParams) -> list[dict]:
    """初版交叉续接（仅作迭代对照）：逐个交叉区，在接触该区的分支之间按走向 × 剖面相似配对。"""
    links = []
    nj, jlab = cv2.connectedComponents(jd.astype(np.uint8), connectivity=8)
    jmax = ndi.grey_dilation(jlab, size=3)
    sel = branches & (jmax > 0)
    touch: dict = {}
    if sel.any():
        for jl, b in np.unique(np.stack([jmax[sel], blab[sel]], 1), axis=0):
            touch.setdefault(int(jl), []).append(int(b))
    jobj = ndi.find_objects(jlab)
    for jl, bs in touch.items():
        bs = sorted({b for b in bs if b > 0})
        if len(bs) < 2:
            continue
        sl = jobj[jl - 1]
        jy, jx = np.nonzero(jlab[sl] == jl)
        o = np.array([jy.mean() + sl[0].start, jx.mean() + sl[1].start])
        ends = {}
        for b in bs:
            ch = chains[b]
            if len(ch) >= 2:
                at_start = np.hypot(*(ch[0] - o)) <= np.hypot(*(ch[-1] - o))
                ends[b] = end_dir(ch, at_start, n=10, skip=0)
        cands = []
        for a_, b_ in itertools.combinations(sorted(ends), 2):
            d = dev_deg(ends[a_], ends[b_])
            if d > p.junction_max_dev:
                continue
            ps = profile_sim(prof[a_], prof[b_])
            s = math.exp(-(d / 25.0) ** 2) * (0.4 + 0.6 * ps)
            if s >= p.junction_min_score:
                cands.append((s, a_, b_, d, ps))
        used = set()
        for s, a_, b_, d, ps in sorted(cands, reverse=True):
            if a_ in used or b_ in used:
                continue
            used |= {a_, b_}
            uf.union(a_, b_)
            links.append({"type": "crossing", "a": a_, "b": b_, "score": round(s, 3), "dev": round(d, 1),
                          "profile_sim": round(ps, 3), "pos": [int(o[1]), int(o[0])]})
    return links


def orientation_field(line_mask: np.ndarray, arg: np.ndarray, p: TradParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1/4 分辨率的划痕走向场：倍角平均得到局部主走向 θ（弧度，自 +x 轴转向 +y 轴）、相干度与线密度。"""
    h, w = line_mask.shape
    s = (w // 4, h // 4)
    th = np.pi * arg.astype(np.float32) / p.n_orient
    m = line_mask.astype(np.float32)
    c2 = cv2.resize(np.cos(2 * th) * m, s, interpolation=cv2.INTER_AREA)
    s2 = cv2.resize(np.sin(2 * th) * m, s, interpolation=cv2.INTER_AREA)
    d = cv2.resize(m, s, interpolation=cv2.INTER_AREA)
    g = p.field_sigma / 4
    c2, s2, d = (cv2.GaussianBlur(x, (0, 0), g) for x in (c2, s2, d))
    coh = np.hypot(c2, s2) / np.maximum(d, 1e-6)
    return 0.5 * np.arctan2(s2, c2), np.clip(coh, 0, 1), d


def pca_orient(yy: np.ndarray, xx: np.ndarray) -> tuple[float, float]:
    """像素集合的主轴走向（弧度）与长短轴比。"""
    if len(yy) < 3:
        return 0.0, 1.0
    cov = np.cov(np.stack([xx, yy]).astype(np.float64))
    ev, vec = np.linalg.eigh(cov)
    v = vec[:, 1]
    return float(math.atan2(v[1], v[0])), float(math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-2)))


def ang_diff(a: float, b: float) -> float:
    """两个无向走向的夹角（度，0–90）。"""
    d = abs((a - b) % math.pi)
    return math.degrees(min(d, math.pi - d))


def end_curvature(chain: np.ndarray, at_start: bool, n: int, skip: int = 6, floor: float = 0.0) -> float:
    """端部曲率（弧度/像素），沿“向端点外行进”方向计。跳过紧邻端点（交叉点处骨架弯折）的 skip 像素；
    取样不足 20 像素或 |κ| 低于噪声下限 floor 时按直线返回 0。"""
    pts = chain[:n] if at_start else chain[::-1][:n]
    pts = pts[skip:]
    if len(pts) < 20:
        return 0.0
    q = pts[::-1][::3].astype(np.float64)               # 由内向端点
    dv = np.diff(q, axis=0)
    ang = np.unwrap(np.arctan2(dv[:, 0], dv[:, 1]))     # (y, x) → atan2(dy, dx)
    sl = np.cumsum(np.hypot(dv[:, 0], dv[:, 1]))
    if len(ang) < 3 or sl[-1] - sl[0] < 1:
        return 0.0
    k = float(np.polyfit(sl, ang, 1)[0])
    return k if abs(k) >= floor else 0.0


def rotate(u: np.ndarray, phi: float) -> np.ndarray:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([u[0] * c - u[1] * s, u[0] * s + u[1] * c])


def max_weight_matching(edges: list[tuple[float, int, int]], exact_max: int = 14) -> list[tuple[int, int]]:
    """候选图的最大权匹配：按连通分量求解，节点不多于 exact_max 时状态压缩求精确解，否则贪心。"""
    adj: dict = {}
    for s, i, j in edges:
        adj.setdefault(i, {})[j] = s
        adj.setdefault(j, {})[i] = s
    seen, out = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) > exact_max:
            used = set()
            for s, i, j in sorted(((adj[i][j], i, j) for i in comp for j in adj[i] if i < j), reverse=True):
                if i not in used and j not in used:
                    used |= {i, j}
                    out.append((i, j))
            continue
        idx = {x: k for k, x in enumerate(comp)}
        memo: dict = {}

        def best(mask: int) -> tuple[float, tuple]:
            if mask == 0:
                return 0.0, ()
            if mask in memo:
                return memo[mask]
            k = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << k)
            res = best(rest)
            x = comp[k]
            for y, s in adj[x].items():
                ky = idx[y]
                if rest >> ky & 1:
                    v, pr = best(rest & ~(1 << ky))
                    if v + s > res[0]:
                        res = (v + s, pr + ((x, y),))
            memo[mask] = res
            return res
        out += list(best((1 << len(comp)) - 1)[1])
    return out


def footprint_damage(defect: np.ndarray, valid: np.ndarray, p: TradParams) -> tuple[np.ndarray, np.ndarray]:
    """足迹内缺陷像素占比 f（1/8 分辨率计算后放大）与碎裂区掩膜（f ≥ damage_f_min，面积 ≥ 一个足迹）。"""
    h, w = defect.shape
    s = (w // 8, h // 8)
    r = max(1, int(round(p.footprint_mm / 2 / p.pixel_mm / 8)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2).astype(np.float32)
    d = cv2.resize((defect & valid).astype(np.float32), s, interpolation=cv2.INTER_AREA)
    v = cv2.resize(valid.astype(np.float32), s, interpolation=cv2.INTER_AREA)
    f = cv2.filter2D(d, -1, k, borderType=cv2.BORDER_CONSTANT) / np.maximum(cv2.filter2D(v, -1, k, borderType=cv2.BORDER_CONSTANT), 1e-6)
    m = ((f >= p.damage_f_min) & (v > 0.5)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] * 64 >= math.pi * (p.footprint_mm / 2 / p.pixel_mm) ** 2
    m = keep[lab].astype(np.uint8)
    up = lambda x, interp: cv2.resize(x, (w, h), interpolation=interp)  # noqa: E731
    return up(f, cv2.INTER_LINEAR), (up(m, cv2.INTER_NEAREST) > 0) & valid


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def segment(img: np.ndarray, valid: np.ndarray, sigma_img: float, p: TradParams | None = None,
            keep_maps: bool = False) -> TradResult:
    p = p or TradParams()
    h, w = img.shape
    sig_i = max(sigma_img, p.sigma_floor)
    sm, fl = flatten(img, p)
    R, arg = line_response(fl, p)
    R[~valid] = 0
    # 线响应噪声取背景像素上的 MAD，并限制在图像噪声的 [1/4, 0.6] 倍之间：
    # 标准输入在 0 处截断，干净背景上 MAD 会严重低估（108l 仅 0.05）；
    # 密集磨损镜片的“背景”里混有细密划痕，MAD 会把真实结构当成噪声（76r 达 2.9）
    bg = valid & (fl < 2 * sig_i)
    noise = lambda r: float(np.clip(robust_sigma(r[bg]) if bg.any() else sig_i,  # noqa: E731
                                    p.line_noise_min * sig_i, p.line_noise_max * sig_i))
    sig_r = max(robust_sigma(R[valid]), sig_i / 4.0) if p.line_noise_mode == "mad_floor" else noise(R)
    line_mask = apply_hysteresis_threshold(R / sig_r, p.line_low, p.line_high) & valid
    Rn = R / sig_r                     # 以噪声为单位的线响应（间隙证据用）
    # 长尺度弱划痕层：61 像素两半线核把浅而密的划痕从碎段连成整条（短核信噪比不足，只能检出碎段）
    sig_f = None
    faint = None
    if p.faint_lengths:
        q = TradParams(**{**p.__dict__, "line_lengths": tuple(p.faint_lengths)})
        # 在残差图上检测：已检出的线（外扩 3 像素）置零，否则长核在亮线两侧也有强响应，会给每条亮线镶边
        done = cv2.dilate(line_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        Rf, argf = line_response(np.where(done, 0, fl).astype(np.float32), q)
        Rf[~valid] = 0
        sig_f = noise(Rf)
        faint = apply_hysteresis_threshold(Rf / sig_f, p.faint_low, p.faint_high) & valid & ~line_mask
        arg = np.where(faint, argf, arg).astype(np.uint8)
        line_mask |= faint
        Rn = np.maximum(Rn, Rf / sig_f)
    # 有效区边界残留的亮边会被当成沿边界的长划痕：去掉边界带内的线响应
    inner = cv2.erode(valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.edge_guard_px + 1,) * 2),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    line_mask &= inner > 0
    line_mask_all = line_mask.copy()   # 方向熵按抑制斑点光晕之前的线掩膜计算
    spot_mask = detect_spots(fl, valid, sig_i, p)
    guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.spot_guard_px + 1,) * 2)
    line_mask &= ~cv2.dilate(spot_mask.astype(np.uint8), guard).astype(bool)
    # 骨架与分支。淡线与亮线分别骨架化，且淡线骨架与亮线至少相隔 2 像素：
    # 否则淡线与亮线相交处会在亮线骨架上新增分叉点，把原本完整的亮线切开
    if faint is not None:
        strong = line_mask & ~faint
        fsk = line_mask & faint & ~(cv2.dilate(strong.astype(np.uint8), K3) > 0)
        sk = prune_spurs(skeletonize(strong), p.spur_len) | prune_spurs(skeletonize(fsk), p.spur_len)
    else:
        sk = prune_spurs(skeletonize(line_mask), p.spur_len)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, K3, borderType=cv2.BORDER_CONSTANT) - sk
    endpoints = sk & (nb == 1)
    junction = sk & (nb >= 3)
    jd = cv2.dilate(junction.astype(np.uint8), K3) > 0
    branches = sk & ~jd
    nbr, blab = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    # 小角度交叉在骨架上常表现为两个三叉点夹一小段：两端都接交叉点的短分支并入交叉区
    nj0, jlab0 = cv2.connectedComponents(jd.astype(np.uint8), connectivity=8)
    jnb = ndi.grey_dilation(jlab0, size=3)
    blen = np.bincount(blab.ravel(), minlength=nbr)
    pairs_ = np.unique(np.stack([blab[branches & (jnb > 0)], jnb[branches & (jnb > 0)]], 1), axis=0)
    nj_per_b = np.bincount(pairs_[:, 0], minlength=nbr)
    short = (nj_per_b >= 2) & (blen < p.junction_merge_len)
    short[0] = False
    if short.any() and p.junction_mode == "ends":
        jd |= short[blab]
        branches = sk & ~jd
        nbr, blab = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    # 宽度扩展得到划痕像素
    near = cv2.dilate(line_mask.astype(np.uint8), K3) > 0
    scratch_pix = (line_mask | (near & (fl > p.width_k * sig_i))) & valid & ~spot_mask
    # 受限生长：从线掩膜出发，沿亮于 grow_k·σ 的连通像素生长至多 grow_radius 像素，
    # 找回弯曲段、端部延续和贴近亮划痕的弱划痕（两半线最小值检测对这些位置偏保守）
    lim = cv2.dilate(scratch_pix.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.grow_radius + 1,) * 2)) > 0
    cand = ((fl > p.grow_k * sig_i) & lim & valid & ~spot_mask) | scratch_pix
    glab, ng = ndi.label(cand, np.ones((3, 3)))
    gkeep = np.zeros(ng + 1, bool)
    gkeep[np.unique(glab[scratch_pix])] = True
    gkeep[0] = False
    grown = gkeep[glab] & ~scratch_pix
    scratch_pix |= grown
    fg = scratch_pix | spot_mask

    objs = ndi.find_objects(blab)
    chains: list[np.ndarray] = [np.zeros((0, 2), int)]
    for i in range(1, nbr):
        sl = objs[i - 1]
        yy, xx = np.nonzero(blab[sl] == i)
        chains.append(order_branch(np.stack([yy + sl[0].start, xx + sl[1].start], 1)))
    prof = profiles(chains, sm)
    uf = UF(nbr)
    links = []

    # 交叉点穿越：走向连续 × 截面相似。按“止于交叉区的分支端”两两配对，而不是逐个交叉区配对，
    # 这样一条划痕穿过相邻的几个三叉点（骨架把小角度交叉拆开）时也能接上
    jnb = ndi.grey_dilation(cv2.connectedComponents(jd.astype(np.uint8), connectivity=8)[1], size=3)
    jends = []
    used_ends: set = set()
    bridges = []
    for b in range(1, nbr):
        ch = chains[b]
        if len(ch) < 2:
            continue
        for at_start in (True, False):
            q = ch[0] if at_start else ch[-1]
            at_j = jnb[q[0], q[1]] > 0
            if at_j or endpoints[q[0], q[1]]:   # 骨架在交叉处断开时，自由端也参与
                jends.append((b, q, end_dir(ch, at_start), at_start, at_j))
    if p.junction_mode == "cluster":
        jends = []
        links += pair_by_cluster(jd, branches, blab, chains, prof, uf, p)
    if len(jends) > 1:
        pts = np.array([[e[1][1], e[1][0]] for e in jends], float)
        v2 = p.junction_mode == "ends2"
        if v2:
            # 端部局部的截面剖面（沿轴线最近 end_len 像素）与端部曲率，代替整条分支的中位剖面
            subs = [np.zeros((0, 2), int)] + [chains[e[0]][:p.end_len] if e[3] else chains[e[0]][-p.end_len:]
                                              for e in jends]
            eprof = profiles(subs, sm)[1:]
            ecurv = [end_curvature(chains[e[0]], e[3], p.end_len, floor=p.curv_min) for e in jends]
        acos_deg = lambda x: math.degrees(math.acos(float(np.clip(x, -1, 1))))  # noqa: E731
        cands = []
        for i, j in cKDTree(pts).query_pairs(p.junction_radius):
            (a_, qa, ua, _, _), (b_, qb, ub, _, _) = jends[i], jends[j]
            if a_ == b_:
                continue
            v = pts[j] - pts[i]
            dist = float(np.hypot(*v))
            if v2:
                # 曲率延拓：沿各自端部曲率行进到对端，弦方向 = 切向旋转 κ·d/2，到达方向 = 切向旋转 κ·d
                d = dev_deg(rotate(ua, ecurv[i] * dist), ub)
                if dist >= 4:
                    vn = v / dist
                    d = max(d, acos_deg(np.dot(rotate(ua, ecurv[i] * dist / 2), vn)),
                            acos_deg(np.dot(rotate(ub, ecurv[j] * dist / 2), -vn)))
            else:
                d = dev_deg(ua, ub)
                if dist >= 6:   # 两端相距较远时，连线也须与两端走向一致
                    vn = v / dist
                    d = max(d, acos_deg(np.dot(ua, vn)), acos_deg(np.dot(ub, -vn)))
            if d > p.junction_max_dev:
                continue
            if v2:
                # 端部局部剖面可能被交叉线干扰，取端部与整条分支剖面相似度的较大者
                ps = max(profile_sim(eprof[i], eprof[j]), profile_sim(prof[a_], prof[b_]))
                # 同一条弧穿过交叉点：两端“向外”曲率大小相等、符号相反；只有两端都估出可信曲率时才比较
                wc = math.exp(-((ecurv[i] + ecurv[j]) / p.curv_scale) ** 2) if ecurv[i] and ecurv[j] else 1.0
                s = math.exp(-(d / 25.0) ** 2) * (0.4 + 0.6 * ps) * (0.7 + 0.3 * wc)
            else:
                ps = profile_sim(prof[a_], prof[b_])
                s = math.exp(-(d / 25.0) ** 2) * (0.4 + 0.6 * ps)
            if s >= p.junction_min_score:
                cands.append((s, i, j, d, ps))
        info = {(i, j): (s, d, ps) for s, i, j, d, ps in cands}
        if v2 and p.junction_match == "optimal":   # 交叉区内整体最优配对（平方加权，偏向高分配对）
            chosen = [(min(i, j), max(i, j)) for i, j in max_weight_matching([(s * s, i, j) for s, i, j, _, _ in cands])]
        else:
            chosen, used = [], set()
            for s, i, j, d, ps in sorted(cands, reverse=True):
                if i not in used and j not in used:
                    used |= {i, j}
                    chosen.append((i, j))
        for i, j in chosen:
            s, d, ps = info[(i, j)]
            a_, b_ = jends[i][0], jends[j][0]
            uf.union(a_, b_)
            used_ends |= {(a_, jends[i][3]), (b_, jends[j][3])}
            if np.hypot(*(pts[j] - pts[i])) > 2:
                bridges.append((jends[i][1], jends[j][1], a_))
            o = (pts[i] + pts[j]) / 2
            kind = "crossing" if (jends[i][4] or jends[j][4]) else "gap"   # 两端都是自由端点时实为短间断
            links.append({"type": kind, "a": a_, "b": b_, "score": round(s, 3), "dev": round(d, 1),
                          "profile_sim": round(ps, 3), "pos": [int(o[0]), int(o[1])]})

    # 断点桥接：距离、共线、间隙证据、截面相似
    free = []
    for b in range(1, nbr):
        ch = chains[b]
        if len(ch) < 6:
            continue
        for at_start in (True, False):
            q = ch[0] if at_start else ch[-1]
            if endpoints[q[0], q[1]] and (b, at_start) not in used_ends:
                free.append((b, at_start, q, end_dir(ch, at_start)))
    if len(free) > 1:
        pts = np.array([[f[2][1], f[2][0]] for f in free], float)
        tree = cKDTree(pts)
        cand = []
        for i, j in tree.query_pairs(p.gap_max_dist):
            bi, _, qi, ui = free[i]
            bj, _, qj, uj = free[j]
            if uf.find(bi) == uf.find(bj):
                continue
            v = pts[j] - pts[i]
            d = float(np.hypot(*v))
            if d < 1:
                continue
            vn = v / d
            a1 = math.degrees(math.acos(float(np.clip(np.dot(ui, vn), -1, 1))))
            a2 = math.degrees(math.acos(float(np.clip(np.dot(uj, -vn), -1, 1))))
            dv = dev_deg(ui, uj)
            if max(a1, a2, dv) > p.gap_max_dev:
                continue
            n = max(2, int(d))
            lx = np.linspace(pts[i][0], pts[j][0], n).astype(int)
            ly = np.linspace(pts[i][1], pts[j][1], n).astype(int)
            ev = float(np.mean(Rn[ly, lx]))
            if d > p.gap_need_evidence and ev < 1.0:   # 较远的断点须有间隙脊线证据
                continue
            ps = profile_sim(prof[bi], prof[bj])
            s = (math.exp(-(max(a1, a2, dv) / 20.0) ** 2) * math.exp(-d / 40.0) * (0.4 + 0.6 * ps)
                 * (0.5 + 0.5 * min(1.0, ev / 2.0)))
            if s >= p.gap_min_score:
                cand.append((s, i, j, d, ps, ev))
        used = set()
        for s, i, j, d, ps, ev in sorted(cand, reverse=True):
            if i in used or j in used:
                continue
            used |= {i, j}
            bi, bj = free[i][0], free[j][0]
            uf.union(bi, bj)
            bridges.append((free[i][2], free[j][2], bi))
            links.append({"type": "gap", "a": bi, "b": bj, "score": round(s, 3), "dist": round(d, 1),
                          "profile_sim": round(ps, 3), "evidence": round(ev, 2),
                          "pos": [int(pts[i][0]), int(pts[i][1]), int(pts[j][0]), int(pts[j][1])]})

    # 像素归属到分支 → 实例
    skel_lab = np.where(branches, blab, 0).astype(np.int32)
    jfill = ndi.grey_dilation(skel_lab, size=3)
    skel_lab = np.where(jd & sk & (skel_lab == 0), jfill, skel_lab)
    bridge_mask = np.zeros((h, w), np.uint8)
    for qa, qb, b in bridges:
        wdt = max(1, int(round(prof[b]["width"] or 1)))
        cv2.line(bridge_mask, (int(qa[1]), int(qa[0])), (int(qb[1]), int(qb[0])), 1, wdt)
        n = int(max(abs(int(qb[0]) - int(qa[0])), abs(int(qb[1]) - int(qa[1])))) + 1
        ly = np.linspace(qa[0], qb[0], n).round().astype(int)
        lx = np.linspace(qa[1], qb[1], n).round().astype(int)
        empty = skel_lab[ly, lx] == 0
        skel_lab[ly[empty], lx[empty]] = int(b)       # 桥接段也作为该分支的骨架，便于像素归属
    pix = (scratch_pix | (bridge_mask > 0)) & valid
    root = np.array([uf.find(i) for i in range(nbr)], np.int32)
    inst = np.zeros((h, w), np.int32)
    cls = np.zeros((h, w), np.uint8)
    if skel_lab.any():
        dist, (iy, ix) = ndi.distance_transform_edt(skel_lab == 0, return_indices=True)
        owner = skel_lab[iy, ix]
        ok = pix & ((dist <= 6) | (grown & (dist <= 6 + p.grow_radius))) & (owner > 0)
        inst[ok] = root[owner[ok]]
        rest = pix & ~ok
    else:
        rest = pix
    # 实例属性与分类
    instances = []
    roots = np.unique(inst[inst > 0])
    blen = np.bincount(skel_lab.ravel(), minlength=nbr)
    members: dict = {}
    for b in range(1, nbr):
        members.setdefault(int(root[b]), []).append(b)
    new_id = np.zeros(int(inst.max()) + 1, np.int32)
    nid = 0
    for r in roots:
        bs = members.get(int(r), [int(r)])
        L = int(sum(blen[b] for b in bs))
        widths = [prof[b]["width"] for b in bs if prof[b]["width"] > 0]
        amps = [prof[b]["amp"] for b in bs if prof[b]["amp"] > 0]
        W = float(np.median(widths)) if widths else 1.0
        A = float(np.median(amps)) if amps else 0.0
        if A < p.noise_amp_k * sig_i and L < 40:
            continue
        nid += 1
        new_id[r] = nid
        elong = L / max(W, 1.0)
        c = "scratch" if (L >= p.min_scratch_len and elong >= p.min_elong) else "uncertain"
        instances.append({"id": nid, "class": c, "branches": bs, "len_px": L, "width_px": round(W, 1),
                          "amp": round(A, 1), "n_links": 0, "flags": []})
    inst = new_id[inst]
    # 划痕像素按“中心线 + 截面半高宽”取：2σ 迟滞/生长会把光学模糊的外沿一并划入（掩膜宽于可见线宽 2–3 倍）
    is_scr = np.zeros(nid + 1, bool)
    for it in instances:
        is_scr[it["id"]] = it["class"] == "scratch"
    traced = is_scr[inst]          # 收窄前的划痕归属：光晕外沿也算“已追踪”，不计入碎裂判定
    if skel_lab.any() and instances:
        half_w = np.full(nid + 1, np.inf, np.float32)
        for it in instances:
            if it["class"] == "scratch":
                half_w[it["id"]] = max(1.0, it["width_px"] / 2.0) + p.width_margin
        inst[(inst > 0) & (dist > half_w[inst])] = 0
    for it in instances:
        bs = set(it["branches"])
        it["n_links"] = sum(1 for lk in links if lk["a"] in bs or lk["b"] in bs)
    cls_of = {it["id"]: it["class"] for it in instances}
    lut = np.zeros(nid + 1, np.uint8)
    for i, c in cls_of.items():
        lut[i] = CLASS_ID[c]
    cls = lut[inst]
    # 碎裂区（v1/初版：按局部纹理判定，在斑点之前处理；footprint 模式在全部类别确定后按光束足迹判定）
    if p.damage_mode == "legacy":
        dmg = damage_regions_legacy(line_mask_all, arg, fg, sk, endpoints, valid, p)
    elif p.damage_mode == "untraced":
        # 受限生长得到的弱延续像素不算“已追踪”，否则密集区会被生长整体吞并
        untraced = (fl > 3 * sig_i) & valid & (~traced | grown) & ~spot_mask
        dmg = damage_regions(untraced, line_mask_all, arg, valid, p)
    else:
        dmg = np.zeros((h, w), bool)
    if dmg.any():
        ids, cnt = np.unique(inst[dmg & (inst > 0)], return_counts=True)
        tot = np.bincount(inst.ravel(), minlength=nid + 1)
        for i, c in zip(ids, cnt):
            it = instances[i - 1]
            if c / max(tot[i], 1) <= 0.5:
                continue
            if it["len_px"] < p.damage_keep_len:
                it["class"] = "damage"
            else:
                it["flags"].append("位于密集区")
        lut = np.zeros(nid + 1, np.uint8)
        for it in instances:
            lut[it["id"]] = CLASS_ID[it["class"]]
        cls = lut[inst]
        cls[dmg & (fg | (cls > 0)) & (cls != CLASS_ID["scratch"])] = CLASS_ID["damage"]
    rest &= cls == 0
    cls[rest & dmg] = CLASS_ID["damage"]
    cls[rest & ~dmg] = CLASS_ID["uncertain"]
    if p.grey_confirm:
        # 无骨架的零散像素按连通域成为待确认实例，便于整块确认或删除
        rl, nr = ndi.label(rest & (cls == CLASS_ID["uncertain"]), np.ones((3, 3)))
        for k, sl in enumerate(ndi.find_objects(rl), 1):
            m = rl[sl] == k
            nid += 1
            inst[sl][m] = nid
            instances.append({"id": nid, "class": "uncertain", "branches": [], "len_px": 0,
                              "width_px": 0.0, "amp": round(float(sm[sl][m].max()), 1), "n_links": 0,
                              "flags": ["无骨架的零散亮像素"]})
    # 划痕走向场与实例走向
    th_f, coh_f, dens_f = orientation_field(line_mask, arg, p)
    objs_i = ndi.find_objects(inst)
    for it in instances:
        sl = objs_i[it["id"] - 1] if it["id"] - 1 < len(objs_i) else None
        if sl is None:
            continue
        cs = [chains[b] for b in it["branches"] if len(chains[b])]
        if cs:
            yx = np.vstack(cs)
            yy, xx = yx[:, 0], yx[:, 1]
        else:
            yy, xx = np.nonzero(inst[sl] == it["id"])
            yy, xx = yy + sl[0].start, xx + sl[1].start
        th, el = pca_orient(yy, xx)
        fy, fx = min(int(yy.mean()) // 4, th_f.shape[0] - 1), min(int(xx.mean()) // 4, th_f.shape[1] - 1)
        it["orient_deg"] = round(math.degrees(th) % 180, 1)
        it["axis_ratio"] = round(el, 2)
        it["field_coh"] = round(float(coh_f[fy, fx]), 2)
        if p.grey_confirm and it["class"] == "uncertain":
            span = float(np.hypot(np.ptp(yy), np.ptp(xx)))
            if (el >= p.grey_min_elong and coh_f[fy, fx] >= p.grey_min_coh
                    and ang_diff(th, float(th_f[fy, fx])) <= p.grey_max_dev):
                it["class"] = "scratch"
                it["flags"].append("浅密划痕：与局部走向场一致")
            elif el >= p.grey_line_ratio and span >= p.grey_line_len and it["amp"] >= 3 * sig_i:
                it["class"] = "scratch"
                it["flags"].append("浅划痕：细长线状片段")
    lut = np.zeros(nid + 1, np.uint8)
    for it in instances:
        lut[it["id"]] = CLASS_ID[it["class"]]
    cls = np.where(inst > 0, lut[inst], cls).astype(np.uint8)
    # 斑点实例
    n_s, slab = cv2.connectedComponents(spot_mask.astype(np.uint8), connectivity=8)
    for i in range(1, n_s):
        m = slab == i
        nid += 1
        inst[m] = nid
        cls[m] = CLASS_ID["spot"] if not dmg[m].mean() > 0.5 else CLASS_ID["damage"]
        yy, xx = np.nonzero(m)
        instances.append({"id": nid, "class": "spot" if cls[yy[0], xx[0]] == 2 else "damage", "branches": [],
                          "len_px": 0, "width_px": round(2 * math.sqrt(len(yy) / math.pi), 1),
                          "amp": round(float(sm[m].max()), 1), "n_links": 0, "flags": []})
    # 交叉区二元叠加所属：桥接段（暗线穿过亮线的部分）按线宽栅格化，已被另一实例占据的像素记为叠加，
    # 亮者为主、暗者为次；无人占据的像素直接归该实例
    inst2 = np.zeros((h, w), np.int32)
    if p.crossing_overlap and bridges:
        amp_of = np.zeros(nid + 1, np.float32)
        wid_of = np.ones(nid + 1, np.float32)
        orient_of = np.zeros(nid + 1, np.float32)
        for it in instances:
            amp_of[it["id"]] = it["amp"]
            wid_of[it["id"]] = max(1.0, it["width_px"])
            orient_of[it["id"]] = it.get("orient_deg", 0.0)
        for qa, qb, b in bridges:
            r = int(root[b])
            own = int(new_id[r]) if r < len(new_id) else 0
            if own == 0 or instances[own - 1]["class"] != "scratch":
                continue
            wd = int(round(wid_of[own]))
            y0, x0 = max(0, min(qa[0], qb[0]) - wd), max(0, min(qa[1], qb[1]) - wd)
            y1, x1 = min(h, max(qa[0], qb[0]) + wd + 1), min(w, max(qa[1], qb[1]) + wd + 1)
            band = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.line(band, (int(qa[1]) - x0, int(qa[0]) - y0), (int(qb[1]) - x0, int(qb[0]) - y0), 1, max(1, wd))
            by, bx = np.nonzero(band)
            by, bx = by + y0, bx + x0
            ok = valid[by, bx] & (cls[by, bx] != CLASS_ID["spot"])
            by, bx = by[ok], bx[ok]
            cur = inst[by, bx]
            e = cur == 0
            inst[by[e], bx[e]] = own
            cls[by[e], bx[e]] = CLASS_ID["scratch"]
            o = (cur > 0) & (cur != own) & (inst2[by, bx] == 0)
            # 只在真正的交叉处记叠加：与桥接段近乎平行的实例（贴边并行的划痕）不算
            bdir = math.degrees(math.atan2(qb[0] - qa[0], qb[1] - qa[1])) % 180
            ang = np.array([ang_diff(math.radians(bdir), math.radians(orient_of[c])) for c in cur[o]]) if o.any() else np.zeros(0)
            keep = np.zeros(len(by), bool)
            keep[np.flatnonzero(o)[ang >= p.overlap_min_angle]] = True
            o = keep
            oy, ox, oc = by[o], bx[o], cur[o]
            brighter = amp_of[own] > amp_of[oc]
            inst[oy[brighter], ox[brighter]] = own
            cls[oy[brighter], ox[brighter]] = CLASS_ID["scratch"]
            inst2[oy, ox] = np.where(brighter, oc, own)
    # 碎裂区（按光束足迹）：足迹内缺陷像素占比 ≥ damage_f_min 的连通区，面积至少一个足迹
    regions = []
    fmap = None
    cy, cx = (float(v.mean()) for v in np.nonzero(valid))
    if p.damage_mode == "footprint":
        fmap, dmg = footprint_damage(cls > 0, valid, p)
        if dmg.any():
            long_scr = np.zeros(nid + 1, bool)
            for it in instances:
                long_scr[it["id"]] = it["class"] == "scratch" and it["len_px"] * p.pixel_mm >= p.damage_trace_len_mm
            th_px = np.pi * arg.astype(np.float32) / p.n_orient
            tot = np.bincount(inst.ravel(), minlength=nid + 1)
            n_r, rlab, rst, rcen = cv2.connectedComponentsWithStats(dmg.astype(np.uint8), connectivity=8)
            for k in range(1, n_r):
                x, y, ww, hh = rst[k, :4]
                sl = np.s_[y:y + hh, x:x + ww]
                m = rlab[sl] == k
                dm = (cls[sl] > 0) & m
                tw = th_px[sl][line_mask[sl] & m]
                c2, s2 = (float(np.cos(2 * tw).mean()), float(np.sin(2 * tw).mean())) if tw.size else (0.0, 0.0)
                trace = float(long_scr[inst[sl][dm]].mean()) if dm.any() else 0.0
                # 分型：区内缺陷像素过半属于可逐条追踪的划痕（≥0.5 mm）为“密集划痕型”，否则为“碎裂型”
                kind = "dense_scratch" if trace >= p.damage_traceable_max else "fragment"
                regions.append({
                    "id": k, "type": kind, "area_mm2": round(float(m.sum()) * p.pixel_mm ** 2, 2),
                    "footprint_frac_mean": round(float(fmap[sl][m].mean()), 3),
                    "defect_frac": round(float(dm.sum() / m.sum()), 3),
                    "scatter_mean_counts": round(float(fl[sl][m].mean()), 2),
                    "traceable_frac": round(trace, 3),
                    "orient_order": round(math.hypot(c2, s2), 3),
                    "orient_dominant_deg": round(math.degrees(0.5 * math.atan2(s2, c2)) % 180, 1),
                    "center_dist_mm": round(math.hypot(rcen[k][0] - cx, rcen[k][1] - cy) * p.pixel_mm, 2),
                    "pos": [int(rcen[k][0]), int(rcen[k][1])]})
                ids, cnt = np.unique(inst[sl][m & (inst[sl] > 0)], return_counts=True)
                for i, c in zip(ids, cnt):
                    it = instances[i - 1]
                    if c / max(tot[i], 1) <= 0.5:
                        continue
                    if kind == "fragment" and it["class"] == "uncertain":
                        it["class"] = "damage"
                    elif it["class"] != "damage":
                        it["flags"].append("位于碎裂区" if kind == "fragment" else "位于密集划痕区")
                if kind == "fragment":
                    sub = cls[sl]
                    sub[(inst[sl] == 0) & (sub == CLASS_ID["uncertain"]) & m] = CLASS_ID["damage"]
            lut = np.zeros(nid + 1, np.uint8)
            for it in instances:
                lut[it["id"]] = CLASS_ID[it["class"]]
            cls = np.where(inst > 0, lut[inst], cls).astype(np.uint8)
    # 复核标记
    for lk in links:
        if (lk["type"] == "crossing" and lk["score"] < 0.5) or (lk["type"] == "gap" and lk["score"] < 0.4):
            lk["review"] = True
    for it in instances:
        if it["class"] == "uncertain" and "无骨架的零散亮像素" not in it["flags"]:
            it["flags"].append("短碎片或细长比不足")
    # 全镜走向统计：划痕骨架像素的走向分布、取向有序度（0 随机 – 1 完全平行）、处于平行族中的比例
    scr_id = np.zeros(nid + 1, bool)
    for it in instances:
        scr_id[it["id"]] = it["class"] == "scratch"
    sp = sk & scr_id[inst]
    ka = arg[sp].astype(np.float32)
    hist = np.bincount(arg[sp], minlength=p.n_orient)[:p.n_orient]
    tw = np.pi * ka / p.n_orient
    order = float(np.hypot(np.cos(2 * tw).mean(), np.sin(2 * tw).mean())) if tw.size else 0.0
    dom = float(np.degrees(0.5 * np.arctan2(np.sin(2 * tw).mean(), np.cos(2 * tw).mean())) % 180) if tw.size else 0.0
    yy, xx = np.nonzero(sp)
    par = float((coh_f[np.minimum(yy // 4, coh_f.shape[0] - 1), np.minimum(xx // 4, coh_f.shape[1] - 1)] >= 0.7).mean()) if yy.size else 0.0
    va = float(valid.sum())
    cz = (np.hypot(*np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)) <= p.central_zone_mm / 2 / p.pixel_mm) & valid \
        if p.damage_mode == "footprint" else None
    stats = {"sigma_img": round(sig_i, 3), "sigma_line": round(sig_r, 4),
             "sigma_faint": round(sig_f, 4) if sig_f else None, "n_branches": nbr - 1,
             "n_crossing_links": sum(1 for lk in links if lk["type"] == "crossing"),
             "n_gap_links": sum(1 for lk in links if lk["type"] == "gap"),
             "n_review_links": sum(1 for lk in links if lk.get("review")),
             "damage_px": int(dmg.sum()), "fg_px": int((cls > 0).sum()),
             "overlap_px": int((inst2 > 0).sum()),
             "orient_hist": hist.tolist(), "orient_order": round(order, 3), "orient_dominant_deg": round(dom, 1),
             "parallel_frac": round(par, 3),
             "damage_frac_valid": round(float(dmg.sum()) / va, 4) if va else 0.0}
    if cz is not None:
        stats["damage_frac_central"] = round(float((dmg & cz).sum()) / max(float(cz.sum()), 1.0), 4)
        stats["large_damage"] = bool(stats["damage_frac_valid"] >= p.damage_lens_frac)
        frag = sum(r["area_mm2"] for r in regions if r["type"] == "fragment") / (p.pixel_mm ** 2)
        stats["damage_frac_fragment"] = round(frag / va, 4) if va else 0.0
    maps = {"flat": fl, "line": Rn, "line_mask": line_mask, "skeleton": sk, "damage": dmg,
            "spot": spot_mask, "bridge": bridge_mask > 0, "chains": chains, "profiles": prof,
            "bridge_segments": bridges, "field_theta": th_f, "field_coh": coh_f,
            "footprint_frac": fmap} if keep_maps else {}
    return TradResult(cls=cls, inst=inst, instances=instances, links=links, maps=maps, stats=stats,
                      inst2=inst2, regions=regions)
