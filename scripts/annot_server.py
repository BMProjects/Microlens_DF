"""像素级人工标注服务（局域网共享）：按缺陷的产生方式标注，由图像算法定位缺陷区域。

审核单位是 512×512 的窗口（约 3.5 mm）；画面同时显示窗口外 256 像素的边距，长划痕可以画出窗口，
并可沿走向自动追踪到整片镜片。界面左右并排：左为原图（不叠色），右为标注结果，两侧同步缩放平移。

标注方式：
- 划痕：沿划痕脊线拖动鼠标 → 在拖动轨迹两侧 6 像素的走廊内做最小代价路径吸附到亮脊线
  → 沿脊线法向测亮度剖面，按半高宽（不低于 2σ）确定划痕区域；
- 直划痕：单击起点、单击终点（Shift 单击加折点）→ 每段保持直线，按 ±3 像素内的亮度峰值拟合平移/微转对齐脊线，再定宽；
- 斑点：单击 → 在 4 像素内找亮度峰值，按半峰值（不低于 3σ）区域生长；
- 碎裂：圈选区域 → 区域内亮于 3σ 的像素；
- 不计：拉框；
- 长划痕：选中划痕按 E，从两端沿走向逐步追踪（方向惯性穿过交叉点，连续 5 步无脊线证据即停止）。
trad_v2 的结果只作为“建议”（虚线脊线、斑点圆圈）显示，单击才采纳；采纳后同样经吸附与区域定位生成掩膜。

提交时服务端生成窗口标签图（0 背景，1 划痕，2 斑点，3 碎裂，255 不计/有效区外）；
所有标注以整镜坐标保存在状态文件中（含窗口外的长划痕），可追溯、可重建，也可作为训练的稀疏正样本。

用法:
    uv run python scripts/annot_server.py init              # 依据 splits_v1.json 与 trad_v2 结果生成窗口任务
    uv run python scripts/annot_server.py serve             # 局域网访问 http://<本机IP>:8500/?t=<令牌>
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STD_DIR = PROJECT_ROOT / "output/dataset_v3"
TRAD_DIR = PROJECT_ROOT / "output/annotation/trad_v2"
SPLITS = PROJECT_ROOT / "output/annotation/splits_v1.json"
BENCH = PROJECT_ROOT / "output/annotation/bench_v1"
LINKS = PROJECT_ROOT / "output/annotation/ablation/link_review"
WEB = PROJECT_ROOT / "scripts/annot_web"
WIN = 512
MARGIN = 256                # 画面显示的窗口外边距
PIXEL_MM = 0.0068
CENTRAL_MM = 9.0
IGNORE = 255
SEED = 20261026
CORRIDOR = 6                # 划痕吸附走廊半宽（像素）
MAX_HALF_WIDTH = 6          # 划痕区域单侧最大半宽（像素）


# ── 窗口任务 ──────────────────────────────────────────────────────────

def load_valid(lens: str) -> np.ndarray:
    v = cv2.imread(str(STD_DIR / "valid" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if v is None:
        raise FileNotFoundError(f"有效区缺失: {lens}")
    return v > 0


def zone_windows(lens: str, per_zone: int, rng: np.random.Generator, taken: list) -> list[dict]:
    """中央 9 mm 区、内环、外环（以中央区外有效像素半径的中位数为界）各随机取 per_zone 个窗口，
    窗口内有效像素 ≥ 95%，彼此不重叠。"""
    valid = load_valid(lens)
    h, w = valid.shape
    ys, xs = np.nonzero(valid)
    cy, cx = ys.mean(), xs.mean()
    rc = CENTRAL_MM / 2 / PIXEL_MM
    r_all = np.hypot(ys - cy, xs - cx)
    r_mid = float(np.median(r_all[r_all > rc])) if (r_all > rc).any() else rc
    integ = cv2.integral(valid.astype(np.uint8))
    out = []
    for zone, (lo, hi) in (("central", (0, rc)), ("inner", (rc, r_mid)), ("outer", (r_mid, 1e9))):
        got, tries = 0, 0
        while got < per_zone and tries < 4000:
            tries += 1
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(lo, min(hi, r_all.max()))
            y0 = int(cy + rad * math.sin(ang)) - WIN // 2
            x0 = int(cx + rad * math.cos(ang)) - WIN // 2
            if y0 < 0 or x0 < 0 or y0 + WIN > h or x0 + WIN > w:
                continue
            frac = (integ[y0 + WIN, x0 + WIN] - integ[y0, x0 + WIN] - integ[y0 + WIN, x0] + integ[y0, x0]) / WIN ** 2
            if frac < 0.95 or any(abs(y0 - b[0]) < WIN and abs(x0 - b[1]) < WIN for b in taken):
                continue
            taken.append((y0, x0))
            out.append({"lens": lens, "zone": zone, "x0": x0, "y0": y0})
            got += 1
    return out


def cmd_init(args: argparse.Namespace) -> None:
    tasks_path = BENCH / "tasks.json"
    if tasks_path.exists() and not args.force:
        sys.exit(f"{tasks_path} 已存在；窗口位置一经生成即冻结，确需重建请加 --force")
    splits = json.loads(SPLITS.read_text())["splits"]
    missing = [ls for v in splits.values() for ls in v if not (TRAD_DIR / f"{ls}.json").exists()]
    if missing:
        sys.exit(f"trad_v2 结果缺失 {len(missing)} 张（如 {missing[:5]}），请先完成 annot_build.py run --all")
    rng = np.random.default_rng(SEED)
    tasks = []
    # 调参（锚点）与测试：每片 3 区 × 2 窗，完整审核；训练：每片 3 区 × 1 窗，作为第一批训练标签
    plan = [("val", 2, True), ("train", 1, True), ("test", 2, False)]
    for split, per_zone, enabled in plan:
        for ls in splits[split]:
            for w in zone_windows(ls, per_zone, rng, []):
                tasks.append({**w, "split": split, "enabled": enabled})
    # 队列顺序：先调参（锚点），再训练（按镜片轮转，使早期标注覆盖尽量多的镜片）；测试集待 U1 候选就绪后启用
    order = {"val": 0, "train": 1, "test": 2}
    tasks.sort(key=lambda t: (order[t["split"]], {"central": 0, "inner": 1, "outer": 2}[t["zone"]], t["lens"]))
    for i, t in enumerate(tasks):
        t["wid"] = f"{t['split'][:2]}{i:04d}_{t['lens']}_{t['zone'][0]}"
    BENCH.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps({"version": "bench_v1", "win": WIN, "seed": SEED, "created": now(),
                                      "suggestions": ["trad_v2"], "tasks": tasks},
                                     ensure_ascii=False, indent=1))
    from collections import Counter
    print(Counter((t["split"], t["enabled"]) for t in tasks))


# ── 镜片数据与算法建议 ─────────────────────────────────────────────────

class LensCache:
    """最近使用的若干张镜片：标准输入、平滑图（用于吸附与追踪）、有效区、噪声 σ、trad_v2 建议。"""

    def __init__(self, n: int = 6):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()
        self.rep = {r["lens"]: r for r in json.loads((STD_DIR / "report.json").read_text())}

    def get(self, lens: str) -> dict:
        with self.lock:
            if lens in self.d:
                self.d.move_to_end(lens)
                return self.d[lens]
        img = cv2.imread(str(STD_DIR / "images" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(lens)
        e = {"img": img, "f": cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0), "valid": load_valid(lens),
             "sigma": max(float(self.rep[lens]["noise_sigma"]), 0.8), "sugg": load_suggestions(lens)}
        with self.lock:
            self.d[lens] = e
            while len(self.d) > self.n:
                self.d.popitem(last=False)
        return e


def load_suggestions(lens: str) -> list[dict]:
    """trad_v2 的 LabelMe 导出：同一划痕实例的各分支（含桥接段）合为一条建议；斑点给出中心与半径。"""
    p = TRAD_DIR / f"{lens}_labelme.json"
    if not p.exists():
        return []
    shapes = json.loads(p.read_text())["shapes"]
    scr: dict[int, dict] = {}
    out = []
    for s in shapes:
        if s["label"] == "scratch" and s["shape_type"] == "linestrip" and s.get("group_id"):
            it = scr.setdefault(s["group_id"], {"kind": "scratch", "lines": []})
            it["lines"].append(s["points"])
        elif s["label"] == "spot" and s["shape_type"] == "polygon":
            pts = np.asarray(s["points"], np.float32)
            c = pts.mean(0)
            r = float(np.sqrt(max(cv2.contourArea(pts), 1.0) / np.pi))
            out.append({"kind": "spot", "c": [round(float(c[0]), 1), round(float(c[1]), 1)], "r": round(r, 1)})
    out = list(scr.values()) + out
    for i, s in enumerate(out):
        s["sid"] = i + 1
        pts = np.concatenate([np.asarray(ln) for ln in s["lines"]]) if s["kind"] == "scratch" else np.asarray([s["c"]])
        s["bbox"] = [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]
    return out


def region_of(t: dict, shape: tuple) -> dict:
    h, w = shape
    x0, y0 = max(0, t["x0"] - MARGIN), max(0, t["y0"] - MARGIN)
    x1, y1 = min(w, t["x0"] + WIN + MARGIN), min(h, t["y0"] + WIN + MARGIN)
    return {"x0": x0, "y0": y0, "w": x1 - x0, "h": y1 - y0}


# ── 由人工输入定位缺陷区域 ─────────────────────────────────────────────

def densify(pts: np.ndarray, step: float = 1.0) -> np.ndarray:
    """折线按弧长等距重采样（x, y）。"""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return pts[:1]
    u = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], 1)


def snap_ridge(e: dict, pts: list, corridor: int = CORRIDOR) -> np.ndarray:
    """拖动轨迹（或直线折点）→ 走廊内最小代价路径（亮度越高代价越低，常数项抑制绕行）；返回逐像素有序路径 (x, y)。
    直线工具用较窄的走廊（3 像素），使结果保持笔直、只做横向居中。"""
    from skimage.graph import route_through_array
    P = densify(pts, 2.0)
    h, w = e["f"].shape
    pad = corridor + 4
    xa, ya = int(max(0, P[:, 0].min() - pad)), int(max(0, P[:, 1].min() - pad))
    xb, yb = int(min(w, P[:, 0].max() + pad + 1)), int(min(h, P[:, 1].max() + pad + 1))
    f = e["f"][ya:yb, xa:xb]
    corr = np.zeros(f.shape, np.uint8)
    cv2.polylines(corr, [np.round(P - [xa, ya]).astype(np.int32).reshape(-1, 1, 2)], False, 1, 2 * corridor + 1)
    cost = 0.5 + 1.0 / (0.3 + np.clip(f / e["sigma"], 0, None))
    cost[corr == 0] = 1e4

    def snap_end(p):           # 端点吸附到 3 像素内的亮度峰值
        x, y = int(round(p[0])) - xa, int(round(p[1])) - ya
        y0, x0 = max(0, y - 3), max(0, x - 3)
        sub = f[y0:y + 4, x0:x + 4]
        if sub.size == 0:
            return (min(max(y, 0), f.shape[0] - 1), min(max(x, 0), f.shape[1] - 1))
        dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        return (y0 + dy, x0 + dx)
    a, b = snap_end(P[0]), snap_end(P[-1])
    if a == b:
        return np.array([[a[1] + xa, a[0] + ya]], np.float64)
    idx, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
    return np.array([[c + xa, r + ya] for r, c in idx], np.float64)


def snap_straight(e: dict, pts: list, reach: float = 3.0) -> np.ndarray:
    """直线工具：每段保持直线，只做平移与微转对齐到脊线。
    沿段逐像素在法向 ±reach 内找亮度峰值，用峰值高于 2σ 的点拟合“法向偏移 = a + b·弧长”（最小二乘），
    证据点不足段长的 20% 时不移动。返回逐像素有序路径 (x, y)。"""
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    out = []
    T = np.arange(-reach, reach + 0.25, 0.5)
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 1:
            continue
        u = (b - a) / L
        nrm = np.array([-u[1], u[0]])
        sgrid = np.arange(0, L + 1e-9, 1.0)
        base = a[None] + sgrid[:, None] * u[None]
        sx = (base[:, None, 0] + T[None, :] * nrm[0]).astype(np.float32)
        sy = (base[:, None, 1] + T[None, :] * nrm[1]).astype(np.float32)
        prof = cv2.remap(e["f"], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        off, peak = T[prof.argmax(1)], prof.max(1)
        ok = peak >= 2 * e["sigma"]
        if ok.sum() >= max(5, 0.2 * len(sgrid)):
            A = np.stack([np.ones(ok.sum()), sgrid[ok]], 1)
            coef, *_ = np.linalg.lstsq(A, off[ok], rcond=None)
            shift = np.clip(coef[0] + coef[1] * sgrid, -reach, reach)
        else:
            shift = np.zeros_like(sgrid)
        seg = base + shift[:, None] * nrm[None]
        out.append(seg if not out else seg[1:])
    return np.concatenate(out) if out else P[:1]


def ridge_region(e: dict, path: np.ndarray) -> np.ndarray:
    """沿脊线法向测亮度剖面，按半高宽（阈值不低于 2σ）定位划痕区域；返回像素 (x, y)。"""
    h, w = e["f"].shape
    path = np.asarray(path, np.float64).reshape(-1, 2)
    if len(path) >= 2:
        k = min(7, len(path) - 1)
        tang = np.gradient(cv2.blur(path.reshape(-1, 1, 2), (1, k)).reshape(-1, 2), axis=0)
    else:
        tang = np.array([[1.0, 0.0]])
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-6)[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    T = np.arange(-8, 8.5, 0.5)
    sx = (path[:, None, 0] + T[None, :] * nrm[:, None, 0]).astype(np.float32)
    sy = (path[:, None, 1] + T[None, :] * nrm[:, None, 1]).astype(np.float32)
    prof = cv2.remap(e["f"], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mid = len(T) // 2
    core = np.abs(T) <= 2
    peak = prof[:, core].max(1)
    thr = np.maximum(2 * e["sigma"], 0.5 * peak)
    above = prof >= thr[:, None]
    ext = np.zeros((len(path), 2))
    for side, rng_ in ((0, range(mid, -1, -1)), (1, range(mid, len(T)))):
        run = np.ones(len(path), bool)
        for j in rng_:
            run &= above[:, j]
            ext[run, side] = abs(T[j])
    ext = np.clip(ext, 1.0, MAX_HALF_WIDTH)
    ext = ndi.median_filter(ext, size=(min(7, len(path)), 1), mode="nearest")
    m = np.zeros((h, w), bool)
    xs, ys = [], []
    for j, t in enumerate(T):
        ok = (t >= -ext[:, 0]) & (t <= ext[:, 1])
        xs.append(sx[ok, j])
        ys.append(sy[ok, j])
    X = np.clip(np.round(np.concatenate(xs)).astype(int), 0, w - 1)
    Y = np.clip(np.round(np.concatenate(ys)).astype(int), 0, h - 1)
    m[Y, X] = True
    if len(path):
        pi = np.round(path).astype(int)
        m[np.clip(pi[:, 1], 0, h - 1), np.clip(pi[:, 0], 0, w - 1)] = True
    ya, yb = max(0, int(Y.min()) - 2), min(h, int(Y.max()) + 3)
    xa, xb = max(0, int(X.min()) - 2), min(w, int(X.max()) + 3)
    sub = cv2.morphologyEx(m[ya:yb, xa:xb].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    sub &= e["valid"][ya:yb, xa:xb]
    yy, xx = np.nonzero(sub)
    return np.stack([xx + xa, yy + ya], 1)


def track(e: dict, path: np.ndarray, max_len: int = 6000) -> np.ndarray:
    """从路径末端沿走向逐步追踪：每步 5 像素，在 ±15° 扇区内选线上均值最高的方向；
    方向带惯性（可穿过交叉点）；连续 5 步低于 2.5σ 即停止并回退这些步。返回追加的点 (x, y)。"""
    f, valid, s = e["f"], e["valid"], e["sigma"]
    h, w = f.shape
    path = np.asarray(path, np.float64)
    if len(path) < 6:
        return np.zeros((0, 2))
    p = path[-1].copy()
    d = path[-1] - path[max(0, len(path) - 15)]
    d /= max(np.hypot(*d), 1e-6)
    step, out, miss = 5.0, [], 0
    angs = np.deg2rad(np.arange(-15, 16, 5))
    for _ in range(int(max_len / step)):
        best, bq, bscore = None, None, -1e9
        for a in angs:
            c, sn = math.cos(a), math.sin(a)
            dd = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            q = p + step * dd
            seg = p[None] + np.linspace(0.2, 1, 5)[:, None] * (q - p)[None]
            v = cv2.remap(f, seg[:, 0:1].astype(np.float32), seg[:, 1:2].astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).mean()
            score = v - 0.02 * s * abs(math.degrees(a))
            if score > bscore:
                best, bq, bscore = dd, q, score
        if not (0 <= bq[0] < w and 0 <= bq[1] < h) or not valid[int(bq[1]), int(bq[0])]:
            break
        miss = miss + 1 if bscore < 2.5 * s else 0
        if miss > 4:
            break
        out.append(bq)
        d = 0.75 * d + 0.25 * best
        d /= max(np.hypot(*d), 1e-6)
        p = bq
    if miss:
        out = out[:-miss] if miss < len(out) else []
    return np.asarray(out).reshape(-1, 2)


def spot_region(e: dict, x: float, y: float) -> np.ndarray:
    f = e["f"]
    h, w = f.shape
    xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
    r = 4
    y0, x0 = max(0, yi - r), max(0, xi - r)
    sub = f[y0:yi + r + 1, x0:xi + r + 1]
    dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    py, px = y0 + dy, x0 + dx
    R = 25
    ya, xa = max(0, py - R), max(0, px - R)
    loc = f[ya:py + R + 1, xa:px + R + 1]
    thr = max(3 * e["sigma"], 0.5 * float(f[py, px]))
    lab, _ = ndi.label(loc >= thr)
    m = lab == lab[py - ya, px - xa]
    yy, xx = np.ogrid[:loc.shape[0], :loc.shape[1]]
    m &= (yy - (py - ya)) ** 2 + (xx - (px - xa)) ** 2 <= R ** 2
    m[py - ya, px - xa] = True
    m &= e["valid"][ya:ya + loc.shape[0], xa:xa + loc.shape[1]]
    ys, xs = np.nonzero(m)
    return np.stack([xs + xa, ys + ya], 1)


def fragment_region(e: dict, poly: list) -> np.ndarray:
    P = np.round(np.asarray(poly, np.float64)).astype(np.int32)
    h, w = e["f"].shape
    xa, ya = max(0, P[:, 0].min()), max(0, P[:, 1].min())
    xb, yb = min(w, P[:, 0].max() + 1), min(h, P[:, 1].max() + 1)
    m = np.zeros((yb - ya, xb - xa), np.uint8)
    cv2.fillPoly(m, [(P - [xa, ya]).reshape(-1, 1, 2)], 1)
    m = (m > 0) & (e["f"][ya:yb, xa:xb] >= 3 * e["sigma"]) & e["valid"][ya:yb, xa:xb]
    ys, xs = np.nonzero(m)
    return np.stack([xs + xa, ys + ya], 1)


def render_label(e: dict, t: dict, st: dict) -> np.ndarray:
    """窗口标签：碎裂 → 划痕 → 斑点 依次覆盖（斑点在划痕上时以斑点为准），不计框最后覆盖，有效区外为不计。"""
    x0, y0 = t["x0"], t["y0"]
    out = np.zeros((WIN, WIN), np.uint8)
    order = {3: 0, 1: 1, 2: 2}
    for it in sorted(st.get("items", []), key=lambda i: order.get(i["cls"], 9)):
        p = np.asarray(it.get("pixels", []), np.int64).reshape(-1, 2) - [x0, y0]
        p = p[(p[:, 0] >= 0) & (p[:, 0] < WIN) & (p[:, 1] >= 0) & (p[:, 1] < WIN)]
        out[p[:, 1], p[:, 0]] = it["cls"]
    for b in st.get("boxes", []):
        xa, ya, xb, yb = (int(round(v)) for v in b)
        out[max(0, ya - y0):max(0, min(WIN, yb - y0)), max(0, xa - x0):max(0, min(WIN, xb - x0))] = IGNORE
    out[~e["valid"][y0:y0 + WIN, x0:x0 + WIN]] = IGNORE
    return out


# ── 服务 ─────────────────────────────────────────────────────────────

def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def png_bytes(a: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return buf.tobytes()


def stretch(x: np.ndarray, mode: str) -> np.ndarray:
    """g：全局平方根拉伸（1→16，16→64，255→255）；a：区域内以 99.7 分位为白点的平方根拉伸。"""
    x = x.astype(np.float32)
    hi = 255.0 if mode == "g" else max(float(np.percentile(x, 99.7)), 16.0)
    return (255 * np.sqrt(np.clip(x / hi, 0, 1))).astype(np.uint8)


def pix_list(p: np.ndarray) -> list:
    return np.asarray(p, np.int64).reshape(-1, 2).tolist()


def build_app(token: str) -> FastAPI:
    tasks = json.loads((BENCH / "tasks.json").read_text())["tasks"]
    by_id = {t["wid"]: t for t in tasks}
    state_dir, label_dir = BENCH / "state", BENCH / "labels"
    state_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cache = LensCache()
    io_lock = threading.Lock()

    def task(wid: str) -> dict:
        t = by_id.get(wid)
        if t is None:
            raise HTTPException(404, f"无此窗口 {wid}")
        return t

    def lens_of(body: dict) -> dict:
        return cache.get(task(body["wid"])["lens"])

    def read_state(wid: str) -> dict:
        p = state_dir / f"{wid}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def next_task(after: str | None, split: str | None) -> dict | None:
        seq = [t for t in tasks if t["enabled"] and (split in (None, "", "all") or t["split"] == split)]
        start = 0
        if after:
            ids = [t["wid"] for t in seq]
            start = ids.index(after) + 1 if after in ids else 0
        for t in seq[start:] + seq[:start]:
            if not (state_dir / f"{t['wid']}.json").exists():
                return t
        return None

    def prefetch(wid: str) -> None:
        nxt = next_task(wid, None)
        if nxt:
            threading.Thread(target=lambda: cache.get(nxt["lens"]), daemon=True).start()

    app = FastAPI(title="Microlens_DF 标注")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        q = request.query_params.get("t")
        if not (request.cookies.get("annot_t") == token or q == token):
            return Response("需要访问令牌：请使用启动时打印的完整地址（含 ?t=…）。", status_code=401,
                            media_type="text/plain; charset=utf-8")
        resp = await call_next(request)
        if q == token:
            resp.set_cookie("annot_t", token, max_age=30 * 86400, httponly=True, samesite="lax")
        return resp

    @app.get("/")
    def index():
        return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/progress")
    def progress():
        sts = [json.loads(p.read_text()) for p in state_dir.glob("*.json")]
        done = {s["wid"]: s.get("status") for s in sts}
        rows = {}
        for t in tasks:
            r = rows.setdefault(t["split"], {"total": 0, "done": 0, "skip": 0, "enabled": t["enabled"]})
            r["total"] += 1
            r["done"] += done.get(t["wid"]) == "done"
            r["skip"] += done.get(t["wid"]) == "skip"
        secs = [s.get("seconds", 0) for s in sts if s.get("status") == "done"]
        return {"splits": rows, "median_seconds": float(np.median(secs)) if secs else None}

    @app.get("/api/next")
    def api_next(after: str = "", split: str = ""):
        t = next_task(after or None, split or None)
        return {"wid": t["wid"] if t else None}

    @app.get("/api/prev")
    def api_prev(before: str = ""):
        """按提交时间顺序找上一个已提交窗口，便于回看修改。"""
        sts = sorted(state_dir.glob("*.json"), key=lambda p: json.loads(p.read_text()).get("updated", ""))
        ids = [p.stem for p in sts]
        if before in ids:
            i = ids.index(before)
            return {"wid": ids[i - 1] if i > 0 else None}
        return {"wid": ids[-1] if ids else None}

    @app.get("/api/window/{wid}")
    def window(wid: str):
        t = task(wid)
        e = cache.get(t["lens"])
        reg = region_of(t, e["img"].shape)
        rx0, ry0, rx1, ry1 = reg["x0"], reg["y0"], reg["x0"] + reg["w"], reg["y0"] + reg["h"]
        sugg = [s for s in e["sugg"] if s["bbox"][2] >= rx0 and s["bbox"][0] <= rx1
                and s["bbox"][3] >= ry0 and s["bbox"][1] <= ry1]
        prefetch(wid)
        return {"task": t, "win": WIN, "region": reg, "lens_shape": list(e["img"].shape),
                "sigma": e["sigma"], "suggestions": sugg, "state": read_state(wid)}

    @app.get("/api/window/{wid}/img.png")
    def window_img(wid: str, mode: str = "a"):
        t = task(wid)
        e = cache.get(t["lens"])
        reg = region_of(t, e["img"].shape)
        sl = np.s_[reg["y0"]:reg["y0"] + reg["h"], reg["x0"]:reg["x0"] + reg["w"]]
        g = stretch(e["img"][sl], mode)
        g[~e["valid"][sl]] = 0
        return Response(png_bytes(g), media_type="image/png")

    @app.get("/api/window/{wid}/ctx.png")
    def window_ctx(wid: str):
        """整镜缩略图（1/8），前端在上面画出窗口与长划痕。"""
        e = cache.get(task(wid)["lens"])
        g = stretch(cv2.resize(e["img"], None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA), "g")
        return Response(png_bytes(g), media_type="image/png")

    @app.post("/api/ridge")
    async def api_ridge(request: Request):
        body = await request.json()
        e = lens_of(body)
        pts = body.get("pts") or []
        if len(pts) < 2:
            raise HTTPException(400, "划痕至少需要两个点")
        if body.get("mode") == "line":
            path = snap_straight(e, pts)
        else:
            path = snap_ridge(e, pts, int(np.clip(body.get("corridor", CORRIDOR), 1, 12)))
        return {"path": np.round(path, 1).tolist(), "pixels": pix_list(ridge_region(e, path))}

    @app.post("/api/extend")
    async def api_extend(request: Request):
        """长划痕：从路径两端沿走向追踪，返回延长后的路径与区域。"""
        body = await request.json()
        e = lens_of(body)
        path = np.asarray(body["path"], np.float64).reshape(-1, 2)
        tail = track(e, path)
        head = track(e, path[::-1])
        full = np.concatenate([head[::-1], path, tail]) if len(head) or len(tail) else path
        full = densify(full, 1.0)
        return {"path": np.round(full, 1).tolist(), "pixels": pix_list(ridge_region(e, full)),
                "added_px": int(round(5 * (len(head) + len(tail))))}

    @app.post("/api/path_region")
    async def api_path_region(request: Request):
        """按给定路径（不再吸附）重算划痕区域，用于截断追踪结果之后。"""
        body = await request.json()
        e = lens_of(body)
        path = densify(np.asarray(body["path"], np.float64).reshape(-1, 2), 1.0)
        return {"path": np.round(path, 1).tolist(), "pixels": pix_list(ridge_region(e, path))}

    @app.post("/api/spot")
    async def api_spot(request: Request):
        body = await request.json()
        e = lens_of(body)
        return {"pixels": pix_list(spot_region(e, body["x"], body["y"]))}

    @app.post("/api/fragment")
    async def api_fragment(request: Request):
        body = await request.json()
        e = lens_of(body)
        if len(body.get("poly") or []) < 3:
            raise HTTPException(400, "碎裂区域至少需要三个点")
        return {"pixels": pix_list(fragment_region(e, body["poly"]))}

    @app.post("/api/window/{wid}/save")
    async def save(wid: str, request: Request):
        body = await request.json()
        t = task(wid)
        if body.get("status") not in ("done", "skip"):
            raise HTTPException(400, "status 须为 done 或 skip")
        items = []
        for it in body.get("items", []):
            if it.get("cls") not in (1, 2, 3):
                raise HTTPException(400, f"非法类别 {it.get('cls')}")
            items.append({k: it[k] for k in ("id", "cls", "kind", "src", "sid", "input", "path", "pixels") if k in it})
        st = {"wid": wid, "lens": t["lens"], "split": t["split"], "zone": t["zone"], "x0": t["x0"], "y0": t["y0"],
              "win": WIN, "status": body["status"], "items": items, "boxes": body.get("boxes", []),
              "seconds": round(float(body.get("seconds", 0)), 1), "reviewer": str(body.get("reviewer", ""))[:40],
              "note": str(body.get("note", ""))[:500], "updated": now(), "format": "bench_v1/items",
              "suggestion_source": "trad_v2"}
        with io_lock:
            (state_dir / f"{wid}.json").write_text(json.dumps(st, ensure_ascii=False))
            if st["status"] == "done":
                cv2.imwrite(str(label_dir / f"{wid}.png"), render_label(cache.get(t["lens"]), t, st))
            else:
                (label_dir / f"{wid}.png").unlink(missing_ok=True)
        return {"ok": True}

    # 续接判读页（K0）：静态页 + 服务端保存
    @app.get("/api/links")
    def links_get():
        p = LINKS / "results_server.json"
        return JSONResponse(json.loads(p.read_text()) if p.exists() else {"answers": {}})

    @app.post("/api/links")
    async def links_post(request: Request):
        body = await request.json()
        with io_lock:
            (LINKS / "results_server.json").write_text(json.dumps(
                {"saved": now(), "answers": body.get("answers", {})}, ensure_ascii=False, indent=1))
        return {"ok": True}

    if LINKS.exists():
        app.mount("/links", StaticFiles(directory=str(LINKS), html=True), name="links")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    return app


def cmd_serve(args: argparse.Namespace) -> None:
    import socket

    import uvicorn
    tok_file = BENCH / ".token"
    if args.token:
        token = args.token
    elif tok_file.exists():
        token = tok_file.read_text().strip()
    else:
        token = secrets.token_urlsafe(9)
        tok_file.write_text(token)
    app = build_app(token)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    print(f"标注服务：http://{ip}:{args.port}/?t={token}", flush=True)
    print(f"续接判读：http://{ip}:{args.port}/links/?t={token}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--force", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8500)   # 本机防火墙已放行 8500
    s.add_argument("--token", default="")
    args = ap.parse_args()
    {"init": cmd_init, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
