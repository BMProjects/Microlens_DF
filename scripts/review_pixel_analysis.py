"""像素级标注实验的分析：全局坐标、跨象限拼接、栅格化、吸附精修、三助手异同与传统参考。

由 review_pixel_pilot.py analyze 调用；结果写入 <OUT>/analysis/。
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_pixel_pilot import INNER, PIXEL_MM, QUADS, RATERS, load_standard  # noqa: E402

H, W = 3000, 4096
TOL = 6          # 中心线一致性容差（像素，约 0.04 mm）
LINK_DIST = 25   # 跨象限拼接：端点到对方折线的最大距离
LINK_ANGLE = 25  # 跨象限拼接：走向最大夹角（度）
EDGE_NEAR = 12   # 端点距象限内侧边界在该距离内视为“到达边界”


# ── 几何工具 ──────────────────────────────────────────────────────────

def poly_len(pts: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0


def densify(pts: np.ndarray, step: float = 2.0) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out, np.float32)


def end_dir(pts: np.ndarray, at_start: bool) -> np.ndarray:
    seg = pts[: min(len(pts), 4)] if at_start else pts[-min(len(pts), 4):]
    v = (seg[0] - seg[-1]) if at_start else (seg[-1] - seg[0])  # 指向折线外侧
    n = np.hypot(*v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def point_to_poly(p: np.ndarray, pts: np.ndarray) -> float:
    d = densify(pts, 3.0)
    return float(np.min(np.hypot(d[:, 0] - p[0], d[:, 1] - p[1])))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(float(np.dot(u, v)))  # 走向不区分正反
    return math.degrees(math.acos(min(1.0, c)))


# ── 读取与拼接 ────────────────────────────────────────────────────────

def load_rater(out: Path, key: dict, rater: str) -> dict:
    """返回 {镜片: {"scratches": [...], "spots": [...], "damage": [...], "status": {...}}}，坐标为全局。"""
    res: dict = {}
    for pid, info in key.items():
        lens, q = info["lens"], info["quadrant"]
        x0, y0 = QUADS[q][:2]
        d = res.setdefault(lens, {"scratches": [], "spots": [], "damage": [], "status": {}, "notes": {}})
        f = out / "ratings" / f"{rater}_{pid}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"error": "未运行"}
        p = rec.get("parsed")
        d["status"][q] = "ok" if p else (rec.get("error") or rec.get("parse_error") or "无结果")[:120]
        d.setdefault("seconds", {})[q] = rec.get("seconds")
        d.setdefault("cost", {})[q] = rec.get("cost_usd")
        if not p:
            continue
        d["notes"][q] = p.get("notes", "")
        for s in p.get("scratches") or []:
            try:
                pts = np.array(s["points"], np.float32).reshape(-1, 2)
            except (KeyError, ValueError, TypeError):
                continue
            if len(pts) < 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, QUADS[q][2] - x0 - 1) + x0
            pts[:, 1] = np.clip(pts[:, 1], 0, QUADS[q][3] - y0 - 1) + y0
            d["scratches"].append({"q": q, "pts": pts, "w": float(s.get("width_px") or 3),
                                   "exits": list(s.get("exits") or [])})
        for s in p.get("spots") or []:
            try:
                cx, cy = s["center"]
                d["spots"].append({"q": q, "c": (float(cx) + x0, float(cy) + y0), "r": float(s.get("r_px") or 4)})
            except (KeyError, ValueError, TypeError):
                continue
        for s in p.get("damage") or []:
            try:
                poly = np.array(s["polygon"], np.float32).reshape(-1, 2) + np.array([x0, y0], np.float32)
            except (KeyError, ValueError, TypeError):
                continue
            if len(poly) >= 3:
                d["damage"].append({"q": q, "poly": poly})
    return res


def inner_edge_coord(q: str, edge: str) -> tuple[str, float]:
    x0, y0, x1, y1 = QUADS[q]
    return {"right": ("x", x1 - 1), "left": ("x", x0), "bottom": ("y", y1 - 1), "top": ("y", y0)}[edge]


def border_ends(s: dict) -> list[tuple[bool, str]]:
    """划痕在象限内侧边界上的端点：[(是否起点, 边)]。"""
    out = []
    for at_start in (True, False):
        p = s["pts"][0] if at_start else s["pts"][-1]
        for edge in INNER[s["q"]]:
            axis, v = inner_edge_coord(s["q"], edge)
            if abs((p[0] if axis == "x" else p[1]) - v) <= EDGE_NEAR:
                out.append((at_start, edge))
    return out


NEIGHBOR = {("q1", "right"): "q2", ("q2", "left"): "q1", ("q3", "right"): "q4", ("q4", "left"): "q3",
            ("q1", "bottom"): "q3", ("q3", "top"): "q1", ("q2", "bottom"): "q4", ("q4", "top"): "q2"}


def stitch(scr: list[dict]) -> tuple[list[list[int]], dict]:
    """跨象限连续性检测：在重叠带内按位置与走向配对，返回拼接后的划痕分组与统计。"""
    parent = list(range(len(scr)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends = [(i, st, e) for i, s in enumerate(scr) for st, e in border_ends(s)]
    used: set = set()
    cands = []
    for i, st, e in ends:
        nq = NEIGHBOR[(scr[i]["q"], e)]
        p = scr[i]["pts"][0] if st else scr[i]["pts"][-1]
        u = end_dir(scr[i]["pts"], st)
        for j, s2 in enumerate(scr):
            if s2["q"] != nq:
                continue
            dist = point_to_poly(p, s2["pts"])
            if dist > LINK_DIST:
                continue
            # 对方折线在重叠带内的局部走向
            dd = densify(s2["pts"], 3.0)
            k = int(np.argmin(np.hypot(dd[:, 0] - p[0], dd[:, 1] - p[1])))
            a, b = dd[max(0, k - 5)], dd[min(len(dd) - 1, k + 5)]
            v = (b - a) / max(np.hypot(*(b - a)), 1e-6)
            ang = angle_between(u, v)
            if ang <= LINK_ANGLE:
                cands.append((dist + ang, i, st, j))
    cands.sort()
    linked_ends = set()
    for _, i, st, j in cands:
        if (i, st) in linked_ends or (i, j) in used:
            continue
        linked_ends.add((i, st))
        used.add((i, j)); used.add((j, i))
        parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(scr)):
        groups.setdefault(find(i), []).append(i)
    n_border = len(ends)
    n_linked = len(linked_ends)
    stats = {"border_endpoints": n_border, "linked_endpoints": n_linked,
             "unmatched_endpoints": n_border - n_linked,
             "continuity_rate": round(n_linked / n_border, 3) if n_border else None,
             "fragments": len(scr), "stitched_scratches": len(groups)}
    unmatched = [(i, st) for i, st, _ in ends if (i, st) not in linked_ends]
    stats["_unmatched"] = unmatched
    return list(groups.values()), stats


# ── 吸附精修与栅格化 ──────────────────────────────────────────────────

def snap(pts: np.ndarray, img_s: np.ndarray, r: int = 6) -> np.ndarray:
    """沿法向在 ±r 像素内移到亮度脊线（类似标注工具的磁性吸附），再沿路径平滑。"""
    d = densify(pts, 2.0)
    if len(d) < 3:
        return d
    tang = np.gradient(d, axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-6)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    offs = np.arange(-r, r + 1, dtype=np.float32)
    xs = d[:, 0:1] + nrm[:, 0:1] * offs
    ys = d[:, 1:2] + nrm[:, 1:2] * offs
    xi = np.clip(np.rint(xs).astype(int), 0, W - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, H - 1)
    vals = img_s[yi, xi]
    best = offs[np.argmax(vals, axis=1)]
    k = 7
    best = np.convolve(np.pad(best, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return d + nrm * best[:, None]


def raster_lines(lines: list[np.ndarray], thickness: int = 1) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for pts in lines:
        cv2.polylines(m, [np.rint(pts).astype(np.int32)], False, 1, thickness)
    return m > 0


def masks_for(d: dict, img_s: np.ndarray, snapped: bool) -> dict:
    lines = [snap(s["pts"], img_s) if snapped else s["pts"] for s in d["scratches"]]
    scratch = np.zeros((H, W), np.uint8)
    for s, pts in zip(d["scratches"], lines):
        cv2.polylines(scratch, [np.rint(pts).astype(np.int32)], False, 1, max(1, int(round(s["w"]))))
    spot = np.zeros((H, W), np.uint8)
    for s in d["spots"]:
        cv2.circle(spot, (int(round(s["c"][0])), int(round(s["c"][1]))), max(1, int(round(s["r"]))), 1, -1)
    dmg = np.zeros((H, W), np.uint8)
    for s in d["damage"]:
        cv2.fillPoly(dmg, [np.rint(s["poly"]).astype(np.int32)], 1)
    return {"center": raster_lines(lines), "lines": lines, "scratch": scratch > 0, "spot": spot > 0, "damage": dmg > 0}


def center_f1(a: np.ndarray, b: np.ndarray) -> dict:
    """中心线一致性：A 的中心线像素有多少落在 B 中心线 TOL 像素内，反之亦然。"""
    if not a.any() and not b.any():
        return {"p": None, "r": None, "f1": None}
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((db[a] <= TOL).mean()) if a.any() else 0.0
    r = float((da[b] <= TOL).mean()) if b.any() else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3)}


def spot_f1(a: list, b: list) -> float | None:
    if not a and not b:
        return None
    used, tp = set(), 0
    for s in a:
        best, bj = None, None
        for j, t in enumerate(b):
            if j in used:
                continue
            dist = math.hypot(s["c"][0] - t["c"][0], s["c"][1] - t["c"][1])
            if dist <= max(s["r"], t["r"]) + TOL and (best is None or dist < best):
                best, bj = dist, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return round(2 * tp / (len(a) + len(b)), 3)


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = (a | b).sum()
    return None if u == 0 else round(float((a & b).sum() / u), 3)


def zone_map() -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(xx - W // 2, yy - H // 2) / (min(H, W) * 0.43)
    return np.where(r < 0.40, 0, np.where(r < 0.75, 1, 2))


def crossings(lines: list[np.ndarray], groups: list[list[int]]) -> int:
    """不同（拼接后）划痕之间的交叉点数：各自 1 像素中心线叠加后，重叠连通块计数。"""
    count = np.zeros((H, W), np.uint8)
    for g in groups:
        m = np.zeros((H, W), np.uint8)
        for i in g:
            cv2.polylines(m, [np.rint(lines[i]).astype(np.int32)], False, 1, 3)
        count += m
    n, _ = cv2.connectedComponents((count >= 2).astype(np.uint8))
    return n - 1


# ── 主流程 ────────────────────────────────────────────────────────────

def analyze(out: Path) -> None:
    from review_pilot import traditional_segment  # 传统参考沿用试点实现
    from skimage.morphology import skeletonize

    key = json.loads((out / "key.json").read_text())
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    lenses = sorted({v["lens"] for v in key.values()}, key=["108l", "26l", "90r", "76r", "72l"].index)
    data = {r: load_rater(out, key, r) for r in RATERS}
    zones = zone_map()
    report: dict = {"lenses": {}, "config": {"tol_px": TOL, "link_dist_px": LINK_DIST, "link_angle_deg": LINK_ANGLE}}
    for lens in lenses:
        img, valid = load_standard(lens)
        img_s = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
        trad = traditional_segment(img, valid)
        trad_fg = (trad["scratch"] | trad["spot"] | trad["damage"]) & valid
        trad_skel = skeletonize(trad["scratch"] & valid)
        row: dict = {"raters": {}, "pairs": {}}
        cache: dict = {}
        for r in RATERS:
            d = data[r][lens]
            groups, st = stitch(d["scratches"])
            unmatched = st.pop("_unmatched")
            m_raw = masks_for(d, img_s, snapped=False)
            m_snap = masks_for(d, img_s, snapped=True)
            disp = [float(np.median(np.hypot(*(snap(s["pts"], img_s) - densify(s["pts"], 2.0)).T)))
                    for s in d["scratches"][:400]] if d["scratches"] else []
            lengths = [sum(poly_len(m_snap["lines"][i]) for i in g) for g in groups]
            zl = [round(float(m_snap["center"][zones == z].sum()) * PIXEL_MM, 2) for z in range(3)]
            dt_fg = cv2.distanceTransform((~trad_fg).astype(np.uint8), cv2.DIST_L2, 3)
            dt_c = cv2.distanceTransform((~m_snap["center"]).astype(np.uint8), cv2.DIST_L2, 3)
            row["raters"][r] = {
                "status": d["status"], "seconds": d.get("seconds"), "cost": d.get("cost"), "notes": d["notes"],
                "fragments": st["fragments"], "stitched_scratches": st["stitched_scratches"],
                "border_endpoints": st["border_endpoints"], "linked_endpoints": st["linked_endpoints"],
                "unmatched_endpoints": st["unmatched_endpoints"], "continuity_rate": st["continuity_rate"],
                "spots": len(d["spots"]), "damage_regions": len(d["damage"]),
                "damage_area_mm2": round(float(m_snap["damage"].sum()) * PIXEL_MM ** 2, 2),
                "scratch_length_mm": round(float(sum(lengths)) * PIXEL_MM, 1),
                "median_scratch_len_mm": round(float(np.median(lengths)) * PIXEL_MM, 2) if lengths else None,
                "max_scratch_len_mm": round(float(max(lengths)) * PIXEL_MM, 2) if lengths else None,
                "zone_length_mm": dict(zip("CME", zl)),
                "crossings": crossings(m_snap["lines"], groups),
                "snap_shift_median_px": round(float(np.median(disp)), 2) if disp else None,
                "trad_support": round(float((dt_fg[m_snap["center"]] <= 4).mean()), 3) if m_snap["center"].any() else None,
                "trad_coverage": round(float((dt_c[trad_skel] <= TOL).mean()), 3) if trad_skel.any() else None,
            }
            cache[r] = {"raw": m_raw, "snap": m_snap, "groups": groups, "unmatched": unmatched, "d": d}
        for a, b in itertools.combinations(RATERS, 2):
            row["pairs"][f"{a}-{b}"] = {
                "center_raw": center_f1(cache[a]["raw"]["center"], cache[b]["raw"]["center"]),
                "center_snap": center_f1(cache[a]["snap"]["center"], cache[b]["snap"]["center"]),
                "spot_f1": spot_f1(cache[a]["d"]["spots"], cache[b]["d"]["spots"]),
                "damage_iou": iou(cache[a]["snap"]["damage"], cache[b]["snap"]["damage"]),
                "scratch_mask_iou": iou(cache[a]["snap"]["scratch"], cache[b]["snap"]["scratch"]),
            }
        # 三人共识：各自中心线外扩 TOL 后计票
        votes = np.zeros((H, W), np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))
        for r in RATERS:
            votes += cv2.dilate(cache[r]["snap"]["center"].astype(np.uint8), k)
        row["consensus"] = {r: {f"in_{n}": round(float((votes[cache[r]["snap"]["center"]] == n).mean()), 3)
                                for n in (1, 2, 3)} if cache[r]["snap"]["center"].any() else None
                            for r in RATERS}
        report["lenses"][lens] = row
        np.savez_compressed(adir / f"{lens}.npz", votes=votes,
                            trad=trad_fg, **{f"{r}_center": cache[r]["snap"]["center"] for r in RATERS},
                            **{f"{r}_damage": cache[r]["snap"]["damage"] for r in RATERS},
                            **{f"{r}_spot": cache[r]["snap"]["spot"] for r in RATERS})
        geo = {r: {"lines": [ln.tolist() for ln in cache[r]["snap"]["lines"]],
                   "raw": [s["pts"].tolist() for s in cache[r]["d"]["scratches"]],
                   "groups": cache[r]["groups"], "unmatched": cache[r]["unmatched"],
                   "spots": [{"c": s["c"], "r": s["r"]} for s in cache[r]["d"]["spots"]],
                   "damage": [s["poly"].tolist() for s in cache[r]["d"]["damage"]]} for r in RATERS}
        (adir / f"{lens}_geometry.json").write_text(json.dumps(geo))
        print(lens, {r: (row["raters"][r]["stitched_scratches"], row["raters"][r]["continuity_rate"])
                     for r in RATERS}, flush=True)
    (adir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"分析结果: {adir}")
