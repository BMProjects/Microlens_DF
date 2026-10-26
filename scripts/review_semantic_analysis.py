"""语义判定实验的分析：跨窗口合并、像素归属、三人异同、共识与传统规则对比。

由 review_semantic_pilot.py analyze 调用；结果写入 <OUT>/analysis/。
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

CLASSES = ["scratch", "spot", "damage", "not_defect"]
CID = {c: i + 1 for i, c in enumerate(CLASSES)}   # 0 无前景；5 未判定
UNJUDGED = 5
PIXEL_MM = 0.0068
RATERS = ["R1", "R2", "R3"]


class UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def kappa(a: list, b: list) -> float | None:
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def adjacency(seg: np.ndarray, ids: set) -> set:
    """区域内相互接触（2 像素内）的已判定片段对。"""
    pairs = set()
    dil = ndi.grey_dilation(seg, size=5)
    ero = -ndi.grey_dilation(-np.where(seg > 0, seg, np.iinfo(np.int32).max), size=5)
    m = (seg > 0) & (dil != seg)
    for a, b in np.unique(np.stack([seg[m], dil[m]], 1), axis=0):
        if a != b and a in ids and b in ids and b > 0:
            pairs.add((min(a, b), max(a, b)))
    m2 = (seg > 0) & (ero != seg) & (ero != np.iinfo(np.int32).max)
    for a, b in np.unique(np.stack([seg[m2], ero[m2]], 1), axis=0):
        if a != b and a in ids and b in ids and b > 0:
            pairs.add((min(a, b), max(a, b)))
    return pairs


def rater_result(out: Path, key: dict, lens: str, rater: str, shown_all: set, origin_region) -> dict:
    labels: dict = {}
    uf = UF()
    wrong, polys, missed, invalid, notes, secs, costs, status = set(), [], [], 0, {}, [], [], {}
    for pid, info in key.items():
        if info["lens"] != lens:
            continue
        f = out / "ratings" / f"{rater}_{pid}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"error": "未运行"}
        p = rec.get("parsed")
        status[info["window"]] = "ok" if p else (rec.get("error") or rec.get("parse_error") or "无结果")[:100]
        secs.append(rec.get("seconds") or 0)
        costs.append(rec.get("cost_usd") or 0)
        if not p:
            continue
        notes[info["window"]] = p.get("notes", "")
        shown = set(info["ids"])
        ox, oy = info["origin"]
        for it in p.get("labels") or []:
            try:
                i, c = int(it["id"]), str(it["class"])
            except (KeyError, ValueError, TypeError):
                continue
            if i not in shown or c not in CID:
                invalid += 1
                continue
            labels.setdefault(i, []).append(c)
        for grp in p.get("same_object") or []:
            g = [int(x) for x in grp if str(x).lstrip("-").isdigit() and int(x) in shown]
            for a, b in zip(g, g[1:]):
                uf.union(a, b)
        wrong |= {int(x) for x in p.get("wrong_merge") or [] if str(x).isdigit() and int(x) in shown}
        for poly in p.get("damage_polygons") or []:
            try:
                arr = np.array(poly, np.float32).reshape(-1, 2) + [ox - origin_region[0], oy - origin_region[1]]
                if len(arr) >= 3:
                    polys.append(arr)
            except (ValueError, TypeError):
                continue
        missed += [m for m in p.get("missed") or [] if isinstance(m, dict)]
    multi = {i: c for i, c in labels.items() if len(c) > 1}
    conflicts = sum(1 for c in multi.values() if len(set(c)) > 1)
    final = {i: Counter(c).most_common(1)[0][0] for i, c in labels.items()}
    return {"labels": final, "uf": uf, "wrong": wrong, "polys": polys, "missed": missed, "invalid": invalid,
            "notes": notes, "seconds": round(sum(secs), 1), "cost": round(sum(costs), 2), "status": status,
            "multi_window_ids": len(multi), "window_conflicts": conflicts,
            "unlabeled_shown": len(shown_all - set(final))}


def pixel_maps(seg_r: np.ndarray, res: dict) -> tuple[np.ndarray, np.ndarray]:
    """区域内像素类别图与实例图：已判定片段按其类别/实例；灰色碎片在碎裂多边形内归碎裂，否则就近继承。"""
    cls = np.zeros(seg_r.shape, np.uint8)
    inst = np.zeros(seg_r.shape, np.int32)
    lab = res["labels"]
    root_id: dict = {}
    for i in lab:
        r = res["uf"].find(i)
        root_id.setdefault(r, len(root_id) + 1)
    ids = np.unique(seg_r)
    lut_c = np.zeros(int(seg_r.max()) + 1, np.uint8)
    lut_i = np.zeros(int(seg_r.max()) + 1, np.int32)
    for i in ids:
        if i > 0 and int(i) in lab:
            lut_c[i] = CID[lab[int(i)]]
            lut_i[i] = root_id[res["uf"].find(int(i))]
    cls = lut_c[seg_r]
    inst = lut_i[seg_r]
    judged = cls > 0
    fg = seg_r > 0
    dmg = np.zeros(seg_r.shape, np.uint8)
    for poly in res["polys"]:
        cv2.fillPoly(dmg, [np.rint(poly).astype(np.int32)], 1)
    rest = fg & ~judged
    cls[rest & (dmg > 0)] = CID["damage"]
    rest &= dmg == 0
    if judged.any() and rest.any():
        dist, (iy, ix) = ndi.distance_transform_edt(~judged, return_indices=True)
        near = rest & (dist <= 3)
        cls[near] = cls[iy[near], ix[near]]
        inst[near] = inst[iy[near], ix[near]]
        rest &= ~near
    cls[rest] = UNJUDGED
    return cls, inst


def analyze(out: Path) -> None:
    key = json.loads((out / "key.json").read_text())
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    lenses = ["108l", "26l", "90r", "76r", "72l"]
    report: dict = {"lenses": {}, "overall": {}}
    pooled = {p: {"a": [], "b": []} for p in itertools.combinations(RATERS, 2)}
    for lens in lenses:
        meta = json.loads((out / "seg" / f"{lens}_info.json").read_text())
        rx, ry, size = meta["region"]
        seg = np.load(out / "seg" / f"{lens}.npz")["seg"]
        seg_r = seg[ry:ry + size, rx:rx + size]
        info = meta["info"]
        shown_all = set().union(*[set(v["ids"]) for v in key.values() if v["lens"] == lens])
        res = {r: rater_result(out, key, lens, r, shown_all, (rx, ry)) for r in RATERS}
        adj = adjacency(seg_r, shown_all)
        row: dict = {"n_shown": len(shown_all), "n_adjacent_pairs": len(adj), "raters": {}, "pairs": {}}
        maps = {}
        for r in RATERS:
            cls, inst = pixel_maps(seg_r, res[r])
            maps[r] = (cls, inst)
            lab = res[r]["labels"]
            cnt = Counter(lab.values())
            objs: dict = {}
            for i, c in lab.items():
                objs.setdefault(res[r]["uf"].find(i), []).append(c)
            obj_cls = Counter(Counter(v).most_common(1)[0][0] for v in objs.values())
            fgpix = seg_r > 0
            row["raters"][r] = {
                "status": res[r]["status"], "seconds": res[r]["seconds"], "cost": res[r]["cost"],
                "notes": res[r]["notes"], "labeled": len(lab), "unlabeled_shown": res[r]["unlabeled_shown"],
                "invalid_items": res[r]["invalid"], "class_counts": dict(cnt),
                "objects": dict(obj_cls), "merged_objects": sum(1 for v in objs.values() if len(v) > 1),
                "wrong_merge": len(res[r]["wrong"]), "damage_polygons": len(res[r]["polys"]),
                "missed": len(res[r]["missed"]),
                "multi_window_ids": res[r]["multi_window_ids"], "window_conflicts": res[r]["window_conflicts"],
                "pixel_share": {c: round(float((cls[fgpix] == CID[c]).mean()), 3) for c in CLASSES}
                | {"unjudged": round(float((cls[fgpix] == UNJUDGED).mean()), 3)},
                "damage_area_mm2": round(float((cls == CID["damage"]).sum()) * PIXEL_MM ** 2, 3),
            }
            # 与传统规则对比：规则把笔画判为划痕、紧凑小块判为斑点
            rule = {i: ("spot" if info[str(i)]["kind"] == "spot" else "scratch") for i in lab}
            same = sum(lab[i] == rule[i] for i in lab)
            row["raters"][r]["agree_with_rule"] = round(same / len(lab), 3) if lab else None
        for a, b in itertools.combinations(RATERS, 2):
            la, lb = res[a]["labels"], res[b]["labels"]
            common = sorted(set(la) & set(lb))
            xa, xb = [la[i] for i in common], [lb[i] for i in common]
            pooled[(a, b)]["a"] += xa
            pooled[(a, b)]["b"] += xb
            both = set(common)
            adj_ab = [p for p in adj if p[0] in both and p[1] in both]  # 只比较两人都判定过的片段对
            same_a = {p for p in adj_ab if res[a]["uf"].find(p[0]) == res[a]["uf"].find(p[1])}
            same_b = {p for p in adj_ab if res[b]["uf"].find(p[0]) == res[b]["uf"].find(p[1])}
            ga = [p in same_a for p in adj_ab]
            gb = [p in same_b for p in adj_ab]
            fg = seg_r > 0
            ca, cb = maps[a][0][fg], maps[b][0][fg]
            row["pairs"][f"{a}-{b}"] = {
                "class_agree": round(sum(x == y for x, y in zip(xa, xb)) / len(common), 3) if common else None,
                "class_kappa": kappa(xa, xb),
                "group_same_jaccard": round(len(same_a & same_b) / len(same_a | same_b), 3) if (same_a | same_b) else None,
                "group_kappa": kappa(ga, gb),
                "wrong_merge_jaccard": (round(len(res[a]["wrong"] & res[b]["wrong"]) / len(res[a]["wrong"] | res[b]["wrong"]), 3)
                                        if (res[a]["wrong"] | res[b]["wrong"]) else None),
                "pixel_class_agree": round(float((ca == cb).mean()), 3),
                "pixel_iou": {c: (round(float(((ca == CID[c]) & (cb == CID[c])).sum() / max(((ca == CID[c]) | (cb == CID[c])).sum(), 1)), 3)
                                  if ((ca == CID[c]) | (cb == CID[c])).any() else None) for c in CLASSES},
            }
        # 共识：每个片段 ≥2 人一致的类别
        cons, disputed = {}, []
        for i in shown_all:
            votes = [res[r]["labels"].get(i) for r in RATERS]
            c = Counter(v for v in votes if v)
            if c and c.most_common(1)[0][1] >= 2:
                cons[i] = c.most_common(1)[0][0]
            else:
                disputed.append(i)
        row["consensus"] = {"ids_with_majority": len(cons), "disputed": len(disputed),
                            "majority_counts": dict(Counter(cons.values())),
                            "unanimous": sum(1 for i in shown_all if len({res[r]['labels'].get(i) for r in RATERS}) == 1
                                             and res['R1']['labels'].get(i) is not None)}
        np.savez_compressed(adir / f"{lens}.npz", seg_r=seg_r,
                            **{f"{r}_cls": maps[r][0] for r in RATERS}, **{f"{r}_inst": maps[r][1] for r in RATERS})
        (adir / f"{lens}_labels.json").write_text(json.dumps(
            {"consensus": {str(k): v for k, v in cons.items()}, "disputed": disputed,
             **{r: {"labels": {str(k): v for k, v in res[r]["labels"].items()},
                    "wrong": sorted(res[r]["wrong"]),
                    "polys": [p.tolist() for p in res[r]["polys"]], "missed": res[r]["missed"]} for r in RATERS}}))
        report["lenses"][lens] = row
        print(lens, {p: (v["class_kappa"], v["group_kappa"]) for p, v in row["pairs"].items()}, flush=True)
    report["overall"] = {f"{a}-{b}": {"n": len(v["a"]), "class_agree": round(sum(x == y for x, y in zip(v["a"], v["b"])) / len(v["a"]), 3) if v["a"] else None,
                                      "class_kappa": kappa(v["a"], v["b"])} for (a, b), v in pooled.items()}
    (adir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("overall", report["overall"])
