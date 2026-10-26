"""传统基础标注（trad_v1）与前几轮实验的对比分析，结果写入 output/annotation/trad_v1/compare/report.json。

对比项：
1. 像素级试点（pixel5，26l 右上象限）：三名助手确认的可见划痕中心线上，各方法前景的召回率与前景占比；
   方法划痕骨架落在助手划痕 4 像素内的比例（“助手支持率”，助手本身有漏标，只作参考）。
2. 语义试点（semantic5，5 张镜片各一块 736×736 区域）：
   - 类别：多数意见类别 vs trad_v1 在该片段像素上的多数类别；
   - 归组：相邻片段对上，≥2 名助手判“同一缺陷”/“不同缺陷”时，trad_v1 是否归入同一实例；
   - 错误连接：≥2 名助手判为“被错误连在一起”的片段，trad_v1 是否已拆开；
   - 覆盖：semantic5 中未得到语义归属的前景像素占比 vs trad_v1 待确认像素占比；
   - 碎裂：72l 中 R2 圈出的碎裂区与 trad_v1 碎裂区的重叠。
3. 网格试点（pilot5）：三助手多数格子 vs 旧传统规则 / trad_v1 的格子召回与精确率。
4. 第一阶段框标注：框命中召回、框内像素占比（沿用 seg_p1_classical_baselines.evaluate）。

用法:
    uv run python scripts/annot_compare.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from review_pixel_analysis import densify  # noqa: E402
from review_pixel_pilot import QUADS as PIX_QUADS  # noqa: E402
from review_pixel_pilot import load_standard  # noqa: E402

ANN = PROJECT_ROOT / "output/annotation/trad_v2"
OUT = ANN / "compare"
SEM = PROJECT_ROOT / "output/review/semantic5"
PIX = PROJECT_ROOT / "output/review/pixel5"
PIL = PROJECT_ROOT / "output/review/pilot5"
LENSES = ["108l", "26l", "90r", "76r", "72l"]
TRAD_NAME = {0: "background", 1: "scratch", 2: "spot", 3: "damage", 4: "uncertain"}
RATERS = ["R1", "R2", "R3"]


def load_trad(lens: str) -> tuple[np.ndarray, np.ndarray]:
    cls = cv2.imread(str(ANN / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)
    inst = cv2.imread(str(ANN / f"{lens}_inst.png"), cv2.IMREAD_UNCHANGED)
    if cls is None or inst is None:
        raise FileNotFoundError(f"缺少 trad_v1 结果: {lens}")
    return cls, inst.astype(np.int32)


def old_traditional(lens: str) -> np.ndarray:
    from review_pilot import traditional_segment
    img, valid = load_standard(lens)
    t = traditional_segment(img, valid)
    return (t["scratch"] | t["spot"] | t["damage"]) & valid


# ── 1. 像素级试点：26l 右上象限 ───────────────────────────────────────

def pixel_pilot() -> dict:
    lens, q = "26l", "q2"
    pid = next(k for k, v in json.loads((PIX / "key.json").read_text()).items() if v == {"lens": lens, "quadrant": q})
    x0, y0, x1, y1 = PIX_QUADS[q]
    img, valid = load_standard(lens)
    qb = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    ref, all_lines = [], []
    for r in RATERS:
        for s in json.loads((PIX / "ratings" / f"{r}_{pid}.json").read_text())["parsed"]["scratches"]:
            d = densify(np.array(s["points"], np.float32), 2.0) + [x0, y0]
            all_lines.append(np.rint(d).astype(np.int32))
            # 只保留中心线两侧确有亮脊的划痕（助手坐标偶有偏移或虚构）
            t = np.gradient(d, axis=0)
            t /= np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-6)
            n = np.stack([-t[:, 1], t[:, 0]], 1)

            def at(o):
                pt = np.clip(np.rint(d + n * o).astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
                return qb[pt[:, 1], pt[:, 0]]
            if np.max([at(o) for o in range(-3, 4)], axis=0).mean() > np.maximum(at(-8), at(8)).mean() + 1.0:
                ref.append(np.clip(np.rint(d).astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1]))
    ref = np.vstack(ref)
    ai = np.zeros(img.shape, np.uint8)
    for ln in all_lines:
        cv2.polylines(ai, [ln.reshape(-1, 1, 2)], False, 1, 1)
    dt_ai = cv2.distanceTransform((1 - ai).astype(np.uint8), cv2.DIST_L2, 3)
    cls, _ = load_trad(lens)
    hr = np.load(SEM / "seg" / f"{lens}.npz")["fg"]
    methods = {"trad_v1": cls > 0, "trad_v1_scratch": np.isin(cls, (1, 4)),
               "hr_segment": hr, "pilot_traditional": old_traditional(lens)}
    qv = valid[y0:y1, x0:x1]
    out = {"n_ref_points": int(len(ref)), "quadrant": [x0, y0, x1, y1], "methods": {}}
    rq = ref[(ref[:, 0] >= x0) & (ref[:, 0] < x1) & (ref[:, 1] >= y0) & (ref[:, 1] < y1)] - [x0, y0]
    for name, m in methods.items():
        dt = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
        sk = skeletonize(m[y0:y1, x0:x1])
        dsk = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 3)
        out["methods"][name] = {
            "ridge_recall": round(float((dt[ref[:, 1], ref[:, 0]] <= 3).mean()), 3),
            # 与掩膜宽度无关的口径：助手中心线到方法前景骨架的距离
            "center_recall": round(float((dsk[rq[:, 1], rq[:, 0]] <= 3).mean()), 3),
            "fg_frac": round(float(m[y0:y1, x0:x1][qv].mean()), 4),
            "ai_support": round(float((dt_ai[y0:y1, x0:x1][sk] <= 4).mean()), 3) if sk.any() else None,
        }
    return out


# ── 2. 语义试点 ───────────────────────────────────────────────────────

def semantic_pilot() -> dict:
    from review_semantic_analysis import adjacency, rater_result
    key = json.loads((SEM / "key.json").read_text())
    rep = json.loads((SEM / "analysis" / "report.json").read_text())
    out: dict = {"lenses": {}}
    pooled_conf: Counter = Counter()
    pooled = {"same_total": 0, "same_hit": 0, "diff_total": 0, "diff_hit": 0, "wrong_total": 0, "wrong_split": 0}
    for lens in LENSES:
        meta = json.loads((SEM / "seg" / f"{lens}_info.json").read_text())
        rx, ry, size = meta["region"]
        z = np.load(SEM / "seg" / f"{lens}.npz")
        seg_r = z["seg"][ry:ry + size, rx:rx + size]
        labels = json.loads((SEM / "analysis" / f"{lens}_labels.json").read_text())
        cons = {int(k): v for k, v in labels["consensus"].items()}
        cls, inst = load_trad(lens)
        cls_r, inst_r = cls[ry:ry + size, rx:rx + size], inst[ry:ry + size, rx:rx + size]
        img_r = load_standard(lens)[0][ry:ry + size, rx:rx + size]
        # 片段 → trad 多数类别 / 多数实例
        seg_cls, seg_inst, seg_major = {}, {}, {}
        objs = ndi.find_objects(seg_r)
        # 与掩膜宽度无关的口径：HR 片段按 2σ 分割、比 trad 掩膜宽，故看片段骨架（斑点看最亮点）3 像素内
        # 是否有 trad 前景，类别与实例取最近的 trad 前景像素
        dist_t, (ny, nx) = ndi.distance_transform_edt(cls_r == 0, return_indices=True)
        for i in np.unique(seg_r):
            if i == 0 or objs[i - 1] is None:
                continue
            sl = objs[i - 1]
            m = seg_r[sl] == i
            sk = skeletonize(m) if m.sum() > 2 else m
            pk = np.unravel_index(int(np.argmax(np.where(m, img_r[sl].astype(np.int32), -1))), m.shape)
            yy, xx = np.nonzero(sk | (np.arange(m.size).reshape(m.shape) == np.ravel_multi_index(pk, m.shape)))
            yy, xx = yy + sl[0].start, xx + sl[1].start
            near = dist_t[yy, xx] <= 3
            gy, gx = pk[0] + sl[0].start, pk[1] + sl[1].start
            if near.mean() < 0.3 and dist_t[gy, gx] > 3:
                seg_cls[int(i)] = "background"
                seg_inst[int(i)] = 0
                continue
            c = cls_r[ny[yy[near], xx[near]], nx[yy[near], xx[near]]]
            seg_cls[int(i)] = TRAD_NAME[int(np.bincount(c).argmax())]
            ii = inst_r[ny[yy[near], xx[near]], nx[yy[near], xx[near]]]
            seg_inst[int(i)] = int(Counter(ii[ii > 0].tolist()).most_common(1)[0][0]) if (ii > 0).any() else 0
            cnt = Counter(ii[ii > 0].tolist())
            seg_major[int(i)] = {k for k, v in cnt.items() if v >= 0.25 * max(1, sum(cnt.values()))}
        conf = Counter((cons[i], seg_cls.get(i, "background")) for i in cons)
        pooled_conf.update(conf)
        # 归组
        shown_all = set().union(*[set(v["ids"]) for v in key.values() if v["lens"] == lens])
        res = {r: rater_result(SEM, key, lens, r, shown_all, (rx, ry)) for r in RATERS}
        adj = adjacency(seg_r, shown_all)
        same, diff = [], []
        for a, b in adj:
            votes = [res[r]["uf"].find(a) == res[r]["uf"].find(b)
                     for r in RATERS if a in res[r]["labels"] and b in res[r]["labels"]]
            if sum(votes) >= 2:
                same.append((a, b))
            elif len(votes) - sum(votes) >= 2:
                diff.append((a, b))
        tr_same = lambda p: seg_inst.get(p[0], 0) > 0 and seg_inst.get(p[0]) == seg_inst.get(p[1])  # noqa: E731
        # 宽松口径：两片段各自占比 ≥25% 的 trad 实例有交集即算同一实例（HR 片段常跨两个 trad 实例）
        tr_share = lambda p: bool(seg_major.get(p[0], set()) & seg_major.get(p[1], set()))  # noqa: E731
        same_hit = sum(tr_share(p) for p in same)
        diff_hit = sum(not tr_share(p) for p in diff)
        same_hit_strict = sum(tr_same(p) for p in same)
        diff_hit_strict = sum(not tr_same(p) for p in diff)
        # 错误连接：≥2 人判定的片段，trad 是否拆成 ≥2 个实例
        wcount = Counter(i for r in RATERS for i in res[r]["wrong"])
        wrong = [i for i, c in wcount.items() if c >= 2]
        split = 0
        for i in wrong:
            ii = inst_r[seg_r == i]
            ii = ii[ii > 0]
            if ii.size:
                cnt = Counter(ii.tolist())
                split += sum(1 for v in cnt.values() if v >= 0.2 * ii.size) >= 2
        fg = seg_r > 0
        trad_fg = cls_r > 0
        row = {
            "n_consensus": len(cons),
            "confusion": {f"{a}->{b}": n for (a, b), n in sorted(conf.items())},
            "class_agree": round(sum(n for (a, b), n in conf.items()
                                     if a == b or (a == "not_defect" and b in ("background", "uncertain"))) / max(len(cons), 1), 3),
            "group_same_pairs": len(same), "group_same_hit": same_hit,
            "group_diff_pairs": len(diff), "group_diff_hit": diff_hit,
            "group_same_hit_strict": same_hit_strict, "group_diff_hit_strict": diff_hit_strict,
            "wrong_merge_consensus": len(wrong), "wrong_merge_split": split,
            "semantic5_unjudged": {r: rep["lenses"][lens]["raters"][r]["pixel_share"]["unjudged"] for r in RATERS},
            "trad_uncertain_share": round(float((cls_r[trad_fg] == 4).mean()), 3),
            "trad_class_share": {TRAD_NAME[c]: round(float((cls_r[trad_fg] == c).mean()), 3) for c in (1, 2, 3, 4)},
            "hr_fg_px": int(fg.sum()), "trad_fg_px": int(trad_fg.sum()),
            "trad_covers_hr": round(float(trad_fg[fg].mean()), 3),
            "hr_segments": int(len(np.unique(seg_r)) - 1),
            "trad_instances": int(len(np.unique(inst_r)) - 1),
        }
        if lens == "72l":
            a = np.load(SEM / "analysis" / "72l.npz")
            r2d = a["R2_cls"] == 3
            td = cls_r == 3
            row["damage_vs_R2"] = {"R2_px": int(r2d.sum()), "trad_px": int(td.sum()),
                                   "iou": round(float((r2d & td).sum() / max((r2d | td).sum(), 1)), 3),
                                   "R2_covered_by_trad": round(float(td[r2d].mean()), 3) if r2d.any() else None}
        for k in pooled:
            pooled[k] += {"same_total": len(same), "same_hit": same_hit, "diff_total": len(diff), "diff_hit": diff_hit,
                          "wrong_total": len(wrong), "wrong_split": split}[k]
        out["lenses"][lens] = row
        print("semantic", lens, row["class_agree"], f"same {same_hit}/{len(same)} diff {diff_hit}/{len(diff)}", flush=True)
    n = sum(pooled_conf.values())
    out["overall"] = {"confusion": {f"{a}->{b}": v for (a, b), v in sorted(pooled_conf.items())},
                      "class_agree": round(sum(v for (a, b), v in pooled_conf.items()
                                               if a == b or (a == "not_defect" and b in ("background", "uncertain"))) / n, 3),
                      **pooled}
    return out


# ── 3. 网格试点 ───────────────────────────────────────────────────────

def grid_pilot() -> dict:
    from review_pilot import CH, COLS, CW, NCOL, NROW, QUADS
    key = json.loads((PIL / "key.json").read_text())
    old = json.loads((PIL / "traditional.json").read_text())
    out: dict = {"lenses": {}, "overall": {}}
    acc = {c: Counter() for c in ("scratch", "spot", "damage")}
    for pid, lens in key.items():
        cls, _ = load_trad(lens)
        trad_masks = {"scratch": cls == 1, "spot": cls == 2, "damage": cls == 3}
        row = {}
        for c in ("scratch", "spot", "damage"):
            votes: Counter = Counter()
            for r in RATERS:
                f = PIL / "ratings" / f"{r}_{pid}.json"
                p = json.loads(f.read_text()).get("parsed") if f.exists() else None
                for q in QUADS:
                    for cell in ((p or {}).get("cells", {}).get(q, {}).get(c) or []):
                        votes[(q, cell)] += 1
            maj = {k for k, v in votes.items() if v >= 2}
            old_set = {(q, cell) for q in QUADS for cell in old[pid]["cells"][q][c]}
            new_set = set()
            for q, (x0, y0) in QUADS.items():
                m = trad_masks[c][y0:y0 + NROW * CH, x0:x0 + NCOL * CW]
                for rr in range(NROW):
                    for cc in range(NCOL):
                        if m[rr * CH:(rr + 1) * CH, cc * CW:(cc + 1) * CW].sum() >= 20:
                            new_set.add((q, f"{COLS[cc]}{rr + 1}"))
            d = {"majority": len(maj)}
            for name, s in (("old", old_set), ("trad_v1", new_set)):
                d[name] = len(s)
                d[f"{name}_recall"] = round(len(maj & s) / len(maj), 3) if maj else None
                d[f"{name}_precision"] = round(len(maj & s) / len(s), 3) if s else None
                acc[c][f"{name}_hit"] += len(maj & s)
                acc[c][f"{name}_n"] += len(s)
            acc[c]["maj"] += len(maj)
            row[c] = d
        out["lenses"][lens] = row
    for c, a in acc.items():
        out["overall"][c] = {"majority": a["maj"]} | {
            f"{n}_{k}": (round(a[f"{n}_hit"] / a[den], 3) if a[den] else None)
            for n in ("old", "trad_v1") for k, den in (("recall", "maj"), ("precision", f"{n}_n"))}
    return out


# ── 4. 第一阶段框标注 ─────────────────────────────────────────────────

def phase1_boxes() -> dict:
    from seg_p1_classical_baselines import CALIB_DIR, evaluate, load_gt_boxes, load_tile_index
    tile_index = load_tile_index()
    roi = np.load(CALIB_DIR / "roi_mask.npy")
    ring = np.load(CALIB_DIR / "ring_mask.npy").astype(np.uint8)
    ring = (cv2.dilate(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) > 0) & roi
    out: dict = {}
    for lens in LENSES:
        boxes = load_gt_boxes(lens, tile_index)
        cls, _ = load_trad(lens)
        hr = np.load(SEM / "seg" / f"{lens}.npz")["fg"]
        out[lens] = {}
        for name, m in (("trad_v1", cls > 0), ("hr_segment", hr), ("pilot_traditional", old_traditional(lens))):
            r = evaluate(m & roi, boxes, roi, ring)
            out[lens][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        print("phase1", lens, {k: v["box_recall"] for k, v in out[lens].items()}, flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = {ls: json.loads((ANN / f"{ls}.json").read_text())["features"] for ls in LENSES}
    report = {"features": feats}
    report["pixel5"] = pixel_pilot()
    print("pixel5", report["pixel5"]["methods"], flush=True)
    report["semantic5"] = semantic_pilot()
    report["pilot5"] = grid_pilot()
    print("pilot5", report["pilot5"]["overall"], flush=True)
    report["phase1"] = phase1_boxes()
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"对比结果: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
