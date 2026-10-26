"""传统算法基础标注：对 dataset_v3 标准输入运行 trad_segment，导出可人工修订的像素级实例标注。

每张镜片输出（--out 目录下）：
    <lens>_inst.png      uint16 实例图（0 为背景）
    <lens>_inst2.png     uint16 交叉区第二所属实例（亮线为主、暗线为次；0 为无叠加）
    <lens>_cls.png       uint8 类别图（0 背景，1 划痕，2 斑点，3 碎裂，4 待确认）
    <lens>.json          实例属性、连接记录（交叉穿越/断点桥接）、统计量与参数
    <lens>_labelme.json  LabelMe 格式：划痕为中心线折线（同一实例共用 group_id，含桥接段），
                         斑点/待确认为多边形，碎裂为区域多边形
    <lens>_preview.jpg   1/4 分辨率类别叠加预览
    <lens>_review.csv    复核队列（低置信连接、待确认实例、密集区内的长划痕、碎裂区）
另写 index.csv：每张镜片一行的统计特征，供最小人工标注子集选择。

用法:
    uv run python scripts/annot_build.py run --lenses 108l 26l 90r 76r 72l
    uv run python scripts/annot_build.py run --all --workers 4
    uv run python scripts/annot_build.py select --n 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.annotation.trad_segment import CLASS_ID, TradParams, segment  # noqa: E402

STD_DIR = PROJECT_ROOT / "output/dataset_v3"
OUT = PROJECT_ROOT / "output/annotation/trad_v2"
PIXEL_MM = 0.0068
VERSION = "trad_v2"
CLASS_NAME = {v: k for k, v in CLASS_ID.items()}
# BGR
COLORS = {1: (80, 220, 80), 2: (0, 160, 255), 3: (255, 0, 255), 4: (160, 160, 160)}


def load_lens(lens: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(STD_DIR / "images" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(STD_DIR / "valid" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or valid is None:
        raise FileNotFoundError(f"标准输入缺失: {lens}")
    return img, valid > 0


def polyline(chain: np.ndarray) -> list[list[int]]:
    pts = chain[:, ::-1].astype(np.int32).reshape(-1, 1, 2)      # (y, x) → (x, y)
    return cv2.approxPolyDP(pts, 1.0, False).reshape(-1, 2).tolist()


def polygons(mask: np.ndarray, x0: int = 0, y0: int = 0, min_area: float = 4.0) -> list[list[list[int]]]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area and len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append((c + [x0, y0]).tolist())
    return out


def labelme(lens: str, res, h: int, w: int) -> dict:
    chains = res.maps["chains"]
    shapes = []
    inst_objs = ndi.find_objects(res.inst)
    for it in res.instances:
        gid = it["id"]
        desc = f"width_px={it['width_px']};amp={it['amp']};len_px={it['len_px']}"
        if it["flags"]:
            desc += ";flags=" + "|".join(it["flags"])
        if it["class"] == "scratch":
            for b in it["branches"]:
                if len(chains[b]) >= 2:
                    shapes.append({"label": "scratch", "points": polyline(chains[b]), "group_id": gid,
                                   "shape_type": "linestrip", "flags": {}, "description": desc})
        elif it["class"] in ("spot", "uncertain"):
            sl = inst_objs[gid - 1]
            if sl is None:
                continue
            for poly in polygons(res.inst[sl] == gid, sl[1].start, sl[0].start):
                shapes.append({"label": it["class"], "points": poly, "group_id": gid,
                               "shape_type": "polygon", "flags": {}, "description": desc})
    # 桥接段画进对应划痕实例
    for qa, qb, b in res.maps["bridge_segments"]:
        gid = int(res.inst[qa[0], qa[1]]) or int(res.inst[qb[0], qb[1]])
        shapes.append({"label": "scratch", "points": [[int(qa[1]), int(qa[0])], [int(qb[1]), int(qb[0])]],
                       "group_id": gid or None, "shape_type": "linestrip", "flags": {}, "description": "bridge"})
    for poly in polygons(res.maps["damage"], min_area=64):
        shapes.append({"label": "damage", "points": poly, "group_id": None, "shape_type": "polygon",
                       "flags": {}, "description": "damage_region"})
    return {"version": "5.2.1", "flags": {}, "shapes": shapes,
            "imagePath": f"../../dataset_v3/images/{lens}.png", "imageData": None,
            "imageHeight": h, "imageWidth": w}


def review_rows(lens: str, res) -> list[dict]:
    rows = []
    for k, lk in enumerate(res.links):
        if lk.get("review"):
            x, y = lk["pos"][0], lk["pos"][1]
            rows.append({"lens": lens, "kind": f"{lk['type']}_link", "id": k, "x": x, "y": y,
                         "score": lk["score"], "note": f"a={lk['a']} b={lk['b']} ps={lk['profile_sim']}"})
    objs = ndi.find_objects(res.inst)
    for it in res.instances:
        if it["class"] == "uncertain" or "位于密集区" in it["flags"]:
            sl = objs[it["id"] - 1]
            if sl is None:
                continue
            rows.append({"lens": lens, "kind": "uncertain" if it["class"] == "uncertain" else "dense_scratch",
                         "id": it["id"], "x": (sl[1].start + sl[1].stop) // 2, "y": (sl[0].start + sl[0].stop) // 2,
                         "score": "", "note": f"len={it['len_px']} width={it['width_px']} amp={it['amp']}"})
    n, lab, st, cen = cv2.connectedComponentsWithStats(res.maps["damage"].astype(np.uint8), connectivity=8)
    for i in range(1, n):
        rows.append({"lens": lens, "kind": "damage_region", "id": i, "x": int(cen[i][0]), "y": int(cen[i][1]),
                     "score": "", "note": f"area_mm2={st[i, cv2.CC_STAT_AREA] * PIXEL_MM ** 2:.3f}"})
    order = {"damage_region": 0, "crossing_link": 1, "gap_link": 2, "dense_scratch": 3, "uncertain": 4}
    rows.sort(key=lambda r: (order[r["kind"]], r["score"] if r["score"] != "" else 9))
    return rows


def preview(img: np.ndarray, valid: np.ndarray, res) -> np.ndarray:
    disp = np.clip(16.0 * np.sqrt(img.astype(np.float32)), 0, 255).astype(np.uint8)
    disp[~valid] = 0
    out = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    for c, col in COLORS.items():
        m = res.cls == c
        out[m] = (0.35 * out[m] + 0.65 * np.array(col)).astype(np.uint8)
    out[res.maps["bridge"]] = (0, 0, 255)
    return cv2.resize(out, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)


def lens_features(lens: str, res, valid: np.ndarray, rep: dict) -> dict:
    sc = [it for it in res.instances if it["class"] == "scratch"]
    cnt = np.bincount(res.cls.ravel(), minlength=5)
    va = float(valid.sum())
    return {"lens": lens, "eyeglass": lens[:-1], "side": lens[-1],
            "n_scratch": len(sc), "scratch_len_mm": round(sum(it["len_px"] for it in sc) * PIXEL_MM, 2),
            "long_scratch_n": sum(1 for it in sc if it["len_px"] * PIXEL_MM >= 2.0),
            "n_spot": sum(1 for it in res.instances if it["class"] == "spot"),
            "n_uncertain": sum(1 for it in res.instances if it["class"] == "uncertain"),
            "scratch_frac": round(cnt[1] / va, 5), "spot_frac": round(cnt[2] / va, 5),
            "damage_frac": round(cnt[3] / va, 5), "uncertain_frac": round(cnt[4] / va, 5),
            "damage_mm2": round(float(res.maps["damage"].sum()) * PIXEL_MM ** 2, 3),
            "n_crossing": res.stats["n_crossing_links"], "n_gap": res.stats["n_gap_links"],
            "n_review": res.stats["n_review_links"],
            "noise_sigma": rep.get("noise_sigma"), "diffuse_mean": rep.get("diffuse_mean"),
            "mean_scratch_len_mm": round(float(np.mean([it["len_px"] for it in sc])) * PIXEL_MM, 3) if sc else 0.0,
            "overlap_px": res.stats.get("overlap_px", 0),
            "orient_order": res.stats.get("orient_order"), "orient_dominant_deg": res.stats.get("orient_dominant_deg"),
            "parallel_frac": res.stats.get("parallel_frac"),
            "damage_frac_valid": res.stats.get("damage_frac_valid"),
            "damage_frac_central": res.stats.get("damage_frac_central"),
            "large_damage": res.stats.get("large_damage"), "n_damage_regions": len(res.regions),
            "damage_frac_fragment": res.stats.get("damage_frac_fragment")}


def run_lens(lens: str, out: str, overrides: dict | None = None) -> dict:
    out_dir = Path(out)
    rep = {r["lens"]: r for r in json.loads((STD_DIR / "report.json").read_text())}[lens]
    img, valid = load_lens(lens)
    t0 = time.perf_counter()
    p = TradParams(**(overrides or {}))
    res = segment(img, valid, rep["noise_sigma"], p, keep_maps=True)
    dt = time.perf_counter() - t0
    h, w = img.shape
    if res.inst.max() > 65535:
        raise ValueError(f"{lens}: 实例数超过 uint16 上限")
    cv2.imwrite(str(out_dir / f"{lens}_inst.png"), res.inst.astype(np.uint16))
    if res.inst2 is not None:
        cv2.imwrite(str(out_dir / f"{lens}_inst2.png"), res.inst2.astype(np.uint16))
    cv2.imwrite(str(out_dir / f"{lens}_cls.png"), res.cls)
    feats = lens_features(lens, res, valid, rep)
    feats["seconds"] = round(dt, 1)
    rows = review_rows(lens, res)
    feats["n_review_items"] = len(rows)
    (out_dir / f"{lens}.json").write_text(json.dumps(
        {"lens": lens, "version": VERSION, "params": asdict(p), "stats": res.stats, "features": feats,
         "class_id": CLASS_ID, "instances": res.instances, "links": res.links,
         "damage_regions": res.regions}, ensure_ascii=False))
    (out_dir / f"{lens}_labelme.json").write_text(json.dumps(labelme(lens, res, h, w), ensure_ascii=False))
    with open(out_dir / f"{lens}_review.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, ["lens", "kind", "id", "x", "y", "score", "note"])
        wr.writeheader()
        wr.writerows(rows)
    cv2.imwrite(str(out_dir / f"{lens}_preview.jpg"), preview(img, valid, res), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return feats


def all_lenses() -> list[str]:
    return sorted(p.stem for p in (STD_DIR / "images").glob("*.png"))


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lenses = all_lenses() if args.all else args.lenses
    if not args.force:
        lenses = [ls for ls in lenses if not (out / f"{ls}.json").exists()]
    print(f"待处理 {len(lenses)} 张", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_lens, ls, str(out)): ls for ls in lenses}
        for fu in futs:
            ls = futs[fu]
            try:
                f = fu.result()
                print(ls, f["seconds"], "s", f"scratch={f['n_scratch']} spot={f['n_spot']} "
                      f"uncertain={f['n_uncertain']} damage_mm2={f['damage_mm2']}", flush=True)
            except Exception as exc:  # 记录后继续其他镜片
                print(ls, "失败:", exc, flush=True)
    write_index(out)


def write_index(out: Path) -> None:
    rows = []
    for p in sorted(out.glob("*.json")):
        if p.stem.endswith("_labelme"):
            continue
        f = json.loads(p.read_text())["features"]
        if "n_review_items" not in f:   # 早期导出未写入该字段，按复核队列行数补齐
            with open(out / f"{p.stem}_review.csv", encoding="utf-8") as fh:
                f["n_review_items"] = sum(1 for _ in fh) - 1
        rows.append(f)
    if not rows:
        return
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"index.csv: {len(rows)} 张")


PILOT = ["108l", "26l", "90r", "76r", "72l"]
SELECT_FEATURES = ["scratch_len_mm", "scratch_frac", "n_spot", "spot_frac", "damage_frac", "uncertain_frac",
                   "n_crossing", "diffuse_mean"]
LOG_FEATURES = {"scratch_len_mm", "n_spot", "n_crossing"}


def feature_matrix(rows: list[dict]) -> np.ndarray:
    cols = []
    for k in SELECT_FEATURES:
        v = np.array([float(r[k]) for r in rows])
        v = np.log1p(v) if k in LOG_FEATURES else v
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med)) or (v.std() or 1.0)
        cols.append(np.clip((v - med) / mad, -6, 6))     # 稳健标准化，截断极端值以免单个离群镜片主导距离
    return np.stack(cols, 1)


def cmd_select(args: argparse.Namespace) -> None:
    """最小人工标注子集：每副眼镜至多一片；先放入五图试点，再在特征空间做最远点（k-center）贪心。"""
    out = Path(args.out)
    with open(out / "index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = feature_matrix(rows)
    names = [r["lens"] for r in rows]
    eyeglass = [r["eyeglass"] for r in rows]
    chosen = [names.index(p) for p in PILOT if p in names]
    used = {eyeglass[i] for i in chosen}
    d = np.min(np.linalg.norm(X[:, None] - X[chosen][None], axis=2), axis=1)
    curve = [{"n": len(chosen), "max_dist": round(float(d.max()), 3), "mean_dist": round(float(d.mean()), 3)}]
    order = []
    while len(chosen) < args.max_n:
        cand = [i for i in range(len(rows)) if eyeglass[i] not in used]
        if not cand:
            break
        i = max(cand, key=lambda j: d[j])
        chosen.append(i)
        order.append(i)
        used.add(eyeglass[i])
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
        curve.append({"n": len(chosen), "max_dist": round(float(d.max()), 3), "mean_dist": round(float(d.mean()), 3)})
    n = args.n
    sel = chosen[:n]
    sev = np.array([float(r["scratch_frac"]) for r in rows])
    q = np.quantile(sev, [0.25, 0.5, 0.75])
    bins = np.digitize(sev, q)
    ex = {"scratch_len_mm": "划痕总长", "scratch_frac": "划痕面积占比", "n_spot": "斑点数", "spot_frac": "斑点面积",
          "damage_frac": "碎裂面积", "uncertain_frac": "待确认占比", "n_crossing": "交叉数", "diffuse_mean": "雾化程度"}
    selected = []
    for i in sel:
        r = rows[i]
        if r["lens"] in PILOT:
            reason = "五图试点（已有三助手判定，可作对照）"
        else:
            z = X[i]
            k = int(np.argmax(np.abs(z)))
            reason = f"特征空间覆盖：{ex[SELECT_FEATURES[k]]}{'偏高' if z[k] > 0 else '偏低'}（z={z[k]:+.1f}），严重度四分位第 {bins[i] + 1} 档"
        selected.append({"lens": r["lens"], "eyeglass": r["eyeglass"], "reason": reason,
                         **{k: r[k] for k in ("n_scratch", "scratch_len_mm", "n_spot", "damage_mm2", "n_review_items")}})
    cover = Counter(int(bins[i]) for i in sel)
    summary = (f"共 {len(rows)} 张、{len(set(eyeglass))} 副眼镜；选出 {len(sel)} 张（每副眼镜至多一片）。"
               f"特征：{'、'.join(ex[k] for k in SELECT_FEATURES)}（稳健标准化）。"
               f"全体镜片到最近入选镜片的平均特征距离由仅含试点时的 {curve[0]['mean_dist']} 降到 {curve[n - len(PILOT)]['mean_dist']}"
               f"（选到 {args.max_n} 张时为 {curve[-1]['mean_dist']}）。严重度四分位覆盖："
               + "、".join(f"第 {b + 1} 档 {cover.get(b, 0)} 张" for b in range(4)) + "。")
    (out / "subset.json").write_text(json.dumps({"summary": summary, "features": SELECT_FEATURES, "curve": curve,
                                                 "selected": selected, "kcenter_order": [names[i] for i in order]},
                                                ensure_ascii=False, indent=1))
    print(summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--lenses", nargs="+")
    g.add_argument("--all", action="store_true")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--force", action="store_true")
    r.add_argument("--out", default=str(OUT))
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--max-n", type=int, default=60)
    s.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {"run": cmd_run, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
