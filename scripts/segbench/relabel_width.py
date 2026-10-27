"""伪标签 v2：用标注工具同一套区域定位算法，从 trad_v2 的中心线/斑点中心重建标签，统一线宽口径。

- 划痕：trad_v2 LabelMe 导出的每条中心线 → 1 像素重采样 → annot_server.ridge_region（法向剖面半高宽，阈值不低于 2σ）；
- 斑点：trad_v2 斑点多边形的中心 → annot_server.spot_region（半峰值生长，不低于 3σ）；
- 碎裂（trad_v2 足迹判据，模型无法从局部学到）与“待确认”：255 不计；
- trad_v2 标为划痕/斑点、但距重建区域超过 3 像素的像素：255 不计（多为没有中心线的短碎段，不当作背景）；
- 有效区外：255。类别：0 背景，1 划痕，2 斑点（v2 不含碎裂类，碎裂只由人工标注提供）。
输出 output/segbench/data/darkfield_v2/{lab,fg}/<lens>.npy 与 splits.json；图像沿用 darkfield/img。

用法:
    uv run --no-sync python scripts/segbench/relabel_width.py --workers 8
    uv run --no-sync python scripts/segbench/relabel_width.py --lenses 14l   # 只算一张（检查用）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import annot_server as AS  # noqa: E402

V1 = PROJECT_ROOT / "output/segbench/data/darkfield"
OUT = PROJECT_ROOT / "output/segbench/data/darkfield_v2"
CLASSES = ["background", "scratch", "spot"]
MAX_FG = 50000


def one(lens: str) -> dict:
    t0 = time.time()
    img = np.load(V1 / "img" / f"{lens}.npy")
    valid = AS.load_valid(lens)
    sigma = max(_sigma(lens), 0.8)          # 与 annot_server.LensCache 相同的 σ 下限
    e = {"img": img, "f": cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0), "valid": valid, "sigma": sigma}
    trad = cv2.imread(str(AS.TRAD_DIR / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)
    shapes = json.loads((AS.TRAD_DIR / f"{lens}_labelme.json").read_text())["shapes"]
    lab = np.zeros(img.shape, np.uint8)
    n_line = n_spot = 0
    for s in shapes:
        if s["label"] == "scratch" and s["shape_type"] == "linestrip" and len(s["points"]) >= 2:
            path = AS.densify(s["points"], 1.0)
            if len(path) >= 2:
                px = AS.ridge_region(e, path)
                lab[px[:, 1], px[:, 0]] = 1
                n_line += 1
    for s in shapes:
        if s["label"] == "spot" and s["shape_type"] == "polygon":
            c = np.asarray(s["points"], np.float64).mean(0)
            px = AS.spot_region(e, c[0], c[1])
            lab[px[:, 1], px[:, 0]] = 2
            n_spot += 1
    near = cv2.dilate((lab > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    lost = np.isin(trad, (1, 2)) & ~near & (lab == 0)
    lab[lost] = 255
    lab[np.isin(trad, (3, 4)) & (lab == 0)] = 255
    lab[~valid] = 255
    ys, xs = np.nonzero((lab > 0) & (lab < 255))
    if ys.size > MAX_FG:
        k = np.random.default_rng(0).choice(ys.size, MAX_FG, replace=False)
        ys, xs = ys[k], xs[k]
    np.save(OUT / "lab" / f"{lens}.npy", lab)
    np.save(OUT / "fg" / f"{lens}.npy", np.stack([ys, xs], 1).astype(np.int32))
    v1 = np.load(V1 / "lab" / f"{lens}.npy", mmap_mode="r")
    both = (v1 != 255) & (lab != 255)
    return {"lens": lens, "lines": n_line, "spots": n_spot, "sec": round(time.time() - t0, 1),
            "scratch_px_v1": int(((v1 == 1) & both).sum()), "scratch_px_v2": int(((lab == 1) & both).sum()),
            "lost_to_ignore": int(lost.sum()), "px": np.bincount(lab[lab != 255].ravel(), minlength=3).tolist()}


_REP: dict | None = None


def _sigma(lens: str) -> float:
    global _REP
    if _REP is None:
        _REP = {r["lens"]: r for r in json.loads((AS.STD_DIR / "report.json").read_text())}
    return float(_REP[lens]["noise_sigma"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lenses", nargs="*")
    args = ap.parse_args()
    for d in ("lab", "fg"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    sp = json.loads((V1 / "splits.json").read_text())
    lenses = args.lenses or [ls for v in sp["p622"].values() for ls in v]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, lenses))
    for r in rows[:5]:
        print(r)
    if args.lenses:
        return
    tot = np.sum([r["px"] for r in rows], 0)
    v1s, v2s = sum(r["scratch_px_v1"] for r in rows), sum(r["scratch_px_v2"] for r in rows)
    (OUT / "splits.json").write_text(json.dumps(
        {"classes": CLASSES, "source": "trad_v2 中心线/斑点中心 → 标注工具同一区域算法（annot_server.ridge_region/spot_region）",
         "ignore": "碎裂、待确认、距重建区域 >3 像素的 trad 前景、有效区外", "img_dir": str(V1 / "img"),
         "splits_from": "splits_v1", "p622": sp["p622"], "class_pixel_frac": (tot / tot.sum()).round(5).tolist(),
         "scratch_px_ratio_v2_over_v1": round(v2s / max(v1s, 1), 4), "per_lens": rows}, ensure_ascii=False, indent=1))
    print(len(rows), "张；类别像素占比", (tot / tot.sum()).round(5).tolist(), "；划痕像素 v2/v1 =", round(v2s / max(v1s, 1), 3))


if __name__ == "__main__":
    main()
