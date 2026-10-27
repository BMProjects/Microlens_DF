"""暗场标准输入 + trad_v2 初步标注 → 训练用伪标签数据（内存映射 npy），划分沿用 splits_v1。

标签：0 背景，1 划痕，2 斑点，3 碎裂（trad_v2 足迹判据下的碎裂像素），255 不计（“待确认”像素与有效区外）。
人工审核窗口（output/annotation/bench_v1/labels）不在这里转换，评价时直接读取，始终以人工标签为准。
测试集镜片也转换（便于最终一次性评价），但训练与调参脚本不得读取 test。

输出 output/segbench/data/darkfield/：img/<lens>.npy（uint8 H×W）、lab/<lens>.npy、fg/<lens>.npy（前景坐标抽样）、splits.json

用法:
    uv run --no-sync python scripts/segbench/prep_darkfield.py --workers 8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STD = PROJECT_ROOT / "output/dataset_v3"
TRAD = PROJECT_ROOT / "output/annotation/trad_v2"
SPLITS = PROJECT_ROOT / "output/annotation/splits_v1.json"
OUT = PROJECT_ROOT / "output/segbench/data/darkfield"
CLASSES = ["background", "scratch", "spot", "fragment"]
MAX_FG = 50000


def one(lens: str) -> dict:
    img = cv2.imread(str(STD / "images" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(STD / "valid" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    cls = cv2.imread(str(TRAD / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or valid is None or cls is None:
        raise FileNotFoundError(lens)
    lab = cls.copy()
    lab[cls == 4] = 255
    lab[valid == 0] = 255
    ys, xs = np.nonzero((lab > 0) & (lab < 255))
    if ys.size > MAX_FG:
        k = np.random.default_rng(0).choice(ys.size, MAX_FG, replace=False)
        ys, xs = ys[k], xs[k]
    np.save(OUT / "img" / f"{lens}.npy", img)
    np.save(OUT / "lab" / f"{lens}.npy", lab)
    np.save(OUT / "fg" / f"{lens}.npy", np.stack([ys, xs], 1).astype(np.int32))
    cnt = np.bincount(lab[lab != 255].ravel(), minlength=4)
    return {"lens": lens, "px": cnt.tolist(), "ignore": int((lab == 255).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for d in ("img", "lab", "fg"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    sp = json.loads(SPLITS.read_text())["splits"]
    lenses = [ls for v in sp.values() for ls in v]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, lenses))
    tot = np.sum([r["px"] for r in rows], 0)
    (OUT / "splits.json").write_text(json.dumps(
        {"classes": CLASSES, "source": "trad_v2 伪标签（待确认→255）", "splits_from": "splits_v1",
         "p622": {"train": sp["train"], "val": sp["val"], "test": sp["test"]},
         "class_pixel_frac": (tot / tot.sum()).round(5).tolist()}, ensure_ascii=False, indent=1))
    print(len(rows), "张；类别像素占比", (tot / tot.sum()).round(5).tolist())


if __name__ == "__main__":
    main()
