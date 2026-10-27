"""MSD（手机玻璃盖板缺陷分割，北大）→ 评测标准格式，保留原始四类，生成固定 6:2:2 划分。

原始数据：oil/ scratch/ stain/ 各 400 张 1920×1080 RGB；掩膜在 ground_truth_1（Scr、Sta）与 ground_truth_2（Oil），
三通道 PNG，缺陷像素为 128，类别由文件名前缀决定（每张图只含一类缺陷）。
输出（output/segbench/data/msd/）：
    images/<id>.jpg   原图软链接
    masks/<id>.png    单通道类别图：0 背景，1 油污，2 划痕，3 污渍
    splits.json       p622：按类别分层的 240/80/80 × 3（种子固定）；官方仅说明随机 6:2:2，未提供划分文件

用法:
    uv run python scripts/segbench/prep_msd.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/home/bm/Data/MSD")
OUT = PROJECT_ROOT / "output/segbench/data/msd"
CLASSES = ["background", "oil", "scratch", "stain"]
PREFIX = {"Oil": ("oil", "ground_truth_2", 1), "Scr": ("scratch", "ground_truth_1", 2),
          "Sta": ("stain", "ground_truth_1", 3)}
SEED = 20261026


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    stats = {}
    for pre, (img_dir, gt_dir, cid) in PREFIX.items():
        ids = sorted(p.stem for p in (args.src / img_dir).glob(f"{pre}_*.jpg"))
        if len(ids) != 400:
            raise SystemExit(f"{img_dir}: 期望 400 张，实际 {len(ids)}")
        fg = []
        for i in ids:
            m = cv2.imread(str(args.src / gt_dir / f"{i}.png"), cv2.IMREAD_COLOR)
            if m is None:
                raise SystemExit(f"掩膜缺失: {gt_dir}/{i}.png")
            lab = np.where(m.max(2) > 0, cid, 0).astype(np.uint8)
            cv2.imwrite(str(args.out / "masks" / f"{i}.png"), lab)
            fg.append(float((lab > 0).mean()))
            dst = args.out / "images" / f"{i}.jpg"
            if not dst.exists():
                os.symlink(args.src / img_dir / f"{i}.jpg", dst)
        stats[CLASSES[cid]] = {"n": len(ids), "fg_frac_median": round(float(np.median(fg)), 5)}
        perm = list(rng.permutation(ids))
        split["train"] += sorted(perm[:240])
        split["val"] += sorted(perm[240:320])
        split["test"] += sorted(perm[320:])
    (args.out / "splits.json").write_text(json.dumps(
        {"classes": CLASSES, "seed": SEED, "rule": "按类别分层随机 6:2:2（每类 240/80/80）",
         "stats": stats, "p622": split}, ensure_ascii=False, indent=1))
    print({k: len(v) for k, v in split.items()}, stats)


if __name__ == "__main__":
    main()
