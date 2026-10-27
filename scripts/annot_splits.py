"""按眼镜分组冻结 训练/调参/测试 划分（第二阶段计划 v3.1 A0）。

- 分组单位：眼镜（镜片名 <编号><l|r>[-后缀] 中的“编号+后缀”，如 58l-ljj、58r-ljj → 58-ljj），同一副眼镜的两片只进同一集合；
- 不符合命名规则的异型样本（如“单片镜片”）固定进训练集；
- 分层：以眼镜内两片的平均缺陷像素占比（标准输入 > 4σ 的有效区像素比例）排序，每 31 副为一层，层内随机分配，
  使三个集合的磨损程度分布一致；
- 算法调参中反复使用过的试点镜片（108l 26l 90r 76r 72l 55l）所在眼镜固定进训练集，避免测试集被开发过程“看过”。

用法:
    uv run python scripts/annot_splits.py            # 写 output/annotation/splits_v1.json（已存在则拒绝覆盖）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STD_DIR = PROJECT_ROOT / "output/dataset_v3"
OUT = PROJECT_ROOT / "output/annotation/splits_v1.json"
PILOT = ["108l", "26l", "90r", "76r", "72l", "55l"]
TARGET = {"test": 30, "val": 8}          # 其余眼镜为训练集
SEED = 20261026
LENS_RE = re.compile(r"^(\d+)([lr])(-.+)?$")


def eyeglass(lens: str) -> str | None:
    m = LENS_RE.match(lens)
    return f"{m.group(1)}{m.group(3) or ''}" if m else None


def severity(lens: str, sigma: float) -> float:
    img = cv2.imread(str(STD_DIR / "images" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(STD_DIR / "valid" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or valid is None:
        raise SystemExit(f"标准输入缺失: {lens}")
    v = valid > 0
    return float((img[v] > 4 * max(sigma, 0.8)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true", help="覆盖已冻结的划分（会使已有标注的集合归属失效，慎用）")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} 已存在（划分已冻结）；确需重划请加 --force")

    rep = {r["lens"]: r for r in json.loads((STD_DIR / "report.json").read_text())}
    lenses = sorted(rep)
    sev = {ls: severity(ls, rep[ls]["noise_sigma"]) for ls in lenses}
    groups: dict[str, list[str]] = {}
    atypical = [ls for ls in lenses if eyeglass(ls) is None]
    for ls in lenses:
        groups.setdefault(eyeglass(ls) or ls, []).append(ls)
    forced = {eyeglass(ls) for ls in PILOT} | set(atypical)
    free = sorted((g for g in groups if g not in forced), key=lambda g: np.mean([sev[x] for x in groups[g]]))

    rng = np.random.default_rng(SEED)
    n_layer = 4
    layers = np.array_split(np.array(free), n_layer)
    assign = {g: "train" for g in forced}
    for i, layer in enumerate(layers):
        layer = list(rng.permutation(layer))
        # 每层按比例分配，余数按层轮换，保证总数精确等于目标
        quota = {k: v // n_layer + (1 if i < v % n_layer else 0) for k, v in TARGET.items()}
        for k in ("test", "val"):
            for _ in range(quota[k]):
                assign[layer.pop()] = k
        for g in layer:
            assign[g] = "train"

    splits = {k: sorted(ls for g, s in assign.items() if s == k for ls in groups[g]) for k in ("train", "val", "test")}
    summary = {k: {"eyeglasses": sum(1 for s in assign.values() if s == k), "lenses": len(v),
                   "severity_median": round(float(np.median([sev[x] for x in v])), 5)} for k, v in splits.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": "splits_v1", "seed": SEED, "rule": __doc__.split("用法")[0].strip(),
                               "forced_train": sorted(forced), "atypical": atypical, "summary": summary,
                               "splits": splits, "eyeglass": {ls: eyeglass(ls) or ls for ls in lenses},
                               "severity": {k: round(v, 5) for k, v in sev.items()}}, ensure_ascii=False, indent=1))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
