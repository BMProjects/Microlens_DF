"""nnU-Net v2 在暗场 v2 伪标签上的适配：原始格式转换、fold 0 划分、调参集预测评价（对伪标签与人工窗口）。

- 只放训练集与调参集镜片（测试集冻结到最终评价）；镜片名映射为 case_XXX（避免非 ASCII 文件名），映射表 case_map.json；
- 单通道灰度 PNG，通道名 rgb_to_0_1 → 固定 ÷255 缩放（保留原始光强，不做逐图 z-score）；
- 标签：0 背景，1 划痕，2 斑点，3 ignore（v2 的 255）。nnU-Net 的 ignore 标签须为最大整数，只在已标注像素上计损失。

用法:
    uv run --no-sync python scripts/segbench/nnunet_darkfield.py prep
    uv run --no-sync python scripts/segbench/nnunet_darkfield.py splits      # plan_and_preprocess 之后
    uv run --no-sync python scripts/segbench/nnunet_darkfield.py eval --pred <预测目录> --name <结果名>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/segbench"))

from darkfield_defects.eval.seg_metrics import ConfusionMatrix, all_background_pa  # noqa: E402

V2 = PROJECT_ROOT / "output/segbench/data/darkfield_v2"
IMG = PROJECT_ROOT / "output/segbench/data/darkfield/img"
NN = PROJECT_ROOT / "output/segbench/nnunet"
DS = "Dataset502_DarkfieldV2"
IGN = 3


def case_map() -> dict:
    p = NN / "raw" / DS / "case_map.json"
    return json.loads(p.read_text())


def cmd_prep(_args) -> None:
    sp = json.loads((V2 / "splits.json").read_text())
    raw = NN / "raw" / DS
    for d in ("imagesTr", "labelsTr", "imagesTs"):
        (raw / d).mkdir(parents=True, exist_ok=True)
    lenses = sp["p622"]["train"] + sp["p622"]["val"]
    cmap = {ls: f"case_{k:03d}" for k, ls in enumerate(lenses)}
    for ls, cid in cmap.items():
        img = np.load(IMG / f"{ls}.npy")
        lab = np.load(V2 / "lab" / f"{ls}.npy").copy()
        lab[lab == 255] = IGN
        cv2.imwrite(str(raw / "imagesTr" / f"{cid}_0000.png"), img)
        cv2.imwrite(str(raw / "labelsTr" / f"{cid}.png"), lab)
        if ls in sp["p622"]["val"]:              # 调参集另放一份到 imagesTs，供预测
            cv2.imwrite(str(raw / "imagesTs" / f"{cid}_0000.png"), img)
    (raw / "case_map.json").write_text(json.dumps(cmap, ensure_ascii=False, indent=1))
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "rgb_to_0_1"},
        "labels": {"background": 0, "scratch": 1, "spot": 2, "ignore": IGN},
        "numTraining": len(lenses), "file_ending": ".png", "overwrite_image_reader_writer": "NaturalImage2DIO"},
        indent=1))
    print(f"写入 {raw}：{len(lenses)} 例（训练 {len(sp['p622']['train'])}，调参 {len(sp['p622']['val'])}）")


def cmd_splits(_args) -> None:
    sp = json.loads((V2 / "splits.json").read_text())["p622"]
    cm = case_map()
    pre = NN / "preprocessed" / DS
    if not pre.exists():
        raise SystemExit("请先运行 nnUNetv2_plan_and_preprocess")
    (pre / "splits_final.json").write_text(json.dumps(
        [{"train": [cm[x] for x in sp["train"]], "val": [cm[x] for x in sp["val"]]}], indent=1))
    print("splits_final.json：fold 0 = splits_v1 训练/调参")


def cmd_eval(args) -> None:
    import train as T                                     # 复用人工窗口读取逻辑
    data = T.DarkfieldData("darkfield_v2")
    names = data.classes
    cm, cw, n_win = ConfusionMatrix(len(names)), ConfusionMatrix(len(names)), 0
    cmap = case_map()
    for ls in data.split["val"]:
        pr = cv2.imread(str(Path(args.pred) / f"{cmap[ls]}.png"), cv2.IMREAD_GRAYSCALE)
        if pr is None:
            raise SystemExit(f"缺少预测 {ls}（{cmap[ls]}）")
        cm.update(np.load(V2 / "lab" / f"{ls}.npy"), pr)
        for y0, x0, lab in data.windows(ls):
            cw.update(lab, pr[y0:y0 + lab.shape[0], x0:x0 + lab.shape[1]])
            n_win += 1
    r = cm.metrics(names)
    r.update({"all_background_PA": all_background_pa(cm), "final_split": "val", "protocol": "darkfield2_pseudo_val"})
    if n_win:
        rw = cw.metrics(names)
        rw.update({"all_background_PA": all_background_pa(cw), "n_windows": n_win})
        r["human_windows"] = rw
    out = PROJECT_ROOT / "output/segbench/runs" / f"darkfield2_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"[val] PA {r['PA']:.4f} MPA {r['MPA']:.4f} mIoU {r['mIoU']:.4f} FWIoU {r['FWIoU']:.4f} "
          + " ".join(f"{k}:{v['IoU']}" for k, v in r["per_class"].items())
          + (f" | 人工窗口 {n_win} 个 mIoU {r['human_windows']['mIoU']:.4f}" if n_win else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("splits")
    e = sub.add_parser("eval")
    e.add_argument("--pred", required=True)
    e.add_argument("--name", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "splits": cmd_splits, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
