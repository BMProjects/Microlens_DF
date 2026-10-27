"""nnU-Net v2 在 MSD 上的适配：转换为 nnU-Net 原始格式（fold 0 = p622 的训练/验证划分），测试集预测后用 FCN 指标评价。

目录（环境变量由本脚本打印，供 nnUNetv2_* 命令使用）：output/segbench/nnunet/{raw,preprocessed,results}

用法:
    uv run --no-sync python scripts/segbench/nnunet_msd.py prep            # 写 raw/Dataset501_MSD
    uv run --no-sync python scripts/segbench/nnunet_msd.py splits          # plan_and_preprocess 之后写 splits_final.json
    uv run --no-sync python scripts/segbench/nnunet_msd.py eval --pred <预测目录> --name <结果名>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.eval.seg_metrics import ConfusionMatrix, all_background_pa  # noqa: E402

MSD = PROJECT_ROOT / "output/segbench/data/msd"
NN = PROJECT_ROOT / "output/segbench/nnunet"
DS = "Dataset501_MSD"


def cmd_prep(_args) -> None:
    sp = json.loads((MSD / "splits.json").read_text())
    raw = NN / "raw" / DS
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (raw / d).mkdir(parents=True, exist_ok=True)
    for part, (idir, ldir) in (("train", ("imagesTr", "labelsTr")), ("val", ("imagesTr", "labelsTr")),
                               ("test", ("imagesTs", "labelsTs"))):
        for i in sp["p622"][part]:
            img = cv2.imread(str(MSD / "images" / f"{i}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"读图失败 {i}")
            cv2.imwrite(str(raw / idir / f"{i}_0000.png"), img)          # RGB 三通道存于同一 PNG（自然图像格式）
            os.link(MSD / "masks" / f"{i}.png", raw / ldir / f"{i}.png") if not (raw / ldir / f"{i}.png").exists() else None
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {n: k for k, n in enumerate(sp["classes"])},
        "numTraining": len(sp["p622"]["train"]) + len(sp["p622"]["val"]),
        "file_ending": ".png", "overwrite_image_reader_writer": "NaturalImage2DIO",
        "normalization_schemes": ["rgb_to_0_1", "rgb_to_0_1", "rgb_to_0_1"]}, indent=1))
    print(f"写入 {raw}")


def cmd_splits(_args) -> None:
    sp = json.loads((MSD / "splits.json").read_text())["p622"]
    pre = NN / "preprocessed" / DS
    if not pre.exists():
        raise SystemExit("请先运行 nnUNetv2_plan_and_preprocess")
    (pre / "splits_final.json").write_text(json.dumps([{"train": sp["train"], "val": sp["val"]}], indent=1))
    print("splits_final.json：fold 0 = p622 训练/验证")


def cmd_eval(args) -> None:
    sp = json.loads((MSD / "splits.json").read_text())
    names = sp["classes"]
    cm = ConfusionMatrix(len(names))
    for i in sp["p622"]["test"]:
        pr = cv2.imread(str(Path(args.pred) / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(MSD / "masks" / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        if pr is None or gt is None:
            raise SystemExit(f"缺少预测或标签 {i}")
        cm.update(gt, pr)
    r = cm.metrics(names)
    r["all_background_PA"] = all_background_pa(cm)
    r["protocol"] = "msd_p622_fullres"
    out = PROJECT_ROOT / "output/segbench/runs" / f"msd_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"[test] PA {r['PA']:.4f} MPA {r['MPA']:.4f} mIoU {r['mIoU']:.4f} FWIoU {r['FWIoU']:.4f} "
          + " ".join(f"{k}:{v['IoU']}" for k, v in r["per_class"].items()))


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
