"""阶段二单图像判定 δᵢ 评估（镜片级等级一致 + 切片级框 F1）。

不依赖 torch：输入是任意模型导出的 YOLO txt 预测（带置信度列）。
置信度阈值请在 val 上选定后固定，再用于测试集，不要在测试集上调。

用法:
    uv run python scripts/eval_image_level.py \
        --pred-dir output/experiments/phase3b_fn_analysis/predictions/labels \
        --tile-list output/cnas_eval/cnas_val_list.txt \
        --conf 0.2 0.3 --out output/experiments/phase2_image_level/stage2_cleaned
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from darkfield_defects.eval.image_level import evaluate_image_level

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pred-dir", type=Path, required=True, help="预测 YOLO txt 目录（含 conf 列）")
    p.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "output/tile_dataset/labels/val")
    p.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "output/dataset_v2/images")
    p.add_argument("--tile-list", type=Path, default=None,
                   help="切片清单（每行一个路径或文件名）；缺省为 gt-dir 下全部切片")
    p.add_argument("--conf", type=float, nargs="+", default=[0.2])
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--f1-min", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True, help="结果目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.disable(logging.INFO)  # WearScore 每镜一条 info 日志，批量时过于嘈杂
    for d in (args.pred_dir, args.gt_dir, args.image_dir):
        if not d.is_dir():
            raise SystemExit(f"目录不存在: {d}")
    if args.tile_list:
        stems = sorted({Path(line).stem for line in args.tile_list.read_text().split()})
    else:
        stems = sorted(p.stem for p in args.gt_dir.glob("*.txt"))
    if not stems:
        raise SystemExit("切片清单为空")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for conf in args.conf:
        rep = evaluate_image_level(stems, args.gt_dir, args.pred_dir, args.image_dir,
                                   conf=conf, iou_thr=args.iou_thr, f1_min=args.f1_min)
        s = rep.summary()
        summaries.append(s)
        print(json.dumps(s, ensure_ascii=False))
        for name, rows in (("lens", rep.lens_rows), ("tile", rep.tile_rows)):
            with open(args.out / f"{name}_conf{conf:g}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    meta = {
        "pred_dir": str(args.pred_dir), "gt_dir": str(args.gt_dir),
        "image_dir": str(args.image_dir),
        "tile_list": str(args.tile_list) if args.tile_list else None,
        "iou_thr": args.iou_thr, "f1_min": args.f1_min, "summaries": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
