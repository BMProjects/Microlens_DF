"""由原始暗场图像生成第二阶段标准输入图像（output/dataset_v3）。

配准沿用原预处理流程的模板与参数；之后调用 preprocessing/standardize.py：
背景偏置补偿到 0、分离弥散层（离焦晕斑/宽模糊条纹/雾状底光）、剔除边界宽亮带。

输出:
    images/<镜片>.png   标准输入（uint8，背景≈0，原始计数单位，低于 0 的噪声截为 0）
    diffuse/<镜片>.png  弥散层（uint8）；补偿后图像 = 标准输入 + 弥散层
    valid/<镜片>.png    本图有效区域（ROI 减去边界宽亮带）
    report.json         每张图的配准参数、偏置、噪声水平、弥散层与剔除统计

用法:
    uv run python scripts/preprocess_standard.py --raw-dir /mnt/Data/Data/Microlens_df/mingyue20260213
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from darkfield_defects.detection.params import load_params  # noqa: E402
from darkfield_defects.preprocessing.registration import apply_warp, register_to_template  # noqa: E402
from darkfield_defects.preprocessing.standardize import standardize  # noqa: E402

CAL = PROJECT_ROOT / "output/dataset_v2/calibration"
LENS_RE = re.compile(r"^\d+[lr](-ljj)?$")

_G: dict = {}


def _init(raw_dir: str, out: str) -> None:
    logging.disable(logging.WARNING)
    _G["B"] = np.load(CAL / "B_blur.npy").astype(np.float32)
    _G["ring"] = np.load(CAL / "ring_mask.npy").astype(bool)
    _G["roi"] = np.load(CAL / "roi_mask.npy").astype(bool)
    _G["rp"] = load_params(None).roi_pipeline
    _G["raw"], _G["out"] = Path(raw_dir), Path(out)


def process(lens: str) -> dict:
    rp = _G["rp"]
    raw = cv2.imread(str(_G["raw"] / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != _G["B"].shape:
        return {"lens": lens, "error": "原图读取失败或尺寸不符"}
    warp, ecc = register_to_template(
        raw, _G["B"], _G["ring"], max_scale_deviation=rp.max_scale_deviation,
        ecc_max_iter=rp.ecc_max_iterations, ecc_epsilon=rp.ecc_epsilon,
        downsample=rp.registration_downsample, pre_blur_sigma=rp.registration_pre_blur_sigma,
        ecc_score_threshold=rp.registration_ecc_score_threshold)
    aligned = apply_warp(raw.astype(np.float64), warp, _G["B"].shape)
    res = standardize(aligned, _G["B"], _G["roi"], _G["ring"])
    out = _G["out"]
    cv2.imwrite(str(out / "images" / f"{lens}.png"), np.clip(np.rint(res.standard), 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / "diffuse" / f"{lens}.png"), np.clip(np.rint(res.diffuse), 0, 255).astype(np.uint8))
    cv2.imwrite(str(out / "valid" / f"{lens}.png"), res.valid_mask.astype(np.uint8) * 255)
    return {"lens": lens, "ecc": round(float(ecc), 4), "warp": np.round(warp, 6).tolist(), **res.stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "output/dataset_v3")
    ap.add_argument("--lenses", nargs="*", help="只处理指定镜片（默认全部）")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not args.raw_dir.is_dir():
        raise SystemExit(f"原图目录不存在: {args.raw_dir}")
    lenses = args.lenses or sorted(p.stem for p in args.raw_dir.glob("*.png") if LENS_RE.match(p.stem))
    for sub in ("images", "diffuse", "valid"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(args.raw_dir), str(args.out))) as ex:
        for i, r in enumerate(ex.map(process, lenses), 1):
            rows.append(r)
            if i % 25 == 0 or "error" in r:
                print(f"[{i}/{len(lenses)}] {r['lens']} {r.get('error', '')}", flush=True)
    report_path = args.out / "report.json"
    old = {r["lens"]: r for r in json.loads(report_path.read_text())} if report_path.exists() else {}
    old.update({r["lens"]: r for r in rows})
    report_path.write_text(json.dumps(sorted(old.values(), key=lambda r: r["lens"]), ensure_ascii=False, indent=1))
    print(f"完成 {len(rows)} 张 → {args.out}")


if __name__ == "__main__":
    main()
