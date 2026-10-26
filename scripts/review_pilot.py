"""智能助手评审试点：5 张镜片的网格分割与分级，三助手独立评审 + 传统算法对照。

子命令:
    build        生成盲评评审包（概览图 + 4 张带网格的原分辨率象限图），镜片名替换为随机编号
    traditional  传统算法（噪声归一化 + 迟滞阈值 + 形态规则）在同一网格上的分割与分级
    run          在隔离会话中调用评审助手（R1/R2: Claude Code，R3: Codex CLI），只读评审包
    merge        汇总三助手与传统算法结果，计算一致性，输出报告

用法:
    uv run python scripts/review_pilot.py build --lenses 108l 26l 90r 76r 72l
    uv run python scripts/review_pilot.py traditional
    uv run python scripts/review_pilot.py run --raters R1 R2 R3
    uv run python scripts/review_pilot.py merge
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_DIR = PROJECT_ROOT / "output/dataset_v2/images"
ROI_PATH = PROJECT_ROOT / "output/dataset_v2/calibration/roi_mask.npy"
OUT = PROJECT_ROOT / "output/review/pilot5"
PROMPT_DOC = PROJECT_ROOT / "doc/评审/试点_五图分割分级_提示词.md"

QUADS = {"q1": (0, 0), "q2": (2048, 0), "q3": (0, 1500), "q4": (2048, 1500)}  # (x0, y0)
QW, QH, NCOL, NROW = 2048, 1500, 16, 12
CW, CH = QW // NCOL, QH // NROW  # 128 × 125 像素 ≈ 0.87 × 0.85 mm
COLS = "ABCDEFGHIJKLMNOP"
CLASSES = ("scratch", "spot", "damage")
GAIN = 4.0  # 显示增强倍数，与提示词一致
PAD = 44

RATERS = {
    "R1": {"tool": "claude", "model": "claude-opus-5-5", "effort": "high"},
    # Fable 5.1 在当前账号需额外用量额度，R2 改用与 R1 不同且可用的 Sonnet 5.5
    "R2": {"tool": "claude", "model": "claude-sonnet-5-5", "effort": "high"},
    "R3": {"tool": "codex", "model": "gpt-6.1-sol", "effort": "high"},
}


def zone_radii(h: int, w: int) -> tuple[tuple[int, int], float]:
    """与 WearScore 一致的功能区几何：中心取图像中心，半径基准 0.43·min(h, w)。"""
    return (w // 2, h // 2), min(h, w) * 0.43


def enhance(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) * GAIN, 0, 255).astype(np.uint8)


def _grid_image(quad: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(enhance(quad), cv2.COLOR_GRAY2BGR)
    overlay = vis.copy()
    for c in range(1, NCOL):
        cv2.line(overlay, (c * CW, 0), (c * CW, QH), (0, 140, 255), 1)
    for r in range(1, NROW):
        cv2.line(overlay, (0, r * CH), (QW, r * CH), (0, 140, 255), 1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)  # 半透明网格，不遮挡细线
    canvas = np.zeros((QH + PAD, QW + PAD, 3), np.uint8)
    canvas[PAD:, PAD:] = vis
    for c in range(NCOL):
        cv2.putText(canvas, COLS[c], (PAD + c * CW + CW // 2 - 9, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for r in range(NROW):
        cv2.putText(canvas, str(r + 1), (4, PAD + r * CH + CH // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def cmd_build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ids = [f"P{i}" for i in rng.sample(range(1, 100), len(args.lenses))]
    pkg_root = OUT / "packages"
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    key = {}
    for pid, lens in zip(ids, args.lenses):
        img = cv2.imread(str(IMAGE_DIR / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (3000, 4096):
            raise SystemExit(f"图像读取失败或尺寸不符: {lens}")
        d = pkg_root / pid
        d.mkdir(parents=True)
        (cx, cy), radius = zone_radii(*img.shape)
        ov = cv2.cvtColor(enhance(img), cv2.COLOR_GRAY2BGR)
        for frac, tag in ((0.40, "C"), (0.75, "M")):
            r = int(radius * frac)
            for a in range(0, 360, 6):  # 虚线圆
                cv2.ellipse(ov, (cx, cy), (r, r), 0, a, a + 3, (0, 200, 255), 6)
        cv2.putText(ov, "C", (cx - 20, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 6)
        cv2.putText(ov, "M", (cx + int(radius * 0.55), cy), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 6)
        cv2.putText(ov, "E", (cx + int(radius * 0.95), cy), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 200, 255), 6)
        cv2.imwrite(str(d / "overview.jpg"), cv2.resize(ov, (2000, 1465), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        for q, (x0, y0) in QUADS.items():
            cv2.imwrite(str(d / f"{q}.jpg"), _grid_image(img[y0:y0 + QH, x0:x0 + QW]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        key[pid] = lens
    (OUT / "key.json").write_text(json.dumps(key, indent=2))  # 不放进评审包
    print(f"评审包: {pkg_root}  编号: {key}")


# ── 传统算法对照（R0-lite）────────────────────────────────────────────

def traditional_segment(img: np.ndarray, roi: np.ndarray) -> dict[str, np.ndarray]:
    """噪声归一化 → 迟滞阈值 → 形态规则分类，返回三类二值掩膜。"""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.measure import label, regionprops

    f = img.astype(np.float32)
    small = cv2.resize(img, (1024, 750), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 51).astype(np.float32), (4096, 3000),
                    interpolation=cv2.INTER_LINEAR)
    res = f - bg
    v = res[roi]
    low = v[v <= np.percentile(v, 50)]
    sigma = max(1.4826 * float(np.median(np.abs(low - np.median(low)))), 1.0)  # uint8 量化下限
    z = res / sigma
    fg = apply_hysteresis_threshold(z, 3.0, 6.0) & roi
    lab = label(fg, connectivity=2)
    scratch = np.zeros_like(fg)
    spot = np.zeros_like(fg)
    for rp in regionprops(lab):
        if rp.area < 15:
            continue
        elong = rp.axis_major_length / max(rp.axis_minor_length, 1.0)
        target = spot if (rp.area <= 400 and elong < 3.0) else scratch
        sl = rp.slice
        target[sl] |= lab[sl] == rp.label
    # 碎裂：网格内前景占比 > 25% 的格子（交叠划痕与碎片无法再分辨单条）
    damage = np.zeros_like(fg)
    for y in range(0, 3000, CH):
        for x in range(0, 4096, CW):
            cell = fg[y:y + CH, x:x + CW]
            if cell.size and cell.mean() > 0.25:
                damage[y:y + CH, x:x + CW] = cell
    return {"scratch": scratch & ~damage, "spot": spot & ~damage, "damage": damage, "z_sigma": sigma}


def masks_to_cells(masks: dict, min_px: int = 20) -> dict:
    out = {}
    for q, (x0, y0) in QUADS.items():
        out[q] = {}
        for cls in CLASSES:
            m = masks[cls][y0:y0 + QH, x0:x0 + QW]
            cells = []
            for r in range(NROW):
                for c in range(NCOL):
                    if m[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW].sum() >= min_px:
                        cells.append(f"{COLS[c]}{r + 1}")
            out[q][cls] = cells
    return out


def traditional_grade(masks: dict, img: np.ndarray) -> tuple[float, str]:
    """把传统实例换成外接框，走与 GUI 主线相同的 WearScore 链路。"""
    from skimage.measure import label, regionprops

    from darkfield_defects.app_services.inference_service import boxes_to_detection_result
    from darkfield_defects.scoring.quantify import compute_wear_metrics
    from darkfield_defects.scoring.wear_score import compute_wear_score

    boxes = []
    for ci, cls in enumerate(CLASSES):
        for rp in regionprops(label(masks[cls], connectivity=2)):
            if rp.area >= 15:
                y1, x1, y2, x2 = rp.bbox
                boxes.append((ci, float(x1), float(y1), float(x2), float(y2), 1.0))
    det, roi = boxes_to_detection_result(boxes, img)
    a = compute_wear_score(compute_wear_metrics(det, roi))
    return round(float(a.score), 1), a.grade


def cmd_traditional(args: argparse.Namespace) -> None:
    logging.disable(logging.INFO)
    key = json.loads((OUT / "key.json").read_text())
    roi = np.load(ROI_PATH)
    result = {}
    for pid, lens in key.items():
        img = cv2.imread(str(IMAGE_DIR / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        masks = traditional_segment(img, roi)
        score, grade = traditional_grade(masks, img)
        result[pid] = {"cells": masks_to_cells(masks), "grade": grade, "score": score,
                       "sigma": round(masks["z_sigma"], 2),
                       "seconds": round(time.perf_counter() - t0, 1)}
        vis = cv2.cvtColor(enhance(img), cv2.COLOR_GRAY2BGR)
        for cls, color in zip(CLASSES, ((60, 200, 30), (0, 170, 255), (255, 80, 200))):
            vis[masks[cls]] = color
        cv2.imwrite(str(OUT / f"traditional_{pid}.jpg"),
                    cv2.resize(vis, (1365, 1000), interpolation=cv2.INTER_AREA))
        print(pid, lens, grade, score, f"σ={masks['z_sigma']:.2f}")
    (OUT / "traditional.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))


# ── 评审助手调用 ──────────────────────────────────────────────────────

def load_prompt() -> str:
    text = PROMPT_DOC.read_text(encoding="utf-8")
    return text.split("\n## 提示词正文\n", 1)[1].strip()


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("输出中未找到 JSON")
    return json.loads(m.group(0))


def run_one(rater: str, pid: str, work: Path, prompt: str) -> dict:
    cfg = RATERS[rater]
    d = work / pid
    t0 = time.perf_counter()
    if cfg["tool"] == "claude":
        cmd = ["claude", "-p", prompt, "--model", cfg["model"], "--effort", cfg["effort"],
               "--allowedTools", "Read", "--output-format", "json", "--no-session-persistence"]
        proc = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800)
        meta = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
        if proc.returncode != 0 or meta.get("is_error"):
            raise RuntimeError(f"{rater}/{pid} 失败: {meta.get('result') or proc.stderr[-500:]}")
        text, cost = meta.get("result", ""), meta.get("total_cost_usd")
    else:
        out_file = d / f".{rater}_last.txt"
        cmd = ["codex", "exec", "-m", cfg["model"], "-c", f'model_reasoning_effort="{cfg["effort"]}"',
               "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(d),
               "-o", str(out_file)]
        for name in ("overview.jpg", "q1.jpg", "q2.jpg", "q3.jpg", "q4.jpg"):
            cmd += ["-i", str(d / name)]
        # -i 接受多个值，位置参数会被当作图片；提示词改由标准输入传入
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(f"{rater}/{pid} 失败: {proc.stderr[-500:]}")
        text, cost = out_file.read_text(encoding="utf-8"), None
        out_file.unlink()
    return {"rater": rater, "pid": pid, "model": cfg["model"], "seconds": round(time.perf_counter() - t0, 1),
            "cost_usd": cost, "raw": text}


def cmd_run(args: argparse.Namespace) -> None:
    prompt = load_prompt()
    pkgs = sorted(p.name for p in (OUT / "packages").iterdir())
    # 每名评审员一份独立副本，放在项目目录之外，避免读到项目记忆、其他评审结果或算法输出
    work_root = Path(args.workdir)
    jobs = []
    for rater in args.raters:
        w = work_root / rater
        if w.exists():
            shutil.rmtree(w)
        shutil.copytree(OUT / "packages", w)
        order = pkgs[:]
        random.Random(hash(rater) & 0xFFFF).shuffle(order)  # 各评审员顺序独立打乱
        jobs += [(rater, pid, w) for pid in order]
    rdir = OUT / "ratings"
    rdir.mkdir(exist_ok=True)

    def job(j):
        rater, pid, w = j
        try:
            rec = run_one(rater, pid, w, prompt)
            try:
                rec["parsed"] = extract_json(rec["raw"])
            except (ValueError, json.JSONDecodeError) as exc:
                rec["parse_error"] = str(exc)
        except Exception as exc:  # 单次失败记录后继续
            rec = {"rater": rater, "pid": pid, "error": str(exc)}
        (rdir / f"{rater}_{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(rater, pid, rec.get("seconds"), "error" if "error" in rec or "parse_error" in rec else "ok",
              flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, jobs))


# ── 汇总 ──────────────────────────────────────────────────────────────

def cell_set(cells: dict, cls: str) -> set[str]:
    return {f"{q}:{c}" for q in QUADS for c in (cells.get(q, {}).get(cls) or [])}


def jaccard(a: set, b: set) -> float | None:
    return None if not a and not b else len(a & b) / len(a | b)


def cmd_merge(args: argparse.Namespace) -> None:
    key = json.loads((OUT / "key.json").read_text())
    trad = json.loads((OUT / "traditional.json").read_text())
    ratings: dict = {}
    for f in sorted((OUT / "ratings").glob("*.json")):
        rec = json.loads(f.read_text())
        if "parsed" in rec:
            ratings.setdefault(rec["pid"], {})[rec["rater"]] = rec
    report = {"lenses": {}, "summary": {}}
    pair_j = {cls: [] for cls in CLASSES}
    maj_vs_trad = {cls: [] for cls in CLASSES}
    grade_agree = []
    for pid, lens in key.items():
        rs = ratings.get(pid, {})
        row = {"lens": lens, "traditional_grade": trad[pid]["grade"], "traditional_score": trad[pid]["score"],
               "grades": {r: rs[r]["parsed"].get("grade") for r in rs},
               "zone_severity": {r: rs[r]["parsed"].get("zone_severity") for r in rs},
               "n_long_scratches": {r: len(rs[r]["parsed"].get("long_scratches") or []) for r in rs},
               "seconds": {r: rs[r].get("seconds") for r in rs},
               "cells": {}}
        gs = [g for g in row["grades"].values() if g]
        row["grade_majority"] = max(set(gs), key=gs.count) if gs and gs.count(max(set(gs), key=gs.count)) >= 2 else None
        grade_agree.append(len(set(gs)) == 1 and len(gs) == len(rs) and len(rs) >= 2)
        for cls in CLASSES:
            sets = {r: cell_set(rs[r]["parsed"].get("cells", {}), cls) for r in rs}
            tset = cell_set(trad[pid]["cells"], cls)
            counts: dict[str, int] = {}
            for s in sets.values():
                for c in s:
                    counts[c] = counts.get(c, 0) + 1
            maj = {c for c, n in counts.items() if n >= 2}
            pj = [jaccard(sets[a], sets[b]) for a, b in itertools.combinations(sorted(sets), 2)]
            pj = [x for x in pj if x is not None]
            pair_j[cls] += pj
            mj = jaccard(maj, tset)
            if mj is not None:
                maj_vs_trad[cls].append(mj)
            row["cells"][cls] = {"n_cells": {r: len(s) for r, s in sets.items()}, "traditional": len(tset),
                                 "majority": len(maj),
                                 "pairwise_jaccard": [round(x, 3) for x in pj],
                                 "majority_vs_traditional_jaccard": None if mj is None else round(mj, 3)}
        report["lenses"][pid] = row
    report["summary"] = {
        "grade_all_agree": f"{sum(grade_agree)}/{len(grade_agree)}",
        "mean_pairwise_jaccard": {c: round(float(np.mean(v)), 3) if v else None for c, v in pair_j.items()},
        "mean_majority_vs_traditional_jaccard": {c: round(float(np.mean(v)), 3) if v else None
                                                 for c, v in maj_vs_trad.items()},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lenses", nargs="+", required=True)
    b.add_argument("--seed", type=int, default=2026)
    sub.add_parser("traditional")
    r = sub.add_parser("run")
    r.add_argument("--raters", nargs="+", default=list(RATERS))
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--workdir", default="/tmp/microlens_review_pilot")
    sub.add_parser("merge")
    args = ap.parse_args()
    {"build": cmd_build, "traditional": cmd_traditional, "run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
