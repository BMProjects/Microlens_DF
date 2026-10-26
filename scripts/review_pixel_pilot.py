"""智能助手像素级缺陷标注实验：5 张镜片 × 4 象限 × 3 名评审员，传统算法像素分割仅作参考。

助手输出像素坐标的矢量标注（划痕中心线折线 + 线宽、斑点圆、碎裂多边形），脚本栅格化为像素掩膜，
并执行跨象限划痕连续性检测（按端点位置与走向拼接相邻象限的划痕）。

子命令:
    build      由 dataset_v3 标准输入生成象限评审包（原分辨率 + 坐标标尺，每包一张图）
    run        隔离调用评审助手（R1/R2: Claude Code，R3: Codex CLI）
    analyze    转全局坐标、跨象限拼接、栅格化、吸附精修、统计三助手异同与传统参考
    （报告由 scripts/report_review_pixel.py 生成）

用法:
    uv run python scripts/review_pixel_pilot.py build --lenses 108l 26l 90r 76r 72l
    uv run python scripts/review_pixel_pilot.py run --raters R1 R2 R3 --workers 6
    uv run python scripts/review_pixel_pilot.py analyze
"""

from __future__ import annotations

import argparse
import json
import os
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

STD_DIR = PROJECT_ROOT / "output/dataset_v3"
OUT = PROJECT_ROOT / "output/review/pixel5"
PROMPT_DOC = PROJECT_ROOT / "doc/评审/像素级划痕标注_提示词_v3.md"
PIXEL_MM = 0.0068

# 象限（全局坐标 x0, y0, x1, y1，右开）：ROI 外接框 x∈[140, 3900)、y∈[0, 3000)，中线两侧各 48 像素重叠
QUADS = {
    "q1": (140, 0, 2068, 1548),
    "q2": (1972, 0, 3900, 1548),
    "q3": (140, 1452, 2068, 3000),
    "q4": (1972, 1452, 3900, 3000),
}
Q_CN = {"q1": "左上象限", "q2": "右上象限", "q3": "左下象限", "q4": "右下象限"}
INNER = {"q1": ("right", "bottom"), "q2": ("left", "bottom"), "q3": ("right", "top"), "q4": ("left", "top")}
EDGE_CN = {"right": "右侧", "left": "左侧", "top": "上侧", "bottom": "下侧"}
QW, QH = 1928, 1548
PAD = 50

RATERS = {
    "R1": {"tool": "claude", "model": "claude-opus-5-5", "effort": "high"},
    "R2": {"tool": "claude", "model": "claude-sonnet-5-5", "effort": "high"},
    "R3": {"tool": "codex", "model": "gpt-6.1-sol", "effort": "high"},
}


def display(img: np.ndarray) -> np.ndarray:
    """平方根拉伸：1→16，4→32，16→64，64→128，255→255。"""
    return np.clip(16.0 * np.sqrt(img.astype(np.float32)), 0, 255).astype(np.uint8)


def with_ruler(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    canvas = np.full((h + PAD, w + PAD, 3), 30, np.uint8)
    canvas[PAD:, PAD:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x in range(0, w + 1, 50):
        long = x % 200 == 0
        cv2.line(canvas, (PAD + x, PAD - (18 if long else 8)), (PAD + x, PAD - 1), (255, 255, 255), 1)
        if long:
            cv2.putText(canvas, str(x), (PAD + x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for y in range(0, h + 1, 50):
        long = y % 200 == 0
        cv2.line(canvas, (PAD - (18 if long else 8), PAD + y), (PAD - 1, PAD + y), (255, 255, 255), 1)
        if long:
            cv2.putText(canvas, str(y), (2, PAD + y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def load_standard(lens: str) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(STD_DIR / "images" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(STD_DIR / "valid" / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or valid is None:
        raise SystemExit(f"标准输入缺失: {lens}")
    return img, valid > 0


def cmd_build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ids = rng.sample(range(10, 99), len(args.lenses))
    pkg_root = OUT / "packages"
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    key = {}
    for n, lens in zip(ids, args.lenses):
        img, valid = load_standard(lens)
        disp = display(img)
        disp[~valid] = 0
        for q, (x0, y0, x1, y1) in QUADS.items():
            pid = f"P{n}{q}"
            d = pkg_root / pid
            d.mkdir(parents=True)
            cv2.imwrite(str(d / "quadrant.jpg"), with_ruler(disp[y0:y1, x0:x1]), [cv2.IMWRITE_JPEG_QUALITY, 93])
            key[pid] = {"lens": lens, "quadrant": q}
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(f"评审包 {len(key)} 个: {pkg_root}")


def load_prompt(q: str) -> str:
    body = PROMPT_DOC.read_text(encoding="utf-8").split("\n## 提示词正文\n", 1)[1].strip()
    inner = "、".join(EDGE_CN[e] for e in INNER[q])
    return (body.replace("{QUADRANT_CN}", Q_CN[q]).replace("{QUADRANT}", q)
            .replace("{INNER_EDGES}", inner).replace("{W}", str(QW - 1)).replace("{H}", str(QH - 1)))


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("输出中未找到 JSON")
    return json.loads(m.group(0))


def run_one(rater: str, pid: str, d: Path, prompt: str) -> dict:
    cfg = RATERS[rater]
    t0 = time.perf_counter()
    if cfg["tool"] == "claude":
        env = dict(os.environ, CLAUDE_CODE_MAX_OUTPUT_TOKENS="64000")
        cmd = ["claude", "-p", prompt, "--model", cfg["model"], "--effort", cfg["effort"],
               "--allowedTools", "Read", "--output-format", "json", "--no-session-persistence"]
        proc = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=3600, env=env)
        meta = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
        if proc.returncode != 0 or meta.get("is_error"):
            raise RuntimeError(meta.get("result") or proc.stderr[-500:])
        text, cost = meta.get("result", ""), meta.get("total_cost_usd")
        out_tokens = (meta.get("usage") or {}).get("output_tokens")
    else:
        out_file = d / f".{rater}_last.txt"
        cmd = ["codex", "exec", "-m", cfg["model"], "-c", f'model_reasoning_effort="{cfg["effort"]}"',
               "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(d),
               "-o", str(out_file), "-i", str(d / "quadrant.jpg")]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(proc.stderr[-500:])
        text, cost, out_tokens = out_file.read_text(encoding="utf-8"), None, None
        out_file.unlink()
    return {"rater": rater, "pid": pid, "model": cfg["model"], "seconds": round(time.perf_counter() - t0, 1),
            "cost_usd": cost, "output_tokens": out_tokens, "raw": text}


def cmd_run(args: argparse.Namespace) -> None:
    key = json.loads((OUT / "key.json").read_text())
    pkgs = sorted(args.packages or key)
    rdir = OUT / "ratings"
    rdir.mkdir(exist_ok=True)
    work_root = Path(args.workdir)
    jobs = []
    for rater in args.raters:
        w = work_root / rater
        w.mkdir(parents=True, exist_ok=True)
        order = pkgs[:]
        random.Random(sum(map(ord, rater))).shuffle(order)
        for pid in order:
            if (rdir / f"{rater}_{pid}.json").exists() and not args.force:
                rec = json.loads((rdir / f"{rater}_{pid}.json").read_text())
                if "parsed" in rec:
                    continue
            dst = w / pid
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(OUT / "packages" / pid, dst)
            jobs.append((rater, pid, dst))

    def job(j):
        rater, pid, d = j
        try:
            rec = run_one(rater, pid, d, load_prompt(key[pid]["quadrant"]))
            try:
                rec["parsed"] = extract_json(rec["raw"])
            except (ValueError, json.JSONDecodeError) as exc:
                rec["parse_error"] = str(exc)
        except Exception as exc:  # 记录后继续
            rec = {"rater": rater, "pid": pid, "error": str(exc)[:800]}
        (rdir / f"{rater}_{pid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        n = len((rec.get("parsed") or {}).get("scratches") or [])
        status = "error" if "error" in rec else ("parse_error" if "parse_error" in rec else f"ok scratches={n}")
        print(rater, pid, rec.get("seconds"), status, flush=True)

    print(f"待运行 {len(jobs)} 个任务", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lenses", nargs="+", required=True)
    b.add_argument("--seed", type=int, default=2027)
    r = sub.add_parser("run")
    r.add_argument("--raters", nargs="+", default=list(RATERS))
    r.add_argument("--packages", nargs="*")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--force", action="store_true")
    r.add_argument("--workdir", default="/tmp/microlens_review_pixel")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "analyze":
        from review_pixel_analysis import analyze  # 分析部分单独成文件
        analyze(OUT)
        return
    {"build": cmd_build, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
