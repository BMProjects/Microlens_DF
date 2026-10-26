"""传统高召回像素分割 + 三智能助手语义判定实验。

子命令:
    build    对每张镜片做高召回分割与片段划分，选取 2×2 个 384 像素窗口生成评审包
    run      隔离调用评审员（R1、R2: claude-opus-5-5；R3: gpt-6.1-sol）
    analyze  合并窗口结果（相同编号跨窗口衔接）、像素归属、三人异同与传统规则对比
    （报告由 scripts/report_review_semantic.py 生成）

用法:
    uv run python scripts/review_semantic_pilot.py build --lenses 108l 26l 90r 76r 72l
    uv run python scripts/review_semantic_pilot.py run --workers 6
    uv run python scripts/review_semantic_pilot.py analyze
"""

from __future__ import annotations

import argparse
import colorsys
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from hr_segment import HRParams, hr_foreground, segments  # noqa: E402
from review_pixel_pilot import display, load_standard  # noqa: E402

OUT = PROJECT_ROOT / "output/review/semantic5"
PROMPT_DOC = PROJECT_ROOT / "doc/评审/语义标注_提示词_v4.md"
REPORT_V3 = PROJECT_ROOT / "output/dataset_v3/report.json"
WIN, OVL, SCALE, PAD = 384, 32, 2.5, 40
REGION = 2 * WIN - OVL                      # 736
OFFSETS = [(0, 0), (WIN - OVL, 0), (0, WIN - OVL), (WIN - OVL, WIN - OVL)]
MIN_LABEL_LEN, MIN_SPOT_AREA = 40, 12
PARAMS = HRParams(min_area=25)

RATERS = {
    "R1": {"tool": "claude", "model": "claude-opus-5-5", "effort": "high"},
    "R2": {"tool": "claude", "model": "claude-opus-5-5", "effort": "high"},
    "R3": {"tool": "codex", "model": "gpt-6.1-sol", "effort": "high"},
}


def id_color(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def labeled_ids(seg: np.ndarray, info: dict) -> np.ndarray:
    area = np.bincount(seg.ravel(), minlength=seg.max() + 1)
    ok = np.zeros(seg.max() + 1, bool)
    for i, v in info.items():
        i = int(i)
        ok[i] = (v["kind"] == "stroke" and v["len"] >= MIN_LABEL_LEN) or (v["kind"] == "spot" and area[i] >= MIN_SPOT_AREA)
    return ok


def ruler(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    c = np.full((h + PAD, w + PAD, 3), 30, np.uint8)
    c[PAD:, PAD:] = img
    for v in range(0, WIN + 1, 32):
        p = int(v * SCALE)
        long = v % 64 == 0
        cv2.line(c, (PAD + p, PAD - (14 if long else 6)), (PAD + p, PAD - 1), (255, 255, 255), 1)
        cv2.line(c, (PAD - (14 if long else 6), PAD + p), (PAD - 1, PAD + p), (255, 255, 255), 1)
        if long:
            cv2.putText(c, str(v), (PAD + p + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(c, str(v), (1, PAD + p + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return c


def choose_region(fg: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    dens = cv2.boxFilter(fg.astype(np.float32), -1, (REGION, REGION))
    vv = cv2.boxFilter(valid.astype(np.float32), -1, (REGION, REGION))
    dens[vv < 0.97] = -1
    y, x = np.unravel_index(np.argmax(dens), dens.shape)
    h, w = fg.shape
    return int(np.clip(x - REGION // 2, 0, w - REGION)), int(np.clip(y - REGION // 2, 0, h - REGION))


def cmd_build(args: argparse.Namespace) -> None:
    rep = {r["lens"]: r for r in json.loads(REPORT_V3.read_text())}
    rng = random.Random(args.seed)
    nums = rng.sample(range(10, 99), len(args.lenses))
    pkg_root = OUT / "packages"
    if pkg_root.exists():
        shutil.rmtree(pkg_root)
    (OUT / "seg").mkdir(parents=True, exist_ok=True)
    key = {}
    for n, lens in zip(nums, args.lenses):
        img, valid = load_standard(lens)
        fg = hr_foreground(img, valid, rep[lens]["noise_sigma"], PARAMS)
        seg, info = segments(fg, PARAMS)
        ok = labeled_ids(seg, info)
        rx, ry = choose_region(fg, valid)
        np.savez_compressed(OUT / "seg" / f"{lens}.npz", seg=seg, fg=fg)
        (OUT / "seg" / f"{lens}_info.json").write_text(json.dumps(
            {"info": info, "region": [rx, ry, REGION], "labeled": np.nonzero(ok)[0].tolist(),
             "sigma": rep[lens]["noise_sigma"]}))
        disp = display(img)
        disp[~valid] = 0
        ctx = cv2.cvtColor(cv2.resize(disp, (1024, 750), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        sx = 1024 / 4096
        for k, (dx, dy) in enumerate(OFFSETS):
            x0, y0 = rx + dx, ry + dy
            pid = f"P{n}w{k}"
            d = pkg_root / pid
            d.mkdir(parents=True)
            c = ctx.copy()
            cv2.rectangle(c, (int(rx * sx), int(ry * sx)), (int((rx + REGION) * sx), int((ry + REGION) * sx)),
                          (120, 120, 120), 1)
            cv2.rectangle(c, (int(x0 * sx), int(y0 * sx)), (int((x0 + WIN) * sx), int((y0 + WIN) * sx)),
                          (0, 230, 255), 2)
            cv2.imwrite(str(d / "context.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
            size = int(WIN * SCALE)
            raw = cv2.cvtColor(cv2.resize(disp[y0:y0 + WIN, x0:x0 + WIN], (size, size),
                                          interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(d / "raw.jpg"), ruler(raw), [cv2.IMWRITE_JPEG_QUALITY, 92])
            w = seg[y0:y0 + WIN, x0:x0 + WIN]
            wl = cv2.resize(w.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            over = raw.copy()
            ids = [int(i) for i in np.unique(w) if i > 0]
            shown = [i for i in ids if ok[i]]
            over[(wl > 0) & ~ok[wl]] = (110, 110, 110)
            for i in shown:
                over[wl == i] = id_color(i)
            v = cv2.addWeighted(over, 0.7, raw, 0.3, 0)
            for i in shown:
                yy, xx = np.nonzero(w == i)
                # 取片段在窗口内离边界最远的像素放编号，避免编号被窗口边缘裁断
                margin = np.minimum.reduce([xx, yy, WIN - 1 - xx, WIN - 1 - yy])
                k2 = int(np.argmax(margin))
                px = int(np.clip(xx[k2] * SCALE + 3, 2, size - 46))
                py = int(np.clip(yy[k2] * SCALE - 3, 14, size - 4))
                cv2.putText(v, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                cv2.putText(v, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(d / "segments.jpg"), ruler(v), [cv2.IMWRITE_JPEG_QUALITY, 92])
            key[pid] = {"lens": lens, "window": k, "origin": [x0, y0], "ids": shown,
                        "gray_fragments": len(ids) - len(shown)}
        print(lens, f"片段 {len(info)}，待判定 {int(ok.sum())}，区域 ({rx},{ry})，各窗口编号数",
              [len(key[f'P{n}w{k}']['ids']) for k in range(4)], flush=True)
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False))


def load_prompt() -> str:
    body = PROMPT_DOC.read_text(encoding="utf-8").split("\n## 提示词正文\n", 1)[1].strip()
    return body.replace("{WIN}", str(WIN)).replace("{SCALE}", str(SCALE)).replace("{WMAX}", str(WIN - 1))


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
    else:
        out_file = d / f".{rater}_last.txt"
        cmd = ["codex", "exec", "-m", cfg["model"], "-c", f'model_reasoning_effort="{cfg["effort"]}"',
               "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(d), "-o", str(out_file)]
        for name in ("context.jpg", "raw.jpg", "segments.jpg"):
            cmd += ["-i", str(d / name)]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(proc.stderr[-500:])
        text, cost = out_file.read_text(encoding="utf-8"), None
        out_file.unlink()
    return {"rater": rater, "pid": pid, "model": cfg["model"], "seconds": round(time.perf_counter() - t0, 1),
            "cost_usd": cost, "raw": text}


def cmd_run(args: argparse.Namespace) -> None:
    key = json.loads((OUT / "key.json").read_text())
    rdir = OUT / "ratings"
    rdir.mkdir(exist_ok=True)
    prompt = load_prompt()
    jobs = []
    for rater in args.raters:
        w = Path(args.workdir) / rater
        w.mkdir(parents=True, exist_ok=True)
        order = sorted(args.packages or key)
        random.Random(sum(map(ord, rater))).shuffle(order)
        for pid in order:
            f = rdir / f"{rater}_{pid}.json"
            if f.exists() and "parsed" in json.loads(f.read_text()) and not args.force:
                continue
            dst = w / pid
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(OUT / "packages" / pid, dst)
            jobs.append((rater, pid, dst))

    def job(j):
        rater, pid, d = j
        try:
            rec = run_one(rater, pid, d, prompt)
            try:
                rec["parsed"] = extract_json(rec["raw"])
            except (ValueError, json.JSONDecodeError) as exc:
                rec["parse_error"] = str(exc)
        except Exception as exc:
            rec = {"rater": rater, "pid": pid, "error": str(exc)[:800]}
        (rdir / f"{rater}_{pid}.json").write_text(json.dumps(rec, ensure_ascii=False))
        n = len((rec.get("parsed") or {}).get("labels") or [])
        print(rater, pid, rec.get("seconds"), rec.get("cost_usd"),
              "error" if "error" in rec else ("parse_error" if "parse_error" in rec else f"ok labels={n}"), flush=True)

    print(f"待运行 {len(jobs)} 个任务", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lenses", nargs="+", required=True)
    b.add_argument("--seed", type=int, default=2028)
    r = sub.add_parser("run")
    r.add_argument("--raters", nargs="+", default=list(RATERS))
    r.add_argument("--packages", nargs="*")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--force", action="store_true")
    r.add_argument("--workdir", default="/tmp/microlens_review_semantic")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "analyze":
        from review_semantic_analysis import analyze
        analyze(OUT)
        return
    {"build": cmd_build, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
