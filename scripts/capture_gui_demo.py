#!/usr/bin/env python3
"""采集 GUI 操作演示并合成 animated WebP。

流程（浏览器驱动系统 Google Chrome，复现「上传 → 检测 → 结果」）：
1. 打开已运行的 Gradio 应用（默认 http://127.0.0.1:7860）。
2. 上传一张镜片图像 → 点击「开始检测」→ 等待 2×2 结果面板就绪。
3. 在关键节点对应用容器截图为帧。
4. 用 Pillow 把帧合成 animated WebP，并另存关键帧 PNG。

依赖：playwright（channel="chrome" 复用系统 google-chrome，无需下载 chromium）、Pillow。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "doc" / "assets" / "generated"
FRAME_DIR = OUT_DIR / "_gui_demo_frames"
DEFAULT_IMAGE = ROOT / "output" / "dataset_v2" / "images" / "103l.png"
DEFAULT_URL = "http://127.0.0.1:7860"
VIEWPORT = {"width": 1360, "height": 840}


def _shot(locator, path: Path) -> Path:
    """对应用容器截图，避免浏览器外框。"""
    locator.screenshot(path=str(path))
    return path


def capture(url: str, image: Path, frame_dir: Path) -> list[tuple[Path, int]]:
    """返回 (帧路径, 停留毫秒) 列表。"""
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[Path, int]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.goto(url, wait_until="networkidle", timeout=60_000)
        app = page.locator("gradio-app")
        page.wait_for_timeout(1500)

        # ① 初始空界面
        frames.append((_shot(app, frame_dir / "01_initial.png"), 1600))

        # ② 上传图像
        page.locator("input[type=file]").first.set_input_files(str(image))
        # 等上传预览渲染 + 信息栏更新
        page.wait_for_timeout(2500)
        frames.append((_shot(app, frame_dir / "02_uploaded.png"), 1800))

        # ③ 点击开始检测
        page.get_by_role("button", name="开始检测").click()
        page.wait_for_timeout(700)
        frames.append((_shot(app, frame_dir / "03_detecting.png"), 900))

        # ④ 轮询直到评级卡片不再是「等待检测」
        deadline = time.time() + 120
        done = False
        while time.time() < deadline:
            page.wait_for_timeout(1000)
            txt = app.inner_text()
            if "等待检测" not in txt and ("评级" in txt or "WearScore" in txt):
                # 结果区已刷新（评级卡出现 A/B/C/D 或结论）
                if any(g in txt for g in ("极佳", "良好", "警告", "报废")):
                    done = True
                    break
        page.wait_for_timeout(800)
        final = _shot(app, frame_dir / "04_result.png")
        # 结果帧多停留（重复入帧）
        frames.append((final, 2600))
        frames.append((final, 2600))

        browser.close()

    if not done:
        print("[warn] 未确认检测完成（评级卡未出现等级标签），仍输出已采集帧。")
    return frames


def assemble_webp(frames: list[tuple[Path, int]], out_path: Path) -> None:
    """把帧合成 animated WebP（统一画布尺寸，避免抖动）。"""
    imgs = [(Image.open(p).convert("RGB"), d) for p, d in frames]
    max_w = max(im.width for im, _ in imgs)
    max_h = max(im.height for im, _ in imgs)
    canvas: list[Image.Image] = []
    durations: list[int] = []
    for im, d in imgs:
        bg = Image.new("RGB", (max_w, max_h), (13, 17, 23))
        bg.paste(im, ((max_w - im.width) // 2, (max_h - im.height) // 2))
        canvas.append(bg)
        durations.append(d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas[0].save(
        out_path,
        format="WEBP",
        save_all=True,
        append_images=canvas[1:],
        duration=durations,
        loop=0,
        quality=72,
        method=6,
    )
    print(f"[ok] {out_path}  ({out_path.stat().st_size/1024:.0f} KB, {len(canvas)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "gui_demo.webp")
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"样张不存在: {args.image}")

    frames = capture(args.url, args.image, FRAME_DIR)
    assemble_webp(frames, args.out)


if __name__ == "__main__":
    main()
