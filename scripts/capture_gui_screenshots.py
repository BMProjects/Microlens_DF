"""驱动图形界面完成一遍单张检测、批量检测和检测记录操作，截取用户手册插图。

前提：GUI 已启动（uv run python -m darkfield_defects.viz.app），本机有 Google Chrome。
截图写入 doc/软著/assets/，文件名与 doc/软著/用户手册.md 中的引用一致。

用法:
    uv run --with playwright python scripts/capture_gui_screenshots.py \
        --image output/dataset_v2/images/123l.png \
        --batch output/dataset_v2/images/84r.png output/dataset_v2/images/100l.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "doc" / "软著" / "assets"
DETECT_TIMEOUT_MS = 600_000


def shot(page: Page, name: str, selector: str | None = None) -> None:
    path = ASSETS / name
    if selector:
        page.locator(selector).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=True)
    print(f"  {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--image", type=Path, required=True, help="单张检测示例图像")
    ap.add_argument("--batch", type=Path, nargs="+", required=True, help="批量检测示例图像")
    args = ap.parse_args()
    for p in [args.image, *args.batch]:
        if not p.is_file():
            raise SystemExit(f"图像不存在: {p}")
    ASSETS.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("#single-left")
        shot(page, "fig_main.png")

        # ── 单张检测 ──
        page.locator("#single-left input[type=file]").first.set_input_files(str(args.image))
        page.get_by_text(re.compile(r"\d+ × \d+ px")).first.wait_for()
        page.wait_for_function(
            "() => !document.querySelector('#single-left').innerText.includes('processing')")
        page.wait_for_timeout(1000)
        shot(page, "fig_upload.png", "#single-left")
        page.get_by_text("高级设置").first.click()
        page.wait_for_timeout(500)
        shot(page, "fig_settings.png", "#single-left")

        page.get_by_role("button", name="开始检测").click()
        page.get_by_text("检测完成").first.wait_for(timeout=DETECT_TIMEOUT_MS)
        page.wait_for_timeout(1500)
        shot(page, "fig_result.png")
        for name, sel in (("fig_overlay.png", "#overlay-panel"), ("fig_topo.png", "#topo-panel"),
                          ("fig_chart.png", "#chart-panel"), ("fig_grade.png", "#grade-panel")):
            shot(page, name, sel)

        # ── 检测报告（最新一条记录的 HTML）──
        runs_root = Path(os.environ.get("DARKFIELD_RUNS_DIR", PROJECT_ROOT / "output" / "app_runs"))
        runs = sorted(runs_root.glob("*/report.html"))
        if not runs:
            raise SystemExit("未找到检测记录 report.html")
        report = browser.new_page(viewport={"width": 1100, "height": 1400})
        report.goto(runs[-1].as_uri())
        report.wait_for_timeout(500)
        report.screenshot(path=str(ASSETS / "fig_report.png"), full_page=False)
        print("  doc/软著/assets/fig_report.png")
        report.close()

        # ── 批量检测 ──
        page.get_by_role("tab", name="批量检测").click()
        page.wait_for_timeout(500)
        batch_input = page.locator("input[type=file][multiple]").first
        batch_input.set_input_files([str(p) for p in args.batch])
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="批量检测").click()
        page.get_by_text(f"完成 {len(args.batch)} 张").first.wait_for(timeout=DETECT_TIMEOUT_MS)
        page.wait_for_timeout(1500)
        shot(page, "fig_batch.png")

        # ── 检测记录 ──
        page.get_by_role("tab", name="检测记录").click()
        page.get_by_role("button", name="刷新记录").click()
        page.wait_for_timeout(1500)
        page.get_by_label("查看记录").click()
        page.get_by_role("option").first.click()
        page.wait_for_timeout(2500)
        shot(page, "fig_history.png")
        browser.close()
    compose_workflow()


def compose_workflow() -> None:
    """把导入、设置、检测结果、报告四张截图拼成一张带步骤标注的完整流程示例图。"""
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font = (ImageFont.truetype(font_path, 34) if Path(font_path).exists()
            else ImageFont.load_default())
    steps = [("① 导入镜片图像", "fig_upload.png"), ("② 设置置信度阈值", "fig_settings.png"),
             ("③ 开始检测，查看四联结果", "fig_result.png"), ("④ 下载检测报告", "fig_report.png")]
    imgs = [Image.open(ASSETS / f).convert("RGB") for _, f in steps]
    row_h = 900
    scaled = [im.resize((round(im.width * row_h / im.height), row_h)) for im in imgs]
    # 第一行：导入 + 设置（窄）+ 检测结果（宽）；第二行：报告居中
    gap, title_h = 40, 70
    top = scaled[:3]
    width = sum(im.width for im in top) + gap * (len(top) + 1)
    bottom = scaled[3]
    height = title_h * 2 + row_h * 2 + gap * 3
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = gap
    for (label, _), im in zip(steps[:3], top):
        draw.text((x, gap // 2), label, fill=(21, 101, 192), font=font)
        canvas.paste(im, (x, gap // 2 + title_h))
        draw.rectangle([x - 1, gap // 2 + title_h - 1, x + im.width, gap // 2 + title_h + row_h],
                       outline=(160, 170, 185), width=2)
        x += im.width + gap
    y2 = gap // 2 + title_h + row_h + gap
    bx = (width - bottom.width) // 2
    draw.text((bx, y2), steps[3][0], fill=(21, 101, 192), font=font)
    canvas.paste(bottom, (bx, y2 + title_h))
    draw.rectangle([bx - 1, y2 + title_h - 1, bx + bottom.width, y2 + title_h + row_h],
                   outline=(160, 170, 185), width=2)
    canvas.save(ASSETS / "fig_workflow.png", optimize=True)
    print("  doc/软著/assets/fig_workflow.png")


if __name__ == "__main__":
    main()
