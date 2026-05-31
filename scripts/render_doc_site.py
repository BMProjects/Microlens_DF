#!/usr/bin/env python3
"""批量渲染 doc 目录文档为 HTML，并生成统一目录页。"""

from __future__ import annotations

import html
import subprocess
from pathlib import Path

from render_research_logs_html import render_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PROJECT_ROOT / "doc"


def categorize(path: Path) -> str:
    name = path.name
    if name.startswith("研究实验日志_"):
        return "研究实验日志"
    if "测试" in name:
        return "测试与交付"
    if "执行" in name or "规划" in name or "准备" in name:
        return "执行与规划"
    if "技术" in name or "说明" in name or "研究报告" in name or "深度研究" in name:
        return "技术研究与系统文档"
    return "其他文档"


def build_nav(current: Path, md_docs: list[Path]) -> str:
    category = categorize(current)
    peers = [p for p in md_docs if p != current and categorize(p) == category][:4]
    links = ['<a href="index.html">文档总目录</a>']
    if current.with_suffix(".md").exists():
        links.append(f'<a href="{html.escape(current.name)}">Markdown 源文档</a>')
    for peer in peers:
        links.append(f'<a href="{html.escape(peer.with_suffix(".html").name)}">{html.escape(peer.stem)}</a>')
    return f'<nav class="doc-nav">{"".join(links)}</nav>'


def try_convert_docx(docx_path: Path) -> Path | None:
    html_path = docx_path.with_suffix(".html")
    if html_path.exists():
        return html_path

    soffice = subprocess.run(
        ["which", "soffice"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if soffice.returncode != 0:
        return None

    profile = DOC_DIR / ".lo-profile-doc-site"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        "soffice",
        "--headless",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to",
        "html:XHTML Writer File:UTF8",
        "--outdir",
        str(DOC_DIR),
        str(docx_path),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if proc.returncode == 0 and html_path.exists():
        return html_path

    wrapper = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(docx_path.stem)}</title>
  <style>
    body {{ margin:0; background:#f5f7fb; color:#0f172a; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
    .page {{ max-width:1000px; margin:32px auto; background:#fff; padding:36px 42px; border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.08); }}
    .doc-nav {{ display:flex; gap:10px; margin-bottom:20px; }}
    .doc-nav a {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#fff; border:1px solid #bfdbfe; color:#1d4ed8; text-decoration:none; }}
    .notice {{ background:#eff6ff; border:1px solid #dbeafe; padding:16px 18px; border-radius:14px; line-height:1.8; color:#334155; }}
  </style>
</head>
<body>
  <main class="page">
    <nav class="doc-nav">
      <a href="index.html">文档总目录</a>
      <a href="{html.escape(docx_path.name)}">下载 Word 原件</a>
    </nav>
    <h1>{html.escape(docx_path.stem)}</h1>
    <div class="notice">
      当前文档以 Word 交付件形式维护，自动 HTML 直转未成功完成，因此这里提供统一 HTML 入口页。
      后续如需进一步插入图表与结果图，建议将其同步维护到 Markdown/HTML 源稿后再生成正式 Word 版本。
    </div>
  </main>
</body>
</html>
"""
    html_path.write_text(wrapper, encoding="utf-8")
    return html_path


def build_index(md_docs: list[Path], html_from_docx: list[Path]) -> str:
    sections: dict[str, list[str]] = {}

    for doc in md_docs:
        sections.setdefault(categorize(doc), []).append(
            f'<li><a href="{html.escape(doc.with_suffix(".html").name)}">{html.escape(doc.stem)}</a>'
            f' <span style="color:#64748b">| 源文档: {html.escape(doc.name)}</span></li>'
        )

    if html_from_docx:
        sections.setdefault("Word 交付件", [])
        for doc in html_from_docx:
            sections["Word 交付件"].append(
                f'<li><a href="{html.escape(doc.name)}">{html.escape(doc.stem)}</a></li>'
            )

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        "<title>Microlens_DF 文档总目录</title>",
        "<style>",
        "body{margin:0;background:#f5f7fb;color:#0f172a;font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;}",
        ".page{max-width:1200px;margin:28px auto;background:#fff;padding:36px 42px 60px;border-radius:18px;box-shadow:0 8px 24px rgba(15,23,42,.08);}",
        "h1{margin-top:0;font-size:2rem;border-bottom:3px solid #dbeafe;padding-bottom:12px;}",
        "h2{margin-top:28px;border-left:5px solid #60a5fa;padding-left:12px;font-size:1.3rem;}",
        "p,li{line-height:1.75;font-size:15px;} ul{padding-left:20px;} a{color:#2563eb;text-decoration:none;} a:hover{text-decoration:underline;}",
        ".intro{color:#475569;background:#eff6ff;border:1px solid #dbeafe;padding:16px 18px;border-radius:14px;}",
        "</style></head><body><main class='page'>",
        "<h1>Microlens_DF 文档总目录</h1>",
        "<p class='intro'>本页统一索引当前项目的研究日志、技术文档、执行规划与 Word 交付件 HTML 版本，便于在同一目录下补充图像结果、分析图和表格，并通过 HTML 进行交叉查阅。</p>",
    ]
    for section, items in sections.items():
        parts.append(f"<h2>{html.escape(section)}</h2><ul>")
        parts.extend(items)
        parts.append("</ul>")
    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> None:
    md_docs = sorted(DOC_DIR.glob("*.md"))
    rendered = []
    for doc in md_docs:
        nav_html = build_nav(doc, md_docs)
        rendered.append(render_file(doc, nav_html=nav_html))

    html_from_docx: list[Path] = []
    for docx in sorted(DOC_DIR.glob("*.docx")):
        converted = try_convert_docx(docx)
        if converted is not None:
            html_from_docx.append(converted)

    index_path = DOC_DIR / "index.html"
    index_path.write_text(build_index(md_docs, html_from_docx), encoding="utf-8")

    print("Rendered Markdown HTML:")
    for path in rendered:
        print(path)
    print("Rendered Word HTML:")
    for path in html_from_docx:
        print(path)
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
