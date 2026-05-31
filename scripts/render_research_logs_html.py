#!/usr/bin/env python3
"""将研究实验日志 Markdown 渲染为便于插图与表格扩展的 HTML."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


DOC_DIR = Path(__file__).resolve().parent.parent / "doc"
DEFAULT_GLOB = "研究实验日志_*.md"

CSS = """
body {
  margin: 0;
  background: #f5f7fb;
  color: #1f2937;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}
.page {
  max-width: 1100px;
  margin: 32px auto;
  background: #ffffff;
  padding: 36px 48px 56px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  border-radius: 18px;
}
.doc-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
  padding: 14px 16px;
  border-radius: 14px;
  background: #eff6ff;
  border: 1px solid #dbeafe;
}
.doc-nav a {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: #ffffff;
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  font-size: 13px;
}
h1, h2, h3, h4 {
  color: #0f172a;
  line-height: 1.35;
}
h1 { font-size: 2rem; border-bottom: 3px solid #dbeafe; padding-bottom: 12px; }
h2 { font-size: 1.45rem; margin-top: 2rem; border-left: 5px solid #60a5fa; padding-left: 12px; }
h3 { font-size: 1.15rem; margin-top: 1.4rem; }
p, li { font-size: 15px; line-height: 1.8; }
code {
  background: #eff6ff;
  padding: 2px 6px;
  border-radius: 6px;
  font-family: "JetBrains Mono", "Consolas", monospace;
}
pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 16px 18px;
  border-radius: 12px;
  overflow-x: auto;
  line-height: 1.55;
}
pre code { background: transparent; color: inherit; padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 24px;
  font-size: 14px;
}
th, td {
  border: 1px solid #dbe2ea;
  padding: 10px 12px;
  vertical-align: top;
}
th {
  background: #eff6ff;
  font-weight: 700;
}
tr:nth-child(even) td {
  background: #fafcff;
}
img {
  max-width: 100%;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
}
figure {
  margin: 24px 0;
}
figcaption {
  color: #475569;
  font-size: 14px;
  margin-top: 8px;
}
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px solid #dbe2ea; margin: 28px 0; }
.meta {
  color: #64748b;
  margin-bottom: 24px;
  font-size: 14px;
}
"""


def convert_inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    def image_repl(match: re.Match[str]) -> str:
        alt = match.group(1).strip()
        src = match.group(2).strip()
        return f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}" /><figcaption>{html.escape(alt)}</figcaption></figure>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_repl, text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2).strip())}">{m.group(1).strip()}</a>',
        text,
    )
    return text


def render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        stripped = line.strip().strip("|")
        cells = [convert_inline(cell.strip()) for cell in stripped.split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return ""
    header = rows[0]
    body = rows[2:] if len(rows) > 2 else []
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{cell}</th>" for cell in header)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def markdown_to_html(markdown_text: str, title: str, nav_html: str = "") -> str:
    lines = markdown_text.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_kind: str | None = None
    code_lines: list[str] = []
    in_code = False
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{convert_inline(' '.join(s.strip() for s in paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items, list_kind
        if list_items and list_kind:
            tag = "ol" if list_kind == "ol" else "ul"
            parts.append(f"<{tag}>")
            parts.extend(f"<li>{item}</li>" for item in list_items)
            parts.append(f"</{tag}>")
            list_items = []
            list_kind = None

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            table_html = render_table(table_lines)
            if table_html:
                parts.append(table_html)
            table_lines = []

    for line in lines:
        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            if in_code:
                parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        if stripped.startswith("|") and "|" in stripped:
            flush_paragraph()
            flush_list()
            table_lines.append(stripped)
            continue
        flush_table()

        if stripped == "---":
            flush_paragraph()
            flush_list()
            parts.append("<hr />")
            continue

        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if match:
            flush_paragraph()
            flush_list()
            level = len(match.group(1))
            parts.append(f"<h{level}>{convert_inline(match.group(2).strip())}</h{level}>")
            continue

        ul_match = re.match(r"^[-*]\s+(.*)$", stripped)
        ol_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ul_match or ol_match:
            flush_paragraph()
            kind = "ul" if ul_match else "ol"
            item = convert_inline((ul_match or ol_match).group(1).strip())
            if list_kind not in (None, kind):
                flush_list()
            list_kind = kind
            list_items.append(item)
            continue

        flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    if in_code:
        parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")

    body_html = "\n".join(parts)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <main class="page">
    {nav_html}
    <div class="meta">自动渲染的 HTML 研究日志，适合继续补充图片、表格与结果对比图。</div>
    {body_html}
  </main>
</body>
</html>
"""


def render_file(path: Path, nav_html: str = "") -> Path:
    output = path.with_suffix(".html")
    html_text = markdown_to_html(path.read_text(encoding="utf-8"), path.stem, nav_html=nav_html)
    output.write_text(html_text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="批量将研究日志 Markdown 渲染为 HTML")
    parser.add_argument("files", nargs="*", help="指定 Markdown 文件；为空时自动处理所有研究实验日志")
    args = parser.parse_args()

    if args.files:
        targets = [Path(p) for p in args.files]
    else:
        targets = sorted(DOC_DIR.glob(DEFAULT_GLOB))

    rendered = []
    for path in targets:
        if path.suffix.lower() != ".md":
            continue
        rendered.append(render_file(path))

    print("Rendered HTML files:")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
