"""生成软件著作权登记材料：源程序鉴别材料 + 用户手册（docx，可选转 PDF）。

按代理机构材料清单（doc/软著/软件著作权所需材料/）的要求：

- 源程序：仅收录本项目自研代码（src/darkfield_defects，排除含第三方衍生结构的文件），
  按“入口 → 服务 → 算法 → 评分 → 工具”顺序拼接；去除注释、文档字符串与空行；
  每页固定 50 行，超过 60 页时取前 30 页 + 后 30 页；页眉“软件名称 + 版本号”，页脚连续页码。
- 用户手册：由 doc/软著/用户手册.md 转换（标题/段落/列表/表格/图片），页眉页码同上。
- 申请表：把 doc/软著/申请信息表.md 中的已核实内容填入代理提供的“新系统申请表模板.docx”，
  未确定项保留【待填】并标黄。

用法:
    uv run python scripts/build_softcopyright_materials.py [--pdf]
输出: output/softcopyright/
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import tokenize
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src" / "darkfield_defects"
OUT = PROJECT_ROOT / "output" / "softcopyright"
MANUAL_MD = PROJECT_ROOT / "doc" / "软著" / "用户手册.md"
FORM_TEMPLATE = (PROJECT_ROOT / "doc" / "软著" / "软件著作权所需材料" / "3-材料模板"
                 / "新系统申请表模板.docx")

SOFTWARE_NAME = "离焦微结构镜片缺陷检测及评价系统"
VERSION = "V1.0"
LINES_PER_PAGE = 50
WRAP_COLS = 76  # 模板页边距下 8pt 等宽约 86 列；留余量以容纳字体回退造成的宽度差
HALF_PAGES = 30

# 入口与主流程优先，使前 30 页覆盖核心功能
ORDERED_FILES = [
    "viz/__init__.py",
    "viz/app.py",
    "app_services/__init__.py",
    "app_services/inference_service.py",
    "app_services/run_records.py",
    "preprocessing/__init__.py",
    "preprocessing/pipeline.py",
    "preprocessing/background_fusion.py",
    "preprocessing/roi_builder.py",
    "preprocessing/registration.py",
    "preprocessing/brightness_correction.py",
    "preprocessing/circle_fitting.py",
    "preprocessing/arc_extraction.py",
    "detection/__init__.py",
    "detection/base.py",
    "detection/params.py",
    "detection/preprocess.py",
    "detection/classical.py",
    "detection/features.py",
    "detection/rendering.py",
    "ml/__init__.py",
    "ml/models.py",
    "ml/sfe_modules.py",
    "ml/nwd_loss.py",
    "ml/train.py",
    "ml/predict.py",
    "ml/segmentation_factory.py",
    "measurement.py",
    "scoring/__init__.py",
    "scoring/quantify.py",
    "scoring/wear_score.py",
    "scoring/report.py",
    "eval/__init__.py",
    "eval/metrics.py",
    "eval/image_level.py",
    "data/__init__.py",
    "data/dataset.py",
    "data/loader.py",
    "cli/__init__.py",
    "cli/app.py",
    "logging.py",
    "exceptions.py",
    "__init__.py",
]
# 参照官方 HRNet-OCR 结构实现，原创性边界不清，不纳入鉴别材料
EXCLUDED = {"ml/hrnet_ocr.py"}


def _char_width(ch: str) -> int:
    # 歧义宽度字符（· → ① 等）在宋体中按全角渲染，按 2 列计，避免 Word 自动折行打乱每页行数
    return 2 if unicodedata.east_asian_width(ch) in "WFA" else 1


def _wrap(line: str, cols: int = WRAP_COLS) -> list[str]:
    """按显示宽度硬换行，保证每个物理行不会被 Word 再次折行（否则每页行数失控）。"""
    line = line.replace("\t", "    ").rstrip()
    out, cur, w = [], "", 0
    for ch in line:
        cw = _char_width(ch)
        if w + cw > cols:
            out.append(cur)
            cur, w = "    ", 4
        cur += ch
        w += cw
    out.append(cur)
    return out


def strip_comments_and_docstrings(source: str) -> list[str]:
    """去除 # 注释与文档字符串（模块/类/函数体首个字符串语句及其他独立字符串语句）。"""
    drop_lines: set[int] = set()
    # 行号 → 缩进；文档字符串是唯一语句时补 pass 保持语法完整
    replace_with_pass: dict[int, int] = {}
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for stmt in body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                drop_lines.update(range(stmt.lineno, stmt.end_lineno + 1))
                if len(body) == 1:
                    replace_with_pass[stmt.lineno] = stmt.col_offset
    lines = source.splitlines()
    comment_cols: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            comment_cols[tok.start[0]] = tok.start[1]
    out = []
    for no, line in enumerate(lines, 1):
        if no in replace_with_pass:
            out.append(" " * replace_with_pass[no] + "pass")
            continue
        if no in drop_lines:
            continue
        if no in comment_cols:
            line = line[:comment_cols[no]]
        if line.strip():
            out.append(line.rstrip())
    return out


def collect_source() -> tuple[list[str], dict]:
    all_py = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py")}
    missing = [f for f in ORDERED_FILES if f not in all_py]
    if missing:
        raise SystemExit(f"清单中的文件不存在: {missing}")
    unlisted = sorted(all_py - set(ORDERED_FILES) - EXCLUDED)
    if unlisted:
        raise SystemExit(
            f"存在未纳入排序清单的源文件，请补充到 ORDERED_FILES 或 EXCLUDED: {unlisted}")
    lines: list[str] = []
    n_code = 0
    for rel in ORDERED_FILES:
        for raw in strip_comments_and_docstrings((SRC / rel).read_text(encoding="utf-8")):
            n_code += 1
            lines += _wrap(raw)
    stats = {"files": len(ORDERED_FILES), "code_lines": n_code, "physical_lines": len(lines)}
    return lines, stats


def _set_fonts(run, ascii_font: str, east_font: str, size: float) -> None:
    run.font.name = ascii_font
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east_font)


def _add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def _new_document(title_suffix: str) -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = sec.bottom_margin = Cm(2.54)  # 与代理模板一致
    sec.left_margin = sec.right_margin = Cm(3.17)
    sec.header_distance = sec.footer_distance = Cm(1.0)
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_fonts(hp.add_run(f"{SOFTWARE_NAME} {VERSION}　{title_suffix}"), "Arial", "SimSun", 9)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_fonts(fp.add_run("第 "), "Arial", "SimSun", 9)
    _add_page_field(fp)
    _set_fonts(fp.add_run(" 页"), "Arial", "SimSun", 9)
    return doc


def build_source_doc(lines: list[str], out_path: Path) -> int:
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    if len(pages) > 2 * HALF_PAGES:
        pages = pages[:HALF_PAGES] + pages[-HALF_PAGES:]
    doc = _new_document("源程序")
    for pi, page in enumerate(pages):
        for li, line in enumerate(page):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = pf.space_after = Pt(0)
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(12.6)
            run = p.add_run(line if line else " ")
            _set_fonts(run, "Courier New", "SimSun", 8)
            if li == len(page) - 1 and pi < len(pages) - 1:
                run.add_break(WD_BREAK.PAGE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return len(pages)


# ── Markdown → docx（只覆盖手册用到的语法）────────────────────────────

_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def _add_inline(paragraph, text: str, size: float = 11) -> None:
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`"):
            r = paragraph.add_run(part[1:-1])
            _set_fonts(r, "Courier New", "SimSun", size - 1)
            continue
        else:
            r = paragraph.add_run(part)
        _set_fonts(r, "Arial", "SimSun", size)


def build_manual_doc(
    md_path: Path, out_path: Path, allow_missing_images: bool = False
) -> list[str]:
    if not md_path.is_file():
        raise SystemExit(f"缺少用户手册源文件: {md_path}")
    doc = _new_document("用户手册")
    lines = md_path.read_text(encoding="utf-8").splitlines()
    missing: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line in ("<!-- pagebreak -->", "\\pagebreak"):
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            i += 1
            continue
        if not line or line.startswith("<!--"):
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(line.lstrip("# "))
                r.bold = True
                _set_fonts(r, "Arial", "SimHei", 20)
            else:
                h = doc.add_heading(level=min(level - 1, 3))
                r = h.add_run(line.lstrip("# "))
                _set_fonts(r, "Arial", "SimHei", {2: 16, 3: 14}.get(level, 12))
            i += 1
        elif m := _IMG_RE.fullmatch(line.strip()):
            src = (md_path.parent / m["src"]).resolve()
            if src.is_file():
                doc.add_picture(str(src), width=Cm(14.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif allow_missing_images:
                missing.append(m["src"])
                ph = doc.add_paragraph()
                ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = ph.add_run(f"【截图待补：{m['src']}】")
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                _set_fonts(r, "Arial", "SimSun", 11)
            else:
                raise SystemExit(f"手册引用的图片不存在: {src}（草稿可加 --allow-missing-images）")
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_inline(cap, m["alt"], size=10)
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for r, cells in enumerate(rows):
                for c, text in enumerate(cells):
                    para = table.cell(r, c).paragraphs[0]
                    _add_inline(para, text, size=10)
                    if r == 0:
                        for run in para.runs:
                            run.bold = True
            doc.add_paragraph()
        elif re.match(r"^(\s*)([-*]|\d+\.)\s", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            p = doc.add_paragraph(style="List Number" if ordered else "List Bullet")
            _add_inline(p, re.sub(r"^\s*([-*]|\d+\.)\s+", "", line))
            i += 1
        elif line.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                _set_fonts(p.add_run(lines[i] or " "), "Courier New", "SimSun", 9)
                i += 1
            i += 1
        elif line.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            _add_inline(p, line.lstrip("> "), size=10)
            i += 1
        else:
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Pt(22)
            p.paragraph_format.line_spacing = 1.5
            _add_inline(p, line)
            i += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return missing


# ── 申请表：填入代理模板 ────────────────────────────────────────────────

FORM_YAML = PROJECT_ROOT / "doc" / "软著" / "申请表字段.yaml"
# (模板行号, 左列标签前缀, yaml 键)；行号对应模板中唯一表格的行
FORM_ROWS = [
    (2, "软件全称", "软件全称"), (3, "软件简称", "软件简称"), (4, "版本号", "版本号"),
    (5, "软件分类", "软件分类"), (6, "软件开发完成日期", "软件开发完成日期"),
    (7, "是否发表", "是否发表"), (8, "公司成立日期", "公司成立日期"),
    (9, "著作权人信息", "著作权人信息"), (10, "营业执照号", "营业执照号"),
    (11, "开发的硬件环境", "开发的硬件环境"), (12, "运行的硬件环境", "运行的硬件环境"),
    (13, "开发该软件的操作系统", "开发该软件的操作系统"),
    (14, "软件开发环境", "软件开发环境/开发工具"),
    (15, "该软件的运行平台", "该软件的运行平台/操作系统"),
    (16, "软件运行支撑环境", "软件运行支撑环境/支持软件"),
    (17, "编程语言及版本号", "编程语言及版本号"), (18, "程序量", "程序量"),
    (19, "开发目的", "开发目的"), (20, "面向领域", "面向领域/行业"),
    (21, "软件的主要功能", "软件的主要功能"),
    (23, "软件的技术特点", "技术特点标签"), (24, "软件的技术特点", "技术特点"),
    (26, "申请人", "申请人姓名/公司名称"), (27, "详细地址", "详细地址"),
    (28, "公司成立日期", "申请人公司成立日期"), (29, "邮政编码", "邮政编码"),
    (30, "联系人", "联系人"), (31, "电话号码", "电话号码"), (32, "E-mail", "E-mail"),
    (33, "手机号码", "手机号码"), (34, "传真号码", "传真号码"),
]
TECH_FEATURE_LIMIT = 100


def build_form(code_lines: int, out_path: Path) -> list[str]:
    import yaml

    if not FORM_TEMPLATE.is_file():
        raise SystemExit(f"缺少代理申请表模板: {FORM_TEMPLATE}")
    fields = yaml.safe_load(FORM_YAML.read_text(encoding="utf-8"))
    # 联系人、电话、邮箱等不入库的字段写在同目录的 .local.yaml 中（已加入 .gitignore）
    local = FORM_YAML.with_suffix(".local.yaml")
    if local.is_file():
        fields.update(yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    fields["程序量"] = f"{code_lines} 行"
    tech = str(fields["技术特点"]).strip()
    if len(tech) > TECH_FEATURE_LIMIT:
        raise SystemExit(f"技术特点 {len(tech)} 字，超过表单上限 {TECH_FEATURE_LIMIT} 字")

    doc = Document(str(FORM_TEMPLATE))
    table = doc.tables[0]
    pending = []
    for row_idx, label, key in FORM_ROWS:
        left, right = table.rows[row_idx].cells[0], table.rows[row_idx].cells[-1]
        if not left.text.strip().startswith(label):
            actual = left.text.strip()[:20]
            raise SystemExit(f"模板第 {row_idx} 行应为“{label}”，实际“{actual}”，模板可能已变更")
        value = str(fields.get(key, "") or "").strip()
        for para in right.paragraphs[1:]:
            para._element.getparent().remove(para._element)
        para = right.paragraphs[0]
        for run in list(para.runs):
            run._element.getparent().remove(run._element)
        run = para.add_run(value)
        _set_fonts(run, "Arial", "SimSun", 10.5)
        if "【待填" in value or "【待核实" in value:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            pending.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return pending


# ── 源程序行数统计（佐证申请表“程序量”）────────────────────────────────

MIN_CODE_LINES = 3000  # 代理要求：源代码不少于 3000 行，否则需提交全部源代码


def build_line_stats(out_path: Path) -> dict:
    """逐文件统计原始行数与有效代码行数（去除注释、文档字符串、空行），生成统计表 docx。"""
    rows = []
    for rel in ORDERED_FILES:
        src = (SRC / rel).read_text(encoding="utf-8")
        rows.append((rel, len(src.splitlines()), len(strip_comments_and_docstrings(src))))
    total_raw = sum(r[1] for r in rows)
    total_eff = sum(r[2] for r in rows)

    doc = _new_document("源程序行数统计")
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(f"{SOFTWARE_NAME} {VERSION} 源程序行数统计")
    r.bold = True
    _set_fonts(r, "Arial", "SimHei", 16)
    summary = (
        f"本软件源程序共 {len(rows)} 个文件，原始总行数 {total_raw} 行，"
        f"有效代码行数 {total_eff} 行（统计口径：去除注释、文档字符串和空行），"
        f"{'满足' if total_eff >= MIN_CODE_LINES else '未达到'}不少于 {MIN_CODE_LINES} 行的要求。"
        f"源程序鉴别材料按有效代码连续排版、每页 {LINES_PER_PAGE} 行，"
        f"提交前 {HALF_PAGES} 页与后 {HALF_PAGES} 页。"
    )
    for text in (
        summary,
        "统计范围：src/darkfield_defects 目录下全部自研 Python 源文件；不含第三方库代码，"
        "不含参照官方 HRNet-OCR 结构实现的 ml/hrnet_ocr.py。",
    ):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(21)
        p.paragraph_format.line_spacing = 1.5
        _set_fonts(p.add_run(text), "Arial", "SimSun", 10.5)

    table = doc.add_table(rows=len(rows) + 2, cols=4)
    table.style = "Table Grid"
    table.autofit = False
    widths = [Cm(1.4), Cm(7.6), Cm(2.8), Cm(2.8)]  # 合计 14.6cm，等于模板版心宽度
    for col, w in zip(table.columns, widths):  # gridCol 宽度：LibreOffice 只认这一项
        col.width = w
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w
    data = [("序号", "源文件", "原始行数", "有效代码行数")]
    data += [(str(i), rel, str(raw), str(eff)) for i, (rel, raw, eff) in enumerate(rows, 1)]
    data += [("", "合计", str(total_raw), str(total_eff))]
    for ri, cells in enumerate(data):
        for ci, text in enumerate(cells):
            para = table.cell(ri, ci).paragraphs[0]
            if ci != 1:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(text)
            run.bold = ri == 0 or ri == len(data) - 1
            _set_fonts(run, "Courier New" if ci == 1 and 0 < ri < len(data) - 1 else "Arial",
                       "SimSun", 9.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return {"files": len(rows), "raw_lines": total_raw, "code_lines": total_eff}


def to_pdf(docx_path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                    str(docx_path.parent), str(docx_path)], check=True, capture_output=True)
    pdf = docx_path.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError(f"PDF 转换失败: {docx_path}")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pdf", action="store_true", help="用 LibreOffice 同时导出 PDF")
    ap.add_argument("--skip-manual", action="store_true")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="草稿模式：手册缺少截图时以标黄占位代替")
    args = ap.parse_args()

    lines, stats = collect_source()
    src_doc = OUT / f"{SOFTWARE_NAME}{VERSION}_源程序.docx"
    n_pages = build_source_doc(lines, src_doc)
    print(f"源程序: {stats['files']} 个文件, 有效代码行 {stats['code_lines']}, "
          f"物理行 {stats['physical_lines']} → 输出 {n_pages} 页: {src_doc}")
    outputs = [src_doc]
    stats_doc = OUT / f"{SOFTWARE_NAME}{VERSION}_源程序行数统计.docx"
    line_stats = build_line_stats(stats_doc)
    if line_stats["code_lines"] != stats["code_lines"]:
        raise SystemExit("行数统计与源程序材料不一致，请检查统计逻辑")
    flag = "满足" if line_stats["code_lines"] >= MIN_CODE_LINES else "不足"
    print(f"行数统计: 原始 {line_stats['raw_lines']} 行，有效代码 {line_stats['code_lines']} 行"
          f"（≥{MIN_CODE_LINES} 行要求：{flag}）: {stats_doc}")
    outputs.append(stats_doc)
    form_doc = OUT / f"{SOFTWARE_NAME}{VERSION}_申请表.docx"
    pending = build_form(stats["code_lines"], form_doc)
    print(f"申请表: {form_doc}\n  待填项 {len(pending)} 个: {'、'.join(pending)}")
    if not args.skip_manual:
        man_doc = OUT / f"{SOFTWARE_NAME}{VERSION}_用户手册.docx"
        missing = build_manual_doc(MANUAL_MD, man_doc, args.allow_missing_images)
        print(f"用户手册: {man_doc}" + (f"\n  截图待补 {len(missing)} 张" if missing else ""))
        outputs.append(man_doc)
    if args.pdf:
        for d in outputs:
            print(f"PDF: {to_pdf(d)}")


if __name__ == "__main__":
    main()
