#!/usr/bin/env python3
"""将 CNAS Word 文档同步到当前 Markdown 源对应口径."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    PROJECT_ROOT / "doc" / "暗场镜片缺陷检测系统_测试大纲.docx",
    PROJECT_ROOT / "doc" / "暗场镜片缺陷检测系统_测试报告_20260322.docx",
]

REPLACEMENTS = {
    "python scripts/run_cnas_eval.py": "uv run python -m cnas_test.runner.run_eval",
    "python -m cnas_test.runner.run_eval": "uv run python -m cnas_test.runner.run_eval",
    "output/cnas_eval/cnas_eval_results.json": "cnas_test/outputs/latest/metrics/cnas_eval_results.json",
    "output/cnas_eval/": "cnas_test/outputs/latest/",
    "output/audit/test_set_v1.json": "cnas_test/manifests/test_set_v1.json",
}

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def _apply_replacements(text: str) -> str:
    result = text
    for old, new in REPLACEMENTS.items():
        result = result.replace(old, new)
    return result


def _sync_document_xml(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    changed = False

    for para in root.findall(".//w:p", NS):
        texts = para.findall(".//w:t", NS)
        if not texts:
            continue

        original = "".join(node.text or "" for node in texts)
        updated = _apply_replacements(original)
        if updated == original:
            continue

        ppr = para.find(f"{{{W_NS}}}pPr")
        first_run = para.find(f"{{{W_NS}}}r")
        first_rpr = None
        if first_run is not None:
            first_rpr = first_run.find(f"{{{W_NS}}}rPr")

        for child in list(para):
            if child.tag != f"{{{W_NS}}}pPr":
                para.remove(child)

        run = ET.Element(f"{{{W_NS}}}r")
        if first_rpr is not None:
            run.append(deepcopy(first_rpr))
        text_node = ET.Element(f"{{{W_NS}}}t")
        if updated.startswith(" ") or updated.endswith(" "):
            text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        text_node.text = updated
        run.append(text_node)
        if ppr is None:
            para.insert(0, run)
        else:
            para.append(run)
        changed = True

    if not changed:
        return xml_bytes
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def sync_docx(path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with ZipFile(path, "r") as zin, ZipFile(tmp_path, "w", compression=ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = _sync_document_xml(data)
            zout.writestr(item, data)
    tmp_path.replace(path)


def main() -> None:
    for path in DOCS:
        sync_docx(path)
        print(f"synced: {path}")


if __name__ == "__main__":
    main()
