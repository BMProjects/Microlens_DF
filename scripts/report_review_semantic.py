"""生成语义判定实验的本地 HTML 报告（output/review/semantic5/semantic_report.html）。

输入：review_semantic_pilot.py analyze 的结果、提示词、output/review/semantic5/observations.md（人工撰写的分析）。
"""

from __future__ import annotations

import html
import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from review_pixel_pilot import display, load_standard  # noqa: E402
from review_semantic_pilot import OUT, PARAMS, RATERS, id_color, load_prompt  # noqa: E402

ASSETS = OUT / "report_assets"
ORDER = ["108l", "26l", "90r", "76r", "72l"]
CLS_BGR = {1: (60, 210, 60), 2: (0, 160, 255), 3: (220, 70, 220), 4: (140, 90, 40), 5: (110, 110, 110)}
CLS_CN = {"scratch": "划痕", "spot": "斑点", "damage": "碎裂", "not_defect": "非缺陷", "unjudged": "未判定"}


def esc(x) -> str:
    return html.escape(str(x))


def save(img, name):
    cv2.imwrite(str(ASSETS / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"report_assets/{name}"


def md_to_html(text: str) -> str:
    out, in_list = [], False
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        fmt = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(s))  # noqa: E731
        if line.startswith("### "):
            out.append(f"<h4>{esc(line[4:])}</h4>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{fmt(line[2:])}</li>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            out.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in cells) + "</tr>")
        else:
            out.append(f"<p>{fmt(line)}</p>")
    if in_list:
        out.append("</ul>")
    txt = "\n".join(out)
    return re.sub(r"((?:<tr>.*?</tr>\n?)+)", r"<table>\1</table>", txt, flags=re.S)


def overlay(base: np.ndarray, cls: np.ndarray) -> np.ndarray:
    v = base.copy()
    for c, col in CLS_BGR.items():
        v[cls == c] = col
    return cv2.addWeighted(v, 0.75, base, 0.25, 0)


def inst_view(base: np.ndarray, inst: np.ndarray, cls: np.ndarray) -> np.ndarray:
    v = base.copy()
    rng = random.Random(5)
    ids = np.unique(inst)
    pal = {int(i): tuple(rng.randint(60, 255) for _ in range(3)) for i in ids if i > 0}
    for i, col in pal.items():
        v[inst == i] = col
    v[(cls == 5)] = (90, 90, 90)
    v[(cls == 3) & (inst == 0)] = CLS_BGR[3]
    return cv2.addWeighted(v, 0.8, base, 0.2, 0)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rep = json.loads((OUT / "analysis/report.json").read_text())
    key = json.loads((OUT / "key.json").read_text())
    obs_p = OUT / "observations.md"
    obs = md_to_html(obs_p.read_text(encoding="utf-8")) if obs_p.exists() else "<p>（尚未撰写）</p>"
    secs = []
    for lens in ORDER:
        row = rep["lenses"][lens]
        meta = json.loads((OUT / "seg" / f"{lens}_info.json").read_text())
        rx, ry, size = meta["region"]
        z = np.load(OUT / f"analysis/{lens}.npz")
        img, valid = load_standard(lens)
        d = display(img); d[~valid] = 0
        base = cv2.cvtColor(d[ry:ry + size, rx:rx + size], cv2.COLOR_GRAY2BGR)
        seg_r = z["seg_r"]
        labeled = np.zeros(int(seg_r.max()) + 1, bool)
        lab_ids = np.array(meta["labeled"]); lab_ids = lab_ids[lab_ids <= seg_r.max()]
        labeled[lab_ids] = True
        tv = base.copy()
        tv[(seg_r > 0) & ~labeled[seg_r]] = (110, 110, 110)
        for i in np.unique(seg_r):
            if i > 0 and labeled[i]:
                tv[seg_r == i] = id_color(int(i))
        tv = cv2.addWeighted(tv, 0.75, base, 0.25, 0)
        imgs = {"raw": save(base, f"{lens}_raw.jpg"), "trad": save(tv, f"{lens}_trad.jpg")}
        for r in RATERS:
            imgs[f"{r}_cls"] = save(overlay(base, z[f"{r}_cls"]), f"{lens}_{r}_cls.jpg")
            imgs[f"{r}_inst"] = save(inst_view(base, z[f"{r}_inst"], z[f"{r}_cls"]), f"{lens}_{r}_inst.jpg")
        stack = np.stack([z[f"{r}_cls"] for r in RATERS])
        cons = np.zeros(seg_r.shape, np.uint8)
        dis = np.zeros(seg_r.shape, bool)
        for c in range(1, 6):
            cons[(stack == c).sum(0) >= 2] = c
        dis = (seg_r > 0) & (cons == 0)
        cv_ = overlay(base, cons)
        cv_[dis] = (40, 40, 255)
        imgs["cons"] = save(cv_, f"{lens}_consensus.jpg")
        pid0 = next(p for p, v in key.items() if v["lens"] == lens and v["window"] == 0)
        shutil.copy(OUT / "packages" / pid0 / "segments.jpg", ASSETS / f"{lens}_window0_segments.jpg")
        shutil.copy(OUT / "packages" / pid0 / "raw.jpg", ASSETS / f"{lens}_window0_raw.jpg")
        rr = row["raters"]
        stat = "".join(
            f"<tr><td>{r}（{RATERS[r]['model']}）</td><td>{sum(1 for s in rr[r]['status'].values() if s == 'ok')}/4</td>"
            f"<td>{rr[r]['labeled']}/{row['n_shown']}</td>"
            + "".join(f"<td>{rr[r]['class_counts'].get(c, 0)}</td>" for c in ("scratch", "spot", "damage", "not_defect"))
            + f"<td>{rr[r]['merged_objects']}</td><td>{rr[r]['wrong_merge']}</td><td>{rr[r]['damage_polygons']}（{rr[r]['damage_area_mm2']} mm²）</td>"
            f"<td>{rr[r]['missed']}</td><td>{rr[r]['window_conflicts']}/{rr[r]['multi_window_ids']}</td>"
            f"<td>{rr[r]['pixel_share']['unjudged']:.0%}</td><td>{rr[r]['agree_with_rule']}</td>"
            f"<td>{round(rr[r]['seconds'] / 60, 1)} min</td><td>{rr[r]['cost'] or '订阅内'}</td></tr>" for r in RATERS)
        pairs = "".join(
            f"<tr><td>{p}</td><td>{v['class_agree']}</td><td>{v['class_kappa']}</td><td>{v['group_kappa']}</td>"
            f"<td>{v['group_same_jaccard']}</td><td>{v['wrong_merge_jaccard']}</td><td>{v['pixel_class_agree']}</td>"
            + "".join(f"<td>{v['pixel_iou'][c]}</td>" for c in ("scratch", "spot", "damage")) + "</tr>"
            for p, v in row["pairs"].items())
        notes = "".join(f"<tr><td>{r}</td><td class='note'>" + "<br>".join(f"窗口{w}: {esc(t)}" for w, t in rr[r]["notes"].items())
                        + "</td></tr>" for r in RATERS)
        c = row["consensus"]
        secs.append(f"""<section class="lens" id="{lens}"><h3>{lens}　区域左上角 ({rx}, {ry})，{size}×{size} 像素（2×2 窗口）</h3>
<p class="meta">待判定片段 {row['n_shown']} 个；相邻片段对 {row['n_adjacent_pairs']} 个；三人一致 {c['unanimous']} 个，有多数意见 {c['ids_with_majority']} 个，三人各异 {c['disputed']} 个。</p>
<div class="g3"><figure><img src="{imgs['raw']}"><figcaption>区域原图（标准输入，平方根拉伸）</figcaption></figure>
<figure><img src="{imgs['trad']}"><figcaption>传统高召回像素分割：彩色=待判定片段，灰色=短碎片</figcaption></figure>
<figure><img src="{imgs['cons']}"><figcaption>三人多数共识（绿=划痕，橙=斑点，紫=碎裂，棕=非缺陷，灰=未判定，红=无多数）</figcaption></figure></div>
<h4>三助手像素级类别图（由片段语义还原到像素）</h4>
<div class="g3">{''.join(f'<figure><img src="{imgs[r + "_cls"]}"><figcaption>{r} {RATERS[r]["model"]}</figcaption></figure>' for r in RATERS)}</div>
<h4>三助手实例图（同色=被判为同一缺陷）</h4>
<div class="g3">{''.join(f'<figure><img src="{imgs[r + "_inst"]}"><figcaption>{r}</figcaption></figure>' for r in RATERS)}</div>
<h4>助手看到的评审包示例（窗口 0）</h4>
<div class="g2"><figure><img src="report_assets/{lens}_window0_raw.jpg"><figcaption>raw.jpg</figcaption></figure>
<figure><img src="report_assets/{lens}_window0_segments.jpg"><figcaption>segments.jpg</figcaption></figure></div>
<h4>统计</h4>
<table><tr><th>评审员</th><th>成功窗口</th><th>已判定/待判定</th><th>划痕</th><th>斑点</th><th>碎裂</th><th>非缺陷</th><th>合并组</th><th>错误连接</th><th>碎裂多边形</th><th>漏分割</th><th>跨窗口冲突/重复出现</th><th>未判定像素</th><th>与传统规则一致</th><th>用时</th><th>成本 $</th></tr>{stat}</table>
<table><tr><th>两两对比</th><th>类别一致率</th><th>类别 κ</th><th>归组 κ</th><th>“同一缺陷”对 Jaccard</th><th>错误连接 Jaccard</th><th>像素类别一致率</th><th>划痕像素 IoU</th><th>斑点像素 IoU</th><th>碎裂像素 IoU</th></tr>{pairs}</table>
<h4>助手说明（原文）</h4><table>{notes}</table></section>""")
    overall = "".join(f"<tr><td>{p}</td><td>{v['n']}</td><td>{v['class_agree']}</td><td>{v['class_kappa']}</td></tr>"
                      for p, v in rep["overall"].items())
    cfg = "".join(f"<tr><td>{r}</td><td>{'Claude Code' if v['tool'] == 'claude' else 'Codex CLI'}</td><td><code>{v['model']}</code></td><td><code>{v['effort']}</code></td></tr>"
                  for r, v in RATERS.items())
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>语义判定实验</title><style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2430; --mute:#5c6675; --line:#dde2ea; --accent:#2a63b8; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1240px; margin:0 auto; padding:24px 20px 60px; }} h1 {{ font-size:25px; }} h2 {{ font-size:20px; margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:18px; margin:0 0 6px; }} h4 {{ font-size:15px; color:var(--accent); margin:16px 0 8px; }} .meta {{ color:var(--mute); }}
.card, section.lens {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }} th, td {{ border:1px solid var(--line); padding:5px 7px; text-align:left; vertical-align:top; }} th {{ background:#eef2f8; }}
td.note {{ color:var(--mute); font-size:12.5px; }} .g2,.g3 {{ display:grid; gap:10px; }} .g2 {{ grid-template-columns:1fr 1fr; }} .g3 {{ grid-template-columns:repeat(3,1fr); }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; border:1px solid var(--line); background:#000; }} figcaption {{ font-size:12px; color:var(--mute); text-align:center; }}
pre {{ background:#0f1720; color:#e6edf3; padding:14px; border-radius:8px; white-space:pre-wrap; font-size:12.5px; }}
@media (max-width:760px) {{ .g2,.g3 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>传统高召回像素分割 + 三智能助手语义判定实验</h1>
<p class="meta">输入：output/dataset_v3 标准输入　|　5 张镜片 × 4 个窗口（384×384，放大 2.5 倍）× 3 名评审员　|　工具：scripts/hr_segment.py、review_semantic_pilot.py、review_semantic_analysis.py、report_review_semantic.py　|　结果：output/review/semantic5/</p>
<h2>一、分析与结论</h2><div class="card">{obs}</div>
<h2>二、方法</h2><div class="card"><ol>
<li><b>传统高召回像素分割（{esc(PARAMS)}）：</b>在标准输入上以本图噪声 σ 为单位做迟滞阈值（2σ/4σ），并与 Sato 脊线响应的迟滞结果取并集；在单象限测试中，对助手确认的划痕召回由原参数的 60.5% 提高到 88.1%。</li>
<li><b>候选片段：</b>前景骨架化并剪除毛刺；骨架在交叉点断开，按走向把最接近共线的分支两两配对连成笔画；前景像素按最近骨架归属；紧凑小连通域为斑点候选。骨架长度 ≥ 40 像素的笔画与面积 ≥ 12 像素的斑点候选带全镜唯一编号交给助手判定，更短的碎片显示为灰色。</li>
<li><b>评审包：</b>每张镜片取前景最密的 736×736 区域，分为 2×2 个 384×384 窗口（重叠 32 像素），每个窗口含整镜位置图、放大 2.5 倍的原图（带窗口坐标标尺）和片段编号叠加图。</li>
<li><b>助手任务：</b>判定每个编号片段的类别；把属于同一条划痕/同一个斑点/同一片碎裂的片段归组；指出错误连接；圈出碎裂区；标出漏分割。相同编号在相邻窗口中一致，跨窗口连续性由编号自动衔接。</li>
<li><b>像素还原：</b>编号片段取其判定类别与实例；灰色碎片落在碎裂多边形内归碎裂，否则在 3 像素内就近继承相邻片段，仍无归属的计为“未判定”。</li></ol>
<table><tr><th>评审员</th><th>工具</th><th>模型</th><th>推理强度</th></tr>{cfg}</table></div>
<h2>三、提示词</h2><div class="card"><pre>{esc(load_prompt())}</pre></div>
<h2>四、总体类别一致性（全部镜片合并）</h2><div class="card"><table><tr><th>两两对比</th><th>共同判定片段数</th><th>类别一致率</th><th>类别 κ</th></tr>{overall}</table></div>
<h2>五、逐镜片结果</h2>{''.join(secs)}
</main></body></html>"""
    (OUT / "semantic_report.html").write_text(doc, encoding="utf-8")
    print(f"报告: {OUT / 'semantic_report.html'}")


if __name__ == "__main__":
    main()
