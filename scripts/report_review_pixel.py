"""生成像素级标注实验的本地 HTML 报告（output/review/pixel5/pixel_report.html）。

输入：review_pixel_pilot.py analyze 的结果、提示词文档、output/review/pixel5/observations.md（人工撰写的分析）。
"""

from __future__ import annotations

import html
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from review_pixel_pilot import OUT, PROMPT_DOC, QUADS, RATERS, display, load_prompt, load_standard  # noqa: E402

ASSETS = OUT / "report_assets"
S = 0.25
ORDER = ["108l", "26l", "90r", "76r", "72l"]


def esc(x) -> str:
    return html.escape(str(x))


def base(lens: str) -> np.ndarray:
    img, valid = load_standard(lens)
    d = display(img)
    d[~valid] = 0
    return cv2.cvtColor(cv2.resize(d, None, fx=S, fy=S, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def draw_quads(v: np.ndarray, s: float) -> None:
    for x0, y0, x1, y1 in QUADS.values():
        cv2.rectangle(v, (int(x0 * s), int(y0 * s)), (int(x1 * s) - 1, int(y1 * s) - 1), (120, 120, 120), 1)


def rater_view(lens: str, g: dict, b: np.ndarray, s: float = S, crop=None, thick: int = 1) -> np.ndarray:
    v = b.copy()
    rng = random.Random(7)
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    dmg = v.copy()
    for poly in g["damage"]:
        cv2.fillPoly(dmg, [np.rint((np.array(poly) - [ox, oy]) * s).astype(np.int32)], (200, 60, 200))
    v = cv2.addWeighted(dmg, 0.35, v, 0.65, 0)
    for grp in g["groups"]:
        color = tuple(int(c) for c in rng.choice([(60, 220, 60), (255, 200, 40), (40, 160, 255), (255, 90, 90),
                                                   (90, 255, 255), (200, 120, 255), (255, 255, 120)]))
        for i in grp:
            pts = np.rint((np.array(g["lines"][i]) - [ox, oy]) * s).astype(np.int32)
            cv2.polylines(v, [pts], False, color, thick)
    for sp in g["spots"]:
        cv2.circle(v, (int((sp["c"][0] - ox) * s), int((sp["c"][1] - oy) * s)), max(2, int(sp["r"] * s)),
                   (0, 140, 255), 1 if crop is None else 2)
    for i, st in g["unmatched"]:
        p = g["raw"][i][0] if st else g["raw"][i][-1]
        x, y = int((p[0] - ox) * s), int((p[1] - oy) * s)
        cv2.drawMarker(v, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    if crop is None:
        draw_quads(v, s)
    return v


def save(img: np.ndarray, name: str) -> str:
    cv2.imwrite(str(ASSETS / name), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return f"report_assets/{name}"


def md_to_html(text: str) -> str:
    out, in_list = [], False
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        if line.startswith("### "):
            out.append(f"<h4>{esc(line[4:])}</h4>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append("<li>" + re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(line[2:])) + "</li>")
        else:
            out.append("<p>" + re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(line)) + "</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rep = json.loads((OUT / "analysis/report.json").read_text())
    obs_path = OUT / "observations.md"
    obs = md_to_html(obs_path.read_text(encoding="utf-8")) if obs_path.exists() else "<p>（尚未撰写）</p>"
    prompt = load_prompt("q1")
    sections, summary_rows = [], []
    for lens in ORDER:
        row = rep["lenses"][lens]
        geo = json.loads((OUT / f"analysis/{lens}_geometry.json").read_text())
        npz = np.load(OUT / f"analysis/{lens}.npz")
        b = base(lens)
        views = {r: save(rater_view(lens, geo[r], b), f"{lens}_{r}.jpg") for r in RATERS}
        # 共识图
        votes = npz["votes"]
        cv_ = b.copy() // 2
        cen_any = np.zeros(votes.shape, bool)
        for r in RATERS:
            cen_any |= npz[f"{r}_center"]
        lab = np.where(cen_any, votes, 0)
        small = cv2.resize(lab, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv_[small == 1] = (60, 60, 255); cv_[small == 2] = (0, 220, 255); cv_[small >= 3] = (80, 230, 80)
        draw_quads(cv_, S)
        cons = save(cv_, f"{lens}_consensus.jpg")
        tv = b.copy()
        tsm = cv2.resize(npz["trad"].astype(np.uint8), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        tv[tsm] = (60, 200, 30)
        trad = save(tv, f"{lens}_traditional.jpg")
        # 交叉最密集区域放大：三人中心线并集密度最大的 600×600 窗口
        dens = cv2.boxFilter(cen_any.astype(np.float32), -1, (600, 600), normalize=False)
        y, x = np.unravel_index(np.argmax(dens), dens.shape)
        x0, y0 = int(np.clip(x - 300, 0, 4096 - 600)), int(np.clip(y - 300, 0, 3000 - 600))
        img, valid = load_standard(lens)
        full = display(img); full[~valid] = 0
        crop_base = cv2.cvtColor(full[y0:y0 + 600, x0:x0 + 600], cv2.COLOR_GRAY2BGR)
        zooms = [save(crop_base, f"{lens}_zoom_img.jpg")] + [
            save(rater_view(lens, geo[r], crop_base, s=1.0, crop=(x0, y0), thick=2), f"{lens}_zoom_{r}.jpg")
            for r in RATERS]
        rr = row["raters"]
        stat_rows = "".join(
            f"<tr><td>{r}（{RATERS[r]['model']}）</td>"
            f"<td>{sum(1 for v in rr[r]['status'].values() if v == 'ok')}/4</td>"
            f"<td>{rr[r]['fragments']}</td><td>{rr[r]['stitched_scratches']}</td>"
            f"<td>{rr[r]['linked_endpoints']}/{rr[r]['border_endpoints']}"
            f"（{rr[r]['continuity_rate'] if rr[r]['continuity_rate'] is not None else '—'}）</td>"
            f"<td>{rr[r]['scratch_length_mm']}</td><td>{rr[r]['max_scratch_len_mm']}</td><td>{rr[r]['crossings']}</td>"
            f"<td>{rr[r]['spots']}</td><td>{rr[r]['damage_regions']}（{rr[r]['damage_area_mm2']} mm²）</td>"
            f"<td>{rr[r]['zone_length_mm']['C']} / {rr[r]['zone_length_mm']['M']} / {rr[r]['zone_length_mm']['E']}</td>"
            f"<td>{rr[r]['snap_shift_median_px']}</td><td>{rr[r]['trad_support']}</td><td>{rr[r]['trad_coverage']}</td>"
            f"<td>{round(sum(x for x in rr[r]['seconds'].values() if x) / 60, 1)} min</td></tr>" for r in RATERS)
        pair_rows = "".join(
            f"<tr><td>{p}</td><td>{v['center_raw']['f1']}</td><td>{v['center_snap']['f1']}</td>"
            f"<td>{v['center_snap']['p']} / {v['center_snap']['r']}</td><td>{v['scratch_mask_iou']}</td>"
            f"<td>{v['spot_f1']}</td><td>{v['damage_iou']}</td></tr>" for p, v in row["pairs"].items())
        cons_rows = "".join(
            f"<tr><td>{r}</td>" + ("".join(f"<td>{row['consensus'][r][f'in_{n}']:.0%}</td>" for n in (1, 2, 3))
                                    if row["consensus"][r] else "<td colspan=3>—</td>") + "</tr>" for r in RATERS)
        notes = "".join(f"<tr><td>{r}</td><td class='note'>" +
                        "<br>".join(f"{q}: {esc(t)}" for q, t in rr[r]["notes"].items()) + "</td></tr>" for r in RATERS)
        fails = [f"{r}:{q} {s}" for r in RATERS for q, s in rr[r]["status"].items() if s != "ok"]
        sections.append(f"""
<section class="lens" id="{lens}"><h3>{lens}</h3>
{('<p class="warn">未成功的象限：' + esc('；'.join(fails)) + '</p>') if fails else ''}
<h4>三助手像素级标注（每条拼接后划痕一种颜色；橙圈=斑点；紫色=碎裂；红叉=未能跨象限拼接的边界端点；灰框=象限）</h4>
<div class="g3">{''.join(f'<figure><img src="{views[r]}"><figcaption>{r} {RATERS[r]["model"]}</figcaption></figure>' for r in RATERS)}</div>
<div class="g2"><figure><img src="{cons}"><figcaption>三人共识（中心线容差 6 像素）：绿=3 人，黄=2 人，红=仅 1 人</figcaption></figure>
<figure><img src="{trad}"><figcaption>传统算法像素分割（仅作参考）</figcaption></figure></div>
<h4>交叉最密集区域原分辨率放大（600×600，左上角全局坐标 {x0}, {y0}）</h4>
<div class="g4"><figure><img src="{zooms[0]}"><figcaption>标准输入（平方根拉伸）</figcaption></figure>
{''.join(f'<figure><img src="{z}"><figcaption>{r}</figcaption></figure>' for z, r in zip(zooms[1:], RATERS))}</div>
<h4>统计</h4>
<table><tr><th>评审员</th><th>成功象限</th><th>划痕段数</th><th>拼接后划痕数</th><th>跨象限拼接端点（连续率）</th><th>划痕总长 mm</th><th>最长 mm</th><th>交叉点</th><th>斑点</th><th>碎裂区</th><th>C/M/E 区长度 mm</th><th>吸附位移中位 px</th><th>落在传统前景比例</th><th>覆盖传统骨架比例</th><th>用时</th></tr>{stat_rows}</table>
<table><tr><th>两两对比</th><th>中心线 F1（原始）</th><th>中心线 F1（吸附后）</th><th>精度 / 召回（吸附后）</th><th>划痕掩膜 IoU</th><th>斑点 F1</th><th>碎裂 IoU</th></tr>{pair_rows}</table>
<table><tr><th>评审员</th><th>其中心线仅本人标出</th><th>2 人一致</th><th>3 人一致</th></tr>{cons_rows}</table>
<h4>助手说明（原文）</h4><table>{notes}</table>
</section>""")
        summary_rows.append(
            f"<tr><td>{lens}</td>" + "".join(f"<td>{rr[r]['stitched_scratches']}</td>" for r in RATERS)
            + "".join(f"<td>{row['pairs'][p]['center_snap']['f1']}</td>" for p in row["pairs"])
            + "".join(f"<td>{rr[r]['continuity_rate'] if rr[r]['continuity_rate'] is not None else '—'}</td>" for r in RATERS)
            + "</tr>")
    cfg = "".join(f"<tr><td>{r}</td><td>{'Claude Code' if v['tool'] == 'claude' else 'Codex CLI'}</td>"
                  f"<td><code>{v['model']}</code></td><td><code>{v['effort']}</code></td></tr>" for r, v in RATERS.items())
    pairs_hdr = "".join(f"<th>{p} F1</th>" for p in rep["lenses"][ORDER[0]]["pairs"])
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>像素级标注实验</title><style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2430; --mute:#5c6675; --line:#dde2ea; --accent:#2a63b8; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1240px; margin:0 auto; padding:24px 20px 60px; }} h1 {{ font-size:25px; }} h2 {{ font-size:20px; margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:18px; margin:0 0 6px; }} h4 {{ font-size:15px; color:var(--accent); margin:16px 0 8px; }}
.card, section.lens {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }} th, td {{ border:1px solid var(--line); padding:5px 7px; text-align:left; vertical-align:top; }} th {{ background:#eef2f8; }}
td.note {{ color:var(--mute); font-size:12.5px; }} .g2,.g3,.g4 {{ display:grid; gap:10px; }} .g2 {{ grid-template-columns:1fr 1fr; }} .g3 {{ grid-template-columns:repeat(3,1fr); }} .g4 {{ grid-template-columns:repeat(4,1fr); }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; border:1px solid var(--line); background:#000; }} figcaption {{ font-size:12px; color:var(--mute); text-align:center; }}
pre {{ background:#0f1720; color:#e6edf3; padding:14px; border-radius:8px; white-space:pre-wrap; font-size:12.5px; }} .warn {{ color:#b35c00; font-weight:600; }}
@media (max-width:760px) {{ .g2,.g3,.g4 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>像素级缺陷标注实验：三智能助手异同（传统算法仅作参考）</h1>
<p style="color:var(--mute)">输入：output/dataset_v3 标准输入图像　|　5 张镜片 × 4 象限 × 3 名评审员　|　工具：scripts/review_pixel_pilot.py、review_pixel_analysis.py、report_review_pixel.py　|　结果：output/review/pixel5/</p>
<h2>一、分析与结论</h2><div class="card">{obs}</div>
<h2>二、实验方法</h2><div class="card">
<ol><li><b>评审包：</b>ROI 外接框按中线切为 4 个象限（1928×1548 像素，相邻象限重叠 96 像素），保持原分辨率；平方根拉伸显示；上、左两侧加像素坐标标尺。每个评审包只含一张象限图，编号不含镜片名。</li>
<li><b>标注形式：</b>划痕为中心线折线（顶点间距 ≤ 40 像素）+ 平均宽度；斑点为圆；碎裂为多边形。脚本栅格化为像素掩膜。</li>
<li><b>隔离调用：</b>三名评审员在项目目录之外的只读会话中独立标注，提示词完全相同（仅象限名不同）。</li>
<li><b>跨象限连续性检测：</b>端点距象限内侧边界 ≤ 12 像素视为到达边界；在相邻象限中寻找距离 ≤ 25 像素且走向夹角 ≤ 25° 的划痕配对拼接；未配对的边界端点即连续性失败。</li>
<li><b>吸附精修：</b>对三人同样处理，把中心线沿法向在 ±6 像素内移到亮度脊线（类似标注工具的磁性吸附），报告吸附位移以反映原始定位精度。</li>
<li><b>对比指标：</b>中心线一致性（容差 6 像素 ≈ 0.04 mm 的精度/召回/F1）、划痕掩膜 IoU、斑点 F1、碎裂 IoU、拼接后划痕数、总长、交叉点数、分区长度、三人共识；传统算法前景与骨架仅作参考。</li></ol>
<table><tr><th>评审员</th><th>工具</th><th>模型</th><th>推理强度</th></tr>{cfg}</table></div>
<h2>三、提示词（以左上象限为例）</h2><div class="card"><pre>{esc(prompt)}</pre></div>
<h2>四、汇总</h2><div class="card"><table><tr><th>镜片</th><th>R1 划痕数</th><th>R2 划痕数</th><th>R3 划痕数</th>{pairs_hdr}<th>R1 连续率</th><th>R2 连续率</th><th>R3 连续率</th></tr>{''.join(summary_rows)}</table></div>
<h2>五、逐镜片结果</h2>{''.join(sections)}
</main></body></html>"""
    (OUT / "pixel_report.html").write_text(doc, encoding="utf-8")
    print(f"报告: {OUT / 'pixel_report.html'}")


if __name__ == "__main__":
    main()
