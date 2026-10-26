"""生成五图试点的本地 HTML 报告（output/review/pilot5/pilot_report.html）。

读取 review_pilot.py 产出的评审包、各评审员结果、传统算法结果与汇总，生成可视化图与报告。
用法:
    uv run python scripts/report_review_pilot.py
"""

from __future__ import annotations

import html
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from review_pilot import (  # noqa: E402
    CH, CLASSES, COLS, CW, IMAGE_DIR, NCOL, NROW, OUT, PROMPT_DOC, QUADS, RATERS, enhance,
)

ASSETS = OUT / "report_assets"
SCALE = 0.25  # 4096×3000 → 1024×750
CLASS_CN = {"scratch": "划痕", "spot": "斑点", "damage": "碎裂"}
CLASS_BGR = {"scratch": (60, 200, 30), "spot": (0, 170, 255), "damage": (255, 80, 200)}
ORDER = ["108l", "26l", "90r", "76r", "72l"]
# 第一阶段框标注经 WearScore 推得的等级（项目实测，见计划 v2.1 第 8.6 节）
OLD_LABEL_GRADE = {"108l": "B（31.9）", "26l": "C（57.1）", "90r": "C（63.2）", "76r": "D（72.2）", "72l": "D（73.4）"}


def cells_to_rects(cells: dict, cls: str) -> list[tuple[int, int, int, int]]:
    rects = []
    for q, (x0, y0) in QUADS.items():
        for c in cells.get(q, {}).get(cls) or []:
            col, row = COLS.find(c[0]), int(c[1:]) - 1
            if 0 <= col < NCOL and 0 <= row < NROW:
                x, y = x0 + col * CW, y0 + row * CH
                rects.append((x, y, x + CW, y + CH))
    return rects


def cell_set(cells: dict, cls: str) -> set[tuple[int, int, int, int]]:
    return set(cells_to_rects(cells, cls))


def base_image(lens: str) -> np.ndarray:
    img = cv2.imread(str(IMAGE_DIR / f"{lens}.png"), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(enhance(img), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)


def paint(base: np.ndarray, layers: list[tuple[set, tuple[int, int, int]]], alpha: float = 0.45) -> np.ndarray:
    over = base.copy()
    for rects, color in layers:
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(over, (int(x1 * SCALE), int(y1 * SCALE)), (int(x2 * SCALE) - 1, int(y2 * SCALE) - 1),
                          color, -1)
    out = cv2.addWeighted(over, alpha, base, 1 - alpha, 0)
    for qx in (2048,):
        cv2.line(out, (int(qx * SCALE), 0), (int(qx * SCALE), out.shape[0]), (200, 200, 200), 1)
    cv2.line(out, (0, int(1500 * SCALE)), (out.shape[1], int(1500 * SCALE)), (200, 200, 200), 1)
    return out


def save(img: np.ndarray, name: str) -> str:
    cv2.imwrite(str(ASSETS / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"report_assets/{name}"


def esc(s) -> str:
    return html.escape(str(s))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    key = json.loads((OUT / "key.json").read_text())
    inv = {v: k for k, v in key.items()}
    trad = json.loads((OUT / "traditional.json").read_text())
    report = json.loads((OUT / "report.json").read_text())
    ratings: dict = {}
    for f in sorted((OUT / "ratings").glob("*.json")):
        r = json.loads(f.read_text())
        if "parsed" in r:
            ratings.setdefault(r["pid"], {})[r["rater"]] = r
    prompt = PROMPT_DOC.read_text(encoding="utf-8").split("\n## 提示词正文\n", 1)[1].strip()

    # ── 汇总指标：传统 vs 多数 的精度与召回 ──
    prf = {}
    for cls in CLASSES:
        tp = tt = mm = 0
        for pid in key:
            sets = [cell_set(ratings[pid][x]["parsed"]["cells"], cls) for x in RATERS]
            cnt: dict = {}
            for s in sets:
                for c in s:
                    cnt[c] = cnt.get(c, 0) + 1
            maj = {c for c, n in cnt.items() if n >= 2}
            t = cell_set(trad[pid]["cells"], cls)
            tp += len(maj & t); tt += len(t); mm += len(maj)
        prf[cls] = (tp / max(tt, 1), tp / max(mm, 1), tt, mm)

    secs = {r: [ratings[p][r].get("seconds") for p in key if r in ratings[p]] for r in RATERS}
    costs = {r: [ratings[p][r].get("cost_usd") for p in key if r in ratings[p]] for r in RATERS}

    lens_sections = []
    for lens in ORDER:
        pid = inv[lens]
        rs = ratings[pid]
        row = report["lenses"][pid]
        base = base_image(lens)
        ov = save(cv2.imread(str(OUT / "packages" / pid / "overview.jpg")), f"{lens}_overview.jpg")
        q4 = cv2.imread(str(OUT / "packages" / pid / "q4.jpg"))
        q4s = save(cv2.resize(q4, (1046, 772), interpolation=cv2.INTER_AREA), f"{lens}_q4.jpg")
        tr = save(cv2.imread(str(OUT / f"traditional_{pid}.jpg")), f"{lens}_traditional.jpg")
        rater_imgs = {}
        for r in RATERS:
            cells = rs[r]["parsed"]["cells"]
            rater_imgs[r] = save(paint(base, [(cell_set(cells, c), CLASS_BGR[c]) for c in CLASSES]),
                                 f"{lens}_{r}.jpg")
        trad_img = save(paint(base, [(cell_set(trad[pid]["cells"], c), CLASS_BGR[c]) for c in CLASSES]),
                        f"{lens}_trad_cells.jpg")
        diff_imgs = {}
        for c in CLASSES:
            sets = [cell_set(rs[r]["parsed"]["cells"], c) for r in RATERS]
            cnt: dict = {}
            for s in sets:
                for x in s:
                    cnt[x] = cnt.get(x, 0) + 1
            maj = {x for x, n in cnt.items() if n >= 2}
            t = cell_set(trad[pid]["cells"], c)
            diff_imgs[c] = save(paint(base, [(maj & t, (90, 220, 90)), (maj - t, (40, 160, 255)),
                                             (t - maj, (60, 60, 255))], alpha=0.55), f"{lens}_diff_{c}.jpg")
        grades = " / ".join(f"{r}: <b>{esc(rs[r]['parsed'].get('grade'))}</b>" for r in RATERS)
        zone_rows = "".join(
            f"<tr><td>{r}（{RATERS[r]['model']}）</td><td>{esc(rs[r]['parsed'].get('grade'))}</td>"
            f"<td>{esc(json.dumps(rs[r]['parsed'].get('zone_severity'), ensure_ascii=False))}</td>"
            f"<td>{esc(rs[r]['parsed'].get('dominant_defect'))}</td><td>{esc(rs[r]['parsed'].get('confidence'))}</td>"
            f"<td>{esc(rs[r].get('seconds'))} s</td><td class='note'>{esc(rs[r]['parsed'].get('notes'))}</td></tr>"
            for r in RATERS)
        cell_rows = "".join(
            f"<tr><td>{CLASS_CN[c]}</td>" + "".join(f"<td>{row['cells'][c]['n_cells'].get(r, '—')}</td>" for r in RATERS)
            + f"<td>{row['cells'][c]['majority']}</td><td>{row['cells'][c]['traditional']}</td>"
            f"<td>{', '.join(f'{x:.2f}' for x in row['cells'][c]['pairwise_jaccard']) or '—'}</td>"
            f"<td>{row['cells'][c]['majority_vs_traditional_jaccard'] if row['cells'][c]['majority_vs_traditional_jaccard'] is not None else '—'}</td></tr>"
            for c in CLASSES)
        long_rows = ""
        for r in RATERS:
            ls = rs[r]["parsed"].get("long_scratches") or []
            items = "；".join(f"{esc(x.get('quadrant'))} {esc(x.get('start'))}→{esc(x.get('end'))}"
                             f"{'（交叉）' if x.get('crosses_others') else ''}" for x in ls)
            long_rows += f"<tr><td>{r}</td><td>{len(ls)}</td><td class='note'>{items}</td></tr>"
        lens_sections.append(f"""
<section class="lens" id="{lens}">
  <h3>{lens}（评审包编号 {pid}）　第一阶段框标注推得等级：{OLD_LABEL_GRADE[lens]}</h3>
  <p class="meta">三助手等级：{grades}　|　多数：<b>{esc(row['grade_majority'])}</b>　|　传统管线：<b>{esc(row['traditional_grade'])}</b>（{row['traditional_score']}）</p>
  <div class="grid2">
    <figure><img src="{ov}"><figcaption>评审包 overview.jpg（助手所见）</figcaption></figure>
    <figure><img src="{q4s}"><figcaption>评审包 q4.jpg（右下象限，带网格；缩小显示）</figcaption></figure>
  </div>
  <h4>三助手的网格分割结果（绿=划痕，橙=斑点，紫=碎裂）与传统算法</h4>
  <div class="grid4">
    <figure><img src="{rater_imgs['R1']}"><figcaption>R1 {RATERS['R1']['model']}</figcaption></figure>
    <figure><img src="{rater_imgs['R2']}"><figcaption>R2 {RATERS['R2']['model']}</figcaption></figure>
    <figure><img src="{rater_imgs['R3']}"><figcaption>R3 {RATERS['R3']['model']}</figcaption></figure>
    <figure><img src="{trad_img}"><figcaption>传统算法（同一网格）</figcaption></figure>
  </div>
  <h4>多数意见（≥2 人）与传统算法逐类对比（绿=两者都有，橙=只有助手多数，红=只有传统算法）</h4>
  <div class="grid3">
    {''.join(f'<figure><img src="{diff_imgs[c]}"><figcaption>{CLASS_CN[c]}</figcaption></figure>' for c in CLASSES)}
  </div>
  <figure class="wide"><img src="{tr}"><figcaption>传统算法像素级结果（绿=划痕，橙=斑点，紫=碎裂）</figcaption></figure>
  <h4>分级与说明</h4>
  <table><tr><th>评审员</th><th>等级</th><th>各区严重度</th><th>主要缺陷</th><th>置信度</th><th>用时</th><th>说明（原文）</th></tr>{zone_rows}</table>
  <h4>网格统计</h4>
  <table><tr><th>类别</th><th>R1 格数</th><th>R2 格数</th><th>R3 格数</th><th>多数格数</th><th>传统格数</th><th>两两 Jaccard（R1-R2, R1-R3, R2-R3）</th><th>多数 vs 传统 Jaccard</th></tr>{cell_rows}</table>
  <h4>长划痕追踪</h4>
  <table><tr><th>评审员</th><th>条数</th><th>内容（象限 起点→终点）</th></tr>{long_rows}</table>
</section>""")

    grade_rows = "".join(
        f"<tr><td>{lens}</td>" + "".join(f"<td>{esc(ratings[inv[lens]][r]['parsed'].get('grade'))}</td>" for r in RATERS)
        + f"<td><b>{esc(report['lenses'][inv[lens]]['grade_majority'])}</b></td>"
        f"<td>{esc(trad[inv[lens]]['grade'])}（{trad[inv[lens]]['score']}）</td></tr>"
        for lens in ORDER)
    s = report["summary"]
    prf_rows = "".join(
        f"<tr><td>{CLASS_CN[c]}</td><td>{s['mean_pairwise_jaccard'][c]}</td><td>{prf[c][0]:.2f}</td><td>{prf[c][1]:.2f}</td>"
        f"<td>{prf[c][2]}</td><td>{prf[c][3]}</td></tr>" for c in CLASSES)
    cfg_rows = "".join(
        f"<tr><td>{r}</td><td>{'Claude Code' if v['tool'] == 'claude' else 'Codex CLI'}</td><td><code>{v['model']}</code></td>"
        f"<td><code>{v['effort']}</code></td><td>{min(secs[r]):.0f}–{max(secs[r]):.0f} s</td>"
        f"<td>{('%.2f 美元' % sum(x for x in costs[r] if x)) if any(costs[r]) else '订阅内，未报告'}</td></tr>"
        for r, v in RATERS.items())

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>五图试点报告</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2430; --mute:#5c6675; --line:#dde2ea; --accent:#2a63b8; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 20px 60px; }}
h1 {{ font-size:26px; margin:8px 0 4px; }} h2 {{ font-size:20px; margin-top:36px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:17px; margin:0 0 6px; }} h4 {{ font-size:15px; margin:18px 0 8px; color:var(--accent); }}
.card, section.lens {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:16px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:6px 0; }}
th, td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }} th {{ background:#eef2f8; }}
td.note {{ color:var(--mute); font-size:13px; }}
.grid2, .grid3, .grid4 {{ display:grid; gap:10px; }} .grid2 {{ grid-template-columns:1fr 1fr; }} .grid3 {{ grid-template-columns:repeat(3,1fr); }} .grid4 {{ grid-template-columns:repeat(4,1fr); }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; border:1px solid var(--line); background:#000; }}
figcaption {{ font-size:12.5px; color:var(--mute); text-align:center; }} figure.wide img {{ max-width:700px; display:block; margin:10px auto 0; }}
pre {{ background:#0f1720; color:#e6edf3; padding:14px; border-radius:8px; white-space:pre-wrap; font-size:13px; }}
.meta {{ color:var(--mute); margin:0 0 10px; }} .ok {{ color:#1b7f3b; font-weight:600; }} .warn {{ color:#b35c00; font-weight:600; }} .bad {{ color:#b3261e; font-weight:600; }}
nav a {{ margin-right:12px; }}
@media (max-width:760px) {{ .grid2,.grid3,.grid4 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>五图试点：三智能助手网格分割与分级 vs 传统算法</h1>
<p class="meta">2026-10-25　|　数据：output/dataset_v2 预处理图像　|　工具：scripts/review_pilot.py、scripts/report_review_pilot.py　|　结果目录：output/review/pilot5/</p>
<nav><a href="#s1">结论</a><a href="#s2">流程</a><a href="#s3">提示词</a><a href="#s4">汇总</a><a href="#s5">逐镜片结果</a><a href="#s6">问题与改进</a></nav>

<h2 id="s1">一、结论</h2>
<div class="card">
<ol>
<li><span class="ok">镜片分级可行：</span>5 张中 3 张三人完全一致，5 张均有多数意见，分歧都在相邻等级（C/D）。三助手与传统管线都把 108l 判为 C，而旧框标注推得 B，印证旧标注漏标。样本少、缺少 A/B 级，需在 T0/T1 中用 κ 统计确认。</li>
<li><span class="ok">划痕网格标注基本可行：</span>密集镜片上 R1 与 R3 的 Jaccard 达 0.97；传统划痕结果对助手多数意见的精度 {prf['scratch'][0]:.2f}、召回 {prf['scratch'][1]:.2f}。</li>
<li><span class="warn">斑点与碎裂尚不能直接替代人工：</span>两两 Jaccard 仅 {s['mean_pairwise_jaccard']['spot']} 与 {s['mean_pairwise_jaccard']['damage']}；R3 在 5 张中从未标出碎裂，属于定义分歧。</li>
<li><span class="ok">助手适合剔除传统误检：</span>传统算法标出的斑点格只有 {prf['spot'][0]:.0%}、碎裂格只有 {prf['damage'][0]:.0%} 被多数助手认可。</li>
<li><span class="bad">流程问题：</span>R2 列举遗漏（自述“格子清单为目视估计，可能有遗漏”）；长划痕被象限切断；助手不能给出像素级与逐条追踪结果。</li>
</ol>
</div>

<h2 id="s2">二、工作流程</h2>
<div class="card">
<ol>
<li><b>选样：</b>配准良好（ECC ≥ 0.9）、背景残差小、非异型样本，来自 5 副不同眼镜，按旧框标注分数从轻到重选 108l、26l、90r、76r、72l；76r、72l 含交叠长划痕、碎裂与点缀斑点。</li>
<li><b>盲评包：</b>每张镜片随机编号；overview.jpg（2000×1465，亮度 ×4，虚线圆标 C/M/E 区）+ q1–q4.jpg（2048×1500 原分辨率，16×12 网格，每格约 0.87×0.85 mm）。编号对照 key.json 不放进评审包。</li>
<li><b>隔离调用：</b>评审包复制到项目目录之外，每名评审员一份、顺序独立打乱；Claude 用 <code>claude -p --allowedTools Read --no-session-persistence</code>，Codex 用 <code>codex exec -s read-only --ephemeral -i …</code>（提示词经标准输入），三人并行，结果写入 ratings/。</li>
<li><b>传统对照：</b>背景中值估计 → 噪声归一化 → 迟滞阈值（3σ/6σ）→ 形态规则（斑点：面积 ≤ 400 且长宽比 &lt; 3）→ 网格内前景占比 &gt; 25% 记为碎裂；按格内 ≥ 20 像素映射到同一网格；等级走 WearScore 链路。</li>
<li><b>汇总：</b>多数意见（≥ 2 人）、两两 Jaccard、传统 vs 多数的精度与召回。</li>
</ol>
<table><tr><th>评审员</th><th>工具</th><th>模型</th><th>推理强度</th><th>每张用时</th><th>用量</th></tr>{cfg_rows}</table>
<p class="meta">注：R2 原计划使用 claude-fable-5-1，但当前账号调用时提示需额外购买用量额度，改用同样可用且与 R1 不同的 claude-sonnet-5-5。</p>
</div>

<h2 id="s3">三、三助手的提示词（完全相同）</h2>
<div class="card"><pre>{esc(prompt)}</pre></div>

<h2 id="s4">四、汇总结果</h2>
<div class="card">
<h4>分级</h4>
<table><tr><th>镜片</th><th>R1</th><th>R2</th><th>R3</th><th>多数</th><th>传统管线</th></tr>{grade_rows}</table>
<h4>网格分割</h4>
<table><tr><th>类别</th><th>三人两两 Jaccard 均值</th><th>传统 vs 多数：精度</th><th>传统 vs 多数：召回</th><th>传统格数</th><th>多数格数</th></tr>{prf_rows}</table>
</div>

<h2 id="s5">五、逐镜片结果</h2>
{''.join(lens_sections)}

<h2 id="s6">六、问题与改进</h2>
<div class="card">
<table><tr><th>问题</th><th>证据</th><th>改进</th></tr>
<tr><td>列举遗漏</td><td>R2 在 90r、72l 只标出其他两人约 1/3 的划痕格，并自述可能遗漏</td><td>输出改为每象限每类 12×16 固定矩阵，逐格作答（doc/评审/评审提示词_v2_矩阵格式.md）</td></tr>
<tr><td>碎裂定义分歧</td><td>R3 从未标碎裂（“密集处多数线条仍可辨，未判为碎裂”）；R1、R2 只在 72l 上一致</td><td>给出操作性定义；T0 中把共识样例裁图附入规程</td></tr>
<tr><td>斑点定义分歧</td><td>斑点两两 Jaccard {s['mean_pairwise_jaccard']['spot']}；划痕上的亮点是否算斑点不统一</td><td>明确长宽比与“明显亮于所在划痕”规则</td></tr>
<tr><td>长划痕被象限切断</td><td>R3 自述“跨象限长痕以单象限可见段标记”</td><td>长划痕改在整镜概览图上以方位描述；逐条完整性由合成真值度量</td></tr>
<tr><td>像素级与逐条追踪</td><td>助手只能做网格级判断</td><td>像素级与物理量由算法给出，助手负责语义核查</td></tr>
<tr><td>三人全审成本</td><td>每张 40–380 s；Claude 两名评审员 10 次约 3.6 美元</td><td>只把分歧格裁成放大图进入第二轮或用户仲裁</td></tr>
<tr><td>评审包来自旧预处理</td><td>本次使用 dataset_v2；部分图像含离焦晕斑</td><td>T0 起改用 dataset_v3 标准输入图像</td></tr>
</table>
</div>
</main></body></html>"""
    out = OUT / "pilot_report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"报告: {out}")


if __name__ == "__main__":
    main()
