"""生成传统基础标注（trad_v1）实验的本地 HTML 报告（output/annotation/trad_v1/tradseg_report.html）。

输入：annot_build.py 的导出、annot_compare.py 的 compare/report.json、可选的 subset.json（最小人工标注子集）、
人工撰写的 output/annotation/trad_v1/observations.md，以及前几轮试点的报告图（semantic5）。
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from darkfield_defects.annotation.trad_segment import TradParams, flatten, line_response  # noqa: E402
from report_review_semantic import md_to_html  # noqa: E402
from review_pixel_pilot import display, load_standard  # noqa: E402

ANN = PROJECT_ROOT / "output/annotation/trad_v1"
ASSETS = ANN / "report_assets"
SEM = PROJECT_ROOT / "output/review/semantic5"
ORDER = ["108l", "26l", "90r", "76r", "72l"]
CLS_BGR = {1: (60, 210, 60), 2: (0, 160, 255), 3: (220, 70, 220), 4: (150, 150, 150)}
CLS_CN = {"scratch": "划痕", "spot": "斑点", "damage": "碎裂", "uncertain": "待确认", "background": "背景",
          "not_defect": "非缺陷"}

LITERATURE = [
    ("降噪去光晕", "高斯平滑 + 白顶帽（半径 12 像素）", "形态学顶帽背景校正（经典方法）",
     "去掉宽于约 24 像素的柔和光晕与阴影，保留划痕和斑点的原始亮度"),
    ("划痕检测", "12 方向 × 2 长度（15/31 像素）的线检测器，线上均值取两半线段的较小者，以噪声 σ 为单位迟滞阈值",
     "Nguyen 等, Pattern Recognition 46(3):703–715, 2013（多尺度线检测）；RORPO（Merveille 等, IEEE TPAMI 2018, 路径开运算）",
     "两半线取小值是本项目的改动：亮斑周围不再出现环状伪划痕，划痕端点不外延"),
    ("弱延续找回", "从线掩膜沿亮于 2σ 的连通像素受限生长 5 像素", "迟滞/测地重建（经典方法）",
     "找回弯曲段、端部延续与贴近亮划痕的弱划痕"),
    ("划痕掩膜宽度", "实例像素取中心线两侧“截面半高宽/2 + 1 像素”以内", "中心线 + 线宽的矢量标注表示（与 LabelMe 折线一致）",
     "去掉 2σ 阈值带入的光学模糊外沿；外沿像素归背景"),
    ("斑点检测", "高斯差分峰值 + 12 方向各向同性检验 + 区域生长 + 长宽比检验", "LoG/DoG 斑点检测（经典方法）",
     "各向同性检验把交叉点、划痕上的亮节点排除在斑点之外"),
    ("碎裂", "64 像素窗口内“亮而无法逐条追踪”的像素占比 ≥ 0.35 且线方向熵 ≥ 0.85", "按本项目标注定义操作化",
     "评审员对碎裂本身不一致，结果保守并全部进入复核"),
    ("交叉点续接", "相邻三叉点间的短段并入交叉区；止于交叉区的分支端在 30 像素内按“走向连续 × 截面剖面相似”配对",
     "Bekkers 等, JMIV 49(3):583–610, 2014（多方向追踪处理交叉）；Hannink 等, MICCAI 2014（保交叉的血管增强）",
     "截面剖面（峰值、半高宽、形状）沿中心线每 3 像素采样"),
    ("断点桥接", "50 像素内自由端点按距离、共线性、间隙脊线证据、剖面相似性双向最优配对",
     "Carneiro-Esteves 等, arXiv:2408.12943, 2024（无监督重连正则）；Cohen & Kimmel 1997（最小路径）",
     "只在间隙上有脊线证据时桥接较远的断点"),
    ("复核分流", "低分连接、待确认短碎片、密集区内的长划痕、碎裂区写入复核队列", "主动学习/人在回路标注（通用做法）",
     "人工按队列逐项确认，其余区域只需抽查"),
]


def esc(x) -> str:
    return html.escape(str(x))


def save(img, name: str) -> str:
    cv2.imwrite(str(ASSETS / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"report_assets/{name}"


def colorize(base: np.ndarray, cls: np.ndarray) -> np.ndarray:
    v = base.copy()
    for c, col in CLS_BGR.items():
        v[cls == c] = col
    return cv2.addWeighted(v, 0.75, base, 0.25, 0)


def inst_view(base: np.ndarray, inst: np.ndarray, lm: dict, links: list, x0: int, y0: int) -> np.ndarray:
    rng = np.random.default_rng(5)
    lut = rng.integers(60, 255, (int(inst.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    v = cv2.addWeighted(lut[inst], 0.8, base, 0.2, 0)
    for s in lm["shapes"]:
        if s["description"] == "bridge":
            (xa, ya), (xb, yb) = s["points"]
            cv2.line(v, (xa - x0, ya - y0), (xb - x0, yb - y0), (255, 255, 255), 1)
    for lk in links:
        if lk.get("review"):
            x, y = lk["pos"][0] - x0, lk["pos"][1] - y0
            if 0 <= x < v.shape[1] and 0 <= y < v.shape[0]:
                cv2.circle(v, (x, y), 6, (0, 0, 255), 1)
    return v


def lens_section(lens: str, cmp: dict) -> str:
    meta = json.loads((SEM / "seg" / f"{lens}_info.json").read_text())
    rx, ry, size = meta["region"]
    img, valid = load_standard(lens)
    d = display(img)
    d[~valid] = 0
    sl = np.s_[ry:ry + size, rx:rx + size]
    base = cv2.cvtColor(d[sl], cv2.COLOR_GRAY2BGR)
    cls = cv2.imread(str(ANN / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)
    inst = cv2.imread(str(ANN / f"{lens}_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    info = json.loads((ANN / f"{lens}.json").read_text())
    lm = json.loads((ANN / f"{lens}_labelme.json").read_text())
    # 去光晕图与线响应：区域外扩 48 像素计算（均为局部运算），再裁回
    m = 48
    ys, xs = slice(max(ry - m, 0), ry + size + m), slice(max(rx - m, 0), rx + size + m)
    p = TradParams()
    _, fl = flatten(img[ys, xs], p)
    R, _ = line_response(fl, p)
    cy, cx = ry - ys.start, rx - xs.start
    fl, R = fl[cy:cy + size, cx:cx + size], R[cy:cy + size, cx:cx + size]
    fl_v = np.clip(16.0 * np.sqrt(np.maximum(fl, 0)), 0, 255).astype(np.uint8)
    r_v = np.clip(255 * np.sqrt(np.maximum(R, 0) / max(np.percentile(R, 99.5), 1e-3)), 0, 255).astype(np.uint8)
    imgs = {
        "raw": save(base, f"{lens}_raw.jpg"),
        "flat": save(cv2.cvtColor(fl_v, cv2.COLOR_GRAY2BGR), f"{lens}_flat.jpg"),
        "line": save(cv2.applyColorMap(r_v, cv2.COLORMAP_INFERNO), f"{lens}_line.jpg"),
        "cls": save(colorize(base, cls[sl]), f"{lens}_cls.jpg"),
        "inst": save(inst_view(base, inst[sl], lm, info["links"], rx, ry), f"{lens}_inst.jpg"),
    }
    for src, dst in ((SEM / "report_assets" / f"{lens}_trad.jpg", f"{lens}_hr.jpg"),
                     (SEM / "report_assets" / f"{lens}_consensus.jpg", f"{lens}_sem_cons.jpg")):
        if src.exists():
            shutil.copy(src, ASSETS / dst)
            imgs[dst] = f"report_assets/{dst}"
    shutil.copy(ANN / f"{lens}_preview.jpg", ASSETS / f"{lens}_preview.jpg")
    f = info["features"]
    if "n_review_items" not in f:
        f["n_review_items"] = sum(1 for _ in open(ANN / f"{lens}_review.csv", encoding="utf-8")) - 1
    s = cmp["semantic5"]["lenses"][lens]
    conf = s["confusion"]
    cons_cls = sorted({k.split("->")[0] for k in conf})
    trad_cls = ["scratch", "spot", "damage", "uncertain", "background"]
    conf_rows = "".join(f"<tr><td>{CLS_CN.get(a, a)}</td>" + "".join(f"<td>{conf.get(f'{a}->{b}', 0)}</td>" for b in trad_cls)
                        + "</tr>" for a in cons_cls)
    hr_fig = (f'<figure><img src="{imgs[f"{lens}_hr.jpg"]}"><figcaption>上一轮：HR 高召回分割的候选片段（彩色编号，灰=短碎片）</figcaption></figure>'
              if f"{lens}_hr.jpg" in imgs else "")
    cons_fig = (f'<figure><img src="{imgs[f"{lens}_sem_cons.jpg"]}"><figcaption>上一轮：三助手多数共识（灰=未判定，红=无多数）</figcaption></figure>'
                if f"{lens}_sem_cons.jpg" in imgs else "")
    dmg = s.get("damage_vs_R2")
    dmg_txt = (f"<p class='meta'>碎裂：R2 圈出 {dmg['R2_px']} 像素，trad_v1 判为碎裂 {dmg['trad_px']} 像素，"
               f"IoU {dmg['iou']}，R2 碎裂区被覆盖 {dmg['R2_covered_by_trad']:.0%}。</p>") if dmg else ""
    return f"""<section class="lens" id="{lens}"><h3>{lens}　区域左上角 ({rx}, {ry})，{size}×{size} 像素（与语义试点同一区域）</h3>
<p class="meta">整镜：划痕实例 {f['n_scratch']} 条（总长 {f['scratch_len_mm']} mm，≥2 mm 的 {f['long_scratch_n']} 条），斑点 {f['n_spot']} 个，
待确认 {f['n_uncertain']} 个，碎裂 {f['damage_mm2']} mm²；交叉穿越 {f['n_crossing']} 处、断点桥接 {f['n_gap']} 处，复核队列 {f['n_review_items']} 项；用时 {f['seconds']} s。</p>
<div class="g3"><figure><img src="{imgs['raw']}"><figcaption>标准输入（平方根拉伸）</figcaption></figure>
<figure><img src="{imgs['flat']}"><figcaption>① 降噪去光晕（平滑 + 白顶帽）</figcaption></figure>
<figure><img src="{imgs['line']}"><figcaption>② 多方向线响应（两半线取小值）</figcaption></figure>
<figure><img src="{imgs['cls']}"><figcaption>③ 初步分类：绿=划痕，橙=斑点，紫=碎裂，灰=待确认</figcaption></figure>
<figure><img src="{imgs['inst']}"><figcaption>④ 重连后的实例（同色=同一缺陷；白线=桥接，红圈=低分连接待复核）</figcaption></figure>
<figure><img src="report_assets/{lens}_preview.jpg"><figcaption>整镜预览（1/4 分辨率；红=桥接段）</figcaption></figure></div>
<h4>与上一轮（HR 分割 + 三助手语义判定）同区域对照</h4>
<div class="g3">{hr_fig}{cons_fig}<figure><img src="{imgs['cls']}"><figcaption>本轮 trad_v1 类别图</figcaption></figure></div>
<table><tr><th>助手多数类别 ＼ trad_v1</th>{''.join(f'<th>{CLS_CN[b]}</th>' for b in trad_cls)}</tr>{conf_rows}</table>
<p class="meta">类别一致率 {s['class_agree']:.0%}（“非缺陷”对应 trad 背景或待确认也计一致）；
相邻片段对中，≥2 名助手判“同一缺陷”的 {s['group_same_pairs']} 对里 trad_v1 归入同一实例 {s['group_same_hit']} 对（严格口径 {s['group_same_hit_strict']}），
判“不同缺陷”的 {s['group_diff_pairs']} 对里分开 {s['group_diff_hit']} 对（严格口径 {s['group_diff_hit_strict']}）；
≥2 人指出的错误连接 {s['wrong_merge_consensus']} 处，trad_v1 已拆开 {s['wrong_merge_split']} 处。
上一轮未判定像素 {', '.join(f'{r} {v:.0%}' for r, v in s['semantic5_unjudged'].items())}；本轮前景全部有类别，其中待确认 {s['trad_uncertain_share']:.0%}。</p>{dmg_txt}
</section>"""


def subset_section() -> str:
    p = ANN / "subset.json"
    if not p.exists():
        return "<div class='card'><p>（全量运行完成后由 <code>annot_build.py select</code> 生成）</p></div>"
    sub = json.loads(p.read_text())
    rows = "".join(f"<tr><td>{i + 1}</td><td>{esc(r['lens'])}</td><td>{esc(r['reason'])}</td><td>{r['n_scratch']}</td>"
                   f"<td>{r['scratch_len_mm']}</td><td>{r['n_spot']}</td><td>{r['damage_mm2']}</td><td>{r['n_review_items']}</td></tr>"
                   for i, r in enumerate(sub["selected"]))
    return f"""<div class="card"><p>{esc(sub['summary'])}</p>
<table><tr><th>#</th><th>镜片</th><th>入选原因</th><th>划痕实例</th><th>划痕总长 mm</th><th>斑点</th><th>碎裂 mm²</th><th>复核项</th></tr>{rows}</table></div>"""


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cmp = json.loads((ANN / "compare" / "report.json").read_text())
    obs_p = ANN / "observations.md"
    obs = md_to_html(obs_p.read_text(encoding="utf-8")) if obs_p.exists() else "<p>（尚未撰写）</p>"
    lit = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>" for a, b, c, d in LITERATURE)
    params = json.loads((ANN / "26l.json").read_text())["params"]
    prm = "".join(f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>" for k, v in params.items())
    px = cmp["pixel5"]["methods"]
    px_name = {"trad_v1": "本轮 trad_v1（全部前景）", "trad_v1_scratch": "本轮 trad_v1（划痕 + 待确认）",
               "hr_segment": "上一轮 HR 高召回分割", "pilot_traditional": "首轮试点传统规则"}
    px_rows = "".join(f"<tr><td>{px_name[k]}</td><td>{v['ridge_recall']:.1%}</td><td>{v['fg_frac']:.1%}</td><td>{v['ai_support']:.1%}</td></tr>"
                      for k, v in px.items())
    so = cmp["semantic5"]["overall"]
    sem_rows = "".join(
        f"<tr><td>{ls}</td><td>{v['class_agree']:.0%}</td><td>{v['group_same_hit']}/{v['group_same_pairs']}</td>"
        f"<td>{v['group_diff_hit']}/{v['group_diff_pairs']}</td><td>{v['wrong_merge_split']}/{v['wrong_merge_consensus']}</td>"
        f"<td>{max(v['semantic5_unjudged'].values()):.0%}</td><td>{v['trad_uncertain_share']:.0%}</td><td>{v['hr_segments']}</td><td>{v['trad_instances']}</td></tr>"
        for ls, v in cmp["semantic5"]["lenses"].items())
    g = cmp["pilot5"]["overall"]
    grid_rows = "".join(f"<tr><td>{CLS_CN[c]}</td><td>{v['majority']}</td><td>{v['old_recall']}</td><td>{v['old_precision']}</td>"
                        f"<td>{v['trad_v1_recall']}</td><td>{v['trad_v1_precision']}</td></tr>" for c, v in g.items())
    p1 = cmp["phase1"]
    p1_rows = "".join(f"<tr><td>{ls}</td><td>{v['trad_v1']['n_gt']}</td>"
                      + "".join(f"<td>{v[m]['box_recall']:.3f} / {v[m]['in_box_precision']:.2f} / {v[m]['fg_ratio']:.1%}</td>"
                                for m in ("trad_v1", "hr_segment", "pilot_traditional")) + "</tr>" for ls, v in p1.items())
    secs = "".join(lens_section(ls, cmp) for ls in ORDER)
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>传统基础标注实验</title><style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2430; --mute:#5c6675; --line:#dde2ea; --accent:#2a63b8; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1240px; margin:0 auto; padding:24px 20px 60px; }} h1 {{ font-size:25px; }} h2 {{ font-size:20px; margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:18px; margin:0 0 6px; }} h4 {{ font-size:15px; color:var(--accent); margin:16px 0 8px; }} .meta {{ color:var(--mute); }}
.card, section.lens {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }} th, td {{ border:1px solid var(--line); padding:5px 7px; text-align:left; vertical-align:top; }} th {{ background:#eef2f8; }}
.g3 {{ display:grid; gap:10px; grid-template-columns:repeat(3,1fr); }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; border:1px solid var(--line); background:#000; }} figcaption {{ font-size:12px; color:var(--mute); text-align:center; }}
.tw {{ overflow-x:auto; }}
@media (max-width:760px) {{ .g3 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>传统算法基础标注：分割、初步分类与划痕重连（五图试点）</h1>
<p class="meta">输入：output/dataset_v3 标准输入（背景≈0，原始计数）　|　代码：src/darkfield_defects/annotation/trad_segment.py、scripts/annot_build.py、annot_compare.py、report_annot_trad.py　|　结果：output/annotation/trad_v1/</p>
<h2>一、结论与分析</h2><div class="card">{obs}</div>
<h2>二、算法流程与文献对应</h2><div class="card">
<p>流程：降噪去光晕 → 划痕/斑点/碎裂分割与初步分类 → 骨架化并在交叉点断开 → 沿中心线采样截面灰度剖面 → 交叉点按“走向连续 × 剖面相似”续接、断点按“距离 + 共线 + 间隙证据 + 剖面相似”桥接 → 实例、类别、复核队列。</p>
<div class="tw"><table><tr><th>环节</th><th>本项目实现</th><th>借鉴来源</th><th>说明</th></tr>{lit}</table></div>
<details><summary>全部参数（trad_v1）</summary><table>{prm}</table></details></div>
<h2>三、导出格式与人工修订流程</h2><div class="card"><ol>
<li><b>每张镜片导出：</b><code>&lt;lens&gt;_inst.png</code>（uint16 实例图）、<code>&lt;lens&gt;_cls.png</code>（0 背景/1 划痕/2 斑点/3 碎裂/4 待确认）、
<code>&lt;lens&gt;.json</code>（实例属性、连接记录与参数）、<code>&lt;lens&gt;_labelme.json</code>（划痕为中心线折线，同一实例共用 group_id，含桥接段；斑点/待确认为多边形；碎裂为区域多边形）、<code>&lt;lens&gt;_review.csv</code>（复核队列）。</li>
<li><b>人工修订顺序：</b>① 碎裂区：确认或删除；② 低分交叉/桥接连接：确认、断开或改接；③ 密集区内的长划痕：检查是否错连；④ 待确认短碎片：归入相邻划痕、改为斑点或删除；⑤ 全图抽查漏检，补画中心线。</li>
<li><b>回写：</b>修订后的 LabelMe 文件按“中心线 + 宽度”栅格化回像素掩膜，作为第二阶段分割模型的训练与评估真值；trad_v1 原始输出保留，用于统计人工改动量。</li></ol></div>
<h2>四、与前几轮实验的对比</h2>
<h4>4.1 像素级试点参考（26l 右上象限，三名助手确认的可见划痕中心线，{cmp['pixel5']['n_ref_points']} 个采样点）</h4>
<div class="card"><table><tr><th>方法</th><th>中心线召回（3 像素内）</th><th>象限前景占比</th><th>骨架落在助手划痕 4 像素内</th></tr>{px_rows}</table>
<p class="meta">“助手支持率”受助手漏标影响（助手只画了部分弱划痕），只作参考。</p></div>
<h4>4.2 语义试点参考（5 张镜片各 736×736 区域；上一轮 HR 片段 + 三助手多数意见）</h4>
<div class="card"><table><tr><th>镜片</th><th>类别一致率</th><th>“同一缺陷”对归入同一实例</th><th>“不同缺陷”对分开</th><th>错误连接已拆开</th><th>上一轮未判定像素（最多一人）</th><th>本轮待确认像素</th><th>上一轮片段数</th><th>本轮实例数</th></tr>{sem_rows}
<tr><td><b>合计</b></td><td><b>{so['class_agree']:.0%}</b></td><td><b>{so['same_hit']}/{so['same_total']}</b></td><td><b>{so['diff_hit']}/{so['diff_total']}</b></td><td><b>{so['wrong_split']}/{so['wrong_total']}</b></td><td colspan="4"></td></tr></table></div>
<h4>4.3 网格试点参考（pilot5：16×12 网格，三助手 ≥2 人标出的格子；当时 R2 为 Sonnet）</h4>
<div class="card"><table><tr><th>类别</th><th>多数格子数</th><th>首轮传统规则 召回</th><th>精确率</th><th>trad_v1 召回</th><th>精确率</th></tr>{grid_rows}</table></div>
<h4>4.4 第一阶段框标注（框命中召回 / 前景落在框内比例 / ROI 前景占比）</h4>
<div class="card"><table><tr><th>镜片</th><th>框数</th><th>trad_v1</th><th>HR 分割</th><th>首轮传统规则</th></tr>{p1_rows}</table>
<p class="meta">第一阶段框只覆盖部分缺陷，三种方法的框命中率都接近 100%，区分度很低；框内比例低主要说明框标注不全，不能直接当作误检率。</p></div>
<h2>五、逐镜片结果</h2>{secs}
<h2>六、最小人工标注子集</h2>{subset_section()}
</main></body></html>"""
    (ANN / "tradseg_report.html").write_text(doc, encoding="utf-8")
    print(f"报告: {ANN / 'tradseg_report.html'}")


if __name__ == "__main__":
    main()
