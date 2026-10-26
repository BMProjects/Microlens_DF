"""传统基础标注算法的迭代复现：按改进环节逐步累积开关，在 6 张不同磨损程度的镜片上各跑一遍，
供开发者查看每一步改了什么、效果如何，并确认最终参数后再全量运行。

阶段 S0 为初版算法（近似复现），S8 为最终方案（= TradParams 默认值）。每个阶段在前一阶段基础上只改一个环节。

用法:
    uv run python scripts/annot_ablation.py run --workers 4
    uv run python scripts/annot_ablation.py analyze
    uv run python scripts/annot_ablation.py report
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import annot_compare as AC  # noqa: E402
from annot_build import STD_DIR, run_lens  # noqa: E402
from darkfield_defects.annotation.trad_segment import TradParams  # noqa: E402
from review_pixel_pilot import display, load_standard  # noqa: E402

OUT = PROJECT_ROOT / "output/annotation/ablation"
LENSES = ["108l", "26l", "90r", "76r", "72l", "55l"]
LENS_NOTE = {"108l": "轻度", "26l": "中度", "90r": "中重度", "76r": "重度", "72l": "重度、密集碎短线",
             "55l": "雾化最重（新增，检验去光晕）"}
SEM_LENSES = ["108l", "26l", "90r", "76r", "72l"]

# 每个阶段：(编号, 名称, 本阶段改动, 改动原因, 相对前一阶段的参数)
STAGES = [
    ("S0", "初版", "经典多方向线检测（整条线均值）、阈值 3σ/6σ、全图 MAD 估噪（下限 σ/4）、DoG 斑点（无各向同性检验）、"
     "逐交叉区配对、按前景填充率/方向熵/碎片长度判碎裂、12 像素以上的断点须有间隙证据",
     "第一个可运行版本",
     dict(line_two_sided=False, line_low=3.0, line_high=6.0, line_noise_mode="mad_floor",
          spot_isotropy=False, spot_grow_frac=0.4, spot_max_area=900, spot_max_elong=3.0, spot_guard_px=1,
          junction_mode="cluster", edge_guard_px=0, grow_radius=0, width_margin=math.inf, width_k=2.0,
          damage_mode="legacy", damage_fill=0.30, damage_entropy=0.85, damage_keep_len=150,
          gap_need_evidence=12.0, gap_min_score=0.25,
          faint_lengths=(), grey_confirm=False, crossing_overlap=False)),
    ("S1", "斑点判别", "斑点峰值增加 12 方向各向同性检验与“不与其他亮结构相连”检验；长宽比上限 3→4；斑点周围 7 像素抑制线响应",
     "初版把划痕交叉点、划痕上的亮节点当成斑点（72l 斑点像素 79 万），并在斑点保护区挖掉大量划痕",
     dict(spot_isotropy=True, spot_grow_frac=0.3, spot_max_area=1500, spot_max_elong=4.0, spot_guard_px=7)),
    ("S2", "两半线检测", "线上均值改为“两个半线段均值取小”，阈值相应降为 1.5σ/3σ",
     "经典线检测在亮斑周围半径约 L/2 处产生环状伪划痕，划痕端点也向外延伸 L/2",
     dict(line_two_sided=True, line_low=1.5, line_high=3.0)),
    ("S3", "交叉续接", "相邻三叉点之间的短段并入交叉区；按“止于交叉区的分支端”在 30 像素内配对，骨架断开处的自由端也参与；端部走向跳过紧邻端点的 3 像素",
     "小角度交叉在骨架上裂成两个三叉点夹一小段，逐交叉区配对接不上，长划痕被切成多段",
     dict(junction_mode="ends")),
    ("S4", "噪声估计", "线响应噪声改在背景像素上估计，并限制在图像噪声的 0.25–0.6 倍",
     "密集镜片的全图 MAD 被细密划痕抬高（72l 达 1.7、76r 达 2.9），阈值随之抬高 5 倍以上，弱划痕大量漏检",
     dict(line_noise_mode="bg_bounded")),
    ("S5", "边界带", "有效区边界 8 像素内不检测划痕", "有效区边缘残留的亮边被当成沿边界的长划痕（108l 左缘）",
     dict(edge_guard_px=8)),
    ("S6", "受限生长", "从线掩膜沿亮于 2σ 的连通像素生长至多 5 像素",
     "两半线检测在弯曲段、端部和贴近亮划痕处偏保守，26l 助手确认中心线召回只有 80%",
     dict(grow_radius=5)),
    ("S7", "半高宽掩膜", "划痕实例像素收窄到中心线两侧“截面半高宽/2 + 1 像素”",
     "2σ 阈值把光学模糊外沿也划入，掩膜宽于可见线宽 2–3 倍且边缘毛糙",
     dict(width_margin=1.0, width_k=3.0)),
    ("S8", "碎裂定义与断点（第一轮最终 = 当前方案）", "碎裂改按标注定义：“亮而无法逐条追踪”的像素占比 ≥0.35 且方向熵 ≥0.85；30 像素内的共线断点不再要求间隙证据（低分进复核）",
     "初版碎裂判据把亮划痕周围和边缘带误判为碎裂；24 像素的共线断点因间隙无脊线证据而接不上",
     dict(damage_mode="untraced", damage_fill=0.35, damage_entropy=0.85, damage_keep_len=40,
          gap_need_evidence=30.0, gap_min_score=0.2)),
    ("S9", "长尺度弱划痕层", "在已检出线条置零后的残差图上，用 61 像素两半线核再检测一遍（阈值 2σ/4σ），并入线掩膜；"
     "淡线与亮线分别骨架化，淡线骨架与亮线相隔至少 2 像素，不在亮线上新增分叉点",
     "灰色待确认区域大多是浅而密的划痕：31 像素核信噪比不足，只检出碎段",
     dict(faint_lengths=(61,))),
    ("S10", "走向场确认浅密划痕", "计算 32 像素尺度的划痕走向场（倍角平均）；待确认片段若细长（轴比 ≥2）、所在处走向相干度 ≥0.5、"
     "且与局部走向夹角 ≤20°，确认为划痕；无骨架的零散像素按连通域成为实例",
     "浅密划痕多成平行族，单个碎段虽短，但与周围走向一致",
     dict(grey_confirm=True)),
    ("S11", "交叉续接 v2", "端部局部剖面（沿轴线最近 60 像素）代替整条分支剖面；按端部曲率延拓判断走向连续，并要求两端曲率一致；"
     "剖面相似取端部与整条分支的较大者，曲率只在两端都可信时比较；交叉区内平方加权的整体最优配对；"
     "重建交叉区二元叠加所属（亮线为主、暗线为次，夹角 < 15° 的贴边段不记）",
     "密集交叉处暗线被亮线打断，且交叉像素只能归一方；弯曲划痕按直线外推会接错",
     dict(junction_mode="ends2", crossing_overlap=True)),
    ("S12", "损伤区（光束足迹判据，第二轮最终）", "3 mm 光束足迹内缺陷像素占比 ≥35% 的连通区为损伤区（面积至少一个足迹），"
     "按区内可追踪划痕（≥0.5 mm）像素占比分为“密集划痕型”（≥50%）与“碎裂型”；损伤区占有效区 ≥10% 为“大面积损伤”；碎裂型区内的待确认碎片归碎裂",
     "原碎裂定义缺乏光学含义，评审员之间也不一致；改从视功能（散射光、对比度）反推",
     dict(damage_mode="footprint")),
]


def stage_params() -> dict[str, dict]:
    acc: dict = {}
    out = {}
    for sid, *_rest, ov in STAGES:
        acc = {**acc, **ov}
        out[sid] = dict(acc)
    return out


def regions() -> dict[str, tuple[int, int, int]]:
    reg = {}
    for ls in SEM_LENSES:
        reg[ls] = tuple(json.loads((AC.SEM / "seg" / f"{ls}_info.json").read_text())["region"])
    d = cv2.imread(str(STD_DIR / "diffuse" / "55l.png"), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError("缺少 55l 弥散层")
    b = cv2.blur(d.astype(np.float32), (301, 301))
    y, x = np.unravel_index(int(b.argmax()), b.shape)
    reg["55l"] = (int(np.clip(x - 368, 0, d.shape[1] - 736)), int(np.clip(y - 368, 0, d.shape[0] - 736)), 736)
    return reg


# ── run ──────────────────────────────────────────────────────────────

def _job(args):
    sid, lens, ov = args
    out = OUT / sid
    try:
        f = run_lens(lens, str(out), ov)
        return sid, lens, f, None
    except Exception as exc:  # 记录后继续
        return sid, lens, None, str(exc)


def cmd_run(args: argparse.Namespace) -> None:
    params = stage_params()
    last = list(params)[-1]
    if params[last] != {k: v for k, v in TradParams().__dict__.items() if k in params[last]}:
        raise SystemExit(f"{last} 与 TradParams 默认值不一致，请同步")
    jobs = []
    for sid in args.stages or params:
        (OUT / sid).mkdir(parents=True, exist_ok=True)
        for ls in LENSES:
            if args.force or not (OUT / sid / f"{ls}.json").exists():
                jobs.append((sid, ls, params[sid]))
    print(f"待运行 {len(jobs)} 个任务", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for sid, ls, f, err in ex.map(_job, jobs):
            print(sid, ls, err or f"{f['seconds']} s scratch={f['n_scratch']} spot={f['n_spot']} "
                  f"uncertain={f['n_uncertain']} damage_mm2={f['damage_mm2']}", flush=True)


# ── analyze ──────────────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> None:
    res: dict = {"stages": {}}
    for sid, *_ in STAGES:
        AC.ANN = OUT / sid
        feats = {ls: json.loads((OUT / sid / f"{ls}.json").read_text())["features"] for ls in LENSES}
        px = AC.pixel_pilot()["methods"]
        sem = AC.semantic_pilot()
        grid = AC.grid_pilot()["overall"]
        res["stages"][sid] = {"features": feats, "pixel5": px, "semantic5": sem, "pilot5": grid}
        print(sid, "26l 中心线召回", px["trad_v1"]["center_recall"], "掩膜召回", px["trad_v1"]["ridge_recall"], "类别一致", sem["overall"]["class_agree"], flush=True)
    res["reference"] = {k: v for k, v in px.items() if k in ("hr_segment", "pilot_traditional")}
    (OUT / "ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"结果: {OUT / 'ablation.json'}")


# ── report ───────────────────────────────────────────────────────────

CLS_BGR = {1: (60, 210, 60), 2: (0, 160, 255), 3: (220, 70, 220), 4: (150, 150, 150)}


def esc(x) -> str:
    return html.escape(str(x))


def lens_figs(lens: str, reg: tuple[int, int, int]) -> dict:
    rx, ry, size = reg
    img, valid = load_standard(lens)
    d = display(img)
    d[~valid] = 0
    base = cv2.cvtColor(d[ry:ry + size, rx:rx + size], cv2.COLOR_GRAY2BGR)
    adir = OUT / "report_assets"
    cv2.imwrite(str(adir / f"{lens}_raw.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 88])
    figs = {"raw": f"report_assets/{lens}_raw.jpg"}
    rng = np.random.default_rng(5)
    for sid, *_ in STAGES:
        cls = cv2.imread(str(OUT / sid / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)[ry:ry + size, rx:rx + size]
        inst = cv2.imread(str(OUT / sid / f"{lens}_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)[ry:ry + size, rx:rx + size]
        v = base.copy()
        for c, col in CLS_BGR.items():
            v[cls == c] = col
        cv2.imwrite(str(adir / f"{lens}_{sid}_cls.jpg"), cv2.addWeighted(v, 0.75, base, 0.25, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        lut = rng.integers(60, 255, (int(inst.max()) + 1, 3)).astype(np.uint8)
        lut[0] = 0
        cv2.imwrite(str(adir / f"{lens}_{sid}_inst.jpg"), cv2.addWeighted(lut[inst], 0.8, base, 0.2, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        figs[sid] = (f"report_assets/{lens}_{sid}_cls.jpg", f"report_assets/{lens}_{sid}_inst.jpg")
    return figs


# ── 第二轮改进（S8 → S12）专节 ─────────────────────────────────────────

V1, V2 = "S8", "S12"
TYPE_CN = {"dense_scratch": "密集划痕型", "fragment": "碎裂型"}


def _load(sid: str, lens: str):
    d = OUT / sid
    cls = cv2.imread(str(d / f"{lens}_cls.png"), cv2.IMREAD_GRAYSCALE)
    inst = cv2.imread(str(d / f"{lens}_inst.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    p2 = d / f"{lens}_inst2.png"
    inst2 = cv2.imread(str(p2), cv2.IMREAD_UNCHANGED).astype(np.int32) if p2.exists() else np.zeros_like(inst)
    return cls, inst, inst2, json.loads((d / f"{lens}.json").read_text())


def lens_metrics(sid: str, lens: str) -> dict:
    cls, inst, inst2, J = _load(sid, lens)
    fg = cls > 0
    sc = [it for it in J["instances"] if it["class"] == "scratch"]
    lens_mm = np.array([it["len_px"] for it in sc]) * 0.0068
    return {"grey_share": float((cls[fg] == 4).mean()), "n_scratch": len(sc),
            "mean_len_mm": float(lens_mm.mean()) if len(sc) else 0.0,
            "len_ge2_n": int((lens_mm >= 2).sum()), "len_ge2_mm": float(lens_mm[lens_mm >= 2].sum()),
            "scratch_mm": float(lens_mm.sum()),
            "n_crossing": J["stats"]["n_crossing_links"], "n_gap": J["stats"]["n_gap_links"],
            "overlap_px": int((inst2 > 0).sum()), "seconds": J["features"]["seconds"]}


def field_view(mask: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """由划痕掩膜的结构张量得到走向场示意：色相 = 走向，饱和度 = 相干度，亮度 = 线密度。"""
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(m, cv2.CV_32F, 1, 0), cv2.Sobel(m, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(x, (0, 0), sigma) for x in (gx * gx, gy * gy, gx * gy))
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2          # 梯度主方向的垂直方向 = 线走向
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(jxx + jyy, 1e-6)
    dens = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    hsv = np.stack([(np.degrees(th) % 180).astype(np.uint8), np.clip(coh * 255, 0, 255).astype(np.uint8),
                    np.clip(dens / max(np.percentile(dens, 99), 1e-6) * 255, 0, 255).astype(np.uint8)], -1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rose(hist: list, size: int = 160) -> np.ndarray:
    """走向玫瑰图（0–180°，双向绘制）。"""
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    h = np.asarray(hist, float)
    h = h / max(h.max(), 1e-9)
    n = len(h)
    for k, v in enumerate(h):
        for sgn in (0, 180):
            a0, a1 = 180 * k / n + sgn, 180 * (k + 1) / n + sgn
            pts = [(c, c)] + [(int(c + v * (c - 8) * math.cos(math.radians(a))), int(c + v * (c - 8) * math.sin(math.radians(a))))
                              for a in np.linspace(a0, a1, 6)]
            cv2.fillPoly(img, [np.array(pts, np.int32)], (184, 120, 42))
    cv2.circle(img, (c, c), c - 8, (200, 200, 200), 1)
    cv2.putText(img, "0", (size - 14, c + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    cv2.putText(img, "90", (c - 7, size - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    return img


def v2_lens_figs(lens: str, reg: tuple[int, int, int]) -> dict:
    from darkfield_defects.annotation.trad_segment import footprint_damage
    rx, ry, size = reg
    img, valid = load_standard(lens)
    d = display(img)
    d[~valid] = 0
    adir = OUT / "report_assets"
    c1, i1, _, _ = _load(V1, lens)
    c2, i2, j2, J2 = _load(V2, lens)
    sl = np.s_[ry:ry + size, rx:rx + size]
    # 交叉区放大：语义区域内叠加像素最密的 240×240 窗口
    ov = cv2.blur((j2[sl] > 0).astype(np.float32), (121, 121))
    oy, ox = np.unravel_index(int(ov.argmax()), ov.shape)
    zy, zx = int(np.clip(ry + oy - 120, 0, img.shape[0] - 240)), int(np.clip(rx + ox - 120, 0, img.shape[1] - 240))
    zs = np.s_[zy:zy + 240, zx:zx + 240]
    zb = cv2.cvtColor(d[zs], cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(7)
    def iv(inst):
        lut = rng.integers(60, 255, (int(inst.max()) + 1, 3)).astype(np.uint8)
        lut[0] = 0
        return cv2.addWeighted(lut[inst], 0.85, zb, 0.15, 0)
    z1, z2 = iv(i1[zs]), iv(i2[zs])
    z3 = z2.copy()
    z3[j2[zs] > 0] = (0, 0, 255)
    big = lambda x: cv2.resize(x, (480, 480), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    out = {}
    for name, im in (("zoom_raw", zb), ("zoom_v1", z1), ("zoom_v2", z2), ("zoom_ov", z3)):
        cv2.imwrite(str(adir / f"{lens}_{name}.jpg"), big(im), [cv2.IMWRITE_JPEG_QUALITY, 90])
        out[name] = f"report_assets/{lens}_{name}.jpg"
    out["zoom_pos"] = (zx, zy)
    # 走向场（整镜 1/4）
    small = cv2.resize(((c2 == 1) & valid).astype(np.uint8), (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    fv = field_view(small > 0)
    fv[cv2.resize(valid.astype(np.uint8), (fv.shape[1], fv.shape[0])) == 0] = 0
    cv2.imwrite(str(adir / f"{lens}_field.jpg"), fv, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["field"] = f"report_assets/{lens}_field.jpg"
    cv2.imwrite(str(adir / f"{lens}_rose.png"), rose(J2["stats"]["orient_hist"]))
    out["rose"] = f"report_assets/{lens}_rose.png"
    # 足迹缺陷占比热图 + 损伤区轮廓
    fmap, dmg = footprint_damage(c2 > 0, valid, TradParams())
    q = (img.shape[1] // 4, img.shape[0] // 4)
    hm = cv2.applyColorMap(np.clip(cv2.resize(fmap, q) / 0.6 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm[cv2.resize(valid.astype(np.uint8), q) == 0] = 0
    lab = cv2.resize(dmg.astype(np.uint8), q, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    types = {tuple(r["pos"]): r["type"] for r in J2.get("damage_regions", [])}
    for c in cs:
        cx, cy = (c.reshape(-1, 2).mean(0) * 4).astype(int)
        kind = min(types.items(), key=lambda kv: math.hypot(kv[0][0] - cx, kv[0][1] - cy))[1] if types else "fragment"
        cv2.drawContours(hm, [c], -1, (255, 255, 255) if kind == "dense_scratch" else (255, 0, 255), 3)
    cv2.imwrite(str(adir / f"{lens}_footprint.jpg"), hm, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["footprint"] = f"report_assets/{lens}_footprint.jpg"
    return out


def v2_section(st: dict, reg: dict) -> str:
    v2ids = ["S8", "S9", "S10", "S11", "S12"]
    met = {s: {ls: lens_metrics(s, ls) for ls in LENSES} for s in v2ids}
    def row(name, fn):
        return f"<tr><td>{name}</td>" + "".join(f"<td>{fn(s)}</td>" for s in v2ids) + "</tr>"
    tot = lambda s, k: sum(met[s][ls][k] for ls in LENSES)  # noqa: E731
    so = lambda s: st[s]["semantic5"]["overall"]  # noqa: E731
    summary = "<table><tr><th>指标（6 张合计/平均）</th>" + "".join(f"<th>{s}</th>" for s in v2ids) + "</tr>" + "".join([
        row("灰色待确认像素 / 全部缺陷像素（平均）", lambda s: f"{np.mean([met[s][ls]['grey_share'] for ls in LENSES]):.1%}"),
        row("划痕总长 mm", lambda s: f"{tot(s, 'scratch_mm'):.0f}"),
        row("划痕实例平均长度 mm", lambda s: f"{np.mean([met[s][ls]['mean_len_mm'] for ls in LENSES]):.2f}"),
        row("≥2 mm 长划痕 条数 / 总长 mm", lambda s: f"{tot(s, 'len_ge2_n')} / {tot(s, 'len_ge2_mm'):.0f}"),
        row("交叉续接 / 断点桥接", lambda s: f"{tot(s, 'n_crossing')} / {tot(s, 'n_gap')}"),
        row("交叉区叠加像素", lambda s: f"{tot(s, 'overlap_px')}"),
        row("26l 中心线召回", lambda s: f"{st[s]['pixel5']['trad_v1']['center_recall']:.1%}"),
        row("26l 骨架落在助手划痕旁", lambda s: f"{st[s]['pixel5']['trad_v1']['ai_support']:.1%}"),
        row("语义试点：类别一致", lambda s: f"{so(s)['class_agree']:.0%}"),
        row("语义试点：“同一缺陷”对合并", lambda s: f"{so(s)['same_hit']}/{so(s)['same_total']}"),
        row("语义试点：“不同缺陷”对分开", lambda s: f"{so(s)['diff_hit']}/{so(s)['diff_total']}"),
        row("6 张总用时 s", lambda s: f"{tot(s, 'seconds'):.0f}")]) + "</table>"
    per = "<table><tr><th>镜片</th><th>灰色占比 S8→S12</th><th>划痕平均长度 mm S8→S12</th><th>≥2 mm 长划痕 S8→S12</th><th>叠加像素</th></tr>" + "".join(
        f"<tr><td>{ls}</td><td>{met[V1][ls]['grey_share']:.1%} → {met[V2][ls]['grey_share']:.1%}</td>"
        f"<td>{met[V1][ls]['mean_len_mm']:.2f} → {met[V2][ls]['mean_len_mm']:.2f}</td>"
        f"<td>{met[V1][ls]['len_ge2_n']} → {met[V2][ls]['len_ge2_n']}</td><td>{met[V2][ls]['overlap_px']}</td></tr>" for ls in LENSES) + "</table>"
    f12 = st[V2]["features"]
    dmg_rows, ori_rows, figs_html = "", "", ""
    for ls in LENSES:
        J = json.loads((OUT / V2 / f"{ls}.json").read_text())
        f = J["features"]
        regs = J.get("damage_regions", [])
        dmg_rows += (f"<tr><td>{ls}</td><td>{f['damage_frac_valid']:.1%}</td><td>{f.get('damage_frac_central', 0) or 0:.1%}</td>"
                     f"<td>{'是' if f['large_damage'] else '否'}</td><td>" + "<br>".join(
                         f"{TYPE_CN[r['type']]}：{r['area_mm2']} mm²，足迹占比 {r['footprint_frac_mean']:.2f}，可追踪 {r['traceable_frac']:.0%}，"
                         f"有序度 {r['orient_order']:.2f}，暗场强度 {r['scatter_mean_counts']}" for r in regs) + "</td></tr>")
        ori_rows += (f"<tr><td>{ls}</td><td>{f['orient_order']:.2f}</td><td>{f['orient_dominant_deg']:.0f}°</td>"
                     f"<td>{f['parallel_frac']:.0%}</td><td>{f['scratch_frac']:.1%}</td><td>{f['diffuse_mean']}</td></tr>")
        g = v2_lens_figs(ls, reg[ls])
        figs_html += f"""<section class="lens"><h3>{ls}（{esc(LENS_NOTE[ls])}）</h3>
<h4>密集交叉区放大（窗口左上角 {g['zoom_pos']}，240×240 像素）</h4>
<div class="g4"><figure><img src="{g['zoom_raw']}"><figcaption>标准输入</figcaption></figure>
<figure><img src="{g['zoom_v1']}"><figcaption>S8 当前：实例（同色 = 同一条）</figcaption></figure>
<figure><img src="{g['zoom_v2']}"><figcaption>S12 改进：实例</figcaption></figure>
<figure><img src="{g['zoom_ov']}"><figcaption>S12：红 = 交叉区第二所属（暗线穿过亮线）</figcaption></figure></div>
<h4>划痕走向与损伤区（整镜）</h4>
<div class="g3"><figure><img src="{g['field']}"><figcaption>走向场：色相 = 走向，饱和度 = 平行程度，亮度 = 密度</figcaption></figure>
<figure><img src="{g['rose']}" style="max-width:220px;background:#fff"><figcaption>划痕走向玫瑰图（长度加权）</figcaption></figure>
<figure><img src="{g['footprint']}"><figcaption>3 mm 足迹内缺陷像素占比（0–60%）；白框 = 密集划痕型损伤区，紫框 = 碎裂型</figcaption></figure></div>
<div class="g3"><figure><img src="{lens_cls_pair(ls, 'S8')}"><figcaption>S8 当前：类别图</figcaption></figure>
<figure><img src="{lens_cls_pair(ls, 'S12')}"><figcaption>S12 改进：类别图</figcaption></figure></div></section>"""
    samp_html, samp = sample_panels()
    xs = None
    if (OUT / "X_S8_crossv2").exists():
        ev, sr = _cover_stats(V1, "X_S8_crossv2")
        xs = (sum(e["kind"] == "merge" for e in ev), sum(e["kind"] == "split" for e in ev), sr)
    ms_note = (f"<p>相对 S8：S12 中有 {samp['merge']} 条 ≥1 mm 划痕由 S8 的多段合并而来，S8 的 ≥1 mm 划痕有 {samp['split_rate']:.0%} 在 S12 中被拆分"
               + (f"；只加交叉续接 v2（不加长尺度层）时为合并 {xs[0]} 条、拆分 {xs[2]:.0%}" if xs else "")
               + "。合并与拆分都包含改对与改错的情况，需要按下方抽样逐例判断。</p>")
    return f"""<div class="card"><h4>汇总</h4>{summary}<h4>逐镜片</h4>{per}
<h4>续接变化：合并与拆分</h4>{ms_note}{samp_html}
<h4>损伤区（S12，3 mm 足迹判据）</h4><table><tr><th>镜片</th><th>损伤区 / 有效区</th><th>损伤区 / 中央 9 mm 区</th><th>大面积损伤（≥10%）</th><th>各损伤区描述</th></tr>{dmg_rows}</table>
<h4>划痕走向统计（S12）</h4><table><tr><th>镜片</th><th>取向有序度（0 随机 – 1 平行）</th><th>主走向</th><th>处于平行族的划痕比例</th><th>划痕面积占比</th><th>预处理弥散层均值</th></tr>{ori_rows}</table>
<p class="meta">走向角从图像 +x 轴量起、向 +y（向下）为正。6 张镜片不足以判断平行划痕与雾化/光晕的相关性，需要在 247 张全量结果上检验。</p></div>
{figs_html}"""



def _cover_stats(ref: str, cand: str) -> tuple[list, float]:
    """cand 中 ≥1 mm 划痕各覆盖 ref 的实例数（≥10% 像素计一个），以及 ref 中 ≥1 mm 划痕被 cand 拆分的比例。"""
    from collections import Counter
    from scipy import ndimage as ndi
    events, split_n, ref_n = [], 0, 0
    for ls in LENSES:
        _, ir, _, Jr = _load(ref, ls)
        _, ic, _, Jc = _load(cand, ls)
        for (a, ia, Ja, b, ib, kind) in ((cand, ic, Jc, ref, ir, "merge"), (ref, ir, Jr, cand, ic, "split")):
            objs = ndi.find_objects(ia)
            for it in Ja["instances"]:
                if it["class"] != "scratch" or it["len_px"] * 0.0068 < 1.0 or it["id"] - 1 >= len(objs):
                    continue
                sl = objs[it["id"] - 1]
                if sl is None:
                    continue
                v = ib[sl][ia[sl] == it["id"]]
                v = v[v > 0]
                if v.size < 20:
                    continue
                parts = [k for k, c in Counter(v.tolist()).items() if c >= 0.1 * v.size]
                if kind == "split":
                    ref_n += 1
                    split_n += len(parts) >= 2
                if len(parts) >= 2:
                    events.append({"lens": ls, "kind": kind, "id": it["id"], "parts": parts, "sl": sl})
    return events, split_n / max(ref_n, 1)


def sample_panels(cand: str = V2, n: int = 12, seed: int = 0) -> tuple[str, dict]:
    """抽样核对：随机抽取 cand 相对 S8 的“合并”与“拆分”事件，并排显示原图 / S8 / cand。"""
    events, split_rate = _cover_stats(V1, cand)
    rng = np.random.default_rng(seed)
    adir = OUT / "report_assets"
    html_parts, stats = {}, {"merge": sum(e["kind"] == "merge" for e in events),
                             "split": sum(e["kind"] == "split" for e in events), "split_rate": split_rate}
    cache = {}
    for kind in ("merge", "split"):
        pool = [e for e in events if e["kind"] == kind and e["lens"] != "108l"]
        pick = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)] if pool else []
        cells = []
        for k, e in enumerate(pick):
            ls = e["lens"]
            if ls not in cache:
                img, valid = load_standard(ls)
                d = display(img)
                d[~valid] = 0
                cache[ls] = (d, _load(V1, ls)[1], _load(cand, ls)[1])
            d, i1, i2 = cache[ls]
            sl = e["sl"]
            cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
            hh = min(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) // 2 + 20, 200)
            y0, x0 = max(0, cy - hh), max(0, cx - hh)
            win = np.s_[y0:cy + hh, x0:cx + hh]
            base = cv2.cvtColor(d[win], cv2.COLOR_GRAY2BGR)
            own_ref, own_cand = (i2, i1) if kind == "merge" else (i1, i2)   # 事件实例所在的结果 / 被比较的结果
            pal = [(60, 200, 255), (80, 220, 80), (255, 120, 60), (220, 80, 220), (60, 60, 255), (255, 255, 80)]
            def paint(inst, ids):
                v = (base * 0.35).astype(np.uint8)
                sub = inst[win]
                v[sub > 0] = (90, 90, 90)
                for q, i in enumerate(ids):
                    v[sub == i] = pal[q % len(pal)]
                return v
            if kind == "merge":
                a_img, b_img = paint(i1, e["parts"]), paint(i2, [e["id"]])
            else:
                a_img, b_img = paint(i1, [e["id"]]), paint(i2, e["parts"])
            strip = np.hstack([cv2.resize(x, (220, 220), interpolation=cv2.INTER_NEAREST) for x in (base, a_img, b_img)])
            name = f"sample_{kind}_{k}.jpg"
            cv2.imwrite(str(adir / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cells.append(f'<figure><img src="report_assets/{name}"><figcaption>#{k + 1} {ls} ({x0}, {y0})：'
                         f'{"S8 的 " + str(len(e["parts"])) + " 段合并为一条" if kind == "merge" else "S8 的一条拆为 " + str(len(e["parts"])) + " 段"}'
                         f'　判断：□对 □错 □不确定</figcaption></figure>')
        html_parts[kind] = "".join(cells)
    html = f"""<p class="meta">每例三联图：标准输入 | S8 当前 | {cand} 改进；彩色为事件涉及的实例，灰色为其他实例。合并例中左图各色为 S8 的多段、右图为改进后的一条；拆分例相反。</p>
<h4>合并（S8 的多段 → 改进后一条），共 {stats['merge']} 例，随机抽 {min(n, stats['merge'])} 例</h4><div class="g2">{html_parts['merge']}</div>
<h4>拆分（S8 的一条 → 改进后多段），共 {stats['split']} 例，随机抽 {min(n, stats['split'])} 例</h4><div class="g2">{html_parts['split']}</div>"""
    return html, stats

def lens_cls_pair(lens: str, sid: str) -> str:
    return f"report_assets/{lens}_{sid}_cls.jpg"

def cmd_report(args: argparse.Namespace) -> None:
    (OUT / "report_assets").mkdir(parents=True, exist_ok=True)
    ab = json.loads((OUT / "ablation.json").read_text())
    st = ab["stages"]
    reg = regions()
    sids = [s[0] for s in STAGES]
    stage_rows = "".join(f"<tr><td><b>{sid}</b></td><td>{esc(name)}</td><td>{esc(chg)}</td><td>{esc(why)}</td>"
                         f"<td><code>{esc(', '.join(f'{k}={v}' for k, v in ov.items()))}</code></td></tr>"
                         for sid, name, chg, why, ov in STAGES)

    def agg(sid):
        s = st[sid]
        f = s["features"]
        so = s["semantic5"]["overall"]
        px = s["pixel5"]["trad_v1"]
        g = s["pilot5"]
        return (f"<tr><td><b>{sid}</b></td><td>{px['center_recall']:.1%}</td><td>{px['ridge_recall']:.1%}</td><td>{px['fg_frac']:.1%}</td><td>{px['ai_support']:.1%}</td>"
                f"<td>{so['class_agree']:.0%}</td><td>{so['same_hit']}/{so['same_total']}</td><td>{so['diff_hit']}/{so['diff_total']}</td>"
                f"<td>{so['wrong_split']}/{so['wrong_total']}</td>"
                f"<td>{g['scratch']['trad_v1_recall']} / {g['scratch']['trad_v1_precision']}</td>"
                f"<td>{g['spot']['trad_v1_recall']} / {g['spot']['trad_v1_precision']}</td>"
                f"<td>{g['damage']['trad_v1_recall']} / {g['damage']['trad_v1_precision']}</td>"
                f"<td>{sum(f[ls]['seconds'] for ls in LENSES):.0f}</td></tr>")
    ref = ab["reference"]
    agg_rows = "".join(agg(s) for s in sids)

    def per_lens_table(key, fmt):
        return "".join(f"<tr><td>{ls}</td>" + "".join(f"<td>{fmt(st[s]['features'][ls][key])}</td>" for s in sids) + "</tr>"
                       for ls in LENSES)
    head = "<tr><th>镜片</th>" + "".join(f"<th>{s}</th>" for s in sids) + "</tr>"
    tabs = [("划痕实例数", "n_scratch", str), ("斑点数", "n_spot", str), ("待确认实例数", "n_uncertain", str),
            ("划痕面积占比", "scratch_frac", lambda v: f"{v:.1%}"), ("碎裂面积 mm²", "damage_mm2", str),
            ("交叉续接数", "n_crossing", str), ("断点桥接数", "n_gap", str), ("复核队列项", "n_review_items", str)]
    lens_tabs = "".join(f"<h4>{t}</h4><table>{head}{per_lens_table(k, f)}</table>" for t, k, f in tabs)
    secs = []
    for ls in LENSES:
        figs = lens_figs(ls, reg[ls])
        rx, ry, size = reg[ls]
        cls_grid = "".join(f'<figure><a href="{figs[s][0]}"><img src="{figs[s][0]}" loading="lazy"></a>'
                           f'<figcaption>{s} {esc(STAGES[i][1])}</figcaption></figure>' for i, s in enumerate(sids))
        inst_grid = "".join(f'<figure><a href="{figs[s][1]}"><img src="{figs[s][1]}" loading="lazy"></a>'
                            f'<figcaption>{s} {esc(STAGES[i][1])}</figcaption></figure>' for i, s in enumerate(sids))
        sem = ""
        if ls in SEM_LENSES:
            sem = "<table><tr><th>指标</th>" + "".join(f"<th>{s}</th>" for s in sids) + "</tr>" + "".join(
                f"<tr><td>{name}</td>" + "".join(f"<td>{fn(st[s]['semantic5']['lenses'][ls])}</td>" for s in sids) + "</tr>"
                for name, fn in (("与助手多数类别一致", lambda v: f"{v['class_agree']:.0%}"),
                                 ("“同一缺陷”对归入同一实例", lambda v: f"{v['group_same_hit']}/{v['group_same_pairs']}"),
                                 ("“不同缺陷”对分开", lambda v: f"{v['group_diff_hit']}/{v['group_diff_pairs']}"),
                                 ("待确认像素占比", lambda v: f"{v['trad_uncertain_share']:.0%}"))) + "</table>"
        secs.append(f"""<section class="lens" id="{ls}"><h3>{ls}（{esc(LENS_NOTE[ls])}）　区域 ({rx}, {ry})，{size}×{size} 像素</h3>
<div class="g3"><figure><img src="{figs['raw']}"><figcaption>标准输入（平方根拉伸）</figcaption></figure>
<figure><img src="{figs['S0'][0]}"><figcaption>S0 初版</figcaption></figure>
<figure><img src="{figs['S8'][0]}"><figcaption>S8 最终</figcaption></figure></div>
<details><summary>各阶段类别图（绿=划痕，橙=斑点，紫=碎裂，灰=待确认；点击放大）</summary><div class="g3">{cls_grid}</div></details>
<details><summary>各阶段实例图（同色 = 同一缺陷，用于看续接效果）</summary><div class="g3">{inst_grid}</div></details>
{sem}</section>""")
    final = TradParams().__dict__
    prm = "".join(f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>" for k, v in final.items())
    obs_p = OUT / "observations.md"
    from report_review_semantic import md_to_html
    obs = md_to_html(obs_p.read_text(encoding="utf-8")) if obs_p.exists() else "<p>（尚未撰写）</p>"
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>基础标注算法迭代</title><style>
:root {{ --bg:#f6f7f9; --card:#fff; --ink:#1d2430; --mute:#5c6675; --line:#dde2ea; --accent:#2a63b8; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1280px; margin:0 auto; padding:24px 20px 60px; }} h1 {{ font-size:25px; }} h2 {{ font-size:20px; margin-top:34px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:18px; margin:0 0 6px; }} h4 {{ font-size:15px; color:var(--accent); margin:16px 0 8px; }} .meta {{ color:var(--mute); }}
.card, section.lens {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }} th, td {{ border:1px solid var(--line); padding:5px 7px; text-align:left; vertical-align:top; }} th {{ background:#eef2f8; }}
.g3 {{ display:grid; gap:10px; grid-template-columns:repeat(3,1fr); margin:8px 0; }} .g4 {{ display:grid; gap:8px; grid-template-columns:repeat(4,1fr); margin:8px 0; }} .g2 {{ display:grid; gap:10px; grid-template-columns:repeat(2,1fr); margin:8px 0; }} details {{ margin:8px 0; }} summary {{ cursor:pointer; color:var(--accent); }}
figure {{ margin:0; }} figure img {{ width:100%; border-radius:6px; border:1px solid var(--line); background:#000; }} figcaption {{ font-size:12px; color:var(--mute); text-align:center; }}
@media (max-width:760px) {{ .g2, .g3, .g4 {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>传统基础标注算法：迭代过程复现与最终参数确认</h1>
<p class="meta">6 张镜片（108l、26l、90r、76r、72l 为前几轮试点；55l 为雾化最重的镜片）× 9 个累积阶段 | 代码：src/darkfield_defects/annotation/trad_segment.py（开关默认即最终方案）、scripts/annot_ablation.py | 结果：output/annotation/ablation/</p>
<h2>一、结论与待确认事项</h2><div class="card">{obs}</div>
<h2>二、迭代阶段</h2><div class="card"><table><tr><th>阶段</th><th>名称</th><th>本阶段改动</th><th>改动原因（试验中发现的问题）</th><th>参数变化</th></tr>{stage_rows}</table>
<p class="meta">S0 是对初版的近似复现：初版的宽度扩展邻域（5×5）等细节没有保留开关，其余行为与当时一致。</p></div>
<h2>三、各阶段指标</h2><div class="card">
<table><tr><th>阶段</th><th>26l 中心线召回（骨架 3 像素内）</th><th>26l 掩膜召回（前景 3 像素内）</th><th>26l 象限前景占比</th><th>骨架落在助手划痕旁</th><th>语义试点类别一致</th><th>“同一缺陷”对合并</th><th>“不同缺陷”对分开</th><th>错误连接已拆开</th><th>网格 划痕 召回/精确</th><th>网格 斑点 召回/精确</th><th>网格 碎裂 召回/精确</th><th>6 张总用时 s</th></tr>{agg_rows}
<tr><td>参考：HR 分割</td><td>{ref['hr_segment']['center_recall']:.1%}</td><td>{ref['hr_segment']['ridge_recall']:.1%}</td><td>{ref['hr_segment']['fg_frac']:.1%}</td><td>{ref['hr_segment']['ai_support']:.1%}</td><td colspan="8">上一轮语义试点所用的高召回分割</td></tr>
<tr><td>参考：首轮传统规则</td><td>{ref['pilot_traditional']['center_recall']:.1%}</td><td>{ref['pilot_traditional']['ridge_recall']:.1%}</td><td>{ref['pilot_traditional']['fg_frac']:.1%}</td><td>{ref['pilot_traditional']['ai_support']:.1%}</td><td colspan="8"></td></tr></table>
<p class="meta">召回参考 = 26l 右上象限三名助手画出、且中心线两侧确有亮脊的划痕；语义试点 = 上一轮 5 块 736×736 区域上三助手的多数意见（“同一/不同缺陷”按宽松口径）；网格 = 首轮 16×12 网格试点的多数格子（当时 R2 为 Sonnet）。助手标注本身有漏标与分歧，这些数只作相对比较。</p>
{lens_tabs}</div>
<h2>四、第二轮改进：S8（当前）→ S12（改进）</h2>{v2_section(st, reg)}
<h2>五、逐镜片各阶段对比</h2>{''.join(secs)}
<h2>六、最终参数（S12 = TradParams 默认值）</h2><div class="card"><table>{prm}</table></div>
</main></body></html>"""
    (OUT / "ablation_report.html").write_text(doc, encoding="utf-8")
    print(f"报告: {OUT / 'ablation_report.html'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stages", nargs="*")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
