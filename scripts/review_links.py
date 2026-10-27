"""划痕续接判读：把“合并/拆分是否正确”统一改问“这两段是不是同一条划痕”，逐例盲判。

事件来自 annot_ablation 的 S8 → S12 续接变化（抽样与报告一致：每类 n 例，seed 相同，不含 108l）：
- 合并：S12 的一条覆盖 S8 的多段 → 对 S8 相邻两段提问，全部答“是”则合并正确；
- 拆分：S8 的一条被 S12 拆成多段 → 对 S12 相邻两段提问，答“否”则拆分正确。
判读页只显示两段（青 A、品红 B）及其交界处，不显示算法做了什么，避免诱导。

每例三幅图：交界处原图（至少 2 倍放大，窗口内自适应平方根拉伸，刻度标出交界点）| 同一窗口叠加 A/B | 4 倍范围的走向上下文。
判读页为静态 HTML：键盘 1 是 / 2 否 / 3 看不清，自动保存在浏览器；经 annot_server 在局域网访问（/links/）时
同时保存到服务端 results_server.json，可直接用 score 统计；本地打开时导出 JSON 后统计。

用法:
    uv run python scripts/review_links.py build
    uv run python scripts/review_links.py score --results ~/Downloads/link_review_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import annot_ablation as AB  # noqa: E402
from review_pixel_pilot import display, load_standard  # noqa: E402

OUT = AB.OUT / "link_review"
SIZE = 480          # 每幅图的显示边长（像素）
HALF = 120          # 交界窗口最小半宽（原图像素，即 2 倍放大）；两段间隙较大时放宽到能同时看到两端
CTX_SCALE = 4       # 上下文窗口是交界窗口的 4 倍
COLOR_A = (255, 255, 0)     # BGR 青
COLOR_B = (255, 0, 255)     # BGR 品红


def sample_events(n: int, seed: int) -> list[dict]:
    """与 annot_ablation.sample_panels 相同的抽样（同一 rng 顺序：先合并后拆分）。"""
    events, _ = AB._cover_stats(AB.V1, AB.V2)
    rng = np.random.default_rng(seed)
    picked = []
    for kind in ("merge", "split"):
        pool = [e for e in events if e["kind"] == kind and e["lens"] != "108l"]
        if pool:
            picked += [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    return picked


def nearest(pa: np.ndarray, pb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    d, j = cKDTree(pb).query(pa)
    i = int(np.argmin(d))
    return float(d[i]), pa[i], pb[j[i]]


def adjacent_pairs(masks: dict[int, np.ndarray]) -> list[tuple[int, int, np.ndarray, float]]:
    """按最近距离做最小生成树（同一事件的各段沿同一条线排列，相邻段之间常隔着交叉区或桥接间隙）；
    返回 (段1, 段2, 交界点 yx, 间隙像素)。"""
    ids = list(masks)
    pts = {i: np.argwhere(masks[i]) for i in ids}
    edges = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            d, pa, pb = nearest(pts[ids[a]], pts[ids[b]])
            edges.append((d, ids[a], ids[b], (pa + pb) / 2))
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    out = []
    for d, a, b, c in sorted(edges, key=lambda e: e[0]):
        if find(a) != find(b):
            parent[find(a)] = find(b)
            out.append((a, b, c, d))
    return out


def crop(arr: np.ndarray, cy: int, cx: int, half: int, fill=0) -> np.ndarray:
    """以 (cy, cx) 为中心裁 2·half 的方窗，越界部分填 fill。"""
    h, w = arr.shape[:2]
    out = np.full((2 * half, 2 * half) + arr.shape[2:], fill, arr.dtype)
    y0, x0 = cy - half, cx - half
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + 2 * half), min(w, x0 + 2 * half)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = arr[ys:ye, xs:xe]
    return out


def tint(gray: np.ndarray, ma: np.ndarray, mb: np.ndarray, alpha: float) -> np.ndarray:
    v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for m, c in ((ma, COLOR_A), (mb, COLOR_B)):
        v[m] = (1 - alpha) * v[m] + alpha * np.array(c, np.float32)
    return v.astype(np.uint8)


def ticks(img: np.ndarray, r0: int = 16, r1: int = 26) -> np.ndarray:
    """在窗口中心外围画四个短刻度，标出交界点而不遮挡交界处的像素。"""
    c = img.shape[0] // 2
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        cv2.line(img, (c + dx * r0, c + dy * r0), (c + dx * r1, c + dy * r1), (0, 200, 255), 2)
    return img


def stretch(x: np.ndarray, floor: float = 16.0) -> np.ndarray:
    """窗口内自适应平方根拉伸：以 99.7 分位为白点（不低于 floor），密集区交叉处不饱和，亮度差异可比。"""
    hi = max(float(np.percentile(x, 99.7)), floor)
    return (255 * np.sqrt(np.clip(x.astype(np.float32) / hi, 0, 1))).astype(np.uint8)


def render(img: np.ndarray, d: np.ndarray, ma: np.ndarray, mb: np.ndarray, cy: int, cx: int,
           half: int) -> tuple[np.ndarray, ...]:
    def win(x, h, interp):
        return cv2.resize(crop(x, cy, cx, h), (SIZE, SIZE), interpolation=interp)
    g = cv2.resize(stretch(crop(img, cy, cx, half)), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    a = win(ma.astype(np.uint8), half, cv2.INTER_NEAREST) > 0
    b = win(mb.astype(np.uint8), half, cv2.INTER_NEAREST) > 0
    raw = ticks(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    ov = tint(g, a, b, 0.5)
    ch = half * CTX_SCALE
    ctx = tint(win(d, ch, cv2.INTER_AREA), win(ma.astype(np.uint8) * 255, ch, cv2.INTER_AREA) > 0,
               win(mb.astype(np.uint8) * 255, ch, cv2.INTER_AREA) > 0, 0.8)
    q = int(round((ch - half) * SIZE / (2 * ch)))
    cv2.rectangle(ctx, (q, q), (SIZE - q, SIZE - q), (0, 200, 255), 1)
    return raw, ov, ctx


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    events = sample_events(args.n, args.seed)
    cache: dict[str, tuple] = {}
    cases, key = [], {}
    for k, e in enumerate(events):
        ls = e["lens"]
        if ls not in cache:
            img, valid = load_standard(ls)
            img[~valid] = 0
            cache[ls] = (img, display(img), AB._load(AB.V1, ls)[1], AB._load(AB.V2, ls)[1])
        img, d, i8, i12 = cache[ls]
        part_inst = i8 if e["kind"] == "merge" else i12   # 提问对象：合并看 S8 的各段，拆分看 S12 的各段
        masks = {p: part_inst == p for p in e["parts"]}
        for a, b, c, gap in adjacent_pairs(masks):
            cid = f"c{len(key) + 1:03d}"
            cy, cx = int(round(c[0])), int(round(c[1]))
            half = max(HALF, int(gap / 2) + 60)
            for tag, im in zip(("raw", "ov", "ctx"), render(img, d, masks[a], masks[b], cy, cx, half)):
                cv2.imwrite(str(out / "img" / f"{cid}_{tag}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cases.append({"id": cid, "lens": ls, "xy": [cx, cy], "zoom": round(SIZE / (2 * half), 2),
                          "gap_px": round(gap)})
            key[cid] = {"event": k, "kind": e["kind"], "lens": ls, "event_inst": e["id"],
                        "event_version": AB.V2 if e["kind"] == "merge" else AB.V1,
                        "pair": [int(a), int(b)], "pair_version": AB.V1 if e["kind"] == "merge" else AB.V2,
                        "n_parts": len(e["parts"]), "xy": [cx, cy], "gap_px": round(gap)}
    order = np.random.default_rng(args.seed + 1).permutation(len(cases))   # 打乱，合并/拆分混排
    cases = [cases[i] for i in order]
    (out / "key.json").write_text(json.dumps({"n": args.n, "seed": args.seed, "versions": [AB.V1, AB.V2],
                                              "cases": key}, ensure_ascii=False, indent=1))
    (out / "index.html").write_text(PAGE.replace("__CASES__", json.dumps(cases, ensure_ascii=False)),
                                    encoding="utf-8")
    print(f"{len(events)} 个事件 → {len(cases)} 个判读例：{out / 'index.html'}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cmd_score(args: argparse.Namespace) -> None:
    out = Path(args.out)
    key = json.loads((out / "key.json").read_text())["cases"]
    ans = json.loads(Path(args.results).expanduser().read_text())["answers"]
    missing = sorted(set(key) - set(ans))
    if missing:
        print(f"未判读 {len(missing)} 例：{' '.join(missing)}")
    by_event: dict[int, list] = {}
    for cid, k in key.items():
        if cid in ans:
            by_event.setdefault(k["event"], []).append((cid, k, ans[cid]["v"]))
    res = {"merge": [], "split": []}
    for ev, items in sorted(by_event.items()):
        kind = items[0][1]["kind"]
        vs = [v for _, _, v in items]
        if "unsure" in vs:
            verdict = None
        elif kind == "merge":
            verdict = all(v == "same" for v in vs)        # 合并正确：相邻各段都是同一条
        else:
            verdict = any(v == "diff" for v in vs)        # 拆分正确：至少一处确实不是同一条
        res[kind].append({"event": ev, "lens": items[0][1]["lens"], "cases": [c for c, _, _ in items],
                          "answers": vs, "correct": verdict})
    summary = {}
    for kind, rows in res.items():
        judged = [r for r in rows if r["correct"] is not None]
        k = sum(r["correct"] for r in judged)
        lo, hi = wilson(k, len(judged))
        summary[kind] = {"correct": k, "judged": len(judged), "unsure": len(rows) - len(judged),
                         "rate": round(k / len(judged), 3) if judged else None, "ci95": [round(lo, 3), round(hi, 3)]}
        name = "合并" if kind == "merge" else "拆分"
        print(f"{name}：正确 {k}/{len(judged)}（95% 区间 {lo:.0%}–{hi:.0%}），看不清 {len(rows) - len(judged)}")
        for r in judged:
            if not r["correct"]:
                print(f"    错误 {name}：{r['lens']} 判读例 {' '.join(r['cases'])}")
    (out / "score.json").write_text(json.dumps({"summary": summary, "events": res}, ensure_ascii=False, indent=1))


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>划痕续接判读</title>
<style>
:root{--bg:#16181c;--fg:#e6e6e6;--muted:#9aa0a6;--card:#202329;--acc:#ffc800;--a:#00ffff;--b:#ff00ff}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 6px}
.help{color:var(--muted);font-size:14px;margin:0 0 12px}
.a{color:var(--a)}.b{color:var(--b)}
.row{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0;background:var(--card);padding:6px;border-radius:6px}
figure img{display:block;width:min(480px,92vw);image-rendering:pixelated}
figcaption{color:var(--muted);font-size:13px;margin-top:4px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}
button{background:var(--card);color:var(--fg);border:1px solid #444;border-radius:6px;padding:8px 14px;font-size:15px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}
#prog{display:flex;gap:3px;flex-wrap:wrap;margin-top:10px}
#prog span{width:18px;height:18px;border-radius:3px;background:#333;cursor:pointer;font-size:10px;text-align:center;line-height:18px}
#prog span.same{background:#2e7d32}#prog span.diff{background:#c62828}#prog span.unsure{background:#777}
#prog span.cur{outline:2px solid var(--acc)}
input[type=text]{background:var(--card);color:var(--fg);border:1px solid #444;border-radius:6px;padding:7px;min-width:260px}
</style></head><body><main>
<h1>划痕续接判读 <span id="pos"></span></h1>
<p class="help">问题：<b><span class="a">青色 A</span> 和 <span class="b">品红 B</span> 是不是<b>同一条</b>划痕（同一次刮擦留下的连续痕迹）？</b>
看交界处（左图黄色刻度中心）两侧的<b>走向是否平滑延续</b>、<b>宽度与亮度是否相近</b>；有别的划痕从交界处横穿过去不影响判断。
右图是 4 倍范围的上下文，便于看长程走向。左图为原图，未叠色，淡线以它为准。<br>
键盘：<b>1</b> 是同一条　<b>2</b> 不是　<b>3</b> 看不清　<b>←/→</b> 翻页。结果自动保存在本浏览器，完成后点“导出结果”。</p>
<div class="row">
 <figure><img id="raw"><figcaption id="cap">交界处原图</figcaption></figure>
 <figure><img id="ov"><figcaption>同一窗口，A/B 叠色</figcaption></figure>
 <figure><img id="ctx"><figcaption>上下文（黄框为左图范围）</figcaption></figure>
</div>
<div class="bar">
 <button data-v="same">1 是同一条</button><button data-v="diff">2 不是</button><button data-v="unsure">3 看不清</button>
 <input type="text" id="note" placeholder="备注（可选）">
 <button id="prev">← 上一例</button><button id="next">下一例 →</button>
 <button id="export">导出结果</button><span id="meta" class="help"></span>
</div>
<div id="prog"></div>
</main><script>
const CASES = __CASES__;
const KEY = "link_review_v1";
let st = {answers:{}};
try { st = JSON.parse(localStorage.getItem(KEY)) || st; } catch (e) {}
let i = 0;
const $ = s => document.querySelector(s);
const SERVED = location.protocol.startsWith("http");     // 经 annot_server 访问时同步保存到服务端
function save(){
  try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {}
  if (SERVED) fetch("/api/links", {method: "POST", headers: {"Content-Type": "application/json"},
                                   body: JSON.stringify(st)}).catch(() => {});
}
function show(){
  const c = CASES[i];
  for (const t of ["raw","ov","ctx"]) $("#"+t).src = `img/${c.id}_${t}.jpg`;
  $("#pos").textContent = `${i+1} / ${CASES.length}`;
  $("#meta").textContent = `${c.id} · ${c.lens} · (${c.xy[0]}, ${c.xy[1]})`;
  $("#cap").textContent = `交界处原图（${c.zoom} 倍放大；两段相距 ${c.gap_px} 像素）`;
  const a = st.answers[c.id];
  document.querySelectorAll("button[data-v]").forEach(b => b.classList.toggle("on", !!a && a.v === b.dataset.v));
  $("#note").value = a && a.note ? a.note : "";
  $("#prog").innerHTML = CASES.map((c, k) => {
    const v = st.answers[c.id] ? st.answers[c.id].v : "";
    return `<span class="${v} ${k===i?"cur":""}" data-k="${k}">${k+1}</span>`;
  }).join("");
}
function answer(v){
  const c = CASES[i];
  st.answers[c.id] = {v, note: $("#note").value, t: new Date().toISOString()};
  save();
  if (i < CASES.length - 1) i++;
  show();
}
document.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => answer(b.dataset.v));
$("#prev").onclick = () => { if (i > 0) { i--; show(); } };
$("#next").onclick = () => { if (i < CASES.length - 1) { i++; show(); } };
$("#prog").onclick = e => { if (e.target.dataset.k) { i = +e.target.dataset.k; show(); } };
$("#note").onchange = () => { const a = st.answers[CASES[i].id]; if (a) { a.note = $("#note").value; save(); } };
document.addEventListener("keydown", e => {
  if (e.target.id === "note") return;
  const m = {"1":"same","2":"diff","3":"unsure"}[e.key];
  if (m) answer(m);
  else if (e.key === "ArrowLeft") $("#prev").click();
  else if (e.key === "ArrowRight") $("#next").click();
});
$("#export").onclick = () => {
  const n = Object.keys(st.answers).length;
  if (n < CASES.length && !confirm(`还有 ${CASES.length - n} 例未判读，仍要导出吗？`)) return;
  const blob = new Blob([JSON.stringify({exported: new Date().toISOString(), answers: st.answers}, null, 1)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "link_review_results.json"; a.click();
};
if (SERVED) fetch("/api/links").then(r => r.ok ? r.json() : null).then(s => {
  if (s && s.answers) { st.answers = Object.assign({}, s.answers, st.answers); show(); }
}).catch(() => {});
show();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=12, help="每类（合并/拆分）抽样事件数")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(OUT))
    s = sub.add_parser("score")
    s.add_argument("--results", required=True, help="判读页导出的 JSON")
    s.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {"build": cmd_build, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
