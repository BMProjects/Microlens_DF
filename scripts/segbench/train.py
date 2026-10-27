"""分割候选统一训练与评测（固定预算、同一数据管线、FCN 指标）。

协议（MSD，p622 划分）：
- 训练：全分辨率随机裁剪 crop×crop，一半裁剪以随机前景像素为中心（缺陷像素占比 < 1%）；翻转 + 90° 旋转 + 轻度亮度对比度扰动；
  AdamW + 多项式学习率；交叉熵 + 多类 Dice；bf16 自动混合精度；
- 选模：每 val_every 次迭代在验证集全图推理，保存验证集 mIoU 最高的检查点；
- 测试：训练结束后用最佳检查点在测试集全图推理一次，报告 PA、MPA、mIoU、FWIoU、分类别 IoU/Dice
  以及“全部预测为背景”的 PA 参照。
输出：output/segbench/runs/<data>_<model>_<tag>/{config.json, history.json, best.pt, test_metrics.json, train.log}

用法:
    uv run --no-sync python scripts/segbench/train.py --model unet_r34 --iters 200 --tag smoke
    uv run --no-sync python scripts/segbench/train.py --model dinov3_unet --iters 16000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import models as M  # noqa: E402
from darkfield_defects.eval.seg_metrics import ConfusionMatrix, all_background_pa  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_rgb(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_mask(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(p)
    return m


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class MSDData:
    """MSD：RGB jpg + 单通道类别 png。"""
    offset = 10.0          # 亮度扰动的加性偏移上限

    def __init__(self):
        self.root = PROJECT_ROOT / "output/segbench/data/msd"
        sp = json.loads((self.root / "splits.json").read_text())
        self.classes, self.split, self.split_seed = sp["classes"], sp["p622"], sp["seed"]
        self._fg: dict[str, np.ndarray] = {}

    def full(self, i: str) -> tuple[np.ndarray, np.ndarray]:
        return read_rgb(self.root / "images" / f"{i}.jpg"), read_mask(self.root / "masks" / f"{i}.png")

    def fg(self, i: str) -> tuple[np.ndarray, tuple]:
        if i not in self._fg:
            m = read_mask(self.root / "masks" / f"{i}.png")
            self._fg[i] = (np.argwhere((m > 0) & (m != 255)), m.shape)
        return self._fg[i]

    def crop(self, i: str, y0: int, x0: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        img, m = self.full(i)
        return img[y0:y0 + c, x0:x0 + c], m[y0:y0 + c, x0:x0 + c]

    def windows(self, i: str) -> list:
        return []


class DarkfieldData:
    """暗场标准输入（单通道，复制为 3 通道，固定 ÷255）+ 伪标签；评价时另读人工审核窗口标签。
    v1：trad_v2 原标签（4 类）；v2：标注工具同一区域算法重建（3 类，碎裂为不计，见 relabel_width.py）。"""
    offset = 2.0           # 背景 ≈ 0、噪声 σ≈1，只做很小的加性扰动

    def __init__(self, version: str = "darkfield"):
        self.root = PROJECT_ROOT / "output/segbench/data" / version
        self.img_root = PROJECT_ROOT / "output/segbench/data/darkfield"
        sp = json.loads((self.root / "splits.json").read_text())
        self.classes, self.split, self.split_seed = sp["classes"], sp["p622"], "splits_v1"
        self.bench = PROJECT_ROOT / "output/annotation/bench_v1"

    def _img(self, i):
        return np.load(self.img_root / "img" / f"{i}.npy", mmap_mode="r")

    def _lab(self, i):
        return np.load(self.root / "lab" / f"{i}.npy", mmap_mode="r")

    def full(self, i: str) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(np.asarray(self._img(i))[..., None], 3, 2), np.asarray(self._lab(i))

    def fg(self, i: str) -> tuple[np.ndarray, tuple]:
        return np.load(self.root / "fg" / f"{i}.npy"), self._lab(i).shape

    def crop(self, i: str, y0: int, x0: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        g = np.asarray(self._img(i)[y0:y0 + c, x0:x0 + c])
        return np.repeat(g[..., None], 3, 2), np.asarray(self._lab(i)[y0:y0 + c, x0:x0 + c])

    def windows(self, i: str) -> list:
        """该镜片已提交（done）的人工审核窗口：[(y0, x0, 标签图)]。"""
        out = []
        for st in (self.bench / "state").glob(f"*_{i}_*.json"):
            d = json.loads(st.read_text())
            lab = cv2.imread(str(self.bench / "labels" / f"{d['wid']}.png"), cv2.IMREAD_GRAYSCALE)
            if d.get("lens") == i and d.get("status") == "done" and lab is not None:
                lab = lab.copy()
                lab[(lab >= len(self.classes)) & (lab != 255)] = 255   # 模型不含的类别（如 v2 的碎裂）按不计
                out.append((d["y0"], d["x0"], lab))
        return out


DATASETS = {"msd": MSDData, "darkfield": DarkfieldData, "darkfield2": lambda: DarkfieldData("darkfield_v2")}


class CropSet(Dataset):
    def __init__(self, data, ids: list[str], crop: int, n: int, fg_frac: float = 0.5):
        self.data, self.ids, self.crop, self.n, self.fg_frac = data, ids, crop, n, fg_frac

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = random.Random()
        i = rng.choice(self.ids)
        fg, (h, w) = self.data.fg(i)
        c = self.crop
        if len(fg) and rng.random() < self.fg_frac:
            cy, cx = fg[rng.randrange(len(fg))]
            cy, cx = cy + rng.randint(-c // 4, c // 4), cx + rng.randint(-c // 4, c // 4)
        else:
            cy, cx = rng.randrange(h), rng.randrange(w)
        y0, x0 = int(np.clip(cy - c // 2, 0, h - c)), int(np.clip(cx - c // 2, 0, w - c))
        img, m = self.data.crop(i, y0, x0, c)
        if rng.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if rng.random() < 0.5:
            img, m = img[::-1], m[::-1]
        r = rng.randrange(4)
        img, m = np.rot90(img, r), np.rot90(m, r)
        o = self.data.offset
        img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-o, o), 0, 255).astype(np.uint8)
        return to_tensor(img), torch.from_numpy(m.astype(np.int64).copy())


def dice_loss(logits: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    keep = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * keep
    oh = F.one_hot(y.clamp(max=n - 1), n).permute(0, 3, 1, 2).float() * keep
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    d = 1 - (2 * inter + 1) / (den + 1)
    return d[present].mean() if present.any() else d.mean() * 0


def predict(model, img: np.ndarray, n_cls: int, device: str, tile: int = 0, overlap: int = 64) -> np.ndarray:
    """全图推理（tile=0）或滑窗推理（重叠区 logits 取平均），返回类别图。"""
    h, w = img.shape[:2]
    x = to_tensor(img)[None]
    if tile <= 0:
        H, W = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
        xx = F.pad(x, (0, W - w, 0, H - h), mode="reflect").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg = model(xx)
        return lg[0, :, :h, :w].float().argmax(0).cpu().numpy().astype(np.uint8)
    acc = torch.zeros(n_cls, h, w)
    cnt = torch.zeros(1, h, w)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) + ([h - tile] if h > tile and (h - tile) % step else [])
    xs = list(range(0, max(w - tile, 0) + 1, step)) + ([w - tile] if w > tile and (w - tile) % step else [])
    for y0 in ys:
        for x0 in xs:
            t = x[:, :, y0:y0 + tile, x0:x0 + tile].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                lg = model(t)[0].float().cpu()
            acc[:, y0:y0 + tile, x0:x0 + tile] += lg
            cnt[:, y0:y0 + tile, x0:x0 + tile] += 1
    return (acc / cnt).argmax(0).numpy().astype(np.uint8)


@torch.inference_mode()
def evaluate(model, data, ids: list[str], n_cls: int, names: list[str], device: str, max_n: int = 0,
             tile: int = 0) -> dict:
    """整图（或滑窗）推理；对全部图像累计混淆矩阵。若有人工审核窗口，另报告窗口内对人工标签的指标。"""
    model.eval()
    cm, cw, n_win = ConfusionMatrix(n_cls), ConfusionMatrix(n_cls), 0
    for i in (ids[:max_n] if max_n else ids):
        img, m = data.full(i)
        pred = predict(model, img, n_cls, device, tile)
        cm.update(m, pred)
        for y0, x0, lab in data.windows(i):
            cw.update(lab, pred[y0:y0 + lab.shape[0], x0:x0 + lab.shape[1]])
            n_win += 1
    model.train()
    r = cm.metrics(names)
    r["all_background_PA"] = all_background_pa(cm)
    if n_win:
        rw = cw.metrics(names)
        rw["all_background_PA"] = all_background_pa(cw)
        rw["n_windows"] = n_win
        r["human_windows"] = rw
    return r


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="msd", choices=list(DATASETS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr-mult", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-max", type=int, default=0, help="验证集最多评估张数（0 为全部；冒烟测试用）")
    ap.add_argument("--eval-tile", type=int, default=0, help="滑窗推理的窗口边长（0 为全图推理）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="b1")
    args = ap.parse_args()

    data = DATASETS[args.data]()
    names, split = data.classes, data.split
    n_cls = len(names)
    out = PROJECT_ROOT / "output/segbench/runs" / f"{args.data}_{args.model}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = M.build(args.model, n_cls, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(M.param_groups(model, args.lr, args.enc_lr_mult), weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    ds = CropSet(data, split["train"], args.crop, args.iters * args.bs)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    cfg = {**vars(args), "n_params": n_params, "classes": names, "git": git_commit(), "torch": torch.__version__,
           "split_seed": data.split_seed, "n_train": len(split["train"]), "started": time.strftime("%F %T")}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    log = open(out / "train.log", "a", encoding="utf-8")

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"{args.model}: {n_params / 1e6:.2f}M 参数，{args.iters} 次迭代 × bs {args.bs}")
    hist, best, t0, it = [], -1.0, time.time(), 0
    model.train()
    for x, y in dl:
        it += 1
        f = (1 - (it - 1) / args.iters) ** 0.9
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lg = model(x)
        lg = lg.float()
        loss = F.cross_entropy(lg, y, ignore_index=255) + dice_loss(lg, y, n_cls)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {it} 次迭代损失非有限值")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % 100 == 0:
            say(f"it {it} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s")
        if it % args.val_every == 0 or it == args.iters:
            r = evaluate(model, data, split["val"], n_cls, names, device, args.val_max, args.eval_tile)
            hist.append({"it": it, **{k: r[k] for k in ("PA", "MPA", "mIoU", "FWIoU", "mDice_defect")},
                         "per_class_IoU": {k: v["IoU"] for k, v in r["per_class"].items()}, "sec": time.time() - t0})
            (out / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1))
            hw = r.get("human_windows")
            if hw:
                hist[-1]["human_windows"] = {k: hw[k] for k in ("PA", "MPA", "mIoU", "FWIoU", "n_windows")}
                (out / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1))
            say(f"[val] it {it} PA {r['PA']:.4f} MPA {r['MPA']:.4f} mIoU {r['mIoU']:.4f} FWIoU {r['FWIoU']:.4f}"
                + (f" | 人工窗口 {hw['n_windows']} 个 PA {hw['PA']:.4f} mIoU {hw['mIoU']:.4f}" if hw else ""))
            if r["mIoU"] > best:
                best = r["mIoU"]
                torch.save({"model": model.state_dict(), "it": it, "val": r, "config": cfg}, out / "best.pt")
        if it >= args.iters:
            break
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    # 暗场测试集冻结到最终一次性评价，这里只在调参集上报告；MSD 在测试集上评价一次
    final = "test" if args.data == "msd" else "val"
    r = evaluate(model, data, split[final], n_cls, names, device, tile=args.eval_tile)
    r["final_split"] = final
    r.update({"best_it": ck["it"], "val_at_best": {k: ck["val"][k] for k in ("PA", "MPA", "mIoU", "FWIoU")},
              "train_sec": round(time.time() - t0), "n_params": n_params,
              "protocol": "msd_p622_fullres" if args.data == "msd" else f"{args.data}_pseudo_val"})
    (out / "test_metrics.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    say(f"[{final}] PA {r['PA']:.4f} MPA {r['MPA']:.4f} mIoU {r['mIoU']:.4f} FWIoU {r['FWIoU']:.4f} "
        + " ".join(f"{k}:{v['IoU']}" for k, v in r["per_class"].items()))


if __name__ == "__main__":
    main()
