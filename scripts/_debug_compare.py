"""
Debug script: compare "good" vs "bad" images to understand registration failures.
Prints statistics and saves radial profile + side-by-side visualisation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("/media/bm/Data/Data/Microlens_df/mingyue20260213")
CAL_DIR    = Path("/media/bm/Data/Data/Microlens_df/mingyue20260213_preprocessed/calibration")
OUT_DIR    = Path("/media/bm/Data/Dev/Microlens_DF/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAD_NAMES  = ["1l.png", "106r.png", "53r.png", "111l.png"]
GOOD_NAMES = ["100l.png", "104r.png", "48l.png", "91r.png"]

# ── load calibration data ──────────────────────────────────────────────
B_blur    = np.load(CAL_DIR / "B_blur.npy")
ring_mask = np.load(CAL_DIR / "ring_mask.npy").astype(bool)

print(f"B_blur shape: {B_blur.shape}, dtype: {B_blur.dtype}")
print(f"ring_mask shape: {ring_mask.shape}, dtype: {ring_mask.dtype}, "
      f"True pixels: {ring_mask.sum()}")
print()

# ── helper: radial profile ─────────────────────────────────────────────
def radial_profile(img, bin_width=10, max_r=2048):
    """Mean pixel value in concentric annuli centred on the image."""
    cy, cx = img.shape[0] / 2.0, img.shape[1] / 2.0
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    bins = np.arange(0, max_r + bin_width, bin_width)
    profile = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (R >= bins[i]) & (R < bins[i + 1])
        if mask.any():
            profile[i] = img[mask].mean()
        else:
            profile[i] = np.nan
    centres = (bins[:-1] + bins[1:]) / 2.0
    return centres, profile

# ── helper: circle mask ────────────────────────────────────────────────
def circle_mask(shape, radius):
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return np.sqrt((X - cx)**2 + (Y - cy)**2) <= radius

# ── analyse each image ─────────────────────────────────────────────────
def analyse(name, label):
    path = DATA_DIR / name
    img = np.array(Image.open(path)).astype(np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)  # to grayscale if needed

    h, w = img.shape
    # centre 40 % crop
    ch, cw = int(h * 0.3), int(w * 0.3)
    centre_region = img[ch:h - ch, cw:w - cw]
    centre_mean = centre_region.mean()

    # ring mask stats (crop ring_mask to image size if needed)
    rm = ring_mask[:h, :w] if (ring_mask.shape[0] >= h and ring_mask.shape[1] >= w) else ring_mask
    ring_mean = img[rm].mean() if rm.any() else float("nan")

    # outside ring but inside r=1900
    big_circle = circle_mask(img.shape, 1900)
    outside_ring = big_circle & (~rm)
    outside_mean = img[outside_ring].mean() if outside_ring.any() else float("nan")

    ratio = ring_mean / centre_mean if centre_mean != 0 else float("nan")

    print(f"[{label:4s}] {name:12s}  "
          f"mean={img.mean():8.2f}  std={img.std():8.2f}  "
          f"min={img.min():8.2f}  max={img.max():8.2f}  "
          f"center40%={centre_mean:8.2f}  "
          f"ring={ring_mean:8.2f}  outside={outside_mean:8.2f}  "
          f"ring/center={ratio:.4f}")
    return img

print("=" * 130)
print(f"{'':>6} {'name':12s}  {'mean':>8s}  {'std':>8s}  "
      f"{'min':>8s}  {'max':>8s}  {'center40%':>10s}  "
      f"{'ring':>8s}  {'outside':>8s}  {'ring/ctr':>10s}")
print("=" * 130)

bad_imgs  = {n: analyse(n, "BAD")  for n in BAD_NAMES}
print("-" * 130)
good_imgs = {n: analyse(n, "GOOD") for n in GOOD_NAMES}
print("=" * 130)

# ── radial profile plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
# pick first 2 of each
for name in BAD_NAMES[:2]:
    r, p = radial_profile(bad_imgs[name])
    ax.plot(r, p, label=f"BAD  {name}", linestyle="--")
for name in GOOD_NAMES[:2]:
    r, p = radial_profile(good_imgs[name])
    ax.plot(r, p, label=f"GOOD {name}")
ax.set_xlabel("Radius (px)")
ax.set_ylabel("Mean intensity")
ax.set_title("Radial intensity profile: good vs bad images")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "debug_radial_profiles.png", dpi=150)
plt.close(fig)
print(f"\nSaved radial profile plot -> {OUT_DIR / 'debug_radial_profiles.png'}")

# ── side-by-side visualisation ─────────────────────────────────────────
def norm255(img):
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-9:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for col, name in enumerate(BAD_NAMES):
    axes[0, col].imshow(norm255(bad_imgs[name]), cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"BAD: {name}", fontsize=10)
    axes[0, col].axis("off")
for col, name in enumerate(GOOD_NAMES):
    axes[1, col].imshow(norm255(good_imgs[name]), cmap="gray", vmin=0, vmax=255)
    axes[1, col].set_title(f"GOOD: {name}", fontsize=10)
    axes[1, col].axis("off")
fig.suptitle("Top row: BAD (registration fails)    Bottom row: GOOD", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "debug_good_vs_bad.png", dpi=150)
plt.close(fig)
print(f"Saved side-by-side   plot -> {OUT_DIR / 'debug_good_vs_bad.png'}")
