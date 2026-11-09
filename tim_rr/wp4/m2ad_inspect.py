"""WP4: structural checks on one downloaded M2AD category (read-only on the data).

Checks
  1. mask coverage: every anomalous image in the metadata has a mask file on disk
  2. image geometry: size, channels, bit depth
  3. alignment across the 10 illuminations of one view (phase correlation on
     gradient magnitude, relative to I10)
  4. per-illumination exposure (mean, dark and saturated pixel shares) and
     defect contrast (|mean inside mask - mean in a 15-px ring| / ring std)
  5. contact sheet: one NG specimen, one view, all illuminations, mask contour

Usage: python -I -B m2ad_inspect.py <m2ad_root> <meta_json> <category> <out_dir>
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT, META, CLS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 1  # distributed images are already 1024x1024


def load(path: Path, gray: bool = True) -> np.ndarray:
    im = Image.open(path)
    if gray:
        im = im.convert("L")
    w, h = im.size
    return np.asarray(im.resize((w // SCALE, h // SCALE), Image.BILINEAR), dtype=np.float64)


def grad(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    f = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    r = np.fft.ifft2(f / (np.abs(f) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    dy = dy - r.shape[0] if dy > r.shape[0] // 2 else dy
    dx = dx - r.shape[1] if dx > r.shape[1] // 2 else dx
    return dy * SCALE, dx * SCALE  # full-resolution pixels


meta = json.load(open(META, encoding="utf-8"))
items = [i for s in ("train", "test") for i in meta[s].get(CLS, [])]
anom = [i for i in items if i["image_anomaly"]]

# 1. mask coverage
missing = [i["mask_path"] for i in anom if not (ROOT / i["mask_path"]).exists()]
imgs_missing = [i["img_path"] for i in items if not (ROOT / i["img_path"]).exists()]
print(f"images in metadata {len(items)}, missing on disk {len(imgs_missing)}")
print(f"anomalous images {len(anom)}, mask files missing {len(missing)}")
print("distinct mask paths:", len({i['mask_path'] for i in anom}),
      "(one per specimen-view if shared across illuminations)")

# 2. geometry
probe = Image.open(ROOT / items[0]["img_path"])
print("defect area px (>=128) over sampled masks: computed below")
print("image size", probe.size, "mode", probe.mode)
mprobe = Image.open(ROOT / anom[0]["mask_path"])
print("mask size", mprobe.size, "mode", mprobe.mode, "values", np.unique(np.asarray(mprobe))[:6])

# Group by specimen-view.
groups: dict[tuple[str, str, str], dict[str, dict]] = collections.defaultdict(dict)
for i in items:
    groups[(i["img_path"].split("/")[1], i["object_name"], i["view"])][i["illumination"]] = i
rng = np.random.default_rng(0)
ng_keys = sorted(k for k in groups if k[0] == "NG" and any(v["image_anomaly"] for v in groups[k].values()))
good_keys = sorted(k for k in groups if k[0] == "Good")
sample = [ng_keys[j] for j in rng.choice(len(ng_keys), size=min(24, len(ng_keys)), replace=False)]
sample_good = [good_keys[j] for j in rng.choice(len(good_keys), size=min(12, len(good_keys)), replace=False)]
illums = sorted(groups[sample[0]])

# 3-4. alignment, exposure, contrast
shifts = collections.defaultdict(list)
expo = collections.defaultdict(lambda: collections.defaultdict(list))
contrast = collections.defaultdict(list)
for key in sample + sample_good:
    ref = load(ROOT / groups[key]["10"]["img_path"])
    fg = ref > 20  # object foreground from the brightest regime (background is black)
    mask = None
    if key in sample:
        rec = next(v for v in groups[key].values() if v["image_anomaly"])
        mask = load(ROOT / rec["mask_path"]) > 127
        from scipy import ndimage
        ring = ndimage.binary_dilation(mask, iterations=15) & ~mask & fg
    for il in illums:
        a = load(ROOT / groups[key][il]["img_path"])
        if il != "10":
            shifts[il].append(np.hypot(*shift(grad(ref), grad(a))))
        expo[il]["mean"].append(a[fg].mean())
        expo[il]["dark"].append((a[fg] < 10).mean())
        expo[il]["sat"].append((a[fg] > 245).mean())
        expo[il]["fg"].append(fg.mean())
        if mask is not None and mask.any() and ring.any():
            contrast[il].append(abs(a[mask].mean() - a[ring].mean()) / (a[ring].std() + 1e-6))

print("\nmax/median shift vs I10 (full-res px):",
      {il: (round(max(v), 1), round(float(np.median(v)), 1)) for il, v in sorted(shifts.items())})
print("\nillum  mean  dark%  sat%  defect-contrast(median over NG views)")
for il in illums:
    e = expo[il]
    c = contrast[il]
    print(f"I{il}  {np.mean(e['mean']):6.1f}  {100*np.mean(e['dark']):5.1f}  {100*np.mean(e['sat']):5.1f}  "
          f"{np.median(c) if c else float('nan'):.2f}")

# 5. contact sheet
key = sample[0]
rec = next(v for v in groups[key].values() if v["image_anomaly"])
m = np.asarray(Image.open(ROOT / rec["mask_path"]).convert("L")) > 127
ys, xs = np.nonzero(m)
cy, cx = int(ys.mean()), int(xs.mean())
half = 600
tiles = []
for il in illums:
    im = Image.open(ROOT / groups[key][il]["img_path"]).convert("RGB")
    a = np.asarray(im)
    y0, x0 = max(0, cy - half), max(0, cx - half)
    crop = a[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    mc = m[y0:y0 + 2 * half, x0:x0 + 2 * half]
    from scipy import ndimage
    edge = mc & ~ndimage.binary_erosion(mc, iterations=3)
    crop[edge] = (0, 160, 255)
    tiles.append(Image.fromarray(crop).resize((300, 300)))
sheet = Image.new("RGB", (300 * 5, 300 * 2), "white")
for j, t in enumerate(tiles):
    sheet.paste(t, ((j % 5) * 300, (j // 5) * 300))
name = f"contact_{CLS}_{key[1]}_view{key[2]}.png"
sheet.save(OUT / name)
print("\ncontact sheet:", OUT / name, "illumination order:", illums)
