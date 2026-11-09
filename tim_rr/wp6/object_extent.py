"""WP6: object bounding-box extent per category on development specimens (I10), to size crops."""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage
root = sys.argv[1]
meta = json.load(open(sys.argv[2], encoding="utf-8"))
roles = json.load(open("splits/m2ad_rr_v1.json", encoding="utf-8"))["roles"]
rng = np.random.default_rng(0)
for cls in meta["train"]:
    items = [i for s in ("train", "test") for i in meta[s][cls]
             if i["illumination"] == "10"
             and roles[f"{cls}/{i['img_path'].split('/')[1]}/{i['object_name']}"] in ("train", "detector_val")]
    sides, cover = [], []
    for it in [items[j] for j in rng.choice(len(items), 40, replace=False)]:
        g = np.asarray(Image.open(f"{root}/{it['img_path']}").convert("L"))
        fg = ndimage.binary_opening(g > 20, iterations=2)
        lab, n = ndimage.label(fg)
        if n == 0:
            continue
        big = lab == (1 + np.argmax(ndimage.sum(fg, lab, range(1, n + 1))))
        ys, xs = np.nonzero(big)
        sides.append(max(ys.max() - ys.min(), xs.max() - xs.min()))
        if it["image_anomaly"]:
            m = np.asarray(Image.open(f"{root}/{it['mask_path']}")) >= 128
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            y0, x0 = np.clip(cy - 256, 0, 512), np.clip(cx - 256, 0, 512)
            cover.append(m[y0:y0 + 512, x0:x0 + 512].sum() / max(1, m.sum()))
    print(f"{cls:7s} object side px p50 {np.median(sides):.0f} p95 {np.percentile(sides,95):.0f} max {max(sides)}"
          f"  mask share inside 512 crop: min {min(cover) if cover else float('nan'):.2f} mean {np.mean(cover) if cover else float('nan'):.3f}")
