"""WP4: share of anomalous images flagged not detectable, by illumination and category."""
import collections
import json
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
tot, nd = collections.Counter(), collections.Counter()
cls_tot, cls_nd = collections.Counter(), collections.Counter()
for split in ("train", "test"):
    for cls in meta[split]:
        for it in meta[split][cls]:
            if it["image_anomaly"]:
                tot[it["illumination"]] += 1
                cls_tot[cls] += 1
                if it["detectable"] is False:
                    nd[it["illumination"]] += 1
                    cls_nd[cls] += 1
print("not-detectable share by illumination:", {k: round(nd[k] / tot[k], 3) for k in sorted(tot)})
print("not-detectable share by category:", {k: round(cls_nd[k] / cls_tot[k], 3) for k in sorted(cls_tot)})
