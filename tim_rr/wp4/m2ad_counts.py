"""WP4: specimen counts per M2AD category from meta_unsupervised.json (read-only)."""
import collections
import json
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
first = meta["test"]["Motor"][0]
print("example record:", first)
total = collections.Counter()
for cls in meta["train"]:
    objs = collections.defaultdict(lambda: [0, 0])  # images, anomalous images
    for split in ("train", "test"):
        for item in meta[split][cls]:
            parts = item["img_path"].replace("\\", "/").split("/")
            o = objs[(parts[1], parts[2])]
            o[0] += 1
            o[1] += int(item.get("image_anomaly", 0))
    good = [k for k in objs if k[0] == "Good"]
    ng = [k for k in objs if k[0] == "NG"]
    ng_anom = [k for k in ng if objs[k][1] > 0]
    per_obj = collections.Counter(v[0] for v in objs.values())
    frac = sum(objs[k][1] for k in ng) / max(1, sum(objs[k][0] for k in ng))
    total.update(good=len(good), ng=len(ng), ng_anom=len(ng_anom))
    print(f"{cls:7s} Good {len(good):4d}  NG {len(ng):4d}  NG with masked anomaly {len(ng_anom):4d}"
          f"  images/specimen {dict(per_obj)}  anomalous-image share within NG {frac:.2f}")
print("TOTAL", dict(total))

# Masked-image structure within NG specimens: per illumination and per view.
by_illum = collections.Counter()
by_view = collections.Counter()
n_ng = collections.Counter()
views_with_defect = collections.Counter()
for split in ("train", "test"):
    for cls in meta[split]:
        for item in meta[split][cls]:
            if item["object_anomaly"]:
                n_ng[(item["illumination"])] += 1
                if item["image_anomaly"]:
                    by_illum[item["illumination"]] += 1
                    by_view[item["view"]] += 1
print("anomalous share of NG images by illumination:",
      {k: round(by_illum[k] / n_ng[k], 2) for k in sorted(n_ng)})
print("anomalous NG images by view:", dict(sorted(by_view.items())))
print("mask_path set on anomalous images:",
      sum(bool(i["mask_path"]) for s in ("train", "test") for c in meta[s] for i in meta[s][c] if i["image_anomaly"]),
      "of", sum(by_illum.values()))
print("detectable field values:", collections.Counter(i["detectable"] for s in ("train", "test") for c in meta[s] for i in meta[s][c]).most_common(5))
