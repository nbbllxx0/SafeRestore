"""WP4: verify fetched M2AD files against the metadata and the fetch manifests.

For each category: every metadata image at the chosen illuminations and every
mask referenced by an anomalous image must exist on disk, and every manifest
member must exist with its recorded size (CRC-32 was checked at fetch time).
"""
import json
import sys
from pathlib import Path

root, meta_path = Path(sys.argv[1]), Path(sys.argv[2])
cats = sys.argv[3:]
illums = ("07", "10")
meta = json.load(open(meta_path, encoding="utf-8"))
ok = True
for cls in cats:
    items = [i for s in ("train", "test") for i in meta[s][cls]]
    imgs = {i["img_path"] for i in items if i["illumination"] in illums}
    masks = {i["mask_path"] for i in items if i["image_anomaly"]}
    miss_i = sorted(p for p in imgs if not (root / p).exists())
    miss_m = sorted(p for p in masks if not (root / p).exists())
    man_path = root / f"{cls}_subset_manifest.json"
    bad_size = []
    n_man = 0
    if man_path.exists():
        man = json.load(open(man_path, encoding="utf-8"))["members"]
        n_man = len(man)
        bad_size = [p for p, m in man.items() if not (root / p).exists() or (root / p).stat().st_size != m["size"]]
    parts = list(root.joinpath(cls).rglob("*.part"))
    status = "OK" if not (miss_i or miss_m or bad_size or parts) else "FAIL"
    ok &= status == "OK"
    print(f"{cls:7s} {status}: images {len(imgs)-len(miss_i)}/{len(imgs)}  masks {len(masks)-len(miss_m)}/{len(masks)}"
          f"  manifest {n_man} (size mismatch {len(bad_size)})  .part {len(parts)}")
print("ALL OK" if ok else "SOME FAILED")
