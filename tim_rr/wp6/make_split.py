"""WP6: frozen specimen-level role split for the M2AD confirmatory study.

Unit: one specimen (all its views and illuminations stay in one role).
Within each (category, Good/NG) stratum, specimens are ranked by
sha256("m2ad-rr-v1:{category}/{object}") and assigned by rank:
  train 30% | detector_val 15% | tuning 15% | holdout 40%
(cut points rounded half-to-even: 15 | 7 | 8 | 20 for a 50-specimen stratum).

The pilot and every design step may read train/detector_val only; tuning is
read only by the frozen gate-ordering step; holdout is opened once.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="utf-8"))
out = Path(sys.argv[2])
strata = defaultdict(set)
for split in ("train", "test"):
    for cls, items in meta[split].items():
        for it in items:
            kind = it["img_path"].split("/")[1]
            strata[(cls, kind)].add(it["object_name"])

FRACTIONS = (("train", 0.30), ("detector_val", 0.15), ("tuning", 0.15))
roles = {}
for (cls, kind), objs in sorted(strata.items()):
    ranked = sorted(objs, key=lambda o: hashlib.sha256(f"m2ad-rr-v1:{cls}/{o}".encode()).hexdigest())
    n = len(ranked)
    cuts, acc = [], 0.0
    for _, f in FRACTIONS:
        acc += f
        cuts.append(round(acc * n))
    for rank, obj in enumerate(ranked):
        role = next((r for (r, _), c in zip(FRACTIONS, cuts) if rank < c), "holdout")
        roles[f"{cls}/{kind}/{obj}"] = role

counts = defaultdict(lambda: defaultdict(int))
for key, role in roles.items():
    cls, kind, _ = key.split("/")
    counts[role][kind] += 1
doc = {"version": "m2ad-rr-v1", "unit": "specimen", "rule": __doc__.strip().splitlines()[3:8],
       "counts": {r: dict(c) for r, c in counts.items()}, "roles": roles}
out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
print(json.dumps(doc["counts"]), hashlib.sha256(out.read_bytes()).hexdigest())
