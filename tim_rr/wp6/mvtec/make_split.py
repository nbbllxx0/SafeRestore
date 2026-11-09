"""Freeze the MVTec AD role split (splits/mvtec_rr_v1.json) and record its SHA-256.

The rule is mvtec_data.split_records; it reads only identifiers and the
defect-free / defective status implied by the folder layout, never images.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mvtec_data  # noqa: E402

# Dataset root: MVTEC_AD_ROOT, else <project parent>/data/raw/mvtec_ad/extracted.
ROOT = Path(os.environ.get("MVTEC_AD_ROOT", Path(__file__).resolve().parents[3].parent / "data" / "raw" / "mvtec_ad" / "extracted"))


def main() -> None:
    out = HERE / "splits" / "mvtec_rr_v1.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the split is frozen")
    records = mvtec_data.split_records(mvtec_data.inventory(ROOT))
    counts = collections.Counter((r["split"], r["has_defect"]) for r in records)
    payload = {
        "dataset": "mvtec_ad",
        "version": "mvtec-rr-v1",
        "rule": mvtec_data.split_records.__doc__.strip(),
        "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "counts": {f"{s}/{'defective' if d else 'defect_free'}": n for (s, d), n in sorted(counts.items())},
        "records": records,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "mvtec_rr_v1.sha256.txt").write_text(f"{digest}  mvtec_rr_v1.json\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=1))
    print("sha256", digest)


if __name__ == "__main__":
    main()
