"""Build the frozen, powered KolektorSDD2 split for the TIM upgrade.

Design constraints:
- The official train/test boundary is preserved: every official test image is
  assigned to the held-out test role and nothing else.
- The official training images are subdivided by a deterministic
  identifier-hash rule into train / detector_val / calibration roles with
  positivity-stratified fractions so that the calibration role carries enough
  defect-positive images for the 0.15 dual-risk targets to be attainable
  (>= 30 accepted certification positives is the planning requirement; the
  budget below targets ~100 calibration positives before gating).
- One image is the sampling unit; KolektorSDD2 metadata expose no coarser
  physical grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ksdd2_data import inventory_sha256, load_ksdd2

POSITIVE_FRACTIONS = {"train": 0.35, "detector_val": 0.22, "calibration": 0.43}
CLEAN_FRACTIONS = {"train": 0.62, "detector_val": 0.15, "calibration": 0.23}


def stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(sample_id: str, positive: bool, seed: int) -> str:
    fractions = POSITIVE_FRACTIONS if positive else CLEAN_FRACTIONS
    value = stable_fraction(sample_id, seed)
    accumulated = 0.0
    for split in ("train", "detector_val", "calibration"):
        accumulated += fractions[split]
        if value < accumulated:
            return split
    return "calibration"


def build_split(root: Path, seed: int) -> dict:
    items = load_ksdd2(root, measure_positivity=True)
    records = []
    for item in items:
        official_part = item.sample_id.split("/", 1)[0]
        if official_part == "test":
            split = "test"
        else:
            split = assign(item.sample_id, item.has_defect, seed)
        records.append(
            {
                "sample_id": item.sample_id,
                "split": split,
                "has_defect": int(item.has_defect),
            }
        )
    counts = Counter(record["split"] for record in records)
    positives = Counter(
        record["split"] for record in records if record["has_defect"]
    )
    return {
        "dataset": "kolektor_sdd2",
        "version": 1,
        "seed": seed,
        "policy": (
            "official test images all map to the test role; official training "
            "images are hash-subdivided with positivity-stratified fractions "
            "so the calibration role is powered for the 0.15 dual-risk targets"
        ),
        "counts": dict(sorted(counts.items())),
        "positive_counts": dict(sorted(positives.items())),
        "inventory_sha256": inventory_sha256(items),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    payload = build_split(args.root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("counts", "positive_counts", "inventory_sha256")}, indent=2))


if __name__ == "__main__":
    main()
