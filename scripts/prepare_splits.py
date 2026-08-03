from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from saferestore.datasets import load_carinthia_s, load_kolektor, load_official_splits


def stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def carinthia_split(root: Path, seed: int = 20260723, *, version: int = 1) -> dict:
    items = load_carinthia_s(root)
    common = {3, 4, 6}
    records = []
    for item in items:
        if item.label not in common:
            split = "shift_test"
        else:
            value = stable_fraction(item.sample_id, seed)
            if value < 0.70:
                split = "train"
            elif value < 0.80:
                split = "detector_val"
            elif value < 0.90:
                split = "calibration"
            else:
                split = "test"
        records.append({"sample_id": item.sample_id, "label": item.label, "split": split})
    counts = Counter(record["split"] for record in records)
    by_class = defaultdict(Counter)
    for record in records:
        by_class[str(record["label"])][record["split"]] += 1
    return {
        "dataset": "carinthia_s",
        "version": version,
        "seed": seed,
        "policy": (
            "classes 1/2/5 reserved for morphology-shift test; "
            "stable hash split within classes 3/4/6"
        ),
        "counts": dict(sorted(counts.items())),
        "class_split_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(by_class.items())
        },
        "records": records,
    }


def kolektor_split(root: Path, split_file: Path, fold: int = 0, seed: int = 20260723) -> dict:
    items = load_kolektor(root)
    train_folds, test_folds, _ = load_official_splits(split_file)
    official_train = set(train_folds[fold])
    official_test = set(test_folds[fold])
    part_assignment = {}
    for part in sorted(official_train):
        value = stable_fraction(part, seed)
        if value < 0.72:
            part_assignment[part] = "train"
        elif value < 0.86:
            part_assignment[part] = "detector_val"
        else:
            part_assignment[part] = "calibration"
    for part in official_test:
        part_assignment[part] = "test"
    records = [
        {"sample_id": item.sample_id, "part_id": item.part_id, "split": part_assignment[item.part_id]}
        for item in items
    ]
    counts = Counter(record["split"] for record in records)
    part_counts = Counter(part_assignment.values())
    return {
        "dataset": "kolektor_sdd",
        "version": 1,
        "seed": seed,
        "official_fold": fold,
        "policy": "official item-level fold 0; stable part-level subdivision of official training items",
        "counts": dict(sorted(counts.items())),
        "part_counts": dict(sorted(part_counts.items())),
        "records": records,
    }


def kolektor_powered_split(
    root: Path,
    *,
    seed: int = 20260725,
    min_calibration_positives: int = 18,
) -> dict:
    """Item-level split engineered so calibration has enough positive mass.

    This is not the official Kolektor fold. Mask positivity is measured from
    on-disk labels.
    """

    from PIL import Image
    import numpy as np

    items = load_kolektor(root)
    part_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pos": 0})
    item_positive: dict[str, bool] = {}
    for item in items:
        mask = np.asarray(Image.open(item.mask_path).convert("L"))
        has = bool((mask > 0).any())
        item_positive[item.sample_id] = has
        part_stats[item.part_id]["n"] += 1
        part_stats[item.part_id]["pos"] += int(has)

    # Prefer parts with defects into calibration until the positive budget is met.
    defect_parts = sorted(
        (part for part, stats in part_stats.items() if stats["pos"] > 0),
        key=lambda part: (stable_fraction(f"powered|{part}", seed), part),
    )
    clean_parts = sorted(
        (part for part, stats in part_stats.items() if stats["pos"] == 0),
        key=lambda part: (stable_fraction(f"powered|{part}", seed), part),
    )

    part_assignment: dict[str, str] = {}
    calib_pos = 0
    for part in defect_parts:
        if calib_pos < min_calibration_positives:
            part_assignment[part] = "calibration"
            calib_pos += part_stats[part]["pos"]
            continue
        value = stable_fraction(f"assign|{part}", seed)
        if value < 0.45:
            part_assignment[part] = "train"
        elif value < 0.65:
            part_assignment[part] = "detector_val"
        else:
            part_assignment[part] = "test"
    for part in clean_parts:
        value = stable_fraction(f"clean|{part}", seed)
        if value < 0.55:
            part_assignment[part] = "train"
        elif value < 0.70:
            part_assignment[part] = "detector_val"
        elif value < 0.82:
            part_assignment[part] = "calibration"
        else:
            part_assignment[part] = "test"

    records = []
    positives_by_split: Counter[str] = Counter()
    for item in items:
        split = part_assignment[item.part_id]
        records.append(
            {
                "sample_id": item.sample_id,
                "part_id": item.part_id,
                "split": split,
                "has_defect": int(item_positive[item.sample_id]),
            }
        )
        positives_by_split[split] += int(item_positive[item.sample_id])
    if positives_by_split["calibration"] < min_calibration_positives:
        raise RuntimeError(
            "powered Kolektor split failed positive budget: "
            f"{positives_by_split['calibration']} < {min_calibration_positives}"
        )
    counts = Counter(record["split"] for record in records)
    part_counts = Counter(part_assignment.values())
    return {
        "dataset": "kolektor_sdd",
        "version": 4,
        "seed": seed,
        "official_fold": None,
        "policy": (
            "powered item-level split for second-domain authorization; "
            f"calibration targets >= {min_calibration_positives} defect-positive images; "
            "not the official Kolektor protocol fold"
        ),
        "counts": dict(sorted(counts.items())),
        "positive_counts": dict(sorted(positives_by_split.items())),
        "part_counts": dict(sorted(part_counts.items())),
        "min_calibration_positives": min_calibration_positives,
        "records": records,
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create frozen public-data split manifests")
    parser.add_argument("--carinthia-root", type=Path, required=True)
    parser.add_argument("--kolektor-root", type=Path, required=True)
    parser.add_argument("--kolektor-split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=project / "splits")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--also-v3", action="store_true", help="Also write carinthia_s_v3.json with seed 20260724")
    parser.add_argument("--also-v4", action="store_true", help="Also write confirmatory v4 Carinthia + powered Kolektor splits")
    args = parser.parse_args()
    carinthia = carinthia_split(args.carinthia_root, seed=args.seed, version=1)
    kolektor = kolektor_split(
        args.kolektor_root, args.kolektor_split_file, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "carinthia_s_v1.json": carinthia,
        "kolektor_sdd_fold0_v1.json": kolektor,
    }
    if args.also_v3:
        payloads["carinthia_s_v3.json"] = carinthia_split(
            args.carinthia_root, seed=20260724, version=3
        )
    if args.also_v4:
        payloads["carinthia_s_v4.json"] = carinthia_split(
            args.carinthia_root, seed=20260725, version=4
        )
        payloads["kolektor_sdd_powered_v4.json"] = kolektor_powered_split(
            args.kolektor_root, seed=20260725, min_calibration_positives=18
        )
    for name, payload in payloads.items():
        path = args.output / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(name, payload["counts"], payload.get("positive_counts"))


if __name__ == "__main__":
    main()
