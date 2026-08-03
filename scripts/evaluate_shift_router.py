from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (41, 53, 67, 79, 97)
ROOT = Path("experiments")


def describe(values: list[float | None]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_endpoint(directory: str) -> dict[str, object]:
    rows = []
    for seed in SEEDS:
        folder = ROOT / f"mixed_carinthia_seed{seed}" / directory
        training = read_rows(folder / "detector_val_records.csv")
        shift = read_rows(folder / "shift_test_records.csv")
        models = {
            method: fit_model(training, method, seed, FEATURE_VARIANTS["full"])
            for method in METHODS
        }
        actions = build_actions(
            shift,
            models,
            POOL_VARIANTS["all"],
            deletion_target=0.15,
            invention_target=0.15,
        )
        primary = json.loads(
            (folder / "sequential_router.json").read_text(encoding="utf-8")
        )
        result = evaluate_policy(
            actions["restored_pool"],
            actions["raw"],
            float(primary["pool_threshold"]),
            float(primary["raw_threshold"]),
        )
        rows.append(
            {
                "seed": seed,
                "nominal_certificate_pass": int(primary["certificate"]["passes"]),
                "shift_coverage": result["coverage"],
                "shift_deletion_risk": result["deletion_risk"],
                "shift_invention_risk": result["invention_risk"],
                "pool_accepted": result["pool_accepted"],
                "raw_accepted": result["raw_accepted"],
                "reviewed": result["reviewed"],
            }
        )
    return {
        "directory": directory,
        "note": (
            "Descriptive application of nominally fitted gates to the reserved "
            "rare-class split. No certificate is claimed under this shift."
        ),
        "coverage": describe([row["shift_coverage"] for row in rows]),
        "deletion_risk": describe([row["shift_deletion_risk"] for row in rows]),
        "invention_risk": describe([row["shift_invention_risk"] for row in rows]),
        "rows": rows,
    }


def main() -> None:
    from scripts.analyze_router_pool import read_rows
    from scripts.evaluate_sequential_router import evaluate_policy
    from scripts.run_supporting_ablations import (
        FEATURE_VARIANTS,
        METHODS,
        POOL_VARIANTS,
        build_actions,
        fit_model,
    )

    output = ROOT / "shift_router_summary"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_floor025": run_endpoint("incident_floor025"),
        "secondary_floor050": run_endpoint("incident_floor050"),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    flat = []
    for endpoint, summary in payload.items():
        for row in summary["rows"]:
            flat.append({"endpoint": endpoint, **row})
    with (output / "seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
