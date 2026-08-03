from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from scripts.analyze_router_pool import (
    action_arrays,
    choose_threshold,
    evaluate,
    fit_models,
    read_rows,
)


def stable_tuning_mask(sample_ids: list[str], seed: int | None = None) -> np.ndarray:
    """Backward-compatible wrapper; confirmatory analysis ignores ``seed``."""

    del seed  # seed must not affect the confirmatory tune/certify partition
    values = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"v3tunecert:{sample_id}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") % 10 < 3)
    mask = np.asarray(values, dtype=bool)
    if mask.all() or (~mask).all():
        raise RuntimeError("degenerate calibration subdivision")
    return mask


def subset(action: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {key: np.asarray(value)[mask] for key, value in action.items()}


def binomial_ucb(errors: int, count: int, delta: float) -> float:
    if count <= 0 or errors >= count:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, count - errors))


def certify(
    action: dict[str, Any],
    threshold: float,
    *,
    delta: float,
) -> dict[str, Any]:
    accepted = action["score"] <= threshold
    count = int(accepted.sum())
    deletion_errors = int(action["deletion"][accepted].sum())
    invention_errors = int(action["invention"][accepted].sum())
    return {
        **evaluate(action, threshold),
        "deletion_errors": deletion_errors,
        "invention_errors": invention_errors,
        "deletion_ucb": binomial_ucb(deletion_errors, count, delta / 2),
        "invention_ucb": binomial_ucb(invention_errors, count, delta / 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.85)
    parser.add_argument("--min-accept", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = read_rows(args.experiment / "detector_val_records.csv")
    calibration_rows = read_rows(args.experiment / "calibration_records.csv")
    test_rows = read_rows(args.experiment / "test_records.csv")
    models = fit_models(training, args.seed)
    raw_calibration = [
        row for row in calibration_rows if row["method"] == "raw_nearest"
    ]
    sample_ids = [row["sample_id"] for row in raw_calibration]
    tuning_mask = stable_tuning_mask(sample_ids, args.seed)

    operating_points: list[dict[str, Any]] = []
    for deletion_target in (0.30, 0.35, 0.40, 0.45, 0.50):
        for invention_target in (0.05, 0.10, 0.15):
            calibration = action_arrays(
                calibration_rows, models, deletion_target, invention_target
            )
            test = action_arrays(test_rows, models, deletion_target, invention_target)
            action = calibration["restored_pool"]
            tuning = subset(action, tuning_mask)
            certification = subset(action, ~tuning_mask)
            threshold = choose_threshold(
                tuning,
                deletion_target=deletion_target,
                invention_target=invention_target,
                margin=args.margin,
                min_accept=args.min_accept,
            )
            certificate = certify(certification, threshold, delta=args.delta)
            certificate["passes"] = (
                certificate["deletion_ucb"] <= deletion_target
                and certificate["invention_ucb"] <= invention_target
            )
            operating_points.append(
                {
                    "deletion_target": deletion_target,
                    "invention_target": invention_target,
                    "threshold": threshold,
                    "tuning": evaluate(tuning, threshold),
                    "certificate": certificate,
                    "test": evaluate(test["restored_pool"], threshold),
                }
            )

    result = {
        "experiment": str(args.experiment.resolve()),
        "method": "fixed restoration-pool selector plus independent tune/certify split",
        "delta": args.delta,
        "tuning_n": int(tuning_mask.sum()),
        "certification_n": int((~tuning_mask).sum()),
        "test_n": len(test_rows) // 6,
        "operating_points": operating_points,
    }
    output = args.output or args.experiment / "router_certification.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

