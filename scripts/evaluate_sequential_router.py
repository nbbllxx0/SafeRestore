from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_router_pool import (
    action_arrays,
    choose_threshold,
    fit_models,
    read_rows,
)
from scripts.certify_router import binomial_ucb, stable_tuning_mask, subset


def evaluate_policy(
    pool: dict[str, Any],
    raw: dict[str, Any],
    pool_threshold: float,
    raw_threshold: float,
) -> dict[str, Any]:
    pool_accept = pool["score"] <= pool_threshold
    raw_accept = (~pool_accept) & (raw["score"] <= raw_threshold)
    accepted = pool_accept | raw_accept
    deletion = np.where(pool_accept, pool["deletion"], raw["deletion"])
    invention = np.where(pool_accept, pool["invention"], raw["invention"])
    selected_method = np.where(pool_accept, pool["method"], raw["method"])
    count = int(accepted.sum())
    methods, counts = np.unique(selected_method[accepted], return_counts=True)
    return {
        "n": int(len(accepted)),
        "accepted": count,
        "coverage": count / len(accepted),
        "pool_accepted": int(pool_accept.sum()),
        "raw_accepted": int(raw_accept.sum()),
        "reviewed": int((~accepted).sum()),
        "deletion_errors": int(deletion[accepted].sum()),
        "invention_errors": int(invention[accepted].sum()),
        "deletion_risk": float(deletion[accepted].mean()) if count else None,
        "invention_risk": float(invention[accepted].mean()) if count else None,
        "method_counts": {
            str(method): int(value) for method, value in zip(methods, counts)
        },
    }


def choose_raw_fallback_threshold(
    pool: dict[str, Any],
    raw: dict[str, Any],
    pool_threshold: float,
    *,
    deletion_target: float,
    invention_target: float,
    margin: float,
) -> float:
    pool_accept = pool["score"] <= pool_threshold
    remaining_scores = np.asarray(raw["score"])[~pool_accept]
    candidates = np.concatenate(
        (np.asarray([float("-inf")]), np.unique(remaining_scores))
    )
    best = float("-inf")
    best_coverage = -1.0
    for threshold in candidates:
        result = evaluate_policy(pool, raw, pool_threshold, float(threshold))
        if not result["accepted"]:
            continue
        if (
            result["deletion_risk"] <= deletion_target * margin
            and result["invention_risk"] <= invention_target * margin
            and result["coverage"] >= best_coverage
        ):
            best = float(threshold)
            best_coverage = float(result["coverage"])
    return best


def certify_policy(
    pool: dict[str, Any],
    raw: dict[str, Any],
    pool_threshold: float,
    raw_threshold: float,
    *,
    delta: float,
    deletion_target: float,
    invention_target: float,
) -> dict[str, Any]:
    result = evaluate_policy(pool, raw, pool_threshold, raw_threshold)
    count = int(result["accepted"])
    result["deletion_ucb"] = binomial_ucb(
        int(result["deletion_errors"]), count, delta / 2
    )
    result["invention_ucb"] = binomial_ucb(
        int(result["invention_errors"]), count, delta / 2
    )
    result["passes"] = (
        result["deletion_ucb"] <= deletion_target
        and result["invention_ucb"] <= invention_target
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deletion-target", type=float, default=0.15)
    parser.add_argument("--invention-target", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--min-accept", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = read_rows(args.experiment / "detector_val_records.csv")
    calibration_rows = read_rows(args.experiment / "calibration_records.csv")
    test_rows = read_rows(args.experiment / "test_records.csv")
    models = fit_models(training, args.seed)
    calibration = action_arrays(
        calibration_rows,
        models,
        args.deletion_target,
        args.invention_target,
    )
    test = action_arrays(
        test_rows,
        models,
        args.deletion_target,
        args.invention_target,
    )
    sample_ids = [
        row["sample_id"]
        for row in calibration_rows
        if row["method"] == "raw_nearest"
    ]
    tuning_mask = stable_tuning_mask(sample_ids, args.seed)
    pool_tuning = subset(calibration["restored_pool"], tuning_mask)
    raw_tuning = subset(calibration["raw"], tuning_mask)
    pool_certification = subset(calibration["restored_pool"], ~tuning_mask)
    raw_certification = subset(calibration["raw"], ~tuning_mask)

    pool_threshold = choose_threshold(
        pool_tuning,
        deletion_target=args.deletion_target,
        invention_target=args.invention_target,
        margin=args.margin,
        min_accept=args.min_accept,
    )
    raw_threshold = choose_raw_fallback_threshold(
        pool_tuning,
        raw_tuning,
        pool_threshold,
        deletion_target=args.deletion_target,
        invention_target=args.invention_target,
        margin=args.margin,
    )
    certificate = certify_policy(
        pool_certification,
        raw_certification,
        pool_threshold,
        raw_threshold,
        delta=args.delta,
        deletion_target=args.deletion_target,
        invention_target=args.invention_target,
    )
    test_result = evaluate_policy(
        test["restored_pool"],
        test["raw"],
        pool_threshold,
        raw_threshold,
    )
    test_result["deployable_coverage"] = (
        test_result["coverage"] if certificate["passes"] else 0.0
    )
    result = {
        "experiment": str(args.experiment.resolve()),
        "policy": "restoration pool, then raw display, then review",
        "seed": args.seed,
        "deletion_target": args.deletion_target,
        "invention_target": args.invention_target,
        "delta": args.delta,
        "margin": args.margin,
        "tuning_n": int(tuning_mask.sum()),
        "certification_n": int((~tuning_mask).sum()),
        "pool_threshold": pool_threshold,
        "raw_threshold": raw_threshold,
        "tuning": evaluate_policy(
            pool_tuning, raw_tuning, pool_threshold, raw_threshold
        ),
        "certificate": certificate,
        "test": test_result,
    }
    output = args.output or args.experiment / "sequential_router.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
