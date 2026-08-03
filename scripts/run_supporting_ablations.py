"""Feature and pool ablations using the primary reported-analysis path.

Unlike the legacy router-ablation script, this uses:
- seed-independent tune/certify hash (default ``v4tunecert``)
- deletion labels fitted only on defect-positive images
- primary minimax action selection without raw-fallback routing
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from saferestore.certification import (
    DualRiskTargets,
    certify_threshold,
    choose_threshold,
    evaluate_threshold,
    select_minimax_actions,
)
from scripts.analyze_router_pool import read_rows
from scripts.run_dataset_pipeline import FEATURES
from scripts.run_reported_analysis import METHODS, stable_tuning_mask, subset


FEATURE_VARIANTS = {
    "full": tuple(FEATURES),
    "no_entropy": tuple(name for name in FEATURES if name != "entropy"),
    "no_consistency": tuple(
        name for name in FEATURES if name != "forward_consistency"
    ),
    "no_disagreement": tuple(
        name for name in FEATURES if name != "detector_disagreement"
    ),
    "no_evidence_shift": tuple(name for name in FEATURES if name != "evidence_shift"),
    "no_residual": tuple(name for name in FEATURES if name != "restoration_residual"),
    "no_probability_area": tuple(
        name for name in FEATURES if name != "high_probability_area"
    ),
    "uncertainty_only": ("entropy", "high_probability_area"),
}

POOL_VARIANTS = {
    "all": METHODS,
    "no_raw": METHODS[1:],
    "no_learned": METHODS[:-1],
    "learned_only": ("learned",),
    "interpolation_only": ("bilinear", "bicubic"),
    "no_sharpening": (
        "raw_nearest",
        "bilinear",
        "bicubic",
        "smooth_bicubic",
        "learned",
    ),
    "no_smoothing": (
        "raw_nearest",
        "bilinear",
        "bicubic",
        "unsharp_bicubic",
        "learned",
    ),
}


def _target_value(row: dict[str, Any], target: str, deletion_floor: float) -> int:
    if target == "deletion_error":
        return int(row["has_defect"] and row["pixel_recall"] < deletion_floor)
    if target == "invention_error":
        return int(row["invention_error"])
    raise ValueError(target)


def fit_risk_models(
    rows: list[dict[str, Any]],
    seed: int,
    deletion_floor: float,
    features: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        model: dict[str, Any] = {"method": method, "features": list(features)}
        for target in ("deletion_error", "invention_error"):
            population = (
                [row for row in local if row["has_defect"]]
                if target == "deletion_error"
                else local
            )
            x = np.asarray(
                [[row[name] for name in features] for row in population],
                dtype=np.float64,
            )
            y = np.asarray(
                [_target_value(row, target, deletion_floor) for row in population],
                dtype=np.int8,
            )
            if y.size == 0 or np.unique(y).size < 2:
                model[target] = {"constant": float(y[0] if y.size else 1.0)}
                continue
            scaler = StandardScaler().fit(x)
            estimator = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            ).fit(scaler.transform(x), y)
            model[target] = {
                "coef": estimator.coef_[0],
                "intercept": float(estimator.intercept_[0]),
                "mean": scaler.mean_,
                "scale": scaler.scale_,
            }
        models[method] = model
    return models


def predict_model(
    model: dict[str, Any], rows: list[dict[str, Any]], features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[row[name] for name in features] for row in rows], dtype=np.float64)
    predictions = []
    for target in ("deletion_error", "invention_error"):
        spec = model[target]
        if "constant" in spec:
            predictions.append(np.full(len(rows), spec["constant"], dtype=np.float64))
            continue
        normalized = (x - spec["mean"]) / spec["scale"]
        logits = normalized @ spec["coef"] + spec["intercept"]
        predictions.append(1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0))))
    return predictions[0], predictions[1]


def build_pool_policy(
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    targets: DualRiskTargets,
    deletion_floor: float,
    features: tuple[str, ...],
    pool_methods: tuple[str, ...],
) -> dict[str, Any]:
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    sample_ids = [row["sample_id"] for row in by_method["raw_nearest"]]
    deletion_probability: dict[str, np.ndarray] = {}
    invention_probability: dict[str, np.ndarray] = {}
    for method in METHODS:
        deletion_probability[method], invention_probability[method] = predict_model(
            models[method], by_method[method], features
        )
    selected_method, selected_score = select_minimax_actions(
        deletion_probability, invention_probability, pool_methods, targets
    )
    selected_rows = [
        by_method[str(method)][index] for index, method in enumerate(selected_method)
    ]
    return {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "score": selected_score,
        "positive": np.asarray([row["has_defect"] for row in selected_rows], dtype=bool),
        "deletion": np.asarray(
            [
                int(row["has_defect"] and row["pixel_recall"] < deletion_floor)
                for row in selected_rows
            ],
            dtype=np.int8,
        ),
        "invention": np.asarray(
            [row["invention_error"] for row in selected_rows], dtype=np.int8
        ),
        "method": selected_method,
    }


def run_variant(
    folder: Path,
    seed: int,
    features: tuple[str, ...],
    pool_methods: tuple[str, ...],
    *,
    deletion_floor: float,
    targets: DualRiskTargets,
    tune_key: str,
) -> dict[str, Any]:
    training_rows = read_rows(folder / "detector_val_records.csv")
    calibration_rows = read_rows(folder / "calibration_records.csv")
    test_rows = read_rows(folder / "test_records.csv")
    models = fit_risk_models(training_rows, seed, deletion_floor, features)
    calibration = build_pool_policy(
        calibration_rows, models, targets, deletion_floor, features, pool_methods
    )
    test = build_pool_policy(
        test_rows, models, targets, deletion_floor, features, pool_methods
    )
    tuning_mask = stable_tuning_mask(calibration["sample_id"].tolist(), tune_key=tune_key)
    tuning = subset(calibration, tuning_mask)
    certification = subset(calibration, ~tuning_mask)
    threshold = choose_threshold(
        tuning["score"],
        tuning["positive"],
        tuning["deletion"],
        tuning["invention"],
        targets,
    )
    certificate = certify_threshold(
        certification["score"],
        threshold,
        certification["positive"],
        certification["deletion"],
        certification["invention"],
        targets,
    )
    test_eval = evaluate_threshold(
        test["score"],
        threshold,
        test["positive"],
        test["deletion"],
        test["invention"],
    )
    passes = bool(certificate["passes"])
    return {
        "certificate_pass": int(passes),
        "threshold": None if threshold == float("-inf") else float(threshold),
        "certification_coverage": certificate["coverage"],
        "deletion_ucb": certificate["deletion_ucb"],
        "invention_ucb": certificate["invention_ucb"],
        "test_coverage": test_eval["coverage"],
        "deployable_test_coverage": test_eval["coverage"] if passes else 0.0,
        "test_deletion_risk": test_eval["deletion_risk"],
        "test_invention_risk": test_eval["invention_risk"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--folder-template", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-key", default="v4tunecert")
    parser.add_argument("--deletion-floor", type=float, default=0.25)
    parser.add_argument("--deletion-target", type=float, default=0.15)
    parser.add_argument("--invention-target", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--min-accept", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = DualRiskTargets(
        deletion=args.deletion_target,
        invention=args.invention_target,
        delta=args.delta,
        tuning_margin=args.margin,
        min_accept=args.min_accept,
        min_positive_accept=args.min_accept,
    )
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        folder = args.root / args.folder_template.format(seed=seed)
        for variant, features in FEATURE_VARIANTS.items():
            result = run_variant(
                folder,
                seed,
                features,
                POOL_VARIANTS["all"],
                deletion_floor=args.deletion_floor,
                targets=targets,
                tune_key=args.tune_key,
            )
            rows.append(
                {
                    "ablation_type": "feature",
                    "variant": variant,
                    "seed": seed,
                    **result,
                }
            )
        for variant, pool in POOL_VARIANTS.items():
            result = run_variant(
                folder,
                seed,
                FEATURE_VARIANTS["full"],
                pool,
                deletion_floor=args.deletion_floor,
                targets=targets,
                tune_key=args.tune_key,
            )
            rows.append(
                {
                    "ablation_type": "pool",
                    "variant": variant,
                    "seed": seed,
                    **result,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (args.output / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    keys = {(row["ablation_type"], row["variant"]) for row in rows}
    for ablation_type, variant in sorted(keys):
        local = [
            row
            for row in rows
            if row["ablation_type"] == ablation_type and row["variant"] == variant
        ]
        coverages = [float(row["deployable_test_coverage"]) for row in local]
        summary_rows.append(
            {
                "ablation_type": ablation_type,
                "variant": variant,
                "certificate_passes": int(sum(row["certificate_pass"] for row in local)),
                "coverage_mean": float(np.mean(coverages)),
                "coverage_sd": float(np.std(coverages, ddof=1)) if len(coverages) > 1 else 0.0,
            }
        )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "note": (
                    "Feature and action-pool ablations use the same deterministic "
                    "tune/certify partition, positive-only evidence-loss fitting, "
                    "and minimax action selection as the reported analysis. The "
                    "full-feature/all-action condition must match the primary "
                    "result before component comparisons."
                ),
                "tune_key": args.tune_key,
                "deletion_floor": args.deletion_floor,
                "deletion_target": args.deletion_target,
                "invention_target": args.invention_target,
                "delta": args.delta,
                "margin": args.margin,
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
