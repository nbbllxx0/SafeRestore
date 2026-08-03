from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


RESTORATION_METHODS = (
    "bilinear",
    "bicubic",
    "smooth_bicubic",
    "unsharp_bicubic",
    "learned",
)
ALL_ACTION_METHODS = ("raw_nearest",) + RESTORATION_METHODS


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    float_fields = {
        "pixel_recall",
        "clean_fpr",
        "entropy",
        "forward_consistency",
        "detector_disagreement",
        "evidence_shift",
        "restoration_residual",
        "high_probability_area",
        "psnr",
    }
    int_fields = {"has_defect", "deletion_error", "invention_error"}
    for row in rows:
        for field in float_fields:
            row[field] = float(row[field])
        for field in int_fields:
            row[field] = int(row[field])
    return rows


def method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["method"] == method]


def fit_models(training: list[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    from scripts.run_dataset_pipeline import fit_risk_score

    return {
        method: fit_risk_score(training, method, seed)
        for method in ("raw_nearest",) + RESTORATION_METHODS
    }


def action_arrays(
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    deletion_target: float,
    invention_target: float,
) -> dict[str, dict[str, Any]]:
    from scripts.run_dataset_pipeline import predict_risk

    by_method = {method: method_rows(rows, method) for method in models}
    sample_ids = [row["sample_id"] for row in by_method["raw_nearest"]]
    for method, subset in by_method.items():
        if [row["sample_id"] for row in subset] != sample_ids:
            raise ValueError(f"sample ordering mismatch for {method}")

    method_scores: dict[str, np.ndarray] = {}
    for method, subset in by_method.items():
        p_del, p_inv = predict_risk(models[method], subset)
        method_scores[method] = np.maximum(
            p_del / deletion_target,
            p_inv / invention_target,
        )

    restored_matrix = np.stack([method_scores[method] for method in RESTORATION_METHODS], axis=1)
    best_index = restored_matrix.argmin(axis=1)
    selected_method = np.asarray(RESTORATION_METHODS, dtype=object)[best_index]
    restored_rows = [
        by_method[str(method)][index] for index, method in enumerate(selected_method)
    ]
    raw_rows = by_method["raw_nearest"]
    return {
        "raw": {
            "score": method_scores["raw_nearest"],
            "deletion": np.asarray([row["deletion_error"] for row in raw_rows], dtype=np.int8),
            "invention": np.asarray([row["invention_error"] for row in raw_rows], dtype=np.int8),
            "method": np.full(len(raw_rows), "raw_nearest", dtype=object),
        },
        "restored_pool": {
            "score": restored_matrix[np.arange(len(best_index)), best_index],
            "deletion": np.asarray([row["deletion_error"] for row in restored_rows], dtype=np.int8),
            "invention": np.asarray([row["invention_error"] for row in restored_rows], dtype=np.int8),
            "method": selected_method,
        },
        **{
            method: {
                "score": method_scores[method],
                "deletion": np.asarray(
                    [row["deletion_error"] for row in by_method[method]], dtype=np.int8
                ),
                "invention": np.asarray(
                    [row["invention_error"] for row in by_method[method]], dtype=np.int8
                ),
                "method": np.full(len(sample_ids), method, dtype=object),
            }
            for method in RESTORATION_METHODS
        },
    }


def choose_threshold(
    calibration: dict[str, Any],
    *,
    deletion_target: float,
    invention_target: float,
    margin: float,
    min_accept: int,
) -> float:
    score = calibration["score"]
    order = np.argsort(score, kind="stable")
    deletion = calibration["deletion"][order]
    invention = calibration["invention"][order]
    best = float("-inf")
    for count in range(min_accept, len(order) + 1):
        if (
            deletion[:count].mean() <= deletion_target * margin
            and invention[:count].mean() <= invention_target * margin
        ):
            best = float(score[order[count - 1]])
    return best


def evaluate(action: dict[str, Any], threshold: float) -> dict[str, Any]:
    accepted = action["score"] <= threshold
    count = int(accepted.sum())
    methods, method_counts = np.unique(action["method"][accepted], return_counts=True)
    return {
        "n": int(len(accepted)),
        "accepted": count,
        "coverage": count / len(accepted),
        "deletion_risk": float(action["deletion"][accepted].mean()) if count else None,
        "invention_risk": float(action["invention"][accepted].mean()) if count else None,
        "method_counts": {
            str(method): int(value) for method, value in zip(methods, method_counts)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margin", type=float, default=0.85)
    parser.add_argument("--min-accept", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = read_rows(args.experiment / "detector_val_records.csv")
    calibration_rows = read_rows(args.experiment / "calibration_records.csv")
    test_rows = read_rows(args.experiment / "test_records.csv")
    models = fit_models(training, args.seed)
    sweeps: list[dict[str, Any]] = []
    for deletion_target in (0.30, 0.40, 0.50, 0.60):
        for invention_target in (0.05, 0.10, 0.15):
            calibration = action_arrays(
                calibration_rows, models, deletion_target, invention_target
            )
            test = action_arrays(test_rows, models, deletion_target, invention_target)
            for action_name in ("raw", "learned", "smooth_bicubic", "restored_pool"):
                threshold = choose_threshold(
                    calibration[action_name],
                    deletion_target=deletion_target,
                    invention_target=invention_target,
                    margin=args.margin,
                    min_accept=args.min_accept,
                )
                sweeps.append(
                    {
                        "deletion_target": deletion_target,
                        "invention_target": invention_target,
                        "action": action_name,
                        "threshold": threshold,
                        "calibration": evaluate(calibration[action_name], threshold),
                        "test": evaluate(test[action_name], threshold),
                    }
                )
    result = {
        "experiment": str(args.experiment.resolve()),
        "note": "Diagnostic threshold sweep; final inferential claims require independent certification.",
        "margin": args.margin,
        "sweeps": sweeps,
    }
    output = args.output or args.experiment / "router_pool_diagnostic.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
