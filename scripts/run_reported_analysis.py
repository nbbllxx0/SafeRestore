from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy
import sklearn
import torch
import torchvision
from PIL import Image
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


METHODS = (
    "raw_nearest",
    "bilinear",
    "bicubic",
    "smooth_bicubic",
    "unsharp_bicubic",
    "learned",
)
RESTORATION_METHODS = METHODS[1:]
POLICIES = METHODS + ("restoration_pool", "action_pool")


def json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stable_tuning_mask(
    sample_ids: list[str],
    *,
    tune_key: str = "v4tunecert",
) -> np.ndarray:
    """Seed-independent ~30% tuning mask for confirmatory protocol v3."""

    values = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"{tune_key}:{sample_id}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") % 10 < 3)
    mask = np.asarray(values, dtype=bool)
    if mask.all() or (~mask).all():
        raise RuntimeError("degenerate tune/certify subdivision")
    return mask


def stable_fraction_mask(
    sample_ids: np.ndarray,
    fraction: float,
    *,
    sensitivity_key: str = "v3sensitivity",
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    limit = int(fraction * 2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{sensitivity_key}:{sample_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            < limit
            for sample_id in sample_ids
        ],
        dtype=bool,
    )


def _target_value(row: dict[str, Any], target: str, deletion_floor: float) -> int:
    if target == "deletion_error":
        return int(row["has_defect"] and row["pixel_recall"] < deletion_floor)
    if target == "invention_error":
        return int(row["invention_error"])
    raise ValueError(target)


def fit_risk_models(
    rows: list[dict[str, Any]], seed: int, deletion_floor: float
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        model: dict[str, Any] = {"method": method, "features": list(FEATURES)}
        for target in ("deletion_error", "invention_error"):
            population = (
                [row for row in local if row["has_defect"]]
                if target == "deletion_error"
                else local
            )
            x = np.asarray(
                [[row[name] for name in FEATURES] for row in population],
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
                "training_n": int(y.size),
                "incidents": int(y.sum()),
            }
        models[method] = model
    return models


def predict_model(model: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
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


def build_policy_arrays(
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    targets: DualRiskTargets,
    deletion_floor: float,
) -> dict[str, dict[str, Any]]:
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    sample_ids = [row["sample_id"] for row in by_method["raw_nearest"]]
    for method, local in by_method.items():
        if [row["sample_id"] for row in local] != sample_ids:
            raise ValueError(f"sample ordering mismatch for {method}")

    deletion_probability: dict[str, np.ndarray] = {}
    invention_probability: dict[str, np.ndarray] = {}
    for method in METHODS:
        deletion_probability[method], invention_probability[method] = predict_model(
            models[method], by_method[method]
        )

    def pack(methods: tuple[str, ...]) -> dict[str, Any]:
        selected_method, selected_score = select_minimax_actions(
            deletion_probability, invention_probability, methods, targets
        )
        selected_rows = [
            by_method[str(method)][index]
            for index, method in enumerate(selected_method)
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

    output = {
        method: pack((method,))
        for method in METHODS
    }
    output["restoration_pool"] = pack(RESTORATION_METHODS)
    output["action_pool"] = pack(METHODS)
    return output


def subset(policy: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {key: np.asarray(value)[mask] for key, value in policy.items()}


def evaluate_policy(policy: dict[str, Any], threshold: float) -> dict[str, Any]:
    result = evaluate_threshold(
        policy["score"],
        threshold,
        policy["positive"],
        policy["deletion"],
        policy["invention"],
    )
    accepted = policy["score"] <= threshold
    methods, counts = np.unique(policy["method"][accepted], return_counts=True)
    result["method_counts"] = {
        str(method): int(count) for method, count in zip(methods, counts)
    }
    return result


def run_seed(
    folder: Path,
    seed: int,
    deletion_floor: float,
    targets: DualRiskTargets,
    *,
    tune_key: str = "v4tunecert",
) -> dict[str, Any]:
    training_rows = read_rows(folder / "detector_val_records.csv")
    calibration_rows = read_rows(folder / "calibration_records.csv")
    test_rows = read_rows(folder / "test_records.csv")
    shift_path = folder / "shift_test_records.csv"
    shift_rows = read_rows(shift_path) if shift_path.exists() else []

    models = fit_risk_models(training_rows, seed, deletion_floor)
    calibration = build_policy_arrays(calibration_rows, models, targets, deletion_floor)
    test = build_policy_arrays(test_rows, models, targets, deletion_floor)
    shift = (
        build_policy_arrays(shift_rows, models, targets, deletion_floor)
        if shift_rows
        else None
    )
    tuning_mask = stable_tuning_mask(
        calibration["action_pool"]["sample_id"].tolist(),
        tune_key=tune_key,
    )

    policy_results: dict[str, Any] = {}
    for name in POLICIES:
        tuning = subset(calibration[name], tuning_mask)
        certification = subset(calibration[name], ~tuning_mask)
        threshold = choose_threshold(
            tuning["score"],
            tuning["positive"],
            tuning["deletion"],
            tuning["invention"],
            targets,
        )
        tune_result = evaluate_policy(tuning, threshold)
        certificate = certify_threshold(
            certification["score"],
            threshold,
            certification["positive"],
            certification["deletion"],
            certification["invention"],
            targets,
        )
        accepted = certification["score"] <= threshold
        cert_methods, cert_counts = np.unique(
            certification["method"][accepted], return_counts=True
        )
        certificate["method_counts"] = {
            str(method): int(count)
            for method, count in zip(cert_methods, cert_counts)
        }
        test_result = evaluate_policy(test[name], threshold)
        test_result["deployable_coverage"] = (
            test_result["coverage"] if certificate["passes"] else 0.0
        )
        test_result["deployable_positive_coverage"] = (
            test_result["positive_coverage"] if certificate["passes"] else 0.0
        )
        test_result["deployable_clean_coverage"] = (
            test_result["clean_coverage"] if certificate["passes"] else 0.0
        )
        shift_result = evaluate_policy(shift[name], threshold) if shift else None
        policy_results[name] = {
            "threshold": threshold,
            "tuning": tune_result,
            "certificate": certificate,
            "test": test_result,
            "shift_diagnostic": shift_result,
        }

    action_policy = calibration["action_pool"]
    action_certification = subset(action_policy, ~tuning_mask)
    sensitivity = []
    threshold = policy_results["action_pool"]["threshold"]
    for fraction in (0.25, 0.50, 0.75, 1.00):
        fraction_mask = stable_fraction_mask(
            action_certification["sample_id"], fraction
        )
        local = subset(action_certification, fraction_mask)
        result = certify_threshold(
            local["score"],
            threshold,
            local["positive"],
            local["deletion"],
            local["invention"],
            targets,
        )
        sensitivity.append({"fraction": fraction, **result})

    return {
        "seed": seed,
        "deletion_recall_floor": deletion_floor,
        "invention_fpr_limit": 0.002,
        "targets": {
            "deletion": targets.deletion,
            "invention": targets.invention,
            "delta": targets.delta,
            "tuning_margin": targets.tuning_margin,
            "min_accept": targets.min_accept,
            "min_positive_accept": targets.min_positive_accept,
        },
        "sampling_assumption": (
            "Certification observations are independent and identically distributed "
            "image-level units from the nominal population; thresholds are fixed on a "
            "disjoint tuning subset."
        ),
        "tuning_n": int(tuning_mask.sum()),
        "certification_n": int((~tuning_mask).sum()),
        "policies": policy_results,
        "certificate_size_sensitivity": sensitivity,
    }


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analysis_scope": "reported five-repetition evaluation",
        "interpretation": (
            "Seed repetitions are descriptive robustness checks; image identities recur "
            "across repetitions and are not treated as independent experimental units."
        ),
        "floors": {},
    }
    for floor in sorted({result["deletion_recall_floor"] for result in results}):
        local_results = [
            result for result in results if result["deletion_recall_floor"] == floor
        ]
        floor_summary: dict[str, Any] = {}
        for policy in POLICIES:
            local = [result["policies"][policy] for result in local_results]
            passes = [bool(item["certificate"]["passes"]) for item in local]
            deployable = [float(item["test"]["deployable_coverage"]) for item in local]
            conditional = [float(item["test"]["coverage"]) for item in local]
            positive = [
                float(item["test"]["deployable_positive_coverage"]) for item in local
            ]
            clean = [
                float(item["test"]["deployable_clean_coverage"]) for item in local
            ]
            passed_items = [item for item, passed in zip(local, passes) if passed]
            deployable_mean, deployable_sd = _mean_sd(deployable)
            conditional_mean, conditional_sd = _mean_sd(conditional)
            positive_mean, positive_sd = _mean_sd(positive)
            clean_mean, clean_sd = _mean_sd(clean)
            floor_summary[policy] = {
                "certificate_passes": int(sum(passes)),
                "repetitions": len(local),
                "deployable_coverage_mean": deployable_mean,
                "deployable_coverage_sd": deployable_sd,
                "conditional_coverage_mean": conditional_mean,
                "conditional_coverage_sd": conditional_sd,
                "deployable_positive_coverage_mean": positive_mean,
                "deployable_positive_coverage_sd": positive_sd,
                "deployable_clean_coverage_mean": clean_mean,
                "deployable_clean_coverage_sd": clean_sd,
                "test_deletion_risk_mean_among_passes": (
                    float(np.mean([item["test"]["deletion_risk"] for item in passed_items]))
                    if passed_items
                    else None
                ),
                "test_invention_risk_mean_among_passes": (
                    float(np.mean([item["test"]["invention_risk"] for item in passed_items]))
                    if passed_items
                    else None
                ),
                "certification_deletion_denominators": [
                    int(item["certificate"]["deletion_denominator"]) for item in local
                ],
                "certification_invention_denominators": [
                    int(item["certificate"]["invention_denominator"]) for item in local
                ],
            }
        summary["floors"][f"{floor:.2f}"] = floor_summary
    return summary


def write_seed_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for result in results:
        for policy, payload in result["policies"].items():
            certificate = payload["certificate"]
            test = payload["test"]
            rows.append(
                {
                    "seed": result["seed"],
                    "deletion_recall_floor": result["deletion_recall_floor"],
                    "policy": policy,
                    "certificate_pass": int(certificate["passes"]),
                    "certification_accepted": certificate["accepted"],
                    "certification_positive": certificate["accepted_positive"],
                    "certification_deletion_errors": certificate["deletion_errors"],
                    "certification_deletion_ucb": certificate["deletion_ucb"],
                    "certification_invention_errors": certificate["invention_errors"],
                    "certification_invention_ucb": certificate["invention_ucb"],
                    "test_coverage": test["coverage"],
                    "deployable_test_coverage": test["deployable_coverage"],
                    "deployable_positive_coverage": test["deployable_positive_coverage"],
                    "deployable_clean_coverage": test["deployable_clean_coverage"],
                    "test_deletion_risk": test["deletion_risk"],
                    "test_invention_risk": test["invention_risk"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the reported population-specific SafeRestore analysis"
    )
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 53, 67, 79, 97])
    parser.add_argument(
        "--folder-template",
        default="mixed_carinthia_seed{seed}",
        help="Experiment folder pattern relative to --root",
    )
    parser.add_argument("--deletion-floors", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--output", type=Path, default=Path("experiments/reported_analysis"))
    parser.add_argument("--deletion-target", type=float, default=0.15)
    parser.add_argument("--invention-target", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--min-accept", type=int, default=15)
    parser.add_argument("--min-positive-accept", type=int, default=15)
    parser.add_argument(
        "--external-folder",
        type=Path,
        default=Path("experiments/mixed_kolektor_balanced_seed7"),
    )
    parser.add_argument("--tune-key", default="v4tunecert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = DualRiskTargets(
        deletion=args.deletion_target,
        invention=args.invention_target,
        delta=args.delta,
        tuning_margin=args.margin,
        min_accept=args.min_accept,
        min_positive_accept=args.min_positive_accept,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for floor in args.deletion_floors:
        for seed in args.seeds:
            folder = args.root / args.folder_template.format(seed=seed)
            result = run_seed(folder, seed, floor, targets, tune_key=args.tune_key)
            results.append(result)
            output = args.output / f"seed_{seed}_floor_{floor:.2f}.json"
            output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")

    summary = summarize(results)
    summary["runtime"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "folder_template": args.folder_template,
        "seeds": list(args.seeds),
    }
    if args.external_folder.exists():
        external = run_seed(args.external_folder, 7, 0.50, targets)
        (args.output / "kolektor_fold0_stress.json").write_text(
            json.dumps(json_safe(external), indent=2, allow_nan=False), encoding="utf-8"
        )
        summary["external_stress"] = {
            "dataset": "KolektorSDD official fold 0",
            "role": "finite-sample stress; not external validation",
            "action_pool": external["policies"]["action_pool"],
        }
    (args.output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    write_seed_csv(results, args.output / "seed_policy_results.csv")
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
