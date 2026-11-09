"""Two-domain TIM analysis: marginal (legacy) plus familywise certification.

For each domain (Carinthia-S primary v4 records, new KolektorSDD2 records) and
each training seed, this driver:

1. reproduces the legacy per-policy marginal analysis exactly as in
   ``scripts/run_reported_analysis.py`` (tuned gate + single marginal
   certificate), for continuity with the arXiv study; and
2. runs the new fixed-sequence familywise procedure of
   ``familywise_certification.py`` in two prespecified readings:
   (a) single-policy: full delta spent on one policy's gate sequence, and
   (b) portfolio: Bonferroni delta across the eight-policy family with valid
       max-coverage selection.

All inputs are frozen record CSVs; no model is retrained here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from saferestore.certification import DualRiskTargets
import scripts.run_reported_analysis as reported
from scripts.analyze_router_pool import read_rows
from scripts.run_reported_analysis import (
    POLICIES,
    build_policy_arrays,
    evaluate_policy,
    fit_risk_models,
    json_safe,
    run_seed,
    stable_tuning_mask,
    subset,
)
from familywise_certification import (
    fixed_sequence_certify,
    portfolio_certify,
    tuning_selected_certify,
)
from run_ksdd2_pipeline import KSDD2_FEATURES

SEEDS = (201, 203, 207, 211, 223)
CARINTHIA_FEATURES = reported.FEATURES

DOMAIN_FEATURES = {
    "carinthia_s": CARINTHIA_FEATURES,
    "kolektor_sdd2": KSDD2_FEATURES,
}

_FIT_BALANCED = reported.fit_risk_models


def fit_risk_models_calibrated(rows, seed, deletion_floor):
    """reported.fit_risk_models without class balancing.

    On sparse-activation domains the class-balanced sigmoids of the two
    endpoints are not scale-comparable, so the worst-endpoint score of
    \\eqref{eq:minimax-score} is dominated by whichever model is more
    rebalanced; plain maximum-likelihood fits keep both scores on an
    approximate probability scale.  Spec format is unchanged.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as _np

    models = {}
    for method in reported.METHODS:
        local = [row for row in rows if row["method"] == method]
        model = {"method": method, "features": list(reported.FEATURES)}
        for target in ("deletion_error", "invention_error"):
            population = (
                [row for row in local if row["has_defect"]]
                if target == "deletion_error"
                else local
            )
            x = _np.asarray(
                [[row[name] for name in reported.FEATURES] for row in population],
                dtype=_np.float64,
            )
            y = _np.asarray(
                [reported._target_value(row, target, deletion_floor) for row in population],
                dtype=_np.int8,
            )
            if y.size == 0 or _np.unique(y).size < 2:
                model[target] = {"constant": float(y[0] if y.size else 1.0)}
                continue
            scaler = StandardScaler().fit(x)
            estimator = LogisticRegression(
                max_iter=2000, random_state=seed, solver="liblinear"
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


DOMAIN_SCORE_FIT = {
    "carinthia_s": _FIT_BALANCED,
    "kolektor_sdd2": fit_risk_models_calibrated,
}


def analyze_domain_seed(
    folder: Path,
    seed: int,
    floor: float,
    targets: DualRiskTargets,
) -> dict:
    legacy = run_seed(folder, seed, floor, targets, tune_key="v4tunecert")

    training_rows = read_rows(folder / "detector_val_records.csv")
    calibration_rows = read_rows(folder / "calibration_records.csv")
    test_rows = read_rows(folder / "test_records.csv")
    models = reported.fit_risk_models(training_rows, seed, floor)
    calibration = build_policy_arrays(calibration_rows, models, targets, floor)
    test = build_policy_arrays(test_rows, models, targets, floor)
    tuning_mask = stable_tuning_mask(
        calibration["action_pool"]["sample_id"].tolist(), tune_key="v4tunecert"
    )

    arrays = {
        name: {
            "score": np.asarray(calibration[name]["score"], dtype=np.float64),
            "positive": np.asarray(calibration[name]["positive"], dtype=bool),
            "deletion": np.asarray(calibration[name]["deletion"]),
            "invention": np.asarray(calibration[name]["invention"]),
        }
        for name in POLICIES
    }

    single_policy = {}
    for name in POLICIES:
        result = fixed_sequence_certify(
            tuning_scores=arrays[name]["score"][tuning_mask],
            tuning_positive=arrays[name]["positive"][tuning_mask],
            certification_scores=arrays[name]["score"][~tuning_mask],
            certification_positive=arrays[name]["positive"][~tuning_mask],
            certification_deletion=arrays[name]["deletion"][~tuning_mask],
            certification_invention=arrays[name]["invention"][~tuning_mask],
            targets=targets,
            delta_alloc=targets.delta,
            tuning_deletion=arrays[name]["deletion"][tuning_mask],
            tuning_invention=arrays[name]["invention"][tuning_mask],
        ).to_dict()
        if result["certified_threshold"] is not None:
            test_eval = evaluate_policy(test[name], result["certified_threshold"])
            result["test"] = test_eval
        single_policy[name] = result

    portfolio = portfolio_certify(arrays, tuning_mask, targets)
    if portfolio["selected_policy"] is not None:
        selected = portfolio["selected_policy"]
        portfolio["selected_test"] = evaluate_policy(
            test[selected], portfolio["selected_threshold"]
        )

    tuning_selected = tuning_selected_certify(arrays, tuning_mask, targets)
    if tuning_selected["certificate"]["certified_threshold"] is not None:
        tuning_selected["selected_test"] = evaluate_policy(
            test[tuning_selected["selected_policy"]],
            tuning_selected["certificate"]["certified_threshold"],
        )

    return {
        "seed": seed,
        "deletion_recall_floor": floor,
        "tuning_n": int(tuning_mask.sum()),
        "certification_n": int((~tuning_mask).sum()),
        "legacy_marginal": legacy,
        "familywise_single_policy": single_policy,
        "familywise_portfolio": portfolio,
        "familywise_tuning_selected": tuning_selected,
    }


def summarize(results: list[dict], floor: float) -> dict:
    local = [r for r in results if r["deletion_recall_floor"] == floor]
    policy_summary = {}
    for name in POLICIES:
        legacy_passes = sum(
            bool(r["legacy_marginal"]["policies"][name]["certificate"]["passes"])
            for r in local
        )
        fam = [r["familywise_single_policy"][name] for r in local]
        fam_certified = [f for f in fam if f["certified_threshold"] is not None]
        policy_summary[name] = {
            "legacy_marginal_passes": legacy_passes,
            "familywise_single_certified": len(fam_certified),
            "familywise_single_mean_test_coverage": (
                float(np.mean([f["test"]["coverage"] for f in fam_certified]))
                if fam_certified
                else 0.0
            ),
        }
    portfolio_rows = [r["familywise_portfolio"] for r in local]
    selections = [p["selected_policy"] for p in portfolio_rows]
    ts_rows = [r["familywise_tuning_selected"] for r in local]
    return {
        "floor": floor,
        "repetitions": len(local),
        "policies": policy_summary,
        "portfolio_selections": selections,
        "portfolio_certified_repetitions": sum(s is not None for s in selections),
        "portfolio_selected_test_coverage": [
            (r["familywise_portfolio"].get("selected_test") or {}).get("coverage")
            for r in local
        ],
        "tuning_selected_policies": [t["selected_policy"] for t in ts_rows],
        "tuning_selected_certified": sum(
            t["certificate"]["certified_threshold"] is not None for t in ts_rows
        ),
        "tuning_selected_test_coverage": [
            (r["familywise_tuning_selected"].get("selected_test") or {}).get("coverage")
            for r in local
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "tim_upgrade" / "experiments" / "tim_analysis")
    parser.add_argument("--floors", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument(
        "--ksdd2-template",
        default="ksdd2_primary_trainseed{seed}",
        help="KSDD2 experiment folder pattern under tim_upgrade/experiments",
    )
    args = parser.parse_args()
    targets = DualRiskTargets(
        deletion=0.15, invention=0.15, delta=args.delta,
        tuning_margin=0.50, min_accept=15, min_positive_accept=15,
    )
    domains = {
        "carinthia_s": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
        "kolektor_sdd2": REPO / "tim_upgrade" / "experiments" / args.ksdd2_template,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    overall: dict[str, dict] = {}
    for domain, template in domains.items():
        results = []
        reported.FEATURES = DOMAIN_FEATURES[domain]
        reported.fit_risk_models = DOMAIN_SCORE_FIT[domain]
        for floor in args.floors:
            for seed in SEEDS:
                folder = Path(str(template).format(seed=seed))
                if not (folder / "test_records.csv").exists():
                    print(f"skip missing {folder}")
                    continue
                result = analyze_domain_seed(folder, seed, floor, targets)
                results.append(result)
                out = args.output / f"{domain}_seed{seed}_floor{floor:.2f}.json"
                out.write_text(
                    json.dumps(json_safe(result), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
        overall[domain] = {
            f"{floor:.2f}": summarize(results, floor) for floor in args.floors
        }
    (args.output / "tim_summary.json").write_text(
        json.dumps(json_safe(overall), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(overall), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
