"""WP2 shared harness: re-run the definitive TIM analysis on frozen records
with relabelled endpoints (r0, f0) and arbitrary targets (alpha, delta).

Read-only on tim_upgrade/, scripts/, src/, experiments/.  Nothing is written
outside tim_rr/wp2/.  Import pattern copied from tim_rr/wp1/ordering_variants.py.

Relabelling.  The stored ``invention_error`` column uses f0 = 0.002 and the
stored ``deletion_error`` column uses r0 = 0.25.  The definitive code derives
the evidence-loss label from ``pixel_recall < floor`` (already parametrised),
but reads the activation label from the stored ``invention_error`` column in
both ``_target_value`` (score-model labels) and ``build_policy_arrays``
(qualification labels).  We therefore wrap ``read_rows`` so that every row is
relabelled at read time:

    invention_error = int(clean_fpr > f0)
    deletion_error  = int(has_defect and pixel_recall < r0)

so the logistic score models are refitted on the new labels and the gates are
qualified against them.  At (r0, f0) = (0.25, 0.002) the relabelling is the
identity (verified: 0 mismatches in every records file checked).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tim_upgrade" / "scripts"))

import familywise_certification as fc  # noqa: E402
import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
from scripts.analyze_router_pool import read_rows as _orig_read_rows  # noqa: E402
from saferestore.certification import DualRiskTargets, clopper_pearson_upper  # noqa: E402

POLICIES = tuple(reported.POLICIES)
SEEDS = tuple(rta.SEEDS)
DOMAINS = {
    "carinthia_s": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
    "kolektor_sdd2": REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed{seed}",
}
DEFINITIVE_DIR = REPO / "tim_upgrade" / "experiments" / "tim_analysis"

# Current endpoint thresholds used by the relabelling read_rows wrapper.
_STATE = {"r0": 0.25, "f0": 0.002, "domain": None}
_RAW_CACHE: dict[str, list] = {}
_LABEL_CACHE: dict[tuple, list] = {}
_FIT_CACHE: dict[tuple, dict] = {}


def _relabelled_read_rows(path):
    key_path = str(Path(path).resolve())
    if key_path not in _RAW_CACHE:
        _RAW_CACHE[key_path] = _orig_read_rows(Path(path))
    r0, f0 = _STATE["r0"], _STATE["f0"]
    key = (key_path, r0, f0)
    if key not in _LABEL_CACHE:
        rows = copy.deepcopy(_RAW_CACHE[key_path])
        for row in rows:
            row["invention_error"] = int(row["clean_fpr"] > f0)
            row["deletion_error"] = int(bool(row["has_defect"]) and row["pixel_recall"] < r0)
        _LABEL_CACHE[key] = rows
    return _LABEL_CACHE[key]


def _memo_fit(base_fit):
    def fit(rows, seed, deletion_floor):
        key = (_STATE["domain"], id(rows), seed, float(deletion_floor), _STATE["f0"])
        if key not in _FIT_CACHE:
            _FIT_CACHE[key] = base_fit(rows, seed, deletion_floor)
        return _FIT_CACHE[key]
    return fit


_MEMO_FITS = {d: _memo_fit(f) for d, f in rta.DOMAIN_SCORE_FIT.items()}

# Patch the module-level names that the definitive code looks up at call time.
rta.read_rows = _relabelled_read_rows
reported.read_rows = _relabelled_read_rows


def set_domain(domain: str) -> None:
    reported.FEATURES = rta.DOMAIN_FEATURES[domain]
    reported.fit_risk_models = _MEMO_FITS[domain]
    _STATE["domain"] = domain


def make_targets(alpha: float, delta: float) -> DualRiskTargets:
    # tuning_margin and minimum-evidence requirement fixed at the prespecified values.
    return DualRiskTargets(deletion=alpha, invention=alpha, delta=delta,
                           tuning_margin=0.50, min_accept=15, min_positive_accept=15)


def analyze(domain: str, seed: int, r0: float, f0: float, alpha: float, delta: float) -> dict:
    """Full definitive analysis for one (domain, seed, cell); returns rta's result dict."""
    assert fc.candidate_sequence.__module__ == "familywise_certification"
    set_domain(domain)
    _STATE["r0"], _STATE["f0"] = float(r0), float(f0)
    folder = Path(str(DOMAINS[domain]).format(seed=seed))
    return rta.analyze_domain_seed(folder, seed, float(r0), make_targets(alpha, delta))


def summarize_seed(result: dict) -> dict:
    """Flat per-seed summary used by every WP2 table."""
    legacy = result["legacy_marginal"]["policies"]
    fam = result["familywise_single_policy"]
    pf = result["familywise_portfolio"]
    ts = result["familywise_tuning_selected"]
    out = {
        "tuning_n": result["tuning_n"],
        "holdout_n": result["certification_n"],
        "portfolio_selected": pf["selected_policy"],
        "portfolio_qualified": pf["selected_policy"] is not None,
    }
    sel_test = pf.get("selected_test") or {}
    out["portfolio_test_coverage_fi"] = float(sel_test.get("coverage", 0.0)) if pf["selected_policy"] else 0.0
    out["portfolio_test_deletion_risk"] = sel_test.get("deletion_risk")
    out["portfolio_test_invention_risk"] = sel_test.get("invention_risk")
    if pf["selected_policy"]:
        ev = pf["policies"][pf["selected_policy"]]
        out["portfolio_holdout_accepted"] = ev["invention_denominator"]
        out["portfolio_holdout_accepted_positive"] = ev["deletion_denominator"]
        out["portfolio_holdout_coverage"] = ev["coverage"]
    ts_cert = ts["certificate"]["certified_threshold"] is not None
    out["tuning_selected_policy"] = ts["selected_policy"]
    out["tuning_selected_qualified"] = ts_cert
    out["tuning_selected_test_coverage_fi"] = (
        float(ts["selected_test"]["coverage"]) if ts_cert else 0.0)
    for name in POLICIES:
        cert = legacy[name]["certificate"]
        out[f"marg_pass__{name}"] = bool(cert["passes"])
        out[f"marg_cov_fi__{name}"] = float(legacy[name]["test"]["deployable_coverage"])
        out[f"marg_test_del__{name}"] = legacy[name]["test"]["deletion_risk"] if cert["passes"] else None
        out[f"marg_test_inv__{name}"] = legacy[name]["test"]["invention_risk"] if cert["passes"] else None
        q = fam[name]["certified_threshold"] is not None
        out[f"fam_pass__{name}"] = q
        out[f"fam_cov_fi__{name}"] = float(fam[name]["test"]["coverage"]) if q else 0.0
        out[f"fam_test_del__{name}"] = fam[name]["test"]["deletion_risk"] if q else None
        out[f"fam_test_inv__{name}"] = fam[name]["test"]["invention_risk"] if q else None
        out[f"port_pass__{name}"] = pf["policies"][name]["certified_threshold"] is not None
    return out


def min_n(e: int, alpha: float, delta_end: float, n_max: int = 100000) -> int:
    """Smallest n with one-sided Clopper-Pearson upper bound (e of n, level delta_end) <= alpha."""
    n = e + 1
    while n <= n_max:
        if clopper_pearson_upper(e, n, delta_end) <= alpha:
            return n
        n += 1
    raise RuntimeError("n_max exceeded")
