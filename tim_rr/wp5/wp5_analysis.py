"""WP5 analysis harness: the definitive TIM analysis imported unchanged.

Same outside-import pattern as tim_rr/wp1/ordering_variants.py and
tim_rr/wp2/wp2_common.py.  Read-only on tim_upgrade/, scripts/, src/.

Two optional hooks, both off by default (default = definitive analysis):

* ``label="strict"`` - at read time each row's ``invention_error`` is replaced
  by the ``strict_invention_error`` column written by wp5_eval (calls in the
  excluded tolerance ring count as clean-area activation).  Score models are
  refitted on, and gates qualified against, the strict labels.
* ``methods=(...)`` - restricts the action set (native-acquisition arm, where
  bilinear / bicubic coincide with the raw image).  The policy family becomes
  ``methods + (restoration_pool, action_pool)`` and the portfolio Bonferroni
  allocation is delta / len(family), as in portfolio_certify.
"""

from __future__ import annotations

import copy
import json
import math
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
from saferestore.certification import DualRiskTargets  # noqa: E402

DEFINITIVE_DIR = REPO / "tim_upgrade" / "experiments" / "tim_analysis"
ORIG_METHODS = tuple(reported.METHODS)
ORIG_POLICIES = tuple(reported.POLICIES)
_STATE = {"label": "standard"}
_RAW_CACHE: dict[str, list] = {}


def _read_rows(path):
    key = str(Path(path).resolve())
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = _orig_read_rows(Path(path))
    rows = _RAW_CACHE[key]
    if _STATE["label"] == "standard":
        return copy.deepcopy(rows)
    if _STATE["label"] == "strict":
        out = copy.deepcopy(rows)
        for row in out:
            row["invention_error"] = int(row["strict_invention_error"])
        return out
    raise ValueError(_STATE["label"])


rta.read_rows = _read_rows
reported.read_rows = _read_rows


def clear_cache() -> None:
    _RAW_CACHE.clear()


def set_domain(domain: str, methods: tuple[str, ...] | None = None) -> None:
    methods = tuple(methods) if methods else ORIG_METHODS
    policies = methods + ("restoration_pool", "action_pool")
    reported.METHODS = methods
    reported.RESTORATION_METHODS = methods[1:]
    reported.POLICIES = policies
    rta.POLICIES = policies
    reported.FEATURES = rta.DOMAIN_FEATURES[domain]
    reported.fit_risk_models = rta.DOMAIN_SCORE_FIT[domain]


def targets(alpha: float = 0.15, delta: float = 0.10) -> DualRiskTargets:
    return DualRiskTargets(deletion=alpha, invention=alpha, delta=delta,
                           tuning_margin=0.50, min_accept=15, min_positive_accept=15)


def analyze(folder: Path, domain: str, seed: int, *, floor: float = 0.25, alpha: float = 0.15,
            delta: float = 0.10, label: str = "standard", methods=None) -> dict:
    assert fc.candidate_sequence.__module__ == "familywise_certification"
    set_domain(domain, methods)
    _STATE["label"] = label
    try:
        return rta.analyze_domain_seed(Path(folder), seed, floor, targets(alpha, delta))
    finally:
        _STATE["label"] = "standard"
        set_domain(domain, None)


def json_safe(value):
    return reported.json_safe(value)


def alpha_star(result: dict) -> tuple[float | None, str | None]:
    """Seed-level descriptive alpha*: min over single-policy sequences (as tab_ksdd2_frontier)."""
    best_name, best = None, None
    for name, payload in result["familywise_single_policy"].items():
        value = payload.get("min_certifiable_alpha_descriptive")
        if value is None:
            continue
        if best is None or value < best:
            best, best_name = value, name
    return best, best_name


def alpha_star_ties(result: dict, tol: float = 1e-12) -> list[str]:
    best, _ = alpha_star(result)
    if best is None:
        return []
    return [n for n, p in result["familywise_single_policy"].items()
            if p.get("min_certifiable_alpha_descriptive") is not None
            and abs(p["min_certifiable_alpha_descriptive"] - best) <= tol]


def summarize(result: dict) -> dict:
    pf = result["familywise_portfolio"]
    single = result["familywise_single_policy"]
    a_star, a_name = alpha_star(result)
    out = {
        "tuning_n": result["tuning_n"],
        "holdout_n": result["certification_n"],
        "portfolio_selected": pf["selected_policy"],
        "portfolio_qualified": int(pf["selected_policy"] is not None),
        "portfolio_delta_alloc": pf["delta_alloc_per_policy"],
        "single_qualified_policies": ";".join(n for n, p in single.items()
                                              if p["certified_threshold"] is not None),
        "n_single_qualified": sum(p["certified_threshold"] is not None for p in single.values()),
        "alpha_star": a_star,
        "alpha_star_policy": a_name,
        "alpha_star_ties": ";".join(alpha_star_ties(result)),
        "tuning_selected_policy": result["familywise_tuning_selected"]["selected_policy"],
        "tuning_selected_qualified": int(
            result["familywise_tuning_selected"]["certificate"]["certified_threshold"] is not None),
    }
    sel = pf.get("selected_test") or {}
    out["portfolio_test_coverage"] = sel.get("coverage")
    out["portfolio_test_loss"] = sel.get("deletion_risk")
    out["portfolio_test_activation"] = sel.get("invention_risk")
    if pf["selected_policy"]:
        ev = pf["policies"][pf["selected_policy"]]
        out["portfolio_holdout_coverage"] = ev["coverage"]
        out["portfolio_holdout_loss"] = f"{ev['deletion_errors']}/{ev['deletion_denominator']}"
        out["portfolio_holdout_activation"] = f"{ev['invention_errors']}/{ev['invention_denominator']}"
    for name, p in single.items():
        out[f"single_q__{name}"] = int(p["certified_threshold"] is not None)
        out[f"alpha_desc__{name}"] = p.get("min_certifiable_alpha_descriptive")
        out[f"marg_pass__{name}"] = int(bool(result["legacy_marginal"]["policies"][name]["certificate"]["passes"]))
    return out


def diff(a, b, path="", out=None):
    """Exact field-by-field comparison (floats compared with ==)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def compare_with_definitive(result: dict, domain: str, seed: int, floor: float = 0.25) -> list[str]:
    ref = json.loads((DEFINITIVE_DIR / f"{domain}_seed{seed}_floor{floor:.2f}.json").read_text(encoding="utf-8"))
    mine = json.loads(json.dumps(json_safe(result), allow_nan=False))
    return diff(mine, ref)
