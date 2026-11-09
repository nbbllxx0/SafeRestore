"""WP3 data access: frozen records -> per-policy arrays for the tuning,
hold-out and test roles, exactly as the published analysis builds them.

Import pattern copied from tim_rr/wp1/ordering_variants.py (monkeypatching
reported.FEATURES / reported.fit_risk_models per domain).  Read-only on
tim_upgrade/, experiments/, scripts/, src/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src", REPO / "tim_upgrade" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import familywise_certification as fc  # noqa: E402
import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
from saferestore.certification import DualRiskTargets, choose_threshold  # noqa: E402
from scripts.analyze_router_pool import read_rows  # noqa: E402

POLICIES = tuple(reported.POLICIES)  # published order
POLICY_NAMES_SORTED = tuple(sorted(POLICIES))  # portfolio iteration order in fc
SEEDS = tuple(rta.SEEDS)
FLOOR = 0.25
TARGETS = DualRiskTargets(deletion=0.15, invention=0.15, delta=0.10,
                          tuning_margin=0.50, min_accept=15, min_positive_accept=15)
DOMAINS = {
    "carinthia_s": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
    "kolektor_sdd2": REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed{seed}",
}
SHORT = {"raw_nearest": "raw", "bilinear": "bil", "bicubic": "bic",
         "smooth_bicubic": "smooth", "unsharp_bicubic": "sharp", "learned": "learned",
         "restoration_pool": "resto", "action_pool": "all"}


@dataclass
class PolicyData:
    """One policy's arrays. Keys: score, positive, deletion, invention (bool/float)."""
    tune: dict
    hold: dict
    test: dict
    grid: np.ndarray = field(default=None)       # eligible tuning gates, ascending
    order: np.ndarray = field(default=None)      # FS order: indices into grid
    tuned_threshold: float = float("-inf")       # legacy marginal tuned gate
    tuning_promise: float = 0.0                  # tuning-selected statistic


@dataclass
class SeedData:
    domain: str
    seed: int
    policies: dict  # name -> PolicyData
    test_full: dict  # name -> full test arrays incl. method (for evaluate_policy)
    hold_ids: np.ndarray = None
    test_ids: np.ndarray = None


def _as_arrays(policy: dict, mask: np.ndarray | None = None) -> dict:
    out = {
        "score": np.asarray(policy["score"], dtype=np.float64),
        "positive": np.asarray(policy["positive"], dtype=bool),
        "deletion": np.asarray(policy["deletion"]).astype(bool),
        "invention": np.asarray(policy["invention"]).astype(bool),
    }
    if mask is not None:
        out = {k: v[mask] for k, v in out.items()}
    return out


def set_domain(domain: str) -> None:
    reported.FEATURES = rta.DOMAIN_FEATURES[domain]
    reported.fit_risk_models = rta.DOMAIN_SCORE_FIT[domain]


def tuning_promise(arr: dict, targets: DualRiskTargets) -> float:
    """Copy of fc.tuning_selected_certify's inner statistic (tuning role only)."""
    scores, positive = arr["score"], arr["positive"]
    deletion, invention = arr["deletion"], arr["invention"]
    order = np.argsort(scores, kind="stable")
    best = 0.0
    for count in range(targets.min_accept, len(order) + 1):
        local = order[:count]
        lp = positive[local]
        if int(lp.sum()) < targets.min_positive_accept:
            continue
        if (float(deletion[local][lp].mean()) <= targets.deletion
                and float(invention[local].mean()) <= targets.invention):
            best = count / len(order)
    return best


def load_seed(domain: str, seed: int, targets: DualRiskTargets = TARGETS,
              floor: float = FLOOR) -> SeedData:
    set_domain(domain)
    folder = Path(str(DOMAINS[domain]).format(seed=seed))
    training_rows = read_rows(folder / "detector_val_records.csv")
    calibration_rows = read_rows(folder / "calibration_records.csv")
    test_rows = read_rows(folder / "test_records.csv")
    models = reported.fit_risk_models(training_rows, seed, floor)
    calibration = reported.build_policy_arrays(calibration_rows, models, targets, floor)
    test = reported.build_policy_arrays(test_rows, models, targets, floor)
    tuning_mask = reported.stable_tuning_mask(
        calibration["action_pool"]["sample_id"].tolist(), tune_key="v4tunecert")
    n_tune, n_hold = int(tuning_mask.sum()), int((~tuning_mask).sum())
    pols = {}
    for name in POLICIES:
        tune = _as_arrays(calibration[name], tuning_mask)
        hold = _as_arrays(calibration[name], ~tuning_mask)
        tst = _as_arrays(test[name])
        grid = np.sort(fc.candidate_sequence(tune["score"], tune["positive"], targets))
        seq = fc.candidate_sequence(
            tune["score"], tune["positive"], targets,
            tuning_deletion=tune["deletion"], tuning_invention=tune["invention"],
            certification_scale=n_hold / max(n_tune, 1))
        order = np.searchsorted(grid, seq)
        assert np.array_equal(grid[order], seq)
        tuned = choose_threshold(tune["score"], tune["positive"], tune["deletion"],
                                 tune["invention"], targets)
        pols[name] = PolicyData(tune=tune, hold=hold, test=tst, grid=grid, order=order,
                                tuned_threshold=float(tuned),
                                tuning_promise=tuning_promise(tune, targets))
    return SeedData(domain=domain, seed=seed, policies=pols, test_full=test,
                    hold_ids=np.asarray(calibration["action_pool"]["sample_id"])[~tuning_mask],
                    test_ids=np.asarray(test["action_pool"]["sample_id"]))


def gate_counts(arr: dict, thresholds: np.ndarray) -> dict:
    """Counts at each threshold: e_del, n_del (accepted positives), e_act, n_act."""
    s = arr["score"]
    acc = s[None, :] <= np.asarray(thresholds)[:, None]
    pos = arr["positive"][None, :]
    return {
        "e_del": (acc & pos & arr["deletion"][None, :]).sum(1),
        "n_del": (acc & pos).sum(1),
        "e_act": (acc & arr["invention"][None, :]).sum(1),
        "n_act": acc.sum(1),
    }
