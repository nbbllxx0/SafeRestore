"""Unit tests for the fixed-sequence familywise certification module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tim_upgrade" / "scripts"))

from saferestore.certification import DualRiskTargets, clopper_pearson_upper
from familywise_certification import (
    candidate_sequence,
    fixed_sequence_certify,
    portfolio_certify,
    tuning_selected_certify,
)

TARGETS = DualRiskTargets(
    deletion=0.15, invention=0.15, delta=0.10,
    tuning_margin=0.50, min_accept=15, min_positive_accept=15,
)


def _clean_policy(n: int = 400, incident_rate: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=n)
    positive = np.ones(n, dtype=bool)
    positive[::4] = False
    deletion = np.zeros(n, dtype=np.int8)
    invention = np.zeros(n, dtype=np.int8)
    if incident_rate > 0:
        hit = rng.uniform(size=n) < incident_rate
        deletion[hit & positive] = 1
        invention[hit] = 1
    return {
        "score": scores,
        "positive": positive,
        "deletion": deletion,
        "invention": invention,
    }


def _mask(n: int = 400) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[: n // 3] = True
    return mask


def test_candidate_sequence_respects_minimum_evidence():
    scores = np.linspace(0.0, 1.0, 40)
    positive = np.ones(40, dtype=bool)
    sequence = candidate_sequence(scores, positive, TARGETS)
    # First 14 prefixes cannot satisfy min_accept=15.
    assert sequence.size == 40 - 14
    assert sequence[0] == pytest.approx(scores[14])


def test_candidate_sequence_empty_when_underpowered():
    scores = np.linspace(0.0, 1.0, 10)
    positive = np.ones(10, dtype=bool)
    assert candidate_sequence(scores, positive, TARGETS).size == 0


def test_zero_incident_policy_certifies_full_coverage():
    arrays = _clean_policy()
    mask = _mask()
    result = fixed_sequence_certify(
        tuning_scores=arrays["score"][mask],
        tuning_positive=arrays["positive"][mask],
        certification_scores=arrays["score"][~mask],
        certification_positive=arrays["positive"][~mask],
        certification_deletion=arrays["deletion"][~mask],
        certification_invention=arrays["invention"][~mask],
        targets=TARGETS,
        delta_alloc=TARGETS.delta,
    )
    assert result.certified_threshold is not None
    assert result.evidence["coverage"] == pytest.approx(1.0)
    assert result.evidence["deletion_ucb"] <= TARGETS.deletion
    assert result.evidence["invention_ucb"] <= TARGETS.invention


def test_high_risk_policy_certifies_nothing():
    arrays = _clean_policy(incident_rate=0.6, seed=3)
    mask = _mask()
    result = fixed_sequence_certify(
        tuning_scores=arrays["score"][mask],
        tuning_positive=arrays["positive"][mask],
        certification_scores=arrays["score"][~mask],
        certification_positive=arrays["positive"][~mask],
        certification_deletion=arrays["deletion"][~mask],
        certification_invention=arrays["invention"][~mask],
        targets=TARGETS,
        delta_alloc=TARGETS.delta,
    )
    assert result.certified_threshold is None
    assert "first_failure" in result.evidence


def test_sequence_stops_at_first_failure():
    """Incidents concentrated at high scores: certificates stop before them."""

    n = 600
    scores = np.linspace(0.0, 1.0, n)
    positive = np.ones(n, dtype=bool)
    deletion = np.zeros(n, dtype=np.int8)
    invention = np.zeros(n, dtype=np.int8)
    deletion[scores > 0.7] = 1
    invention[scores > 0.7] = 1
    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    result = fixed_sequence_certify(
        tuning_scores=scores[mask],
        tuning_positive=positive[mask],
        certification_scores=scores[~mask],
        certification_positive=positive[~mask],
        certification_deletion=deletion[~mask],
        certification_invention=invention[~mask],
        targets=TARGETS,
        delta_alloc=TARGETS.delta,
        tuning_deletion=deletion[mask],
        tuning_invention=invention[mask],
    )
    assert result.certified_threshold is not None
    assert result.certified_threshold < 0.8
    accepted = scores[~mask] <= result.certified_threshold
    empirical = deletion[~mask][accepted].mean()
    assert empirical <= TARGETS.deletion


def test_certified_gate_bounds_match_direct_computation():
    arrays = _clean_policy(incident_rate=0.05, seed=11)
    mask = _mask()
    result = fixed_sequence_certify(
        tuning_scores=arrays["score"][mask],
        tuning_positive=arrays["positive"][mask],
        certification_scores=arrays["score"][~mask],
        certification_positive=arrays["positive"][~mask],
        certification_deletion=arrays["deletion"][~mask],
        certification_invention=arrays["invention"][~mask],
        targets=TARGETS,
        delta_alloc=TARGETS.delta,
        tuning_deletion=arrays["deletion"][mask],
        tuning_invention=arrays["invention"][mask],
    )
    assert result.certified_threshold is not None
    scores = arrays["score"][~mask]
    positive = arrays["positive"][~mask]
    accepted = scores <= result.certified_threshold
    e_del = int(arrays["deletion"][~mask][accepted & positive].sum())
    n_del = int((accepted & positive).sum())
    expected = clopper_pearson_upper(e_del, n_del, TARGETS.delta / 2)
    assert result.evidence["deletion_ucb"] == pytest.approx(expected)


def test_portfolio_uses_bonferroni_allocation():
    policies = {
        "clean": _clean_policy(seed=1),
        "risky": _clean_policy(incident_rate=0.5, seed=2),
    }
    mask = _mask()
    result = portfolio_certify(policies, mask, TARGETS)
    assert result["delta_alloc_per_policy"] == pytest.approx(TARGETS.delta / 2)
    assert result["selected_policy"] == "clean"
    assert result["policies"]["risky"]["certified_threshold"] is None


def test_tuning_selection_prefers_low_risk_policy():
    policies = {
        "clean": _clean_policy(seed=5),
        "risky": _clean_policy(incident_rate=0.5, seed=6),
    }
    mask = _mask()
    result = tuning_selected_certify(policies, mask, TARGETS)
    assert result["selected_policy"] == "clean"
    assert result["certificate"]["certified_threshold"] is not None
    assert result["certificate"]["delta_alloc"] == pytest.approx(TARGETS.delta)

