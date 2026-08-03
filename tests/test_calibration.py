import numpy as np
import pytest

from saferestore.calibration import calibrate_sequential_router
from saferestore.router import route_sequential
from saferestore.types import Action, RiskTargets


def test_risk_targets_reject_invalid_values():
    with pytest.raises(ValueError):
        RiskTargets(deletion=0.0)
    with pytest.raises(ValueError):
        RiskTargets(min_accept=0)


def test_sequential_calibration_routes_safe_cases():
    n = 600
    restored_score = np.linspace(0.0, 1.0, n)
    raw_score = np.linspace(0.1, 1.1, n)
    restored_error = (restored_score > 0.7).astype(np.int8)
    raw_error = (raw_score > 0.9).astype(np.int8)
    scores = {Action.RESTORED: restored_score, Action.RAW: raw_score}
    deletion = {Action.RESTORED: restored_error, Action.RAW: raw_error}
    invention = {
        Action.RESTORED: np.zeros(n, dtype=np.int8),
        Action.RAW: np.zeros(n, dtype=np.int8),
    }
    result = calibrate_sequential_router(
        scores,
        deletion,
        invention,
        targets=RiskTargets(deletion=0.35, invention=0.20, delta=0.20, min_accept=40),
        seed=3,
    )
    decisions = route_sequential(scores, result)
    assert np.any(decisions == Action.RESTORED.value)
    assert np.all(np.isin(decisions, [a.value for a in Action]))


def test_calibration_rejects_too_small_dataset():
    scores = {Action.RESTORED: np.zeros(10), Action.RAW: np.zeros(10)}
    errors = {Action.RESTORED: np.zeros(10), Action.RAW: np.zeros(10)}
    with pytest.raises(ValueError):
        calibrate_sequential_router(
            scores,
            errors,
            errors,
            targets=RiskTargets(min_accept=6),
        )
