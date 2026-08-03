"""Legacy two-action sequential prototype.

This module is retained only to reproduce historical development artifacts.
The manuscript algorithm and public API are implemented in
``saferestore.certification``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import Action, RiskTargets


@dataclass(frozen=True)
class ActionCalibration:
    action: Action
    threshold: float
    calibration_count: int
    accepted_count: int
    deletion_errors: int
    invention_errors: int
    deletion_ucb: float
    invention_ucb: float
    fold: int


@dataclass(frozen=True)
class SequentialCalibration:
    order: tuple[Action, ...]
    actions: Mapping[Action, ActionCalibration]
    targets: RiskTargets
    seed: int
    calibration_size: int


def _as_binary(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isin(arr, (0, 1, False, True))):
        raise ValueError(f"{name} must be binary")
    return arr.astype(np.int8, copy=False)


def _hoeffding_ucb(errors: int, count: int, delta: float) -> float:
    if count <= 0:
        return 1.0
    empirical = errors / count
    radius = np.sqrt(np.log(1.0 / delta) / (2.0 * count))
    return float(min(1.0, empirical + radius))


def _calibrate_action(
    *,
    action: Action,
    scores: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    eligible: np.ndarray,
    targets: RiskTargets,
    fold: int,
    simultaneous_slots: int,
) -> ActionCalibration:
    local_scores = np.asarray(scores, dtype=np.float64)[eligible]
    local_del = _as_binary(deletion_errors, "deletion_errors")[eligible]
    local_inv = _as_binary(invention_errors, "invention_errors")[eligible]
    calibration_count = int(local_scores.size)

    if calibration_count == 0:
        return ActionCalibration(action, float("-inf"), 0, 0, 0, 0, 1.0, 1.0, fold)
    if not np.all(np.isfinite(local_scores)):
        raise ValueError(f"non-finite safety score for action {action.value}")

    candidates = np.unique(local_scores)
    # Two risks and every candidate threshold are covered by a union bound.
    per_test_delta = targets.delta / max(1, simultaneous_slots * 2 * len(candidates))
    best: ActionCalibration | None = None

    for threshold in candidates:
        accepted = local_scores <= threshold
        count = int(accepted.sum())
        if count < targets.min_accept:
            continue
        d_err = int(local_del[accepted].sum())
        i_err = int(local_inv[accepted].sum())
        d_ucb = _hoeffding_ucb(d_err, count, per_test_delta)
        i_ucb = _hoeffding_ucb(i_err, count, per_test_delta)
        if d_ucb <= targets.deletion and i_ucb <= targets.invention:
            candidate = ActionCalibration(
                action=action,
                threshold=float(threshold),
                calibration_count=calibration_count,
                accepted_count=count,
                deletion_errors=d_err,
                invention_errors=i_err,
                deletion_ucb=d_ucb,
                invention_ucb=i_ucb,
                fold=fold,
            )
            if best is None or candidate.accepted_count > best.accepted_count:
                best = candidate

    if best is not None:
        return best
    return ActionCalibration(
        action=action,
        threshold=float("-inf"),
        calibration_count=calibration_count,
        accepted_count=0,
        deletion_errors=0,
        invention_errors=0,
        deletion_ucb=1.0,
        invention_ucb=1.0,
        fold=fold,
    )


def calibrate_sequential_router(
    scores: Mapping[Action, np.ndarray],
    deletion_errors: Mapping[Action, np.ndarray],
    invention_errors: Mapping[Action, np.ndarray],
    *,
    order: Sequence[Action] = (Action.RESTORED, Action.RAW),
    targets: RiskTargets = RiskTargets(),
    seed: int = 0,
) -> SequentialCalibration:
    """Calibrate a predeclared sequential routing policy on disjoint folds.

    Each action is calibrated on its own fold. Later-action folds are filtered
    by already-fixed earlier thresholds, matching the deployed routing rule.
    The resulting high-probability bounds are a conservative prototype under
    exchangeable calibration/evaluation sampling; arbitrary shift is outside
    the guarantee.
    """

    order_tuple = tuple(Action(a) for a in order)
    if not order_tuple or len(set(order_tuple)) != len(order_tuple):
        raise ValueError("order must contain unique non-abstain actions")
    if Action.ABSTAIN in order_tuple:
        raise ValueError("abstain is the implicit terminal action")

    first = np.asarray(scores[order_tuple[0]])
    if first.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    n = int(first.size)
    if n < len(order_tuple) * targets.min_accept:
        raise ValueError("calibration set is too small for the requested folds")

    for action in order_tuple:
        for collection, label in (
            (scores, "scores"),
            (deletion_errors, "deletion_errors"),
            (invention_errors, "invention_errors"),
        ):
            arr = np.asarray(collection[action])
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"{label}[{action.value}] has incompatible shape")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    folds = np.empty(n, dtype=np.int32)
    folds[permutation] = np.arange(n, dtype=np.int32) % len(order_tuple)

    calibrated: dict[Action, ActionCalibration] = {}
    for fold, action in enumerate(order_tuple):
        eligible = folds == fold
        # Earlier actions use thresholds learned on independent folds.
        for prior in order_tuple[:fold]:
            eligible &= np.asarray(scores[prior], dtype=np.float64) > calibrated[prior].threshold
        calibrated[action] = _calibrate_action(
            action=action,
            scores=np.asarray(scores[action]),
            deletion_errors=np.asarray(deletion_errors[action]),
            invention_errors=np.asarray(invention_errors[action]),
            eligible=eligible,
            targets=targets,
            fold=fold,
            simultaneous_slots=len(order_tuple),
        )

    return SequentialCalibration(
        order=order_tuple,
        actions=calibrated,
        targets=targets,
        seed=seed,
        calibration_size=n,
    )
