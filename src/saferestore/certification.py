from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class DualRiskTargets:
    """Targets for a fixed policy certified on an independent i.i.d. sample."""

    deletion: float = 0.15
    invention: float = 0.15
    delta: float = 0.10
    tuning_margin: float = 0.50
    min_accept: int = 15
    min_positive_accept: int = 15

    def __post_init__(self) -> None:
        for name in ("deletion", "invention", "delta", "tuning_margin"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")
        if self.min_accept < 1 or self.min_positive_accept < 1:
            raise ValueError("minimum accepted counts must be positive")


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isin(array, (0, 1, False, True))):
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=False)


def _validated_arrays(
    scores: np.ndarray,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be a finite one-dimensional array")
    positive_array = _binary(positive, "positive")
    deletion_array = _binary(deletion_errors, "deletion_errors")
    invention_array = _binary(invention_errors, "invention_errors")
    if not (
        len(score_array)
        == len(positive_array)
        == len(deletion_array)
        == len(invention_array)
    ):
        raise ValueError("policy arrays must have equal length")
    if np.any(deletion_array & ~positive_array):
        raise ValueError("deletion incidents are defined only for positive images")
    return score_array, positive_array, deletion_array, invention_array


def clopper_pearson_upper(errors: int, count: int, delta: float) -> float:
    """One-sided exact binomial upper confidence bound."""

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if count <= 0 or errors >= count:
        return 1.0
    if errors < 0 or errors > count:
        raise ValueError("errors must lie between zero and count")
    return float(beta.ppf(1.0 - delta, errors + 1, count - errors))


def evaluate_threshold(
    scores: np.ndarray,
    threshold: float,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    *,
    prevalences: tuple[float, ...] = (0.01, 0.10, 0.50),
) -> dict[str, Any]:
    """Evaluate a fixed gate with population-specific risk denominators."""

    scores, positive, deletion_errors, invention_errors = _validated_arrays(
        scores, positive, deletion_errors, invention_errors
    )
    accepted = scores <= threshold
    accepted_positive = accepted & positive
    accepted_clean = accepted & ~positive
    n = len(scores)
    accepted_n = int(accepted.sum())
    positive_n = int(positive.sum())
    clean_n = int((~positive).sum())
    deletion_n = int(accepted_positive.sum())
    invention_n = accepted_n
    positive_coverage = deletion_n / positive_n if positive_n else None
    clean_coverage = int(accepted_clean.sum()) / clean_n if clean_n else None
    standardized = {
        f"{prevalence:.2f}": (
            prevalence * positive_coverage
            + (1.0 - prevalence) * clean_coverage
            if positive_coverage is not None and clean_coverage is not None
            else None
        )
        for prevalence in prevalences
    }
    deletion_count = int(deletion_errors[accepted_positive].sum())
    invention_count = int(invention_errors[accepted].sum())
    return {
        "n": n,
        "accepted": accepted_n,
        "reviewed": n - accepted_n,
        "coverage": accepted_n / n if n else 0.0,
        "positive_n": positive_n,
        "clean_n": clean_n,
        "accepted_positive": deletion_n,
        "accepted_clean": int(accepted_clean.sum()),
        "positive_coverage": positive_coverage,
        "clean_coverage": clean_coverage,
        "prevalence_standardized_coverage": standardized,
        "deletion_denominator": deletion_n,
        "invention_denominator": invention_n,
        "deletion_errors": deletion_count,
        "invention_errors": invention_count,
        "deletion_risk": deletion_count / deletion_n if deletion_n else None,
        "invention_risk": invention_count / invention_n if invention_n else None,
    }


def choose_threshold(
    scores: np.ndarray,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    targets: DualRiskTargets,
) -> float:
    """Choose the largest tuning prefix satisfying conservative risk margins."""

    scores, positive, deletion_errors, invention_errors = _validated_arrays(
        scores, positive, deletion_errors, invention_errors
    )
    order = np.argsort(scores, kind="stable")
    best = float("-inf")
    for count in range(targets.min_accept, len(order) + 1):
        local = order[:count]
        local_positive = positive[local]
        positive_count = int(local_positive.sum())
        if positive_count < targets.min_positive_accept:
            continue
        deletion_risk = float(deletion_errors[local][local_positive].mean())
        invention_risk = float(invention_errors[local].mean())
        if (
            deletion_risk <= targets.deletion * targets.tuning_margin
            and invention_risk <= targets.invention * targets.tuning_margin
        ):
            best = float(scores[order[count - 1]])
    return best


def choose_threshold_empirical_target(
    scores: np.ndarray,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    targets: DualRiskTargets,
) -> float:
    """Choose the largest tuning prefix whose empirical risks meet the targets."""

    scores, positive, deletion_errors, invention_errors = _validated_arrays(
        scores, positive, deletion_errors, invention_errors
    )
    order = np.argsort(scores, kind="stable")
    best = float("-inf")
    for count in range(targets.min_accept, len(order) + 1):
        local = order[:count]
        local_positive = positive[local]
        positive_count = int(local_positive.sum())
        if positive_count < targets.min_positive_accept:
            continue
        deletion_risk = float(deletion_errors[local][local_positive].mean())
        invention_risk = float(invention_errors[local].mean())
        if deletion_risk <= targets.deletion and invention_risk <= targets.invention:
            best = float(scores[order[count - 1]])
    return best


def choose_threshold_tuning_ucb(
    scores: np.ndarray,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    targets: DualRiskTargets,
) -> float:
    """Choose the largest tuning prefix whose two exact upper bounds meet target."""

    scores, positive, deletion_errors, invention_errors = _validated_arrays(
        scores, positive, deletion_errors, invention_errors
    )
    order = np.argsort(scores, kind="stable")
    best = float("-inf")
    for count in range(targets.min_accept, len(order) + 1):
        local = order[:count]
        local_positive = positive[local]
        positive_count = int(local_positive.sum())
        if positive_count < targets.min_positive_accept:
            continue
        deletion_count = int(deletion_errors[local][local_positive].sum())
        invention_count = int(invention_errors[local].sum())
        deletion_ucb = clopper_pearson_upper(
            deletion_count, positive_count, targets.delta / 2
        )
        invention_ucb = clopper_pearson_upper(
            invention_count, count, targets.delta / 2
        )
        if deletion_ucb <= targets.deletion and invention_ucb <= targets.invention:
            best = float(scores[order[count - 1]])
    return best


def certify_threshold(
    scores: np.ndarray,
    threshold: float,
    positive: np.ndarray,
    deletion_errors: np.ndarray,
    invention_errors: np.ndarray,
    targets: DualRiskTargets,
) -> dict[str, Any]:
    """Certify one threshold fixed before observing this i.i.d. sample."""

    result = evaluate_threshold(
        scores, threshold, positive, deletion_errors, invention_errors
    )
    result["deletion_ucb"] = clopper_pearson_upper(
        result["deletion_errors"], result["deletion_denominator"], targets.delta / 2
    )
    result["invention_ucb"] = clopper_pearson_upper(
        result["invention_errors"], result["invention_denominator"], targets.delta / 2
    )
    result["passes"] = bool(
        result["deletion_ucb"] <= targets.deletion
        and result["invention_ucb"] <= targets.invention
    )
    return result


def select_minimax_actions(
    deletion_probabilities: Mapping[str, np.ndarray],
    invention_probabilities: Mapping[str, np.ndarray],
    methods: tuple[str, ...],
    targets: DualRiskTargets,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the action minimizing the larger target-normalized risk score."""

    if not methods:
        raise ValueError("methods cannot be empty")
    columns = []
    expected_shape: tuple[int, ...] | None = None
    for method in methods:
        deletion = np.asarray(deletion_probabilities[method], dtype=np.float64)
        invention = np.asarray(invention_probabilities[method], dtype=np.float64)
        if deletion.ndim != 1 or deletion.shape != invention.shape:
            raise ValueError(f"incompatible probability arrays for {method}")
        if expected_shape is None:
            expected_shape = deletion.shape
        elif deletion.shape != expected_shape:
            raise ValueError("candidate actions must share the same sample ordering")
        columns.append(
            np.maximum(deletion / targets.deletion, invention / targets.invention)
        )
    matrix = np.stack(columns, axis=1)
    selected_index = matrix.argmin(axis=1)
    selected_method = np.asarray(methods, dtype=object)[selected_index]
    selected_score = matrix[np.arange(len(selected_index)), selected_index]
    return selected_method, selected_score
