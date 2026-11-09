"""Familywise-valid selective-restoration certification (TIM upgrade).

Replaces the single tuned-gate marginal certificate with a
fixed-sequence-testing construction in the Learn-then-Test tradition
(Angelopoulos et al., 2021): the threshold-tuning role fixes an ordered
sequence of candidate gates; the disjoint certification role then tests the
gates in that order, each with two one-sided exact binomial bounds at joint
level ``delta_alloc``, stopping at the first failure.  Because the ordering is
measurable with respect to pre-certification information and testing stops at
the first non-rejection, the probability that any certified gate violates
either population target is at most ``delta_alloc`` (fixed-sequence FWER
control).  Selecting the largest certified gate is therefore valid — unlike
selecting among independently certified marginal gates.

A portfolio wrapper allocates ``delta`` across a prespecified policy family
by Bonferroni, runs one fixed sequence per policy, and selects the certified
policy with the largest certification-role coverage.  The selection is then
covered by the familywise statement: with probability at least
``1 - delta`` every certified (policy, gate) pair — including the selected
one — satisfies both population targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from saferestore.certification import DualRiskTargets, clopper_pearson_upper


@dataclass(frozen=True)
class FixedSequenceResult:
    certified_threshold: float | None
    certified_count: int
    tested_count: int
    sequence_start_threshold: float | None
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified_threshold": self.certified_threshold,
            "certified_count": self.certified_count,
            "tested_count": self.tested_count,
            "sequence_start_threshold": self.sequence_start_threshold,
            **self.evidence,
        }


def _counts_at_threshold(
    threshold: float,
    scores: np.ndarray,
    positive: np.ndarray,
    deletion: np.ndarray,
    invention: np.ndarray,
) -> tuple[int, int, int, int]:
    accepted = scores <= threshold
    accepted_positive = accepted & positive
    return (
        int(deletion[accepted_positive].sum()),
        int(accepted_positive.sum()),
        int(invention[accepted].sum()),
        int(accepted.sum()),
    )


def candidate_sequence(
    tuning_scores: np.ndarray,
    tuning_positive: np.ndarray,
    targets: DualRiskTargets,
    *,
    tuning_deletion: np.ndarray | None = None,
    tuning_invention: np.ndarray | None = None,
    certification_scale: float = 1.0,
) -> np.ndarray:
    """Candidate gates ordered by decreasing tuning-predicted pass margin.

    All quantities are computed on the tuning role only, so the ordering is
    measurable with respect to pre-certification information and
    fixed-sequence testing on the certification role remains valid.  Gates
    whose tuning prefix cannot meet the minimum-evidence requirement are
    excluded.  For each remaining gate the tuning empirical incident rates are
    combined with certification denominators predicted by
    ``certification_scale`` (the certification/tuning size ratio) to form
    anticipated exact upper bounds; gates are tested in order of decreasing
    anticipated margin, with coverage as tie-break, so the most promising
    gates are reached before any failure stops the sequence.  When incident
    arrays are unavailable the ordering degrades to ascending gate size.
    """

    order = np.argsort(tuning_scores, kind="stable")
    sorted_scores = tuning_scores[order]
    sorted_positive = tuning_positive[order]
    accepted = np.arange(1, len(sorted_scores) + 1)
    accepted_positive = np.cumsum(sorted_positive)
    eligible = (accepted >= targets.min_accept) & (
        accepted_positive >= targets.min_positive_accept
    )
    if not eligible.any():
        return np.empty(0, dtype=np.float64)
    start = int(np.argmax(eligible))
    thresholds = np.unique(sorted_scores[start:])
    if tuning_deletion is None or tuning_invention is None:
        return thresholds

    deletion = np.asarray(tuning_deletion).astype(bool)
    invention = np.asarray(tuning_invention).astype(bool)
    margins = np.empty(len(thresholds), dtype=np.float64)
    coverages = np.empty(len(thresholds), dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        mask = tuning_scores <= threshold
        mask_positive = mask & tuning_positive
        n_del = max(int(round(int(mask_positive.sum()) * certification_scale)), 1)
        n_inv = max(int(round(int(mask.sum()) * certification_scale)), 1)
        # Laplace-smoothed tuning rates: (e+1)/(n+2) guards the ordering
        # against optimistic small-sample tuning estimates, so the sequence
        # leads with genuinely safe gates (ordering affects power only).
        rate_del = (
            (float(deletion[mask_positive].sum()) + 1.0)
            / (int(mask_positive.sum()) + 2.0)
            if mask_positive.any()
            else 1.0
        )
        rate_inv = (float(invention[mask].sum()) + 1.0) / (int(mask.sum()) + 2.0)
        e_del = int(round(rate_del * n_del))
        e_inv = int(round(rate_inv * n_inv))
        anticipated_del = clopper_pearson_upper(min(e_del, n_del), n_del, targets.delta / 2)
        anticipated_inv = clopper_pearson_upper(min(e_inv, n_inv), n_inv, targets.delta / 2)
        margins[index] = min(
            targets.deletion - anticipated_del, targets.invention - anticipated_inv
        )
        coverages[index] = int(mask.sum()) / len(tuning_scores)
    ranking = np.lexsort((-coverages, -margins))
    return thresholds[ranking]


def fixed_sequence_certify(
    *,
    tuning_scores: np.ndarray,
    tuning_positive: np.ndarray,
    certification_scores: np.ndarray,
    certification_positive: np.ndarray,
    certification_deletion: np.ndarray,
    certification_invention: np.ndarray,
    targets: DualRiskTargets,
    delta_alloc: float,
    tuning_deletion: np.ndarray | None = None,
    tuning_invention: np.ndarray | None = None,
) -> FixedSequenceResult:
    """Fixed-sequence testing over tuning-ordered gates at joint level delta_alloc."""

    tuning_scores = np.asarray(tuning_scores, dtype=np.float64)
    tuning_positive = np.asarray(tuning_positive, dtype=bool)
    scores = np.asarray(certification_scores, dtype=np.float64)
    positive = np.asarray(certification_positive, dtype=bool)
    deletion = np.asarray(certification_deletion).astype(bool)
    invention = np.asarray(certification_invention).astype(bool)
    sequence = candidate_sequence(
        tuning_scores,
        tuning_positive,
        targets,
        tuning_deletion=tuning_deletion,
        tuning_invention=tuning_invention,
        certification_scale=(len(scores) / max(len(tuning_scores), 1)),
    )

    certified: float | None = None
    best_coverage = -1.0
    certified_evidence: dict[str, Any] = {}
    failure_evidence: dict[str, Any] = {}
    tested = 0
    for threshold in sequence:
        e_del, n_del, e_inv, n_inv = _counts_at_threshold(
            float(threshold), scores, positive, deletion, invention
        )
        deletion_ucb = clopper_pearson_upper(e_del, n_del, delta_alloc / 2)
        invention_ucb = clopper_pearson_upper(e_inv, n_inv, delta_alloc / 2)
        tested += 1
        if deletion_ucb <= targets.deletion and invention_ucb <= targets.invention:
            coverage = n_inv / len(scores) if len(scores) else 0.0
            if coverage > best_coverage:
                best_coverage = coverage
                certified = float(threshold)
                certified_evidence = {
                    "deletion_errors": e_del,
                    "deletion_denominator": n_del,
                    "deletion_ucb": deletion_ucb,
                    "invention_errors": e_inv,
                    "invention_denominator": n_inv,
                    "invention_ucb": invention_ucb,
                    "accepted": n_inv,
                    "coverage": coverage,
                    "delta_alloc": delta_alloc,
                }
            continue
        failure_evidence = {
            "threshold": float(threshold),
            "deletion_errors": e_del,
            "deletion_denominator": n_del,
            "deletion_ucb": deletion_ucb,
            "invention_errors": e_inv,
            "invention_denominator": n_inv,
            "invention_ucb": invention_ucb,
        }
        break
    evidence = dict(certified_evidence)
    if failure_evidence:
        evidence["first_failure"] = failure_evidence
    # Descriptive diagnostic only (uses certification outcomes at every gate,
    # so it is NOT a certificate): the smallest target level at which some
    # gate in the sequence would have both exact bounds pass.
    if sequence.size:
        best = float("inf")
        best_at = None
        for threshold in sequence:
            e_del, n_del, e_inv, n_inv = _counts_at_threshold(
                float(threshold), scores, positive, deletion, invention
            )
            worst = max(
                clopper_pearson_upper(e_del, n_del, delta_alloc / 2),
                clopper_pearson_upper(e_inv, n_inv, delta_alloc / 2),
            )
            if worst < best:
                best = worst
                best_at = float(threshold)
        evidence["min_certifiable_alpha_descriptive"] = best
        evidence["min_certifiable_alpha_threshold"] = best_at
    return FixedSequenceResult(
        certified_threshold=certified,
        certified_count=tested - (1 if failure_evidence else 0),
        tested_count=tested,
        sequence_start_threshold=float(sequence[0]) if sequence.size else None,
        evidence=evidence,
    )


def tuning_selected_certify(
    policies: Mapping[str, dict[str, np.ndarray]],
    tuning_mask: np.ndarray,
    targets: DualRiskTargets,
) -> dict[str, Any]:
    """Select one policy on the tuning role, then spend full delta on its sequence.

    The selection statistic is the tuning-role coverage of the largest score
    prefix whose empirical dual risks meet the stated targets (with the
    minimum-evidence requirement).  Because the selection uses only
    pre-certification information, the subsequent fixed-sequence test at full
    ``delta`` retains its familywise guarantee for the selected policy.
    """

    def tuning_promise(arrays: dict[str, np.ndarray]) -> float:
        scores = arrays["score"][tuning_mask]
        positive = arrays["positive"][tuning_mask]
        deletion = np.asarray(arrays["deletion"])[tuning_mask].astype(bool)
        invention = np.asarray(arrays["invention"])[tuning_mask].astype(bool)
        order = np.argsort(scores, kind="stable")
        best = 0.0
        for count in range(targets.min_accept, len(order) + 1):
            local = order[:count]
            local_positive = positive[local]
            if int(local_positive.sum()) < targets.min_positive_accept:
                continue
            deletion_risk = float(deletion[local][local_positive].mean())
            invention_risk = float(invention[local].mean())
            if deletion_risk <= targets.deletion and invention_risk <= targets.invention:
                best = count / len(order)
        return best

    names = sorted(policies)
    promises = {name: tuning_promise(policies[name]) for name in names}
    selected = max(names, key=lambda name: (promises[name], name))
    arrays = policies[selected]
    result = fixed_sequence_certify(
        tuning_scores=arrays["score"][tuning_mask],
        tuning_positive=arrays["positive"][tuning_mask],
        certification_scores=arrays["score"][~tuning_mask],
        certification_positive=arrays["positive"][~tuning_mask],
        certification_deletion=arrays["deletion"][~tuning_mask],
        certification_invention=arrays["invention"][~tuning_mask],
        targets=targets,
        delta_alloc=targets.delta,
        tuning_deletion=np.asarray(arrays["deletion"])[tuning_mask],
        tuning_invention=np.asarray(arrays["invention"])[tuning_mask],
    )
    return {
        "tuning_promise": promises,
        "selected_policy": selected,
        "certificate": result.to_dict(),
        "interpretation": (
            "Policy selection is measurable with respect to the tuning role; "
            "the fixed-sequence certificate for the selected policy therefore "
            "holds at the full joint level delta."
        ),
    }


def portfolio_certify(
    policies: Mapping[str, dict[str, np.ndarray]],
    tuning_mask: np.ndarray,
    targets: DualRiskTargets,
) -> dict[str, Any]:
    """Bonferroni-allocated fixed sequences across a prespecified policy family.

    ``policies`` maps policy name to arrays with keys ``score``, ``positive``,
    ``deletion``, ``invention`` over the full calibration role; ``tuning_mask``
    selects the tuning subset.  Returns per-policy results plus the valid
    max-coverage selection among certified policies.
    """

    names = sorted(policies)
    delta_alloc = targets.delta / len(names)
    results: dict[str, Any] = {}
    best_name: str | None = None
    best_coverage = -1.0
    for name in names:
        arrays = policies[name]
        tuning = tuning_mask
        certification = ~tuning_mask
        result = fixed_sequence_certify(
            tuning_scores=arrays["score"][tuning],
            tuning_positive=arrays["positive"][tuning],
            certification_scores=arrays["score"][certification],
            certification_positive=arrays["positive"][certification],
            certification_deletion=arrays["deletion"][certification],
            certification_invention=arrays["invention"][certification],
            targets=targets,
            delta_alloc=delta_alloc,
            tuning_deletion=np.asarray(arrays["deletion"])[tuning],
            tuning_invention=np.asarray(arrays["invention"])[tuning],
        )
        results[name] = result.to_dict()
        if result.certified_threshold is not None:
            coverage = result.evidence["coverage"]
            if coverage > best_coverage:
                best_coverage = coverage
                best_name = name
    return {
        "delta": targets.delta,
        "policy_family": names,
        "delta_alloc_per_policy": delta_alloc,
        "policies": results,
        "selected_policy": best_name,
        "selected_coverage": best_coverage if best_name else None,
        "selected_threshold": (
            results[best_name]["certified_threshold"] if best_name else None
        ),
        "interpretation": (
            "With probability at least 1 - delta over the certification sample, "
            "every certified (policy, gate) pair satisfies both population "
            "targets; selecting the max-coverage certified pair is therefore "
            "covered by the same statement."
        ),
    }
