import numpy as np

from saferestore import (
    DualRiskTargets,
    certify_threshold,
    choose_threshold,
    choose_threshold_empirical_target,
    choose_threshold_tuning_ucb,
    clopper_pearson_upper,
    evaluate_threshold,
    select_minimax_actions,
)


def test_deletion_uses_only_accepted_positive_denominator() -> None:
    scores = np.zeros(6)
    positive = np.array([1, 1, 0, 0, 0, 0])
    deletion = np.array([1, 0, 0, 0, 0, 0])
    invention = np.array([0, 0, 0, 1, 0, 0])
    result = evaluate_threshold(scores, 0.0, positive, deletion, invention)
    assert result["deletion_denominator"] == 2
    assert result["invention_denominator"] == 6
    assert result["deletion_risk"] == 0.5
    assert result["invention_risk"] == 1 / 6


def test_exact_bound_has_conservative_edge_cases() -> None:
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_upper(4, 4, 0.05) == 1.0
    assert 0.0 < clopper_pearson_upper(0, 100, 0.05) < 0.05


def test_tune_then_certify_with_separate_denominators() -> None:
    n = 200
    scores = np.linspace(0.0, 1.0, n)
    positive = np.arange(n) % 4 < 2
    deletion = np.zeros(n, dtype=np.int8)
    invention = np.zeros(n, dtype=np.int8)
    targets = DualRiskTargets(min_accept=20, min_positive_accept=10)
    threshold = choose_threshold(
        scores[::2], positive[::2], deletion[::2], invention[::2], targets
    )
    certificate = certify_threshold(
        scores[1::2],
        threshold,
        positive[1::2],
        deletion[1::2],
        invention[1::2],
        targets,
    )
    assert threshold > 0.0
    assert certificate["deletion_denominator"] > 0
    assert certificate["passes"]


def test_action_selection_includes_raw_in_the_same_pool() -> None:
    deletion = {
        "raw": np.array([0.01, 0.20]),
        "restored": np.array([0.10, 0.02]),
    }
    invention = {
        "raw": np.array([0.01, 0.01]),
        "restored": np.array([0.02, 0.02]),
    }
    selected, score = select_minimax_actions(
        deletion,
        invention,
        ("raw", "restored"),
        DualRiskTargets(),
    )
    assert selected.tolist() == ["raw", "restored"]
    assert np.all(np.isfinite(score))


def test_empirical_target_selector_uses_the_exact_target() -> None:
    scores = np.arange(20, dtype=float)
    positive = np.ones(20, dtype=np.int8)
    incidents = np.zeros(20, dtype=np.int8)
    incidents[-2:] = 1
    targets = DualRiskTargets(min_accept=15, min_positive_accept=15)

    conservative = choose_threshold(scores, positive, incidents, incidents, targets)
    at_target = choose_threshold_empirical_target(
        scores, positive, incidents, incidents, targets
    )

    assert conservative == 18.0
    assert at_target == 19.0


def test_tuning_ucb_selector_requires_enough_exact_evidence() -> None:
    scores = np.arange(25, dtype=float)
    positive = np.ones(25, dtype=np.int8)
    incidents = np.zeros(25, dtype=np.int8)
    targets = DualRiskTargets(min_accept=15, min_positive_accept=15)

    threshold = choose_threshold_tuning_ucb(
        scores, positive, incidents, incidents, targets
    )

    assert threshold == 24.0
    assert clopper_pearson_upper(0, 25, targets.delta / 2) <= targets.deletion

