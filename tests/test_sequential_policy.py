from __future__ import annotations

import numpy as np

from scripts.evaluate_sequential_router import (
    certify_policy,
    choose_raw_fallback_threshold,
    evaluate_policy,
)


def action(score, deletion, invention, method):
    return {
        "score": np.asarray(score, dtype=float),
        "deletion": np.asarray(deletion, dtype=np.int8),
        "invention": np.asarray(invention, dtype=np.int8),
        "method": np.full(len(score), method, dtype=object),
    }


def test_sequential_policy_uses_raw_only_after_pool_rejects() -> None:
    pool = action(
        [0.1, 0.8, 0.9, 0.2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        "restored",
    )
    raw = action(
        [0.9, 0.2, 0.8, 0.1],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        "raw",
    )
    result = evaluate_policy(pool, raw, pool_threshold=0.3, raw_threshold=0.3)
    assert result["pool_accepted"] == 2
    assert result["raw_accepted"] == 1
    assert result["reviewed"] == 1
    assert result["accepted"] == 3
    assert result["deletion_errors"] == 0
    assert result["invention_errors"] == 0


def test_raw_fallback_threshold_respects_combined_tuning_margin() -> None:
    pool = action(
        [0.1, 0.2, 0.8, 0.9],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        "restored",
    )
    raw = action(
        [0.9, 0.9, 0.1, 0.2],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        "raw",
    )
    threshold = choose_raw_fallback_threshold(
        pool,
        raw,
        pool_threshold=0.3,
        deletion_target=0.5,
        invention_target=0.5,
        margin=0.5,
    )
    assert threshold == 0.2
    result = evaluate_policy(pool, raw, 0.3, threshold)
    assert result["coverage"] == 1.0
    assert result["deletion_risk"] == 0.25


def test_joint_certificate_bounds_the_combined_policy() -> None:
    pool = action(
        np.linspace(0.0, 1.0, 100),
        np.zeros(100),
        np.zeros(100),
        "restored",
    )
    raw = action(
        np.linspace(1.0, 0.0, 100),
        np.zeros(100),
        np.zeros(100),
        "raw",
    )
    result = certify_policy(
        pool,
        raw,
        pool_threshold=0.50,
        raw_threshold=0.50,
        delta=0.10,
        deletion_target=0.15,
        invention_target=0.15,
    )
    assert result["accepted"] == 100
    assert result["passes"]
    assert result["deletion_ucb"] < 0.15
    assert result["invention_ucb"] < 0.15

