import math

from scripts.evaluate_shift_router import describe


def test_describe_ignores_undefined_endpoint_values() -> None:
    summary = describe([None, 0.25, float("nan"), 0.75])
    assert summary["mean"] == 0.5
    assert math.isclose(summary["sd"], math.sqrt(0.125))
    assert summary["min"] == 0.25
    assert summary["max"] == 0.75


def test_describe_returns_nulls_when_endpoint_is_never_observed() -> None:
    assert describe([None, float("nan")]) == {
        "mean": None,
        "sd": None,
        "min": None,
        "max": None,
    }
