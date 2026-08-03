import torch

from saferestore.degradation_schedule import (
    DEFAULT_CORRUPTION_KEY,
    degrade_mixed_batch,
    level_for_sample,
)


def test_level_assignment_is_stable_across_model_seeds():
    first = level_for_sample("sample-1", split="test")
    again = level_for_sample("sample-1", split="test")
    other_key = level_for_sample(
        "sample-1", split="test", corruption_key="v3corr:A"
    )
    shifted = level_for_sample("sample-1", split="calibration")
    assert first == again
    assert first.name in {"mild", "moderate", "severe"}
    assert shifted.name in {"mild", "moderate", "severe"}
    # Different corruption keys may differ; default key is seed-independent.
    assert level_for_sample("sample-1", split="test", corruption_key=DEFAULT_CORRUPTION_KEY) == first
    assert other_key.name in {"mild", "moderate", "severe"}


def test_mixed_degradation_preserves_batch_shapes():
    high = torch.rand(3, 1, 32, 48)
    low, bicubic, names = degrade_mixed_batch(
        high,
        ["a", "b", "c"],
        split="test",
    )
    assert low.shape == (3, 1, 16, 24)
    assert bicubic.shape == high.shape
    assert len(names) == 3
    assert torch.isfinite(low).all()
    assert torch.isfinite(bicubic).all()


def test_evaluation_severity_does_not_depend_on_training_draw():
    high = torch.rand(2, 1, 32, 48)
    ids = ["x", "y"]
    _, _, names_a = degrade_mixed_batch(high, ids, split="test")
    _, _, names_b = degrade_mixed_batch(
        high, ids, split="test", training_draw_seed=999
    )
    assert names_a == names_b
