import numpy as np

from saferestore.metrics import evaluate_routing
from saferestore.types import Action


def test_metrics_use_selected_action_errors_only():
    decisions = np.array([Action.RESTORED.value, Action.RAW.value, Action.ABSTAIN.value], dtype=object)
    deletion = {
        Action.RESTORED: np.array([1, 1, 1]),
        Action.RAW: np.array([0, 0, 1]),
    }
    invention = {
        Action.RESTORED: np.array([0, 1, 1]),
        Action.RAW: np.array([1, 1, 1]),
    }
    result = evaluate_routing(decisions, deletion, invention)
    assert result.accepted_count == 2
    assert result.coverage == 2 / 3
    assert result.deletion_risk == 0.5
    assert result.invention_risk == 0.5
    assert result.abstain_count == 1
