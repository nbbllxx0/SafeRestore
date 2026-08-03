import numpy as np

from saferestore.scores import build_action_scores, quantile_evidence_scores
from saferestore.types import Action


def test_proxy_scores_detect_deletion_and_invention():
    raw = np.array([[[0.9, 0.1]], [[0.1, 0.1]]])
    restored = np.array([[[0.2, 0.1]], [[0.1, 0.8]]])
    features = quantile_evidence_scores(raw, restored, quantile=1.0)
    assert features["deletion_proxy"][0] > 0.0
    assert features["invention_proxy"][1] > 0.0
    action_scores = build_action_scores(np.array([0.2, 0.3]), features)
    assert set(action_scores) == {Action.RAW, Action.RESTORED}
