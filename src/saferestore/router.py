from __future__ import annotations

from typing import Mapping

import numpy as np

from .calibration import SequentialCalibration
from .types import Action


def route_sequential(
    scores: Mapping[Action, np.ndarray],
    calibration: SequentialCalibration,
) -> np.ndarray:
    """Apply restored/raw/abstain routing in the calibrated order."""

    first = np.asarray(scores[calibration.order[0]], dtype=np.float64)
    if first.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    n = first.size
    decisions = np.full(n, Action.ABSTAIN.value, dtype=object)
    remaining = np.ones(n, dtype=bool)

    for action in calibration.order:
        current = np.asarray(scores[action], dtype=np.float64)
        if current.shape != first.shape:
            raise ValueError(f"score shape mismatch for {action.value}")
        threshold = calibration.actions[action].threshold
        accept = remaining & (current <= threshold)
        decisions[accept] = action.value
        remaining[accept] = False

    return decisions
