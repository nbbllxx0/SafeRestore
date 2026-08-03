from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .types import Action


@dataclass(frozen=True)
class RoutingMetrics:
    sample_count: int
    accepted_count: int
    coverage: float
    review_burden: float
    deletion_risk: float
    invention_risk: float
    joint_risk: float
    raw_count: int
    restored_count: int
    abstain_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "accepted_count": self.accepted_count,
            "coverage": self.coverage,
            "review_burden": self.review_burden,
            "deletion_risk": self.deletion_risk,
            "invention_risk": self.invention_risk,
            "joint_risk": self.joint_risk,
            "raw_count": self.raw_count,
            "restored_count": self.restored_count,
            "abstain_count": self.abstain_count,
        }


def evaluate_routing(
    decisions: np.ndarray,
    deletion_errors: Mapping[Action, np.ndarray],
    invention_errors: Mapping[Action, np.ndarray],
) -> RoutingMetrics:
    decisions = np.asarray(decisions, dtype=object)
    if decisions.ndim != 1:
        raise ValueError("decisions must be one-dimensional")
    n = decisions.size
    valid = {action.value for action in Action}
    if any(value not in valid for value in decisions):
        raise ValueError("decisions contain an unknown action")

    accepted = decisions != Action.ABSTAIN.value
    selected_deletion = np.zeros(n, dtype=np.int8)
    selected_invention = np.zeros(n, dtype=np.int8)

    for action in (Action.RAW, Action.RESTORED):
        d = np.asarray(deletion_errors[action], dtype=np.int8)
        i = np.asarray(invention_errors[action], dtype=np.int8)
        if d.shape != (n,) or i.shape != (n,):
            raise ValueError(f"error shape mismatch for {action.value}")
        mask = decisions == action.value
        selected_deletion[mask] = d[mask]
        selected_invention[mask] = i[mask]

    accepted_count = int(accepted.sum())
    if accepted_count:
        deletion_risk = float(selected_deletion[accepted].mean())
        invention_risk = float(selected_invention[accepted].mean())
        joint_risk = float(np.maximum(selected_deletion, selected_invention)[accepted].mean())
    else:
        deletion_risk = invention_risk = joint_risk = float("nan")

    raw_count = int(np.sum(decisions == Action.RAW.value))
    restored_count = int(np.sum(decisions == Action.RESTORED.value))
    abstain_count = int(np.sum(decisions == Action.ABSTAIN.value))
    coverage = accepted_count / n if n else 0.0

    return RoutingMetrics(
        sample_count=n,
        accepted_count=accepted_count,
        coverage=coverage,
        review_burden=1.0 - coverage,
        deletion_risk=deletion_risk,
        invention_risk=invention_risk,
        joint_risk=joint_risk,
        raw_count=raw_count,
        restored_count=restored_count,
        abstain_count=abstain_count,
    )
