from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """Available inspection actions."""

    RESTORED = "restored"
    RAW = "raw"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RiskTargets:
    """Predeclared accepted-case risk targets.

    The statistical prototype assumes exchangeable calibration and evaluation
    examples. It does not guarantee control under arbitrary distribution shift.
    """

    deletion: float = 0.15
    invention: float = 0.08
    delta: float = 0.10
    min_accept: int = 50

    def __post_init__(self) -> None:
        for name in ("deletion", "invention", "delta"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")
        if self.min_accept < 1:
            raise ValueError("min_accept must be positive")
