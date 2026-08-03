from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

from .degradation import degrade_batch


@dataclass(frozen=True)
class DegradationLevel:
    name: str
    blur_sigma: float
    noise_std: float


LEVELS = (
    DegradationLevel("mild", blur_sigma=0.6, noise_std=0.010),
    DegradationLevel("moderate", blur_sigma=1.2, noise_std=0.025),
    DegradationLevel("severe", blur_sigma=2.0, noise_std=0.045),
)

# Default evaluation schedule for confirmatory protocol v3. Model-fitting seeds
# must not enter this key; otherwise severity changes across "repetitions."
DEFAULT_CORRUPTION_KEY = "v3corr:eval"


def level_for_sample(
    sample_id: str,
    *,
    split: str,
    corruption_key: str = DEFAULT_CORRUPTION_KEY,
) -> DegradationLevel:
    digest = hashlib.sha256(
        f"{corruption_key}:{split}:{sample_id}".encode("utf-8")
    ).digest()
    return LEVELS[int.from_bytes(digest[:8], "big") % len(LEVELS)]


def degrade_mixed_batch(
    high_resolution: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    split: str,
    corruption_key: str = DEFAULT_CORRUPTION_KEY,
    training_draw_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Apply a deterministic severity schedule.

    Evaluation uses ``corruption_key`` and sample identity only. When
    ``training_draw_seed`` is set (restorer augmentation), an additional draw
    salt varies noise within the same severity without changing the evaluation
    schedule used for certification.
    """

    if len(sample_ids) != high_resolution.shape[0]:
        raise ValueError("sample_ids must match the batch dimension")
    lows: list[torch.Tensor] = []
    bicubics: list[torch.Tensor] = []
    names: list[str] = []
    for index, sample_id in enumerate(sample_ids):
        level = level_for_sample(
            sample_id, split=split, corruption_key=corruption_key
        )
        if training_draw_seed is None:
            noise_salt = f"noise:{corruption_key}:{split}:{sample_id}"
        else:
            noise_salt = (
                f"noise:{corruption_key}:{split}:{sample_id}:{training_draw_seed}"
            )
        digest = hashlib.sha256(noise_salt.encode("utf-8")).digest()
        generator = torch.Generator(device=high_resolution.device).manual_seed(
            int.from_bytes(digest[:8], "big")
        )
        low, bicubic = degrade_batch(
            high_resolution[index : index + 1],
            blur_sigma=level.blur_sigma,
            noise_std=level.noise_std,
            generator=generator,
        )
        lows.append(low)
        bicubics.append(bicubic)
        names.append(level.name)
    return torch.cat(lows, dim=0), torch.cat(bicubics, dim=0), names
