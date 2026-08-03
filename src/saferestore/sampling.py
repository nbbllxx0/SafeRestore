from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from .torch_data import PublicSegmentationDataset


def defect_balanced_sampler(
    dataset: PublicSegmentationDataset,
    *,
    seed: int,
    positive_weight: float = 8.0,
) -> WeightedRandomSampler | None:
    """Oversample defect-bearing images for sparse-positive training sets."""

    if dataset.dataset != "kolektor_sdd":
        return None
    weights: list[float] = []
    for record in dataset.records:
        _, mask_path = dataset._paths(record)
        with Image.open(mask_path) as source:
            positive = bool(np.asarray(source.convert("L"), dtype=np.uint8).any())
        weights.append(positive_weight if positive else 1.0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
