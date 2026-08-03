from __future__ import annotations

from typing import Mapping

import numpy as np

from .types import Action


def _flatten_samples(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have sample and spatial dimensions")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr.reshape(arr.shape[0], -1)


def quantile_evidence_scores(
    raw_probability: np.ndarray,
    restored_probability: np.ndarray,
    *,
    restoration_ensemble: np.ndarray | None = None,
    data_consistency: np.ndarray | None = None,
    quantile: float = 0.95,
) -> dict[str, np.ndarray]:
    """Construct model-agnostic deletion/invention proxy scores.

    These are ranking features, not calibrated probabilities. Ground-truth
    errors on an independent calibration set are required before routing.
    """

    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0.5, 1]")
    raw = _flatten_samples(raw_probability, "raw_probability")
    restored = _flatten_samples(restored_probability, "restored_probability")
    if raw.shape != restored.shape:
        raise ValueError("raw and restored probability maps must match")

    deletion = np.quantile(np.maximum(raw - restored, 0.0), quantile, axis=1)
    invention = np.quantile(np.maximum(restored - raw, 0.0), quantile, axis=1)

    if restoration_ensemble is None:
        disagreement = np.zeros(raw.shape[0], dtype=np.float64)
    else:
        ensemble = np.asarray(restoration_ensemble, dtype=np.float64)
        if ensemble.ndim < 3 or ensemble.shape[0] != raw.shape[0]:
            raise ValueError("ensemble must have shape [sample, member, ...]")
        member_flat = ensemble.reshape(ensemble.shape[0], ensemble.shape[1], -1)
        if member_flat.shape[2] != raw.shape[1]:
            raise ValueError("ensemble spatial shape must match probability maps")
        disagreement = np.quantile(np.std(member_flat, axis=1), quantile, axis=1)

    if data_consistency is None:
        consistency = np.zeros(raw.shape[0], dtype=np.float64)
    else:
        consistency = np.asarray(data_consistency, dtype=np.float64)
        if consistency.ndim != 1 or consistency.size != raw.shape[0]:
            raise ValueError("data_consistency must be one-dimensional per sample")
        if np.any(consistency < 0) or not np.all(np.isfinite(consistency)):
            raise ValueError("data_consistency must be finite and nonnegative")

    joint = np.maximum(deletion, invention) + 0.5 * disagreement + 0.5 * consistency
    return {
        "deletion_proxy": deletion,
        "invention_proxy": invention,
        "disagreement": disagreement,
        "data_consistency": consistency,
        "joint_proxy": joint,
    }


def build_action_scores(
    raw_uncertainty: np.ndarray,
    restoration_features: Mapping[str, np.ndarray],
) -> dict[Action, np.ndarray]:
    raw = np.asarray(raw_uncertainty, dtype=np.float64)
    restored = np.asarray(restoration_features["joint_proxy"], dtype=np.float64)
    if raw.ndim != 1 or raw.shape != restored.shape:
        raise ValueError("raw and restored scores must be matching vectors")
    return {Action.RAW: raw, Action.RESTORED: restored}
