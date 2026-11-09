"""CPU tests for the WP6 confirmatory pipeline definitions (no sealed data read)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import common as C


def test_sealed_roles_refused_without_freeze():
    if C.MANIFEST.exists():
        pytest.skip("protocol already frozen")
    for role in ("tuning", "holdout"):
        with pytest.raises(PermissionError):
            C.build_views({role}, purpose="test")
    assert not C.ACCESS_LOG.exists() or "purpose=test" not in C.ACCESS_LOG.read_text(encoding="utf-8")


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        C.open_roles({"test"}, "test")


def test_split_hash_and_dev_views():
    views = C.build_views({"detector_val"}, purpose="test")
    assert len(views) == 1680 and {v["role"] for v in views} == {"detector_val"}
    units = C.select_units(views)
    assert len(units) == 140
    assert all(u["anomalous"] for u in units if u["kind"] == "NG")
    assert [u["sample_id"] for u in C.select_units(list(reversed(views)))] == [u["sample_id"] for u in units]


def test_clean_region_excludes_ring():
    fg = np.ones((64, 64), bool)
    mask = np.zeros((64, 64), bool)
    mask[30:33, 30:33] = True
    clean = C.clean_region(fg, mask)
    assert not clean[30 - C.TOLERANCE:33 + C.TOLERANCE, 30 - C.TOLERANCE:33 + C.TOLERANCE].any()
    assert clean[0, 0] and clean.sum() == 64 * 64 - (3 + 2 * C.TOLERANCE) ** 2


def test_actions_shape_and_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 120, size=(32, 48, 3), dtype=np.uint8)
    photo = {"gain": 2.0, "gamma": 0.6}
    assert C.apply_classical("raw", img, photo) is img
    for m in ("gain", "gamma", "clahe", "gain_unsharp"):
        out = C.apply_classical(m, img, photo)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert np.allclose(C.apply_classical("gain", img, photo), np.clip(img * 2.0 + 0.5, 0, 255).astype(np.uint8), atol=1)
    # on a smooth image, unsharp masking keeps the gained image almost unchanged
    ramp = np.repeat(np.linspace(10, 100, 48, dtype=np.float32)[None, :, None], 32, axis=0)
    ramp = np.repeat(ramp, 3, axis=2).astype(np.uint8)
    gain, sharp = C.apply_classical("gain", ramp, photo), C.apply_classical("gain_unsharp", ramp, photo)
    assert np.abs(sharp[:, 4:-4].astype(int) - gain[:, 4:-4].astype(int)).max() <= 2


def test_record_row_endpoints():
    shape = (64, 64)
    mask = np.zeros(shape, bool)
    mask[10:14, 10:14] = True
    fg = np.ones(shape, bool)
    clean = C.clean_region(fg, mask)
    prob = np.zeros(shape, np.float32)
    prob[10:12, 10:14] = 0.9            # half the defect detected -> recall 0.5
    prob[50:60, 50:60] = 0.9            # 100 clean false positives
    v = {"sample_id": "A/NG/1/v", "specimen": "A/NG/1", "cls": "A", "role": "detector_val",
         "detectable": {"07": True}}
    img = np.zeros(shape + (3,), np.uint8)
    row = C.record_row(v, "raw", img, img, prob, prob, 0.5, mask, clean)
    assert row["has_defect"] == 1 and row["pixel_recall"] == pytest.approx(0.5)
    assert row["deletion_error"] == 0
    assert row["clean_fpr"] == pytest.approx(100 / clean.sum()) and row["invention_error"] == 1
    assert row["active_components"] == 2.0 and row["secondary_peak"] == pytest.approx(0.9)
    assert set(C.FEATURES) <= set(row)


def _synthetic_rows(n_units: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        positive = i % 2 == 0
        hard = rng.random()
        for k, m in enumerate(C.METHODS):
            risk = hard * (1.0 - 0.1 * k)
            row = {"sample_id": f"S/{i:04d}/v", "specimen": f"S/{i:04d}", "role": "holdout",
                   "category": C.CATEGORIES[i % len(C.CATEGORIES)],
                   "method": m, "has_defect": int(positive),
                   "deletion_error": int(positive and rng.random() < 0.3 * risk),
                   "invention_error": int(rng.random() < 0.2 * risk * (k / 5)),
                   "pixel_recall": 0.5, "clean_fpr": 0.0}
            for j, f in enumerate(C.FEATURES):
                row[f] = float(risk + 0.3 * rng.normal() + 0.01 * j)
            rows.append(row)
    return rows


def test_analysis_end_to_end_synthetic():
    import analyze as A

    targets = A.DualRiskTargets(deletion=0.15, invention=0.15, delta=0.10, min_accept=15, min_positive_accept=15)
    models = A.fit_models(_synthetic_rows(400, 1), seed=1)
    qual = A.policy_arrays(_synthetic_rows(600, 2), models, targets)
    test = A.policy_arrays(_synthetic_rows(200, 3), models, targets)
    tuning_mask = np.arange(600) % 10 < 3
    res = A.readings(qual, tuning_mask, test, targets)
    assert set(res["single_policy"]) == set(C.POLICIES)
    comp = A.comparators(qual, tuning_mask, test, targets, res["tuning_selected"]["selected_policy"])
    assert comp["fs"]["selected_policy"] == res["portfolio"]["selected_policy"]
    if comp["fs"]["selected_policy"] is not None:
        assert comp["fs"]["threshold"] == res["portfolio"]["selected_threshold"]
    assert set(comp) == set(A.COMPARATORS)
