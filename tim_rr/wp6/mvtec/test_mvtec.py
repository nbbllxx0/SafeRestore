"""Tests for the MVTec AD data layer: frozen split integrity and cache equivalence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mvtec_data  # noqa: E402
import run_mvtec  # noqa: E402

SPLIT = HERE / "splits" / "mvtec_rr_v1.json"


def test_split_hash_matches_record():
    recorded = (HERE / "splits" / "mvtec_rr_v1.sha256.txt").read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == recorded


def test_split_roles_disjoint_and_balanced():
    records = json.loads(SPLIT.read_text(encoding="utf-8"))["records"]
    assert len({r["sample_id"] for r in records}) == len(records)
    for role in ("calibration", "test"):
        part = [r for r in records if r["split"] == role]
        assert sum(r["has_defect"] for r in part) * 2 == len(part)


@pytest.mark.skipif(not run_mvtec.DATA_ROOT.is_dir(), reason="MVTec AD not available")
def test_cached_items_equal_direct_decoding():
    ds = mvtec_data.MvtecSegmentationDataset(dataset="mvtec_ad", root=run_mvtec.DATA_ROOT, split_file=SPLIT,
                                             split="detector_val")
    for index in (0, 1, len(ds) - 1):
        record = ds.records[index]
        with Image.open(ds.root / record["image"]) as source:
            image = source.convert("L").resize((ds.size[1], ds.size[0]), Image.Resampling.BILINEAR)
            direct = np.asarray(image, dtype=np.float32).copy() / 255.0
        for _ in range(2):  # first call fills the cache, second reads it
            item = ds[index]
            assert np.array_equal(item["image"][0].numpy(), direct)
        if record["mask"] is not None:
            with Image.open(ds.root / record["mask"]) as source:
                mask = source.convert("L").resize((ds.size[1], ds.size[0]), Image.Resampling.NEAREST)
            assert np.array_equal(item["mask"][0].numpy(), (np.asarray(mask) > 0).astype(np.float32))


def test_sealed_roles_refused():
    run_mvtec.install_data_source(run_mvtec.DEV_ROLES)
    with pytest.raises(PermissionError):
        run_mvtec.k.Ksdd2SegmentationDataset(dataset="kolektor_sdd2", root=run_mvtec.DATA_ROOT,
                                             split_file=SPLIT, split="calibration")
