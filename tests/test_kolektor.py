from pathlib import Path
import pickle

import pytest

from saferestore.datasets.kolektor import load_kolektor, load_official_splits


def test_kolektor_inventory_pairs_files(tmp_path: Path):
    part = tmp_path / "kos01"
    part.mkdir()
    (part / "Part0.jpg").write_bytes(b"image")
    (part / "Part0_label.bmp").write_bytes(b"mask")
    items = load_kolektor(tmp_path)
    assert len(items) == 1
    assert items[0].part_id == "kos01"


def test_data_only_unpickler_loads_lists(tmp_path: Path):
    split = ((["kos01"], ["kos02"], ["kos03"]), (["kos02", "kos03"], ["kos01", "kos03"], ["kos01", "kos02"]), ["kos01", "kos02", "kos03"])
    path = tmp_path / "split.pyb"
    path.write_bytes(pickle.dumps(split, protocol=0))
    train, test, all_parts = load_official_splits(path)
    assert train[0] == ("kos01",)
    assert set(all_parts) == {"kos01", "kos02", "kos03"}


def test_data_only_unpickler_rejects_globals(tmp_path: Path):
    path = tmp_path / "bad.pyb"
    path.write_bytes(pickle.dumps(Path("x"), protocol=0))
    with pytest.raises(pickle.UnpicklingError):
        load_official_splits(path)
