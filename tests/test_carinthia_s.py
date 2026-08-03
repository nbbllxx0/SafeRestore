from pathlib import Path

import pytest

from saferestore.datasets.carinthia_s import inventory_sha256, load_carinthia_s


def test_carinthia_inventory_pairs_images_and_masks(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"image")
    (tmp_path / "masks" / "a.png").write_bytes(b"mask")
    (tmp_path / "carinthia-s.csv").write_text(
        "image_path;mask_path;filename;label\nimages/a.jpg;masks/a.png;a;2\n",
        encoding="utf-8",
    )
    items = load_carinthia_s(tmp_path)
    assert len(items) == 1
    assert items[0].label == 2
    assert len(inventory_sha256(items)) == 64


def test_carinthia_inventory_blocks_path_escape(tmp_path: Path):
    (tmp_path / "carinthia-s.csv").write_text(
        "image_path;mask_path;filename;label\n../a.jpg;../a.png;a;2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_carinthia_s(tmp_path)
