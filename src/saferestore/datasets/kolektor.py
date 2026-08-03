from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class KolektorItem:
    sample_id: str
    part_id: str
    image_path: Path
    mask_path: Path
    has_defect: bool


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"global object forbidden in split file: {module}.{name}")


def load_official_splits(path: str | Path):
    """Load the official protocol-0 data-only split without allowing globals."""
    with Path(path).open("rb") as stream:
        value = _DataOnlyUnpickler(stream).load()
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("unexpected Kolektor split structure")
    train, test, all_parts = value
    if len(train) != 3 or len(test) != 3:
        raise ValueError("expected three official folds")
    return tuple(tuple(str(x) for x in fold) for fold in train), tuple(
        tuple(str(x) for x in fold) for fold in test
    ), tuple(str(x) for x in all_parts)


def load_kolektor(root: str | Path) -> list[KolektorItem]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    items: list[KolektorItem] = []
    for part_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for image in sorted(part_dir.glob("Part*.jpg")):
            mask = image.with_name(f"{image.stem}_label.bmp")
            if not mask.is_file():
                raise FileNotFoundError(mask)
            # Dataset masks are uncompressed BMP; any non-zero byte after the
            # header is insufficient, so defer exact positivity to the audit.
            items.append(
                KolektorItem(
                    sample_id=f"{part_dir.name}/{image.stem}",
                    part_id=part_dir.name,
                    image_path=image,
                    mask_path=mask,
                    has_defect=False,
                )
            )
    if not items:
        raise ValueError("Kolektor inventory is empty")
    return items


def inventory_sha256(items: list[KolektorItem]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.sample_id):
        digest.update(f"{item.sample_id}|{item.image_path.name}|{item.mask_path.name}\n".encode())
    return digest.hexdigest().upper()
