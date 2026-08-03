from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CarinthiaSItem:
    sample_id: str
    image_path: Path
    mask_path: Path
    label: int


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"dataset path escapes root: {relative}") from exc
    return candidate


def load_carinthia_s(root: str | Path) -> list[CarinthiaSItem]:
    """Load and validate the public Carinthia-S inventory without editing it."""

    root_path = Path(root).resolve()
    csv_path = root_path / "carinthia-s.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    items: list[CarinthiaSItem] = []
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        expected = {"image_path", "mask_path", "filename", "label"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected Carinthia-S columns: {reader.fieldnames}")
        for row in reader:
            sample_id = row["filename"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"empty or duplicate sample id: {sample_id}")
            image = _safe_child(root_path, row["image_path"].strip())
            mask = _safe_child(root_path, row["mask_path"].strip())
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"missing pair for {sample_id}")
            items.append(CarinthiaSItem(sample_id, image, mask, int(row["label"])))
            seen.add(sample_id)
    if not items:
        raise ValueError("Carinthia-S inventory is empty")
    return items


def inventory_sha256(items: list[CarinthiaSItem]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.sample_id):
        line = f"{item.sample_id}|{item.image_path.name}|{item.mask_path.name}|{item.label}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest().upper()
