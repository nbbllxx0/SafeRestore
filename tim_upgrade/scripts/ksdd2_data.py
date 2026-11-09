"""KolektorSDD2 inventory and torch dataset for the TIM upgrade.

Lives entirely under tim_upgrade/; imports but never modifies the frozen
`saferestore` library.  KolektorSDD2 layout after official extraction:

    dataset/train/{id}.png + {id}_GT.png
    dataset/test/{id}.png + {id}_GT.png

Each image is one product surface; the public metadata expose no coarser
grouping unit, so one image is the sampling unit.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# (height, width) after resize; KSDD2 sources are approximately 630x230
# portrait, so a 640x256 canvas preserves aspect while staying divisible by 8.
KSDD2_SIZE = (640, 256)


@dataclass(frozen=True)
class Ksdd2Item:
    sample_id: str  # "train/10429" or "test/20101"
    image_path: Path
    mask_path: Path
    has_defect: bool


def load_ksdd2(root: str | Path, *, measure_positivity: bool = True) -> list[Ksdd2Item]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    items: list[Ksdd2Item] = []
    for part in ("train", "test"):
        part_dir = root_path / part
        if not part_dir.is_dir():
            raise FileNotFoundError(part_dir)
        for image in sorted(part_dir.glob("*.png")):
            if image.stem.endswith("_GT"):
                continue
            # The official archive ships literal "(copy)" duplicates of
            # train/10301; exclude them so each surface appears once.
            if "(copy)" in image.stem:
                continue
            mask = image.with_name(f"{image.stem}_GT.png")
            if not mask.is_file():
                raise FileNotFoundError(mask)
            has_defect = False
            if measure_positivity:
                with Image.open(mask) as source:
                    has_defect = bool(np.any(np.asarray(source.convert("L")) > 0))
            items.append(
                Ksdd2Item(
                    sample_id=f"{part}/{image.stem}",
                    image_path=image,
                    mask_path=mask,
                    has_defect=has_defect,
                )
            )
    if not items:
        raise ValueError("KolektorSDD2 inventory is empty")
    return items


def inventory_sha256(items: list[Ksdd2Item]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda value: value.sample_id):
        digest.update(
            f"{item.sample_id}|{item.image_path.name}|{item.mask_path.name}|{int(item.has_defect)}\n".encode()
        )
    return digest.hexdigest().upper()


class Ksdd2SegmentationDataset(Dataset):
    """Drop-in replacement for PublicSegmentationDataset (same record API)."""

    def __init__(
        self,
        *,
        dataset: str,
        root: str | Path,
        split_file: str | Path,
        split: str,
        augment: bool = False,
    ) -> None:
        if dataset != "kolektor_sdd2":
            raise ValueError(f"unexpected dataset for KSDD2 loader: {dataset}")
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.size = KSDD2_SIZE
        self.augment = augment
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if payload["dataset"] != dataset:
            raise ValueError("split file does not match dataset")
        self.records = [record for record in payload["records"] if record["split"] == split]
        if not self.records:
            raise ValueError(f"no records for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def _paths(self, record: dict[str, Any]) -> tuple[Path, Path]:
        part, stem = record["sample_id"].split("/", 1)
        folder = self.root / part
        return folder / f"{stem}.png", folder / f"{stem}_GT.png"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path, mask_path = self._paths(record)
        with Image.open(image_path) as source:
            image = source.convert("L").resize(
                (self.size[1], self.size[0]), Image.Resampling.BILINEAR
            )
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(mask_path) as source:
            mask = source.convert("L").resize(
                (self.size[1], self.size[0]), Image.Resampling.NEAREST
            )
            mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32, copy=False).copy()
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        if self.augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-1,))
                mask_tensor = torch.flip(mask_tensor, dims=(-1,))
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-2,))
                mask_tensor = torch.flip(mask_tensor, dims=(-2,))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": record["sample_id"],
            "split": record["split"],
        }
