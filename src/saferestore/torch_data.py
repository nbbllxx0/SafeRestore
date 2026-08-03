from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SIZES = {"carinthia_s": (256, 256), "kolektor_sdd": (256, 640)}


class PublicSegmentationDataset(Dataset):
    def __init__(
        self,
        *,
        dataset: str,
        root: str | Path,
        split_file: str | Path,
        split: str,
        augment: bool = False,
    ) -> None:
        if dataset not in SIZES:
            raise ValueError(f"unknown dataset: {dataset}")
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.size = SIZES[dataset]
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
        sample_id = record["sample_id"]
        if self.dataset == "carinthia_s":
            return self.root / "images" / f"{sample_id}.jpg", self.root / "masks" / f"{sample_id}.png"
        part, stem = sample_id.split("/", 1)
        return self.root / part / f"{stem}.jpg", self.root / part / f"{stem}_label.bmp"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path, mask_path = self._paths(record)
        with Image.open(image_path) as source:
            image = source.convert("L").resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(mask_path) as source:
            mask = source.convert("L").resize((self.size[1], self.size[0]), Image.Resampling.NEAREST)
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
