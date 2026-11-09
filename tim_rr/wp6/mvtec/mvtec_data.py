"""MVTec AD inventory, frozen specimen split, and torch dataset for the WP6 study.

Layout after official extraction (root = extracted folder):
    <category>/train/good/<id>.png
    <category>/test/<defect type | good>/<id>.png
    <category>/ground_truth/<defect type>/<id>_mask.png

Every image shows a different physical object, so one image is the unit.
The dataset class is a drop-in replacement for tim_upgrade's
Ksdd2SegmentationDataset (same constructor and record API), so the paper's
KolektorSDD2 pipeline runs unchanged except for the data source.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CATEGORIES = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
SIZE = (512, 512)  # (height, width); every category is square at 700-1024 px


def inventory(root: Path) -> list[dict[str, Any]]:
    """One record per image: sample_id, category, has_defect, image and mask paths (relative)."""

    root = Path(root)
    records = []
    for cat in CATEGORIES:
        for part in ("train", "test"):
            for image in sorted((root / cat / part).glob("*/*.png")):
                kind = image.parent.name
                mask = None
                if kind != "good":
                    mask = root / cat / "ground_truth" / kind / f"{image.stem}_mask.png"
                    if not mask.is_file():
                        raise FileNotFoundError(mask)
                records.append({
                    "sample_id": f"{cat}/{part}/{kind}/{image.stem}",
                    "category": cat,
                    "has_defect": int(kind != "good"),
                    "image": image.relative_to(root).as_posix(),
                    "mask": None if mask is None else mask.relative_to(root).as_posix(),
                })
    if not records:
        raise ValueError(f"empty MVTec AD inventory under {root}")
    return records


def rank_key(sample_id: str) -> str:
    return hashlib.sha256(f"mvtec-rr-v1:{sample_id}".encode()).hexdigest()


def split_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen role assignment, stratified by category.

    Defective images, ranked by rank_key within each category: train 30 %,
    detector_val 15 %, calibration 40 %, test 15 % (cut points round half to
    even).  Defect-free images (train/good and test/good pooled), ranked the
    same way: calibration and test each receive as many as the category's
    defective images in that role, so both are 50 % positive; of the rest,
    15 % go to detector_val and 85 % to train.  "calibration" is the
    qualification split, later subdivided into tuning and hold-out by the
    paper's seed-independent hash rule.
    """

    out = []
    for cat in CATEGORIES:
        pos = sorted((r for r in records if r["category"] == cat and r["has_defect"]),
                     key=lambda r: rank_key(r["sample_id"]))
        neg = sorted((r for r in records if r["category"] == cat and not r["has_defect"]),
                     key=lambda r: rank_key(r["sample_id"]))
        n = len(pos)
        c1, c2, c3 = (round(n * f) for f in (0.30, 0.45, 0.85))
        roles = ["train"] * c1 + ["detector_val"] * (c2 - c1) + ["calibration"] * (c3 - c2) + ["test"] * (n - c3)
        for r, role in zip(pos, roles):
            out.append({**r, "split": role})
        n_cal, n_test = c3 - c2, n - c3
        if len(neg) < n_cal + n_test + 2:
            raise ValueError(f"{cat}: too few defect-free images")
        rest = neg[n_cal + n_test:]
        n_dv = round(len(rest) * 0.15)
        neg_roles = (["calibration"] * n_cal + ["test"] * n_test + ["detector_val"] * n_dv
                     + ["train"] * (len(rest) - n_dv))
        for r, role in zip(neg, neg_roles):
            out.append({**r, "split": role})
    return sorted(out, key=lambda r: r["sample_id"])


class MvtecSegmentationDataset(Dataset):
    """Same constructor and item format as Ksdd2SegmentationDataset (grayscale, resized)."""

    def __init__(self, *, dataset: str, root: str | Path, split_file: str | Path, split: str,
                 augment: bool = False) -> None:
        self.dataset = dataset
        self.root = Path(root).resolve()
        self.size = SIZE
        self.augment = augment
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if payload["dataset"] != "mvtec_ad":
            raise ValueError("split file is not an MVTec AD split")
        self.records = [r for r in payload["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"no records for split {split}")
        # Decoded, resized uint8 arrays per index: full-resolution PNG decoding
        # dominated epoch time.  Cached values are the arrays computed below, so
        # items are bitwise identical with or without the cache.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        if index not in self._cache:
            record = self.records[index]
            with Image.open(self.root / record["image"]) as source:
                image = np.asarray(source.convert("L").resize((self.size[1], self.size[0]),
                                                              Image.Resampling.BILINEAR), dtype=np.uint8).copy()
            mask = None
            if record["mask"] is not None:
                with Image.open(self.root / record["mask"]) as source:
                    mask = np.asarray(source.convert("L").resize((self.size[1], self.size[0]),
                                                                 Image.Resampling.NEAREST), dtype=np.uint8).copy()
            self._cache[index] = (image, mask)
        return self._cache[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = self._load(index)
        image_array = image.astype(np.float32) / 255.0
        if mask is None:
            mask_array = np.zeros(self.size, dtype=np.float32)
        else:
            mask_array = (mask > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        if self.augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-1,))
                mask_tensor = torch.flip(mask_tensor, dims=(-1,))
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-2,))
                mask_tensor = torch.flip(mask_tensor, dims=(-2,))
        return {"image": image_tensor, "mask": mask_tensor, "sample_id": record["sample_id"],
                "split": record["split"]}
