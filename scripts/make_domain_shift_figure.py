"""Illustrate nominal and reserved morphologies at readable journal scale."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from saferestore.degradation import degrade_batch
from saferestore.degradation_schedule import LEVELS
from saferestore.torch_data import PublicSegmentationDataset
import scripts.plot_style_manuscript as style
from scripts.plot_style_manuscript import BLUE, FIG_W, INK, LIGHT, MID, ORANGE, save, setup_style


PROJECT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT / "splits" / "carinthia_s_v4.json"
CORRUPTION_KEY = "v4corr:eval"
SEVERE = LEVELS[2]
# Row headers replace the old left gutter, so panels use the full text width.
ROW_HEADER_Y = (0.955, 0.448)
CASES = (
    ("a", "Nominal pit morphology", "4", "22e4c1a4e8aa4599ab8d5b52a42b68b2", "test", BLUE),
    ("b", "Reserved ridge morphology", "1", "4beeb8dfc4df4dceb7eb1d850a87923e", "shift_test", ORANGE),
)


def resolve_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    manifest = PROJECT / "configs" / "carinthia_s.local.json"
    if manifest.exists():
        return Path(json.loads(manifest.read_text(encoding="utf-8"))["local_root"])
    raise SystemExit("Provide --data-root or create configs/carinthia_s.local.json")


def noise_generator(sample_id: str, split: str) -> torch.Generator:
    digest = hashlib.sha256(
        f"noise:{CORRUPTION_KEY}:{split}:{sample_id}:severe-display".encode("utf-8")
    ).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))


def square_roi(mask: np.ndarray, minimum: int = 88, padding: int = 20) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise RuntimeError("cannot form a defect ROI from an empty mask")
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    size = max(minimum, x1 - x0 + 2 * padding, y1 - y0 + 2 * padding)
    size = min(size, mask.shape[0], mask.shape[1])
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    left = min(max(0, cx - size // 2), mask.shape[1] - size)
    top = min(max(0, cy - size // 2), mask.shape[0] - size)
    return left, top, left + size, top + size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "paper" / "figure_outputs")
    args = parser.parse_args()

    style.OUTPUT = args.output_dir
    root = resolve_data_root(args.data_root)
    datasets = {
        split: PublicSegmentationDataset(
            dataset="carinthia_s", root=root, split_file=SPLIT, split=split, augment=False
        )
        for split in ("test", "shift_test")
    }
    wanted = {sample_id for _, _, _, sample_id, _, _ in CASES}
    items = {
        item["sample_id"]: item
        for dataset in datasets.values()
        for item in dataset
        if item["sample_id"] in wanted
    }
    if set(items) != wanted:
        raise RuntimeError(f"missing illustration samples: {sorted(wanted - set(items))}")

    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W, 4.30))
    column_titles = ("Reference image", "Severe observation", "Reference ROI", "Severe ROI")
    source_rows: list[dict[str, object]] = []

    for row, (letter, population, label, sample_id, split_name, contour_color) in enumerate(CASES):
        item = items[sample_id]
        high = item["image"].unsqueeze(0)
        mask = item["mask"].squeeze(0).numpy() > 0.5
        low, severe = degrade_batch(
            high,
            blur_sigma=SEVERE.blur_sigma,
            noise_std=SEVERE.noise_std,
            generator=noise_generator(sample_id, split_name),
        )
        reference = high[0, 0].numpy()
        degraded = severe[0, 0].numpy()
        x0, y0, x1, y1 = square_roi(mask)
        panels = (
            ("reference_full", reference, mask, False),
            ("severe_full", degraded, mask, False),
            ("reference_roi", reference[y0:y1, x0:x1], mask[y0:y1, x0:x1], True),
            ("severe_roi", degraded[y0:y1, x0:x1], mask[y0:y1, x0:x1], True),
        )
        for column, (display, image, local_mask, is_roi) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.contour(local_mask, levels=[0.5], colors=[contour_color], linewidths=1.05)
            if not is_roi:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=contour_color,
                                       linewidth=0.9, linestyle=(0, (3, 2))))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.65)
                spine.set_color(LIGHT)
            if row == 0:
                ax.set_title(column_titles[column], fontsize=7.8, fontweight="bold", pad=4)
            source_rows.append(
                {
                    "panel": letter,
                    "population": population,
                    "class_label": label,
                    "sample_id": sample_id,
                    "split": split_name,
                    "display": display,
                    "mask_pixels": int(mask.sum()),
                    "roi_x0": x0,
                    "roi_y0": y0,
                    "roi_x1": x1,
                    "roi_y1": y1,
                }
            )
        fig.text(
            0.015, ROW_HEADER_Y[row],
            rf"$\bf{{{letter}}}$  {population} (Carinthia-S class {label})",
            ha="left", va="center", fontsize=8.2, color=INK,
        )

    fig.text(
        0.995, 0.015,
        "Dashed boxes define the displayed pixel-space ROIs; physical scale is unavailable in the public metadata.",
        ha="right", va="bottom", fontsize=6.6, color=MID,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.880, bottom=0.075, wspace=0.06, hspace=0.22)
    save(fig, "fig_domain_shift")

    with (style.OUTPUT / "fig_domain_shift_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()
