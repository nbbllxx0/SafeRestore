"""Generate the reader-facing evaluation-design and data-role figure.

Layout revision for manuscript v7: panel (b) is rebuilt as two aligned rows on a
shared grid -- the nominal roles above, the reserved stress population below --
so the source card, the row headers, and the row notes share edges instead of
leaving a ragged block of white space on the right.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from saferestore.degradation import degrade_batch
from saferestore.degradation_schedule import LEVELS
from saferestore.torch_data import PublicSegmentationDataset
import scripts.plot_style_manuscript as style

from scripts.plot_style_manuscript import (
    BLUE,
    BLUE_DARK,
    BLUE_LIGHT,
    FIG_W,
    INK,
    LIGHT,
    MID,
    ORANGE,
    ORANGE_LIGHT,
    save,
    setup_style,
)

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT / "splits" / "carinthia_s_v4.json"
# Fixed visible defect from the current test split (v3 sample is not in this partition).
SAMPLE_ID = "2efea7a24d684a84acde1db1a9b94aec"

# Shared grid for panel (b): source card on the left, role track on the right.
SRC_X, SRC_W = 0.012, 0.168
TRACK_X = 0.232
TRACK_RIGHT = 0.988
TRACK_W = TRACK_RIGHT - TRACK_X

ROLE_Y, ROLE_H = 0.455, 0.360
STRESS_Y, STRESS_H = 0.055, 0.200


def resolve_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    manifest = PROJECT / "configs" / "carinthia_s.local.json"
    if manifest.exists():
        return Path(json.loads(manifest.read_text(encoding="utf-8"))["local_root"])
    raise SystemExit("Provide --data-root or create configs/carinthia_s.local.json")


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = BLUE) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            transform=ax.transAxes,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "paper" / "figure_outputs")
    args = parser.parse_args()

    style.OUTPUT = args.output_dir
    style.OUTPUT.mkdir(parents=True, exist_ok=True)

    dataset = PublicSegmentationDataset(
        dataset="carinthia_s",
        root=resolve_data_root(args.data_root),
        split_file=SPLIT,
        split="test",
        augment=False,
    )
    item = next((candidate for candidate in dataset if candidate["sample_id"] == SAMPLE_ID), None)
    if item is None:
        raise RuntimeError(f"study-design sample {SAMPLE_ID} absent from the current test split")
    high = item["image"].unsqueeze(0)
    mask = item["mask"].squeeze(0).numpy() > 0.5
    if mask.sum() == 0:
        raise RuntimeError("study-design sample has an empty mask")

    observations: list[tuple[str, torch.Tensor, float, float]] = []
    for level in LEVELS:
        digest = hashlib.sha256(f"study-design:{SAMPLE_ID}:{level.name}".encode("utf-8")).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
        low, _ = degrade_batch(
            high,
            blur_sigma=level.blur_sigma,
            noise_std=level.noise_std,
            generator=generator,
        )
        raw = F.interpolate(low, size=high.shape[-2:], mode="nearest")
        observations.append((level.name, raw, level.blur_sigma, level.noise_std))

    counts = json.loads(SPLIT.read_text(encoding="utf-8"))["counts"]
    nominal = counts["train"] + counts["detector_val"] + counts["calibration"] + counts["test"]
    setup_style()
    fig = plt.figure(figsize=(FIG_W, 3.78))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.58, 1.10], hspace=0.30, wspace=0.05)

    # ------------------------------------------------------------- panel (a)
    fig.text(
        0.012,
        0.978,
        r"$\bf{a}$  Controlled degradation with fixed spatial correspondence",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK,
    )
    images = [("Reference", high, None, None), *observations]
    for column, (name, tensor, blur_sigma, noise_std) in enumerate(images):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(tensor[0, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(mask, levels=[0.5], colors=[BLUE], linewidths=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name.capitalize(), fontsize=7.4, fontweight="bold", pad=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color(LIGHT)
    caption_y = 0.428
    fig.text(
        0.012,
        caption_y,
        "Matched pixel coordinates; physical scale is not supplied in the public metadata",
        ha="left",
        va="center",
        fontsize=6.0,
        color=MID,
    )
    fig.text(
        0.988,
        caption_y,
        "Blue contour: provided annotation boundary",
        ha="right",
        va="center",
        fontsize=6.0,
        color=BLUE_DARK,
    )

    # ------------------------------------------------------------- panel (b)
    flow = fig.add_subplot(grid[1, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    flow.text(
        0.0,
        1.10,
        r"$\bf{b}$  Disjoint data roles and separate population stress",
        transform=flow.transAxes,
        fontsize=8.5,
        va="top",
        color=INK,
    )

    source_top = ROLE_Y + ROLE_H
    source_bottom = STRESS_Y
    flow.add_patch(
        FancyBboxPatch(
            (SRC_X, source_bottom),
            SRC_W,
            source_top - source_bottom,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            transform=flow.transAxes,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.85,
        )
    )
    source_mid = (source_top + source_bottom) / 2
    flow.text(
        SRC_X + SRC_W / 2,
        source_mid + 0.095,
        "Carinthia-S",
        transform=flow.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
    )
    flow.text(
        SRC_X + SRC_W / 2,
        source_mid - 0.075,
        "4,591 images\n6 morphologies",
        transform=flow.transAxes,
        ha="center",
        va="center",
        fontsize=6.1,
        color=MID,
        linespacing=1.3,
    )

    # Nominal role track.
    flow.text(
        TRACK_X,
        ROLE_Y + ROLE_H + 0.075,
        f"NOMINAL POPULATION   |   {nominal:,} images   |   classes 3, 4, 6",
        transform=flow.transAxes,
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=BLUE_DARK,
    )
    flow.text(
        TRACK_RIGHT,
        ROLE_Y + ROLE_H + 0.075,
        "role widths are schematic",
        transform=flow.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color=MID,
    )
    role_specs = (
        ("Train", counts["train"], "fit models", "#EAF1F5"),
        ("Detector val.", counts["detector_val"], "fit ranking scores", "#DCEAF1"),
        ("Calibration", counts["calibration"], "30% tune / 70% certify", "#CEDFE8"),
        ("Test", counts["test"], "held-out evaluation", "#BDD5E1"),
    )
    gap = 0.010
    segment_width = (TRACK_W - gap * (len(role_specs) - 1)) / len(role_specs)
    for index, (title, count, body, fill) in enumerate(role_specs):
        x = TRACK_X + index * (segment_width + gap)
        flow.add_patch(
            Rectangle(
                (x, ROLE_Y),
                segment_width,
                ROLE_H,
                transform=flow.transAxes,
                facecolor=fill,
                edgecolor=BLUE_LIGHT,
                linewidth=0.75,
            )
        )
        center = x + segment_width / 2
        flow.text(center, ROLE_Y + ROLE_H - 0.088, title, transform=flow.transAxes,
                  ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK)
        flow.text(center, ROLE_Y + ROLE_H / 2 - 0.020, f"{count:,} images", transform=flow.transAxes,
                  ha="center", va="center", fontsize=5.9, color=INK)
        flow.text(center, ROLE_Y + 0.070, body, transform=flow.transAxes,
                  ha="center", va="center", fontsize=5.3, color=MID)

    # Reserved stress population, on the same left edge and width as the role track.
    flow.text(
        TRACK_X,
        STRESS_Y + STRESS_H + 0.075,
        "RESERVED STRESS POPULATION   |   classes 1, 2, 5",
        transform=flow.transAxes,
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color="#8D4D12",
    )
    flow.text(
        TRACK_RIGHT,
        STRESS_Y + STRESS_H + 0.075,
        "never used for certification",
        transform=flow.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color=MID,
    )
    flow.add_patch(
        FancyBboxPatch(
            (TRACK_X, STRESS_Y),
            TRACK_W,
            STRESS_H,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            transform=flow.transAxes,
            facecolor=ORANGE_LIGHT,
            edgecolor=ORANGE,
            linewidth=0.8,
        )
    )
    flow.text(
        TRACK_X + 0.018,
        STRESS_Y + STRESS_H / 2,
        f"Shift stress   |   {counts['shift_test']} images",
        transform=flow.transAxes,
        ha="left",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color="#8D4D12",
    )
    flow.text(
        TRACK_RIGHT - 0.018,
        STRESS_Y + STRESS_H / 2,
        "descriptive diagnostic only; not certificate evidence",
        transform=flow.transAxes,
        ha="right",
        va="center",
        fontsize=5.8,
        color="#8D4D12",
    )

    _arrow(flow, (SRC_X + SRC_W, ROLE_Y + ROLE_H / 2), (TRACK_X, ROLE_Y + ROLE_H / 2), color=BLUE_DARK)
    _arrow(flow, (SRC_X + SRC_W, STRESS_Y + STRESS_H / 2), (TRACK_X, STRESS_Y + STRESS_H / 2), color=ORANGE)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.912, bottom=0.03)
    save(fig, "fig_study_design")

    rows: list[dict[str, object]] = [{"record_type": "example", "name": "reference", "sample_id": SAMPLE_ID}]
    rows.extend(
        {
            "record_type": "degradation",
            "name": name,
            "sample_id": SAMPLE_ID,
            "blur_sigma": blur_sigma,
            "noise_std": noise_std,
        }
        for name, _, blur_sigma, noise_std in observations
    )
    rows.extend({"record_type": "split", "name": name, "count": count} for name, count in counts.items())
    fields = ["record_type", "name", "sample_id", "blur_sigma", "noise_std", "count"]
    with (style.OUTPUT / "fig_study_design_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("study-design sample", SAMPLE_ID, "counts", counts)


if __name__ == "__main__":
    main()
