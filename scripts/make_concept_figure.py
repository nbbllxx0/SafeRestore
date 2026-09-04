"""Generate the reader-facing SafeRestore decision and certification diagram.

The two stages sit on white and are separated by a hairline rule rather than
tinted bands.  Both stages use one shared column grid, and the certification
stage lists its two endpoints as stacked rows keyed by colour, so the differing
denominators read as sentences rather than as an unlabelled two-column table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

# One shared horizontal grid for both stages.
LEFT = 0.032
RIGHT = 0.968
SPAN = RIGHT - LEFT
GAP = 0.034
PAD_X = 0.016

TITLE_FS = 7.4
BODY_FS = 6.6


def track(units: tuple[float, ...]) -> list[tuple[float, float]]:
    """Return (x, width) pairs for relative column units across the shared grid."""
    total = SPAN - GAP * (len(units) - 1)
    scale = total / sum(units)
    boxes: list[tuple[float, float]] = []
    x = LEFT
    for unit in units:
        width = unit * scale
        boxes.append((x, width))
        x += width + GAP
    return boxes


def card(ax, x, y, w, h, title, body, *, edge=LIGHT, linewidth=0.9) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor="white",
            transform=ax.transAxes,
        )
    )
    ax.text(x + PAD_X, y + h - 0.045, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=TITLE_FS, fontweight="bold", color=INK)
    if body:
        ax.text(x + PAD_X, y + h - 0.115, body, transform=ax.transAxes,
                ha="left", va="top", fontsize=BODY_FS, color=MID, linespacing=1.35)


def arrow(ax, x0, x1, y, *, color=MID) -> None:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color=color, transform=ax.transAxes,
                                 shrinkA=0.0, shrinkB=0.0))


def connect(ax, boxes, y, *, color=MID) -> None:
    """Draw one arrow in every gap of a card track."""
    for (x, w), (x_next, _) in zip(boxes, boxes[1:]):
        arrow(ax, x + w + 0.005, x_next - 0.005, y, color=color)


def stage_header(ax, y, text, *, rule_above=False) -> None:
    if rule_above:
        ax.plot([LEFT, RIGHT], [y + 0.055, y + 0.055], transform=ax.transAxes,
                color=LIGHT, linewidth=0.8)
    ax.text(LEFT, y, text, transform=ax.transAxes, ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=INK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "paper" / "figure_outputs")
    args = parser.parse_args()
    style.OUTPUT = args.output_dir
    setup_style()
    fig, ax = plt.subplots(figsize=(FIG_W, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------------- stage 1
    stage_header(ax, 0.945, "RANK ONE DISPLAY FOR EACH IMAGE")
    row1 = track((1.0, 1.0, 1.0, 1.0))
    y1, h1 = 0.660, 0.225
    specs1 = (
        ("Observation", "Measured\ninspection image", BLUE_LIGHT, 0.9),
        ("Candidate displays", "Raw display and five\nrestoration actions", BLUE_LIGHT, 0.9),
        ("Fitted ranking scores", "Evidence-loss score\nExcess-activation score", BLUE_LIGHT, 0.9),
        ("Selected display", "Smallest score, returned\nonly below the fixed gate", BLUE, 1.4),
    )
    for (x, w), (title, body, edge, lw) in zip(row1, specs1):
        card(ax, x, y1, w, h1, title, body, edge=edge, linewidth=lw)
    connect(ax, row1, y1 + h1 / 2, color=BLUE_DARK)

    ax.text(0.5, 0.590,
            "Combined normalized score   "
            r"$q_k=\max\{\widehat s_{\rm loss,k}/\alpha_{\rm loss},\;"
            r"\widehat s_{\rm act,k}/\alpha_{\rm act}\}$"
            "     the smallest-$q_k$ display is returned only when it clears the gate",
            transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color=BLUE_DARK)

    # ---------------------------------------------------------------- stage 2
    stage_header(ax, 0.455, "TEST THE FIXED POLICY ON HELD-OUT CERTIFICATION IMAGES",
                 rule_above=True)
    row2 = track((1.0, 1.62, 1.0, 1.0))
    y2, h2 = 0.075, 0.290
    (x_tune, w_tune), (x_pop, w_pop), (x_evi, w_evi), (x_dec, w_dec) = row2

    card(ax, x_tune, y2, w_tune, h2, "Threshold tuning",
         "One calibration\nsubset fixes\nthe gate")

    card(ax, x_pop, y2, w_pop, h2, "Held-out certification sets", "")
    endpoints = (
        ("EVIDENCE LOSS", BLUE_DARK, "incidents among accepted positives"),
        ("EXCESS ACTIVATION", ORANGE, "incidents among all accepted images"),
    )
    for index, (label, colour, detail) in enumerate(endpoints):
        row_y = y2 + h2 - 0.130 - index * 0.085
        ax.plot([x_pop + PAD_X, x_pop + PAD_X], [row_y - 0.048, row_y + 0.018],
                transform=ax.transAxes, color=colour, linewidth=2.2, solid_capstyle="butt")
        ax.text(x_pop + PAD_X + 0.012, row_y, label, transform=ax.transAxes,
                fontsize=6.1, fontweight="bold", color=colour, va="center")
        ax.text(x_pop + PAD_X + 0.012, row_y - 0.036, detail, transform=ax.transAxes,
                fontsize=6.1, color=MID, va="center")

    card(ax, x_evi, y2, w_evi, h2, "Exact evidence",
         "Two one-sided\nupper bounds,\none per endpoint")

    card(ax, x_dec, y2, w_dec, h2, "Policy decision", "")
    pill_w = w_dec - 2 * PAD_X
    pill_h = 0.050
    for pill_y, label, fill, edge, text_color in (
        (y2 + h2 - 0.160, "PASS", BLUE_LIGHT, BLUE, BLUE_DARK),
        (y2 + h2 - 0.220, "REVIEW", ORANGE_LIGHT, ORANGE, "#874711"),
    ):
        ax.add_patch(FancyBboxPatch((x_dec + PAD_X, pill_y), pill_w, pill_h,
                                    boxstyle="round,pad=0.004,rounding_size=0.010",
                                    transform=ax.transAxes, facecolor=fill,
                                    edgecolor=edge, linewidth=0.8))
        ax.text(x_dec + w_dec / 2, pill_y + pill_h / 2, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.2, fontweight="bold", color=text_color)

    connect(ax, row2, y2 + h2 / 2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_concept")

    rows = [
        {"panel": "selection", "step": 1, "node": "observation"},
        {"panel": "selection", "step": 2, "node": "candidate displays"},
        {"panel": "selection", "step": 3, "node": "fitted ranking scores"},
        {"panel": "selection", "step": 4, "node": "selected display and fixed gate"},
        {"panel": "certification", "step": 1, "node": "threshold tuning"},
        {"panel": "certification", "step": 2, "node": "held-out certification sets"},
        {"panel": "certification", "step": 3, "node": "separate exact upper bounds"},
        {"panel": "certification", "step": 4, "node": "pass or review"},
    ]
    with (style.OUTPUT / "fig_concept_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
