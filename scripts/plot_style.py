from __future__ import annotations

"""Shared plotting helpers sized for arXiv single-column (textwidth ~6.5in)."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"

# Readable single-column widths (inches).
FIG_W = 6.2
FIG_W_NARROW = 5.6

BLUE = "#276FBF"
BLUE_DARK = "#164A7B"
BLUE_LIGHT = "#A9C8E8"
ORANGE = "#D97706"
ORANGE_LIGHT = "#F4C98B"
INK = "#202833"
MID = "#687587"
LIGHT = "#D7DEE7"
PALE = "#F4F7FA"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": LIGHT,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "xtick.color": MID,
            "ytick.color": MID,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
