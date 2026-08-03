from __future__ import annotations

"""Shared publication style for the SafeRestore journal manuscript."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figure_outputs"

# Full-width journal figure geometry in inches. Typography is chosen to remain
# readable after moderate reduction in a two-column journal layout.
FIG_W = 7.05
FIG_W_NARROW = 6.35

# Restrained, color-vision-friendly palette. Blue marks supported/retained
# evidence; amber marks incidents or population stress; greys carry context.
BLUE = "#1F6E9C"
BLUE_DARK = "#174E6B"
BLUE_LIGHT = "#B9D7E7"
ORANGE = "#C56B1A"
ORANGE_LIGHT = "#F2D2AD"
INK = "#202A32"
MID = "#65727C"
LIGHT = "#D9E0E4"
PALE = "#F6F8F9"
WHITE = "#FFFFFF"

METHOD_COLORS = {
    "raw_nearest": INK,
    "bilinear": "#8797A2",
    "bicubic": BLUE,
    "smooth_bicubic": "#91B7CB",
    "unsharp_bicubic": BLUE_DARK,
    "learned": ORANGE,
    "restoration_pool": "#6C9DB6",
    "action_pool": BLUE,
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "axes.edgecolor": "#8D9AA3",
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "grid.color": LIGHT,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "xtick.color": MID,
            "ytick.color": MID,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "xtick.major.size": 2.7,
            "ytick.major.size": 0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 7.2,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "hatch.linewidth": 0.4,
        }
    )


def quiet_grid(ax: plt.Axes, *, axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=LIGHT, linewidth=0.55, alpha=0.7)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left", pad=7.0)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
