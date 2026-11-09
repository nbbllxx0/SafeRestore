"""Redraw figures for the TIM-26-11920 revision. Writes only to tim_rr/manuscript/figures/.

Reads frozen analysis outputs from tim_upgrade/ (read-only). Numeric traces
are unchanged; R2.6/R2.7: larger text and explicit numeric axes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tim_upgrade" / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    PALE,
    WHITE,
    setup_style,
)

OUT = Path(__file__).resolve().parents[1] / "figures"
ANALYSIS = REPO / "tim_upgrade" / "experiments" / "tim_analysis"
SEEDS = (201, 203, 207, 211, 223)

CHARCOAL = "#3a3a3a"
GREEN = "#2e7d32"
AMBER = "#c8811a"

POLICY_LABELS = {
    "raw_nearest": "Acquired",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "smooth_bicubic": "Smooth bic.",
    "unsharp_bicubic": "Sharp bic.",
    "learned": "Learned",
    "restoration_pool": "Resto. pool",
    "action_pool": "All-action",
}


def fig_policy_comparison() -> None:
    """Same bars as the submitted figure; I&M axis and legend wording."""
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 200,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    marg, fam = [], []
    for name in POLICY_LABELS:
        m = f = 0
        for seed in SEEDS:
            data = json.loads(
                (ANALYSIS / f"carinthia_s_seed{seed}_floor0.25.json").read_text()
            )
            m += int(bool(data["legacy_marginal"]["policies"][name]["certificate"]["passes"]))
            f += int(data["familywise_single_policy"][name]["certified_threshold"] is not None)
        marg.append(m)
        fam.append(f)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax = axes[0]
    x = np.arange(len(POLICY_LABELS))
    ax.bar(x - 0.19, marg, 0.38, color=CHARCOAL, label="marginal tuned gate")
    ax.bar(x + 0.19, fam, 0.38, color=BLUE, label="familywise sequence")
    ax.set_xticks(x, list(POLICY_LABELS.values()), rotation=28, ha="right")
    ax.set_ylabel("qualified repetitions (of 5)")
    ax.set_ylim(0, 5.6)
    ax.set_yticks(range(6))
    ax.set_title("(a) per-policy qualifications, Carinthia-S", loc="left")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    hold, test, labels = [], [], []
    for seed in SEEDS:
        data = json.loads(
            (ANALYSIS / f"carinthia_s_seed{seed}_floor0.25.json").read_text()
        )
        pf = data["familywise_portfolio"]
        name = pf["selected_policy"]
        hold.append(pf["policies"][name]["coverage"] * 100)
        test.append(pf["selected_test"]["coverage"] * 100)
        labels.append(f"{seed}\n{POLICY_LABELS[name]}")
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.19, hold, 0.38, color=BLUE, label="hold-out role")
    ax.bar(x + 0.19, test, 0.38, color=GREEN, label="held-out test")
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.set_ylabel("auto-acc. fraction (%)")
    ax.set_ylim(0, 66)
    ax.set_title("(b) portfolio selection per repetition", loc="left")
    ax.legend(
        frameon=False, loc="upper left", ncols=2,
        columnspacing=0.9, borderaxespad=0.1,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig_policy_comparison.pdf")
    plt.close(fig)
    print("wrote fig_policy_comparison.pdf")


def fig_sequence_walk() -> None:
    """Same traces as the submitted figure; axis label, target legend entry, two-column legend."""
    import make_tim_figures as tf

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    cases = [
        ("carinthia_s", REPO / "experiments" / "v4_carinthia_primary_trainseed211",
         211, "action_pool", "(a) Carinthia-S, seed 211, all-action"),
        ("kolektor_sdd2", REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed201",
         201, "action_pool", "(b) KolektorSDD2, seed 201, all-action"),
    ]
    for ax, (domain, folder, seed, policy, title) in zip(axes, cases):
        rows = tf.walk(domain, folder, seed, policy)
        stop = next(
            (r for r in rows if r["del_ucb"] > 0.15 or r["inv_ucb"] > 0.15),
            None,
        )
        rows.sort(key=lambda r: r["coverage"])
        cov = np.array([r["coverage"] for r in rows])
        du = np.array([r["del_ucb"] for r in rows])
        iu = np.array([r["inv_ucb"] for r in rows])
        ax.plot(cov, du, color=AMBER, lw=1.2, label="evidence-loss bound")
        ax.plot(cov, iu, color=BLUE, lw=1.2, label="excess-activation bound")
        ax.axhline(0.15, color=CHARCOAL, lw=0.8, ls="--", label="target 0.15")
        passing = (du <= 0.15) & (iu <= 0.15)
        if passing.any():
            ax.plot(
                cov[passing], np.maximum(du, iu)[passing], "o",
                color="#2e7d32", ms=3.5, label="both bounds pass",
            )
        worst = np.maximum(du, iu)
        j = int(np.argmin(worst))
        ax.plot(
            cov[j], worst[j], "D", color=CHARCOAL, ms=4,
            label=f"min level {worst[j]:.3f}",
        )
        if stop is not None:
            ax.plot(
                stop["coverage"], max(stop["del_ucb"], stop["inv_ucb"]),
                "x", color="#b3261e", ms=7, mew=1.8,
                label="first failure",
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("automatic-acceptance fraction")
        ax.set_ylabel("one-sided exact upper bound")
        ax.set_ylim(0, 0.75)
        ax.set_xlim(0, 1.02)
        ax.legend(
            frameon=False, loc="upper left", ncol=2, handlelength=1.4,
            columnspacing=1.0, borderaxespad=0.2,
        )
    fig.savefig(OUT / "fig_sequence_walk.pdf")
    plt.close(fig)
    print("wrote fig_sequence_walk.pdf")


def fig_ksdd2_diagnostics() -> None:
    """Same bars as the submitted figure; panel (b) title only."""
    import make_tim_figures as tf

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    d, pos, clean = tf.tolerance_sweep()
    ax = axes[0]
    width = 0.35
    x = np.arange(len(d))
    ax.bar(x - width / 2, pos, width, color=AMBER, label="positive images")
    ax.bar(x + width / 2, clean, width, color=BLUE, label="clean images")
    ax.set_xticks(x, [f"{v} px" for v in d])
    ax.set_xlabel("annotation tolerance radius")
    ax.set_ylabel("excess-activation incidence")
    ax.set_title("(a) boundary overrun vs. tolerance", loc="left")
    ax.legend(frameon=False)

    ax = axes[1]
    folder = REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed201"
    from scripts.analyze_router_pool import read_rows

    rows = [
        r for r in read_rows(folder / "calibration_records.csv")
        if r["method"] == "raw_nearest"
    ]
    posm = np.array([bool(int(r["has_defect"])) for r in rows])
    ydel = np.array([float(r["pixel_recall"]) < 0.25 for r in rows])[posm]
    yinv = np.array([int(r["invention_error"]) for r in rows])

    def auc(x, y):
        order = np.argsort(x)
        ranks = np.empty(len(x))
        ranks[order] = np.arange(1, len(x) + 1)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    feats = (
        "entropy", "forward_consistency", "high_probability_area",
        "peak_probability", "secondary_peak", "active_components",
    )
    labels = (
        "mean entropy", "forward consist.", "high-prob. area",
        "peak prob.", "secondary peak", "num. components",
    )
    n_whole = 3
    adel, ainv = [], []
    for f in feats:
        xv = np.array([float(r[f]) for r in rows])
        adel.append(abs(auc(xv[posm], ydel) - 0.5) + 0.5)
        ainv.append(abs(auc(xv, yinv) - 0.5) + 0.5)
    x = np.arange(len(feats))
    ax.bar(x - 0.18, adel, 0.36, color=AMBER, label="evidence loss (positives)")
    ax.bar(x + 0.18, ainv, 0.36, color=BLUE, label="excess activation (all)")
    ax.axhline(0.5, color=CHARCOAL, lw=0.8)
    ax.axvline(n_whole - 0.5, color=CHARCOAL, lw=0.8, ls=":")
    ax.text(
        (n_whole - 1) / 2, 1.035, "whole-image", ha="center",
        fontsize=7, color=CHARCOAL, style="italic",
    )
    ax.text(
        n_whole + (len(feats) - n_whole - 1) / 2, 1.035, "localized",
        ha="center", fontsize=7, color=CHARCOAL, style="italic",
    )
    ax.set_xticks(x, labels, rotation=24, ha="right", fontsize=7)
    ax.set_ylabel("|AUC| (direction-folded)")
    ax.set_ylim(0.0, 1.42)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("(b) feature ranking power, qualification split", loc="left")
    ax.legend(
        frameon=False, loc="upper center", ncols=2,
        borderaxespad=0.1, handlelength=1.4,
    )
    fig.savefig(OUT / "fig_ksdd2_diagnostics.pdf")
    plt.close(fig)
    print("wrote fig_ksdd2_diagnostics.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_policy_comparison()
    fig_sequence_walk()
    fig_ksdd2_diagnostics()


if __name__ == "__main__":
    main()
