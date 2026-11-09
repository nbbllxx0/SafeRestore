"""WP5 figures (3.5-inch width, all text 8 pt, numeric axes with gridlines).

Reads only CSV/JSON in tim_rr/wp5/; writes fig_{a,b,c,d}_*.{pdf,png} next to this file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})
BLUE, AMBER, GREY, RED, GREEN = "#1f5a99", "#d08a1d", "#6e6e6e", "#b2342b", "#3a7d44"
SEEDS = (201, 203, 207, 211, 223)


def read(name: str) -> list[dict]:
    return list(csv.DictReader((HERE / name).open(encoding="utf-8")))


def fnum(x):
    return float(x) if x not in ("", None) else float("nan")


def grid(ax):
    ax.grid(True, lw=0.4, color="0.85")
    ax.set_axisbelow(True)


def save(fig, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("wrote", stem)


def fig_a() -> None:
    inc = [r for r in read("exp_a_incidence.csv") if r["role"] == "calibration"]
    per = read("exp_a_per_seed.csv")
    radii = sorted({int(r["radius"]) for r in inc})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), constrained_layout=True)
    ax = axes[0]
    series = (("act_pos_e", "n_pos", "positives (ring excluded)", AMBER, "o", "-"),
              ("strict_act_pos_e", "n_pos", "positives (strict: ring counts)", RED, "s", "--"),
              ("act_clean_e", "n_clean", "clean images", BLUE, "^", "-"))
    for key, den, label, color, marker, ls in series:
        means, lo, hi = [], [], []
        for r_ in radii:
            vals = [int(x[key]) / int(x[den]) for x in inc if int(x["radius"]) == r_]
            means.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
        means, lo, hi = map(np.asarray, (means, lo, hi))
        ax.errorbar(radii, means, yerr=[means - lo, hi - means], color=color, marker=marker, ms=3.5,
                    lw=1.0, ls=ls, capsize=2, label=label)
    ax.axvline(8, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(radii)
    ax.set_xlabel("tolerance radius R (px)")
    ax.set_ylabel("activation incidence\n(raw action, qualification split)")
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, handletextpad=0.4)
    ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top")
    grid(ax)
    ax = axes[1]
    for label, color, marker, dx in (("standard", BLUE, "o", -0.25), ("strict", RED, "s", 0.25)):
        for i, seed in enumerate(SEEDS):
            ys = [fnum(next(x["alpha_star"] for x in per if int(x["radius"]) == r_ and int(x["seed"]) == seed
                            and x["label"] == label)) for r_ in radii]
            ax.plot(np.asarray(radii) + dx, ys, ls="none", marker=marker, ms=3, color=color, alpha=0.8,
                    label=f"{label} labels" if i == 0 else None)
    ax.axhline(0.15, color="k", lw=0.8, ls="--")
    ax.text(10.0, 0.155, "α = 0.15", va="bottom", ha="center")
    ax.axvline(8, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(radii)
    ax.set_xlabel("tolerance radius R (px)")
    ax.set_ylabel("descriptive α* per seed")
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, 0.97, "(b)", transform=ax.transAxes, va="top")
    grid(ax)
    save(fig, "fig_a_tolerance")


def fig_b() -> None:
    summ = [r for r in read("exp_b_summary.csv") if r["role"] == "calibration"]
    inc = [r for r in read("exp_b_incidence.csv") if r["role"] == "calibration"]
    actions = ("raw", "smoothed", "sharpened", "learned")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), constrained_layout=True, sharex=True)
    for ax, domain, title in zip(axes, ("carinthia_s", "kolektor_sdd2"), ("(a) Carinthia-S", "(b) KolektorSDD2")):
        x = np.arange(len(actions))
        bars = (("native", "loss_rate", "loss, native", AMBER, None),
                ("nominal (published records)", "loss_rate", "loss, nominal", AMBER, "////"),
                ("native", "act_rate_all", "activation, native", BLUE, None),
                ("nominal (published records)", "act_rate_all", "activation, nominal", BLUE, "////"))
        width = 0.2
        for j, (arm, key, label, color, hatch) in enumerate(bars):
            means, lo, hi = [], [], []
            for a in actions:
                vals = [fnum(r[key]) for r in inc if r["domain"] == domain and r["arm"] == arm and r["action"] == a]
                means.append(np.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
            means, lo, hi = map(np.asarray, (means, lo, hi))
            ax.bar(x + (j - 1.5) * width, means, width, color="white" if hatch else color,
                   edgecolor=color, hatch=hatch, lw=0.8, label=label,
                   yerr=[means - lo, hi - means], error_kw={"lw": 0.6, "capsize": 1.5})
        ax.axhline(0.15, color="k", lw=0.8, ls="--")
        top = 1.0 if domain == "carinthia_s" else 0.7
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top + 1e-9, 0.1 if top < 0.8 else 0.2))
        ax.set_ylabel("incidence rate\n(qualification split)")
        ax.text(0.01, 0.97, title, transform=ax.transAxes, va="top")
        grid(ax)
    axes[1].legend(frameon=False, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.88))
    axes[1].set_xticks(np.arange(len(actions)), actions)
    axes[1].set_xlabel("action")
    save(fig, "fig_b_native")


def fig_c() -> None:
    rows = read("exp_c_portfolio.csv")
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    labels = {"nominal": "nominal", "motion": "motion", "defocus": "defocus",
              "poisson_gaussian": "Pois.-Gauss.", "jpeg30": "JPEG 30", "jpeg50": "JPEG 50", "jpeg70": "JPEG 70"}
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.6), constrained_layout=True, sharex=True)
    keys = (("auto_acceptance", "test auto-acceptance", (0, 0.7), 0.1),
            ("loss_rate", "test loss rate", (0, 0.7), 0.1),
            ("act_rate", "test activation rate", (0, 1.0), 0.2))
    offs = np.linspace(-0.24, 0.24, len(SEEDS))
    markers = ("o", "s", "^", "D", "v")
    for k, (ax, (key, ylab, ylim, step)) in enumerate(zip(axes, keys)):
        for i, seed in enumerate(SEEDS):
            for j, fam in enumerate(fams):
                r = next(x for x in rows if x["family"] == fam and int(x["seed"]) == seed)
                y = fnum(r[key])
                empty = int(r["accepted"]) == 0 or (key == "loss_rate" and int(r["loss_n"]) == 0)
                if empty and key != "auto_acceptance":
                    ax.plot(j + offs[i], 0.0, marker="x", color=GREY, ms=3.5, ls="none")
                    continue
                exceeds = key != "auto_acceptance" and y > 0.15
                ax.plot(j + offs[i], y, marker=markers[i], ms=3.2, ls="none",
                        color=RED if exceeds else BLUE, mfc="none" if exceeds else BLUE,
                        label=f"seed {seed}" if (k == 0 and j == 0) else None)
        if key != "auto_acceptance":
            ax.axhline(0.15, color="k", lw=0.8, ls="--")
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, step))
        ax.set_ylabel(ylab)
        ax.text(0.01, 0.97, f"({'abc'[k]})", transform=ax.transAxes, va="top")
        grid(ax)
    handles, labs = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labs, frameon=False, loc="upper center", ncols=3, handletextpad=0.2,
                   columnspacing=0.8, bbox_to_anchor=(0.5, 0.9))
    axes[2].set_xticks(np.arange(len(fams)), [labels[f] for f in fams], rotation=30, ha="right")
    axes[2].set_xlabel("degradation family of the test role (gate fixed)")
    save(fig, "fig_c_family_shift")


def fig_d() -> None:
    rows = read("exp_d_prevalence.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3.3), constrained_layout=True)
    x = np.array([fnum(r["test_act_rate"]) - fnum(r["alpha"]) for r in rows])
    y = np.array([fnum(r["test_act_rate_reweighted_to_qual"]) - fnum(r["alpha"]) for r in rows])
    ex = np.array([r["exceeds_alpha_test_act"] == "1" for r in rows])
    ax.plot(x[~ex], y[~ex], "o", ms=3.2, color=BLUE, label=f"test rate ≤ α ({int((~ex).sum())})")
    ax.plot(x[ex], y[ex], "s", ms=3.6, color=RED, mfc="none", label=f"test rate > α ({int(ex.sum())})")
    lim = (-0.27, 0.10)
    ax.plot(lim, lim, color=GREY, lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = np.round(np.arange(-0.25, 0.101, 0.05), 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("test activation rate − α")
    ax.set_ylabel("reweighted to qualification mixture − α")
    ax.legend(frameon=False, loc="upper left")
    grid(ax)
    save(fig, "fig_d_prevalence")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    for w in which:
        {"a": fig_a, "b": fig_b, "c": fig_c, "d": fig_d}[w]()
