"""WP3 figures (3.5-inch single-column width; tick labels >= 8 pt).

Run after compare_records.py and montecarlo.py:
    python -B tim_rr/wp3/figures.py
Writes fig_wp3_power.{pdf,png} and fig_wp3_validity.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

SHOW = ["naive", "marginal", "ltt_bonf", "holm", "sgr", "fs_asc", "fs_hb", "fallback",
        "fs_tuning_sel", "fs"]
LABEL = {"naive": "Naive select-best*", "marginal": "Marginal tuned gate", "ltt_bonf": "LTT-Bonferroni",
         "holm": "LTT-Holm", "sgr": "SGR", "fs_asc": "FS, ascending order", "fs_hb": "FS, Hoeffding-Bentkus",
         "fallback": "Fallback, k = 5", "fs_tuning_sel": "FS, tuning-selected", "fs": "FS, margin (ours)"}


def cp_ci(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def power_figure(rec: dict, mc: dict) -> None:
    rows = rec["rows"]["carinthia_s"]
    y = np.arange(len(SHOW))
    test_cov, n_q, pop_cov = [], [], []
    for m in SHOW:
        port = [r["methods"][m]["portfolio"] for r in rows]
        n_q.append(sum(p is not None for p in port))
        test_cov.append(np.mean([p["test"]["coverage"] if p else 0.0 for p in port]))
        pop_cov.append(np.mean([u["methods"][m]["port_cov"] for u in mc["records"]["carinthia_s"].values()]))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    h = 0.38
    ax.barh(y + h / 2, test_cov, h, color="#2b6cb0", label="Frozen records: test (5 seeds)")
    ax.barh(y - h / 2, pop_cov, h, color="#9ec5e8", label="Resampled hold-outs: population")
    for yi, v, k in zip(y, test_cov, n_q):
        ax.text(v + 0.008, yi + h / 2, f"{k}/5", va="center", fontsize=7)
    ax.set_yticks(y, [LABEL[m] for m in SHOW])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.65)
    ax.set_xticks(np.arange(0, 0.61, 0.1))
    ax.set_xlabel("Mean acceptance fraction (failure-inclusive)\nCarinthia-S, portfolio selection; *invalid", loc="right")
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False,
              borderaxespad=0.2, handlelength=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_wp3_power.{ext}", dpi=300)
    plt.close(fig)


def validity_figure(mc: dict) -> None:
    reps = mc["reps"]
    settings = [("Carinthia-S records, resampled", mc["records"]["carinthia_s"], "o", "#2b6cb0"),
                ("KSDD2 records, resampled", mc["records"]["kolektor_sdd2"], "s", "#c05621"),
                ("Synthetic, all gates at 0.153", mc["synthetic"]["boundary_null"], "^", "#2f855a"),
                ("Synthetic, loss risk crosses $\\alpha$", mc["synthetic"]["crossing_loss"], "D", "#6b46c1")]
    floor = 1e-4
    y = np.arange(len(SHOW))
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    off = np.linspace(-0.27, 0.27, len(settings))
    for (name, units, mk, col), dy in zip(settings, off):
        for yi, m in zip(y, SHOW):
            k = max(u["methods"][m]["port_fwer_k"] for u in units.values())
            lo, hi = cp_ci(k, reps)
            v = k / reps
            ax.plot([max(lo, floor), max(hi, floor)], [yi + dy] * 2, color=col, linewidth=0.9)
            ax.plot(max(v, floor), yi + dy, mk, color=col, markersize=3.6,
                    markerfacecolor=col if k > 0 else "white",
                    label=name if m == SHOW[0] else None)
    ax.axvline(0.10, color="crimson", linestyle="--", linewidth=1.0)
    ax.text(0.105, len(SHOW) - 0.55, "$\\delta$ = 0.10", color="crimson", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.8, 1.0)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["1e-4", "1e-3", "0.01", "0.1", "1"])
    ax.set_yticks(y, [LABEL[m] for m in SHOW])
    ax.invert_yaxis()
    ax.set_xlabel(f"Realised FWER, max of 5 seeds or draws\n"
                  f"{reps} draws each; 95% CI; open = 0", loc="right")
    ax.grid(axis="x", which="major", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False,
              handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_wp3_validity.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    rec = json.loads((OUT / "records_comparison.json").read_text(encoding="utf-8"))
    mc = json.loads((OUT / "montecarlo_results.json").read_text(encoding="utf-8"))
    power_figure(rec, mc)
    validity_figure(mc)


if __name__ == "__main__":
    main()
