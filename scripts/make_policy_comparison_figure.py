"""Generate the headline policy-comparison figure.

Every row is a separately fit, tuned, and certified policy under the same five
training repetitions.  The left panel counts marginal certificates that pass;
the right panel contrasts the workload the tuned gate would accept with the
workload the certificate decision actually supports.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import scripts.plot_style_manuscript as style
from scripts.plot_style_manuscript import (
    BLUE,
    BLUE_DARK,
    BLUE_LIGHT,
    FIG_W,
    INK,
    LIGHT,
    MID,
    panel_title,
    quiet_grid,
    save,
    setup_style,
)

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "experiments" / "v4_reported_analysis_primary"
ABLATIONS = PROJECT / "experiments" / "v4_supporting_ablations_primarypath" / "seed_results.csv"
SEEDS = (201, 203, 207, 211, 223)

# (row kind, label, source key). "head" rows carry a group name only.
LAYOUT = (
    ("head", "Fixed actions", None),
    ("policy", "Raw", "raw_nearest"),
    ("policy", "Bilinear", "bilinear"),
    ("policy", "Bicubic", "bicubic"),
    ("policy", "Smoothed bicubic", "smooth_bicubic"),
    ("policy", "Sharpened bicubic", "unsharp_bicubic"),
    ("policy", "Learned", "learned"),
    ("head", "Adaptive pools", None),
    ("policy", "Restoration-only pool", "restoration_pool"),
    ("policy", "All-action policy", "action_pool"),
    ("head", "Simplifications of the all-action policy", None),
    ("ablation", "Remove consistency feature", ("feature", "no_consistency")),
    ("ablation", "Interpolation-only pool", ("pool", "interpolation_only")),
    ("ablation", "Remove learned action", ("pool", "no_learned")),
    ("ablation", "Remove raw action", ("pool", "no_raw")),
    ("ablation", "Learned action only", ("pool", "learned_only")),
)
PRIMARY_LABEL = "All-action policy"


def summarize(passes: list[bool], conditional: list[float], gated: list[float]) -> dict[str, float]:
    return {
        "passes": sum(passes),
        "conditional_mean": statistics.mean(conditional),
        "gated_mean": statistics.mean(gated),
        "gated_sd": statistics.stdev(gated),
    }


def policy_summary(key: str) -> dict[str, float]:
    passes, conditional, gated = [], [], []
    for seed in SEEDS:
        payload = json.loads((RESULTS / f"seed_{seed}_floor_0.25.json").read_text(encoding="utf-8"))
        item = payload["policies"][key]
        test = item.get("test") or {}
        passes.append(bool(item["certificate"]["passes"]))
        conditional.append(100.0 * float(test.get("coverage") or 0.0))
        gated.append(100.0 * float(test.get("deployable_coverage") or 0.0))
    return summarize(passes, conditional, gated)


def ablation_summary(rows: list[dict[str, str]], kind: str, variant: str) -> dict[str, float]:
    selected = [row for row in rows if row["ablation_type"] == kind and row["variant"] == variant]
    if len(selected) != len(SEEDS):
        raise RuntimeError(f"expected {len(SEEDS)} repetitions for {kind}/{variant}, found {len(selected)}")
    return summarize(
        [row["certificate_pass"] == "1" for row in selected],
        [100.0 * float(row["test_coverage"] or 0.0) for row in selected],
        [100.0 * float(row["deployable_test_coverage"] or 0.0) for row in selected],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "paper" / "figure_outputs")
    args = parser.parse_args()
    style.OUTPUT = args.output_dir

    with ABLATIONS.open(encoding="utf-8", newline="") as handle:
        ablation_rows = list(csv.DictReader(handle))

    entries: list[tuple[str, str, dict[str, float] | None]] = []
    for kind, label, key in LAYOUT:
        if kind == "head":
            entries.append((kind, label, None))
        elif kind == "policy":
            entries.append((kind, label, policy_summary(str(key))))
        else:
            entries.append((kind, label, ablation_summary(ablation_rows, *key)))

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 4.05), gridspec_kw={"width_ratios": (0.80, 1.20)})
    y = np.arange(len(entries))
    data_rows = [(index, label, summary) for index, (kind, label, summary) in enumerate(entries) if summary]

    for ax in axes:
        ax.set_ylim(len(entries) - 0.5, -0.5)
        for index, (kind, _, _) in enumerate(entries):
            if kind == "head" and index:
                ax.axhline(index - 0.5, color=LIGHT, linewidth=0.7)
        primary_index = next(i for i, (_, label, _) in enumerate(entries) if label == PRIMARY_LABEL)
        ax.axhspan(primary_index - 0.42, primary_index + 0.42, color="#EEF3F6", linewidth=0, zorder=0)

    ax = axes[0]
    track = [5.0 if summary else 0.0 for _, _, summary in entries]
    ax.barh(y, track, height=0.56, color="#EDF1F3", edgecolor=LIGHT, linewidth=0.6, zorder=1)
    for index, label, summary in data_rows:
        primary = label == PRIMARY_LABEL
        ax.barh([index], [summary["passes"]], height=0.56, color=BLUE if not primary else BLUE_DARK,
                edgecolor=BLUE_DARK, linewidth=0.6, zorder=2)
        ax.text(summary["passes"] + 0.16, index, f"{summary['passes']}/5", va="center", ha="left",
                fontsize=6.4, fontweight="bold" if primary else "normal", color=BLUE_DARK, zorder=3)
    ax.set_xlim(0, 5.9)
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("Marginal certificates passed")
    ax.set_yticks(y, [label for _, label, _ in entries])
    for index, tick in enumerate(ax.get_yticklabels()):
        kind, label, _ = entries[index]
        if kind == "head":
            tick.set_fontweight("bold")
            tick.set_fontsize(6.9)
            tick.set_color(INK)
        else:
            tick.set_fontsize(6.9)
            tick.set_color(INK if label == PRIMARY_LABEL else MID)
            if label == PRIMARY_LABEL:
                tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    panel_title(ax, "a", "Certificates passed")
    quiet_grid(ax, axis="x")

    ax = axes[1]
    for index, label, summary in data_rows:
        low = max(0.0, summary["gated_mean"] - summary["gated_sd"])
        high = summary["gated_mean"] + summary["gated_sd"]
        ax.plot([low, high], [index, index], color=BLUE_LIGHT, linewidth=2.4, solid_capstyle="round",
                zorder=2)
        ax.scatter([summary["conditional_mean"]], [index], s=26, facecolors="white",
                   edgecolors=MID, linewidths=0.9, zorder=3)
        ax.scatter([summary["gated_mean"]], [index], s=30, marker="D", color=BLUE_DARK,
                   edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_yticks(y, ["" for _ in entries])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.5, 58)
    ax.set_xlabel("Test coverage (%)")
    panel_title(ax, "b", "Workload before and after the certificate")
    quiet_grid(ax, axis="x")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                       markeredgecolor=MID, label="conditional: accepted by the tuned gate"),
            plt.Line2D([], [], marker="D", linestyle="none", color=BLUE_DARK,
                       label="pass-gated: supported by the certificate"),
            plt.Line2D([], [], color=BLUE_LIGHT, linewidth=2.4,
                       label="pass-gated spread across repetitions"),
        ],
        loc="upper center", bbox_to_anchor=(0.44, -0.115), ncol=1, fontsize=6.3,
        handletextpad=0.6, labelspacing=0.35,
    )

    fig.subplots_adjust(left=0.245, right=0.985, top=0.925, bottom=0.225, wspace=0.045)
    save(fig, "fig_policy_comparison")

    source = [
        {
            "group": "fixed action" if index < 7 else ("adaptive pool" if index < 10 else "simplification"),
            "policy": label,
            "certificates_passed": summary["passes"],
            "repetitions": len(SEEDS),
            "conditional_test_coverage_percent": round(summary["conditional_mean"], 3),
            "pass_gated_test_coverage_percent": round(summary["gated_mean"], 3),
            "pass_gated_test_coverage_sd": round(summary["gated_sd"], 3),
        }
        for index, label, summary in data_rows
    ]
    with (style.OUTPUT / "fig_policy_comparison_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)


if __name__ == "__main__":
    main()
