"""Figures for the TIM manuscript.

fig_sequence_walk: per-gate exact bounds along the tuning-ordered sequence,
certification role, for a Carinthia-S repetition (certifying) and a
KolektorSDD2 repetition (near-miss), with the certified set and the
descriptive minimum certifiable level marked.

fig_ksdd2_diagnostics: (a) endpoint incidence among positives/cleans versus
boundary-tolerance radius; (b) single-feature ranking AUCs for the two
endpoints on fitting and calibration roles.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scripts.run_reported_analysis as reported
from scripts.analyze_router_pool import read_rows
from saferestore.certification import DualRiskTargets, clopper_pearson_upper
from familywise_certification import candidate_sequence, _counts_at_threshold
from run_ksdd2_pipeline import KSDD2_FEATURES
from run_tim_analysis import fit_risk_models_calibrated, DOMAIN_FEATURES

FIGDIR = REPO / "tim_upgrade" / "manuscript" / "figures"
TARGETS = DualRiskTargets()

BLUE = "#2b5f9e"
AMBER = "#c8811a"
CHARCOAL = "#3a3a3a"

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
    # IEEE PDF eXpress rejects Type 3 fonts; 42 embeds TrueType instead.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def walk(domain: str, folder: Path, seed: int, policy: str):
    reported.FEATURES = DOMAIN_FEATURES[domain]
    fit = (
        fit_risk_models_calibrated
        if domain == "kolektor_sdd2"
        else reported.fit_risk_models
        if reported.fit_risk_models.__name__ == "fit_risk_models"
        else reported.fit_risk_models
    )
    train = read_rows(folder / "detector_val_records.csv")
    cal = read_rows(folder / "calibration_records.csv")
    if domain == "kolektor_sdd2":
        models = fit_risk_models_calibrated(train, seed, 0.25)
    else:
        from run_tim_analysis import _FIT_BALANCED

        models = _FIT_BALANCED(train, seed, 0.25)
    arrays = reported.build_policy_arrays(cal, models, TARGETS, 0.25)
    a = arrays[policy]
    tm = reported.stable_tuning_mask([r for r in a["sample_id"]])
    tune_scores = np.asarray(a["score"])[tm]
    tune_pos = np.asarray(a["positive"])[tm]
    cert_scores = np.asarray(a["score"])[~tm]
    cert_pos = np.asarray(a["positive"])[~tm]
    cert_del = np.asarray(a["deletion"])[~tm].astype(bool)
    cert_inv = np.asarray(a["invention"])[~tm].astype(bool)
    sequence = candidate_sequence(
        tune_scores, tune_pos, TARGETS,
        tuning_deletion=np.asarray(a["deletion"])[tm],
        tuning_invention=np.asarray(a["invention"])[tm],
        certification_scale=len(cert_scores) / max(len(tune_scores), 1),
    )
    rows = []
    for threshold in sequence:
        e_del, n_del, e_inv, n_inv = _counts_at_threshold(
            float(threshold), cert_scores, cert_pos, cert_del, cert_inv
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": n_inv / len(cert_scores),
                "del_ucb": clopper_pearson_upper(e_del, n_del, TARGETS.delta / 2),
                "inv_ucb": clopper_pearson_upper(e_inv, n_inv, TARGETS.delta / 2),
            }
        )
    return rows


def fig_sequence_walk() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    cases = [
        ("carinthia_s", REPO / "experiments" / "v4_carinthia_primary_trainseed211",
         211, "action_pool", "(a) Carinthia-S, seed 211, all-action"),
        ("kolektor_sdd2", REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed201",
         201, "action_pool", "(b) KolektorSDD2, seed 201, all-action"),
    ]
    for ax, (domain, folder, seed, policy, title) in zip(axes, cases):
        rows = walk(domain, folder, seed, policy)
        # First failure along the tested (sequence) order = where the
        # certified procedure stops.
        stop = next((r for r in rows
                     if r["del_ucb"] > 0.15 or r["inv_ucb"] > 0.15), None)
        rows.sort(key=lambda r: r["coverage"])
        cov = np.array([r["coverage"] for r in rows])
        du = np.array([r["del_ucb"] for r in rows])
        iu = np.array([r["inv_ucb"] for r in rows])
        ax.plot(cov, du, color=AMBER, lw=1.2, label="evidence-loss bound")
        ax.plot(cov, iu, color=BLUE, lw=1.2, label="excess-activation bound")
        ax.axhline(0.15, color=CHARCOAL, lw=0.8, ls="--")
        ax.text(1.0, 0.157, "target 0.15", fontsize=7, color=CHARCOAL,
                ha="right", va="bottom")
        passing = (du <= 0.15) & (iu <= 0.15)
        if passing.any():
            ax.plot(cov[passing], np.maximum(du, iu)[passing], "o",
                    color="#2e7d32", ms=3.5, label="both bounds pass")
        worst = np.maximum(du, iu)
        j = int(np.argmin(worst))
        ax.plot(cov[j], worst[j], "D", color=CHARCOAL, ms=4,
                label=f"min level {worst[j]:.3f}")
        if stop is not None:
            ax.plot(stop["coverage"], max(stop["del_ucb"], stop["inv_ucb"]),
                    "x", color="#b3261e", ms=7, mew=1.8,
                    label="first failure (testing stops)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("certification coverage")
        ax.set_ylabel("one-sided exact upper bound")
        ax.set_ylim(0, 0.75)
        ax.set_xlim(0, 1.02)
        ax.legend(frameon=False, loc="upper left", handlelength=1.6,
                  borderaxespad=0.2)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig_sequence_walk.pdf")
    plt.close(fig)
    print("wrote fig_sequence_walk.pdf")


TOLERANCE_RUNS = {
    0: "ksdd2_tolerance0px_trainseed201",
    4: "ksdd2_tolerance4px_trainseed201",
    8: "ksdd2_rich_trainseed201",
}


def tolerance_sweep() -> tuple[list[int], list[float], list[float]]:
    """Excess-activation incidence vs. annotation-tolerance radius.

    Each radius reads the calibration-role records regenerated from the same
    seed-201 checkpoints at that tolerance (no retraining); reproduce with

        python scripts/regenerate_ksdd2_records.py \\
            --source experiments/ksdd2_rich_trainseed201 \\
            --output experiments/ksdd2_tolerance{R}px_trainseed201 \\
            --root <KSDD2 root> --split-file splits/kolektor_sdd2_v1.json \\
            --boundary-tolerance {R}
    """
    radii, pos, clean = [], [], []
    for radius, folder in sorted(TOLERANCE_RUNS.items()):
        rows = [r for r in read_rows(
            REPO / "tim_upgrade" / "experiments" / folder
            / "calibration_records.csv") if r["method"] == "raw_nearest"]
        p = [r for r in rows if int(r["has_defect"])]
        c = [r for r in rows if not int(r["has_defect"])]
        radii.append(radius)
        pos.append(sum(int(r["invention_error"]) for r in p) / len(p))
        clean.append(sum(int(r["invention_error"]) for r in c) / len(c))
    return radii, pos, clean


def fig_ksdd2_diagnostics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    # (a) tolerance-ring incidence, recomputed from the regenerated records.
    d, pos, clean = tolerance_sweep()
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
    # (b) feature AUCs, calibration role, raw action, seed 201.
    ax = axes[1]
    folder = REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed201"
    rows = [r for r in read_rows(folder / "calibration_records.csv")
            if r["method"] == "raw_nearest"]
    posm = np.array([bool(int(r["has_defect"])) for r in rows])
    ydel = np.array([float(r["pixel_recall"]) < 0.25 for r in rows])[posm]
    yinv = np.array([int(r["invention_error"]) for r in rows])

    def auc(x, y):
        order = np.argsort(x); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    # Six features defined for the raw action (the four raw-relative
    # difference features are identically zero there), grouped as
    # whole-image (first three) versus localized (last three).
    feats = ("entropy", "forward_consistency", "high_probability_area",
             "peak_probability", "secondary_peak", "active_components")
    labels = ("mean entropy", "forward consist.", "high-prob. area",
              "peak prob.", "secondary peak", "num. components")
    n_whole = 3
    adel, ainv = [], []
    for f in feats:
        x = np.array([float(r[f]) for r in rows])
        adel.append(abs(auc(x[posm], ydel) - 0.5) + 0.5)
        ainv.append(abs(auc(x, yinv) - 0.5) + 0.5)
    x = np.arange(len(feats))
    ax.bar(x - 0.18, adel, 0.36, color=AMBER, label="evidence loss (positives)")
    ax.bar(x + 0.18, ainv, 0.36, color=BLUE, label="excess activation (all)")
    ax.axhline(0.5, color=CHARCOAL, lw=0.8)
    ax.axvline(n_whole - 0.5, color=CHARCOAL, lw=0.8, ls=":")
    ax.text((n_whole - 1) / 2, 1.035, "whole-image", ha="center",
            fontsize=7, color=CHARCOAL, style="italic")
    ax.text(n_whole + (len(feats) - n_whole - 1) / 2, 1.035, "localized",
            ha="center", fontsize=7, color=CHARCOAL, style="italic")
    ax.set_xticks(x, labels, rotation=24, ha="right", fontsize=7)
    ax.set_ylabel("|AUC| (direction-folded)")
    ax.set_ylim(0.45, 1.34)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title("(b) feature ranking power, calibration role", loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=1,
              borderaxespad=0.1, handlelength=1.4)
    fig.savefig(FIGDIR / "fig_ksdd2_diagnostics.pdf")
    plt.close(fig)
    print("wrote fig_ksdd2_diagnostics.pdf")


if __name__ == "__main__":
    fig_sequence_walk()
    fig_ksdd2_diagnostics()
