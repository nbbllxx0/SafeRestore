"""Generate quantitative manuscript figures and their source tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from saferestore.certification import (
    DualRiskTargets,
    certify_threshold,
    choose_threshold,
    choose_threshold_empirical_target,
    choose_threshold_tuning_ucb,
    clopper_pearson_upper,
    evaluate_threshold,
)
from scripts.analyze_router_pool import read_rows
from scripts.run_reported_analysis import (
    build_policy_arrays,
    fit_risk_models,
    predict_model,
    stable_tuning_mask,
    subset,
)
import scripts.plot_style_manuscript as style
from scripts.plot_style_manuscript import (
    BLUE,
    BLUE_DARK,
    BLUE_LIGHT,
    FIG_W,
    INK,
    LIGHT,
    METHOD_COLORS,
    MID,
    ORANGE,
    panel_title,
    quiet_grid,
    save,
    setup_style,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figure_outputs"
RESULTS = ROOT / "experiments" / "v4_reported_analysis_primary"
SEEDS = (201, 203, 207, 211, 223)
FOLDER_TEMPLATE = "v4_carinthia_primary_trainseed{seed}"
METHODS = (
    "raw_nearest",
    "bilinear",
    "bicubic",
    "smooth_bicubic",
    "unsharp_bicubic",
    "learned",
)
METHOD_LABELS = {
    "raw_nearest": "Raw",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "smooth_bicubic": "Smoothed bicubic",
    "unsharp_bicubic": "Sharpened bicubic",
    "learned": "Learned",
}
SEVERITIES = ("mild", "moderate", "severe")
SEVERITY_LABELS = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}
TUNE_KEY = "v4tunecert"


def load_seed(seed: int, floor: float) -> dict[str, Any]:
    return json.loads((RESULTS / f"seed_{seed}_floor_{floor:.2f}.json").read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def action_tradeoffs() -> None:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        records = read_rows(ROOT / "experiments" / FOLDER_TEMPLATE.format(seed=seed) / "test_records.csv")
        for method in METHODS:
            local = [row for row in records if row["method"] == method]
            positive = [row for row in local if row["has_defect"]]
            rows.append(
                {
                    "seed": seed,
                    "action": method,
                    "psnr_db": float(np.mean([row["psnr"] for row in local])),
                    "evidence_loss_rate_positive": float(np.mean([row["pixel_recall"] < 0.25 for row in positive])),
                    "excess_activation_rate_all": float(np.mean([row["invention_error"] for row in local])),
                    "positive_images": len(positive),
                    "all_images": len(local),
                }
            )
    write_csv(OUTPUT / "fig_action_tradeoffs_source.csv", rows)

    means = {
        method: float(np.mean([row["psnr_db"] for row in rows if row["action"] == method]))
        for method in METHODS
    }
    ordered = sorted(METHODS, key=lambda method: means[method], reverse=True)
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.05), sharey=True,
                             gridspec_kw={"width_ratios": (0.95, 1.05, 1.05)})
    fields = (
        ("psnr_db", "PSNR (dB)", "a", "Reconstruction fidelity", 1.0),
        ("evidence_loss_rate_positive", "Evidence-loss incidents (%)", "b", "Positive images", 100.0),
        ("excess_activation_rate_all", "Excess-activation incidents (%)", "c", "All images", 100.0),
    )
    y = np.arange(len(ordered))
    for ax, (field, xlabel, letter, title, scale) in zip(axes, fields):
        for position, method in zip(y, ordered):
            values = scale * np.asarray([row[field] for row in rows if row["action"] == method], dtype=float)
            color = ORANGE if method == "learned" else BLUE if method == "bicubic" else INK
            ax.hlines(position, values.min(), values.max(), color=color, linewidth=1.1, alpha=0.45, zorder=1)
            offsets = np.linspace(-0.15, 0.15, len(values))
            ax.scatter(values, position + offsets, s=20, facecolors="white", edgecolors=color,
                       linewidths=0.9, zorder=2)
            ax.scatter([values.mean()], [position], s=38, marker="D", color=color,
                       edgecolors="white", linewidths=0.6, zorder=3)
        panel_title(ax, letter, title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(len(ordered) - 0.5, -0.5)
        quiet_grid(ax, axis="x")
    axes[0].set_yticks(y, [METHOD_LABELS[method] for method in ordered])
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK,
                   label="model-fitting repetition"),
            Line2D([], [], marker="D", linestyle="none", markerfacecolor=INK, markeredgecolor="white",
                   label="five-repetition mean"),
            Line2D([], [], color=INK, linewidth=1.1, alpha=0.45, label="observed range"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.57, -0.01), fontsize=7.0,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.90, bottom=0.22, wspace=0.22)
    save(fig, "fig_action_tradeoffs")


def boundary_evidence() -> None:
    primary = {seed: load_seed(seed, 0.25) for seed in SEEDS}
    strict = {seed: load_seed(seed, 0.50) for seed in SEEDS}
    source_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for floor, payload in ((0.25, primary[seed]), (0.50, strict[seed])):
            item = payload["policies"]["action_pool"]
            shift = item.get("shift_diagnostic") or {}
            test = item.get("test") or {}
            source_rows.append(
                {
                    "record_type": "endpoint",
                    "seed": seed,
                    "recall_floor": floor,
                    "certificate_pass": int(item["certificate"]["passes"]),
                    "certified_coverage": test.get("deployable_coverage", 0.0) or 0.0,
                    "nominal_evidence_loss_rate": test.get("deletion_risk"),
                    "shift_coverage": shift.get("coverage"),
                    "shift_evidence_loss_rate": shift.get("deletion_risk"),
                }
            )
        for item in primary[seed]["certificate_size_sensitivity"]:
            source_rows.append(
                {
                    "record_type": "evidence_volume",
                    "seed": seed,
                    "certificate_fraction": item["fraction"],
                    "certificate_pass": int(item["passes"]),
                    "evidence_loss_denominator": item["deletion_denominator"],
                    "activation_denominator": item["invention_denominator"],
                }
            )
    write_csv(OUTPUT / "fig_boundary_evidence_source.csv", source_rows)

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.18), gridspec_kw={"width_ratios": (1.05, 0.90, 1.15)})

    ax = axes[0]
    y = np.arange(len(SEEDS))
    primary_cov = [100.0 * float(primary[s]["policies"]["action_pool"]["test"].get("deployable_coverage") or 0.0) for s in SEEDS]
    strict_cov = [100.0 * float(strict[s]["policies"]["action_pool"]["test"].get("deployable_coverage") or 0.0) for s in SEEDS]
    offset = 0.15
    for yy, p_cov, s_cov in zip(y, primary_cov, strict_cov):
        ax.plot([p_cov, s_cov], [yy - offset, yy + offset], color=LIGHT, linewidth=1.3, zorder=1)
        ax.scatter([p_cov], [yy - offset], s=34, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter([s_cov], [yy + offset], s=34, marker="s", facecolors="white", edgecolors=ORANGE, linewidths=1.1, zorder=3)
    ax.set_yticks(y, [str(seed) for seed in SEEDS])
    ax.invert_yaxis()
    ax.set_xlim(-4, max(primary_cov + strict_cov) + 8)
    ax.set_xlabel("Pass-gated test coverage (%)")
    ax.set_ylabel("Training repetition")
    panel_title(ax, "a", "Endpoint sensitivity")
    quiet_grid(ax, axis="x")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="white", label="recall floor 0.25"),
            Line2D([], [], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=ORANGE, label="recall floor 0.50"),
        ],
        loc="lower right", fontsize=6.4,
    )

    ax = axes[1]
    fractions = (0.25, 0.50, 0.75, 1.00)
    pass_counts = [
        sum(next(item["passes"] for item in primary[seed]["certificate_size_sensitivity"]
                 if abs(item["fraction"] - fraction) < 1e-9) for seed in SEEDS)
        for fraction in fractions
    ]
    xx = np.arange(len(fractions))
    ax.bar(xx, [5] * len(xx), width=0.62, color="#EDF1F3", edgecolor=LIGHT, linewidth=0.7)
    ax.bar(xx, pass_counts, width=0.62, color=BLUE, edgecolor=BLUE_DARK, linewidth=0.6)
    for xpos, count in zip(xx, pass_counts):
        ax.text(xpos, max(0.18, count + 0.12), f"{count}/5", ha="center", va="bottom",
                fontsize=7.0, fontweight="bold", color=BLUE_DARK)
    ax.set_xticks(xx, ["25", "50", "75", "100"])
    ax.set_ylim(0, 5.4)
    ax.set_yticks(range(0, 6))
    ax.set_xlabel("Certification images retained (%)")
    ax.set_ylabel("Policies passing")
    panel_title(ax, "b", "Evidence volume")
    quiet_grid(ax, axis="y")

    ax = axes[2]
    pairs: list[tuple[int, float, float]] = []
    missing: list[int] = []
    for seed in SEEDS:
        policy = primary[seed]["policies"]["action_pool"]
        threshold = policy["threshold"]
        test = policy["test"]
        shift = policy.get("shift_diagnostic") or {}
        if threshold is None:
            missing.append(seed)
            continue
        if test.get("deletion_risk") is not None and shift.get("deletion_risk") is not None:
            pairs.append((seed, 100.0 * float(test["deletion_risk"]), 100.0 * float(shift["deletion_risk"])))
        else:
            missing.append(seed)
    for seed, nominal, shifted in pairs:
        ax.plot([0, 1], [nominal, shifted], color=LIGHT, linewidth=1.35, zorder=1)
        ax.scatter([0], [nominal], s=34, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter([1], [shifted], s=38, marker="D", facecolors="white", edgecolors=ORANGE, linewidths=1.1, zorder=3)
        offset = {207: -3.0, 223: 3.0}.get(seed, 0.0)
        ax.annotate(str(seed), xy=(1, shifted), xytext=(1.08, shifted + offset),
                    ha="left", va="center", fontsize=7.2, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MID, "linewidth": 0.55,
                                "shrinkA": 0, "shrinkB": 3})
    ax.axhspan(0, 15, color=BLUE_LIGHT, alpha=0.20, linewidth=0)
    ax.axhline(15, color=INK, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(-0.03, 16.8, "15% target", fontsize=6.4, color=BLUE_DARK)
    ax.set_xticks([0, 1], ["Nominal\ntest", "Reserved\nmorphologies"])
    ax.set_xlim(-0.12, 1.30)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Evidence-loss incident rate (%)")
    panel_title(ax, "c", "Population transfer")
    quiet_grid(ax, axis="y")
    if missing:
        ax.text(0.50, -0.28, f"No eligible gate for repetitions {', '.join(map(str, missing))}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=MID)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.26, wspace=0.40)
    save(fig, "fig_boundary_evidence")


def collect_selected_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    targets = DualRiskTargets()
    for seed in SEEDS:
        folder = ROOT / "experiments" / FOLDER_TEMPLATE.format(seed=seed)
        records = read_rows(folder / "test_records.csv")
        models = fit_risk_models(read_rows(folder / "detector_val_records.csv"), seed, 0.25)
        policy = build_policy_arrays(records, models, targets, 0.25)["action_pool"]
        by_method = {method: [row for row in records if row["method"] == method] for method in METHODS}
        probabilities = {method: predict_model(models[method], by_method[method]) for method in METHODS}
        result = load_seed(seed, 0.25)["policies"]["action_pool"]
        if result["threshold"] is None or float(result["threshold"]) == float("-inf"):
            continue
        threshold = float(result["threshold"])
        passed = bool(result["certificate"]["passes"])
        for index, sample_id in enumerate(policy["sample_id"]):
            method = str(policy["method"][index])
            record = by_method[method][index]
            if record["sample_id"] != str(sample_id):
                raise RuntimeError("selected-row ordering mismatch")
            loss_score, activation_score = probabilities[method]
            output.append(
                {
                    "seed": seed,
                    "sample_id": str(sample_id),
                    "severity": record["severity"],
                    "has_defect": int(record["has_defect"]),
                    "selected_action": method,
                    "evidence_loss_score": float(loss_score[index]),
                    "excess_activation_score": float(activation_score[index]),
                    "gate_score": float(policy["score"][index]),
                    "gate_threshold": threshold,
                    "accepted": int(float(policy["score"][index]) <= threshold),
                    "certificate_pass": int(passed),
                    "evidence_loss_incident": int(record["has_defect"] and float(record["pixel_recall"]) < 0.25),
                    "excess_activation_incident": int(record["invention_error"]),
                    "pixel_recall": float(record["pixel_recall"]),
                    "clean_fpr": float(record["clean_fpr"]),
                }
            )
    return output


def bin_rows(rows: list[dict[str, object]], score: str, outcome: str, bins: int = 6) -> list[dict[str, float]]:
    order = np.argsort(np.asarray([float(row[score]) for row in rows]))
    groups = np.array_split(order, bins)
    result: list[dict[str, float]] = []
    for index, group in enumerate(groups, start=1):
        local = [rows[int(position)] for position in group]
        errors = int(sum(int(row[outcome]) for row in local))
        result.append(
            {
                "score_quantile": float(index),
                "n": float(len(local)),
                "incidents": float(errors),
                "mean_score": float(np.mean([float(row[score]) for row in local])),
                "observed_rate": errors / len(local),
            }
        )
    return result


def risk_score_behavior(rows: list[dict[str, object]]) -> None:
    panels = (
        ("evidence_loss_score", "evidence_loss_incident", "evidence_loss", "a", "Evidence-loss score ordering", BLUE, True),
        ("excess_activation_score", "excess_activation_incident", "excess_activation", "b", "Excess-activation score ordering", ORANGE, False),
    )
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0), sharey=True)
    source_rows: list[dict[str, object]] = []
    for ax, (score, outcome, endpoint, letter, title, color, positive_only) in zip(axes, panels):
        curves: list[list[dict[str, float]]] = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            local = [row for row in rows if int(row["seed"]) == seed and (not positive_only or int(row["has_defect"]) == 1)]
            if len(local) < 6:
                continue
            binned = bin_rows(local, score, outcome)
            curves.append(binned)
            x = np.asarray([item["score_quantile"] for item in binned])
            y = 100.0 * np.asarray([item["observed_rate"] for item in binned])
            ax.plot(x, y, "o-", color=color, alpha=0.45, linewidth=1.0, markersize=3.8,
                    markerfacecolor="white", markeredgewidth=0.8)
            for item in binned:
                source_rows.append({"endpoint": endpoint, "seed": seed, **item})
        mean_y = 100.0 * np.mean([[item["observed_rate"] for item in curve] for curve in curves], axis=0)
        x = np.arange(1, len(mean_y) + 1)
        ax.plot(x, mean_y, "o-", color=color, linewidth=2.0, markersize=4.8,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.7)
        ax.set_xticks(x, [str(i) for i in x])
        ax.set_xlabel("Within-repetition score quantile (low to high)")
        panel_title(ax, letter, title)
        quiet_grid(ax, axis="y")
    axes[0].set_ylabel("Observed incident rate (%)")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=INK, alpha=0.55, markerfacecolor="white", label="individual repetition"),
            Line2D([], [], marker="o", color=INK, linewidth=2.0, markerfacecolor=INK, label="mean across available repetitions"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.52, -0.01), fontsize=6.6,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.26, wspace=0.16)
    save(fig, "fig_risk_score_behavior")
    write_csv(OUTPUT / "fig_risk_score_behavior_source.csv", source_rows)


def severity_behavior(rows: list[dict[str, object]]) -> None:
    policy_rows = [row for row in rows if int(row["seed"]) == 211]
    accepted_rows = [row for row in policy_rows if int(row["accepted"]) == 1]
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.15), gridspec_kw={"width_ratios": (1.20, 0.86, 1.05)})

    ax = axes[0]
    left = np.zeros(len(SEVERITIES))
    hatches = ("", "//", "..", "xx", "\\\\", "++")
    for method, hatch in zip(METHODS, hatches):
        values = []
        for severity in SEVERITIES:
            local = [row for row in accepted_rows if row["severity"] == severity]
            values.append(100.0 * sum(row["selected_action"] == method for row in local) / len(local) if local else 0.0)
        ax.barh(np.arange(len(SEVERITIES)), values, left=left, height=0.60,
                color=METHOD_COLORS[method], edgecolor="white", linewidth=0.6,
                hatch=hatch, label=METHOD_LABELS[method])
        left += np.asarray(values)
    returned_counts = {
        severity: sum(1 for row in accepted_rows if row["severity"] == severity)
        for severity in SEVERITIES
    }
    ax.set_yticks(
        np.arange(len(SEVERITIES)),
        [SEVERITY_LABELS[s] + "\n" + f"{returned_counts[s]} returned" for s in SEVERITIES],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Selected-action composition (%)")
    panel_title(ax, "a", "Returned action mix")
    ax.legend(ncol=2, fontsize=5.9, loc="lower center", bbox_to_anchor=(0.52, -0.45),
              columnspacing=0.7, handletextpad=0.35)

    ax = axes[1]
    coverage_values = []
    coverage_labels = []
    for severity in SEVERITIES:
        local = [row for row in policy_rows if row["severity"] == severity]
        accepted = sum(int(row["accepted"]) for row in local)
        coverage_values.append(100.0 * accepted / len(local))
        coverage_labels.append(f"{accepted}/{len(local)}")
    xx = np.arange(len(SEVERITIES))
    ax.bar(xx, coverage_values, width=0.60, color=BLUE, edgecolor=BLUE_DARK, linewidth=0.6)
    for xpos, value, label in zip(xx, coverage_values, coverage_labels):
        ax.text(xpos, min(98, value + 3), label, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(xx, ["Mild", "Mod.", "Severe"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Conditional coverage (%)")
    panel_title(ax, "b", "Automatic-return workload")
    quiet_grid(ax, axis="y")

    ax = axes[2]
    width = 0.34
    loss_rates, activation_rates, loss_labels, activation_labels = [], [], [], []
    for severity in SEVERITIES:
        local = [row for row in accepted_rows if row["severity"] == severity]
        positive = [row for row in local if int(row["has_defect"]) == 1]
        loss = sum(int(row["evidence_loss_incident"]) for row in positive)
        activation = sum(int(row["excess_activation_incident"]) for row in local)
        loss_rates.append(100.0 * loss / len(positive) if positive else 0.0)
        activation_rates.append(100.0 * activation / len(local) if local else 0.0)
        loss_labels.append(f"{loss}/{len(positive)}")
        activation_labels.append(f"{activation}/{len(local)}")
    ax.bar(xx - width / 2, loss_rates, width, color=BLUE, edgecolor=BLUE_DARK, linewidth=0.6, label="Evidence loss")
    ax.bar(xx + width / 2, activation_rates, width, color=ORANGE, edgecolor="#8C4A12", linewidth=0.6, label="Excess activation")
    for index, (value, label) in enumerate(zip(loss_rates, loss_labels)):
        ax.text(index - width / 2, value + 1.2, label, ha="center", va="bottom", fontsize=6.0)
    for index, (value, label) in enumerate(zip(activation_rates, activation_labels)):
        ax.text(index + width / 2, value + 3.5, label, ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(xx, ["Mild", "Mod.", "Severe"])
    ax.set_ylim(0, max(32.0, max(loss_rates + activation_rates) * 1.5))
    ax.set_ylabel("Incident rate (%)")
    panel_title(ax, "c", "Returned-image incidents")
    ax.legend(loc="upper left", fontsize=6.1)
    quiet_grid(ax, axis="y")

    fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.25, wspace=0.40)
    save(fig, "fig_severity_behavior")
    write_csv(OUTPUT / "fig_severity_behavior_source.csv", rows)


def tuning_rule_sensitivity(targets: DualRiskTargets) -> list[dict[str, Any]]:
    """Reconstruct the post-hoc tuning-rule sensitivity reported in the supplement."""

    selectors = (
        ("half_margin", "Half-margin", choose_threshold),
        ("empirical_target", "Empirical target", choose_threshold_empirical_target),
        ("tuning_ucb", "Tuning UCB", choose_threshold_tuning_ucb),
    )
    rows: list[dict[str, Any]] = []
    for rule_key, rule_label, selector in selectors:
        for seed in SEEDS:
            folder = ROOT / "experiments" / FOLDER_TEMPLATE.format(seed=seed)
            models = fit_risk_models(
                read_rows(folder / "detector_val_records.csv"), seed, 0.25
            )
            calibration = build_policy_arrays(
                read_rows(folder / "calibration_records.csv"), models, targets, 0.25
            )["action_pool"]
            test = build_policy_arrays(
                read_rows(folder / "test_records.csv"), models, targets, 0.25
            )["action_pool"]
            tuning_mask = stable_tuning_mask(
                calibration["sample_id"].tolist(), tune_key=TUNE_KEY
            )
            tuning = subset(calibration, tuning_mask)
            certification = subset(calibration, ~tuning_mask)
            threshold = selector(
                tuning["score"],
                tuning["positive"],
                tuning["deletion"],
                tuning["invention"],
                targets,
            )
            if not np.isfinite(threshold):
                rows.append(
                    {
                        "rule": rule_key,
                        "rule_label": rule_label,
                        "seed": seed,
                        "threshold": None,
                        "loss_errors": None,
                        "loss_count": None,
                        "loss_ucb": 1.0,
                        "activation_errors": None,
                        "activation_count": None,
                        "activation_ucb": 1.0,
                        "certificate_decision": "no eligible gate",
                        "conditional_test_coverage": 0.0,
                        "pass_gated_test_coverage": 0.0,
                    }
                )
                continue

            certificate = certify_threshold(
                certification["score"],
                threshold,
                certification["positive"],
                certification["deletion"],
                certification["invention"],
                targets,
            )
            test_result = evaluate_threshold(
                test["score"],
                threshold,
                test["positive"],
                test["deletion"],
                test["invention"],
            )
            rows.append(
                {
                    "rule": rule_key,
                    "rule_label": rule_label,
                    "seed": seed,
                    "threshold": float(threshold),
                    "loss_errors": certificate["deletion_errors"],
                    "loss_count": certificate["deletion_denominator"],
                    "loss_ucb": certificate["deletion_ucb"],
                    "activation_errors": certificate["invention_errors"],
                    "activation_count": certificate["invention_denominator"],
                    "activation_ucb": certificate["invention_ucb"],
                    "certificate_decision": (
                        "pass" if certificate["passes"] else "fail"
                    ),
                    "conditional_test_coverage": test_result["coverage"],
                    "pass_gated_test_coverage": (
                        test_result["coverage"] if certificate["passes"] else 0.0
                    ),
                }
            )

    table_dir = ROOT / "paper" / "table_sources"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "table_tuning_rule_sensitivity_source.csv", rows)
    return rows


def risk_coverage_evidence() -> None:
    """Show certification risk-coverage and a clean-component planning bound."""

    targets = DualRiskTargets()
    tuning_rule_sensitivity(targets)
    source_rows: list[dict[str, Any]] = []

    seed = 211
    folder = ROOT / "experiments" / FOLDER_TEMPLATE.format(seed=seed)
    models = fit_risk_models(
        read_rows(folder / "detector_val_records.csv"), seed, 0.25
    )
    calibration = build_policy_arrays(
        read_rows(folder / "calibration_records.csv"), models, targets, 0.25
    )["action_pool"]
    tuning_mask = stable_tuning_mask(
        calibration["sample_id"].tolist(), tune_key=TUNE_KEY
    )
    certification = subset(calibration, ~tuning_mask)
    scores = np.asarray(certification["score"], dtype=float)
    order = np.argsort(scores, kind="stable")
    selected_threshold = float(
        load_seed(seed, 0.25)["policies"]["action_pool"]["threshold"]
    )
    gate_count = int(np.sum(scores <= selected_threshold))

    first_count = next(
        count
        for count in range(targets.min_accept, len(order) + 1)
        if int(np.asarray(certification["positive"])[order[:count]].sum())
        >= targets.min_positive_accept
    )
    prefix_counts = {
        int(value)
        for value in np.linspace(first_count, len(order), 36)
    }
    prefix_counts.add(gate_count)

    coverages: list[float] = []
    loss_ucbs: list[float] = []
    activation_ucbs: list[float] = []
    gate_result: dict[str, Any] | None = None
    for count in sorted(prefix_counts):
        threshold = float(scores[order[count - 1]])
        certificate = certify_threshold(
            certification["score"],
            threshold,
            certification["positive"],
            certification["deletion"],
            certification["invention"],
            targets,
        )
        coverage_pct = 100.0 * certificate["coverage"]
        loss_ucb_pct = 100.0 * certificate["deletion_ucb"]
        activation_ucb_pct = 100.0 * certificate["invention_ucb"]
        coverages.append(coverage_pct)
        loss_ucbs.append(loss_ucb_pct)
        activation_ucbs.append(activation_ucb_pct)
        is_gate = count == gate_count
        if is_gate:
            gate_result = certificate
        source_rows.append(
            {
                "panel": "certification_risk_coverage",
                "seed": seed,
                "accepted_prefix": count,
                "threshold": threshold,
                "coverage": certificate["coverage"],
                "accepted_positive": certificate["accepted_positive"],
                "accepted_clean": certificate["accepted_clean"],
                "loss_ucb": certificate["deletion_ucb"],
                "activation_ucb": certificate["invention_ucb"],
                "selected_tuning_gate": int(is_gate),
            }
        )

    if gate_result is None:
        raise RuntimeError("The tuning-selected gate is absent from the plotted prefixes")

    setup_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_W, 3.02),
        gridspec_kw={"width_ratios": (1.18, 0.92)},
    )

    ax = axes[0]
    ax.plot(
        coverages,
        loss_ucbs,
        color=BLUE,
        linewidth=1.8,
        marker="o",
        markersize=2.8,
        markevery=4,
        label="Evidence-loss UCB",
    )
    ax.plot(
        coverages,
        activation_ucbs,
        color=ORANGE,
        linewidth=1.8,
        marker="s",
        markersize=2.6,
        markevery=4,
        label="Excess-activation UCB",
    )
    ax.axhline(
        100.0 * targets.deletion,
        color=MID,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        label="15% target",
    )
    gate_coverage = 100.0 * gate_result["coverage"]
    ax.axvline(
        gate_coverage,
        color=INK,
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        alpha=0.8,
    )
    ax.scatter(
        [gate_coverage],
        [100.0 * gate_result["deletion_ucb"]],
        marker="D",
        s=38,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Tuning-selected gate",
    )
    ax.scatter(
        [gate_coverage],
        [100.0 * gate_result["invention_ucb"]],
        marker="D",
        s=38,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(
        0,
        max(
            32.0,
            5.0 * np.ceil(max(loss_ucbs + activation_ucbs) / 5.0),
        ),
    )
    ax.set_xlabel("Certification coverage (%)")
    ax.set_ylabel("One-sided exact upper bound (%)")
    panel_title(ax, "a", "Certification risk-coverage")
    quiet_grid(ax, axis="y")
    ax.legend(loc="upper left", fontsize=6.3, ncol=2)

    ax = axes[1]
    clean_counts = np.arange(1, 61)
    zero_incident_ucbs = np.asarray(
        [clopper_pearson_upper(0, int(count), targets.delta / 2) for count in clean_counts]
    )
    needed = int(
        next(
            count
            for count, bound in zip(clean_counts, zero_incident_ucbs)
            if bound <= targets.invention
        )
    )
    ax.plot(
        clean_counts,
        100.0 * zero_incident_ucbs,
        color=BLUE_DARK,
        linewidth=1.9,
        label="Best case: zero incidents",
    )
    ax.axhline(
        100.0 * targets.invention,
        color=MID,
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        label="15% target",
    )
    ax.axvline(
        needed,
        color=ORANGE,
        linestyle=(0, (2, 2)),
        linewidth=1.0,
        label=f"Minimum n = {needed}",
    )

    first_point = True
    for point_seed in SEEDS:
        policy = load_seed(point_seed, 0.25)["policies"]["action_pool"]
        threshold = policy.get("threshold")
        if threshold is None or not np.isfinite(float(threshold)):
            continue
        clean_count = int(policy["certificate"].get("accepted_clean") or 0)
        if clean_count <= 0:
            continue
        bound_pct = 100.0 * clopper_pearson_upper(
            0, clean_count, targets.delta / 2
        )
        ax.scatter(
            [clean_count],
            [bound_pct],
            s=34,
            marker="o",
            color=ORANGE,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label="Eligible primary gates" if first_point else None,
        )
        first_point = False
        source_rows.append(
            {
                "panel": "eligible_gate_clean_counts",
                "seed": point_seed,
                "accepted_clean": clean_count,
                "zero_incident_ucb": bound_pct / 100.0,
            }
        )

    for count, bound in zip(clean_counts, zero_incident_ucbs):
        source_rows.append(
            {
                "panel": "clean_component_planning_bound",
                "accepted_clean": int(count),
                "zero_incident_ucb": float(bound),
                "minimum_n_for_target": needed,
            }
        )

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Accepted clean images, n")
    ax.set_ylabel("Best-case U(0,n; 0.05) (%)")
    panel_title(ax, "b", "Clean-component evidence planning")
    quiet_grid(ax, axis="both")
    ax.legend(loc="upper right", fontsize=6.1)

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.89,
        bottom=0.20,
        wspace=0.31,
    )
    save(fig, "fig_risk_coverage")
    write_csv(OUTPUT / "fig_risk_coverage_source.csv", source_rows)

def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    style.OUTPUT = OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    action_tradeoffs()
    boundary_evidence()
    rows = collect_selected_rows()
    risk_score_behavior(rows)
    severity_behavior(rows)
    risk_coverage_evidence()
    print("Wrote quantitative figures to", OUTPUT, "from", len(rows), "selected-policy rows")


if __name__ == "__main__":
    main()
