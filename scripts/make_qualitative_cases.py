"""Generate readable illustrative outcomes for one marginally certified policy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

from saferestore.certification import DualRiskTargets
from saferestore.degradation import restore_methods
from saferestore.degradation_schedule import degrade_mixed_batch
from saferestore.torch_data import PublicSegmentationDataset
from saferestore.torch_models import CompactRestorer, TinyUNet
from scripts.analyze_router_pool import read_rows
from scripts.run_reported_analysis import METHODS, build_policy_arrays, fit_risk_models
import scripts.plot_style_manuscript as style
from scripts.plot_style_manuscript import BLUE, FIG_W, INK, LIGHT, MID, ORANGE, save, setup_style


SEED = 211
PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / f"experiments/v4_carinthia_primary_trainseed{SEED}"
SPLIT = PROJECT / "splits/carinthia_s_v4.json"
ANALYSIS = PROJECT / "experiments/v4_reported_analysis_primary" / f"seed_{SEED}_floor_0.25.json"
# Row headers replace the old left gutter, so panels use the full text width.
ROW_HEADER_Y = (0.968, 0.658, 0.348)
ACTION_LABELS = {
    "raw_nearest": "Raw",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "smooth_bicubic": "Smoothed bicubic",
    "unsharp_bicubic": "Sharpened bicubic",
    "learned": "Learned",
}


def resolve_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    manifest = PROJECT / "configs" / "carinthia_s.local.json"
    if manifest.exists():
        return Path(json.loads(manifest.read_text(encoding="utf-8"))["local_root"])
    raise SystemExit("Provide --data-root or create configs/carinthia_s.local.json")


def square_roi(mask: np.ndarray, minimum: int = 88, padding: int = 20) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise RuntimeError("cannot form a defect ROI from an empty mask")
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    size = min(max(minimum, x1 - x0 + 2 * padding, y1 - y0 + 2 * padding), mask.shape[0], mask.shape[1])
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = min(max(0, cx - size // 2), mask.shape[1] - size)
    top = min(max(0, cy - size // 2), mask.shape[0] - size)
    return left, top, left + size, top + size


def choose_cases(policy, records, threshold: float) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for index, sample_id in enumerate(policy["sample_id"]):
        method = str(policy["method"][index])
        score = float(policy["score"][index])
        rec = next(r for r in records if r["sample_id"] == sample_id and r["method"] == method)
        if method == "learned" and score <= threshold and rec["severity"] == "severe":
            selected.append((str(sample_id), "Returned learned display"))
            break
    for index, sample_id in enumerate(policy["sample_id"]):
        method = str(policy["method"][index])
        score = float(policy["score"][index])
        rec = next(r for r in records if r["sample_id"] == sample_id and r["method"] == method)
        if score <= threshold and int(rec["invention_error"]) == 1:
            selected.append((str(sample_id), "Returned display with activation incident"))
            break
    review_candidates: list[tuple[str, str]] = []
    for index, sample_id in enumerate(policy["sample_id"]):
        method = str(policy["method"][index])
        rec = next(r for r in records if r["sample_id"] == sample_id and r["method"] == method)
        if float(policy["score"][index]) > threshold and int(rec["has_defect"]) == 1:
            review_candidates.append((str(sample_id), str(rec["severity"])))
    severe = next((sample_id for sample_id, severity in review_candidates if severity == "severe"), None)
    if severe is None and review_candidates:
        severe = review_candidates[0][0]
    if severe is not None:
        selected.append((severe, "Assigned to review"))
    if len(selected) != 3:
        raise RuntimeError(f"could not locate three illustration cases: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "paper" / "figure_outputs")
    args = parser.parse_args()

    style.OUTPUT = args.output_dir
    dataset = PublicSegmentationDataset(
        dataset="carinthia_s",
        root=resolve_data_root(args.data_root),
        split_file=SPLIT,
        split="test",
        augment=False,
    )
    detector = TinyUNet()
    detector.load_state_dict(torch.load(EXPERIMENT / "detector.pt", map_location="cpu", weights_only=True))
    restorer = CompactRestorer()
    restorer.load_state_dict(torch.load(EXPERIMENT / "restorer.pt", map_location="cpu", weights_only=True))
    detector.eval()
    restorer.eval()

    records = read_rows(EXPERIMENT / "test_records.csv")
    models = fit_risk_models(read_rows(EXPERIMENT / "detector_val_records.csv"), SEED, 0.25)
    policy = build_policy_arrays(records, models, DualRiskTargets(), 0.25)["action_pool"]
    policy_index = {str(sample_id): index for index, sample_id in enumerate(policy["sample_id"])}
    result = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    gate_threshold = float(result["policies"]["action_pool"]["threshold"])
    detector_threshold = json.loads((EXPERIMENT / "results.json").read_text(encoding="utf-8"))["detector_threshold"]
    cases = choose_cases(policy, records, gate_threshold)
    wanted = {sample_id for sample_id, _ in cases}
    items = {item["sample_id"]: item for item in dataset if item["sample_id"] in wanted}
    if set(items) != wanted:
        raise RuntimeError("an illustration sample is absent from the test split")

    setup_style()
    fig, axes = plt.subplots(3, 4, figsize=(FIG_W, 5.75))
    titles = ("Reference context", "Raw ROI", "Bicubic ROI", "Ranked-action ROI")
    source_rows: list[dict[str, object]] = []

    for row_index, (sample_id, case_label) in enumerate(cases):
        item = items[sample_id]
        high = item["image"].unsqueeze(0)
        mask = item["mask"].squeeze(0).numpy() > 0.5
        low, bicubic, severities = degrade_mixed_batch(high, [sample_id], split="test", corruption_key="v4corr:eval")
        with torch.no_grad():
            candidates = restore_methods(low, tuple(high.shape[-2:]))
            candidates["learned"] = restorer(bicubic)
            probabilities = {
                method: torch.sigmoid(detector(candidate))[0, 0].numpy()
                for method, candidate in candidates.items()
            }
        row_lookup = {row["method"]: row for row in records if row["sample_id"] == sample_id}
        selected_index = policy_index[sample_id]
        selected_method = str(policy["method"][selected_index])
        selected_score = float(policy["score"][selected_index])
        returned = selected_score <= gate_threshold
        anchor = mask.copy()
        if not anchor.any():
            selected_calls = probabilities[selected_method] >= detector_threshold
            if selected_calls.any():
                anchor = selected_calls
            else:
                anchor = np.zeros_like(mask, dtype=bool)
                anchor[np.unravel_index(np.argmax(probabilities[selected_method]), anchor.shape)] = True
        x0, y0, x1, y1 = square_roi(anchor)
        displayed = (
            ("reference", high[0, 0].numpy(), None, False),
            ("raw_nearest", candidates["raw_nearest"][0, 0].numpy(), probabilities["raw_nearest"], True),
            ("bicubic", candidates["bicubic"][0, 0].numpy(), probabilities["bicubic"], True),
            (selected_method, candidates[selected_method][0, 0].numpy(), probabilities[selected_method], True),
        )
        # One decimal convention per row so the incident margin is legible.
        fpr_decimals = 4 if case_label.startswith("Returned display with activation") else 2
        for column, (method, full_image, probability, crop) in enumerate(displayed):
            ax = axes[row_index, column]
            if crop:
                image = full_image[y0:y1, x0:x1]
                local_mask = mask[y0:y1, x0:x1]
                local_probability = probability[y0:y1, x0:x1]
            else:
                image = full_image
                local_mask = mask
                local_probability = None
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if local_mask.any():
                ax.contour(local_mask, levels=[0.5], colors=[BLUE], linewidths=1.0)
            if not crop:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor=BLUE, linewidth=0.9,
                                       linestyle=(0, (3, 2))))
            elif local_probability is not None:
                calls = local_probability >= detector_threshold
                if calls.min() != calls.max():
                    ax.contour(calls, levels=[0.5], colors=[ORANGE], linewidths=0.85, linestyles="dashed")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2 if column == 3 else 0.6)
                spine.set_color(BLUE if column == 3 else LIGHT)
            if row_index == 0:
                ax.set_title(titles[column], fontsize=7.8, fontweight="bold", pad=4)
            if method == "reference":
                ax.set_xlabel(f"Example {chr(65 + row_index)}", fontsize=6.6, color=MID, labelpad=2)
                recall = None
                clean_fpr = None
            else:
                metric = row_lookup[method]
                recall = 100.0 * float(metric["pixel_recall"])
                clean_fpr = 100.0 * float(metric["clean_fpr"])
                activation_incident_panel = (
                    column == 3
                    and case_label.startswith("Returned display with activation")
                    and clean_fpr > 0.20
                )
                fpr_text = f"{clean_fpr:.{fpr_decimals}f}"
                ax.set_xlabel(
                    f"recall {recall:.0f}%  |  clean FPR {fpr_text}%",
                    fontsize=6.7,
                    color=INK,
                    labelpad=2,
                )
            if (
                column == 3
                and case_label.startswith("Returned display with activation")
                and clean_fpr is not None
                and clean_fpr > 0.20
            ):
                ax.text(
                    0.50,
                    -0.155,
                    "ACTIVATION INCIDENT\nclean FPR > 0.20%",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=5.2,
                    fontweight="bold",
                    linespacing=1.25,
                    multialignment="center",
                    color=ORANGE,
                    bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FFF3E8",
                          "edgecolor": ORANGE, "linewidth": 0.6},
                    clip_on=False,
                )
            if column == 3:
                # The panels are square and height-constrained, so the badge is
                # stacked rather than set on one line that would overrun them.
                badge = "RETURNED" if returned else "REVIEW"
                ax.text(0.955, 0.955, f"{ACTION_LABELS[selected_method]}\n{badge}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=5.6, fontweight="bold", color="white",
                        linespacing=1.3, multialignment="right",
                        bbox={"boxstyle": "round,pad=0.26", "facecolor": BLUE if returned else MID,
                              "edgecolor": "none"})
            source_rows.append(
                {
                    "case": row_index + 1,
                    "case_label": case_label,
                    "example_label": f"Example {chr(65 + row_index)}",
                    "sample_id": sample_id,
                    "severity": severities[0],
                    "displayed_action": method,
                    "ranked_action": selected_method,
                    "gate_score": selected_score,
                    "gate_threshold": gate_threshold,
                    "returned": int(returned),
                    "pixel_recall": recall,
                    "clean_fpr_percent": clean_fpr,
                    "roi_x0": x0,
                    "roi_y0": y0,
                    "roi_x1": x1,
                    "roi_y1": y1,
                }
            )
        letter = chr(97 + row_index)
        fig.text(0.015, ROW_HEADER_Y[row_index],
                 rf"$\bf{{{letter}}}$  {case_label}, {severities[0]} observation",
                 ha="left", va="center", fontsize=8.0, color=INK)

    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, linewidth=1.2, label="provided annotation boundary"),
            Line2D([], [], color=ORANGE, linewidth=1.0, linestyle="dashed", label="detector call"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.57, 0.002), fontsize=7.1,
    )
    fig.text(0.995, 0.064,
             "Dashed boxes define pixel-space ROIs; physical scale is unavailable in the public metadata.",
             ha="right", va="bottom", fontsize=6.3, color=MID)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.895, bottom=0.115, wspace=0.06, hspace=0.53)
    save(fig, "fig_qualitative_cases")

    with (style.OUTPUT / "fig_qualitative_cases_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()

