"""Experiment A, step 2: tolerance sweep through the definitive analysis (R2.4).

Inputs: tim_rr/wp5/ksdd2_tol{R}_seed{S}/ (run_ksdd2_tolerance.py).
Outputs (all in tim_rr/wp5/):
  exp_a_reproduction.json   seed-level comparison of the 8-px run with the definitive JSON
  exp_a_results/            full analysis JSON per (radius, seed, label)
  exp_a_per_seed.csv        one row per (radius, seed, label): qualification summary
  exp_a_incidence.csv       one row per (radius, seed, role): incidence, boundary band,
                            secondary_peak statistics (raw action)
  exp_a_summary.csv         one row per (radius, label): aggregated over seeds
Labels: standard (ring excluded from the clean region, as published) and strict
(ring calls count as clean-area activation; see wp5_eval.BAND_COLUMNS).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import wp5_analysis as A

HERE = Path(__file__).resolve().parent
RADII = (0, 2, 4, 6, 8, 12, 16)
SEEDS = (201, 203, 207, 211, 223)
LABELS = ("standard", "strict")
ROLES = {"detector_val": "detector_val_records.csv", "calibration": "calibration_records.csv",
         "test": "test_records.csv"}
F0 = 0.002


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def auc(x: np.ndarray, y: np.ndarray) -> float | None:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    # average ranks for ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def incidence_rows(radius: int, seed: int) -> list[dict]:
    folder = HERE / f"ksdd2_tol{radius}_seed{seed}"
    rows_out = []
    for role, filename in ROLES.items():
        rows = [r for r in read(folder / filename) if r["method"] == "raw_nearest"]
        pos = [r for r in rows if r["has_defect"] == "1"]
        cln = [r for r in rows if r["has_defect"] == "0"]
        total_calls_pos = sum(int(r["total_calls"]) for r in pos)
        ring_calls_pos = sum(int(r["ring_calls"]) for r in pos)
        clean_calls_pos = sum(int(r["clean_calls"]) for r in pos)
        in_mask_pos = total_calls_pos - ring_calls_pos - clean_calls_pos
        sec = np.array([float(r["secondary_peak"]) for r in rows])
        inv = np.array([int(r["invention_error"]) for r in rows])
        rows_out.append({
            "radius": radius, "seed": seed, "role": role, "action": "raw",
            "n_pos": len(pos), "n_clean": len(cln),
            "act_pos_e": sum(int(r["invention_error"]) for r in pos),
            "act_clean_e": sum(int(r["invention_error"]) for r in cln),
            "strict_act_pos_e": sum(int(r["strict_invention_error"]) for r in pos),
            "strict_act_clean_e": sum(int(r["strict_invention_error"]) for r in cln),
            "loss_pos_e": sum(int(r["deletion_error"]) for r in pos),
            # boundary band (positive images; the ring is empty on clean images)
            "pos_calls_total": total_calls_pos,
            "pos_calls_in_mask": in_mask_pos,
            "pos_calls_in_ring": ring_calls_pos,
            "pos_calls_clean": clean_calls_pos,
            "ring_share_of_pos_calls": ring_calls_pos / total_calls_pos if total_calls_pos else None,
            "ring_share_of_outside_mask_calls": (ring_calls_pos / (ring_calls_pos + clean_calls_pos)
                                                 if ring_calls_pos + clean_calls_pos else None),
            "pos_images_any_ring_call": sum(int(r["ring_calls"]) > 0 for r in pos),
            "pos_images_ring_only_excess": sum(
                int(r["ring_calls"]) / max(int(r["clean_pixels"]), 1) > F0 for r in pos),
            "mean_ring_pixels_pos": float(np.mean([int(r["ring_pixels"]) for r in pos])) if pos else None,
            # secondary_peak (raw action, all images in the role)
            "secondary_peak_zero_frac": float(np.mean(sec == 0.0)),
            "secondary_peak_median": float(np.median(sec)),
            "secondary_peak_mean": float(np.mean(sec)),
            "secondary_peak_auc_activation": auc(sec, inv),
        })
    return rows_out


def main() -> None:
    (HERE / "exp_a_results").mkdir(exist_ok=True)
    # Reproduction gate: 8 px must reproduce the definitive analysis exactly (all seeds).
    repro = {}
    for seed in SEEDS:
        result = A.analyze(HERE / f"ksdd2_tol8_seed{seed}", "kolektor_sdd2", seed)
        d = A.compare_with_definitive(result, "kolektor_sdd2", seed)
        a, name = A.alpha_star(result)
        repro[str(seed)] = {"identical_to_definitive": not d, "n_diffs": len(d), "first_diffs": d[:5],
                            "alpha_star": a, "alpha_star_policy": name,
                            "alpha_star_ties": A.alpha_star_ties(result)}
    inc = incidence_rows(8, 201)
    cal = [r for r in inc if r["role"] == "calibration"][0]
    repro["seed201_qualification_raw_incidence"] = {
        "positive": f"{cal['act_pos_e']}/{cal['n_pos']}", "clean": f"{cal['act_clean_e']}/{cal['n_clean']}"}
    (HERE / "exp_a_reproduction.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    print(json.dumps(repro, indent=2))
    if not all(repro[str(s)]["identical_to_definitive"] for s in SEEDS):
        raise SystemExit("REPRODUCTION FAILED - stop")

    per_seed, incidence = [], []
    for radius in RADII:
        for seed in SEEDS:
            incidence.extend(incidence_rows(radius, seed))
            for label in LABELS:
                result = A.analyze(HERE / f"ksdd2_tol{radius}_seed{seed}", "kolektor_sdd2", seed, label=label)
                (HERE / "exp_a_results" / f"r{radius}_seed{seed}_{label}.json").write_text(
                    json.dumps(A.json_safe(result), indent=1, allow_nan=False), encoding="utf-8")
                row = {"radius": radius, "seed": seed, "label": label}
                row.update(A.summarize(result))
                per_seed.append(row)
            print("done", radius, seed, flush=True)
        A.clear_cache()

    with (HERE / "exp_a_per_seed.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)
    with (HERE / "exp_a_incidence.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(incidence[0]))
        w.writeheader()
        w.writerows(incidence)

    summary = []
    for radius in RADII:
        cal = [r for r in incidence if r["radius"] == radius and r["role"] == "calibration"]
        tst = [r for r in incidence if r["radius"] == radius and r["role"] == "test"]
        for label in LABELS:
            local = [r for r in per_seed if r["radius"] == radius and r["label"] == label]
            astars = [r["alpha_star"] for r in local]
            pre = "strict_" if label == "strict" else ""
            summary.append({
                "radius": radius, "label": label,
                # incidence: raw action, qualification split, summed over seeds (images recur across seeds)
                "qual_act_pos_rate_mean": float(np.mean([r[pre + "act_pos_e"] / r["n_pos"] for r in cal])),
                "qual_act_clean_rate_mean": float(np.mean([r[pre + "act_clean_e"] / r["n_clean"] for r in cal])),
                "qual_act_pos_e_by_seed": ";".join(str(r[pre + "act_pos_e"]) for r in cal),
                "qual_act_clean_e_by_seed": ";".join(str(r[pre + "act_clean_e"]) for r in cal),
                "qual_n_pos": cal[0]["n_pos"], "qual_n_clean": cal[0]["n_clean"],
                "test_act_pos_rate_mean": float(np.mean([r[pre + "act_pos_e"] / r["n_pos"] for r in tst])),
                "test_act_clean_rate_mean": float(np.mean([r[pre + "act_clean_e"] / r["n_clean"] for r in tst])),
                "qual_ring_share_of_pos_calls_mean": float(np.mean([r["ring_share_of_pos_calls"] for r in cal])),
                "qual_ring_share_of_outside_mask_calls_mean": float(np.mean(
                    [r["ring_share_of_outside_mask_calls"] for r in cal])),
                "qual_pos_ring_only_excess_by_seed": ";".join(str(r["pos_images_ring_only_excess"]) for r in cal),
                "qual_secondary_peak_zero_frac_mean": float(np.mean([r["secondary_peak_zero_frac"] for r in cal])),
                "qual_secondary_peak_auc_act_mean": float(np.mean([r["secondary_peak_auc_activation"] for r in cal])),
                "portfolio_qualified_reps": sum(r["portfolio_qualified"] for r in local),
                "portfolio_selections": ";".join(str(r["portfolio_selected"]) for r in local),
                "single_qualified_total": sum(r["n_single_qualified"] for r in local),
                "tuning_selected_qualified": sum(r["tuning_selected_qualified"] for r in local),
                "alpha_star_by_seed": ";".join(f"{a:.3f}" for a in astars),
                "alpha_star_min": min(astars), "alpha_star_median": float(np.median(astars)),
                "alpha_star_max": max(astars),
            })
    with (HERE / "exp_a_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("wrote exp_a_*")


if __name__ == "__main__":
    main()
