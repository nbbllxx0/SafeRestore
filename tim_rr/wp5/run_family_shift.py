"""Experiment C: degradation-family shift stress test on Carinthia-S (outside the guarantee).

Fixed: detector/restorer checkpoints, detector threshold, the score models
(refitted exactly as in the definitive analysis from the frozen
detector_val_records.csv; class-balanced fits), and the QUALIFIED gates of the
definitive analysis (tim_upgrade/experiments/tim_analysis/
carinthia_s_seed{S}_floor0.25.json: familywise_portfolio selected policy and
certified threshold; and every familywise single-policy certified threshold).

Re-rendered: the TEST role only, under the published severity schedule with one
operator component replaced (wp5_eval.degrade_family_batch):
  nominal           published operator (control; must reproduce Table V)
  motion            linear motion PSF, length sqrt(12)*sigma_b (same 2nd moment), hashed angle; + Gaussian noise
  defocus           uniform disk PSF, radius 2*sigma_b (per-axis sigma = sigma_b); + Gaussian noise
  poisson_gaussian  Gaussian blur; noise var a*x + b, a = sigma_n^2, b = sigma_n^2/2 (= sigma_n^2 at x = 0.5)
  jpeg30/50/70      Gaussian blur; JPEG at fixed quality replaces the additive noise
All families keep the 2x bicubic downsampling and the per-image severity level.

Outputs (tim_rr/wp5/): family_shift/carinthia_seed{S}_{family}_test_records.csv,
exp_c_nominal_check.json, exp_c_portfolio.csv, exp_c_single.csv, exp_c_operator.csv
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import wp5_analysis as A
import wp5_eval as E

HERE = Path(__file__).resolve().parent
SEEDS = (201, 203, 207, 211, 223)
FAMILIES = ("nominal", "motion", "defocus", "poisson_gaussian", "jpeg30", "jpeg50", "jpeg70")
ALPHA = 0.15
FLOOR = 0.25


def main() -> None:
    device = torch.device("cuda")
    outdir = HERE / "family_shift"
    outdir.mkdir(exist_ok=True)
    # ---- inference
    nominal_check = {}
    for seed in SEEDS:
        started = time.time()
        detector, restorer, threshold, meta, folder = E.load_models("carinthia_s", seed, device)
        for family in FAMILIES:
            path = outdir / f"carinthia_seed{seed}_{family}_test_records.csv"
            if path.exists():
                continue
            mode = "nominal" if family == "nominal" else f"family:{family}"
            rows = E.evaluate("carinthia_s", "test", seed, detector, restorer, threshold, device,
                              mode=mode)[None]
            E.write_csv(path, rows)
        print("inference", seed, f"{time.time() - started:.0f}s", flush=True)
        del detector, restorer
        torch.cuda.empty_cache()

    # ---- evaluation with fixed score models and fixed qualified gates
    A.set_domain("carinthia_s")
    reported = A.reported
    targets = A.targets(ALPHA, 0.10)
    portfolio_rows, single_rows, operator_rows = [], [], []
    for seed in SEEDS:
        folder = Path(str(E.DATA["carinthia_s"]["run"]).format(seed=seed))
        definitive = json.loads((A.DEFINITIVE_DIR / f"carinthia_s_seed{seed}_floor{FLOOR:.2f}.json")
                                .read_text(encoding="utf-8"))
        training = A._orig_read_rows(folder / "detector_val_records.csv")
        models = reported.fit_risk_models(training, seed, FLOOR)
        pf = definitive["familywise_portfolio"]
        selected, gate = pf["selected_policy"], pf["selected_threshold"]
        import csv as _csv
        nominal_rows = list(_csv.DictReader((outdir / f"carinthia_seed{seed}_nominal_test_records.csv").open(encoding="utf-8")))
        rep = E.compare_records(nominal_rows, folder / "test_records.csv")
        nominal_check[str(seed)] = {"records_exact": rep.get("exact"),
                                    "label_mismatches": rep["label_mismatches"], "rows": rep["n_new"]}
        for family in FAMILIES:
            rows = A._orig_read_rows(outdir / f"carinthia_seed{seed}_{family}_test_records.csv")
            arrays = reported.build_policy_arrays(rows, models, targets, FLOOR)
            res = reported.evaluate_policy(arrays[selected], gate)
            row = {"seed": seed, "family": family, "selected_policy": selected, "gate": gate,
                   "test_n": res["n"], "accepted": res["accepted"],
                   "auto_acceptance": res["coverage"],
                   "loss_e": res["deletion_errors"], "loss_n": res["deletion_denominator"],
                   "loss_rate": res["deletion_risk"],
                   "act_e": res["invention_errors"], "act_n": res["invention_denominator"],
                   "act_rate": res["invention_risk"]}
            # "" when the gate accepts no test image (vacuous: nothing automated)
            row["meets_both_0.15"] = ("" if res["accepted"] == 0 else
                                      int((row["loss_rate"] or 0) <= ALPHA and (row["act_rate"] or 0) <= ALPHA))
            row["loss_exceeds"] = int(row["loss_rate"] is not None and row["loss_rate"] > ALPHA)
            row["act_exceeds"] = int(row["act_rate"] is not None and row["act_rate"] > ALPHA)
            if family == "nominal":
                ref = pf["selected_test"]
                same = all(res[k] == ref[k] for k in ("coverage", "deletion_risk", "invention_risk",
                                                       "accepted", "deletion_errors", "invention_errors"))
                nominal_check.setdefault(str(seed), {})["portfolio_test_identical_to_definitive"] = same
                nominal_check[str(seed)]["definitive"] = {k: ref[k] for k in ("coverage", "deletion_risk", "invention_risk")}
            portfolio_rows.append(row)
            for name, payload in definitive["familywise_single_policy"].items():
                thr = payload["certified_threshold"]
                if thr is None:
                    continue
                r2 = reported.evaluate_policy(arrays[name], thr)
                single_rows.append({"seed": seed, "family": family, "policy": name, "gate": thr,
                                    "auto_acceptance": r2["coverage"], "loss_rate": r2["deletion_risk"],
                                    "act_rate": r2["invention_risk"],
                                    "loss_e": r2["deletion_errors"], "loss_n": r2["deletion_denominator"],
                                    "act_e": r2["invention_errors"], "act_n": r2["invention_denominator"],
                                    "accepted": r2["accepted"],
                                    "loss_exceeds": int((r2["deletion_risk"] or 0) > ALPHA),
                                    "act_exceeds": int((r2["invention_risk"] or 0) > ALPHA),
                                    "exceeds": int((r2["deletion_risk"] or 0) > ALPHA or (r2["invention_risk"] or 0) > ALPHA)})
            # operator-level descriptors (no gate): raw and bicubic actions
            for method in ("raw_nearest", "bicubic", "learned"):
                local = [r for r in rows if r["method"] == method]
                pos = [r for r in local if r["has_defect"]]
                operator_rows.append({
                    "seed": seed, "family": family, "method": method,
                    "mean_psnr_db": float(np.mean([r["psnr"] for r in local])),
                    "loss_rate_pos": float(np.mean([r["pixel_recall"] < FLOOR for r in pos])),
                    "act_rate_all": float(np.mean([r["invention_error"] for r in local])),
                })
    # PSF second moments actually used (per published level), for the matched-severity statement
    import math
    from torchvision.transforms.functional import gaussian_blur
    from saferestore.degradation_schedule import LEVELS
    kernel_rows = []
    for level in LEVELS:
        size = max(3, int(round(level.blur_sigma * 6)) | 1)
        delta = torch.zeros(1, 1, 41, 41)
        delta[0, 0, 20, 20] = 1.0
        g = gaussian_blur(delta, [size, size], [level.blur_sigma, level.blur_sigma])[0, 0].numpy()
        yy, xx = np.mgrid[-20:21, -20:21]
        m = E.motion_kernel(math.sqrt(12.0) * level.blur_sigma, 0.0)
        hm = m.shape[0] // 2
        my, mx = np.mgrid[-hm:hm + 1, -hm:hm + 1]
        d = E.disk_kernel(2.0 * level.blur_sigma)
        hd = d.shape[0] // 2
        dy, dx = np.mgrid[-hd:hd + 1, -hd:hd + 1]
        kernel_rows.append({
            "level": level.name, "sigma_b": level.blur_sigma, "sigma_n": level.noise_std,
            "gaussian_sigma_x_px": float(np.sqrt((g * xx ** 2).sum())),
            "motion_length_px": math.sqrt(12.0) * level.blur_sigma,
            "motion_sigma_along_px": float(np.sqrt((m * mx ** 2).sum())),
            "defocus_radius_px": 2.0 * level.blur_sigma,
            "defocus_sigma_x_px": float(np.sqrt((d * dx ** 2).sum())),
            "pg_a": level.noise_std ** 2, "pg_b": level.noise_std ** 2 / 2.0,
            "jpeg_quality_variants": "30;50;70 (fixed per family)",
        })
    with (HERE / "exp_c_kernel_moments.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(kernel_rows[0]))
        w.writeheader()
        w.writerows(kernel_rows)
    (HERE / "exp_c_nominal_check.json").write_text(json.dumps(nominal_check, indent=2), encoding="utf-8")
    for name, rows in (("exp_c_portfolio.csv", portfolio_rows), ("exp_c_single.csv", single_rows),
                       ("exp_c_operator.csv", operator_rows)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(nominal_check, indent=2))


if __name__ == "__main__":
    main()
