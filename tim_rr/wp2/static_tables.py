"""WP2 deliverables 1, 4, 5 (no grid needed).

1. Exact attribute-sampling tables (one-sided Clopper-Pearson), with an
   independent binomial-CDF cross-check, and the hold-out accepted counts of
   every Carinthia-S portfolio selection from the definitive JSONs.
4. Design-effect tables and the largest intra-cluster correlation at which each
   Carinthia-S portfolio selection would still pass with effective n.
5. Failure-inclusive rebuild of Table IV (both domains) and the qualification
   vs test prevalence of each domain.

Writes sample_size.csv, holdout_accepted_counts.csv, design_effect.csv,
design_effect_max_rho.csv, table4_failure_inclusive.csv,
table4_per_seed.csv, prevalence.csv, static_tables.json.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

import wp2_common as W

HERE = Path(__file__).resolve().parent
DELTA = 0.10
ALLOC = {"single_delta/2": DELTA / 2, "portfolio_delta/16": DELTA / 16}


def write_csv(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def min_n_binom(e, alpha, d):
    """Independent route: smallest n with P(X <= e | n, alpha) <= d."""
    n = e + 1
    while binom.cdf(e, n, alpha) > d:
        n += 1
    return n


def cp_real(e, n, d):
    """Clopper-Pearson upper bound allowing real-valued (effective) e and n."""
    if n <= e:
        return 1.0
    return float(beta.ppf(1.0 - d, e + 1.0, n - e))


def min_nominal(e, alpha, d, de, scale_e):
    """Smallest nominal n whose effective sample n/de passes.

    scale_e=False: e incidents kept, n_eff = n/de (conservative).
    scale_e=True : Korn-Graubard-style, e_eff = e/de and n_eff = n/de.
    """
    n = e + 1
    while True:
        ee = e / de if scale_e else e
        if cp_real(ee, n / de, d) <= alpha:
            return n
        n += 1


def max_rho(e, n, m, d, alpha, scale_e):
    def passes(rho):
        de = 1 + (m - 1) * rho
        ee = e / de if scale_e else e
        return cp_real(ee, n / de, d) <= alpha
    if not passes(0.0):
        return None
    if passes(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if passes(mid) else (lo, mid)
    return lo


def main() -> None:
    out = {}
    # ---- 1. sample-size table -------------------------------------------
    rows, mismatches = [], 0
    for alpha in (0.10, 0.15, 0.20):
        for label, d in ALLOC.items():
            row = {"alpha": alpha, "allocation": label, "endpoint_level": d}
            for e in range(6):
                n1 = W.min_n(e, alpha, d)
                n2 = min_n_binom(e, alpha, d)
                mismatches += int(n1 != n2)
                row[f"e{e}"] = n1
            rows.append(row)
    write_csv("sample_size.csv", rows)
    out["sample_size_crosscheck_mismatches"] = mismatches

    # hold-out accepted counts of the definitive Carinthia-S portfolio selections
    acc_rows = []
    for seed in W.SEEDS:
        ref = json.loads((W.DEFINITIVE_DIR / f"carinthia_s_seed{seed}_floor0.25.json").read_text())
        pf = ref["familywise_portfolio"]
        sel = pf["selected_policy"]
        ev = pf["policies"][sel]
        n_pos, n_all = ev["deletion_denominator"], ev["invention_denominator"]
        d = pf["delta_alloc_per_policy"] / 2
        # largest incident count each endpoint could have absorbed at these n
        max_e_pos = max(e for e in range(n_pos) if cp_real(e, n_pos, d) <= 0.15)
        max_e_all = max(e for e in range(n_all) if cp_real(e, n_all, d) <= 0.15)
        acc_rows.append({
            "seed": seed, "selected_policy": sel,
            "holdout_n": ref["certification_n"],
            "accepted_positives": n_pos, "loss_incidents": ev["deletion_errors"],
            "loss_U": round(ev["deletion_ucb"], 4),
            "accepted_images": n_all, "activation_incidents": ev["invention_errors"],
            "activation_U": round(ev["invention_ucb"], 4),
            "holdout_coverage": round(ev["coverage"], 4),
            "endpoint_level": d,
            "max_loss_incidents_allowed": max_e_pos,
            "max_activation_incidents_allowed": max_e_all,
            "source": f"tim_upgrade/experiments/tim_analysis/carinthia_s_seed{seed}_floor0.25.json",
        })
    write_csv("holdout_accepted_counts.csv", acc_rows)

    # ---- 4. design effect ------------------------------------------------
    de_rows = []
    for m in (5, 10, 25):
        for rho in (0.0, 0.01, 0.05, 0.1, 0.2):
            de = 1 + (m - 1) * rho
            row = {"m": m, "rho": rho, "design_effect": round(de, 3),
                   "n_eff_per_100_nominal": round(100 / de, 2)}
            for label, d in ALLOC.items():
                short = "single" if label.startswith("single") else "portfolio"
                for e in (0, 1, 2):
                    row[f"{short}_e{e}_nominal_n"] = min_nominal(e, 0.15, d, de, False)
                    row[f"{short}_e{e}_nominal_n_KG"] = min_nominal(e, 0.15, d, de, True)
            de_rows.append(row)
    write_csv("design_effect.csv", de_rows)

    rho_rows = []
    for a in acc_rows:
        for m in (5, 10, 25):
            r = {"seed": a["seed"], "selected_policy": a["selected_policy"], "m": m}
            for scale_e, tag in ((False, "conservative"), (True, "KG")):
                rl = max_rho(a["loss_incidents"], a["accepted_positives"], m, a["endpoint_level"], 0.15, scale_e)
                ra = max_rho(a["activation_incidents"], a["accepted_images"], m, a["endpoint_level"], 0.15, scale_e)
                r[f"max_rho_loss_{tag}"] = None if rl is None else round(rl, 4)
                r[f"max_rho_activation_{tag}"] = None if ra is None else round(ra, 4)
                r[f"max_rho_both_{tag}"] = None if rl is None or ra is None else round(min(rl, ra), 4)
            rho_rows.append(r)
    write_csv("design_effect_max_rho.csv", rho_rows)

    # ---- 5. failure-inclusive Table IV ----------------------------------
    per_seed, t4 = [], []
    for domain in W.DOMAINS:
        reps = []
        for seed in W.SEEDS:
            ref = json.loads((W.DEFINITIVE_DIR / f"{domain}_seed{seed}_floor0.25.json").read_text())
            test_rows = W._orig_read_rows(Path(str(W.DOMAINS[domain]).format(seed=seed)) / "test_records.csv")
            rec = {"domain": domain, "seed": seed}
            for method in ("raw_nearest", "learned"):
                local = [r for r in test_rows if r["method"] == method]
                pos = [r for r in local if r["has_defect"]]
                rec[f"ungated_{method}_n"] = len(local)
                rec[f"ungated_{method}_positives"] = len(pos)
                rec[f"ungated_{method}_loss_rate"] = sum(r["pixel_recall"] < 0.25 for r in pos) / len(pos)
                rec[f"ungated_{method}_activation_rate"] = sum(r["clean_fpr"] > 0.002 for r in local) / len(local)
            fam = ref["familywise_single_policy"]
            for name in ("raw_nearest", "learned", "restoration_pool", "action_pool"):
                q = fam[name]["certified_threshold"] is not None
                t = fam[name].get("test") or {}
                rec[f"fam_{name}_qualified"] = q
                rec[f"fam_{name}_cov_fi"] = t.get("coverage", 0.0) if q else 0.0
                rec[f"fam_{name}_loss_per_positive_fi"] = (t["deletion_errors"] / t["positive_n"]) if q else 0.0
                rec[f"fam_{name}_activation_per_image_fi"] = (t["invention_errors"] / t["n"]) if q else 0.0
                rec[f"fam_{name}_loss_rate_cond"] = t.get("deletion_risk") if q else None
                rec[f"fam_{name}_activation_rate_cond"] = t.get("invention_risk") if q else None
            pf = ref["familywise_portfolio"]
            q = pf["selected_policy"] is not None
            t = pf.get("selected_test") or {}
            rec["portfolio_selected"] = pf["selected_policy"]
            rec["portfolio_qualified"] = q
            rec["portfolio_cov_fi"] = t.get("coverage", 0.0) if q else 0.0
            rec["portfolio_loss_per_positive_fi"] = (t["deletion_errors"] / t["positive_n"]) if q else 0.0
            rec["portfolio_activation_per_image_fi"] = (t["invention_errors"] / t["n"]) if q else 0.0
            rec["portfolio_loss_rate_cond"] = t.get("deletion_risk") if q else None
            rec["portfolio_activation_rate_cond"] = t.get("invention_risk") if q else None
            reps.append(rec)
            per_seed.append(rec)

        def mean(key):
            vals = [r[key] for r in reps if r[key] is not None]
            return (float(np.mean(vals)) if vals else None), len(vals)

        def rng(key):
            vals = [r[key] for r in reps if r[key] is not None]
            return (float(min(vals)), float(max(vals))) if vals else (None, None)

        for method, label in (("raw_nearest", "Always acquired image (ungated)"),
                              ("learned", "Always learned restore (ungated)")):
            t4.append({
                "domain": domain, "rule": label, "qualified_reps": "n/a (no qualification)",
                "auto_acceptance_fi_mean": 1.0, "auto_acceptance_fi_min": 1.0, "auto_acceptance_fi_max": 1.0,
                "loss_rate_cond_mean": mean(f"ungated_{method}_loss_rate")[0],
                "loss_rate_cond_range": rng(f"ungated_{method}_loss_rate"),
                "activation_rate_cond_mean": mean(f"ungated_{method}_activation_rate")[0],
                "activation_rate_cond_range": rng(f"ungated_{method}_activation_rate"),
                "loss_per_test_positive_fi_mean": mean(f"ungated_{method}_loss_rate")[0],
                "activation_per_test_image_fi_mean": mean(f"ungated_{method}_activation_rate")[0],
                "cond_auto_acceptance_mean_qualified_only": None,
            })
        for name, label in (("raw_nearest", "Acquired image, familywise gate (full delta)"),
                            ("learned", "Learned restore, familywise gate (full delta)"),
                            ("restoration_pool", "Restoration-only pool, familywise gate (full delta; selection not licensed)"),
                            ("action_pool", "All-action pool, familywise gate (full delta)")):
            qn = sum(r[f"fam_{name}_qualified"] for r in reps)
            covs = [r[f"fam_{name}_cov_fi"] for r in reps]
            qcovs = [c for c, r in zip(covs, reps) if r[f"fam_{name}_qualified"]]
            t4.append({
                "domain": domain, "rule": label, "qualified_reps": f"{qn}/5",
                "auto_acceptance_fi_mean": float(np.mean(covs)),
                "auto_acceptance_fi_min": float(min(covs)), "auto_acceptance_fi_max": float(max(covs)),
                "loss_rate_cond_mean": mean(f"fam_{name}_loss_rate_cond")[0],
                "loss_rate_cond_range": rng(f"fam_{name}_loss_rate_cond"),
                "activation_rate_cond_mean": mean(f"fam_{name}_activation_rate_cond")[0],
                "activation_rate_cond_range": rng(f"fam_{name}_activation_rate_cond"),
                "loss_per_test_positive_fi_mean": mean(f"fam_{name}_loss_per_positive_fi")[0],
                "activation_per_test_image_fi_mean": mean(f"fam_{name}_activation_per_image_fi")[0],
                "cond_auto_acceptance_mean_qualified_only": float(np.mean(qcovs)) if qcovs else None,
            })
        qn = sum(r["portfolio_qualified"] for r in reps)
        covs = [r["portfolio_cov_fi"] for r in reps]
        qcovs = [c for c, r in zip(covs, reps) if r["portfolio_qualified"]]
        t4.append({
            "domain": domain, "rule": "Qualified selective rule (portfolio, delta/8 per policy)",
            "qualified_reps": f"{qn}/5",
            "auto_acceptance_fi_mean": float(np.mean(covs)),
            "auto_acceptance_fi_min": float(min(covs)), "auto_acceptance_fi_max": float(max(covs)),
            "loss_rate_cond_mean": mean("portfolio_loss_rate_cond")[0],
            "loss_rate_cond_range": rng("portfolio_loss_rate_cond"),
            "activation_rate_cond_mean": mean("portfolio_activation_rate_cond")[0],
            "activation_rate_cond_range": rng("portfolio_activation_rate_cond"),
            "loss_per_test_positive_fi_mean": mean("portfolio_loss_per_positive_fi")[0],
            "activation_per_test_image_fi_mean": mean("portfolio_activation_per_image_fi")[0],
            "cond_auto_acceptance_mean_qualified_only": float(np.mean(qcovs)) if qcovs else None,
        })
    for r in t4:
        for k in ("loss_rate_cond_range", "activation_rate_cond_range"):
            lo, hi = r[k]
            r[k] = "" if lo is None else f"{lo:.4f}-{hi:.4f}"
    write_csv("table4_failure_inclusive.csv", t4)
    write_csv("table4_per_seed.csv", per_seed)

    # ---- prevalence --------------------------------------------------------
    prev = []
    for domain in W.DOMAINS:
        for seed in W.SEEDS:
            folder = Path(str(W.DOMAINS[domain]).format(seed=seed))
            cal = [r for r in W._orig_read_rows(folder / "calibration_records.csv") if r["method"] == "raw_nearest"]
            tst = [r for r in W._orig_read_rows(folder / "test_records.csv") if r["method"] == "raw_nearest"]
            mask = W.reported.stable_tuning_mask([r["sample_id"] for r in cal], tune_key="v4tunecert")
            pos = np.asarray([r["has_defect"] for r in cal], dtype=bool)
            prev.append({
                "domain": domain, "seed": seed,
                "qualification_n": len(cal), "qualification_positives": int(pos.sum()),
                "tuning_n": int(mask.sum()), "tuning_positives": int(pos[mask].sum()),
                "holdout_n": int((~mask).sum()), "holdout_positives": int(pos[~mask].sum()),
                "test_n": len(tst), "test_positives": int(sum(r["has_defect"] for r in tst)),
            })
    write_csv("prevalence.csv", prev)
    out["holdout_accepted_counts"] = acc_rows
    (HERE / "static_tables.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("crosscheck mismatches:", mismatches)
    for r in rows:
        print(r)
    for r in acc_rows:
        print(r)


if __name__ == "__main__":
    main()
