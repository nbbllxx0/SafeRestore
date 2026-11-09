"""Experiment D: are the WP2 KSDD2 portfolio test exceedances a prevalence-mismatch effect?

WP2 (tim_rr/wp2/grid_cells.csv) found 9 of 67 qualified KSDD2 portfolio
selections whose held-out TEST activation rate exceeds alpha.  The
qualification split is positivity-stratified (109/604 positive) while the test
role has 110/1004 positive, and excess activation is more frequent on clean
images.  For each of the 67 qualified (cell, seed) selections this script
re-runs the WP2 harness (tim_rr/wp2/wp2_common.py, imported read-only), splits
the accepted test set into positives and clean images, and reweights the test
activation rate to the qualification-split mixture:

  w_pos = pi_q / pi_t,  w_clean = (1 - pi_q) / (1 - pi_t)
  act_rw = (w_pos * e_pos + w_clean * e_clean) / (w_pos * n_pos + w_clean * n_clean)

with e_* / n_* the activation incidents / accepted counts among test
positives / clean images.  The same split is reported for the hold-out.

Outputs (tim_rr/wp5/): exp_d_prevalence.csv, exp_d_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WP2 = HERE.parent / "wp2"
sys.path.insert(0, str(WP2))
import wp2_common as W  # noqa: E402  (read-only import; run with python -B)

reported = W.reported


def split_counts(policy: dict, threshold: float, mask=None) -> dict:
    score = np.asarray(policy["score"], dtype=np.float64)
    pos = np.asarray(policy["positive"], dtype=bool)
    inv = np.asarray(policy["invention"]).astype(bool)
    if mask is not None:
        score, pos, inv = score[mask], pos[mask], inv[mask]
    acc = score <= threshold
    return {"n": int(len(score)), "n_pos_total": int(pos.sum()),
            "acc_pos": int((acc & pos).sum()), "acc_clean": int((acc & ~pos).sum()),
            "act_pos": int((acc & pos & inv).sum()), "act_clean": int((acc & ~pos & inv).sum())}


def main() -> None:
    cells = list(csv.DictReader((WP2 / "grid_cells.csv").open(encoding="utf-8")))
    k_cells = [c for c in cells if c["domain"] == "kolektor_sdd2"]
    total_q = sum(int(c["port_qualified_pairs"]) for c in k_cells)
    total_x = sum(int(c["port_test_exceed_pairs"]) for c in k_cells)
    seeds = list(csv.DictReader((WP2 / "grid_seed_kolektor_sdd2.csv").open(encoding="utf-8")))
    qualified = [r for r in seeds if r["portfolio_qualified"] == "True"]
    rows = []
    for r in sorted(qualified, key=lambda x: (int(x["seed"]), float(x["r0"]), float(x["f0"]))):
        seed = int(r["seed"])
        r0, f0, alpha, delta = (float(r[k]) for k in ("r0", "f0", "alpha", "delta"))
        res = W.analyze("kolektor_sdd2", seed, r0, f0, alpha, delta)
        pf = res["familywise_portfolio"]
        sel, thr = pf["selected_policy"], pf["selected_threshold"]
        assert sel == r["portfolio_selected"], (sel, r["portfolio_selected"])
        folder = Path(str(W.DOMAINS["kolektor_sdd2"]).format(seed=seed))
        training = W.rta.read_rows(folder / "detector_val_records.csv")
        models = reported.fit_risk_models(training, seed, r0)
        targets = W.make_targets(alpha, delta)
        cal_rows = W.rta.read_rows(folder / "calibration_records.csv")
        test_rows = W.rta.read_rows(folder / "test_records.csv")
        cal = reported.build_policy_arrays(cal_rows, models, targets, r0)
        test = reported.build_policy_arrays(test_rows, models, targets, r0)
        tuning_mask = reported.stable_tuning_mask(cal["action_pool"]["sample_id"].tolist(), tune_key="v4tunecert")
        t = split_counts(test[sel], thr)
        h = split_counts(cal[sel], thr, ~tuning_mask)
        q_all = split_counts(cal[sel], thr)  # whole qualification split, for pi_q
        pi_q = q_all["n_pos_total"] / q_all["n"]
        pi_t = t["n_pos_total"] / t["n"]
        w_pos, w_clean = pi_q / pi_t, (1 - pi_q) / (1 - pi_t)
        n_acc = t["acc_pos"] + t["acc_clean"]
        act_test = (t["act_pos"] + t["act_clean"]) / n_acc
        act_rw = ((w_pos * t["act_pos"] + w_clean * t["act_clean"])
                  / (w_pos * t["acc_pos"] + w_clean * t["acc_clean"]))
        # consistency with WP2's stored test activation
        assert abs(act_test - float(r["portfolio_test_invention_risk"])) < 1e-12
        rows.append({
            "seed": seed, "r0": r0, "f0": f0, "alpha": alpha, "delta": delta, "selected": sel,
            "pi_qual": pi_q, "pi_test": pi_t,
            "test_acc_pos": t["acc_pos"], "test_acc_clean": t["acc_clean"],
            "test_act_pos": t["act_pos"], "test_act_clean": t["act_clean"],
            "test_act_rate_pos": t["act_pos"] / t["acc_pos"] if t["acc_pos"] else None,
            "test_act_rate_clean": t["act_clean"] / t["acc_clean"] if t["acc_clean"] else None,
            "test_act_rate": act_test,
            "test_act_rate_reweighted_to_qual": act_rw,
            "test_loss_rate": float(r["portfolio_test_deletion_risk"]) if r["portfolio_test_deletion_risk"] else None,
            "exceeds_alpha_test": int(act_test > alpha or (r["portfolio_test_deletion_risk"] and float(r["portfolio_test_deletion_risk"]) > alpha)),
            "exceeds_alpha_test_act": int(act_test > alpha),
            "exceeds_alpha_reweighted_act": int(act_rw > alpha),
            "holdout_acc_pos": h["acc_pos"], "holdout_acc_clean": h["acc_clean"],
            "holdout_act_pos": h["act_pos"], "holdout_act_clean": h["act_clean"],
            "holdout_act_rate": (h["act_pos"] + h["act_clean"]) / max(h["acc_pos"] + h["acc_clean"], 1),
            "holdout_frac_pos_accepted": h["acc_pos"] / max(h["acc_pos"] + h["acc_clean"], 1),
            "test_frac_pos_accepted": t["acc_pos"] / max(n_acc, 1),
        })
    with (HERE / "exp_d_prevalence.csv").open("w", newline="", encoding="utf-8") as hdl:
        w = csv.DictWriter(hdl, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ex = [x for x in rows if x["exceeds_alpha_test"]]
    pooled = lambda key, sub: int(sum(x[key] for x in sub))
    summary = {
        "grid_cells_port_qualified_pairs": total_q, "grid_cells_port_test_exceed_pairs": total_x,
        "reproduced_qualified": len(rows), "reproduced_exceed": len(ex),
        "exceed_act_only": sum(x["exceeds_alpha_test_act"] for x in rows),
        "exceed_after_reweighting": sum(x["exceeds_alpha_reweighted_act"] for x in rows),
        "exceed_cases": [{k: x[k] for k in ("seed", "r0", "f0", "alpha", "delta", "selected",
                                             "test_act_rate", "test_act_rate_reweighted_to_qual",
                                             "test_act_rate_pos", "test_act_rate_clean",
                                             "test_acc_pos", "test_acc_clean")} for x in ex],
        "pooled_over_67": {
            "test_act_rate_pos": pooled("test_act_pos", rows) / pooled("test_acc_pos", rows),
            "test_act_rate_clean": pooled("test_act_clean", rows) / max(pooled("test_acc_clean", rows), 1),
            "holdout_act_rate_pos": pooled("holdout_act_pos", rows) / pooled("holdout_acc_pos", rows),
            "holdout_act_rate_clean": pooled("holdout_act_clean", rows) / max(pooled("holdout_acc_clean", rows), 1),
        },
    }
    (HERE / "exp_d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
