"""Print the markdown tables of WP5_REPORT.md from the WP5 CSV/JSON outputs (stdout only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (201, 203, 207, 211, 223)
PNAME = {"raw_nearest": "raw", "bilinear": "bilinear", "bicubic": "bicubic", "smooth_bicubic": "smoothed",
         "unsharp_bicubic": "sharpened", "learned": "learned", "restoration_pool": "restoration pool",
         "action_pool": "all-action pool", "None": "none", "": "none"}


def read(name):
    return list(csv.DictReader((HERE / name).open(encoding="utf-8")))


def f(x, nd=3):
    return "n/a" if x in ("", None) else f"{float(x):.{nd}f}"


def pct(x):
    return "n/a" if x in ("", None) else f"{100 * float(x):.1f}"


def table_a():
    s = read("exp_a_summary.csv")
    print("| R (px) | Labels | Qual. positives act. e (of 109), per seed | Qual. clean act. e (of 495), per seed | Test pos. / clean act. rate (mean) | Portfolio qualified reps | Single-policy qualifications (of 40) | α* per seed |")
    print("|---|---|---|---|---|---|---|---|")
    for r in s:
        print(f"| {r['radius']} | {r['label']} | {r['qual_act_pos_e_by_seed'].replace(';', ', ')} (mean {f(r['qual_act_pos_rate_mean'])}) | "
              f"{r['qual_act_clean_e_by_seed'].replace(';', ', ')} (mean {f(r['qual_act_clean_rate_mean'])}) | "
              f"{f(r['test_act_pos_rate_mean'])} / {f(r['test_act_clean_rate_mean'])} | {r['portfolio_qualified_reps']}/5 | "
              f"{r['single_qualified_total']} | {r['alpha_star_by_seed'].replace(';', ', ')} |")
    print()
    print("| R (px) | Ring share of all calls on positives | Ring share of outside-mask calls on positives | Positives whose ring calls alone exceed f0, per seed (of 109) | secondary_peak = 0 (fraction) | secondary_peak AUC for activation |")
    print("|---|---|---|---|---|---|")
    for r in s:
        if r["label"] != "standard":
            continue
        print(f"| {r['radius']} | {f(r['qual_ring_share_of_pos_calls_mean'])} | {f(r['qual_ring_share_of_outside_mask_calls_mean'])} | "
              f"{r['qual_pos_ring_only_excess_by_seed'].replace(';', ', ')} | {f(r['qual_secondary_peak_zero_frac_mean'])} | "
              f"{f(r['qual_secondary_peak_auc_act_mean'])} |")


def table_b():
    s = read("exp_b_summary.csv")
    print("| Domain | Arm | Action | Loss rate (positives) | Act. rate positives | Act. rate clean | Act. rate all |")
    print("|---|---|---|---|---|---|---|")
    for r in s:
        if r["role"] != "calibration" or r["action"] in ("bilinear", "bicubic"):
            continue
        arm = "native" if r["arm"] == "native" else "nominal"
        print(f"| {r['domain']} | {arm} | {r['action']} | {f(r['loss_rate_mean'])} | {f(r['act_pos_rate_mean'])} | "
              f"{f(r['act_clean_rate_mean'])} | {f(r['act_all_rate_mean'])} |")
    print()
    p = read("exp_b_per_seed.csv")
    print("| Domain | Seed | Portfolio (δ/6) selection | Hold-out loss e/n | Hold-out act. e/n | Test auto-acc. (%) | Test loss (%) | Test act. (%) | Single-policy qualified | Tuning-selected | α* (policy) | Portfolio at δ/8 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in p:
        if r["variant"] != "delta_over_6":
            continue
        r8 = next(x for x in p if x["variant"] != "delta_over_6" and x["domain"] == r["domain"] and x["seed"] == r["seed"])
        sel = PNAME[r["portfolio_selected"]] if r["portfolio_qualified"] == "1" else "none"
        single = ", ".join(PNAME[x] for x in r["single_qualified_policies"].split(";")) if r["single_qualified_policies"] else "none"
        ts = PNAME[r["tuning_selected_policy"]] + (" (qualified)" if r["tuning_selected_qualified"] == "1" else " (not qualified)")
        sel8 = PNAME[r8["portfolio_selected"]] if r8["portfolio_qualified"] == "1" else "none"
        print(f"| {r['domain']} | {r['seed']} | {sel} | {r.get('portfolio_holdout_loss') or '-'} | {r.get('portfolio_holdout_activation') or '-'} | "
              f"{pct(r['portfolio_test_coverage'])} | {pct(r['portfolio_test_loss'])} | {pct(r['portfolio_test_activation'])} | "
              f"{single} | {ts} | {f(r['alpha_star'])} ({PNAME[r['alpha_star_policy']]}) | {sel8} |")


def table_c():
    rows = read("exp_c_portfolio.csv")
    single = read("exp_c_single.csv")
    op = read("exp_c_operator.csv")
    fams = list(dict.fromkeys(r["family"] for r in rows))
    print("| Family | Mean PSNR raw / bicubic (dB) | Raw-action loss / act. rate | Portfolio gate, per seed 201/203/207/211/223: test auto-acc. % / loss % (e/n) / act. % | Seeds with an endpoint > 0.15 | Seeds with < 5% auto-acc. | Single-policy gates with an endpoint > 0.15 (of 27) |")
    print("|---|---|---|---|---|---|---|")
    for fam in fams:
        loc = [r for r in rows if r["family"] == fam]
        o_raw = [r for r in op if r["family"] == fam and r["method"] == "raw_nearest"]
        o_bic = [r for r in op if r["family"] == fam and r["method"] == "bicubic"]
        cells = []
        for r in loc:
            cells.append(f"{pct(r['auto_acceptance'])} / {pct(r['loss_rate'])} ({r['loss_e']}/{r['loss_n']}) / {pct(r['act_rate'])}")
        exceed = sum(int(r["loss_exceeds"]) or int(r["act_exceeds"]) for r in loc)
        collapse = sum(float(r["auto_acceptance"]) < 0.05 for r in loc)
        s_loc = [r for r in single if r["family"] == fam]
        print(f"| {fam} | {np.mean([float(r['mean_psnr_db']) for r in o_raw]):.2f} / {np.mean([float(r['mean_psnr_db']) for r in o_bic]):.2f} | "
              f"{np.mean([float(r['loss_rate_pos']) for r in o_raw]):.3f} / {np.mean([float(r['act_rate_all']) for r in o_raw]):.3f} | "
              f"{'; '.join(cells)} | {exceed}/5 | {collapse}/5 | {sum(int(r['exceeds']) for r in s_loc)}/{len(s_loc)} |")


def table_d():
    rows = read("exp_d_prevalence.csv")
    print("| Seed | r0 | f0 | α | δ | Selected | Test act. pos e/n | Test act. clean e/n | Test act. rate | Reweighted | Hold-out act. pos e/n | Hold-out act. clean e/n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["exceeds_alpha_test"] != "1":
            continue
        print(f"| {r['seed']} | {r['r0']} | {r['f0']} | {r['alpha']} | {r['delta']} | {PNAME[r['selected']]} | "
              f"{r['test_act_pos']}/{r['test_acc_pos']} | {r['test_act_clean']}/{r['test_acc_clean']} | {f(r['test_act_rate'])} | "
              f"{f(r['test_act_rate_reweighted_to_qual'])} | {r['holdout_act_pos']}/{r['holdout_acc_pos']} | {r['holdout_act_clean']}/{r['holdout_acc_clean']} |")
    s = json.loads((HERE / "exp_d_summary.json").read_text(encoding="utf-8"))
    print()
    print(json.dumps({k: s[k] for k in s if k != "exceed_cases"}, indent=1))
    rw = [float(r["test_act_rate_reweighted_to_qual"]) - float(r["alpha"]) for r in rows]
    raw = [float(r["test_act_rate"]) - float(r["alpha"]) for r in rows]
    print("max reweighted - alpha over 67:", max(rw), " max test - alpha:", max(raw))


if __name__ == "__main__":
    for name, fn in (("A", table_a), ("B", table_b), ("C", table_c), ("D", table_d)):
        print(f"\n### Table {name}\n")
        fn()
