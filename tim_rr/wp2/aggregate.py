"""WP2: aggregate per-seed grid / sweep rows into cell-level tables, a compact
markdown summary, and the stricter-requirement figure.

Inputs : grid_seed_<domain>.csv, sweep_seed_<domain>.csv (from run_grid.py)
Outputs: grid_cells.csv, grid_summary.md, sweep_curves.csv,
         fig_alpha_sweep.pdf, fig_alpha_sweep.png

All results are DESCRIPTIVE / RETROSPECTIVE: the records were used during
method development (records/2026-10-25-tim-rr-wp1-provenance.md), and the
grid contains 108 cells per domain (no multiplicity control across cells).

Definitions (per cell = domain x r0 x f0 x alpha x delta; 5 repetitions):
  marg_passes__p    marginal tuned-gate certificate passes for policy p (of 5)
  fam_passes__p     familywise single-policy fixed-sequence qualifications (full delta)
  portfolio_qualified  Bonferroni portfolio (delta/8 per policy) qualified repetitions
  *_cov_fi          FAILURE-INCLUSIVE mean held-out test automatic-acceptance
                    fraction: a repetition without a qualified/passing gate contributes 0
  adv_reps_total    sum_p fam_passes - sum_p marg_passes
  adv_cov_fi_mean   mean over the 8 policies of (fam_cov_fi - marg_cov_fi)
  port_minus_best_marg_cov_fi  portfolio_cov_fi minus the best single marginal policy's
                    failure-inclusive coverage, the best policy picked post hoc (an
                    unlicensed oracle; favours the marginal side)
  flag_fam_not_better  adv_reps_total <= 0
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
POLICIES = ("raw_nearest", "bilinear", "bicubic", "smooth_bicubic", "unsharp_bicubic",
            "learned", "restoration_pool", "action_pool")
SHORT = {"raw_nearest": "raw", "bilinear": "bil", "bicubic": "bic", "smooth_bicubic": "smo",
         "unsharp_bicubic": "shp", "learned": "lrn", "restoration_pool": "rpool",
         "action_pool": "apool"}
DOMAINS = ("carinthia_s", "kolektor_sdd2")
DNAME = {"carinthia_s": "Carinthia-S", "kolektor_sdd2": "KolektorSDD2"}


def read(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("True", "False"):
                r[k] = v == "True"
            elif v == "":
                r[k] = None
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def group(rows):
    cells = {}
    for r in rows:
        key = (r["domain"], r["r0"], r["f0"], r["alpha"], r["delta"])
        cells.setdefault(key, []).append(r)
    return cells


def _vals(reps, key):
    return [r[key] for r in reps if r[key] is not None]


def cell_summary(key, reps):
    domain, r0, f0, alpha, delta = key
    assert len(reps) == 5, (key, len(reps))
    out = {"domain": domain, "r0": r0, "f0": f0, "alpha": alpha, "delta": delta}
    marg_tot = fam_tot = 0
    advs, n_lt, n_gt = [], 0, 0
    marg_cov, fam_cov = {}, {}
    exceed = {"marg": [0, 0], "fam": [0, 0], "port": [0, 0]}  # [qualified pairs, pairs with a test rate > alpha]
    for p in POLICIES:
        m = sum(bool(r[f"marg_pass__{p}"]) for r in reps)
        f = sum(bool(r[f"fam_pass__{p}"]) for r in reps)
        out[f"marg_passes__{p}"] = m
        out[f"fam_passes__{p}"] = f
        marg_tot += m
        fam_tot += f
        n_lt += f < m
        n_gt += f > m
        marg_cov[p] = float(np.mean([r[f"marg_cov_fi__{p}"] for r in reps]))
        fam_cov[p] = float(np.mean([r[f"fam_cov_fi__{p}"] for r in reps]))
        out[f"marg_cov_fi__{p}"] = round(marg_cov[p], 4)
        out[f"fam_cov_fi__{p}"] = round(fam_cov[p], 4)
        advs.append(fam_cov[p] - marg_cov[p])
        for r in reps:
            for tag, pre in (("marg", "marg"), ("fam", "fam")):
                if r[f"{pre}_pass__{p}"]:
                    exceed[tag][0] += 1
                    d, i = r[f"{pre}_test_del__{p}"], r[f"{pre}_test_inv__{p}"]
                    exceed[tag][1] += int((d is not None and d > alpha) or (i is not None and i > alpha))
    out["marg_passes_total"] = marg_tot
    out["fam_passes_total"] = fam_tot
    out["adv_reps_total"] = fam_tot - marg_tot
    out["n_policies_fam_gt_marg"] = n_gt
    out["n_policies_fam_lt_marg"] = n_lt
    out["adv_cov_fi_mean"] = round(float(np.mean(advs)), 4)
    pq = sum(bool(r["portfolio_qualified"]) for r in reps)
    out["portfolio_qualified"] = pq
    out["portfolio_selections"] = "|".join(SHORT.get(r["portfolio_selected"], "-") if r["portfolio_selected"] else "-" for r in reps)
    out["portfolio_cov_fi"] = round(float(np.mean([r["portfolio_test_coverage_fi"] for r in reps])), 4)
    dels = _vals(reps, "portfolio_test_deletion_risk")
    invs = _vals(reps, "portfolio_test_invention_risk")
    out["portfolio_test_loss_mean"] = round(float(np.mean(dels)), 4) if dels else None
    out["portfolio_test_loss_max"] = round(float(max(dels)), 4) if dels else None
    out["portfolio_test_act_mean"] = round(float(np.mean(invs)), 4) if invs else None
    out["portfolio_test_act_max"] = round(float(max(invs)), 4) if invs else None
    for r in reps:
        if r["portfolio_qualified"]:
            exceed["port"][0] += 1
            d, i = r["portfolio_test_deletion_risk"], r["portfolio_test_invention_risk"]
            exceed["port"][1] += int((d is not None and d > alpha) or (i is not None and i > alpha))
    for tag in ("marg", "fam", "port"):
        out[f"{tag}_qualified_pairs"] = exceed[tag][0]
        out[f"{tag}_test_exceed_pairs"] = exceed[tag][1]
    best_marg = max(POLICIES, key=lambda p: marg_cov[p])
    best_fam = max(POLICIES, key=lambda p: fam_cov[p])
    out["best_marg_policy_posthoc"] = best_marg
    out["best_marg_cov_fi_posthoc"] = round(marg_cov[best_marg], 4)
    out["best_fam_policy_posthoc"] = best_fam
    out["best_fam_cov_fi_posthoc"] = round(fam_cov[best_fam], 4)
    out["port_minus_best_marg_cov_fi"] = round(out["portfolio_cov_fi"] - marg_cov[best_marg], 4)
    out["tuning_selected_qualified"] = sum(bool(r["tuning_selected_qualified"]) for r in reps)
    out["tuning_selected_cov_fi"] = round(float(np.mean([r["tuning_selected_test_coverage_fi"] for r in reps])), 4)
    out["flag_fam_not_better"] = out["adv_reps_total"] <= 0
    out["flag_both_zero"] = marg_tot == 0 and fam_tot == 0
    out["flag_any_policy_fam_lt_marg"] = n_lt > 0
    out["flag_portfolio_below_best_marg_cov"] = out["port_minus_best_marg_cov_fi"] < 0
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def grid_outputs():
    rows = []
    for d in DOMAINS:
        rows += read(HERE / f"grid_seed_{d}.csv")
    cells = group(rows)
    out = [cell_summary(k, v) for k, v in sorted(cells.items())]
    write_csv(HERE / "grid_cells.csv", out)

    lines = ["# WP2 sensitivity grid: compact summary",
             "",
             "DESCRIPTIVE / RETROSPECTIVE. The records were used during method development "
             "(records/2026-10-25-tim-rr-wp1-provenance.md). 108 cells per domain, 5 repetitions "
             "each; no multiplicity control across cells. Source: grid_cells.csv "
             "(per-seed rows: grid_seed_<domain>.csv).",
             "",
             "Failure-inclusive (FI) acceptance: mean held-out test automatic-acceptance fraction "
             "over the 5 repetitions, a repetition with no qualified gate contributing 0.",
             ""]
    for d in DOMAINS:
        loc = [c for c in out if c["domain"] == d]
        nb = [c for c in loc if c["flag_fam_not_better"]]
        nb_zero = [c for c in nb if c["flag_both_zero"]]
        lt = [c for c in loc if c["flag_any_policy_fam_lt_marg"]]
        pb = [c for c in loc if c["flag_portfolio_below_best_marg_cov"]]
        lines += [f"## {DNAME[d]}", "",
                  f"- Cells: {len(loc)}. Familywise single-policy qualifications exceed marginal passes "
                  f"(summed over 8 policies) in {len(loc) - len(nb)} cells; not better in {len(nb)} "
                  f"({len(nb_zero)} of these are 0 vs 0).",
                  f"- Cells where at least one policy has fewer familywise qualifications than marginal passes: {len(lt)}.",
                  f"- Cells where portfolio FI acceptance is below the post-hoc best marginal policy's FI acceptance: {len(pb)}.",
                  f"- Portfolio qualified repetitions, distribution over cells: "
                  + ", ".join(f"{k}/5: {sum(c['portfolio_qualified'] == k for c in loc)}" for k in range(6)) + ".",
                  f"- Qualified (policy, repetition) pairs with a held-out test incident rate above alpha "
                  f"(either endpoint): marginal {sum(c['marg_test_exceed_pairs'] for c in loc)}/"
                  f"{sum(c['marg_qualified_pairs'] for c in loc)}, familywise single "
                  f"{sum(c['fam_test_exceed_pairs'] for c in loc)}/{sum(c['fam_qualified_pairs'] for c in loc)}, "
                  f"portfolio selection {sum(c['port_test_exceed_pairs'] for c in loc)}/"
                  f"{sum(c['port_qualified_pairs'] for c in loc)} (descriptive; test rates are point estimates).",
                  "",
                  "Portfolio qualified reps / FI acceptance (%) by cell. Rows r0, f0; columns alpha x delta.",
                  ""]
        alphas = sorted({c["alpha"] for c in loc})
        deltas = sorted({c["delta"] for c in loc})
        hdr = "| r0 | f0 | " + " | ".join(f"a={a:.2f} d={dd:.2f}" for a in alphas for dd in deltas) + " |"
        lines += [hdr, "|" + "---|" * (2 + len(alphas) * len(deltas))]
        idx = {(c["r0"], c["f0"], c["alpha"], c["delta"]): c for c in loc}
        for r0 in sorted({c["r0"] for c in loc}):
            for f0 in sorted({c["f0"] for c in loc}):
                cells_txt = []
                for a in alphas:
                    for dd in deltas:
                        c = idx[(r0, f0, a, dd)]
                        cells_txt.append(f"{c['portfolio_qualified']}/{pct(c['portfolio_cov_fi'])}")
                lines.append(f"| {r0:.2f} | {f0:.3f} | " + " | ".join(cells_txt) + " |")
        lines += ["", "Familywise-minus-marginal qualified repetitions, summed over 8 policies "
                  "(adv_reps_total; marked * where <= 0 with a nonzero side, 0z where both are 0).", "",
                  hdr, "|" + "---|" * (2 + len(alphas) * len(deltas))]
        for r0 in sorted({c["r0"] for c in loc}):
            for f0 in sorted({c["f0"] for c in loc}):
                cells_txt = []
                for a in alphas:
                    for dd in deltas:
                        c = idx[(r0, f0, a, dd)]
                        if c["flag_both_zero"]:
                            cells_txt.append("0z")
                        else:
                            cells_txt.append(f"{c['adv_reps_total']:+d}" + ("*" if c["flag_fam_not_better"] else ""))
                lines.append(f"| {r0:.2f} | {f0:.3f} | " + " | ".join(cells_txt) + " |")
        flagged = [c for c in nb if not c["flag_both_zero"]] + [c for c in lt if not c["flag_fam_not_better"]]
        if flagged:
            lines += ["", "Flagged cells (familywise not better in total, or some policy familywise < marginal):", "",
                      "| r0 | f0 | alpha | delta | marginal total | familywise total | policies fw<marg | portfolio q | portfolio FI % | best marginal FI % (post hoc) |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for c in flagged:
                lines.append(f"| {c['r0']:.2f} | {c['f0']:.3f} | {c['alpha']:.2f} | {c['delta']:.2f} | "
                             f"{c['marg_passes_total']} | {c['fam_passes_total']} | {c['n_policies_fam_lt_marg']} | "
                             f"{c['portfolio_qualified']} | {pct(c['portfolio_cov_fi'])} | {pct(c['best_marg_cov_fi_posthoc'])} |")
        lines.append("")
    (HERE / "grid_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def sweep_outputs():
    curves = []
    for d in DOMAINS:
        rows = read(HERE / f"sweep_seed_{d}.csv")
        for key, reps in sorted(group(rows).items()):
            c = cell_summary(key, reps)
            curves.append({k: c[k] for k in (
                "domain", "alpha", "delta", "r0", "f0", "portfolio_qualified", "portfolio_cov_fi",
                "portfolio_selections", "best_fam_policy_posthoc", "best_fam_cov_fi_posthoc",
                "tuning_selected_qualified", "tuning_selected_cov_fi",
                "best_marg_policy_posthoc", "best_marg_cov_fi_posthoc",
                "portfolio_test_loss_max", "portfolio_test_act_max")})
    write_csv(HERE / "sweep_curves.csv", curves)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
    for ax, d, tag in zip(axes, DOMAINS, ("(a)", "(b)")):
        loc = [c for c in curves if c["domain"] == d]
        a = [c["alpha"] for c in loc]
        ax.plot(a, [100 * c["portfolio_cov_fi"] for c in loc], "-o", ms=2.5, lw=1.2,
                color="#1f4e9c", label="Portfolio (δ/8 per policy)")
        ax.plot(a, [100 * c["best_fam_cov_fi_posthoc"] for c in loc], "--s", ms=2.2, lw=1.0,
                color="#c4572a", label="Best single policy (post hoc, full δ)")
        ax.plot(a, [100 * c["tuning_selected_cov_fi"] for c in loc], ":^", ms=2.2, lw=1.0,
                color="#3a8a3a", label="Tuning-selected single policy")
        ax.axvline(0.15, color="0.4", lw=0.8, ls="-.")
        ax.set_xlim(0.045, 0.305)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        ax.grid(True, lw=0.4, color="0.85")
        ax.set_ylabel("Test auto-acceptance (%)")
        ax.text(0.02, 0.93, f"{tag} {DNAME[d]}", transform=ax.transAxes, fontsize=8, va="top")
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, bbox_to_anchor=(0.0, 0.84))
    axes[1].set_xlabel("Target incident probability α (both endpoints)")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(HERE / "fig_alpha_sweep.pdf")
    fig.savefig(HERE / "fig_alpha_sweep.png", dpi=300)
    return curves


if __name__ == "__main__":
    g = grid_outputs()
    s = sweep_outputs()
    print("cells", len(g), "sweep points", len(s))
