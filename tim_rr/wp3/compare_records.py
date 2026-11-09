"""WP3 part 1: reproduce the published numbers and run the matched
comparison on the frozen records (retrospective, descriptive).

Run:  python -B tim_rr/wp3/compare_records.py
Writes: tim_rr/wp3/records_comparison.json, tim_rr/wp3/records_comparison.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import wp3_data as D
import wp3_methods as M

OUT = Path(__file__).resolve().parent


def tuning_selected_name(sd: D.SeedData) -> str:
    names = sorted(sd.policies)
    return max(names, key=lambda n: (sd.policies[n].tuning_promise, n))


def prepare(sd: D.SeedData, role: str = "hold"):
    """Hold-out gate counts (R = 1) and p-values for every policy."""
    t = D.TARGETS
    pv_cp, pv_hb, n_acc = {}, {}, {}
    for name, pd in sd.policies.items():
        c = D.gate_counts(getattr(pd, role), pd.grid)
        c = {k: v[None, :] for k, v in c.items()}
        pv_cp[name] = M.gate_pvalues(c, t.deletion, t.invention, "cp")
        pv_hb[name] = M.gate_pvalues(c, t.deletion, t.invention, "hb")
        n_acc[name] = c["n_act"]
        idx = np.searchsorted(pd.grid, pd.tuned_threshold)
        pd.tuned_idx = int(idx) if (idx < len(pd.grid) and pd.grid[idx] == pd.tuned_threshold) else -1
        if np.isfinite(pd.tuned_threshold):
            assert pd.tuned_idx >= 0, "tuned gate not on grid"
    return pv_cp, pv_hb, n_acc


def crc_gate(arr: dict, grid: np.ndarray, alpha_loss: float, alpha_act: float) -> int:
    """Conformal risk control (arXiv 2208.02814) operating point, B = 1.

    Losses (nondecreasing in the gate):  loss endpoint on positives,
    l_i = 1{accepted_i and loss incident_i};  activation endpoint on all
    units, l_i = 1{accepted_i and activation incident_i}.  For each
    endpoint the largest gate with (n/(n+1)) Rhat + 1/(n+1) <= alpha; the
    deployed gate is the smaller of the two.  Guarantee: E[l_{n+1}] <= alpha
    (expectation over calibration and test draw), i.e. a bound on the
    JOINT rate P(accepted and incident | positive) -- not the conditional
    risk and not with probability 1 - delta."""
    c = D.gate_counts(arr, grid)
    n_pos = int(arr["positive"].sum())
    n_all = len(arr["score"])
    ok_l = (n_pos / (n_pos + 1)) * c["e_del"] / max(n_pos, 1) + 1 / (n_pos + 1) <= alpha_loss
    ok_a = (n_all / (n_all + 1)) * c["e_act"] / n_all + 1 / (n_all + 1) <= alpha_act
    # monotone losses -> admissible set is a prefix; take its last element
    def last_prefix(ok):
        if not ok[0]:
            return -1
        bad = np.flatnonzero(~ok)
        return (bad[0] - 1) if bad.size else len(ok) - 1
    return min(last_prefix(ok_l), last_prefix(ok_a))


def evaluate(sd: D.SeedData, name: str, threshold: float) -> dict:
    r = D.reported.evaluate_policy(sd.test_full[name], threshold)
    return {"coverage": r["coverage"], "deletion_risk": r["deletion_risk"],
            "invention_risk": r["invention_risk"], "accepted": r["accepted"],
            "accepted_positive": r["accepted_positive"]}


def main() -> None:
    t = D.TARGETS
    names_sorted = D.POLICY_NAMES_SORTED
    all_rows = {}
    repro = {}
    for domain in D.DOMAINS:
        rows = []
        for seed in D.SEEDS:
            # --- 1. published pipeline, unmodified
            D.set_domain(domain)
            folder = Path(str(D.DOMAINS[domain]).format(seed=seed))
            orig = D.rta.analyze_domain_seed(folder, seed, D.FLOOR, t)
            # --- 2. WP3 implementations on the same arrays
            sd = D.load_seed(domain, seed)
            pv_cp, pv_hb, n_acc = prepare(sd)
            ts = tuning_selected_name(sd)
            res = M.run_methods(sd.policies, t.delta, names_sorted, ts, pv_cp, pv_hb, n_acc)
            # --- 3. equality checks against the published code
            checks = {}
            for nm in D.POLICIES:
                pd = sd.policies[nm]
                dep = res["fs"]["single"][nm][1][0]
                mine = float(pd.grid[dep]) if dep >= 0 else None
                checks[f"fs_single_{nm}"] = mine == orig["familywise_single_policy"][nm]["certified_threshold"]
                q = res["marginal"]["single"][nm][0][0]
                checks[f"marginal_{nm}"] = bool(q.any()) == bool(
                    orig["legacy_marginal"]["policies"][nm]["certificate"]["passes"])
            pi, gi = res["fs"]["sel"]
            sel = names_sorted[pi[0]] if pi[0] >= 0 else None
            checks["fs_portfolio_selection"] = sel == orig["familywise_portfolio"]["selected_policy"]
            if sel is not None:
                checks["fs_portfolio_threshold"] = float(sd.policies[sel].grid[gi[0]]) == \
                    orig["familywise_portfolio"]["selected_threshold"]
            checks["tuning_selected_policy"] = ts == orig["familywise_tuning_selected"]["selected_policy"]
            q_ts = res["fs_tuning_sel"]["sel"][1][0]
            checks["tuning_selected_qualified"] = (q_ts >= 0) == (
                orig["familywise_tuning_selected"]["certificate"]["certified_threshold"] is not None)
            # --- 4. per-method deployed outcome on the TEST role
            row = {"seed": seed, "tuning_selected": ts, "checks": checks, "methods": {},
                   "grid_sizes": {nm: int(len(sd.policies[nm].grid)) for nm in D.POLICIES},
                   "n_hold": int(len(sd.policies["action_pool"].hold["score"])),
                   "n_test": int(len(sd.policies["action_pool"].test["score"]))}
            for method in M.METHODS:
                r = res[method]
                single = {}
                for nm in D.POLICIES:
                    q, dep = r["single"][nm]
                    d = int(dep[0])
                    single[nm] = None if d < 0 else {
                        "threshold": float(sd.policies[nm].grid[d]),
                        "n_qualified_gates": int(q[0].sum()),
                        "test": evaluate(sd, nm, float(sd.policies[nm].grid[d]))}
                pi, gi = r["sel"]
                sel = names_sorted[pi[0]] if pi[0] >= 0 else None
                port = None
                if sel is not None:
                    thr = float(sd.policies[sel].grid[gi[0]])
                    port = {"policy": sel, "threshold": thr,
                            "n_qualified_pairs": int(sum(r["portfolio"][nm][0][0].sum() for nm in names_sorted)),
                            "test": evaluate(sd, sel, thr)}
                row["methods"][method] = {"single": single, "portfolio": port}
            # --- 5. CRC operating point (expectation-level; not tabulated as equivalent)
            crc = {}
            for nm in D.POLICIES:
                pd = sd.policies[nm]
                g = crc_gate(pd.hold, pd.grid, t.deletion, t.invention)
                crc[nm] = None if g < 0 else {"threshold": float(pd.grid[g]),
                                               "test": evaluate(sd, nm, float(pd.grid[g]))}
            row["crc"] = crc
            rows.append(row)
            print(domain, seed, "checks ok" if all(checks.values()) else
                  f"CHECK FAILURES {[k for k, v in checks.items() if not v]}")
        all_rows[domain] = rows
        # published-number reproduction summary
        repro[domain] = {
            "portfolio_qualified": sum(r["methods"]["fs"]["portfolio"] is not None for r in rows),
            "single_fs": [sum(r["methods"]["fs"]["single"][nm] is not None for r in rows) for nm in D.POLICIES],
            "single_marginal": [sum(r["methods"]["marginal"]["single"][nm] is not None for r in rows)
                                for nm in D.POLICIES],
            "single_fs_ascending": [sum(r["methods"]["fs_asc"]["single"][nm] is not None for r in rows)
                                    for nm in D.POLICIES],
            "portfolio_fs_ascending": sum(r["methods"]["fs_asc"]["portfolio"] is not None for r in rows),
            "tuning_selected_qualified": sum(r["methods"]["fs_tuning_sel"]["portfolio"] is not None for r in rows),
            "all_checks_pass": all(all(r["checks"].values()) for r in rows),
        }
    payload = {"targets": {"alpha_loss": t.deletion, "alpha_act": t.invention, "delta": t.delta,
                           "r0": D.FLOOR, "min_accept": t.min_accept,
                           "min_positive_accept": t.min_positive_accept},
               "policies": list(D.POLICIES), "reproduction": repro, "rows": all_rows}
    (OUT / "records_comparison.json").write_text(json.dumps(payload, indent=1, default=float),
                                                  encoding="utf-8")
    print(json.dumps(repro, indent=1))
    write_markdown(payload)


def _fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def write_markdown(payload: dict) -> None:
    lines = ["# WP3 record-based comparison (retrospective, descriptive)", "",
             "alpha = 0.15 (both endpoints), delta = 0.10, r0 = 0.25, f0 = 0.002, 5 training seeds.",
             "Failure-inclusive mean test acceptance: 0 for a repetition in which nothing qualifies.",
             "Test incident rates: mean over repetitions that deployed something "
             "(pooled counts in brackets: incidents/denominator).", ""]
    covers = {
        "naive": ("no", "no"), "marginal": ("no gate selection (one gate fixed on tuning)",
                                             "no; yes only for the Bonferroni 1/8 portfolio variant"),
        "ltt_bonf": ("yes", "yes (Bonferroni 1/8)"), "holm": ("yes", "yes (joint Holm over all pairs)"),
        "sgr": ("yes (tested gates)", "yes (Bonferroni 1/8)"), "fs": ("yes", "yes (Bonferroni 1/8)"),
        "fs_tuning_sel": ("yes", "yes (tuning-role selection)"), "fallback": ("yes", "yes (Bonferroni 1/8)"),
        "fs_hb": ("yes", "yes (Bonferroni 1/8)"), "fs_asc": ("yes", "yes (Bonferroni 1/8)"), "fs_iut": ("yes", "yes (Bonferroni 1/8)"),
        "holm_iut": ("yes", "yes (joint Holm)")}
    short = [D.SHORT[p] for p in payload["policies"]]
    for domain, rows in payload["rows"].items():
        lines += [f"## {domain}", "",
                  "| Method | Portfolio qualified | Portfolio failure-incl. mean test acceptance | "
                  "Portfolio test loss rate [pooled] | Portfolio test activation rate [pooled] | "
                  f"Single-policy qualified ({', '.join(short)}) | Single-policy failure-incl. mean test acceptance, "
                  "mean over 8 policies | Covers gate selection | Covers policy selection |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for method in M.METHODS:
            port = [r["methods"][method]["portfolio"] for r in rows]
            n_q = sum(p is not None for p in port)
            cov = np.mean([p["test"]["coverage"] if p else 0.0 for p in port])
            dl = [p["test"] for p in port if p]
            def pooled(key_e, key_n):
                e = sum((x[key_e] or 0) * x[key_n] for x in dl)
                n = sum(x[key_n] for x in dl)
                return f"[{int(round(e))}/{n}]" if n else ""
            lr = np.mean([x["deletion_risk"] for x in dl if x["deletion_risk"] is not None]) if dl else None
            ar = np.mean([x["invention_risk"] for x in dl]) if dl else None
            singles = [sum(r["methods"][method]["single"][nm] is not None for r in rows)
                       for nm in payload["policies"]]
            scov = np.mean([[r["methods"][method]["single"][nm]["test"]["coverage"]
                             if r["methods"][method]["single"][nm] else 0.0 for r in rows]
                            for nm in payload["policies"]])
            port_cell = f"{n_q}/5" if method not in ("marginal",) else f"{n_q}/5 (Bonferroni 1/8 variant)"
            lines.append(
                f"| {M.METHOD_LABELS[method]} | {port_cell} | {cov:.3f} | {_fmt(lr)} {pooled('deletion_risk', 'accepted_positive')} | "
                f"{_fmt(ar)} {pooled('invention_risk', 'accepted')} | {','.join(map(str, singles))} | {scov:.3f} | "
                f"{covers[method][0]} | {covers[method][1]} |")
        lines += [""]
        lines += ["CRC operating point (expectation-level joint-rate guarantee; NOT equivalent):", "",
                  "| Policy | reps with a gate | mean test acceptance | mean test loss rate | mean test activation rate |",
                  "|---|---|---|---|---|"]
        for nm in payload["policies"]:
            c = [r["crc"][nm] for r in rows]
            got = [x for x in c if x]
            lines.append(f"| {D.SHORT[nm]} | {len(got)}/5 | {np.mean([x['test']['coverage'] if x else 0 for x in c]):.3f} | "
                         f"{_fmt(np.mean([x['test']['deletion_risk'] for x in got if x['test']['deletion_risk'] is not None]) if got else None)} | "
                         f"{_fmt(np.mean([x['test']['invention_risk'] for x in got]) if got else None)} |")
        lines += [""]
    (OUT / "records_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
