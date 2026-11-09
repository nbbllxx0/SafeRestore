"""WP3 part 2: Monte Carlo validity study.

(a) Record-based: for each domain and training seed, the pooled hold-out +
    test records form a finite population whose gate-level conditional
    risks are KNOWN.  The tuning role stays fixed (it is F0: grid, ordering,
    tuned gate, tuning-selected policy).  Each replication draws a hold-out
    of the original size with replacement and runs every procedure.
(b) Synthetic stress: populations in which many gates sit just above or
    just below alpha (deterministic incident patterns, so every population
    risk is known exactly), 5 independent tuning draws each.

Realised familywise error = P(any qualified (policy, gate) has population
risk > alpha on either endpoint).  Also reported: P(deployed gate
violates), P(anything qualifies), mean population acceptance fraction of
the deployed gate (0 if none).

Run:  python -B tim_rr/wp3/montecarlo.py [--reps 4000]
Writes: tim_rr/wp3/montecarlo_results.json, tim_rr/wp3/montecarlo_results.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import beta

import wp3_data as D
import wp3_methods as M
from compare_records import tuning_selected_name

OUT = Path(__file__).resolve().parent


@dataclass
class Pol:
    grid: np.ndarray
    order: np.ndarray
    tuned_idx: int


def cp_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def tuned_index(grid: np.ndarray, thr: float) -> int:
    if not np.isfinite(thr):
        return -1
    i = int(np.searchsorted(grid, thr))
    assert i < len(grid) and grid[i] == thr
    return i


def population_truth(pop: dict, grid: np.ndarray, alpha_l: float, alpha_a: float):
    c = D.gate_counts(pop, grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.where(c["n_del"] > 0, c["e_del"] / np.maximum(c["n_del"], 1), 0.0)
        ra = np.where(c["n_act"] > 0, c["e_act"] / np.maximum(c["n_act"], 1), 0.0)
    return {"risk_loss": rl, "risk_act": ra, "violates": (rl > alpha_l) | (ra > alpha_a),
            "coverage": c["n_act"] / len(pop["score"])}


def mc_engine(pols: dict, pop: dict, n_hold: int, reps: int, rng: np.random.Generator,
              ts_name: str, targets=D.TARGETS, chunk: int = 1000) -> dict:
    names = tuple(sorted(pols))
    al, aa, delta = targets.deletion, targets.invention, targets.delta
    N = len(next(iter(pop.values()))["score"])
    truth = {nm: population_truth(pop[nm], pols[nm].grid, al, aa) for nm in names}
    bins = {nm: np.searchsorted(pols[nm].grid, pop[nm]["score"], side="left") for nm in names}
    acc = {m: {"port_fwer": 0, "port_dep_viol": 0, "port_qual": 0, "port_cov": 0.0,
               "single_fwer": {nm: 0 for nm in names}, "single_qual": {nm: 0 for nm in names},
               "single_cov": {nm: 0.0 for nm in names}} for m in M.METHODS}
    crc = {nm: {"viol": 0, "cov": 0.0, "none": 0} for nm in names}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        idx = rng.integers(0, N, size=(r, n_hold))
        pv_cp, pv_hb, n_acc, counts = {}, {}, {}, {}
        for nm in names:
            m = len(pols[nm].grid)
            b = bins[nm][idx] + (m + 1) * np.arange(r)[:, None]
            p = pop[nm]
            def h(w):
                x = np.bincount(b.ravel(), weights=None if w is None else w[idx].ravel().astype(float),
                                minlength=r * (m + 1)).reshape(r, m + 1)
                return np.cumsum(x[:, :m], axis=1)
            c = {"n_act": h(None), "n_del": h(p["positive"]),
                 "e_del": h(p["positive"] & p["deletion"]), "e_act": h(p["invention"])}
            counts[nm] = c
            pv_cp[nm] = M.gate_pvalues(c, al, aa, "cp")
            pv_hb[nm] = M.gate_pvalues(c, al, aa, "hb")
            n_acc[nm] = c["n_act"].astype(np.int64)
        res = M.run_methods(pols, delta, names, ts_name, pv_cp, pv_hb, n_acc)
        rows = np.arange(r)
        for method in M.METHODS:
            a = acc[method]
            anyv = np.zeros(r, dtype=bool)
            for nm in names:
                q, _ = res[method]["portfolio"][nm]
                anyv |= (q & truth[nm]["violates"][None, :]).any(1)
            a["port_fwer"] += int(anyv.sum())
            pi, gi = res[method]["sel"]
            dep_viol = np.zeros(r, dtype=bool)
            cov = np.zeros(r)
            for i, nm in enumerate(names):
                sel = pi == i
                if sel.any():
                    dep_viol[sel] = truth[nm]["violates"][gi[sel]]
                    cov[sel] = truth[nm]["coverage"][gi[sel]]
            a["port_dep_viol"] += int(dep_viol.sum())
            a["port_qual"] += int((pi >= 0).sum())
            a["port_cov"] += float(cov.sum())
            for nm in names:
                q, dep = res[method]["single"][nm]
                a["single_fwer"][nm] += int((q & truth[nm]["violates"][None, :]).any(1).sum())
                a["single_qual"][nm] += int((dep >= 0).sum())
                a["single_cov"][nm] += float(np.where(dep >= 0, truth[nm]["coverage"][np.maximum(dep, 0)], 0).sum())
        # CRC operating point (B = 1), expectation-level joint-rate rule
        for nm in names:
            c = counts[nm]
            n_pos = pop[nm]["positive"][idx].sum(1)
            ok_l = (c["e_del"] + 1) / (n_pos[:, None] + 1) <= al
            ok_a = (c["e_act"] + 1) / (n_hold + 1) <= aa
            ok = ok_l & ok_a
            # monotone losses: admissible set is a prefix; last index of the prefix
            first_bad = np.where(ok.all(1), ok.shape[1], np.argmin(ok, axis=1))
            g = first_bad - 1
            has = g >= 0
            crc[nm]["none"] += int((~has).sum())
            crc[nm]["viol"] += int(truth[nm]["violates"][np.maximum(g, 0)][has].sum())
            crc[nm]["cov"] += float(truth[nm]["coverage"][np.maximum(g, 0)][has].sum())
        done += r
    out = {"reps": reps, "n_hold": n_hold, "N_pop": N,
           "population": {nm: {"grid_size": int(len(pols[nm].grid)),
                               "violating_gates": int(truth[nm]["violates"].sum()),
                               "fs_first_gate_violates": bool(truth[nm]["violates"][pols[nm].order[0]])
                               if len(pols[nm].order) else None,
                               "max_safe_coverage": float(truth[nm]["coverage"][~truth[nm]["violates"]].max())
                               if (~truth[nm]["violates"]).any() else 0.0,
                               "nested_safe_coverage": nested_safe(truth[nm])}
                          for nm in names},
           "methods": {}, "crc": {}}
    for method, a in acc.items():
        out["methods"][method] = {
            "port_fwer": a["port_fwer"] / reps, "port_fwer_k": a["port_fwer"],
            "port_dep_viol": a["port_dep_viol"] / reps,
            "port_qual": a["port_qual"] / reps, "port_cov": a["port_cov"] / reps,
            "single_fwer": {nm: v / reps for nm, v in a["single_fwer"].items()},
            "single_fwer_k": a["single_fwer"],
            "single_qual": {nm: v / reps for nm, v in a["single_qual"].items()},
            "single_cov": {nm: v / reps for nm, v in a["single_cov"].items()},
        }
    for nm, c in crc.items():
        out["crc"][nm] = {"cond_viol": c["viol"] / reps, "no_gate": c["none"] / reps,
                          "cov": c["cov"] / reps}
    return out


def nested_safe(t: dict) -> float:
    """Largest population coverage among gates below the first violating gate
    (ascending), i.e. the 'oracle' gate if one deploys a nested prefix."""
    v = t["violates"]
    first = int(np.argmax(v)) if v.any() else len(v)
    return float(t["coverage"][first - 1]) if first > 0 else 0.0


# ------------------------------------------------------------ record-based
def record_study(reps: int, seed0: int = 20261025) -> dict:
    out = {}
    for di, domain in enumerate(D.DOMAINS):
        out[domain] = {}
        for seed in D.SEEDS:
            t0 = time.time()
            sd = D.load_seed(domain, seed)
            pols, pop = {}, {}
            for nm, pd in sd.policies.items():
                pols[nm] = Pol(grid=pd.grid, order=pd.order, tuned_idx=tuned_index(pd.grid, pd.tuned_threshold))
                pop[nm] = {k: np.concatenate([pd.hold[k], pd.test[k]]) for k in pd.hold}
            n_hold = len(sd.policies["action_pool"].hold["score"])
            rng = np.random.default_rng([seed0, di, seed])
            r = mc_engine(pols, pop, n_hold, reps, rng, tuning_selected_name(sd))
            out[domain][str(seed)] = r
            print(f"{domain} {seed}: {time.time() - t0:.1f}s  "
                  + "  ".join(f"{m}={r['methods'][m]['port_fwer']:.4f}" for m in ("naive", "ltt_bonf", "sgr", "fs", "fallback")))
    return out


# --------------------------------------------------------------- synthetic
def _pattern(n: int, rate) -> np.ndarray:
    """Deterministic incidents for units ranked 1..n: cumulative count
    ceil(k * rate(k)) so that every prefix k has rate >= rate(k) - 0 and
    < rate(k) + 1/k."""
    k = np.arange(1, n + 1)
    target = np.ceil(k * rate(k / n) - 1e-12)
    target = np.maximum.accumulate(np.clip(target, 0, k))
    inc = np.diff(np.concatenate([[0], target]))
    assert np.all((inc == 0) | (inc == 1)), "rate curve too steep"
    return inc.astype(bool)


def make_population(scenario: str, rng: np.random.Generator, N: int, n_pol: int = 8,
                    prevalence: float = 0.3) -> dict:
    base = rng.uniform(size=N)
    positive = rng.uniform(size=N) < prevalence
    pop = {}
    for j in range(n_pol):
        score = base + rng.normal(0, 0.05, size=N)
        order = np.argsort(score, kind="stable")
        deletion = np.zeros(N, dtype=bool)
        invention = np.zeros(N, dtype=bool)
        pos_order = order[positive[order]]
        shift = 0.02 * (j - (n_pol - 1) / 2) / n_pol  # small per-policy offset
        if scenario == "boundary_null":
            # every gate: loss risk just above alpha (>= 0.153), activation 0.03
            deletion[pos_order] = _pattern(len(pos_order), lambda u: 0.153 + 0 * u)
            invention[order] = _pattern(N, lambda u: 0.03 + 0 * u)
        elif scenario == "crossing_loss":
            # loss risk rises slowly 0.10 -> 0.20 across the score range
            deletion[pos_order] = _pattern(len(pos_order), lambda u: 0.10 + shift + 0.10 * u)
            invention[order] = _pattern(N, lambda u: 0.03 + 0 * u)
        elif scenario == "crossing_both":
            # loss crosses alpha at ~60 % of positives, activation at ~40 % of units
            deletion[pos_order] = _pattern(len(pos_order), lambda u: 0.12 + shift + 0.05 * u)
            invention[order] = _pattern(N, lambda u: 0.13 + shift + 0.05 * u)
        else:
            raise ValueError(scenario)
        pop[f"p{j}"] = {"score": score, "positive": positive.copy(),
                        "deletion": deletion & positive, "invention": invention}
    return pop


def synthetic_study(reps: int, N: int = 20000, n_tune: int = 200, n_hold: int = 400,
                    draws: int = 5, seed0: int = 7) -> dict:
    t = D.TARGETS
    out = {}
    for si, scenario in enumerate(("boundary_null", "crossing_loss", "crossing_both")):
        rng_pop = np.random.default_rng([seed0, si])
        pop = make_population(scenario, rng_pop, N)
        out[scenario] = {}
        for d in range(draws):
            t0 = time.time()
            rng = np.random.default_rng([seed0, si, d])
            tune_idx = rng.integers(0, N, size=n_tune)  # tuning role: i.i.d. draw, then fixed (F0)
            pols, promises = {}, {}
            for nm, p in pop.items():
                tune = {k: v[tune_idx] for k, v in p.items()}
                grid = np.sort(D.fc.candidate_sequence(tune["score"], tune["positive"], t))
                seq = D.fc.candidate_sequence(tune["score"], tune["positive"], t,
                                              tuning_deletion=tune["deletion"],
                                              tuning_invention=tune["invention"],
                                              certification_scale=n_hold / n_tune)
                order = np.searchsorted(grid, seq)
                tuned = D.choose_threshold(tune["score"], tune["positive"], tune["deletion"],
                                           tune["invention"], t)
                pols[nm] = Pol(grid=grid, order=order, tuned_idx=tuned_index(grid, tuned))
                promises[nm] = D.tuning_promise(tune, t)
            ts = max(sorted(pols), key=lambda n: (promises[n], n))
            r = mc_engine(pols, pop, n_hold, reps, rng, ts, chunk=500)
            out[scenario][str(d)] = r
            print(f"synthetic {scenario} draw {d}: {time.time() - t0:.1f}s  "
                  + "  ".join(f"{m}={r['methods'][m]['port_fwer']:.4f}" for m in ("naive", "ltt_bonf", "sgr", "fs", "fallback")))
    return out


# ------------------------------------------------------------------ report
def write_markdown(payload: dict) -> None:
    reps = payload["reps"]
    lines = ["# WP3 Monte Carlo validity study", "",
             f"{reps} replications per (domain, seed) or (scenario, tuning draw); delta = 0.10.",
             "FWER = P(any qualified (policy, gate) violates a population target). "
             "95% Clopper-Pearson intervals on the Monte Carlo proportion.", ""]

    def block(title, units, unit_label):
        lines.extend([f"## {title}", "",
                      f"| Method | Portfolio FWER, max over {unit_label} [95% CI] | Portfolio FWER, mean | "
                      "P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | "
                      f"Single-policy FWER, max over policies x {unit_label} [95% CI] |",
                      "|---|---|---|---|---|---|---|"])
        for method in M.METHODS:
            rs = [u["methods"][method] for u in units.values()]
            ks = [x["port_fwer_k"] for x in rs]
            kmax = max(ks)
            lo, hi = cp_ci(kmax, reps)
            sk = max(max(x["single_fwer_k"].values()) for x in rs)
            slo, shi = cp_ci(sk, reps)
            flag = " **>delta**" if lo > 0.10 else ""
            sflag = " **>delta**" if slo > 0.10 else ""
            lines.append(
                f"| {M.METHOD_LABELS[method]} | {kmax / reps:.4f} [{lo:.4f}, {hi:.4f}]{flag} | "
                f"{np.mean([x['port_fwer'] for x in rs]):.4f} | {np.mean([x['port_dep_viol'] for x in rs]):.4f} | "
                f"{np.mean([x['port_qual'] for x in rs]):.3f} | {np.mean([x['port_cov'] for x in rs]):.3f} | "
                f"{sk / reps:.4f} [{slo:.4f}, {shi:.4f}]{sflag} |")
        crc_v = max(max(c["cond_viol"] for c in u["crc"].values()) for u in units.values())
        crc_m = np.mean([np.mean([c["cond_viol"] for c in u["crc"].values()]) for u in units.values()])
        lines.extend(["", f"CRC operating point: P(conditional risk of the CRC gate > alpha), "
                      f"mean over policies x {unit_label} = {crc_m:.3f}, max = {crc_v:.3f} "
                      "(CRC does not claim to control this quantity).", ""])
        pops = [u["population"] for u in units.values()]
        vg = [sum(p[nm]["violating_gates"] for nm in p) for p in pops]
        gs = [sum(p[nm]["grid_size"] for nm in p) for p in pops]
        ff = [sum(bool(p[nm]["fs_first_gate_violates"]) for nm in p) for p in pops]
        lines.extend([f"Population context per {unit_label[:-1]}: violating gates / grid gates (all 8 policies) = "
                      + ", ".join(f"{a}/{b}" for a, b in zip(vg, gs))
                      + f"; policies whose first FS gate violates = {ff}.", ""])

    for domain, units in payload["records"].items():
        block(f"Record-based resampling: {domain}", units, "seeds")
    for scen, units in payload["synthetic"].items():
        block(f"Synthetic stress: {scen}", units, "draws")
    (OUT / "montecarlo_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    payload = {"reps": args.reps, "records": record_study(args.reps)}
    payload["synthetic"] = {} if args.skip_synthetic else synthetic_study(args.reps)
    (OUT / "montecarlo_results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    write_markdown(payload)


if __name__ == "__main__":
    main()
