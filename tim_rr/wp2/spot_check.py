"""WP2 spot check: independent minimal re-implementation for two grid cells.

Does NOT import the definitive analysis code, saferestore, or wp2_common.
Re-implemented from the written procedure: CSV parsing, relabelling,
logistic score fits, minimax action scores, the hash tuning mask, the
Laplace margin ordering, fixed-sequence testing, the Bonferroni portfolio,
and the marginal tuned-gate certificate.  The one-sided Clopper-Pearson bound
is computed by bisection on the binomial CDF (not scipy.stats.beta), and pass
decisions use the dual form P(X <= e | n, alpha) <= level.

Usage: python -B spot_check.py DOMAIN R0 F0 ALPHA DELTA
Compares against grid_seed_<domain>.csv and prints/saves spot_check_<...>.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FOLDERS = {
    "carinthia_s": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
    "kolektor_sdd2": REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed{seed}",
}
FEATS = {
    "carinthia_s": ["entropy", "forward_consistency", "detector_disagreement", "evidence_shift",
                    "restoration_residual", "high_probability_area"],
    "kolektor_sdd2": ["entropy", "forward_consistency", "detector_disagreement", "evidence_shift",
                      "restoration_residual", "high_probability_area", "peak_probability",
                      "peak_disagreement", "secondary_peak", "active_components"],
}
BALANCED = {"carinthia_s": True, "kolektor_sdd2": False}
METHODS = ["raw_nearest", "bilinear", "bicubic", "smooth_bicubic", "unsharp_bicubic", "learned"]
POOLS = {m: [m] for m in METHODS}
POOLS["restoration_pool"] = METHODS[1:]
POOLS["action_pool"] = METHODS
MIN_ACC = MIN_POS = 15
SEEDS = (201, 203, 207, 211, 223)


def cp_upper(e, n, level):
    if n <= 0 or e >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom.cdf(e, n, mid) > level:
            lo = mid
        else:
            hi = mid
    return hi


def passes(e, n, level, alpha):
    if n <= 0 or e >= n:
        return False
    return binom.cdf(e, n, alpha) <= level


def load(path, feats, r0, f0):
    by = {m: [] for m in METHODS}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pos = int(r["has_defect"]) == 1
            by[r["method"]].append({
                "id": r["sample_id"], "pos": pos,
                "dele": int(pos and float(r["pixel_recall"]) < r0),
                "inv": int(float(r["clean_fpr"]) > f0),
                "x": [float(r[f]) for f in feats],
            })
    return by


def fit(train, seed, balanced):
    models = {}
    for m in METHODS:
        rows = train[m]
        spec = {}
        for tgt in ("dele", "inv"):
            pop = [r for r in rows if r["pos"]] if tgt == "dele" else rows
            X = np.array([r["x"] for r in pop])
            y = np.array([r[tgt] for r in pop])
            if len(np.unique(y)) < 2:
                spec[tgt] = ("const", float(y[0]))
                continue
            sc = StandardScaler().fit(X)
            lr = LogisticRegression(class_weight="balanced" if balanced else None, max_iter=2000,
                                    random_state=seed, solver="liblinear").fit(sc.transform(X), y)
            spec[tgt] = ("lr", sc, lr)
        models[m] = spec
    return models


def predict(spec, X):
    if spec[0] == "const":
        return np.full(len(X), spec[1])
    _, sc, lr = spec
    z = np.clip(lr.decision_function(sc.transform(X)), -40, 40)
    return 1 / (1 + np.exp(-z))


def policy_arrays(data, models, alpha):
    ids = [r["id"] for r in data["raw_nearest"]]
    cols = {}
    for m in METHODS:
        assert [r["id"] for r in data[m]] == ids
        X = np.array([r["x"] for r in data[m]])
        cols[m] = np.maximum(predict(models[m]["dele"], X) / alpha, predict(models[m]["inv"], X) / alpha)
    out = {}
    for name, ms in POOLS.items():
        mat = np.stack([cols[m] for m in ms], axis=1)
        k = mat.argmin(axis=1)
        sel = [data[ms[j]][i] for i, j in enumerate(k)]
        out[name] = {"score": mat[np.arange(len(k)), k],
                     "pos": np.array([r["pos"] for r in sel]),
                     "dele": np.array([r["dele"] for r in sel]),
                     "inv": np.array([r["inv"] for r in sel]),
                     "ids": ids}
    return out


def tune_mask(ids):
    return np.array([int.from_bytes(hashlib.sha256(f"v4tunecert:{s}".encode()).digest()[:8], "big") % 10 < 3
                     for s in ids])


def sequence(ts, tp, td, ti, alpha, delta, scale):
    order = np.argsort(ts, kind="stable")
    cum_n = np.arange(1, len(ts) + 1)
    cum_p = np.cumsum(tp[order])
    ok = (cum_n >= MIN_ACC) & (cum_p >= MIN_POS)
    if not ok.any():
        return []
    thr = np.unique(ts[order][int(np.argmax(ok)):])
    keyed = []
    for idx, t in enumerate(thr):
        a = ts <= t
        ap = a & tp
        nd = max(int(round(ap.sum() * scale)), 1)
        ni = max(int(round(a.sum() * scale)), 1)
        rd = (td[ap].sum() + 1) / (ap.sum() + 2) if ap.any() else 1.0
        ri = (ti[a].sum() + 1) / (a.sum() + 2)
        ed, ei = int(round(rd * nd)), int(round(ri * ni))
        margin = min(alpha - cp_upper(min(ed, nd), nd, delta / 2), alpha - cp_upper(min(ei, ni), ni, delta / 2))
        keyed.append((-margin, -a.sum() / len(ts), idx, t))
    return [k[3] for k in sorted(keyed)]


def fixed_sequence(arr, mask, alpha, delta, level_joint):
    ts, tp = arr["score"][mask], arr["pos"][mask]
    td, ti = arr["dele"][mask].astype(bool), arr["inv"][mask].astype(bool)
    cs, cp = arr["score"][~mask], arr["pos"][~mask]
    cd, ci = arr["dele"][~mask].astype(bool), arr["inv"][~mask].astype(bool)
    best, best_cov = None, -1.0
    for t in sequence(ts, tp, td, ti, alpha, delta, len(cs) / len(ts)):
        a = cs <= t
        ap = a & cp
        if passes(int(cd[ap].sum()), int(ap.sum()), level_joint / 2, alpha) and \
           passes(int(ci[a].sum()), int(a.sum()), level_joint / 2, alpha):
            cov = a.sum() / len(cs)
            if cov > best_cov:
                best, best_cov = t, cov
        else:
            break
    return best


def marginal(arr, mask, alpha, delta):
    ts, tp = arr["score"][mask], arr["pos"][mask]
    td, ti = arr["dele"][mask], arr["inv"][mask]
    order = np.argsort(ts, kind="stable")
    thr = -np.inf
    for c in range(MIN_ACC, len(order) + 1):
        loc = order[:c]
        lp = tp[loc]
        if lp.sum() < MIN_POS:
            continue
        if td[loc][lp].mean() <= alpha * 0.5 and ti[loc].mean() <= alpha * 0.5:
            thr = ts[order[c - 1]]
    cs, cp = arr["score"][~mask], arr["pos"][~mask]
    a = cs <= thr
    ap = a & cp
    ok = passes(int(arr["dele"][~mask][ap].sum()), int(ap.sum()), delta / 2, alpha) and \
        passes(int(arr["inv"][~mask][a].sum()), int(a.sum()), delta / 2, alpha)
    return ok, thr


def main():
    domain, r0, f0, alpha, delta = sys.argv[1], *map(float, sys.argv[2:6])
    mine = []
    for seed in SEEDS:
        folder = Path(str(FOLDERS[domain]).format(seed=seed))
        feats = FEATS[domain]
        models = fit(load(folder / "detector_val_records.csv", feats, r0, f0), seed, BALANCED[domain])
        cal = policy_arrays(load(folder / "calibration_records.csv", feats, r0, f0), models, alpha)
        tst = policy_arrays(load(folder / "test_records.csv", feats, r0, f0), models, alpha)
        mask = tune_mask(cal["action_pool"]["ids"])
        rec = {"seed": seed}
        best_name, best_cov = None, -1.0
        for name in sorted(POOLS):
            t_single = fixed_sequence(cal[name], mask, alpha, delta, delta)
            rec[f"fam_pass__{name}"] = t_single is not None
            rec[f"fam_cov_fi__{name}"] = float((tst[name]["score"] <= t_single).mean()) if t_single is not None else 0.0
            ok, thr = marginal(cal[name], mask, alpha, delta)
            rec[f"marg_pass__{name}"] = bool(ok)
            t_port = fixed_sequence(cal[name], mask, alpha, delta, delta / len(POOLS))
            if t_port is not None:
                cov = float((cal[name]["score"][~mask] <= t_port).mean())
                if cov > best_cov:
                    best_name, best_cov, best_t = name, cov, t_port
        rec["portfolio_selected"] = best_name
        rec["portfolio_test_coverage_fi"] = float((tst[best_name]["score"] <= best_t).mean()) if best_name else 0.0
        mine.append(rec)

    grid = [r for r in csv.DictReader(open(HERE / f"grid_seed_{domain}.csv", encoding="utf-8"))
            if float(r["r0"]) == r0 and float(r["f0"]) == f0 and float(r["alpha"]) == alpha and float(r["delta"]) == delta]
    assert len(grid) == 5
    diffs = []
    for a, g in zip(mine, sorted(grid, key=lambda r: int(r["seed"]))):
        for k, v in a.items():
            if k == "seed":
                continue
            gv = g[k]
            if isinstance(v, bool):
                same = (gv == "True") == v
            elif isinstance(v, float):
                same = abs(float(gv) - v) < 1e-9
            else:
                same = (gv if gv != "" else None) == v
            if not same:
                diffs.append({"seed": a["seed"], "field": k, "independent": v, "grid": gv})
    report = {"cell": {"domain": domain, "r0": r0, "f0": f0, "alpha": alpha, "delta": delta},
              "fields_compared": len(mine) * (len(mine[0]) - 1), "n_differences": len(diffs),
              "differences": diffs, "independent": mine}
    name = f"spot_check_{domain}_r{r0}_f{f0}_a{alpha}_d{delta}.json"
    (HERE / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(name, "fields", report["fields_compared"], "differences", len(diffs))
    for d in diffs[:20]:
        print(d)


if __name__ == "__main__":
    main()
