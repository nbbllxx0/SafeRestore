"""WP3: matched risk-control procedures on a common candidate gate grid.

Every procedure works on exact one-sided binomial p-values computed from
hold-out counts, vectorised over R replications (R = 1 for the frozen
records; R = number of Monte Carlo draws otherwise).

For gate j with e incidents among n units and target alpha, the endpoint
p-value for H0: risk > alpha is p = P(Bin(n, alpha) <= e) (p = 1 when
n = 0).  The Clopper-Pearson bound U(e, n; g) <= alpha  iff  p <= g, so a
CP test at level g/2 per endpoint is the gate p-value
    p_gate = min(1, 2 * max(p_loss, p_act))  tested at level g
(Bonferroni over the two endpoints, as in the published method).
The intersection-union alternative p_gate = max(p_loss, p_act) is also
valid for the union null "at least one endpoint violated" and is reported
as a sensitivity variant ("IUT").

Shapes: p-value matrices are (R, m) with columns = grid gates in ascending
threshold order (= ascending acceptance).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

E_CONST = math.e


# ----------------------------------------------------------------- p-values
def binom_pvalue(e: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = binom.cdf(e, n, alpha)
    return np.where(n > 0, p, 1.0)


def hb_pvalue(e: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """Hoeffding-Bentkus p-value, LTT (arXiv 2110.01052) Proposition 1:
    min(exp(-n h1(min(R,alpha), alpha)), e * P(Bin(n, alpha) <= ceil(n R)))."""
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n > 0, e / np.maximum(n, 1), 1.0)
        a = np.minimum(r, alpha)
        # h1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b)), with 0 log 0 = 0
        t1 = np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0) / alpha), 0.0)
        t2 = (1 - a) * np.log((1 - a) / (1 - alpha))
        hoeff = np.exp(-n * (t1 + t2))
    bent = E_CONST * binom.cdf(np.ceil(n * r - 1e-9), n, alpha)
    p = np.minimum(np.minimum(hoeff, bent), 1.0)
    return np.where(n > 0, p, 1.0)


def gate_pvalues(c: dict, alpha_loss: float, alpha_act: float, kind: str = "cp") -> dict:
    """Return dict of gate p-values (R, m) for the combination rules used."""
    f = binom_pvalue if kind == "cp" else hb_pvalue
    pl = f(c["e_del"], c["n_del"], alpha_loss)
    pa = f(c["e_act"], c["n_act"], alpha_act)
    mx = np.maximum(pl, pa)
    return {"bonf": np.minimum(1.0, 2.0 * mx), "iut": mx}


# --------------------------------------------------------------- selection
def deploy_counts(qualified: np.ndarray, n_acc: np.ndarray, rank: np.ndarray | None = None) -> np.ndarray:
    """deploy() on integer accepted counts (exact ties)."""
    R, m = qualified.shape
    if m == 0:
        return np.full(R, -1, dtype=np.int64)
    if rank is None:
        rank = np.arange(m)
    big = (m + 1)
    key = np.where(qualified, n_acc.astype(np.int64) * big + (m - rank[None, :]), -1)
    idx = np.argmax(key, axis=1)
    return np.where(qualified.any(axis=1), idx, -1)


# --------------------------------------------------------------- procedures
def proc_naive(pg: np.ndarray, gamma: float) -> np.ndarray:
    """Every gate tested at the full level (no multiplicity correction)."""
    return pg <= gamma


def proc_single(pg: np.ndarray, gate_idx: int, gamma: float) -> np.ndarray:
    """Legacy marginal: only the tuning-chosen gate is tested."""
    q = np.zeros_like(pg, dtype=bool)
    if gate_idx >= 0:
        q[:, gate_idx] = pg[:, gate_idx] <= gamma
    return q


def proc_bonferroni(pg: np.ndarray, gamma: float) -> np.ndarray:
    m = pg.shape[1]
    return pg <= gamma / max(m, 1)


def proc_holm(pg: np.ndarray, gamma: float) -> np.ndarray:
    """Holm step-down over all columns of pg (FWER gamma)."""
    R, M = pg.shape
    if M == 0:
        return np.zeros_like(pg, dtype=bool)
    order = np.argsort(pg, axis=1, kind="stable")
    sp = np.take_along_axis(pg, order, axis=1)
    thr = gamma / (M - np.arange(M))
    ok = sp <= thr[None, :]
    k = np.where(ok.all(axis=1), M, np.argmin(ok, axis=1))  # number rejected
    rej_sorted = np.arange(M)[None, :] < k[:, None]
    q = np.zeros_like(pg, dtype=bool)
    np.put_along_axis(q, order, rej_sorted, axis=1)
    return q


def proc_fixed_sequence(pg: np.ndarray, order: np.ndarray, gamma: float) -> np.ndarray:
    R = pg.shape[0]
    q = np.zeros_like(pg, dtype=bool)
    alive = np.ones(R, dtype=bool)
    for j in order:
        ok = alive & (pg[:, j] <= gamma)
        q[:, j] = ok
        alive = ok
        if not alive.any():
            break
    return q


def proc_fallback(pg: np.ndarray, order: np.ndarray, gamma: float, k: int = 5) -> np.ndarray:
    """Wiens (2003) fallback / Bretz et al. (2009) graph: weights 1/k on the
    first k ordered gates, a forward chain passing the full level of every
    rejected gate to the next gate in the order; gates after the k-th carry
    weight 0 and are reached only through the chain (fixed sequence)."""
    R = pg.shape[0]
    q = np.zeros_like(pg, dtype=bool)
    kk = min(k, len(order))
    carry = np.zeros(R)
    for t, j in enumerate(order):
        w = 1.0 / kk if t < kk else 0.0
        level = w + carry
        rej = (level > 0) & (pg[:, j] <= level * gamma)
        q[:, j] = rej
        carry = np.where(rej, level, 0.0)
        if t >= kk and not (carry > 0).any():
            break
    return q


def sgr_k(m: int) -> int:
    return max(1, int(math.ceil(math.log2(m)))) if m > 0 else 0


def proc_sgr(pg: np.ndarray, gamma: float) -> np.ndarray:
    """Geifman & El-Yaniv (2017) Algorithm 1 binary search, adapted.

    Gates are indexed z = 1..m in DEcreasing acceptance (z = 1 largest gate,
    as in SGR's ordering by confidence).  k = ceil(log2 m) iterations; each
    tested gate is tested at gamma / k (the two endpoints are split inside
    pg).  z = ceil((zmin+zmax)/2); pass -> zmax = z, fail -> zmin = z.
    Every tested gate that passes is qualified (Theorem 3.2 holds
    simultaneously over the k tested gates)."""
    R, m = pg.shape
    q = np.zeros_like(pg, dtype=bool)
    if m == 0:
        return q
    k = sgr_k(m)
    level = gamma / k
    zmin = np.ones(R, dtype=np.int64)
    zmax = np.full(R, m, dtype=np.int64)
    rows = np.arange(R)
    for _ in range(k):
        z = (zmin + zmax + 1) // 2
        j = m - z  # grid index (ascending acceptance)
        ok = pg[rows, j] <= level
        q[rows[ok], j[ok]] = True
        zmax = np.where(ok, z, zmax)
        zmin = np.where(ok, zmin, z)
    return q


# ----------------------------------------------------------------- drivers
METHOD_LABELS = {
    "naive": "Naive select-best (no correction; invalid)",
    "marginal": "Marginal single tuned gate (legacy)",
    "ltt_bonf": "LTT-Bonferroni over gate grid",
    "holm": "LTT-Holm step-down",
    "sgr": "SGR binary search",
    "fs": "Fixed sequence, margin order (ours)",
    "fs_tuning_sel": "Fixed sequence, tuning-selected policy (ours)",
    "fallback": "Fallback/graphical, k=5 equal weights",
    "fs_hb": "Fixed sequence with Hoeffding-Bentkus",
    "fs_asc": "Fixed sequence, ascending order (first implementation, WP1)",
    "fs_iut": "Fixed sequence, IUT endpoint combination (variant)",
    "holm_iut": "LTT-Holm, IUT endpoint combination (variant)",
}
METHODS = tuple(METHOD_LABELS)


def run_methods(pols: dict, delta: float, names_sorted: tuple, tuning_selected: str,
                pv_cp: dict, pv_hb: dict, n_acc: dict) -> dict:
    """Run every procedure at single-policy and portfolio level.

    pols[name]: object with .order (FS order, grid indices) and
    .tuned_idx (grid index of the legacy tuned gate or -1).
    pv_cp/pv_hb[name]: gate_pvalues() dicts; n_acc[name]: (R, m) hold-out
    accepted counts (deployment key).  Returns
    out[method] = {"single": {name: (q, dep)}, "portfolio": {name: (q, dep)},
                   "sel": (policy_index_in_names_sorted, gate_idx)}.
    """
    P = len(names_sorted)
    out = {}

    def per_policy(fn, budget, rank_fs=False):
        res = {}
        for name in names_sorted:
            q = fn(name, budget)
            rank = None
            if rank_fs:
                rank = np.empty(len(pols[name].order), dtype=np.int64)
                rank[pols[name].order] = np.arange(len(pols[name].order))
            res[name] = (q, deploy_counts(q, n_acc[name], rank))
        return res

    specs = {
        "naive": (lambda nm, g: proc_naive(pv_cp[nm]["bonf"], g), False, False),
        "marginal": (lambda nm, g: proc_single(pv_cp[nm]["bonf"], pols[nm].tuned_idx, g), False, False),
        "ltt_bonf": (lambda nm, g: proc_bonferroni(pv_cp[nm]["bonf"], g), False, False),
        "sgr": (lambda nm, g: proc_sgr(pv_cp[nm]["bonf"], g), False, False),
        "fs": (lambda nm, g: proc_fixed_sequence(pv_cp[nm]["bonf"], pols[nm].order, g), True, False),
        "fallback": (lambda nm, g: proc_fallback(pv_cp[nm]["bonf"], pols[nm].order, g), True, False),
        "fs_hb": (lambda nm, g: proc_fixed_sequence(pv_hb[nm]["bonf"], pols[nm].order, g), True, False),
        "fs_iut": (lambda nm, g: proc_fixed_sequence(pv_cp[nm]["iut"], pols[nm].order, g), True, False),
        "fs_asc": (lambda nm, g: proc_fixed_sequence(pv_cp[nm]["bonf"], np.arange(len(pols[nm].order)), g),
                   False, False),
        "holm": (lambda nm, g: proc_holm(pv_cp[nm]["bonf"], g), False, True),
        "holm_iut": (lambda nm, g: proc_holm(pv_cp[nm]["iut"], g), False, True),
    }
    for method, (fn, rank_fs, joint_holm) in specs.items():
        single = per_policy(fn, delta, rank_fs)
        if method == "naive":
            portfolio = single  # no correction at either level
        elif joint_holm:
            key = "bonf" if method == "holm" else "iut"
            mats = [pv_cp[nm][key] for nm in names_sorted]
            sizes = [m.shape[1] for m in mats]
            qall = proc_holm(np.concatenate(mats, axis=1), delta)
            portfolio, start = {}, 0
            for nm, sz in zip(names_sorted, sizes):
                q = qall[:, start:start + sz]
                start += sz
                portfolio[nm] = (q, deploy_counts(q, n_acc[nm]))
        else:
            portfolio = per_policy(fn, delta / P, rank_fs)
        out[method] = {"single": single, "portfolio": portfolio,
                       "sel": select_policy(portfolio, names_sorted, n_acc)}
    # tuning-selected reading: policy fixed on tuning, full delta (Prop. 1)
    q, dep = out["fs"]["single"][tuning_selected]
    ts_idx = names_sorted.index(tuning_selected)
    R = q.shape[0]
    port = {nm: (np.zeros_like(out["fs"]["single"][nm][0]), np.full(R, -1, dtype=np.int64))
            for nm in names_sorted}
    port[tuning_selected] = (q, dep)
    out["fs_tuning_sel"] = {"single": out["fs"]["single"], "portfolio": port,
                            "sel": (np.where(dep >= 0, ts_idx, -1), dep)}
    return out


def select_policy(portfolio: dict, names_sorted: tuple, n_acc: dict):
    """Max hold-out coverage across policies; first in sorted order on ties
    (strict '>' as in fc.portfolio_certify)."""
    R = next(iter(portfolio.values()))[1].shape[0]
    best = np.full(R, -1, dtype=np.int64)
    best_gate = np.full(R, -1, dtype=np.int64)
    best_cov = np.full(R, -1, dtype=np.int64)
    rows = np.arange(R)
    for i, nm in enumerate(names_sorted):
        q, dep = portfolio[nm]
        cov = np.where(dep >= 0, n_acc[nm][rows, np.maximum(dep, 0)], -1)
        better = (dep >= 0) & (cov > best_cov)
        best = np.where(better, i, best)
        best_gate = np.where(better, dep, best_gate)
        best_cov = np.where(better, cov, best_cov)
    return best, best_gate
