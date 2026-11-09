"""Unit tests for wp3_methods on toy examples with known answers.

Run:  python -B tim_rr/wp3/test_wp3_methods.py      (or pytest)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

import wp3_data as D  # noqa: F401  (sets sys.path for saferestore)
import wp3_methods as M
from saferestore.certification import clopper_pearson_upper


def test_cp_duality():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 400))
        e = int(rng.integers(0, n + 1))
        g = float(rng.choice([0.05, 0.1 / 16, 0.1 / 232, 0.0333]))
        cp_pass = clopper_pearson_upper(e, n, g) <= 0.15
        p_pass = M.binom_pvalue(np.array([e]), np.array([n]), 0.15)[0] <= g
        if cp_pass != p_pass:  # allow only floating-point boundary cases
            assert abs(clopper_pearson_upper(e, n, g) - 0.15) < 1e-9, (e, n, g)
    assert M.binom_pvalue(np.array([0]), np.array([0]), 0.15)[0] == 1.0


def test_hb_valid_and_looser_than_exact():
    for n in (20, 50, 150, 400):
        e = np.arange(n + 1)
        p_hb = M.hb_pvalue(e, np.full_like(e, n), 0.15)
        p_cp = M.binom_pvalue(e, np.full_like(e, n), 0.15)
        assert np.all(p_hb >= p_cp - 1e-12)
        pmf = binom.pmf(e, n, 0.15)  # null boundary: true risk = alpha
        for g in (0.01, 0.05, 0.1):
            assert pmf[p_hb <= g].sum() <= g + 1e-12


def test_bonferroni_and_naive():
    pg = np.array([[0.01, 0.03, 0.026, 0.2]])
    assert M.proc_bonferroni(pg, 0.1).tolist() == [[True, False, False, False]]
    assert M.proc_naive(pg, 0.1).tolist() == [[True, True, True, False]]


def test_holm():
    pg = np.array([[0.01, 0.04, 0.03, 0.5], [0.03, 0.2, 0.01, 0.5]])
    q = M.proc_holm(pg, 0.1)
    # row 0: sorted .01<=.025, .03<=.0333, .04<=.05, .5>.1 -> 3 rejections
    # row 1: sorted .01<=.025, .03<=.0333, .2>.05 -> 2 rejections
    assert q.tolist() == [[True, True, True, False], [True, False, True, False]]


def test_fixed_sequence():
    pg = np.array([[0.01, 0.5, 0.02], [0.2, 0.01, 0.01]])
    q = M.proc_fixed_sequence(pg, np.array([2, 0, 1]), 0.05)
    assert q.tolist() == [[True, False, True], [False, False, True]]


def test_fallback():
    order = np.array([0, 1, 2])
    # k = 2: levels 0.025, 0.025 (+ carry), then chain only
    q = M.proc_fallback(np.array([[0.04, 0.02, 0.5]]), order, 0.05, k=2)
    assert q.tolist() == [[False, True, False]]
    q = M.proc_fallback(np.array([[0.02, 0.04, 0.045]]), order, 0.05, k=2)
    assert q.tolist() == [[True, True, True]]   # 0.025 -> 0.05 -> 0.05
    q = M.proc_fallback(np.array([[0.02, 0.06, 0.01]]), order, 0.05, k=2)
    assert q.tolist() == [[True, False, False]]  # chain broken at gate 2


def test_sgr():
    # 8 gates ascending acceptance; gates 0..4 pass at gamma/3, 5..7 fail
    p = np.array([[0.001] * 5 + [0.9] * 3])
    q = M.proc_sgr(p, 0.1)
    # z (1 = largest gate): z=5 -> j=3 pass; z=3 -> j=5 fail; z=4 -> j=4 pass
    assert np.flatnonzero(q[0]).tolist() == [3, 4]
    assert M.sgr_k(8) == 3 and M.sgr_k(116) == 7 and M.sgr_k(174) == 8
    # level is gamma/k: p = 0.04 fails at 0.1/3
    assert not M.proc_sgr(np.array([[0.04] * 8]), 0.1).any()


def test_deploy_tie_rule():
    q = np.array([[True, True, True, False]])
    n = np.array([[10, 12, 12, 20]])
    assert M.deploy_counts(q, n)[0] == 1                       # smallest index on tie
    assert M.deploy_counts(q, n, np.array([3, 2, 1, 0]))[0] == 2  # rank tie-break
    assert M.deploy_counts(np.zeros_like(q), n)[0] == -1


def test_single_gate_exact_size_and_iut():
    # exact: P(p <= g) <= g at the null boundary, for CP and for IUT
    n = 200
    e = np.arange(n + 1)
    p = M.binom_pvalue(e, np.full_like(e, n), 0.15)
    pmf = binom.pmf(e, n, 0.15)
    for g in (0.1, 0.05, 0.1 / 16):
        assert pmf[p <= g].sum() <= g
    # IUT: endpoint 1 at the boundary, endpoint 2 safe (risk 0.01)
    p2 = M.binom_pvalue(e, np.full_like(e, n), 0.15)
    pmf2 = binom.pmf(e, n, 0.01)
    joint = sum(pmf[i] * pmf2[k] for i in range(n + 1) for k in range(n + 1)
                if max(p[i], p2[k]) <= 0.1)
    assert joint <= 0.1


def test_naive_is_invalid_valid_methods_are_valid():
    """10 independent gates, all exactly at the null boundary (risk = alpha).
    Naive select-best exceeds delta; Bonferroni, Holm, FS, fallback, SGR do not."""
    rng = np.random.default_rng(1)
    R, m, n, a, d = 20000, 10, 100, 0.15, 0.10
    e = rng.binomial(n, a, size=(R, m))
    pg = M.binom_pvalue(e, np.full_like(e, n), a)
    order = np.arange(m)
    fw = {
        "naive": M.proc_naive(pg, d).any(1).mean(),
        "bonf": M.proc_bonferroni(pg, d).any(1).mean(),
        "holm": M.proc_holm(pg, d).any(1).mean(),
        "fs": M.proc_fixed_sequence(pg, order, d).any(1).mean(),
        "fallback": M.proc_fallback(pg, order, d).any(1).mean(),
        "sgr": M.proc_sgr(pg, d).any(1).mean(),
    }
    se = np.sqrt(d * (1 - d) / R)
    assert fw["naive"] > 0.3, fw
    for k in ("bonf", "holm", "fs", "fallback", "sgr"):
        assert fw[k] <= d + 3 * se, (k, fw)


def test_crc_gate():
    from compare_records import crc_gate
    n = 39  # (e + 1)/(n + 1) <= 0.15  <=>  e <= 5 when n = 39
    score = np.arange(n, dtype=float)
    positive = np.ones(n, dtype=bool)
    invention = np.zeros(n, dtype=bool)
    deletion = np.zeros(n, dtype=bool)
    deletion[[5, 10, 15, 20, 25, 30]] = True  # 6th incident at score 30
    arr = {"score": score, "positive": positive, "deletion": deletion, "invention": invention}
    grid = np.array([10.0, 25.0, 29.0, 30.0, 38.0])
    assert crc_gate(arr, grid, 0.15, 0.15) == 2       # gate 29 has e = 5; gate 30 has e = 6
    deletion[0] = True                                 # now gate 10 already has e = 3, 25 -> 6
    assert crc_gate(arr, grid, 0.15, 0.15) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
