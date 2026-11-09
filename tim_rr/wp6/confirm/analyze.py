"""WP6 confirmatory study, step 3: qualification readings from frozen records.

Confirmatory mode (after the freeze): for every seed in protocol_config.json,
risk models are fitted on detector_val records (all views, maximum likelihood),
then the tuning units fix the gate sequence and the holdout units are split by
identifier hash into a qualification part and a test part.  Four readings are
computed from identical records, exactly as in the two-domain study:
marginal tuned gate, single-policy fixed sequence at full delta, Bonferroni
portfolio over the eight policies, and tuning-selected single policy.  The
test part reports post-decision outcomes of every qualified selection.
Secondary, also prespecified: the WP3 comparator procedures (LTT-Bonferroni,
LTT-Holm, SGR, fallback k=5, fixed sequence with Hoeffding-Bentkus, naive
select-best) on the same gate grid, through tim_rr/wp3/wp3_methods.py.

Development mode (--dry-run): uses detector_val records only.  Specimens are
halved by identifier hash; half A fits the risk models, half B (all views,
NOT independent units) plays tuning / qualification / test.  It exercises the
code path and gives a rough power preview; it is not evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C  # noqa: E402  (also puts tim_rr/wp6 and src on sys.path)

sys.path.insert(0, str(C.REPO / "tim_upgrade" / "scripts"))
sys.path.insert(0, str(C.REPO / "tim_rr" / "wp3"))
import wp3_methods as M  # noqa: E402
from familywise_certification import (  # noqa: E402
    candidate_sequence,
    fixed_sequence_certify,
    portfolio_certify,
    tuning_selected_certify,
)
from saferestore.certification import (  # noqa: E402
    DualRiskTargets,
    certify_threshold,
    choose_threshold,
    evaluate_threshold,
    select_minimax_actions,
)

CONFIG = C.HERE / "protocol_config.json"
NUMERIC = set(C.FEATURES) | {"pixel_recall", "clean_fpr"}
INTEGER = {"has_defect", "deletion_error", "invention_error"}


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
        for k in INTEGER:
            r[k] = int(r[k])
    return rows


def hash_fraction(key: str, salt: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big") / 2**64


def design(rows: list[dict]) -> np.ndarray:
    """Nine image features plus product-category indicators (known at application time)."""

    feats = np.asarray([[r[f] for f in C.FEATURES] for r in rows], dtype=np.float64)
    cats = np.asarray([[float(r["category"] == c) for c in C.CATEGORIES] for r in rows], dtype=np.float64)
    return np.concatenate([feats, cats.reshape(len(rows), len(C.CATEGORIES))], axis=1)


def fit_models(rows: list[dict], seed: int) -> dict:
    """Maximum-likelihood logistic risk models per action (the KolektorSDD2 rule)."""

    models = {}
    for method in C.METHODS:
        local = [r for r in rows if r["method"] == method]
        model = {}
        for target in ("deletion_error", "invention_error"):
            pop = [r for r in local if r["has_defect"]] if target == "deletion_error" else local
            x = design(pop)
            y = np.asarray([r[target] for r in pop], dtype=np.int8)
            if y.size == 0 or np.unique(y).size < 2:
                model[target] = {"constant": float(y[0] if y.size else 1.0)}
                continue
            scaler = StandardScaler().fit(x)
            est = LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear").fit(
                scaler.transform(x), y)
            model[target] = {"coef": est.coef_[0], "intercept": float(est.intercept_[0]),
                             "mean": scaler.mean_, "scale": scaler.scale_,
                             "training_n": int(y.size), "incidents": int(y.sum())}
        models[method] = model
    return models


def predict(model: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = design(rows)
    out = []
    for target in ("deletion_error", "invention_error"):
        spec = model[target]
        if "constant" in spec:
            out.append(np.full(len(rows), spec["constant"]))
            continue
        logits = ((x - spec["mean"]) / np.where(spec["scale"] > 0, spec["scale"], 1.0)) @ spec["coef"] + spec["intercept"]
        out.append(1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40))))
    return out[0], out[1]


def policy_arrays(rows: list[dict], models: dict, targets: DualRiskTargets) -> dict:
    by_method = {m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["sample_id"])
                 for m in C.METHODS}
    ids = [r["sample_id"] for r in by_method["raw"]]
    for m, local in by_method.items():
        if [r["sample_id"] for r in local] != ids:
            raise ValueError(f"sample ordering mismatch for {m}")
    p_del, p_inv = {}, {}
    for m in C.METHODS:
        p_del[m], p_inv[m] = predict(models[m], by_method[m])

    def pack(methods: tuple[str, ...]) -> dict:
        chosen, score = select_minimax_actions(p_del, p_inv, methods, targets)
        sel = [by_method[str(m)][i] for i, m in enumerate(chosen)]
        return {"sample_id": np.asarray(ids, dtype=object), "score": score,
                "positive": np.asarray([r["has_defect"] for r in sel], dtype=bool),
                "deletion": np.asarray([r["deletion_error"] for r in sel], dtype=np.int8),
                "invention": np.asarray([r["invention_error"] for r in sel], dtype=np.int8),
                "method": chosen}

    out = {m: pack((m,)) for m in C.METHODS}
    out["restoration_pool"] = pack(C.RESTORATION_METHODS)
    out["action_pool"] = pack(C.METHODS)
    return out


def sub(arr: dict, mask: np.ndarray) -> dict:
    return {k: np.asarray(v)[mask] for k, v in arr.items()}


def evaluate(arr: dict, threshold: float) -> dict:
    res = evaluate_threshold(arr["score"], threshold, arr["positive"], arr["deletion"], arr["invention"])
    acc = arr["score"] <= threshold
    ms, cs = np.unique(arr["method"][acc], return_counts=True)
    res["method_counts"] = {str(m): int(c) for m, c in zip(ms, cs)}
    return res


def readings(qual_arrays: dict, tuning_mask: np.ndarray, test_arrays: dict | None,
             targets: DualRiskTargets) -> dict:
    """All four readings; qual_arrays cover tuning + qualification units."""

    out = {"marginal": {}, "single_policy": {}}
    for name in C.POLICIES:
        tun, cert = sub(qual_arrays[name], tuning_mask), sub(qual_arrays[name], ~tuning_mask)
        thr = choose_threshold(tun["score"], tun["positive"], tun["deletion"], tun["invention"], targets)
        certificate = certify_threshold(cert["score"], thr, cert["positive"], cert["deletion"],
                                        cert["invention"], targets)
        out["marginal"][name] = {"threshold": thr, "certificate": certificate}
        fs = fixed_sequence_certify(
            tuning_scores=tun["score"], tuning_positive=tun["positive"],
            certification_scores=cert["score"], certification_positive=cert["positive"],
            certification_deletion=cert["deletion"], certification_invention=cert["invention"],
            targets=targets, delta_alloc=targets.delta,
            tuning_deletion=tun["deletion"], tuning_invention=tun["invention"]).to_dict()
        if test_arrays is not None:
            out["marginal"][name]["test"] = evaluate(test_arrays[name], thr)
            if fs["certified_threshold"] is not None:
                fs["test"] = evaluate(test_arrays[name], fs["certified_threshold"])
        out["single_policy"][name] = fs
    arrays = {n: {k: qual_arrays[n][k] for k in ("score", "positive", "deletion", "invention")}
              for n in C.POLICIES}
    port = portfolio_certify(arrays, tuning_mask, targets)
    if port["selected_policy"] is not None and test_arrays is not None:
        port["selected_test"] = evaluate(test_arrays[port["selected_policy"]], port["selected_threshold"])
    tsel = tuning_selected_certify(arrays, tuning_mask, targets)
    thr = tsel["certificate"]["certified_threshold"]
    if thr is not None and test_arrays is not None:
        tsel["selected_test"] = evaluate(test_arrays[tsel["selected_policy"]], thr)
    out["portfolio"] = port
    out["tuning_selected"] = tsel
    out["n"] = {"tuning": int(tuning_mask.sum()), "qualification": int((~tuning_mask).sum()),
                "qualification_positive": int(qual_arrays["raw"]["positive"][~tuning_mask].sum()),
                "test": 0 if test_arrays is None else int(len(test_arrays["raw"]["score"]))}
    return out


COMPARATORS = ("naive", "marginal", "ltt_bonf", "holm", "sgr", "fs", "fallback", "fs_hb", "fs_tuning_sel")


def comparators(qual: dict, tuning_mask: np.ndarray, test: dict | None, targets: DualRiskTargets,
                tuning_selected: str) -> dict:
    """Portfolio-level selections of the WP3 procedures on a common gate grid."""

    names_sorted = tuple(sorted(C.POLICIES))
    pols, pv_cp, pv_hb, n_acc, grids = {}, {}, {}, {}, {}
    for name in C.POLICIES:
        arr = {k: np.asarray(qual[name][k]) for k in ("score", "positive", "deletion", "invention")}
        tun = {k: v[tuning_mask] for k, v in arr.items()}
        hold = {k: v[~tuning_mask] for k, v in arr.items()}
        for part in (tun, hold):
            part["positive"] = part["positive"].astype(bool)
            part["deletion"] = part["deletion"].astype(bool)
            part["invention"] = part["invention"].astype(bool)
        grid = np.sort(candidate_sequence(tun["score"], tun["positive"], targets))
        if grid.size:
            seq = candidate_sequence(tun["score"], tun["positive"], targets,
                                     tuning_deletion=tun["deletion"], tuning_invention=tun["invention"],
                                     certification_scale=len(hold["score"]) / max(len(tun["score"]), 1))
            order = np.searchsorted(grid, seq)
        else:  # no gate meets the minimum evidence: one gate accepting nothing (never qualifies)
            grid, order = np.asarray([-np.inf]), np.asarray([0])
        tuned = choose_threshold(tun["score"], tun["positive"], tun["deletion"], tun["invention"], targets)
        idx = int(np.searchsorted(grid, tuned))
        tuned_idx = idx if idx < len(grid) and grid[idx] == tuned else -1
        acc = hold["score"][None, :] <= grid[:, None]
        c = {"e_del": (acc & hold["positive"] & hold["deletion"]).sum(1)[None, :],
             "n_del": (acc & hold["positive"]).sum(1)[None, :],
             "e_act": (acc & hold["invention"]).sum(1)[None, :],
             "n_act": acc.sum(1)[None, :]}
        pv_cp[name] = M.gate_pvalues(c, targets.deletion, targets.invention, "cp")
        pv_hb[name] = M.gate_pvalues(c, targets.deletion, targets.invention, "hb")
        n_acc[name] = c["n_act"]
        pols[name] = type("Pol", (), {"order": order, "tuned_idx": tuned_idx})()
        grids[name] = grid
    res = M.run_methods(pols, targets.delta, names_sorted, tuning_selected, pv_cp, pv_hb, n_acc)
    out = {}
    n_hold = int((~tuning_mask).sum())
    for method in COMPARATORS:
        pi, gi = res[method]["sel"]
        if pi[0] < 0:
            out[method] = {"selected_policy": None}
            continue
        name = names_sorted[pi[0]]
        thr = float(grids[name][gi[0]])
        entry = {"selected_policy": name, "threshold": thr,
                 "qualification_coverage": int(n_acc[name][0, gi[0]]) / n_hold}
        if test is not None:
            entry["test"] = evaluate(test[name], thr)
        out[method] = entry
    return out


def ungated(arrays: dict) -> dict:
    """Always-<action> reference rates on a set of units (no gate, no review)."""

    res = {}
    for m in C.METHODS:
        a = arrays[m]
        pos = a["positive"]
        res[m] = {"evidence_loss_rate": float(a["deletion"][pos].mean()) if pos.any() else None,
                  "excess_activation_rate": float(a["invention"].mean()), "n": int(len(pos)),
                  "positives": int(pos.sum())}
    return res


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return [json_safe(x) for x in v.tolist()]
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--records", default="detector_val_records.csv",
                    help="dry run only: detector_val records file name")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = DualRiskTargets(deletion=cfg["alpha_evidence_loss"], invention=cfg["alpha_excess_activation"],
                              delta=cfg["delta"], min_accept=cfg["min_accept"],
                              min_positive_accept=cfg["min_positive_accept"])
    seeds = args.seeds or cfg["seeds"]
    if not args.dry_run:
        C.verify_freeze()
    results = []
    for seed in seeds:
        folder = C.HERE / "records" / f"seed{seed}"
        dev = read_records(folder / args.records)
        if args.dry_run:
            half_a = [r for r in dev if hash_fraction(r["specimen"], "m2ad-rr-v1-dry") < 0.5]
            half_b = [r for r in dev if hash_fraction(r["specimen"], "m2ad-rr-v1-dry") >= 0.5]
            models = fit_models(half_a, seed)
            pool, salt_key = half_b, "sample_id"
        else:
            models = fit_models(dev, seed)
            tuning_rows = read_records(folder / "tuning_records.csv")
            holdout_rows = read_records(folder / "holdout_records.csv")
            pool, salt_key = tuning_rows + holdout_rows, "specimen"
        # Tuning / qualification / test assignment by identifier hash.
        ids = sorted({(r["sample_id"], r[salt_key], r["role"]) for r in pool})
        if args.dry_run:
            role_of = {sid: ("tuning" if hash_fraction(key, "m2ad-rr-v1-drytune") < 0.3 else "holdout")
                       for sid, key, _ in ids}
        else:
            role_of = {sid: role for sid, _, role in ids}
        part_of = {sid: ("test" if role_of[sid] == "holdout"
                         and hash_fraction(key, "m2ad-rr-v1-test") < cfg["test_fraction"] else
                         "qual" if role_of[sid] == "holdout" else "tuning")
                   for sid, key, _ in ids}
        qual_rows = [r for r in pool if part_of[r["sample_id"]] in ("tuning", "qual")]
        test_rows = [r for r in pool if part_of[r["sample_id"]] == "test"]
        qual = policy_arrays(qual_rows, models, targets)
        tuning_mask = np.asarray([part_of[s] == "tuning" for s in qual["raw"]["sample_id"]])
        test = policy_arrays(test_rows, models, targets) if test_rows else None
        res = readings(qual, tuning_mask, test, targets)
        res["seed"] = seed
        res["comparators"] = comparators(qual, tuning_mask, test, targets,
                                         res["tuning_selected"]["selected_policy"])
        fs_sel = res["comparators"]["fs"]
        if fs_sel["selected_policy"] != res["portfolio"]["selected_policy"] or (
                fs_sel["selected_policy"] is not None
                and fs_sel["threshold"] != res["portfolio"]["selected_threshold"]):
            raise RuntimeError("WP3 fixed-sequence selection disagrees with portfolio_certify")
        res["ungated_qualification"] = ungated(sub_all(qual, ~tuning_mask))
        if test is not None:
            res["ungated_test"] = ungated(test)
        results.append(res)
        port = res["portfolio"]
        print(f"seed {seed}: n={res['n']} portfolio={port['selected_policy']} "
              f"cov={port['selected_coverage']} tuning_selected={res['tuning_selected']['selected_policy']} "
              f"thr={res['tuning_selected']['certificate']['certified_threshold']}", flush=True)
        print("   comparators:", {m: (e["selected_policy"], round(e.get("qualification_coverage", 0), 3))
                                   for m, e in res["comparators"].items()}, flush=True)
        for name in C.POLICIES:
            fs, mg = res["single_policy"][name], res["marginal"][name]["certificate"]
            print(f"   {name:16s} marginal_pass={mg['passes']} fs_cov={fs.get('coverage')} "
                  f"minalpha={fs.get('min_certifiable_alpha_descriptive')}", flush=True)
    name = "dry_run_results.json" if args.dry_run else "confirmatory_results.json"
    summary = {"mode": "dry_run" if args.dry_run else "confirmatory", "config": cfg, "results": results}
    (C.HERE / name).write_text(json.dumps(json_safe(summary), indent=1), encoding="utf-8")


def sub_all(arrays: dict, mask: np.ndarray) -> dict:
    return {n: sub(a, mask) for n, a in arrays.items()}


if __name__ == "__main__":
    main()
