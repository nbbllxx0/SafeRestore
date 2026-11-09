"""Development power preview for the M2AD confirmatory study (detector_val only).

Risk models are fitted on detector_val specimens of hash half A.  From half B,
tuning and holdout samples of the frozen sizes (79+79 and 200+200 specimens,
one view each) are drawn by a specimen-level bootstrap: a specimen is drawn
with replacement, then one of its eligible views (any view for Good, an
anomalous view for NG).  For each candidate test fraction the portfolio and
tuning-selected readings are run on every draw.  Output: probability of
qualification and mean qualified acceptance.  Development evidence only; it
fixes TEST_FRACTION in the protocol and says nothing about the hold-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import analyze as A  # noqa: E402
import common as C  # noqa: E402


def frontier(arrays: dict) -> dict:
    """Per policy, over all half-B views ranked by score: ranking AUCs and the empirical
    loss / activation rates of the best-ranked 10, 20, 30 and 50 % (descriptive)."""

    from sklearn.metrics import roc_auc_score

    out = {}
    for name, a in arrays.items():
        order = np.argsort(a["score"], kind="stable")
        pos = a["positive"][order]
        loss = a["deletion"][order].astype(bool) & pos
        act = a["invention"][order].astype(bool)
        n = np.arange(1, len(order) + 1)
        cp, cl, ca = np.cumsum(pos), np.cumsum(loss), np.cumsum(act)
        entry = {"auc_loss": float(roc_auc_score(a["deletion"][a["positive"]], a["score"][a["positive"]])),
                 "auc_activation": float(roc_auc_score(a["invention"], a["score"]))}
        for c in (0.1, 0.2, 0.3, 0.5):
            k = int(c * len(order)) - 1
            entry[f"top{int(c * 100)}"] = {"loss": float(cl[k] / max(cp[k], 1)), "activation": float(ca[k] / n[k]),
                                           "accepted_positives": int(cp[k])}
        out[name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--test-fractions", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--tag", default="", help="record folder suffix (development variants)")
    args = ap.parse_args()
    cfg = json.loads(A.CONFIG.read_text(encoding="utf-8"))
    targets = A.DualRiskTargets(deletion=cfg["alpha_evidence_loss"], invention=cfg["alpha_excess_activation"],
                                delta=cfg["delta"], min_accept=cfg["min_accept"],
                                min_positive_accept=cfg["min_positive_accept"])
    rows = A.read_records(C.HERE / "records" / f"seed{args.seed}{args.tag}" / "detector_val_records.csv")
    half = {r["specimen"]: A.hash_fraction(r["specimen"], "m2ad-rr-v1-dry") < 0.5 for r in rows}
    models = A.fit_models([r for r in rows if half[r["specimen"]]], args.seed)
    pool = [r for r in rows if not half[r["specimen"]]]
    arrays = A.policy_arrays(pool, models, targets)
    ids = arrays["raw"]["sample_id"]
    positive = arrays["raw"]["positive"]
    by_spec: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        spec = sid.rsplit("/", 1)[0]
        kind = spec.split("/")[1]
        if kind == "NG" and not positive[i]:
            continue  # NG unit must be an anomalous view (mask present)
        by_spec.setdefault(spec, []).append(i)
    good = sorted(s for s in by_spec if s.split("/")[1] == "Good")
    ng = sorted(s for s in by_spec if s.split("/")[1] == "NG")
    rng = np.random.default_rng(args.seed)

    def draw(specs: list[str], n: int) -> list[int]:
        picks = rng.integers(len(specs), size=n)
        return [by_spec[specs[j]][rng.integers(len(by_spec[specs[j]]))] for j in picks]

    out = {"seed": args.seed, "draws": args.draws, "half_b_specimens": {"good": len(good), "ng": len(ng)},
           "ungated_half_b": A.ungated(arrays), "frontier_half_b": frontier(arrays), "results": {}}
    for name, f in out["frontier_half_b"].items():
        print(f"{name:16s} {f}", flush=True)
    for tf in args.test_fractions:
        n_test_each = int(round(200 * tf))
        stats = {"portfolio_qualified": 0, "portfolio_cov": [], "tuning_selected_qualified": 0,
                 "tuning_selected_cov": [], "selected": {}}
        for _ in range(args.draws):
            tune = draw(good, 79) + draw(ng, 79)
            hold = draw(good, 200 - n_test_each) + draw(ng, 200 - n_test_each)
            idx = np.asarray(tune + hold)
            qual = {n: {k: np.asarray(v)[idx] for k, v in a.items()} for n, a in arrays.items()}
            mask = np.zeros(len(idx), bool)
            mask[: len(tune)] = True
            core = {n: {k: qual[n][k] for k in ("score", "positive", "deletion", "invention")} for n in qual}
            port = A.portfolio_certify(core, mask, targets)
            if port["selected_policy"] is not None:
                stats["portfolio_qualified"] += 1
                stats["portfolio_cov"].append(port["selected_coverage"])
                stats["selected"][port["selected_policy"]] = stats["selected"].get(port["selected_policy"], 0) + 1
            ts = A.tuning_selected_certify(core, mask, targets)
            if ts["certificate"]["certified_threshold"] is not None:
                stats["tuning_selected_qualified"] += 1
                stats["tuning_selected_cov"].append(ts["certificate"]["coverage"])
        res = {"p_portfolio": stats["portfolio_qualified"] / args.draws,
               "mean_portfolio_cov_when_qualified": float(np.mean(stats["portfolio_cov"])) if stats["portfolio_cov"] else None,
               "p_tuning_selected": stats["tuning_selected_qualified"] / args.draws,
               "mean_ts_cov_when_qualified": float(np.mean(stats["tuning_selected_cov"])) if stats["tuning_selected_cov"] else None,
               "portfolio_selected_counts": stats["selected"]}
        out["results"][str(tf)] = res
        print(f"test_fraction={tf}: {res}", flush=True)
    print("ungated half B:", json.dumps(out["ungated_half_b"]), flush=True)
    (C.HERE / f"power_preview_seed{args.seed}{args.tag}.json").write_text(json.dumps(A.json_safe(out), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
