"""Development power preview for the MVTec AD study (detector_val records only).

Scores follow the paper's KolektorSDD2 rule (ten features, maximum-likelihood
logistic models, worst-endpoint minimax score, eight-policy family), fitted on
detector_val images of hash half A.  From half B, tuning and hold-out samples
of the frozen sizes are drawn with replacement (positives and defect-free
images separately), and the portfolio and tuning-selected readings of
familywise_certification.py are run on every draw.  Development evidence
only: it decides whether the protocol goes forward; it says nothing about
the hold-out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "tim_upgrade" / "scripts"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
from familywise_certification import portfolio_certify, tuning_selected_certify  # noqa: E402
from saferestore.certification import DualRiskTargets  # noqa: E402
from scripts.analyze_router_pool import read_rows  # noqa: E402

TARGETS = DualRiskTargets(deletion=0.15, invention=0.15, delta=0.10, tuning_margin=0.50,
                          min_accept=15, min_positive_accept=15)


def half(sample_id: str) -> bool:
    return int.from_bytes(hashlib.sha256(f"mvtec-rr-v1-dry:{sample_id}".encode()).digest()[:8], "big") % 2 == 0


def frontier(arrays: dict) -> dict:
    """Per policy, over all half-B images ranked by score: ranking AUCs and the empirical
    loss / activation rates of the best-ranked 10, 20, 30 and 50 % (descriptive)."""

    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        return float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else None

    out = {}
    for name, a in arrays.items():
        score, positive = np.asarray(a["score"]), np.asarray(a["positive"], bool)
        deletion, invention = np.asarray(a["deletion"], bool), np.asarray(a["invention"], bool)
        order = np.argsort(score, kind="stable")
        pos = positive[order]
        cp, cl, ca = np.cumsum(pos), np.cumsum(deletion[order] & pos), np.cumsum(invention[order])
        entry = {"auc_loss": auc(deletion[positive], score[positive]), "auc_activation": auc(invention, score)}
        for c in (0.1, 0.2, 0.3, 0.5):
            k = int(c * len(order)) - 1
            entry[f"top{int(c * 100)}"] = {"loss": float(cl[k] / max(cp[k], 1)), "activation": float(ca[k] / (k + 1)),
                                           "accepted_positives": int(cp[k])}
        out[name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--tag", default="", help="run folder suffix of a development variant")
    args = ap.parse_args()
    reported.FEATURES = rta.DOMAIN_FEATURES["kolektor_sdd2"]
    split = json.loads((HERE / "splits" / "mvtec_rr_v1.json").read_text(encoding="utf-8"))
    cal = [r for r in split["records"] if r["split"] == "calibration"]
    # Exact tuning / hold-out sizes of the frozen hash rule (identifiers only, no outcomes).
    tune = reported.stable_tuning_mask([r["sample_id"] for r in cal], tune_key="v4tunecert")
    defect = np.asarray([r["has_defect"] for r in cal], bool)
    sizes = {"tune_pos": int((tune & defect).sum()), "tune_neg": int((tune & ~defect).sum()),
             "hold_pos": int((~tune & defect).sum()), "hold_neg": int((~tune & ~defect).sum())}

    rows = read_rows(HERE / "runs" / f"seed{args.seed}{args.tag}" / "detector_val_records.csv")
    models = rta.fit_risk_models_calibrated([r for r in rows if half(r["sample_id"])], args.seed, 0.25)
    pool = [r for r in rows if not half(r["sample_id"])]
    arrays = reported.build_policy_arrays(pool, models, TARGETS, 0.25)
    positive = arrays["raw_nearest"]["positive"]
    pos_idx, neg_idx = np.flatnonzero(positive), np.flatnonzero(~positive)
    rng = np.random.default_rng(args.seed)

    ungated = {}
    for m in reported.METHODS:
        a = arrays[m]
        ungated[m] = {"evidence_loss_rate": float(a["deletion"][a["positive"]].mean()),
                      "excess_activation_rate": float(a["invention"].mean())}
    p_port = p_ts = 0
    cov_port, cov_ts, selected = [], [], {}
    for _ in range(args.draws):
        tune = np.concatenate([rng.choice(pos_idx, sizes["tune_pos"]), rng.choice(neg_idx, sizes["tune_neg"])])
        hold = np.concatenate([rng.choice(pos_idx, sizes["hold_pos"]), rng.choice(neg_idx, sizes["hold_neg"])])
        idx = np.concatenate([tune, hold])
        mask = np.zeros(len(idx), bool)
        mask[: len(tune)] = True
        core = {n: {k: np.asarray(a[k])[idx] for k in ("score", "positive", "deletion", "invention")}
                for n, a in arrays.items()}
        port = portfolio_certify(core, mask, TARGETS)
        if port["selected_policy"] is not None:
            p_port += 1
            cov_port.append(port["selected_coverage"])
            selected[port["selected_policy"]] = selected.get(port["selected_policy"], 0) + 1
        ts = tuning_selected_certify(core, mask, TARGETS)
        if ts["certificate"]["certified_threshold"] is not None:
            p_ts += 1
            cov_ts.append(ts["certificate"]["coverage"])
    out = {"seed": args.seed, "draws": args.draws, "sizes": sizes,
           "half_b": {"positives": int(len(pos_idx)), "defect_free": int(len(neg_idx))},
           "ungated_half_b": ungated,
           "p_portfolio": p_port / args.draws,
           "mean_portfolio_cov_when_qualified": float(np.mean(cov_port)) if cov_port else None,
           "p_tuning_selected": p_ts / args.draws,
           "mean_ts_cov_when_qualified": float(np.mean(cov_ts)) if cov_ts else None,
           "portfolio_selected_counts": selected,
           "frontier_half_b": frontier(arrays)}
    print(json.dumps(out, indent=1), flush=True)
    (HERE / f"power_preview_seed{args.seed}{args.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
