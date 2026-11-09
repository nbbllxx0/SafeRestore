"""WP1: re-run familywise qualification under the gate-ordering rules that
existed during development (2026-07-27), on the frozen records.

  ascending       first implementation (run at 16:08:29), gates in ascending size
  margin_raw      margin ordering as introduced 16:11 (raw tuning rates)
  margin_laplace  final rule after the 17:01 edit (Laplace-smoothed rates)

Read-only on tim_upgrade/ and experiments/; writes only next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tim_upgrade" / "scripts"))

import familywise_certification as fc  # noqa: E402
import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
from saferestore.certification import DualRiskTargets, clopper_pearson_upper  # noqa: E402

_FINAL = fc.candidate_sequence


def margin_sequence(laplace: bool):
    def seq(tuning_scores, tuning_positive, targets, *, tuning_deletion=None,
            tuning_invention=None, certification_scale=1.0):
        thresholds = _FINAL(tuning_scores, tuning_positive, targets)  # eligible, ascending
        if thresholds.size == 0 or tuning_deletion is None:
            return thresholds
        deletion = np.asarray(tuning_deletion).astype(bool)
        invention = np.asarray(tuning_invention).astype(bool)
        margins = np.empty(len(thresholds))
        coverages = np.empty(len(thresholds))
        for i, threshold in enumerate(thresholds):
            mask = tuning_scores <= threshold
            mpos = mask & tuning_positive
            n_del = max(int(round(int(mpos.sum()) * certification_scale)), 1)
            n_inv = max(int(round(int(mask.sum()) * certification_scale)), 1)
            if laplace:
                r_del = (deletion[mpos].sum() + 1.0) / (mpos.sum() + 2.0) if mpos.any() else 1.0
                r_inv = (invention[mask].sum() + 1.0) / (mask.sum() + 2.0)
            else:
                r_del = float(deletion[mpos].mean()) if mpos.any() else 1.0
                r_inv = float(invention[mask].mean())
            e_del, e_inv = int(round(r_del * n_del)), int(round(r_inv * n_inv))
            margins[i] = min(
                targets.deletion - clopper_pearson_upper(min(e_del, n_del), n_del, targets.delta / 2),
                targets.invention - clopper_pearson_upper(min(e_inv, n_inv), n_inv, targets.delta / 2),
            )
            coverages[i] = mask.sum() / len(tuning_scores)
        return thresholds[np.lexsort((-coverages, -margins))]
    return seq


def ascending(tuning_scores, tuning_positive, targets, **_):
    return _FINAL(tuning_scores, tuning_positive, targets)


VARIANTS = {"ascending": ascending,
            "margin_raw": margin_sequence(False),
            "margin_laplace": margin_sequence(True)}


def main() -> None:
    targets = DualRiskTargets(deletion=0.15, invention=0.15, delta=0.10,
                              tuning_margin=0.50, min_accept=15, min_positive_accept=15)
    domains = {
        "carinthia_s": REPO / "experiments" / "v4_carinthia_primary_trainseed{seed}",
        "kolektor_sdd2": REPO / "tim_upgrade" / "experiments" / "ksdd2_rich_trainseed{seed}",
    }
    summary = {}
    for variant, fn in VARIANTS.items():
        fc.candidate_sequence = fn
        for domain, template in domains.items():
            reported.FEATURES = rta.DOMAIN_FEATURES[domain]
            reported.fit_risk_models = rta.DOMAIN_SCORE_FIT[domain]
            rows = []
            for seed in rta.SEEDS:
                r = rta.analyze_domain_seed(Path(str(template).format(seed=seed)), seed, 0.25, targets)
                pf = r["familywise_portfolio"]
                sel = pf["selected_policy"]
                single = {k: v["certified_threshold"] is not None
                          for k, v in r["familywise_single_policy"].items()}
                rows.append({
                    "seed": seed,
                    "portfolio_selected": sel,
                    "portfolio_test_coverage": (pf.get("selected_test") or {}).get("coverage"),
                    "single_policy_qualified": single,
                    "tuning_selected": r["familywise_tuning_selected"]["selected_policy"],
                    "tuning_selected_qualified":
                        r["familywise_tuning_selected"]["certificate"]["certified_threshold"] is not None,
                })
            summary.setdefault(variant, {})[domain] = rows
    fc.candidate_sequence = _FINAL
    out = Path(__file__).with_name("ordering_variants.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for variant, doms in summary.items():
        for domain, rows in doms.items():
            port = sum(r["portfolio_selected"] is not None for r in rows)
            ts = sum(r["tuning_selected_qualified"] for r in rows)
            pol = {}
            for r in rows:
                for k, v in r["single_policy_qualified"].items():
                    pol[k] = pol.get(k, 0) + int(v)
            print(f"{variant:15s} {domain:14s} portfolio {port}/5  tuning-selected {ts}/5  "
                  f"single {pol}  selections {[r['portfolio_selected'] for r in rows]}")


if __name__ == "__main__":
    main()
