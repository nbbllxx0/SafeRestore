"""WP6 MVTec AD confirmatory analysis (runs only after the freeze).

For every protocol seed: the paper's analyze_domain_seed (marginal tuned
gate, single-policy fixed sequence, Bonferroni portfolio, tuning-selected
policy, test-role outcomes) with the KolektorSDD2 scoring rule, and the WP3
comparator procedures on the same records (LTT-Bonferroni, LTT-Holm, SGR,
fallback k = 5, fixed sequence with Hoeffding-Bentkus, naive select-best).
Writes confirmatory_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "tim_upgrade" / "scripts", REPO / "tim_rr" / "wp3", HERE):
    sys.path.insert(0, str(p))

import compare_records as CR  # noqa: E402
import run_mvtec  # noqa: E402
import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
import wp3_data as D  # noqa: E402
import wp3_methods as M  # noqa: E402

PROTOCOL_SEEDS = (401, 403, 407, 409, 419)


def main() -> None:
    run_mvtec.verify_freeze()
    rta.DOMAIN_FEATURES["mvtec_ad"] = rta.DOMAIN_FEATURES["kolektor_sdd2"]
    rta.DOMAIN_SCORE_FIT["mvtec_ad"] = rta.DOMAIN_SCORE_FIT["kolektor_sdd2"]
    D.DOMAINS["mvtec_ad"] = HERE / "runs" / "seed{seed}"
    reported.FEATURES = rta.DOMAIN_FEATURES["mvtec_ad"]
    reported.fit_risk_models = rta.DOMAIN_SCORE_FIT["mvtec_ad"]
    names_sorted = D.POLICY_NAMES_SORTED
    results = []
    for seed in PROTOCOL_SEEDS:
        folder = HERE / "runs" / f"seed{seed}"
        main_reading = rta.analyze_domain_seed(folder, seed, 0.25, D.TARGETS)
        sd = D.load_seed("mvtec_ad", seed)
        pv_cp, pv_hb, n_acc = CR.prepare(sd)
        ts = CR.tuning_selected_name(sd)
        res = M.run_methods(sd.policies, D.TARGETS.delta, names_sorted, ts, pv_cp, pv_hb, n_acc)
        comparators = {}
        for method in M.METHODS:
            pi, gi = res[method]["sel"]
            if pi[0] < 0:
                comparators[method] = {"selected_policy": None}
                continue
            name = names_sorted[pi[0]]
            thr = float(sd.policies[name].grid[gi[0]])
            comparators[method] = {"selected_policy": name, "threshold": thr, "test": CR.evaluate(sd, name, thr)}
        results.append({"seed": seed, "main": main_reading, "comparators": comparators})
        port = main_reading["familywise_portfolio"]
        print(f"seed {seed}: portfolio {port['selected_policy']} cov {port['selected_coverage']}", flush=True)
    (HERE / "confirmatory_results.json").write_text(json.dumps(reported.json_safe(results), indent=1),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
