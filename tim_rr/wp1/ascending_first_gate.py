"""WP1: first-tested gate evidence under ascending ordering, Carinthia-S portfolio (delta/8)."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src", REPO / "tim_upgrade" / "scripts"):
    sys.path.insert(0, str(p))
import ordering_variants as ov  # noqa: E402  (same folder)
import familywise_certification as fc  # noqa: E402
import run_tim_analysis as rta  # noqa: E402
import scripts.run_reported_analysis as reported  # noqa: E402
from saferestore.certification import DualRiskTargets  # noqa: E402

fc.candidate_sequence = ov.ascending
t = DualRiskTargets(deletion=0.15, invention=0.15, delta=0.10, tuning_margin=0.50,
                    min_accept=15, min_positive_accept=15)
reported.FEATURES = rta.DOMAIN_FEATURES["carinthia_s"]
reported.fit_risk_models = rta.DOMAIN_SCORE_FIT["carinthia_s"]
for seed in rta.SEEDS:
    r = rta.analyze_domain_seed(REPO / "experiments" / f"v4_carinthia_primary_trainseed{seed}", seed, 0.25, t)
    ff = {k: v.get("first_failure") for k, v in r["familywise_portfolio"]["policies"].items()}
    cells = [f"{k[:6]}:{f['deletion_errors']}/{f['deletion_denominator']},{f['invention_errors']}/{f['invention_denominator']}"
             for k, f in ff.items() if f]
    print(seed, " ".join(cells))
