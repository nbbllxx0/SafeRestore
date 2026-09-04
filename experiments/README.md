# Experiment evidence map

This directory contains the records and summaries used by the SafeRestore
manuscript. Public dataset binaries are not redistributed.

## Authoritative primary evidence

- `v4_carinthia_primary_trainseed{201,203,207,211,223}/` contains the trained
  checkpoint metadata and image-action records for each primary repetition.
- `v4_reported_analysis_primary/summary.json` contains aggregates at the 25% and
  50% recall floors for fixed actions and adaptive pools.
- `v4_reported_analysis_primary/seed_<seed>_floor_<floor>.json` contains the tuned
  gate, endpoint-specific certification counts and bounds, population coverage,
  held-out test behavior, and reserved-morphology diagnostic.

## Supporting evidence

- `v4_reported_analysis_attn/`: attention-gated detector + U-Net restorer family.
- `v4_supporting_ablations_primarypath/`: feature and action-pool ablations,
  including seed-level outcomes.
- `v4_baselines/`: entropy-gated raw and outcome-informed oracle checks.
- `v4_reported_analysis_kolektor/`: item-separated KolektorSDD sample-size and
  detector-competence diagnostic.
- `v4_clean_proxy_sensitivity_seed211/`: explicitly post-hoc inpainted-clean
  sensitivity for the sole passing primary policy.

## Interpretation

A seed-policy pair passes only when both exact upper bounds meet their targets.
Conditional coverage describes the tuned gate before that decision. Pass-gated
coverage equals conditional coverage for a passing marginal certificate and zero
otherwise. Repetition means and standard deviations describe sensitivity to
model fitting on recurring identities; they are not inference over independent
factories and do not justify selecting a passing seed.

Evidence loss uses accepted positives as its denominator. Excess activation uses
all accepted images. The reserved morphology and KolektorSDD results carry no
transferred certificate.
