# SafeRestore

Detector-relative risk certificates for selective industrial image restoration.

An inspection pipeline often restores a measured image before a detector acts on
it. SafeRestore treats that restoration as a decision: for each image it ranks
the raw display and five restored candidates, tunes one return gate, and
evaluates that fixed gate on a disjoint certification sample using two one-sided
exact binomial bounds — one for evidence loss among accepted defect-positive
images, one for excess activation among all accepted images. An image is
returned automatically only when both bounds meet their targets; otherwise it is
assigned to review.

The two endpoints deliberately use different populations. Evidence loss is
meaningful only on defect-positive images; excess activation is defined on the
clean region of every accepted image. Keeping the denominators separate stops
apparent performance from moving with defect prevalence.

The certificate is marginal, for one policy fixed before its certification
outcomes are observed, under an image-level i.i.d. working model. It is not an
image-level safety probability, does not transfer under distribution shift, and
does not cover choosing whichever training seed or ablation happens to pass.

Shaoliang Yang and Jun Wang, Santa Clara University.

## Scope of this repository

**Code only.** This repository holds the analysis and figure-generation source,
the `saferestore` package, the regression tests, the deterministic split
manifests, and dataset manifest examples.

It deliberately does **not** contain:

- per-image or per-action record tables;
- trained detector or restorer checkpoints;
- rendered figures or figure/table source tables;
- any dataset imagery.

Those are distributed as the reproducibility archive that accompanies the
preprint as ancillary material. Carinthia-S and KolektorSDD themselves are
distributed by their original authors under their own licenses and are not
redistributed anywhere in this project.

## Main empirical result

The retrospective Carinthia-S study uses training seeds 201, 203, 207, 211, and
223. The same image identities, observation schedule, and tune/certify partition
recur across seeds, so variation is descriptive model-fitting sensitivity.

- The all-action policy passes in 1/5 repetitions, with `12.0% +/- 26.9%`
  pass-gated test coverage.
- Fixed bicubic passes in 3/5, with `23.3% +/- 22.2%`.
- Removing the consistency feature passes in 3/5 (`24.6% +/- 24.9%`).
- An interpolation-only pool passes in 3/5 (`25.2% +/- 23.6%`).
- Reserved morphologies reach 81.1–90.3% evidence-loss incidence under nominal
  gates; no nominal certificate is transferred to them.
- KolektorSDD has detector-validation Dice near 0.03–0.04 and too few tuning and
  certification positives for the stated 15% target.

The evidence supports certification and review as safeguards, but not
adaptive-policy superiority or external-domain readiness.

## Contents

- `src/saferestore/` — degradation schedule, restoration actions, detector and
  restorer models, metrics, scores, certification;
- `scripts/` — analysis driver, ablations, and the generators for the
  statistical figures;
- `tests/` — regression tests for splits, metrics, scores, and certification;
- `splits/` — deterministic split manifests (image identifiers and roles);
- `configs/` — dataset manifest examples.

## Environment

```powershell
py -3.12 -m pip install -r requirements-analysis-lock.txt
py -3.12 -m pip install -e ".[research,dev]"
pytest
```

The tests run without any dataset present.

## Reproducing the analysis

The analysis reads archived image–action records rather than raw imagery. Fetch
the reproducibility archive distributed with the preprint, unpack its
`experiments/` directory beside this checkout, then:

```powershell
py -3.12 -m scripts.run_reported_analysis --seeds 201 203 207 211 223 `
  --folder-template "v4_carinthia_primary_trainseed{seed}" `
  --tune-key v4tunecert `
  --output experiments/v4_reported_analysis_primary
```

The statistical figures are regenerated with one command each:

```powershell
py -3.12 -m scripts.make_policy_comparison_figure   # Figure 4
py -3.12 -m scripts.make_result_figures             # Figures 5, 6, 7, 9, 10
```

Both run from the archived records alone; pass `--output-dir` to choose where
they land.

The generators for the remaining figures are not part of this release. Figure 1
is a hand-specified schematic that encodes no data, and Figures 2, 3, and 8
render Carinthia-S imagery, which is not redistributed here. Their published
outputs and source tables ship with the reproducibility archive instead.

## Record field names

The archived records predate the terminology used in the article:

| Record field | Article quantity |
| --- | --- |
| `deletion_risk`, `deletion_errors`, `deletion_denominator`, `deletion_ucb` | evidence-loss risk, incidents, accepted positives, exact upper bound |
| `invention_risk`, `invention_errors`, `invention_denominator`, `invention_ucb` | excess-activation risk, incidents, accepted images, exact upper bound |
| `deletion_recall_floor`, `invention_fpr_limit` | recall floor, clean-region FPR limit |
| `coverage` | conditional coverage, before the certificate decision |
| `deployable_coverage` | pass-gated coverage, zero for a failed certificate |
| `action_pool`, `restoration_pool` | all-action policy, restoration-only pool |

The names are kept for stability. The article deliberately does not use
"deletion" or "invention" as claims: both endpoints are detector-relative
surrogates, not evidence that a transformation erased or invented physical
structure.

## Related work

The earlier benchmark [*Does Super-Resolution Preserve Defect Evidence?*](https://arxiv.org/abs/2607.17401)
evaluates fixed reconstruction methods at a predeclared low-false-call operating
point. SafeRestore asks a different question — whether an image-specific
raw/restored/review policy can be certified for automatic return — and shares no
model, operating point, or numerical result with it.

## Citation

The arXiv identifier and a BibTeX entry will be added when the preprint is
announced.

## License

MIT, see [LICENSE](LICENSE). The public datasets remain under the licenses set
by their original authors.
