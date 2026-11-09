# WP2: records-only statistics for TIM-26-11920 (R&R)

**Date:** 2026-10-25. **Scope:** deliverables 1–5 of WP2 in
`records/2026-10-25-tim-rr-revision-plan.md` §5. The R2.4 tolerance sweep needs checkpoint
inference, so it is not part of this records-only package.

**Status label.** All grid, sweep and design-effect results are **descriptive and retrospective**.
The frozen records were used during method development, including the choice of gate ordering
(`records/2026-10-25-tim-rr-wp1-provenance.md` §2–4). The grid has 108 cells per domain and no
multiplicity control across cells. No grid cell is a confirmatory test.

**Inputs (read-only).** Carinthia-S: `experiments/v4_carinthia_primary_trainseed{201,203,207,211,223}/`.
KolektorSDD2 (KSDD2): `tim_upgrade/experiments/ksdd2_rich_trainseed{…}/`. Each folder has
`{detector_val,calibration,test}_records.csv`. Analysis code: `tim_upgrade/scripts/run_tim_analysis.py`
and `familywise_certification.py`, imported unchanged. The final margin ordering with Laplace
smoothing is used throughout.

## Method notes

- **Relabelling** (`wp2_common.py`). The definitive code derives evidence loss from
  `pixel_recall < r0`, which is already a parameter. It reads activation from the stored
  `invention_error` column (f0 = 0.002) in two places: the score-model labels (`_target_value`)
  and the qualification labels (`build_policy_arrays`). The harness wraps `read_rows` in both
  `run_tim_analysis` and `run_reported_analysis`. Each row gets `invention_error = clean_fpr > f0`
  and `deletion_error = has_defect and pixel_recall < r0`. The score models are then refitted on
  the new labels. At (0.25, 0.002) the stored labels equal the relabelled ones: 0 mismatches in
  all six CSVs of seed 201 for both domains.
- **Fixed in every cell:** tuning margin 0.50, minimum evidence 15/15, the eight-policy family,
  the hash tuning/hold-out split, and the per-domain score fits (class-balanced for Carinthia-S,
  MLE for KSDD2). α is set equal for both endpoints.
- **Failure-inclusive (FI) acceptance:** the mean held-out test automatic-acceptance fraction over
  5 repetitions. A repetition that qualifies nothing contributes 0.
- **Familywise-minus-marginal advantage** (`grid_cells.csv`):
  - `adv_reps_total`: Σ over the 8 policies of familywise single-policy qualifications (full δ)
    minus Σ of marginal tuned-gate passes.
  - `adv_cov_fi_mean`: the same difference in FI acceptance, averaged over the policies.
  - `port_minus_best_marg_cov_fi`: portfolio FI minus the FI of the best marginal policy, with
    that policy picked post hoc. This oracle is not licensed and favours the marginal side.
- **Flag:** a cell is flagged when `adv_reps_total ≤ 0`, or when any single policy has fewer
  familywise qualifications than marginal passes.

## Commands

All commands run from `tim_rr/wp2/` with `python -B`.

```
python -B baseline_reproduction.py                       # -> baseline_reproduction.json (38 s)
python -B static_tables.py                               # -> deliverables 1, 4, 5 CSVs
python -B run_grid.py --mode grid  --domain carinthia_s  # -> grid_seed_carinthia_s.csv  (540 rows)
python -B run_grid.py --mode grid  --domain kolektor_sdd2
python -B run_grid.py --mode sweep --domain carinthia_s  # -> sweep_seed_carinthia_s.csv (130 rows)
python -B run_grid.py --mode sweep --domain kolektor_sdd2
python -B aggregate.py                                   # -> grid_cells.csv, grid_summary.md,
                                                         #    sweep_curves.csv, fig_alpha_sweep.{pdf,png}
python -B spot_check.py carinthia_s   0.10 0.005 0.20 0.05
python -B spot_check.py kolektor_sdd2 0.10 0.005 0.25 0.10
```

## 0. Baseline reproduction (passed)

The harness was run at r0 ∈ {0.25, 0.50}, f0 = 0.002, α = 0.15, δ = 0.10, for both domains and
all five seeds (20 runs). Each output was compared field by field with the definitive JSON in
`tim_upgrade/experiments/tim_analysis/`. **All 20 are identical, with 0 differing fields**
(`baseline_reproduction.json`, `all_identical: true`).

Carinthia-S, r0 = 0.25:

| Policy | raw | bilinear | bicubic | smooth | sharp | learned | restoration pool | action pool |
|---|---|---|---|---|---|---|---|---|
| Familywise single (expected 4,4,4,4,4,1,3,3) | 4 | 4 | 4 | 4 | 4 | 1 | 3 | 3 |
| Marginal (expected 2,2,3,2,1,0,0,1) | 2 | 2 | 3 | 2 | 1 | 0 | 0 | 1 |

- Carinthia-S portfolio: 5/5, selections [unsharp_bicubic, bilinear, action_pool, action_pool,
  unsharp_bicubic]. This matches the expected values.
- KSDD2 portfolio: 0/5 at both floors. Every single-policy and marginal count is 0.

## 1. Exact sample sizes (R1.2 / R2.5)

`sample_size.csv` gives the minimum accepted n such that the one-sided Clopper–Pearson upper
bound is ≤ α, for δ = 0.10. A second route (smallest n with P(X ≤ e | n, α) ≤ level, via
`scipy.stats.binom`) agrees in all 36 entries: 0 mismatches.

| α | Allocation per endpoint | e=0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| 0.10 | single δ/2 = 0.05 | 29 | 46 | 61 | 76 | 89 | 103 |
| 0.10 | portfolio δ/16 = 0.00625 | 49 | 69 | 87 | 103 | 119 | 134 |
| 0.15 | single | 19 | 30 | 40 | 50 | 59 | 68 |
| 0.15 | portfolio | 32 | 45 | 57 | 68 | 78 | 88 |
| 0.20 | single | 14 | 22 | 30 | 37 | 44 | 50 |
| 0.20 | portfolio | 23 | 33 | 42 | 50 | 58 | 65 |

The α = 0.15 rows match the expected values and Table VI.

Carinthia-S portfolio selections on the hold-out (`holdout_accepted_counts.csv`, read from the
definitive JSONs). The hold-out has n = 329 images, 310 of them positive. "Max e allowed" is the
largest incident count that would still pass at the observed n.

| Seed | Selected | Accepted positives | Loss e (U) | Max e allowed | Accepted images | Activation e (U) | Max e allowed |
|---|---|---|---|---|---|---|---|
| 201 | unsharp_bicubic | 98 | 4 (0.120) | 6 | 107 | 2 (0.081) | 7 |
| 203 | bilinear | 98 | 6 (0.149) | 6 | 106 | 1 (0.066) | 6 |
| 207 | action_pool | 102 | 6 (0.143) | 6 | 110 | 4 (0.107) | 7 |
| 211 | action_pool | 136 | 9 (0.139) | 10 | 147 | 8 (0.120) | 11 |
| 223 | unsharp_bicubic | 116 | 7 (0.139) | 7 | 124 | 3 (0.084) | 8 |

In three of the five selections (seeds 203, 207, 223), the loss endpoint sits exactly at its
maximum allowed count.

## 2. Sensitivity grid (R1.3 / R2.3)

The grid is r0 {0.10, 0.25, 0.50} × f0 {0.001, 0.002, 0.005} × α {0.10, 0.15, 0.20, 0.25} ×
δ {0.05, 0.10, 0.20}, run for both domains and 5 seeds (1,080 runs). Outputs:

- `grid_seed_<domain>.csv`: per-seed rows.
- `grid_cells.csv`: per-cell rows with every requested quantity.
- `grid_summary.md`: heat-map tables and the list of flagged cells.

**Carinthia-S (108 cells)**
- Familywise beats marginal (`adv_reps_total > 0`) in **103 cells**. It ties in 5 cells: 2 are
  0 vs 0, and 3 are nonzero ties, all at r0 = 0.50 with α = 0.10 (3–3, 1–1 and 5–5). It loses in
  0 cells. Summed over all cells: 2,893 familywise qualifications vs 1,956 marginal passes.
- In 7 cells at least one policy has fewer familywise qualifications than marginal passes, by one
  policy each (listed in `grid_summary.md`).
- The portfolio qualifies ≥ 1 repetition in 98 cells and 5/5 in 58 cells.
- Portfolio FI acceptance falls below the post-hoc best marginal policy in 19 cells.
  - 15 of them are at r0 = 0.50 and 2 at (r0 = 0.25, f0 = 0.001, α = 0.10). These are the
    strict corners.
  - In 7 of the 19 the portfolio qualifies nothing while some marginal policy passes. The
    best-marginal FI in those 7 cells is 5.7–12.0%.
  - The other 2 are at (r0 = 0.10, f0 = 0.005, α = 0.20, δ ∈ {0.10, 0.20}): portfolio 97.5% vs
    99.9%.
  - Column `flag_portfolio_below_best_marg_cov` in `grid_cells.csv` lists all 19.
- Held-out test incident rates never exceed α for any qualified gate: marginal 0/1,956 pairs,
  familywise 0/2,893, portfolio 0/410. These test rates are point estimates.
- f0 is the strongest driver of acceptance. At r0 = 0.25, α = 0.15, δ = 0.10, portfolio FI
  acceptance is 23.6% at f0 = 0.001, 45.1% at 0.002 and 80.9% at 0.005.

**KSDD2 (108 cells)**
- The portfolio qualifies ≥ 1 repetition in only 33 cells, and 5/5 in 1 cell
  (r0 = 0.25, f0 = 0.005, α = 0.25, δ = 0.20; FI 9.3%).
- At α = 0.15 it qualifies in 2 of 27 cells: r0 = 0.10, f0 = 0.005, δ ∈ {0.10, 0.20}, 1/5 each,
  FI 2.0%.
- Familywise beats marginal in 53 cells. It ties in 46 cells, 36 of them 0 vs 0. It is **worse
  in 9 cells**, all with small counts:
  - r0 = 0.10, f0 = 0.001: 6 cells.
  - r0 = 0.10, f0 = 0.002, α = 0.10: 2 cells.
  - r0 = 0.50, f0 = 0.005, α = 0.20, δ = 0.10: 1 cell.
- In 21 cells at least one policy is worse under familywise. Portfolio FI falls below the
  post-hoc best marginal policy in 40 cells, though the best-marginal FI there is at most 17.9%.
- **Test exceedances occur on KSDD2.** Some qualified gates have a held-out test incident rate
  above α (descriptive point estimates):
  - marginal: 38/130 (policy, repetition) pairs;
  - familywise single: 85/627;
  - portfolio selection: 9/67.
- Activation drives most of these (familywise 72 activation vs 13 loss). All 9 portfolio cases are
  activation exceedances (test activation 0.21–0.28 vs α = 0.20–0.25). Their hold-out accepted
  sets are dominated by positives: for example 26/26, 33/35 and 51/68.
- The cause is not established. Candidates are the different test mixture (see §5), sampling
  noise, and the hold-out reuse during KSDD2 development. Treat these as possible validity
  warnings, not as confirmations.

## 3. Stricter-requirement curves (R2.3)

Settings: δ = 0.10, r0 = 0.25, f0 = 0.002, α from 0.05 to 0.30 in steps of 0.01.

- Data: `sweep_curves.csv`.
- Figure: `fig_alpha_sweep.pdf` / `.png`. The figure is 3.5 in wide, all text is ≥ 8 pt, and it
  has gridlines. The y-axis is FI held-out test automatic acceptance in %.
- The α = 0.15 point equals the baseline: portfolio 5/5, 45.1%.
- "Best single" is the single policy (full δ) with the highest FI, picked post hoc per α.
  "Tuning-sel." is the valid tuning-selected single-policy reading.

| α | Carinthia portfolio (q, FI %) | Best single, post hoc | Tuning-sel. | KSDD2 portfolio | KSDD2 best single | KSDD2 tuning-sel. |
|---|---|---|---|---|---|---|
| 0.10 | 1/5, 9.2 | 24.4 | 17.7 | 0/5, 0 | 0 | 0 |
| 0.12 | 5/5, 35.5 | 32.2 | 28.5 | 0/5, 0 | 0 | 0 |
| 0.15 | 5/5, 45.1 | 35.9 | 36.7 | 0/5, 0 | 0 | 0 |
| 0.20 | 5/5, 50.1 | 52.7 | 59.7 | 0/5, 0 | 2.1 | 1.1 |
| 0.25 | 5/5, 56.7 | 56.8 | 58.6 | 2/5, 4.7 | 8.2 | 5.0 |
| 0.30 | 5/5, 68.0 | 62.2 | 66.7 | 3/5, 8.6 | 9.2 | 9.7 |

**Carinthia-S.** Automatic acceptance falls from 45.1% at α = 0.15 to 9.2% at α = 0.10, and to
0 at α ≤ 0.08.
- The portfolio leads the post-hoc best single policy only for α between 0.12 and 0.19, and again
  from 0.26 upwards.
- For α between 0.19 and 0.28, the valid tuning-selected single policy accepts more than the
  portfolio (for example 59.7% vs 50.1% at α = 0.20).
- For α ≤ 0.11 the portfolio's δ/16 allocation costs acceptance (9.2% vs 24.4% at α = 0.10).

**KSDD2.** Nothing qualifies below α = 0.22 under the portfolio, or below α = 0.16 under any single
policy.

## 4. Design effect (R1.4)

`design_effect.csv` uses DE = 1 + (m−1)ρ and n_eff = n/DE. "Nominal n" is the smallest n whose
n_eff passes at α = 0.15.

- Main variant: e is kept unscaled, so nominal n = ⌈DE·n_min(e)⌉. This is conservative.
- `_KG` columns: a Korn–Graubard-style variant that scales both e and n by 1/DE.

Excerpt for m = 10:

| ρ | DE | Single e=0/1/2 | Portfolio e=0/1/2 | Portfolio, KG variant |
|---|---|---|---|---|
| 0 | 1.00 | 19 / 30 / 40 | 32 / 45 / 57 | 32 / 45 / 57 |
| 0.01 | 1.09 | 21 / 33 / 44 | 35 / 49 / 62 | 35 / 48 / 60 |
| 0.05 | 1.45 | 27 / 44 / 58 | 46 / 65 / 82 | 46 / 60 / 72 |
| 0.10 | 1.90 | 36 / 57 / 76 | 60 / 85 / 108 | 60 / 74 / 87 |
| 0.20 | 2.80 | 52 / 84 / 112 | 88 / 126 / 158 | 88 / 102 / 116 |

`design_effect_max_rho.csv` gives the largest ρ at which each Carinthia-S portfolio selection would
still pass both endpoints with n_eff. The values below are the conservative variant:

| Seed | m = 5 | m = 10 | m = 25 |
|---|---|---|---|
| 201 | 0.066 | 0.029 | 0.011 |
| 203 | 0.002 | 0.001 | 0.0003 |
| 207 | 0.012 | 0.005 | 0.002 |
| 211 | 0.021 | 0.010 | 0.004 |
| 223 | 0.022 | 0.010 | 0.004 |

The KG values are about 2–3× larger. Loss is the binding endpoint in every case.

**Reading.** Every selection is fragile to clustering, because each sits close to its boundary.
At m = 10, ρ ≈ 0.01 breaks four of the five.

## 5. Failure-inclusive Table IV (E13) and prevalence (E12)

Sources: `table4_failure_inclusive.csv` (both domains) and `table4_per_seed.csv`. The draft LaTeX
replacement is `tab_measurement_performance_fi.tex`.

Settings: Carinthia-S, r0 = 0.25, f0 = 0.002, α = 0.15, δ = 0.10. Test loss and activation are
conditional rates; gated rows average them over qualified repetitions only. The CSV also gives
per-unit FI incident rates.

| Rule | Qualified reps | FI test auto-acc. % (range) | Test loss % | Test act. % |
|---|---|---|---|---|
| Always acquired image, ungated | n/a | 100 | 48.7 | 3.5 |
| Always learned restore, ungated | n/a | 100 | 6.7 | 51.3 |
| Acquired image, qualified gate (δ) | 4/5 | 27.7 (0–46.4) | 5.2 | 0.8 |
| Learned restore, qualified gate (δ) | 1/5 | 9.0 (0–44.8) | 8.9 | 10.0 |
| Restoration-only pool, qualified gate (δ) | 3/5 | 27.8 (0–52.7) | 5.5 | 2.9 |
| Portfolio selective rule (δ/8 per policy) | 5/5 | 45.1 (39.7–52.9) | 7.3 | 2.0 |

- The old Table IV figures 34.6, 44.8 and 46.3 are means over qualified repetitions only. They
  equal `cond_auto_acceptance_mean_qualified_only` in the CSV.
- The old row "Always learned restore … no qualification" in fact reported the learned policy's
  familywise gate.
- KSDD2:
  - Ungated acquired image: loss 37.3%, activation 4.8%.
  - Ungated learned restore: loss 33.3%, activation 9.6%.
  - Every gated rule: 0/5 qualified, FI 0%.

**Prevalence** (`prevalence.csv`, identical across seeds):

| Domain | Qualification split | Tuning | Hold-out | Test |
|---|---|---|---|---|
| KSDD2 | 109/604 (18.0%) | 36/188 | 73/416 (17.5%) | 110/1,004 (11.0%) |
| Carinthia-S | 438/461 | 128/132 | 310/329 | 430/446 |

- The activation endpoint, P(activation incident | accepted), is a mixture-dependent quantity.
  Its guarantee therefore refers to the qualification-split mixture from which the hold-out is
  drawn. For KSDD2 that is the positivity-stratified mixture with about 18% positives, not the
  official-test mixture with 11%.
- The loss endpoint conditions on accepted positives. It does not depend on the clean fraction.

## Spot checks (independent implementation)

`spot_check.py` re-implements the procedure without importing the definitive code, `saferestore`
or `wp2_common`. It covers:
- CSV parsing and relabelling;
- the logistic fits and minimax scores;
- the hash split;
- Laplace margin ordering, fixed sequence and portfolio;
- the marginal gate.

The Clopper–Pearson bound comes from bisection on the binomial CDF. Results on two non-baseline
cells:

- Carinthia-S (r0 0.10, f0 0.005, α 0.20, δ 0.05): **130/130 fields identical** to the grid.
  Portfolio 5/5, FI 97.5%.
- KSDD2 (r0 0.10, f0 0.005, α 0.25, δ 0.10): **130/130 identical**. Portfolio 4/5, FI 21.1%.

The fields compared are, per seed: familywise pass and FI per policy, marginal pass per policy, the
portfolio selection, and portfolio FI. Outputs: `spot_check_*.json`.

## Problems and caveats

- The background shell wrapper that launched the grid ended with exit code 1. The cause was a bad
  trailing `tail -3` with multiple files. All four runs finished and wrote complete CSVs
  (540/540/130/130 rows); the logs were deleted afterwards.
- On Carinthia-S, f0 = 0.005 gives 80–100% acceptance in many cells. The activation endpoint is
  close to non-binding there. Any heat map should say so.
- The marginal comparison keeps the tuning margin at 0.5·α, as prespecified. A different margin
  would change the marginal counts.
- Not done here: the R2.4 radius sweep (needs inference), and a δ-curve beyond the three grid
  values.
