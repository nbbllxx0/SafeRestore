# WP6 confirmatory study on M²AD — frozen protocol

Status: DRAFT (not frozen). Sealed roles (tuning, holdout) have not been opened.
Version: m2ad-rr-v1 protocol 1. Revision of TIM-26-11920.

## 1. Purpose

This study answers the main validity objection of the decision letter
(AE; R1.1): in the two-domain study, design choices were made while
qualification-split outcomes were visible. Here, every design choice is fixed,
and the code is hashed, before any tuning or hold-out outcome is computed.
The study asks whether the familywise qualification procedure qualifies an
automatic-acceptance rule on a fresh public domain with a physical
acquisition change, not a synthetic degradation (R2.1).

## 2. Data and roles

- Dataset: M²AD (Apache-2.0; HF `ChengYuQi99/M2AD`), 10 categories, 998
  specimens (499 Good, 499 NG), 12 views × 10 illuminations, 1024×1024 RGB.
- Acquired regime: illumination I07 (poor). Reference regime: I10. Both are
  physical acquisitions of the same specimen and view, aligned to 0 px.
- Frozen specimen split `splits/m2ad_rr_v1.json`
  (SHA-256 `1b967024…27f3`, frozen 2026-10-25T11:54:42-07:00), stratified by
  category and Good/NG: train 150/150, detector_val 70/70, tuning 79/79,
  holdout 200/200 specimens (Good/NG).
- Development used only train and detector_val.

## 3. Units

- One unit per specimen: one view. Good specimen: any of its 12 views. NG
  specimen: one of its views flagged `image_anomaly`. The view is the minimum of
  `sha256("m2ad-rr-v1-view:<category>/<kind>/<object>/<view>")` over the
  candidates (`common.select_units`).
- Tuning units: 158 (79 positive). Holdout units: 400 (200 positive).
- The holdout is subdivided by `sha256("m2ad-rr-v1-test:<specimen>")`:
  fraction TEST_FRACTION goes to a test part; the rest is the
  qualification part.
- Independence: specimens are distinct physical objects; one view per
  specimen removes within-specimen correlation. Residual dependence
  (category, acquisition session) is not modelled; Section 9 states the claim
  limit.

## 4. Fixed measurement chain (per seed)

- Detector: four-level RGB U-Net (base 32), trained on I10 views of train
  specimens, DETECTOR_STEPS steps, batch 16, 384-px crops (half
  defect-centred), BCE (pos-weight ≤ 80) plus soft dice, AdamW 1e-3 with
  200-step warm-up and cosine decay. The checkpoint is the one with the
  highest positive-view dice at I10 on a fixed 300-view detector_val subset.
- Pixel threshold τ: the 1 − 10⁻³ quantile of detector probabilities over
  the clean region of detector_val views at I10.
- Six actions applied to the acquired I07 image: raw; global gain; gamma;
  CLAHE (clip 0.01); gain then unsharp mask y = g + (g − G₂ * g), with G₂ a
  Gaussian of σ = 2 px per channel (written out, because scikit-image 0.26
  `unsharp_mask` returned uninitialised values); learned enhancer. Gain and gamma are fitted on train-role foreground means. The
  learned enhancer is a four-level U-Net residual model I07 → I10, L1 loss,
  4000 steps, trained on train specimens, checkpoint by foreground L1 on 60
  detector_val views.
- Eight policies: the six single actions, the restoration pool (five
  enhancements), and the action pool (all six). Within a pool, the action
  minimising the larger target-normalised predicted risk is chosen.

## 5. Endpoints (identical to the two-domain study)

- Evidence loss: positive unit with pixel recall < r₀ = 0.25 on mask ≥ 128.
- Excess activation: clean-region false-positive rate > f₀ = 0.002. Clean
  region = I10 foreground (mean RGB > 20) minus the mask dilated by an 8-px
  ring.
- Detector-relative: both endpoints measure what the fixed detector reports,
  not the true defect state.

## 6. Ranking scores

- Nine application-time features per (unit, action), computed without the
  mask or the I10 reference: entropy, detector disagreement with the raw
  action, evidence shift, restoration residual, high-probability area, top
  0.1 % probability, top 0.1 % disagreement, secondary peak, and number of
  call components. The KolektorSDD2 feature set less forward consistency,
  which needs a known degradation operator that does not exist here.
- Ten product-category indicators are added to the nine features. The
  product type is known at application time on an inspection line, and on
  development data excess activation is category-driven (raw I07: 99 % of
  Car and Tube views, 0–2 % of Bird, Cube and Dice views).
- Per-action logistic models of the two endpoints, maximum likelihood, fitted
  on all detector_val views of the seed.

## 7. Qualification

- Targets α = 0.15 for both endpoints, δ = 0.10, minimum evidence 15
  accepted and 15 accepted positives on the tuning prefix.
- Gate sequence: margin ordering with Laplace-smoothed tuning rates,
  `familywise_certification.py` (SHA-256 in the freeze manifest).
- Primary reading: Bonferroni portfolio over the eight policies (δ/8 per
  policy, δ/16 per endpoint), max-fraction selection.
- Secondary readings, all from the same records: tuning-selected single
  policy at full δ; single-policy fixed sequences; marginal tuned gate; the
  WP3 comparators on the same gate grid (LTT-Bonferroni, LTT-Holm, SGR,
  fallback k = 5, fixed sequence with Hoeffding–Bentkus, naive select-best).
- Seeds: SEEDS (detector and enhancer retrained per seed; units are fixed,
  so seeds are repetitions of training, not independent samples).

## 8. Outcomes reported, whatever they are

1. Primary: per seed, whether the portfolio qualifies a (policy, gate), the
   qualified automatic-acceptance fraction on the qualification part, and
   the selected policy. Count of qualified seeds out of the total.
2. Test part: for every qualified selection, accepted counts and incident
   rates of both endpoints with exact 95 % intervals.
3. Ungated references on the qualification and test parts: always-raw,
   always-<enhancement> incident rates.
4. Comparator readings (Section 7).
5. Any deviation from this protocol, with reason, as a deviation.

No outcome threshold is set for "success": a null result (no qualification)
is reported with the exact-bound frontier as in the KolektorSDD2 analysis.

## 9. Claim limits stated in advance

- The guarantee holds over the qualification draw for the M²AD specimen
  population under the specimen-level i.i.d. working model; categories and
  acquisition sessions are not resampled.
- Prevalence: units are 50 % positive by design. The excess-activation
  bound refers to that mixture.
- Five training seeds share one set of units; they describe training
  variability, not sampling variability.

## 10. Access control

- `FREEZE_MANIFEST.json` lists SHA-256 hashes of this file,
  `protocol_config.json`, the code that defines records and readings, and
  the per-seed model files trained before the freeze.
- `make_records.py` refuses roles tuning and holdout unless every hash
  verifies; each opening is appended to `sealed_role_access_log.txt`.

## 11. Development decisions (made on train and detector_val only)

- Regime pair I07 → I10 chosen on dataset metadata (not-detectable flags,
  dark-pixel fraction, defect contrast): WP4 record.
- Development iterations (all on train and detector_val; logs in `logs/`):
  - pilot v2: detector on I10 only, 12 000 steps; detector_val views: evidence
    loss 45 % at I10, 72 % at raw I07, 80 % with gain (`../pilot_v2_log.txt`;
    the CLAHE evaluation ran out of memory and `pilot_results.json` was not
    written).
  - dev1 (seed 301, I10 only, 24 000 steps, dice 0.42): power preview 0/200
    qualified portfolios at either test fraction; with category indicators,
    tuning-selected 1/200. Evidence loss among the best-ranked 10 % of
    half-B views was 17–36 % for every policy (`power_preview_seed301.json`).
  - dev2 (seed 301, detector on I10 and I07 with gain/gamma jitter, 24 000
    steps, mean I10/I07 dice 0.27): raw I07 excess activation fell from 33 %
    to 2.5 %, but evidence loss stayed at 63 %, and 13–42 % among the
    best-ranked 10 % of half-B views. Power preview: portfolio 1/200 at test
    fraction 0 and 0/200 at 0.25; tuning-selected 2/200 and 3/200
    (`power_preview_seed301_dev2.json`).
  - dev2 with a relaxed pixel rule (10⁻² clean-pixel FPR, τ = 0.929, f₀ =
    0.02; records only, same models): raw evidence loss 43 %, raw excess
    activation 1.8 %, but evidence loss among the best-ranked 10 % of half-B
    views still 19–33 %. Power preview: portfolio 0/200 at both test
    fractions; tuning-selected 0/200 and 3/200
    (`power_preview_seed301_dev2_px0.01.json`).
  - Conclusion of development: with a supervised U-Net trained on 150
    defective specimens, detector recall on the small M²AD defects at the
    10⁻³ pixel-threshold rule is the binding limit; the study as designed has
    near-zero power at α = 0.15. Decision on how to proceed is with the
    authors.
- TEST_FRACTION chosen from the development dry run (`dry_run_results.json`).
