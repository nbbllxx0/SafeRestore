# WP6 prospectively frozen qualification study on MVTec AD — protocol

Status: STOPPED IN DEVELOPMENT (2026-10-25). Never frozen. The sealed roles
(calibration, test) were never opened. Section 12 gives the stop rule and the result.
Version: mvtec-rr-v1 protocol 1. Revision of TIM-26-11920.

## 1. Purpose

The decision letter's main validity objection (AE; R1.1) is that in the
two-domain study, design choices were made while qualification-split
outcomes were visible. This study answers it. Every design choice is fixed,
and the code and models are hashed, before any calibration or test outcome
is computed. The study asks one question: does the paper's frozen KolektorSDD2
policy, run unchanged on a fresh public domain, qualify an
automatic-acceptance rule at the study targets?

Scope note (stated in advance): MVTec AD varies the product, the defect
morphology and the imaging setup, but the observation is still the paper's
controlled degradation instrument. The study therefore answers the
prespecification objection (R1.1). It does not answer the
physical-acquisition objection (R2.1). The physical-acquisition study
planned on M²AD was stopped in development (Section 11).

## 2. Data

- MVTec AD (Bergmann et al., CVPR 2019; CC BY-NC-SA 4.0), archive
  `mvtec_anomaly_detection.tar.xz`, 5,264,982,680 bytes, SHA-256
  `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`. Both
  values equal anomalib's published checksum for the official archive. The
  file was obtained from the Hugging Face mirror
  `micguida1/mvtech_anomaly_detection` because the official download links
  returned HTTP 404 on 2026-10-25.
- Inventory (verified 2026-10-25): 15 categories, 5,354 images, of which
  1,258 are defective with a pixel mask each. Median defect area per category
  ranges from 0.29 % (screw) to 7.9 % (tile) of the image.
- Each image shows a different physical object; MVTec AD exposes no object
  or session identifier. One image is the statistical unit.

## 3. Frozen role split

`splits/mvtec_rr_v1.json` (SHA-256
`2839d2c93829419d9acef18553c6fb5308c13d5ba39b32df13d9f481ffacd522`, frozen
2026-10-25T18:02:06-07:00), rule `mvtec_data.split_records`, stratified by
category. The rule reads identifiers and the defective/defect-free status
implied by the folder layout, never pixels.

| Role | Defective | Defect-free |
|---|---|---|
| train | 379 | 2,894 |
| detector_val | 188 | 511 |
| calibration (qualification split) | 503 | 503 |
| test | 188 | 188 |

The paper's seed-independent hash rule (`stable_tuning_mask`, key
`v4tunecert`) divides the calibration role into tuning (147 defective, 159
defect-free) and hold-out (356 defective, 344 defect-free). Development uses
train and detector_val only.

## 4. Fixed measurement chain (per seed)

The paper's KolektorSDD2 pipeline (`tim_upgrade/scripts/run_ksdd2_pipeline.py`
with `scripts/run_dataset_pipeline.py`) runs unchanged. Only the dataset
class is replaced (`mvtec_data.MvtecSegmentationDataset`: grayscale, bilinear
resize to 512 × 512, masks by nearest neighbour).

- Detector: tiny U-Net, 32 base channels, 15 epochs, batch 8, defect-weighted
  sampling (8:1). The checkpoint is the epoch with the highest
  positive-image dice on detector_val.
- Pixel threshold τ: the 1 − 10⁻³ quantile of detector probabilities over the
  clean pixels of detector_val reference images.
- Observation: the paper's degradation instrument
  (Gaussian blur, 2× bicubic downsampling, additive noise; corruption key
  `v4corr:eval`). Severity is assigned by
  identifier hash.
- Six actions: raw (nearest neighbour), bilinear, bicubic, smoothed
  bicubic, sharpened bicubic, learned residual restorer (6 blocks, 32
  channels, 5 epochs).
- Eight policies: the six single actions, the restoration-only pool and the
  all-action pool.

## 5. Endpoints (identical to the article)

- Evidence loss: a positive image with pixel recall < r₀ = 0.25.
- Excess activation: clean-region false-positive rate > f₀ = 0.002. The
  clean region excludes an 8-px ring around the annotation.
- Both endpoints are detector-relative.

## 6. Ranking scores (identical to the article's KolektorSDD2 rule)

- Ten application-time features (`KSDD2_FEATURES`): entropy, forward
  consistency, detector disagreement, evidence shift, restoration residual,
  high-probability area, peak probability, peak disagreement, secondary peak,
  active components.
- Per action, two maximum-likelihood logistic models
  (`fit_risk_models_calibrated`), fitted on all detector_val images of the
  seed. The worst-endpoint score of the article ranks the actions.

## 7. Qualification (identical to the article)

- Targets α = 0.15 for both endpoints; δ = 0.10; minimum evidence 15
  accepted images and 15 accepted positives on the tuning prefix.
- Gate sequence: margin ordering with Laplace-smoothed tuning rates
  (`familywise_certification.py`).
- Primary reading: Bonferroni portfolio over the eight policies (δ/8 per
  policy, δ/16 per endpoint), max-fraction selection.
- Secondary readings from the same records: the tuning-selected single
  policy at full δ; the single-policy fixed sequences; the marginal tuned
  gate; and the comparator procedures of the article's procedure comparison
  (LTT-Bonferroni, LTT-Holm, SGR, fallback k = 5, fixed sequence with
  Hoeffding–Bentkus, naive select-best).
- Seeds: 401, 403, 407, 409, 419. The detector and restorer are retrained
  per seed. The units are fixed, so seeds are training repetitions, not
  independent samples.
- Code: `analyze_mvtec.py` calls `run_tim_analysis.analyze_domain_seed`
  with the KolektorSDD2 feature set and scoring rule.

## 8. Outcomes reported, whatever they are

1. Primary: per seed, whether the portfolio qualifies a (policy, gate) pair;
   the selected policy; the hold-out automatic-acceptance fraction. The count
   of qualified seeds out of five.
2. Test role: for every qualified selection, the automatic-acceptance
   fraction and both incident rates.
3. Ungated references: always-raw and always-learned incident rates on the
   test role.
4. The secondary and comparator readings of Section 7.
5. Every deviation from this protocol, with its reason.

No success threshold is set. A null result is reported with the
exact-bound frontier, as in the article's KolektorSDD2 analysis.

## 9. Claim limits stated in advance

- The guarantee holds over the hold-out draw for the MVTec AD image
  population of the frozen split, under the image-level i.i.d. working model.
- The calibration role is 50 % defective by design. The excess-activation
  bound refers to that mixture.
- The observation is synthetic (Section 1).

## 10. Access control

- `FREEZE_MANIFEST.json` lists the SHA-256 of this file, the split, the MVTec
  code, the pipeline and analysis code it calls, and every per-seed model and
  development record present at freeze time (`freeze.py`).
- `run_mvtec.py --mode confirm` and `analyze_mvtec.py` refuse to run unless
  every hash verifies. Each opening of the sealed roles is appended to
  `sealed_role_access_log.txt`.

## 11. Development history

- M²AD (physical illumination change, specimen-level units) was the planned
  confirmatory domain. It was stopped in development, before its sealed
  roles were opened. Detector recall on the small M²AD defects left near-zero
  power at α = 0.15 in every development variant (`../confirm/PROTOCOL.md`,
  Section 11; power previews in `../confirm/`).
- MVTec AD was chosen next. Its defects are larger, and the paper's pipeline
  runs on it without domain-specific changes.
- Before the freeze, the MVTec code was tested on a synthetic image tree and
  on development roles only. Changes made before the freeze:
  - a decoded-image cache (bitwise-identical items; `test_mvtec.py`);
  - path-neutral run metadata;
  - the power preview's sample sizes set to the exact hash-rule sizes.
- Development power preview: Section 12.

## 12. Development power preview

Method (`power_preview.py`): fit the ranking scores on hash half A of the
seed's detector_val images (`sha256("mvtec-rr-v1-dry:<id>")`), then draw
tuning and hold-out samples of the exact frozen sizes (Section 3) with
replacement from half B (107 defective, 256 defect-free images), positives
and defect-free images separately; run the portfolio and tuning-selected
readings on each of 200 draws.

- Iteration 1 (written 2026-10-25, before iteration 2 started): the
  article's detector (tiny U-Net, 15 epochs), seed 403
  (`power_preview_seed403.json`). Best detector_val positive dice 0.24.
  Ungated half-B evidence loss is 0.82–0.85 for every action; excess
  activation is 0.14–0.18. Portfolio 0/200; tuning-selected 0/200. Even
  among the 10 % of half-B images ranked safest, evidence loss is
  0.38–0.73. By category, raw evidence loss on detector_val is 1.00 for
  bottle, cable, capsule, metal_nut, pill, screw, toothbrush and
  transistor; the detector works partly only on textures (leather 0.23,
  carpet 0.38). Seed 401 (same configuration, started first, without the
  image cache) was stopped unfinished once iteration 1 failed; it produced
  no output.
- Iteration 2, declared before it was run: the four-level U-Net of
  `../unet4.py` (grayscale input, base 32, GroupNorm), 30 detector epochs.
  Everything else is unchanged. Reason: the article's detector has two
  pooling levels, and its receptive field is far smaller than the
  object-level MVTec defects at 512 px. **Stop rule:** the study goes to the
  freeze only if the iteration-2 preview qualifies the portfolio in at
  least 100 of 200 draws. Otherwise the MVTec study stops in development,
  like M²AD, and no further detector iteration is made.

- Iteration 2 result (`power_preview_seed403_u4.json`; run folder
  `runs/seed403_u4`): best detector_val positive dice 0.36 (30 epochs).
  Ungated half-B evidence loss is 0.74–0.77 for every action; excess
  activation is 0.07–0.11. Portfolio 0/200; tuning-selected 3/200. Among
  the 10 % of half-B images ranked safest, evidence loss is 0.22–0.43.
  The stop rule (at least 100/200 portfolio draws) is not met. **The study
  stops here.** No freeze manifest was written, and `run_mvtec.py --mode
  confirm` was never run.

- Timing record (added 2026-10-25, after the study stopped; nothing above
  was changed). The modification time of this file postdates the
  iteration-2 run, because the result bullet was appended after the run.
  The order of events rests on our own records only; no external timestamp
  exists. All times are UTC, 2026-10-26:
  - 01:50:58: the iteration-2 bullet with the stop rule was written to this
    file (session log of the agent that edited it). The editing tool
    rewrites the whole file, so the file-system creation time records the
    last tool edit. Read at 03:08 UTC, before this timing record was added,
    it was 01:51:00, matching the stop-rule edit; adding this record reset
    it;
  - 01:51:06: the iteration-2 run was launched (session log);
  - 01:51:07: `logs/dev_seed403_u4.log` was created; 01:51:10:
    `runs/seed403_u4` was created (file-system creation times);
  - 02:19:31: the iteration-2 result bullet was appended (session log).
  The portfolio qualified in 0 of 200 draws in both iterations, so any
  stop threshold above zero gives the same decision.
