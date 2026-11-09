# Reproducibility records for the IEEE TIM article

This branch (`tim-rr-resubmission`) holds the code and data for the
IEEE Transactions on Instrumentation and Measurement article *Qualified
Acceptance Decision Rules for Vision-Based Surface Inspection with Learned
Image Enhancement*: the split manifests, the frozen per-image, per-action
records, the trained checkpoints, the analysis outputs and the code.
Together they reconstruct every number reported in the article.

## Requirements

Python 3.12 with numpy, scipy, scikit-learn, pandas, matplotlib, torch,
torchvision, scikit-image, Pillow, and pytest. Run the scripts from the repository root. Each script finds the library
(`src/saferestore`) and the records from its own location, so no
installation step and no `PYTHONPATH` is needed.

## Datasets are not included

Obtain the datasets from their original distributors:

- **Carinthia-S**: Kofler and Hasić, Zenodo record 16895427 (CC BY 4.0).
- **KolektorSDD2**: Božič, Tabernik and Skočaj,
  <https://www.vicos.si/resources/kolektorsdd2/> (CC BY-NC-SA 4.0).
- **MVTec AD**: Bergmann et al., <https://www.mvtec.com/company/research/datasets/mvtec-ad>
  (CC BY-NC-SA 4.0). The study used `mvtec_anomaly_detection.tar.xz`,
  SHA-256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`.

The repository holds only derived per-image measurements and our own trained
weights. Image data is needed only to retrain or re-run inference. Every
qualification result is recomputed from the records alone. Dataset paths in
the code are placeholders (`<path to your ... directory>`). The MVTec AD
code reads the environment variable `MVTEC_AD_ROOT`.

## Layout

```
src/saferestore/                      library
scripts/                              project-level modules imported by the analysis
splits/carinthia_s_v4.json            Carinthia-S split
tim_upgrade/splits/kolektor_sdd2_v1.json
tim_upgrade/scripts/                  two-domain pipeline and familywise qualification
    run_tim_analysis.py               definitive two-domain analysis
    familywise_certification.py       fixed-sequence and portfolio procedures
tim_upgrade/tests/                    regression tests
experiments/v4_carinthia_primary_trainseed{201,203,207,211,223}/
                                      frozen Carinthia-S records and checkpoints
tim_upgrade/experiments/ksdd2_rich_trainseed{201,203,207,211,223}/
                                      KolektorSDD2 records and checkpoints
tim_upgrade/experiments/tim_analysis/ definitive analysis outputs
tim_upgrade/experiments/ksdd2_*       KolektorSDD2 development diagnostics
                                      (tolerance 0/4 px, all-image dice
                                      checkpointing, whole-image features)
tim_rr/wp1/                           gate-ordering provenance analysis
tim_rr/wp2/                           sample size, sensitivity grids, alpha
                                      sweep, failure-inclusive measurement table
tim_rr/wp3/                           comparison with existing risk-control
                                      procedures and Monte Carlo validity study
tim_rr/wp5/                           annotation-tolerance sweep, native-image
                                      arm, degradation-family shift, prevalence
tim_rr/wp6/mvtec/                     MVTec AD prospective study (stopped in development)
tim_rr/wp6/confirm/                   M2AD development study (stopped in
                                      development; code, protocol, power previews)
```

## The two prospective studies (stopped in development)

Both studies froze a role split before any model was trained. Each would
open its hold-out only after a development power preview predicted
adequate power. Neither preview did, so both studies stopped with their
hold-outs unopened. No freeze manifest exists, and no calibration, tuning,
hold-out or test records were ever computed for either domain.

- `tim_rr/wp6/mvtec/PROTOCOL.md`: the MVTec AD protocol, its pre-declared
  stop rule and both development iterations. The split is in `splits/`;
  development models and detector-validation records are in
  `runs/seed403` (article's detector) and `runs/seed403_u4` (four-level
  U-Net). The previews are `power_preview_seed403.json` and
  `power_preview_seed403_u4.json`. `freeze.py`, `analyze_mvtec.py` and the
  `confirm` mode of `run_mvtec.py` were written for the freeze and were
  never run.
- `tim_rr/wp6/confirm/PROTOCOL.md`: the M2AD protocol and development
  history. The development records are in `records/seed301*`; the power
  previews are `power_preview_seed301*.json`.

To recompute the previews from the released development records:

```bash
python tim_rr/wp6/mvtec/power_preview.py --seed 403
python tim_rr/wp6/mvtec/power_preview.py --seed 403 --tag _u4
python tim_rr/wp6/confirm/power_preview.py --seed 301
python tim_rr/wp6/confirm/power_preview.py --seed 301 --tag _dev2
python tim_rr/wp6/confirm/power_preview.py --seed 301 --tag _dev2_px0.01
```

## Reproducing the other results

```bash
python tim_upgrade/scripts/run_tim_analysis.py --ksdd2-template "ksdd2_rich_trainseed{seed}" --output tim_rr/tim_analysis_rebuilt
                                                     # two-domain qualification, written beside the frozen outputs
python tim_rr/wp2/baseline_reproduction.py           # checks it against the frozen outputs
python tim_rr/wp3/compare_records.py                 # procedure comparison table
python tim_rr/wp3/montecarlo.py --reps 10000         # Monte Carlo validity study (~7 min)
python -m pytest tim_upgrade/tests tim_rr/wp3/test_wp3_methods.py
```
