# WP3 Monte Carlo validity study

10000 replications per (domain, seed) or (scenario, tuning draw); delta = 0.10.
FWER = P(any qualified (policy, gate) violates a population target). 95% Clopper-Pearson intervals on the Monte Carlo proportion.

## Record-based resampling: carinthia_s

| Method | Portfolio FWER, max over seeds [95% CI] | Portfolio FWER, mean | P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | Single-policy FWER, max over policies x seeds [95% CI] |
|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 0.1383 [0.1316, 0.1452] **>delta** | 0.1157 | 0.0469 | 1.000 | 0.508 | 0.0459 [0.0419, 0.0502] |
| Marginal single tuned gate (legacy) | 0.0022 [0.0014, 0.0033] | 0.0004 | 0.0002 | 0.645 | 0.267 | 0.0231 [0.0202, 0.0262] |
| LTT-Bonferroni over gate grid | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0000 | 0.866 | 0.317 | 0.0006 [0.0002, 0.0013] |
| LTT-Holm step-down | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0000 | 0.866 | 0.318 | 0.0008 [0.0003, 0.0016] |
| SGR binary search | 0.0023 [0.0015, 0.0034] | 0.0016 | 0.0007 | 0.979 | 0.388 | 0.0048 [0.0035, 0.0064] |
| Fixed sequence, margin order (ours) | 0.0065 [0.0050, 0.0083] | 0.0049 | 0.0020 | 0.962 | 0.398 | 0.0339 [0.0304, 0.0376] |
| Fixed sequence, tuning-selected policy (ours) | 0.0280 [0.0249, 0.0314] | 0.0164 | 0.0153 | 0.781 | 0.346 | 0.0339 [0.0304, 0.0376] |
| Fallback/graphical, k=5 equal weights | 0.0056 [0.0042, 0.0073] | 0.0041 | 0.0020 | 0.978 | 0.397 | 0.0339 [0.0304, 0.0376] |
| Fixed sequence with Hoeffding-Bentkus | 0.0024 [0.0015, 0.0036] | 0.0019 | 0.0008 | 0.925 | 0.366 | 0.0129 [0.0108, 0.0153] |
| Fixed sequence, ascending order (first implementation, WP1) | 0.0058 [0.0044, 0.0075] | 0.0047 | 0.0026 | 0.618 | 0.240 | 0.0359 [0.0323, 0.0397] |
| Fixed sequence, IUT endpoint combination (variant) | 0.0179 [0.0154, 0.0207] | 0.0147 | 0.0064 | 0.979 | 0.423 | 0.0748 [0.0697, 0.0801] |
| LTT-Holm, IUT endpoint combination (variant) | 0.0003 [0.0001, 0.0009] | 0.0002 | 0.0001 | 0.926 | 0.347 | 0.0015 [0.0008, 0.0025] |

CRC operating point: P(conditional risk of the CRC gate > alpha), mean over policies x seeds = 0.969, max = 1.000 (CRC does not claim to control this quantity).

Population context per seed: violating gates / grid gates (all 8 policies) = 638/930, 719/930, 447/930, 431/930, 562/938; policies whose first FS gate violates = [0, 1, 1, 0, 1].

## Record-based resampling: kolektor_sdd2

| Method | Portfolio FWER, max over seeds [95% CI] | Portfolio FWER, mean | P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | Single-policy FWER, max over policies x seeds [95% CI] |
|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 0.0537 [0.0494, 0.0583] | 0.0353 | 0.0261 | 0.256 | 0.020 | 0.0177 [0.0152, 0.0205] |
| Marginal single tuned gate (legacy) | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0001 | 0.001 | 0.000 | 0.0114 [0.0094, 0.0137] |
| LTT-Bonferroni over gate grid | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |
| LTT-Holm step-down | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |
| SGR binary search | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0006 [0.0002, 0.0013] |
| Fixed sequence, margin order (ours) | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0001 | 0.001 | 0.000 | 0.0078 [0.0062, 0.0097] |
| Fixed sequence, tuning-selected policy (ours) | 0.0009 [0.0004, 0.0017] | 0.0003 | 0.0003 | 0.004 | 0.000 | 0.0078 [0.0062, 0.0097] |
| Fallback/graphical, k=5 equal weights | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0006 [0.0002, 0.0013] |
| Fixed sequence with Hoeffding-Bentkus | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0001 | 0.001 | 0.000 | 0.0021 [0.0013, 0.0032] |
| Fixed sequence, ascending order (first implementation, WP1) | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0080 [0.0063, 0.0099] |
| Fixed sequence, IUT endpoint combination (variant) | 0.0006 [0.0002, 0.0013] | 0.0002 | 0.0002 | 0.004 | 0.000 | 0.0264 [0.0233, 0.0297] |
| LTT-Holm, IUT endpoint combination (variant) | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |

CRC operating point: P(conditional risk of the CRC gate > alpha), mean over policies x seeds = 0.943, max = 1.000 (CRC does not claim to control this quantity).

Population context per seed: violating gates / grid gates (all 8 policies) = 1312/1388, 1274/1387, 1378/1379, 1197/1376, 1297/1385; policies whose first FS gate violates = [4, 6, 8, 1, 4].

## Synthetic stress: boundary_null

| Method | Portfolio FWER, max over draws [95% CI] | Portfolio FWER, mean | P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | Single-policy FWER, max over policies x draws [95% CI] |
|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 0.6591 [0.6497, 0.6684] **>delta** | 0.6271 | 0.6271 | 0.627 | 0.491 | 0.1330 [0.1264, 0.1398] **>delta** |
| Marginal single tuned gate (legacy) | 0.0127 [0.0106, 0.0151] | 0.0083 | 0.0083 | 0.008 | 0.006 | 0.0322 [0.0288, 0.0358] |
| LTT-Bonferroni over gate grid | 0.0007 [0.0003, 0.0014] | 0.0005 | 0.0005 | 0.001 | 0.000 | 0.0016 [0.0009, 0.0026] |
| LTT-Holm step-down | 0.0007 [0.0003, 0.0014] | 0.0005 | 0.0005 | 0.001 | 0.000 | 0.0016 [0.0009, 0.0026] |
| SGR binary search | 0.0050 [0.0037, 0.0066] | 0.0047 | 0.0047 | 0.005 | 0.003 | 0.0093 [0.0075, 0.0114] |
| Fixed sequence, margin order (ours) | 0.0271 [0.0240, 0.0305] | 0.0237 | 0.0237 | 0.024 | 0.020 | 0.0332 [0.0298, 0.0369] |
| Fixed sequence, tuning-selected policy (ours) | 0.0332 [0.0298, 0.0369] | 0.0291 | 0.0291 | 0.029 | 0.025 | 0.0332 [0.0298, 0.0369] |
| Fallback/graphical, k=5 equal weights | 0.0062 [0.0048, 0.0079] | 0.0055 | 0.0055 | 0.005 | 0.005 | 0.0120 [0.0100, 0.0143] |
| Fixed sequence with Hoeffding-Bentkus | 0.0091 [0.0073, 0.0112] | 0.0086 | 0.0086 | 0.009 | 0.007 | 0.0122 [0.0101, 0.0145] |
| Fixed sequence, ascending order (first implementation, WP1) | 0.0118 [0.0098, 0.0141] | 0.0084 | 0.0084 | 0.008 | 0.003 | 0.0274 [0.0243, 0.0308] |
| Fixed sequence, IUT endpoint combination (variant) | 0.0543 [0.0499, 0.0589] | 0.0513 | 0.0513 | 0.051 | 0.043 | 0.0683 [0.0634, 0.0734] |
| LTT-Holm, IUT endpoint combination (variant) | 0.0016 [0.0009, 0.0026] | 0.0015 | 0.0015 | 0.001 | 0.001 | 0.0026 [0.0017, 0.0038] |

CRC operating point: P(conditional risk of the CRC gate > alpha), mean over policies x draws = 1.000, max = 1.000 (CRC does not claim to control this quantity).

Population context per draw: violating gates / grid gates (all 8 policies) = 1185/1185, 1164/1164, 1206/1206, 1225/1225, 1309/1309; policies whose first FS gate violates = [8, 8, 8, 8, 8].

## Synthetic stress: crossing_loss

| Method | Portfolio FWER, max over draws [95% CI] | Portfolio FWER, mean | P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | Single-policy FWER, max over policies x draws [95% CI] |
|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 0.3586 [0.3492, 0.3681] **>delta** | 0.3551 | 0.3489 | 0.695 | 0.367 | 0.0589 [0.0544, 0.0637] |
| Marginal single tuned gate (legacy) | 0.0059 [0.0045, 0.0076] | 0.0035 | 0.0035 | 0.013 | 0.005 | 0.0317 [0.0284, 0.0353] |
| LTT-Bonferroni over gate grid | 0.0006 [0.0002, 0.0013] | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.0007 [0.0003, 0.0014] |
| LTT-Holm step-down | 0.0006 [0.0002, 0.0013] | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.0007 [0.0003, 0.0014] |
| SGR binary search | 0.0028 [0.0019, 0.0040] | 0.0022 | 0.0022 | 0.006 | 0.003 | 0.0049 [0.0036, 0.0065] |
| Fixed sequence, margin order (ours) | 0.0086 [0.0069, 0.0106] | 0.0063 | 0.0063 | 0.019 | 0.009 | 0.0306 [0.0273, 0.0342] |
| Fixed sequence, tuning-selected policy (ours) | 0.0306 [0.0273, 0.0342] | 0.0136 | 0.0136 | 0.037 | 0.016 | 0.0306 [0.0273, 0.0342] |
| Fallback/graphical, k=5 equal weights | 0.0025 [0.0016, 0.0037] | 0.0017 | 0.0017 | 0.004 | 0.002 | 0.0082 [0.0065, 0.0102] |
| Fixed sequence with Hoeffding-Bentkus | 0.0029 [0.0019, 0.0042] | 0.0022 | 0.0022 | 0.010 | 0.004 | 0.0106 [0.0087, 0.0128] |
| Fixed sequence, ascending order (first implementation, WP1) | 0.0039 [0.0028, 0.0053] | 0.0017 | 0.0017 | 0.025 | 0.008 | 0.0155 [0.0132, 0.0181] |
| Fixed sequence, IUT endpoint combination (variant) | 0.0223 [0.0195, 0.0254] | 0.0148 | 0.0147 | 0.045 | 0.020 | 0.0693 [0.0644, 0.0745] |
| LTT-Holm, IUT endpoint combination (variant) | 0.0009 [0.0004, 0.0017] | 0.0005 | 0.0005 | 0.001 | 0.000 | 0.0012 [0.0006, 0.0021] |

CRC operating point: P(conditional risk of the CRC gate > alpha), mean over policies x draws = 0.991, max = 1.000 (CRC does not claim to control this quantity).

Population context per draw: violating gates / grid gates (all 8 policies) = 784/1227, 742/1188, 759/1251, 882/1223, 776/1203; policies whose first FS gate violates = [8, 5, 3, 4, 4].

## Synthetic stress: crossing_both

| Method | Portfolio FWER, max over draws [95% CI] | Portfolio FWER, mean | P(deployed violates), mean | P(qualify), mean | Mean deployed population acceptance | Single-policy FWER, max over policies x draws [95% CI] |
|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 0.0381 [0.0344, 0.0420] | 0.0350 | 0.0345 | 0.073 | 0.033 | 0.0060 [0.0046, 0.0077] |
| Marginal single tuned gate (legacy) | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |
| LTT-Bonferroni over gate grid | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |
| LTT-Holm step-down | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |
| SGR binary search | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0002 [0.0000, 0.0007] |
| Fixed sequence, margin order (ours) | 0.0002 [0.0000, 0.0007] | 0.0001 | 0.0001 | 0.000 | 0.000 | 0.0016 [0.0009, 0.0026] |
| Fixed sequence, tuning-selected policy (ours) | 0.0013 [0.0007, 0.0022] | 0.0005 | 0.0005 | 0.001 | 0.001 | 0.0016 [0.0009, 0.0026] |
| Fallback/graphical, k=5 equal weights | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0002 [0.0000, 0.0007] |
| Fixed sequence with Hoeffding-Bentkus | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0004 [0.0001, 0.0010] |
| Fixed sequence, ascending order (first implementation, WP1) | 0.0001 [0.0000, 0.0006] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0015 [0.0008, 0.0025] |
| Fixed sequence, IUT endpoint combination (variant) | 0.0003 [0.0001, 0.0009] | 0.0002 | 0.0002 | 0.000 | 0.000 | 0.0075 [0.0059, 0.0094] |
| LTT-Holm, IUT endpoint combination (variant) | 0.0000 [0.0000, 0.0004] | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.0000 [0.0000, 0.0004] |

CRC operating point: P(conditional risk of the CRC gate > alpha), mean over policies x draws = 0.998, max = 1.000 (CRC does not claim to control this quantity).

Population context per draw: violating gates / grid gates (all 8 policies) = 1006/1104, 1011/1163, 976/1175, 994/1247, 936/1182; policies whose first FS gate violates = [8, 7, 8, 7, 8].
