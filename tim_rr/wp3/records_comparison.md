# WP3 record-based comparison (retrospective, descriptive)

alpha = 0.15 (both endpoints), delta = 0.10, r0 = 0.25, f0 = 0.002, 5 training seeds.
Failure-inclusive mean test acceptance: 0 for a repetition in which nothing qualifies.
Test incident rates: mean over repetitions that deployed something (pooled counts in brackets: incidents/denominator).

## carinthia_s

| Method | Portfolio qualified | Portfolio failure-incl. mean test acceptance | Portfolio test loss rate [pooled] | Portfolio test activation rate [pooled] | Single-policy qualified (raw, bil, bic, smooth, sharp, learned, resto, all) | Single-policy failure-incl. mean test acceptance, mean over 8 policies | Covers gate selection | Covers policy selection |
|---|---|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 5/5 | 0.545 | 0.110 [129/1158] | 0.018 [26/1216] | 5,5,5,5,5,1,3,3 | 0.367 | no | no |
| Marginal single tuned gate (legacy) | 3/5 (Bonferroni 1/8 variant) | 0.226 | 0.053 [27/475] | 0.011 [5/505] | 2,2,3,2,1,0,0,1 | 0.118 | no gate selection (one gate fixed on tuning) | no; yes only for the Bonferroni 1/8 portfolio variant |
| LTT-Bonferroni over gate grid | 1/5 | 0.091 | 0.052 [10/194] | 0.010 [2/204] | 3,3,1,3,3,0,3,2 | 0.168 | yes | yes (Bonferroni 1/8) |
| LTT-Holm step-down | 1/5 | 0.091 | 0.052 [10/194] | 0.010 [2/204] | 3,3,1,3,3,0,3,2 | 0.168 | yes | yes (joint Holm over all pairs) |
| SGR binary search | 4/5 | 0.334 | 0.058 [40/703] | 0.021 [15/745] | 3,4,3,4,5,0,3,3 | 0.254 | yes (tested gates) | yes (Bonferroni 1/8) |
| Fixed sequence, margin order (ours) | 5/5 | 0.451 | 0.073 [70/953] | 0.020 [20/1005] | 4,4,4,4,4,1,3,3 | 0.282 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, tuning-selected policy (ours) | 4/5 | 0.367 | 0.088 [68/776] | 0.008 [7/818] | 4,4,4,4,4,1,3,3 | 0.282 | yes | yes (tuning-role selection) |
| Fallback/graphical, k=5 equal weights | 5/5 | 0.447 | 0.073 [69/944] | 0.020 [19/996] | 3,4,4,4,5,0,3,3 | 0.276 | yes | yes (Bonferroni 1/8) |
| Fixed sequence with Hoeffding-Bentkus | 5/5 | 0.443 | 0.076 [72/936] | 0.019 [18/988] | 3,4,2,4,4,0,3,3 | 0.243 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, ascending order (first implementation, WP1) | 0/5 | 0.000 | --  | --  | 2,3,3,2,3,0,1,2 | 0.150 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, IUT endpoint combination (variant) | 5/5 | 0.454 | 0.073 [70/959] | 0.010 [11/1013] | 4,4,4,4,4,1,3,3 | 0.299 | yes | yes (Bonferroni 1/8) |
| LTT-Holm, IUT endpoint combination (variant) | 1/5 | 0.093 | 0.056 [11/197] | 0.010 [2/207] | 3,3,3,3,5,0,3,3 | 0.219 | yes | yes (joint Holm) |

CRC operating point (expectation-level joint-rate guarantee; NOT equivalent):

| Policy | reps with a gate | mean test acceptance | mean test loss rate | mean test activation rate |
|---|---|---|---|---|
| raw | 5/5 | 0.601 | 0.236 | 0.018 |
| bil | 5/5 | 0.639 | 0.230 | 0.016 |
| bic | 5/5 | 0.616 | 0.239 | 0.014 |
| smooth | 5/5 | 0.638 | 0.224 | 0.016 |
| sharp | 5/5 | 0.607 | 0.230 | 0.017 |
| learned | 5/5 | 0.452 | 0.028 | 0.393 |
| resto | 5/5 | 0.672 | 0.069 | 0.181 |
| all | 5/5 | 0.681 | 0.073 | 0.184 |

## kolektor_sdd2

| Method | Portfolio qualified | Portfolio failure-incl. mean test acceptance | Portfolio test loss rate [pooled] | Portfolio test activation rate [pooled] | Single-policy qualified (raw, bil, bic, smooth, sharp, learned, resto, all) | Single-policy failure-incl. mean test acceptance, mean over 8 policies | Covers gate selection | Covers policy selection |
|---|---|---|---|---|---|---|---|---|
| Naive select-best (no correction; invalid) | 2/5 | 0.024 | 0.037 [4/104] | 0.161 [19/119] | 1,0,1,0,0,2,1,1 | 0.008 | no | no |
| Marginal single tuned gate (legacy) | 0/5 (Bonferroni 1/8 variant) | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | no gate selection (one gate fixed on tuning) | no; yes only for the Bonferroni 1/8 portfolio variant |
| LTT-Bonferroni over gate grid | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (Bonferroni 1/8) |
| LTT-Holm step-down | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (joint Holm over all pairs) |
| SGR binary search | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes (tested gates) | yes (Bonferroni 1/8) |
| Fixed sequence, margin order (ours) | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, tuning-selected policy (ours) | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (tuning-role selection) |
| Fallback/graphical, k=5 equal weights | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (Bonferroni 1/8) |
| Fixed sequence with Hoeffding-Bentkus | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, ascending order (first implementation, WP1) | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (Bonferroni 1/8) |
| Fixed sequence, IUT endpoint combination (variant) | 0/5 | 0.000 | --  | --  | 0,1,1,0,0,1,0,0 | 0.005 | yes | yes (Bonferroni 1/8) |
| LTT-Holm, IUT endpoint combination (variant) | 0/5 | 0.000 | --  | --  | 0,0,0,0,0,0,0,0 | 0.000 | yes | yes (joint Holm) |

CRC operating point (expectation-level joint-rate guarantee; NOT equivalent):

| Policy | reps with a gate | mean test acceptance | mean test loss rate | mean test activation rate |
|---|---|---|---|---|
| raw | 5/5 | 0.167 | 0.142 | 0.199 |
| bil | 5/5 | 0.186 | 0.169 | 0.251 |
| bic | 5/5 | 0.196 | 0.179 | 0.153 |
| smooth | 5/5 | 0.185 | 0.166 | 0.230 |
| sharp | 5/5 | 0.160 | 0.202 | 0.115 |
| learned | 5/5 | 0.260 | 0.162 | 0.273 |
| resto | 5/5 | 0.184 | 0.175 | 0.213 |
| all | 5/5 | 0.171 | 0.169 | 0.213 |
