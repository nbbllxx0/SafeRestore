# WP3 — Matched comparison with existing risk-control procedures and Monte Carlo validity

**Date:** 2026-10-25. **Manuscript:** TIM-26-11920 (Reject & Resubmit), Reviewer 2 comment R2.2.
**Status:** complete for records-only scope. All numbers below were re-read from
`records_comparison.json` / `montecarlo_results.json` in this folder at write time.

## 0. Headline

- **Fixed sequence (FS) clearly beats grid-wide multiplicity corrections and the legacy
  marginal protocol, but it does not dominate SGR or a fallback sequence.** On Carinthia-S
  (portfolio level), FS qualified 5/5 repetitions with failure-inclusive mean test acceptance
  0.451; fallback (k = 5) 5/5, 0.447; SGR 4/5, 0.334; LTT-Bonferroni and LTT-Holm 1/5, 0.091;
  marginal tuned gate with a Bonferroni 1/8 portfolio 3/5, 0.226. Under resampling of the
  pooled population, FS, fallback and SGR are close (mean deployed population acceptance 0.398,
  0.397, 0.388).
- **FS power depends on the gate ordering.** The same FS with ascending order (the first
  implementation, WP1) qualified 0/5 and had mean deployed population acceptance 0.240 under
  resampling. The margin ordering was adopted after that 0/5 run (WP1 provenance), so the
  record-based FS advantage over SGR (5/5 vs 4/5) is retrospective.
- **KSDD2: no valid procedure qualifies any portfolio (0/5 for all).** Only the invalid naive
  rule "qualifies" (2/5), and its deployed gates have a pooled test activation rate of 0.161
  (19/119), above α = 0.15.
- **Validity:** every valid procedure stayed far below δ = 0.10 in all 10 record populations
  and 15 synthetic stress populations (largest valid portfolio-level FWER 0.0543, FS-IUT
  variant; largest single-policy 0.0748, FS-IUT; main FS 0.0271 / 0.0339). Naive select-best
  exceeded δ: up to 0.138 on Carinthia-S records and 0.659 in the boundary stress.
- **Novelty consequence:** the testing machinery is not the contribution, and the paper
  should not claim a power advantage over SGR or fallback procedures. What the comparison
  supports: (i) uncorrected selection is invalid on these records; (ii) placing gate and policy
  selection inside the risk statement is necessary; (iii) among valid procedures, ordered
  procedures (FS, fallback, SGR's binary search) are much more powerful than grid-wide
  Bonferroni/Holm for nested gates; (iv) the exact binomial bound is tighter than
  Hoeffding–Bentkus. The contribution must be the two-endpoint conditional formulation and the
  measurement-decision framing, as plan §6.3 already proposes.

## 1. Scope, data and provenance caveat

- Records: Carinthia-S `experiments/v4_carinthia_primary_trainseed{201,203,207,211,223}/`,
  KSDD2 `tim_upgrade/experiments/ksdd2_rich_trainseed{...}/`. No retraining, no inference.
- Targets: α = 0.15 for both endpoints (loss among accepted positives at r0 = 0.25; excess
  activation among all accepted images at f0 = 0.002), δ = 0.10, minimum evidence 15 accepted /
  15 accepted positives on the tuning prefix, 8-policy family.
- Roles: tuning (≈30 %, hash key `v4tunecert`), hold-out (Carinthia-S 329 images, KSDD2 416),
  test (Carinthia-S 446, KSDD2 1004). The five seeds share the same hold-out and test images, so
  seed counts are not independent replications.
- **Every record-based comparison here is retrospective and descriptive.** The margin
  ordering with Laplace smoothing was adopted after a 0/5 ascending-order run on the same
  hold-out (records/2026-10-25-tim-rr-wp1-provenance.md §4). The competitors have no
  comparable post-hoc tuning, which favours FS in the record table. The KSDD2 pipeline itself
  was adapted while hold-out outcomes were visible (WP1 §2).

## 2. Methods compared (exact adaptations)

Common to all: same score arrays (minimax worst-endpoint score per policy, as in the published
code), same tuning/hold-out split, same **candidate gate grid per policy** = the unique tuning
scores from the first tuning prefix meeting the 15/15 minimum evidence (`fc.candidate_sequence`
without incident arrays; 116–118 gates per policy on Carinthia-S, 168–174 on KSDD2), same two
conditional endpoints, same δ. Deployment = qualified gate with the largest hold-out acceptance;
portfolio selection = largest hold-out acceptance across policies (ties: first in sorted policy
order, as in `fc.portfolio_certify`).

Endpoint combination. For gate j, the exact endpoint p-value for H0: risk > α is
p = P(Bin(n, α) ≤ e) (p = 1 if n = 0). A Clopper–Pearson bound at level γ meets α iff p ≤ γ
(verified numerically in the unit tests). The published method splits δ over the two endpoints;
all main rows therefore use the gate p-value **p_gate = min(1, 2·max(p_loss, p_act))** (Bonferroni
over endpoints). Variants marked "IUT" use p_gate = max(p_loss, p_act), the intersection–union
test, which is valid for the union null "at least one endpoint violated" (Berger 1982).

| # | Method | Single-policy level (budget δ) | Portfolio level (8 policies) |
|---|---|---|---|
| 1 | Naive select-best (invalid) | every gate tested at p_gate ≤ δ (CP at δ/2 per endpoint), no correction | same, then max acceptance across policies |
| 2 | Marginal single tuned gate (legacy) | the gate chosen by `choose_threshold` (tuning margin 0.5) tested at δ | not defined in the paper; shown as a Bonferroni variant: each policy's tuned gate at δ/8 |
| 3 | LTT-Bonferroni (Angelopoulos et al.) | every gate at δ/m (δ/(2m) per endpoint) | δ/(8·m_P) per gate |
| 4 | LTT-Holm | Holm step-down over the m gates of the policy, p_gate (Bonferroni endpoints) | one joint Holm over all Σ m_P (policy, gate) hypotheses at δ |
| 5 | SGR (Geifman & El-Yaniv) | Algorithm 1 binary search: gates indexed z = 1..m by decreasing acceptance, k = ⌈log2 m⌉ iterations, z = ⌈(zmin+zmax)/2⌉, pass → zmax = z, fail → zmin = z; each tested gate at δ/k (δ/(2k) per endpoint). **Adaptation:** two endpoints via the δ/2 split; the deployed gate is the largest-acceptance tested gate that passed (the original outputs the last tested gate and its bound); validity follows from the union bound over the k tested gates (their Theorem 3.2). As in the original, the largest gate (z = 1) is never tested. | δ/8 per policy |
| 6 | Fixed sequence (ours) | published code: margin order with Laplace smoothing, stop at first failure, δ | δ/8 per policy (published portfolio); also the published tuning-selected reading (policy fixed on tuning, full δ) |
| 7 | Fallback / graphical | weights 1/5 on the first k = 5 gates of the margin order; graph = forward chain (Wiens 2003 fallback; Bretz et al. 2009 graph update): a rejected gate passes its whole level to the next gate; gates after the 5th have weight 0 and are reached only through the chain. **k = 5 and equal weights were fixed by the WP3 brief before running; on these records the rule is retrospective.** | δ/8 per policy |
| 8 | FS with Hoeffding–Bentkus | as 6, with the LTT Proposition 1 HB p-value min(exp(−n h1(R̂∧α, α)), e·P(Bin(n, α) ≤ ⌈nR̂⌉)) instead of the exact binomial | δ/8 per policy |
| – | FS ascending order | the first implementation (WP1): gates in ascending size | δ/8 per policy |
| – | FS-IUT, Holm-IUT | as 6 / 4 with IUT endpoint combination | as 6 / 4 |

**Conformal risk control (CRC, Angelopoulos et al. 2022)** is not tabulated as equivalent. Its
operating point was computed with B = 1 and the monotone losses ℓ = 1{accepted ∧ loss incident}
on positives and ℓ = 1{accepted ∧ activation incident} on all images: largest gate with
(e+1)/(n+1) ≤ α per endpoint, deployed gate = the smaller one.

## 3. Commands (working directory `tim_rr/wp3/`, env `torch5090`)

```
python -B test_wp3_methods.py          # 11 unit tests
python -B compare_records.py           # reproduction + record table  (~30 s)
python -B montecarlo.py --reps 10000   # MC validity                 (~7 min)
python -B figures.py                   # fig_wp3_power / fig_wp3_validity (.pdf, .png)
```

Files: `wp3_data.py` (record loading through the published functions, import pattern from
`tim_rr/wp1/ordering_variants.py`), `wp3_methods.py` (all procedures, vectorised),
`records_comparison.{json,md}`, `montecarlo_results.{json,md}`.

## 4. Reproduction status (verified)

`compare_records.py` runs the unmodified `run_tim_analysis.analyze_domain_seed` for every
(domain, seed) and checks the WP3 implementations against it.

- Carinthia-S portfolio **5/5** (selections unsharp, bilinear, all-action, all-action,
  unsharp); single-policy FS counts **4,4,4,4,4,1,3,3**; marginal **2,2,3,2,1,0,0,1**;
  tuning-selected 4/5. KSDD2 **0/5** everywhere. All match the published summary.
- All 195 equality checks pass: WP3 FS thresholds equal the published certified thresholds for
  every policy and seed, marginal pass/fail equals the legacy certificate, portfolio policy and
  threshold equal the published ones, and the tuning-selected policy and outcome match.
- FS ascending reproduces WP1: portfolio 0/5, single 2,3,3,2,3,0,1,2.

Unit tests (`test_wp3_methods.py`, all 11 pass): CP/p-value duality on 3000 random (e, n, γ);
HB validity and HB ≥ exact; Bonferroni, naive, Holm, FS, fallback and SGR on hand-computed toy
cases; deployment tie rule; exact single-gate size and IUT size; a 20 000-replication null
simulation (10 independent boundary gates) in which naive exceeds δ (> 0.3) and all valid
procedures stay within δ + 3 SE; CRC gate on a hand-computed example.

## 5. Record-based comparison (retrospective)

Failure-inclusive mean = average over 5 seeds of the held-out test acceptance fraction, 0 when
nothing qualifies. Test incident rates: mean over repetitions that deployed; pooled
incidents/denominator in brackets. Full table incl. single-policy acceptance: `records_comparison.md`.

**Carinthia-S**

| Method | Portfolio qualified | Mean test acceptance | Test loss rate | Test activation rate | Single-policy qualified (raw, bil, bic, smooth, sharp, learned, resto, all) | Covers gate sel. | Covers policy sel. |
|---|---|---|---|---|---|---|---|
| Naive select-best | 5/5 | 0.545 | 0.110 [129/1158] | 0.018 [26/1216] | 5,5,5,5,5,1,3,3 | no | no |
| Marginal tuned gate | 3/5 (Bonf. 1/8 variant) | 0.226 | 0.053 [27/475] | 0.011 [5/505] | 2,2,3,2,1,0,0,1 | no selection (one prefixed gate) | only in the variant |
| LTT-Bonferroni | 1/5 | 0.091 | 0.052 [10/194] | 0.010 [2/204] | 3,3,1,3,3,0,3,2 | yes | yes |
| LTT-Holm | 1/5 | 0.091 | 0.052 [10/194] | 0.010 [2/204] | 3,3,1,3,3,0,3,2 | yes | yes |
| SGR | 4/5 | 0.334 | 0.058 [40/703] | 0.021 [15/745] | 3,4,3,4,5,0,3,3 | yes (tested gates) | yes |
| **FS, margin (ours)** | **5/5** | **0.451** | 0.073 [70/953] | 0.020 [20/1005] | 4,4,4,4,4,1,3,3 | yes | yes |
| FS, tuning-selected (ours) | 4/5 | 0.367 | 0.088 [68/776] | 0.008 [7/818] | (as FS) | yes | yes (tuning role) |
| Fallback, k = 5 | 5/5 | 0.447 | 0.073 [69/944] | 0.020 [19/996] | 3,4,4,4,5,0,3,3 | yes | yes |
| FS, Hoeffding–Bentkus | 5/5 | 0.443 | 0.076 [72/936] | 0.019 [18/988] | 3,4,2,4,4,0,3,3 | yes | yes |
| FS, ascending | 0/5 | 0.000 | — | — | 2,3,3,2,3,0,1,2 | yes | yes |
| FS-IUT (variant) | 5/5 | 0.454 | 0.073 [70/959] | 0.010 [11/1013] | 4,4,4,4,4,1,3,3 | yes | yes |
| Holm-IUT (variant) | 1/5 | 0.093 | 0.056 [11/197] | 0.010 [2/207] | 3,3,3,3,5,0,3,3 | yes | yes |

**KSDD2:** every valid procedure 0/5 at portfolio level and 0 single-policy qualifications,
except FS-IUT (single-policy 0,1,1,0,0,1,0,0; portfolio 0/5). Naive "qualifies" 2/5 with mean
test acceptance 0.024, test loss rate 0.037 [4/104] and test activation rate 0.161 [19/119].

**CRC operating point (not equivalent).** Carinthia-S: a gate exists for every policy and seed;
mean test acceptance 0.45–0.68 by policy, but the mean test loss rate among accepted positives
is 0.224–0.239 for the five fixed actions and the activation rate is 0.393 (learned) and
0.18 (pools), all above α = 0.15. KSDD2: mean test activation rate 0.115–0.273 by policy.

## 6. Monte Carlo validity (10 000 hold-out draws per population)

Population = pooled hold-out + test records of one (domain, seed) (Carinthia-S N = 775,
KSDD2 N = 1420); the tuning role stays fixed (grid, ordering, tuned gate, tuning-selected
policy = F0). Each draw is a hold-out of the original size sampled with replacement. A gate
violates if its population conditional risk exceeds α on either endpoint. FWER = P(any
qualified (policy, gate) violates). 95 % CIs are Clopper–Pearson on the MC proportion.

Synthetic stress (N = 20 000 units, prevalence 0.3, 8 correlated policies, tuning n = 200 drawn
once per draw and then fixed, hold-out n = 400; incident patterns are deterministic in score
order so every gate's population risk is known exactly): **boundary_null** — every gate has loss
risk in [0.153, 0.153 + 1/k), activation 0.03, so every qualification is false; **crossing_loss**
— loss risk rises slowly from 0.10 to 0.20 across the gates, so many gates lie within ±0.01 of α;
**crossing_both** — loss crosses α at ≈60 % and activation at ≈40 % of the gates.

Portfolio-level FWER, maximum over the 5 seeds or draws [95 % CI] (full table with single-policy
FWER, P(qualify) and power: `montecarlo_results.md`):

| Method | Carinthia-S records | KSDD2 records | Synthetic boundary_null | Synthetic crossing_loss |
|---|---|---|---|---|
| Naive select-best | **0.138 [0.132, 0.145]** | 0.054 [0.049, 0.058] | **0.659 [0.650, 0.668]** | **0.359 [0.349, 0.368]** |
| Marginal (Bonf. 1/8) | 0.0022 | 0.0002 | 0.0127 | 0.0059 |
| LTT-Bonferroni | 0.0002 | 0.0000 | 0.0007 | 0.0006 |
| LTT-Holm | 0.0002 | 0.0000 | 0.0007 | 0.0006 |
| SGR | 0.0023 | 0.0000 | 0.0050 | 0.0028 |
| FS, margin (ours) | 0.0065 [0.0050, 0.0083] | 0.0002 | 0.0271 [0.0240, 0.0305] | 0.0086 |
| FS, tuning-selected | 0.0280 | 0.0009 | 0.0332 | 0.0306 |
| Fallback, k = 5 | 0.0056 | 0.0000 | 0.0062 | 0.0025 |
| FS, Hoeffding–Bentkus | 0.0024 | 0.0002 | 0.0091 | 0.0029 |
| FS, ascending | 0.0058 | 0.0000 | 0.0118 | 0.0039 |
| FS-IUT (variant) | 0.0179 | 0.0006 | 0.0543 [0.0499, 0.0589] | 0.0223 |

crossing_both: naive max 0.038; every valid method ≤ 0.0013.

Findings:

1. **Naive selection is invalid on the real records**, not only in theory: realised FWER
   0.091–0.138 across the five Carinthia-S populations (mean 0.116), above δ beyond MC error in
   3 of 5 seeds (lower CI bounds > 0.10 for seeds 201, 207, 211); seed 203 (0.1007) is at δ and
   seed 223 (0.091) below it.
   The excess comes from **policy** selection: at single-policy level naive stayed below δ on
   the records (max 0.046) and exceeded it only in the boundary stress (0.133).
2. **No valid procedure exceeded δ** in any population; the largest valid value is 0.0748
   (FS-IUT, single policy, Carinthia-S seed 223, CI [0.070, 0.080]). Nothing needed
   investigation for excess error. The valid procedures are conservative here because only one
   endpoint is near its target at the first violating gate (the δ/2 endpoint split then spends
   half the budget on an endpoint that cannot err), because the exact binomial is discrete, and
   because the portfolio Bonferroni split assumes the worst case across policies.
3. **Power under resampling (Carinthia-S, mean of 5 seeds):** mean deployed population acceptance
   (0 if none) and P(qualify): FS 0.398 / 0.962; fallback 0.397 / 0.978; SGR 0.388 / 0.979;
   FS-HB 0.366 / 0.925; tuning-selected FS 0.346 / 0.781; LTT-Holm 0.318 / 0.866;
   LTT-Bonferroni 0.317 / 0.866; marginal (Bonf. 1/8) 0.267 / 0.645; FS ascending 0.240 / 0.618;
   FS-IUT 0.423 / 0.979; naive (invalid) 0.508 / 1.000. FS is not uniformly best by seed: on
   seed 203 SGR qualifies 0.994 of draws and FS 0.838.
4. **The realised hold-out was less favourable than the pooled population** for all methods:
   LTT-Bonferroni qualifies 0.79–1.00 of resampled hold-outs per seed but 1/5 on the actual
   hold-out. Because the five seeds share one hold-out image set, the record counts are close to a
   single draw; MC power figures are the more stable comparison.
5. **CRC gate vs the conditional target:** the CRC operating point violated the conditional
   target in 94–100 % of draws (mean over policies and seeds: Carinthia-S 0.969, KSDD2 0.943,
   synthetic 0.991–1.000). This is expected: CRC does not claim to control that quantity.

## 7. Where fixed sequence wins and loses; what the novelty claim can be

**Wins (descriptive, retrospective).** Against grid-wide Bonferroni and Holm, FS spends the
whole policy budget on each tested gate instead of dividing it over 116–174 nested gates;
this is the largest and most robust gap (records 5/5 vs 1/5; MC acceptance 0.398 vs 0.317).
Holm gains nothing over Bonferroni here because few gates have very small p-values. Against
the legacy marginal protocol, FS qualifies more single policies (4,4,4,4,4,1,3,3 vs
2,2,3,2,1,0,0,1) and, unlike the marginal protocol, covers gate and policy selection. The exact
binomial bound is tighter than Hoeffding–Bentkus (FS-HB loses 4 of 27 single-policy
qualifications and 0.03 MC acceptance).

**Ties or losses.** SGR, a 2017 procedure, is within 0.01 of FS in MC acceptance and higher in
P(qualify); its record deficit (4/5) is one seed (211). The fallback sequence, which keeps
testing after an early failure, matches FS on the records (5/5, 0.447 vs 0.451) and in MC
(0.397), with lower dependence on the first gate. FS with ascending order collapses (0/5; MC
0.240): FS power rests entirely on an ordering that, on these records, was chosen after
seeing a failed run. A further finding: the δ/2 endpoint split in Proposition 1 is not
needed. The intersection–union combination is valid and gave the highest valid power (MC
0.423; KSDD2 three single-policy qualifications). It should not be adopted post hoc for the
existing records; it can be prespecified for the WP6 confirmatory study.

**What the claim can be.** (i) A formulation: the acceptance rule of an inspection system with an
optional enhancement stage as a decision rule with two consumer's-risk-type probabilities
conditioned on different accepted populations, with gate and policy selection inside the risk
statement. (ii) Evidence that this matters: uncorrected selection exceeds δ on real inspection
records (0.14 vs 0.10), and the marginal protocol both lacks selection coverage and has less
power. (iii) A comparison showing that ordered testing over nested gates (FS, fallback, SGR) is
the right family for this problem and that exact bounds are preferable to concentration bounds.
**What it cannot be.** New testing theory; a power advantage of FS over SGR or fallback
procedures; a prospective validation of the margin ordering on these records.

## 8. Draft manuscript paragraph (for the comparison subsection)

> On the same records, score arrays, tuning/hold-out split and candidate gates, we compared the
> procedure with existing risk-control methods (Table [X]); because the gate ordering was
> fixed after an earlier ordering had failed on the same hold-out, this comparison is
> retrospective. Selecting the best of separately qualified gates and policies without
> correction exceeded the nominal level when hold-outs were resampled from the pooled
> Carinthia-S records (realised familywise error up to 0.14 for δ = 0.10), whereas every valid
> procedure stayed below 0.08 in all resampled and synthetic stress populations. Among valid
> procedures, ordered testing over the nested gates was markedly more powerful than Bonferroni
> or Holm correction over the 116–118 candidate gates (Carinthia-S portfolio qualified in 5/5
> versus 1/5 repetitions; mean test automatic-acceptance fraction 0.45 versus 0.09), while
> SGR [Geifman2017] (4/5, 0.33) and a fallback sequence [Wiens2003] (5/5, 0.45) were close to
> the fixed sequence, and no valid procedure qualified a KolektorSDD2 policy. The contribution is
> therefore the two-endpoint conditional formulation with selection inside the risk statement,
> not the testing procedure, for which fixed-sequence, SGR-type and fallback constructions are
> interchangeable to within the resolution of these data.

## 9. References verified in-session

| Key | Entry | How verified |
|---|---|---|
| angelopoulos2021ltt | A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei, "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control," arXiv:2110.01052, v1 3 Oct 2021 | arXiv abstract page; HB p-value (Proposition 1, Eq. 1) read from the PDF |
| geifman2017sgr | Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural Networks," arXiv:1705.08500, v1 23 May 2017 | arXiv abstract page; Algorithm 1 and Theorem 3.2 read from the PDF. Venue (NeurIPS 2017) **not verified** in-session |
| bates2021rcps | S. Bates, A. Angelopoulos, L. Lei, J. Malik, M. I. Jordan, "Distribution-Free, Risk-Controlling Prediction Sets," arXiv:2101.02703, v1 7 Jan 2021 | arXiv abstract page. Journal version **not verified** |
| angelopoulos2022crc | A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, "Conformal Risk Control," arXiv:2208.02814, v1 4 Aug 2022 | arXiv abstract page. Journal version **not verified** |
| wiens2003fallback | B. L. Wiens, "A fixed sequence Bonferroni procedure for testing multiple endpoints," Pharmaceutical Statistics 2(3):211–215, 2003, doi:10.1002/pst.64 | Crossref API |
| bretz2009graphical | F. Bretz, W. Maurer, W. Brannath, M. Posch, "A graphical approach to sequentially rejective multiple test procedures," Statistics in Medicine 28(4):586–604, 2009, doi:10.1002/sim.3495 | Crossref API |
| berger1982iut | R. L. Berger, "Multiparameter hypothesis testing and acceptance sampling," Technometrics 24(4):295–300, 1982, doi:10.1080/00401706.1982.10487790 | Crossref API (only needed if the IUT remark is used) |
| holm1979 | S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of Statistics 6(2):65–70, 1979 (JSTOR 4615733) | secondary sources via web search; no DOI record found on Crossref |

## 10. Caveats

- Retrospective: the margin ordering and the KSDD2 pipeline were adapted with hold-out outcomes
  visible (WP1). The fallback k = 5 and SGR rules were fixed by the WP3 brief, not prespecified
  before the records existed.
- The MC "population" is the pooled hold-out + test records. It tests validity for a known
  finite population, not for the true production distribution. Image-level i.i.d. sampling is
  assumed, as in Proposition 1.
- Seeds share images; record counts are not independent replications.
- The SGR adaptation never tests the largest gate (faithful to Algorithm 1) and adds the
  two-endpoint split and portfolio Bonferroni, which are not in the original paper.
- The marginal "portfolio" row is a WP3 construction (Bonferroni 1/8 over tuned gates); the
  published marginal protocol has no policy selection.
