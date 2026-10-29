# Phase 3 — learned triage signal

Generated 2026-10-29 by scripts/phase3_learned_triage_report.py (models at commit 6fc055d). TAGCN K=3 (3 seeds), w = 20, LKH per window as in the triage experiment; sklearn on the CPU, network on the T4.

## Summary

1. **Does the learned signal beat the disagreement?** Yes, by a small margin that holds on every test set.
   - **Signal (AP).** The AP is +0.04–0.05 higher on every test set:
     - tsp1000: 0.517 against 0.469;
     - tsp10000: 0.454 against 0.403;
     - Fu-10k: 0.454 against 0.414.
   - **Policy, top 20 %.** The learned signal is better than the disagreement on tsp1000 (−0.059 pp, 95 % of the
     instances, p = 6e-6) and on Fu-10k (−0.047 pp, 16 / 16 instances, p = 3e-5).
   - **Policy, top 50 %.** The differences are smaller:
     - −0.022 pp on Fu-10k (p = 6e-5);
     - not significant on tsp1000;
     - 3 / 3 instances better on tsp10000, which has too few instances to test.
   - **Model.** The logistic regression was chosen for all 3 seeds. Gradient boosting does not beat it: the signal
     is close to linear in the features.
2. **How much of the oracle does it recover?** The share of the oracle's gain at the same share of windows sent to
   LKH:
   - at top 20 %: 68–70 % for the learned signal, against 60–64 % for the disagreement (and 46–54 % for random);
   - at top 50 %: 89–90 %, against 85–88 % for the disagreement and 85–89 % for random.
3. **Does the network's cost now pay off at equal time against random at 50 %?** No.
   - **At 10k.** Random at 50 % takes 13.6–13.9 s and reaches 3.787 % (tsp10000) and 3.939 % (Fu-10k).
     - The learned signal at top 20 % is a little faster (12.3–12.5 s) but 0.13–0.16 pp worse.
     - The learned signal at top 50 % gives the same gap (−0.001 pp on tsp10000, −0.019 pp on Fu-10k), but takes
       19.2–19.4 s, 40 % more.
   - **At 1k** the picture is the same: random at 50 % gives 3.888 % in 1.37 s; the learned signal gives 4.031 % in
     1.16 s at top 20 % and 3.838 % in 1.70 s at top 50 %.
   - **Why.** Scoring the windows costs about 4 s per 10k instance: building the window graphs, the network and the
     features. That is more than the LKH time the better ranking saves. The features themselves add only
     0.04 s / 0.4 s (1k / 10k) on top of the disagreement.

## What the features say

- The edge probabilities carry most of the signal. Other features that matter:
  - the longest edge relative to the nearest-neighbour spacing;
  - the number of segment edges outside the 5 nearest neighbours;
  - local density.
- Some features add nothing: the simple rule (length / NN distances) once the others are in, and the position in
  the tour.
- Adding log10 n changes nothing. Adding the beam's logp gains at most +0.003 AP, so the beam is not needed for
  triage.
- The learned signal is also steadier across TAGCN seeds: the std of the gap is 0.006–0.034, against 0.015–0.041
  for the disagreement.

## Caveats

- **Training windows.** The model is trained on first-pass windows of the greedy + 2-opt tour (the window logs). The
  policy applies it in up to 3 passes on a tour that is already being improved. It still ranks better than the
  disagreement there.
- **tsp10000_test has 3 instances.** It has no paired test; the 10k conclusion rests on Fu-10k (16 instances).
- **Fu-1k is no longer a test set.** Half of it is training data and the other half is validation.
- **Regularisation.** The chosen C = 0.01 is the smallest value in the grid. A stronger regularisation might do as
  well or slightly better; given the size of the effect, it would not change the conclusions.
- **Times.** Times are wall-clock with 8 processes on the shared 8-core CPU (as in E2), so absolute times include
  contention. Random, disagreement, learned and oracle were run together on each set so they are comparable. The
  gaps of random, disagreement and oracle reproduce those of E2 exactly.

## Data

| Set | Role | n | Instances | Windows (per seed) | LKH improves (%) | Mean window gain (%) |
|---|---|---|---|---|---|---|
| tsp200_test | train + validation (halves by instance) | 200 | 100 | 2000 | 29.9 | 0.67 |
| tsp1000_test_concorde | train + validation (halves by instance) | 1000 | 128 | 12800 | 23.7 | 0.47 |
| tsp1000_test | test | 1000 | 20 | 2000 | 25.7 | 0.50 |
| tsp10000_test | test | 10000 | 3 | 3000 | 21.3 | 0.42 |
| tsp10000_test_concorde | test (Fu-10k) | 10000 | 16 | 16000 | 21.5 | 0.41 |

Label: LKH improves the window (path shorter than the current one). Features: p mean / min / max on the segment's edges, disagreement, segment length / NN distances, longest edge / mean NN distance, density within 2 mean NN distances, position in the tour, edges outside the 5 nearest neighbours; no feature uses the LKH solution; n and the beam logp only as ablations. Training: half of the instances of tsp200 and of Fu-1k (same halves for the 3 seeds), each set with the same total weight; validation: the other halves; tests never used to fit or choose.

## Model selection (validation)

| Model | Hyper-parameters chosen (per seed) | Validation AP, tsp200 | Validation AP, Fu-1k | Chosen (seeds) |
|---|---|---|---|---|
| logistic regression | {"C": 0.01}; {"C": 0.01}; {"C": 0.01} | 0.550 ± 0.030 | 0.488 ± 0.029 | 3 / 3 |
| gradient boosting (classifier) | {"max_depth": 3, "learning_rate": 0.05}; {"max_depth": 3, "learning_rate": 0.05}; {"max_depth": 3, "learning_rate": 0.05} | 0.532 ± 0.036 | 0.481 ± 0.027 | 0 / 3 |
| gradient boosting (regressor on the gain) | {"max_depth": 3, "learning_rate": 0.05}; {"max_depth": 3, "learning_rate": 0.05}; {"max_depth": 3, "learning_rate": 0.05} | 0.530 ± 0.041 | 0.477 ± 0.026 | 0 / 3 |

## Signal quality: average precision

| Set | Windows | Random (= positive rate) | Simple rule | Disagreement | Logistic | GB classifier | GB regressor | **Chosen** | Chosen + log10 n | Chosen + beam logp | Chosen − disagreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tsp200_test (validation half) | 1000 | 0.286 ± 0.000 | 0.382 ± 0.000 | 0.511 ± 0.046 | 0.550 ± 0.030 | 0.532 ± 0.036 | 0.530 ± 0.041 | 0.550 ± 0.030 | 0.550 ± 0.030 | 0.548 ± 0.031 | +0.038 |
| tsp1000_test_concorde (validation half (Fu-1k)) | 6400 | 0.238 ± 0.000 | 0.301 ± 0.000 | 0.451 ± 0.033 | 0.488 ± 0.029 | 0.481 ± 0.027 | 0.477 ± 0.026 | 0.488 ± 0.029 | 0.488 ± 0.029 | 0.487 ± 0.028 | +0.038 |
| tsp1000_test (test) | 2000 | 0.257 ± 0.000 | 0.348 ± 0.000 | 0.469 ± 0.025 | 0.517 ± 0.023 | 0.504 ± 0.024 | 0.494 ± 0.024 | 0.517 ± 0.023 | 0.517 ± 0.023 | 0.520 ± 0.022 | +0.049 |
| tsp10000_test (test) | 3000 | 0.213 ± 0.000 | 0.282 ± 0.000 | 0.403 ± 0.028 | 0.454 ± 0.021 | 0.445 ± 0.021 | 0.436 ± 0.023 | 0.454 ± 0.021 | 0.454 ± 0.021 | 0.449 ± 0.024 | +0.051 |
| tsp10000_test_concorde (test (Fu-10k)) | 16000 | 0.215 ± 0.000 | 0.288 ± 0.000 | 0.414 ± 0.024 | 0.454 ± 0.021 | 0.447 ± 0.022 | 0.440 ± 0.021 | 0.454 ± 0.021 | 0.455 ± 0.021 | 0.454 ± 0.021 | +0.041 |

Mean ± std over the 3 TAGCN seeds (random, simple and the label do not depend on the seed). Validation rows: the validation halves; test rows: all windows of the set.

## Policy: top x % of the windows to LKH (E2 protocol)

| Set | x | Policy | Gap (%) | Gain (pp) | Share of oracle gain | Time (s) | of which network (s) | of which score (s) | Windows to LKH |
|---|---|---|---|---|---|---|---|---|---|
| tsp1000_test | 20 % | Random (5 seeds) | 4.260 ± 0.038 | 0.444 | 46 % | 0.75 | 0.00 | 0.00 | 20 % |
| tsp1000_test | 20 % | Disagreement (3 seeds) | 4.091 ± 0.041 | 0.613 | 63 % | 1.08 | 0.06 | 0.36 | 20 % |
| tsp1000_test | 20 % | Learned (3 seeds) | 4.031 ± 0.028 | 0.673 | 69 % | 1.16 | 0.06 | 0.40 | 20 % |
| tsp1000_test | 20 % | Oracle: real LKH gain | 3.734 | 0.970 | 100 % | 2.32 * | 0.00 | 0.00 | 20 % |
| tsp1000_test | 50 % | Random (5 seeds) | 3.888 ± 0.016 | 0.816 | 85 % | 1.37 | 0.00 | 0.00 | 50 % |
| tsp1000_test | 50 % | Disagreement (3 seeds) | 3.859 ± 0.017 | 0.845 | 88 % | 1.63 | 0.04 | 0.33 | 50 % |
| tsp1000_test | 50 % | Learned (3 seeds) | 3.838 ± 0.007 | 0.866 | 90 % | 1.70 | 0.04 | 0.36 | 50 % |
| tsp1000_test | 50 % | Oracle: real LKH gain | 3.742 | 0.962 | 100 % | 2.11 * | 0.00 | 0.00 | 50 % |
| tsp10000_test | 20 % | Random (5 seeds) | 4.050 ± 0.013 | 0.407 | 54 % | 7.35 | 0.00 | 0.00 | 20 % |
| tsp10000_test | 20 % | Disagreement (3 seeds) | 4.008 ± 0.030 | 0.449 | 60 % | 11.87 | 0.37 | 3.77 | 20 % |
| tsp10000_test | 20 % | Learned (3 seeds) | 3.945 ± 0.034 | 0.512 | 68 % | 12.45 | 0.32 | 4.17 | 20 % |
| tsp10000_test | 20 % | Oracle: real LKH gain | 3.702 | 0.755 | 100 % | 24.78 * | 0.00 | 0.00 | 20 % |
| tsp10000_test | 50 % | Random (5 seeds) | 3.787 ± 0.018 | 0.670 | 89 % | 13.61 | 0.00 | 0.00 | 50 % |
| tsp10000_test | 50 % | Disagreement (3 seeds) | 3.814 ± 0.023 | 0.643 | 85 % | 18.73 | 0.23 | 3.63 | 50 % |
| tsp10000_test | 50 % | Learned (3 seeds) | 3.786 ± 0.006 | 0.671 | 89 % | 19.19 | 0.23 | 4.00 | 50 % |
| tsp10000_test | 50 % | Oracle: real LKH gain | 3.702 | 0.755 | 100 % | 24.17 * | 0.00 | 0.00 | 50 % |
| tsp10000_test_concorde | 20 % | Random (5 seeds) | 4.212 ± 0.007 | 0.397 | 52 % | 7.39 | 0.00 | 0.00 | 20 % |
| tsp10000_test_concorde | 20 % | Disagreement (3 seeds) | 4.121 ± 0.023 | 0.488 | 64 % | 11.95 | 0.27 | 3.72 | 20 % |
| tsp10000_test_concorde | 20 % | Learned (3 seeds) | 4.074 ± 0.013 | 0.535 | 70 % | 12.32 | 0.28 | 4.09 | 20 % |
| tsp10000_test_concorde | 20 % | Oracle: real LKH gain | 3.844 | 0.765 | 100 % | 25.81 * | 0.00 | 0.00 | 20 % |
| tsp10000_test_concorde | 50 % | Random (5 seeds) | 3.939 ± 0.007 | 0.670 | 88 % | 13.93 | 0.00 | 0.00 | 50 % |
| tsp10000_test_concorde | 50 % | Disagreement (3 seeds) | 3.942 ± 0.015 | 0.667 | 87 % | 18.98 | 0.23 | 3.70 | 50 % |
| tsp10000_test_concorde | 50 % | Learned (3 seeds) | 3.920 ± 0.006 | 0.689 | 90 % | 19.39 | 0.27 | 4.10 | 50 % |
| tsp10000_test_concorde | 50 % | Oracle: real LKH gain | 3.845 | 0.764 | 100 % | 25.80 * | 0.00 | 0.00 | 50 % |

w = 20, offsets 0 and w/2, ≤ 3 passes, from the greedy + 2-opt tour; in every half-pass exactly the top x % of the windows by the score go to LKH (E2 of the benchmark). Gap: mean over instances of the per-instance mean over runs (± std of the run means). Time: greedy + refinement, mean per instance. *score* = time to compute the scores outside the network (for the learned signal: instance context, features and predict). * the oracle runs LKH on every window to know its gain: its time is not that of a usable policy.

### Paired tests

| Set | x | Learned vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |
|---|---|---|---|---|---|---|
| tsp1000_test (20) | 20 % | Disagreement (3 seeds) | 95 / 0 / 5 | -0.059 | 5.7e-06 | **better** |
| tsp1000_test (20) | 20 % | Random (5 seeds) | 95 / 0 / 5 | -0.229 | 1.3e-05 | **better** |
| tsp1000_test (20) | 50 % | Disagreement (3 seeds) | 60 / 5 / 35 | -0.022 | 1.3e-01 | n.s. |
| tsp1000_test (20) | 50 % | Random (5 seeds) | 60 / 0 / 40 | -0.050 | 1.9e-01 | n.s. |
| tsp10000_test (3) | 20 % | Disagreement (3 seeds) | 100 / 0 / 0 | -0.062 | — | no test (n = 3) |
| tsp10000_test (3) | 20 % | Random (5 seeds) | 100 / 0 / 0 | -0.105 | — | no test (n = 3) |
| tsp10000_test (3) | 50 % | Disagreement (3 seeds) | 100 / 0 / 0 | -0.028 | — | no test (n = 3) |
| tsp10000_test (3) | 50 % | Random (5 seeds) | 67 / 0 / 33 | -0.001 | — | no test (n = 3) |
| tsp10000_test_concorde (16) | 20 % | Disagreement (3 seeds) | 100 / 0 / 0 | -0.047 | 3.1e-05 | **better** |
| tsp10000_test_concorde (16) | 20 % | Random (5 seeds) | 100 / 0 / 0 | -0.138 | 3.1e-05 | **better** |
| tsp10000_test_concorde (16) | 50 % | Disagreement (3 seeds) | 94 / 0 / 6 | -0.022 | 6.1e-05 | **better** |
| tsp10000_test_concorde (16) | 50 % | Random (5 seeds) | 81 / 0 / 19 | -0.018 | 6.3e-03 | **better** |

Paired two-sided Wilcoxon on the per-instance gaps (learned and disagreement averaged over their 3 TAGCN seeds, random over its 5 seeds); negative difference = learned shorter. Bonferroni within each set: 0.05 / 4 = 0.0125. tsp10000_test has 3 instances: no test.

### Same time

| Set | Policy (sorted by time) | Time (s) | Gap (%) |
|---|---|---|---|
| tsp1000_test | Random top 20 % | 0.75 | 4.260 |
| tsp1000_test | Disagreement top 20 % | 1.08 | 4.091 |
| tsp1000_test | Learned top 20 % | 1.16 | 4.031 |
| tsp1000_test | Random top 50 % | 1.37 | 3.888 |
| tsp1000_test | Disagreement top 50 % | 1.63 | 3.859 |
| tsp1000_test | Learned top 50 % | 1.70 | 3.838 |
| tsp10000_test | Random top 20 % | 7.35 | 4.050 |
| tsp10000_test | Disagreement top 20 % | 11.87 | 4.008 |
| tsp10000_test | Learned top 20 % | 12.45 | 3.945 |
| tsp10000_test | Random top 50 % | 13.61 | 3.787 |
| tsp10000_test | Disagreement top 50 % | 18.73 | 3.814 |
| tsp10000_test | Learned top 50 % | 19.19 | 3.786 |
| tsp10000_test_concorde | Random top 20 % | 7.39 | 4.212 |
| tsp10000_test_concorde | Disagreement top 20 % | 11.95 | 4.121 |
| tsp10000_test_concorde | Learned top 20 % | 12.32 | 4.074 |
| tsp10000_test_concorde | Random top 50 % | 13.93 | 3.939 |
| tsp10000_test_concorde | Disagreement top 50 % | 18.98 | 3.942 |
| tsp10000_test_concorde | Learned top 50 % | 19.39 | 3.920 |

The same-time question: a policy is worth its cost only if no cheaper policy reaches the same gap.

## Feature importance

| Feature | Logistic coefficient (standardised) | Permutation importance of the chosen model (validation AP drop) |
|---|---|---|
| p_mean | -0.320 ± 0.048 | 0.052 ± 0.014 |
| p_min | -0.080 ± 0.074 | 0.006 ± 0.006 |
| p_max | -0.064 ± 0.077 | 0.003 ± 0.005 |
| disagreement | 0.320 ± 0.048 | 0.052 ± 0.014 |
| seg_len_nn | -0.037 ± 0.039 | 0.000 ± 0.000 |
| max_edge_nn | 0.209 ± 0.006 | 0.032 ± 0.003 |
| density | 0.167 ± 0.010 | 0.008 ± 0.002 |
| position | -0.073 ± 0.006 | 0.000 ± 0.000 |
| out_knn5 | 0.208 ± 0.021 | 0.032 ± 0.008 |

Mean ± std over the 3 TAGCN seeds. Coefficients: sign = direction (positive = more likely that LKH improves); p_mean and disagreement are the same information (disagreement = 1 − p_mean), so their coefficients share it. Permutation importance: drop of the validation AP when the feature is shuffled (5 repeats).

