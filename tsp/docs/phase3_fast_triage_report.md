# Phase 3 — cheap window scoring for the triage

Generated 2026-10-30 by scripts/phase3_fast_triage_report.py. TAGCN K=3 (3 seeds), w = 20, LKH per window; tsp1000 (20 instances) and Fu-10k (16).

## Summary

**At equal time, does the learned signal now beat random? Yes on tsp1000. On Fu-10k only in the lighter-load
diagnostic, not in the 7-process policy run.**
- **tsp1000.** The batched learned signal at top 50 % gives 3.838 % in 1.36 s; random at 50 % gives 3.888 % in
  1.43 s. So it is faster and better, but the difference is not significant (p = 0.19, 20 instances).
- **Fu-10k, 7 processes.** In the 7-process policy run the learned signal at top 50 % is better (3.920 % against
  3.939 %, 81 % of the instances, p = 0.006) but takes 17.4 s against 13.9 s.
  - Scoring is no longer the reason: it takes 0.35 s.
  - The reason is LKH: with the same 1500 windows sent, LKH took 8.8 ms per window in the network processes and
    7.2 ms in the random process running alongside them.
- **Fu-10k, diagnostic.** The diagnostic ran 3 processes together on 4 Fu-10k instances.
  - The extra LKH time falls to +3–6 % per window (7.6–7.8 ms against 7.3 ms). That is about what the window logs
    predict (windows LKH improves take +7 %).
  - The learned signal at top 50 % then beats random at 50 % on both time and gap: 14.6–14.9 s and 3.909 %, against
    15.3 s and 3.926 %. This is only 4 instances, with no test.
  - The torch thread count is not the cause: 1 and 2 threads give the same times.
  - So, at 10k, the answer depends on how loaded the machine is. Under load, the processes that run the network pay
    more for their own LKH calls; I have not isolated why.

## The cheap scoring (version A)

- **Speed.** The windows of a half-pass are built as one graph with numpy and sent to the network in one call; the
  features are vectorised; unchanged windows keep their score.
  - Scoring both half-passes of a 10k tour now takes 0.09 s (disagreement) and 0.14 s (learned), against 1.13 s per
    window.
  - Inside the policy, scoring went from ~4 s to 0.35 s per 10k instance.
- **Equivalence.**
  - On the sample, the Spearman correlation with the per-window scores is 1.0000 and the top-20 % overlap is 100 %;
    the largest difference is 1e-6.
  - Every policy gap is identical, instance by instance, to the per-window run (108 / 108 instance × seed pairs,
    difference 0).
- **The learned signal at equal share of windows.** It still beats the batched disagreement (top 20 %: −0.06 pp on
  tsp1000, −0.05 pp on Fu-10k, both p < 1e-4).

## Version B (one network pass on the whole instance)

- **The global p is saturated.** The network was trained on TSP-20 windows rescaled to [0, 1]²; on the whole instance
  it sees much smaller distances. Mean p is 0.99 at n = 200 and ~0.9 at 10k. No tour edge falls outside the global
  k = 100 graph.
- **Disagreement.** The Spearman correlation of the global disagreement with the current one is 0.02–0.09, below the
  0.5 threshold. Stopped there: no policy runs.
- **Learned signal, retrained on global features.** The Spearman correlation is 0.58–0.59, so it went on, but it is
  clearly worse than A:
  - tsp1000: 4.220 % against 4.031 % at top 20 %, and 3.907 % against 3.838 % at top 50 %;
  - Fu-10k: 4.225 % against 4.074 % at top 20 %, and 3.985 % against 3.920 % at top 50 %.
  - What it keeps comes from the geometric features.
- **Cost.** B is not cheaper than A either: the global graph and the pass take ~0.9 s per 10k instance, against
  0.35 s for A.

## Caveats

- **Load differs between runs.**
  - The B policies on Fu-10k ran one seed at a time, under less load than A and random. The first attempts (3
    together, then 1) hit CUDA out of memory, because another project's Ollama held 12.6–13.2 GiB of the T4. The
    network head is now applied in edge chunks (peak 0.81 GiB).
  - The per-window and oracle rows come from the previous run (8 processes). Their gaps are deterministic; their
    times carry that run's load.
- **The diagnostic is small.** It used 4 instances and one rng / TAGCN seed, so it shows the load effect but is not a
  test.
- **The equal-time answer at 10k is conditional.** A clean answer needs the policies timed under the same, light
  load (e.g. one process at a time). That has not been run.

## Scoring time

| Set | Per window (disagreement + learned) | A disagreement | A learned | B disagreement | B learned | Tour edges outside the global graph |
|---|---|---|---|---|---|---|
| tsp1000_test | 0.123 | 0.014 | 0.018 | 0.064 | 0.066 | 0.00 % |
| tsp10000_test_concorde | 1.132 | 0.090 | 0.137 | 0.941 | 0.905 | 0.00 % |

Seconds per instance to score every window of both half-passes of one pass (network included; one process alone, not under the 8-process load of the policy runs). The per-window column computes the heatmaps once for both scores (the learned features add ~10 %). B includes the global k-NN graph and the network pass.

## Equivalence with the per-window scores

| Set | Version | Spearman vs per-window (mean, min) | Top-20 % overlap (mean, min) | Largest |difference| | Passes to the time table |
|---|---|---|---|---|---|
| tsp1000_test | A disagreement | 1.0000, 1.0000 | 100.0 %, 100.0 % | 2.1e-07 | yes |
| tsp1000_test | A learned | 1.0000, 1.0000 | 100.0 %, 100.0 % | 8.8e-07 | yes |
| tsp1000_test | B disagreement | 0.0904, -0.2217 | 23.3 %, 0.0 % | 2.5e-01 | **no** (Spearman < 0.5) |
| tsp1000_test | B learned | 0.5947, 0.2452 | 58.3 %, 30.0 % | 5.2e-01 | yes |
| tsp10000_test_concorde | A disagreement | 1.0000, 1.0000 | 100.0 %, 100.0 % | 2.4e-07 | yes |
| tsp10000_test_concorde | A learned | 1.0000, 1.0000 | 100.0 %, 100.0 % | 1.2e-06 | yes |
| tsp10000_test_concorde | B disagreement | 0.0153, -0.0647 | 20.1 %, 12.0 % | 2.8e-01 | **no** (Spearman < 0.5) |
| tsp10000_test_concorde | B learned | 0.5830, 0.4400 | 50.8 %, 38.0 % | 5.6e-01 | yes |

Sample: the first 2 instances of each set, 3 TAGCN seeds, both half-passes of the first pass on the greedy + 2-opt tour. B learned is the logistic model retrained on global features (same protocol as the learned triage); it is compared with the current learned score.

### Policy outcome: batched A against per-window

| Set | x | Batched A vs per-window | Instances × seeds with the same gap | Largest gap difference (pp) |
|---|---|---|---|---|
| tsp1000_test | 20 % | Disagreement, batched A | 60 / 60 | 0.0e+00 |
| tsp1000_test | 20 % | Learned, batched A | 60 / 60 | 0.0e+00 |
| tsp1000_test | 50 % | Disagreement, batched A | 60 / 60 | 0.0e+00 |
| tsp1000_test | 50 % | Learned, batched A | 60 / 60 | 0.0e+00 |
| tsp10000_test_concorde | 20 % | Disagreement, batched A | 48 / 48 | 0.0e+00 |
| tsp10000_test_concorde | 20 % | Learned, batched A | 48 / 48 | 0.0e+00 |
| tsp10000_test_concorde | 50 % | Disagreement, batched A | 48 / 48 | 0.0e+00 |
| tsp10000_test_concorde | 50 % | Learned, batched A | 48 / 48 | 0.0e+00 |

## Policies (E2 protocol)

| Set | x | Policy | Gap (%) | Share of oracle gain | Time (s) | of which scoring (s) | of which network (s) |
|---|---|---|---|---|---|---|---|
| tsp1000_test | 20 % | Random (5 seeds) | 4.260 ± 0.038 | 46 % | 0.77 | 0.00 | 0.00 |
| tsp1000_test | 20 % | Disagreement, batched A | 4.091 ± 0.041 | 63 % | 0.78 | 0.05 | 0.04 |
| tsp1000_test | 20 % | Learned, batched A | 4.031 ± 0.028 | 69 % | 0.80 | 0.06 | 0.04 |
| tsp1000_test | 20 % | Learned (retrained), global B | 4.220 ± 0.003 | 50 % | 0.80 | 0.10 | 0.03 |
| tsp1000_test | 20 % | Disagreement, per window (previous run) | 4.091 ± 0.041 | 63 % | 1.08 | 0.36 | 0.06 |
| tsp1000_test | 20 % | Learned, per window (previous run) | 4.031 ± 0.028 | 69 % | 1.16 | 0.40 | 0.06 |
| tsp1000_test | 20 % | Oracle: real LKH gain (previous run) | 3.734 | 100 % | 2.32 * | 0.00 | 0.00 |
| tsp1000_test | 50 % | Random (5 seeds) | 3.888 ± 0.016 | 85 % | 1.43 | 0.00 | 0.00 |
| tsp1000_test | 50 % | Disagreement, batched A | 3.859 ± 0.017 | 88 % | 1.35 | 0.04 | 0.02 |
| tsp1000_test | 50 % | Learned, batched A | 3.838 ± 0.007 | 90 % | 1.36 | 0.05 | 0.02 |
| tsp1000_test | 50 % | Learned (retrained), global B | 3.907 ± 0.011 | 83 % | 1.35 | 0.08 | 0.01 |
| tsp1000_test | 50 % | Disagreement, per window (previous run) | 3.859 ± 0.017 | 88 % | 1.63 | 0.33 | 0.04 |
| tsp1000_test | 50 % | Learned, per window (previous run) | 3.838 ± 0.007 | 90 % | 1.70 | 0.36 | 0.04 |
| tsp1000_test | 50 % | Oracle: real LKH gain (previous run) | 3.742 | 100 % | 2.11 * | 0.00 | 0.00 |
| tsp10000_test_concorde | 20 % | Random (5 seeds) | 4.212 ± 0.007 | 52 % | 7.44 | 0.00 | 0.00 |
| tsp10000_test_concorde | 20 % | Disagreement, batched A | 4.121 ± 0.023 | 64 % | 9.35 | 0.29 | 0.18 |
| tsp10000_test_concorde | 20 % | Learned, batched A | 4.074 ± 0.013 | 70 % | 9.60 | 0.35 | 0.18 |
| tsp10000_test_concorde | 20 % | Learned (retrained), global B | 4.225 ± 0.009 | 50 % | 7.98 | 0.92 | 0.12 |
| tsp10000_test_concorde | 20 % | Disagreement, per window (previous run) | 4.121 ± 0.023 | 64 % | 11.95 | 3.72 | 0.27 |
| tsp10000_test_concorde | 20 % | Learned, per window (previous run) | 4.074 ± 0.013 | 70 % | 12.32 | 4.09 | 0.28 |
| tsp10000_test_concorde | 20 % | Oracle: real LKH gain (previous run) | 3.844 | 100 % | 25.81 * | 0.00 | 0.00 |
| tsp10000_test_concorde | 50 % | Random (5 seeds) | 3.939 ± 0.007 | 88 % | 13.93 | 0.00 | 0.00 |
| tsp10000_test_concorde | 50 % | Disagreement, batched A | 3.942 ± 0.015 | 87 % | 17.21 | 0.28 | 0.17 |
| tsp10000_test_concorde | 50 % | Learned, batched A | 3.920 ± 0.006 | 90 % | 17.37 | 0.35 | 0.18 |
| tsp10000_test_concorde | 50 % | Learned (retrained), global B | 3.985 ± 0.010 | 82 % | 13.97 | 0.92 | 0.12 |
| tsp10000_test_concorde | 50 % | Disagreement, per window (previous run) | 3.942 ± 0.015 | 87 % | 18.98 | 3.70 | 0.23 |
| tsp10000_test_concorde | 50 % | Learned, per window (previous run) | 3.920 ± 0.006 | 90 % | 19.39 | 4.10 | 0.27 |
| tsp10000_test_concorde | 50 % | Oracle: real LKH gain (previous run) | 3.845 | 100 % | 25.80 * | 0.00 | 0.00 |

w = 20, ≤ 3 passes, top x % of the windows of each half-pass to LKH (E2 protocol). Random and the batched scores ran together (7 processes); the per-window and oracle rows are from the learned-triage run (8 processes, same machine, same protocol): their gaps are deterministic, their times carry that run's load. * the oracle runs LKH on every window: its time is not that of a usable policy.

### Same time

| Set | Policy (sorted by time) | Time (s) | Gap (%) |
|---|---|---|---|
| tsp1000_test | Random (5 seeds), top 20 % | 0.77 | 4.260 |
| tsp1000_test | Disagreement, batched A, top 20 % | 0.78 | 4.091 |
| tsp1000_test | Learned (retrained), global B, top 20 % | 0.80 | 4.220 |
| tsp1000_test | Learned, batched A, top 20 % | 0.80 | 4.031 |
| tsp1000_test | Disagreement, batched A, top 50 % | 1.35 | 3.859 |
| tsp1000_test | Learned (retrained), global B, top 50 % | 1.35 | 3.907 |
| tsp1000_test | Learned, batched A, top 50 % | 1.36 | 3.838 |
| tsp1000_test | Random (5 seeds), top 50 % | 1.43 | 3.888 |
| tsp10000_test_concorde | Random (5 seeds), top 20 % | 7.44 | 4.212 |
| tsp10000_test_concorde | Learned (retrained), global B, top 20 % | 7.98 | 4.225 |
| tsp10000_test_concorde | Disagreement, batched A, top 20 % | 9.35 | 4.121 |
| tsp10000_test_concorde | Learned, batched A, top 20 % | 9.60 | 4.074 |
| tsp10000_test_concorde | Random (5 seeds), top 50 % | 13.93 | 3.939 |
| tsp10000_test_concorde | Learned (retrained), global B, top 50 % | 13.97 | 3.985 |
| tsp10000_test_concorde | Disagreement, batched A, top 50 % | 17.21 | 3.942 |
| tsp10000_test_concorde | Learned, batched A, top 50 % | 17.37 | 3.920 |

### Paired tests

| Set | Policy | vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) | Verdict |
|---|---|---|---|---|---|---|
| tsp1000_test (20) | Learned, batched A top 20 % | Random (5 seeds) top 50 % | 5 / 0 / 95 | +0.144 | 5.7e-06 | **worse** |
| tsp1000_test (20) | Learned, batched A top 50 % | Random (5 seeds) top 50 % | 60 / 0 / 40 | -0.050 | 1.9e-01 | n.s. |
| tsp1000_test (20) | Learned, batched A top 20 % | Random (5 seeds) top 20 % | 95 / 0 / 5 | -0.229 | 1.3e-05 | **better** |
| tsp1000_test (20) | Learned, batched A top 20 % | Disagreement, batched A top 20 % | 95 / 0 / 5 | -0.059 | 5.7e-06 | **better** |
| tsp1000_test (20) | Learned, batched A top 50 % | Disagreement, batched A top 50 % | 60 / 5 / 35 | -0.022 | 1.3e-01 | n.s. |
| tsp10000_test_concorde (16) | Learned, batched A top 20 % | Random (5 seeds) top 50 % | 0 / 0 / 100 | +0.135 | 3.1e-05 | **worse** |
| tsp10000_test_concorde (16) | Learned, batched A top 50 % | Random (5 seeds) top 50 % | 81 / 0 / 19 | -0.018 | 6.3e-03 | **better** |
| tsp10000_test_concorde (16) | Learned, batched A top 20 % | Random (5 seeds) top 20 % | 100 / 0 / 0 | -0.138 | 3.1e-05 | **better** |
| tsp10000_test_concorde (16) | Learned, batched A top 20 % | Disagreement, batched A top 20 % | 100 / 0 / 0 | -0.047 | 3.1e-05 | **better** |
| tsp10000_test_concorde (16) | Learned, batched A top 50 % | Disagreement, batched A top 50 % | 94 / 0 / 6 | -0.022 | 6.1e-05 | **better** |

Paired two-sided Wilcoxon on the per-instance gaps; Bonferroni 0.05 / 5 within each set.

## Diagnostic: LKH time of the network policies

| x | Policy | LKH time per window sent (ms) | Total time (s) | Gap (%) |
|---|---|---|---|---|
| 20 % | Random (rng seed 0) | 7.33 | 8.65 | 4.207 |
| 20 % | Learned, batched A, torch threads 2 | 7.77 | 7.87 | 4.057 |
| 20 % | Learned, batched A, torch threads 1 | 7.83 | 7.97 | 4.057 |
| 50 % | Random (rng seed 0) | 7.33 | 15.29 | 3.926 |
| 50 % | Learned, batched A, torch threads 2 | 7.76 | 14.88 | 3.909 |
| 50 % | Learned, batched A, torch threads 1 | 7.56 | 14.57 | 3.909 |

