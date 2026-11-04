# Ablations A1–A5 of the assumptions audit (2026-11-03)

The audit (`docs/assumptions.md`) listed assumptions behind the phase 3, phase 5 and warehouse conclusions that had not been
tested. A1–A5 test five of them. Everything ran on the CPU (`CUDA_VISIBLE_DEVICES=""`, `nice -n 10`), the network
included, so the network's own time is larger here than in the phase 3 reports, which ran it on the T4.

- **Code.** `tsp/src/tsp_gnn/triage.py` (memo modes), `tsp/src/tsp_gnn/softdist.py`, `tsp/scripts/ablation_*.py`,
  `tsp/scripts/run_ablation_a12.sh` and `run_ablation_a345.sh`, `warehouse/scripts/ablation_a3_warehouse.py` and
  `ablation_a5.py`. Commits 2b6946b, 6a751fd and the one with this report.
- **Raw tables.** `tsp/results/ablations/{a12,a3_tsp,a3_warehouse,a4,a5}.md` (outside git). Figure:
  `tsp/docs/figs/ablation_a2_pareto.png`.
- **Time.** A1/A2 09:40–20:50, alone on the machine as far as our processes go (75 measurements, 0 failures). A3–A5
  20:51–22:11, two lanes (0 failures). The A4 rerun (below) 22:14–23:02. About 13.4 h of wall clock against the 11 h
  planned: 2.5 h went to waiting for the load to drop before 5 measurements, and 0.8 h to the A4 rerun.
- **Deviations from the plan.**
  - **A4, τ.** The first grid (0.01–0.5) chose τ = 0.5, its edge. The grid was extended to 0.75, 1, 2, 5 and ∞ and
    all of A4 was rerun. τ = 0.75 is an interior optimum. The first run is kept in `tsp/results/ablations/a4_grid1`.
  - **Load.** The 5 measurements at w = 50 without memo or with the cache on LKH on every window ran with a load of
    6.8–9.0 after 30 min of waiting. They are marked "with load", and the w = 50 "none" times are not used for any
    conclusion. The other 70 measurements started with a load ≤ 4 (median 3.0).
  - **A4 timings.** The A4 runs shared the machine with A5, so A4 compares gaps at the same share of windows, not times.

## A1: window memo in the ranked triage

Ranked triage, w = 20 (and 50), ≤ 3 passes, seed 0. Each cell gives gap % / time s per instance (greedy + 2-opt tour
+ refinement). In brackets is the gap minus random's gap at the same time, interpolated per instance along random's
curve with the same memo; * means Wilcoxon p < 0.05. The memo modes are: none, as in phase 3; (a) cache, where a
window already solved with the same cities reuses its order; (b) exclude, where a window already solved and unchanged
is not selected, so the x % goes to new windows. The cache never changes the gap, which checks it is correct.

| set | policy | x % | none | (a) cache | (b) exclude |
|---|---|---:|---|---|---|
| Fu-10k, w = 20 | LKH on every window | 100 | 3.845 / 23.09 | 3.845 / 10.89 | = cache |
| Fu-10k, w = 20 | random | 10 | 4.367 / 5.06 | 4.367 / 4.61 | 4.349 / 4.78 |
| Fu-10k, w = 20 | random | 20 | 4.205 / 6.98 | 4.205 / 6.20 | 4.155 / 7.03 |
| Fu-10k, w = 20 | random | 50 | 3.931 / 12.99 | 3.931 / 9.72 | 3.844 / 11.36 |
| Fu-10k, w = 20 | disagreement | 10 | 4.222 / 5.93 (−0.069*) | 4.222 / 4.51 (−0.140*) | 4.089 / 5.59 (−0.189*) |
| Fu-10k, w = 20 | disagreement | 20 | 4.091 / 8.06 (−0.066*) | 4.091 / 5.51 (−0.182*) | 3.947 / 7.75 (−0.151*) |
| Fu-10k, w = 20 | disagreement | 50 | 3.923 / 14.09 (+0.001) | 3.923 / 8.13 (−0.129*) | 3.844 / 11.91 (−0.000) |
| Fu-10k, w = 20 | learned | 10 | 4.179 / 5.76 (−0.125*) | 4.179 / 4.92 (−0.145*) | 4.056 / 5.88 (−0.198*) |
| Fu-10k, w = 20 | learned | 20 | 4.055 / 8.03 (−0.103*) | 4.055 / 6.04 (−0.166*) | 3.931 / 8.12 (−0.137*) |
| Fu-10k, w = 20 | learned | 50 | 3.913 / 13.90 (−0.011*) | 3.913 / 8.73 (−0.092*) | 3.845 / 12.30 (+0.000) |
| Fu-1k, w = 20 (64 held-out) | LKH on every window | 100 | 3.994 / 1.98 | 3.994 / 1.14 | = cache |
| Fu-1k, w = 20 (64 held-out) | random | 20 | 4.438 / 0.65 | 4.438 / 0.60 | 4.381 / 0.67 |
| Fu-1k, w = 20 (64 held-out) | random | 50 | 4.112 / 1.25 | 4.112 / 0.94 | 3.992 / 1.15 |
| Fu-1k, w = 20 (64 held-out) | learned | 10 | 4.416 / 0.56 (−0.107*) | 4.416 / 0.49 (−0.149*) | 4.270 / 0.58 (−0.215*) |
| Fu-1k, w = 20 (64 held-out) | learned | 20 | 4.259 / 0.77 (−0.104*) | 4.259 / 0.60 (−0.166*) | 4.101 / 0.80 (−0.167*) |
| Fu-1k, w = 20 (64 held-out) | learned | 50 | 4.082 / 1.28 (−0.054*) | 4.082 / 0.88 (−0.087*) | 3.995 / 1.23 (−0.001) |
| Fu-10k, w = 50 | LKH on every window | 100 | 3.061 / 76.41 † | 3.061 / 42.53 † | = cache |
| Fu-10k, w = 50 | random | 20 | 3.734 / 17.57 † | 3.734 / 14.83 | 3.694 / 15.69 |
| Fu-10k, w = 50 | disagreement | 20 | 3.786 / 22.19 † | 3.786 / 13.90 (−0.003) | 3.544 / 18.63 (−0.058*) |
| Fu-10k, w = 50 | learned | 10 | 3.861 / 14.15 † | 3.861 / 10.26 (−0.136*) | 3.767 / 12.03 (−0.132*) |
| Fu-10k, w = 50 | learned | 20 | 3.615 / 22.26 † | 3.615 / 15.09 (−0.115*) | 3.453 / 18.79 (−0.144*) |

† measured with load (6.8–9.0). The share of LKH calls that repeat a window already solved is 58 % (Fu-10k) and 49 %
(Fu-1k) for LKH on every window, 7–33 % for random and 41–54 % for the disagreement and learned signals at w = 20.

**Seeds 1 and 2** (Fu-10k and Fu-1k, w = 20, none and exclude). The learned signal beats random at equal time at
10–20 % in all 3 seeds and with every memo mode (−0.07 to −0.22 pp, p ≤ 0.008). The disagreement does too on Fu-10k,
except seed 2 without memo (−0.02, n.s.); on Fu-1k, seeds 1 and 2 without memo, it ties random (n.s.). At 50 % with
exclude every policy ends within 0.015 pp of LKH on every window. Full table in `tsp/results/ablations/a12.md`.

**Reading.**
- Half of the LKH time of "LKH on every window" was spent re-solving windows already solved. With the cache it takes
  10.9–11.4 s on Fu-10k (not 23 s), and it then reaches 3.845 % faster than any triage at 50 % without memo
  (13–14 s, 3.91–3.93 %).
- With a memo, triage pays only below about 50 %. At 50 %, exclude reaches the same gap as LKH on every window in about
  the same time, whatever the signal.
- The selectors gain from the memo more than random does, because they pick the same windows again. Against random
  with the same memo they win by 0.09–0.22 pp at equal time at 10–20 %.

## A2: repeated timings and the LKH-3 anchors

Fu-10k, w = 20, seed 0, 5 runs of each decisive point, one process at a time.

| point | median time s / inst. | IQR | gap % (all runs) | load before each run |
|---|---:|---:|---:|---|
| LKH on every window, none | 23.13 | 23.09–23.47 | 3.845 | 2.3, 4.5, 3.7, 3.2, 4.3 |
| LKH on every window, (a) cache | 11.37 | 11.20–11.93 | 3.845 | 3.3, 3.1, 3.3, 2.7, 2.3 |
| random 10 / 20 / 50 % | 4.95 / 6.98 / 13.14 | 4.83–5.06 / 6.87–6.99 / 12.99–13.24 | 4.367 / 4.205 / 3.931 | 3.0, 2.8, 3.2, 2.5, 3.2 |
| learned 10 / 20 / 50 % | 5.79 / 8.03 / 14.28 | 5.79–5.98 / 7.98–8.18 / 13.90–14.60 | 4.179 / 4.055 / 3.913 | 3.3, 2.8, 3.0, 3.1, 2.2 |
| learned − random at equal time, 10 / 20 / 50 % | −0.090 to −0.128 / −0.090 to −0.114 / −0.006 to −0.018 pp, the same sign in 5 of 5 runs | | | |
| LKH-3 QUADRANT 5, Fu-10k, budgets 8.4 / 14.2 / 17.5 / 21.5 s, 1 process | 8.19 / 13.98 / 17.33 / 21.44 | | 0.135 / 0.092 / 0.081 / 0.073 | |
| the same in phase 3 (8 processes) | 8.35 / 14.09 / 17.46 / 21.43 | | 0.150 / 0.105 / 0.093 / 0.081 | |
| LKH-3, Fu-1k, budgets 0.72 / 1.25 / 1.54 / 1.94 s: now / phase 3 | 0.75 / 1.27 / 1.56 / 1.97 and 0.75 / 1.28 / 1.57 / 1.98 | | 0.085 / 0.057 / 0.043 / 0.039 and 0.079 / 0.054 / 0.042 / 0.035 | |

**Reading.**
- Every ordering by time among these points is separated by the IQR, and the equal-time sign holds in every run.
  The variation between runs (IQR ≤ 7 % of the median) is smaller than every difference used in a conclusion.
- The LKH-3 anchors measured alone match those measured with 8 processes within 2 % in time and 0.01 pp in gap.

## A3: a "no translation" baseline

The same routes and episodes solved with no rule at all, judged against the true rules. McNemar is paired and exact.
Warehouse: phase C, 200 requests per level. TSP: phase 5 step 3, part C.

| domain | level | arm | no translation: valid % | arm: valid % | above (pp) | arm only / baseline only | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|
| warehouse | L1 | p1 | 1.0 | 100.0 | +99.0 | 198 / 0 | 5e-60 |
| warehouse | L1 | q14 | 1.0 | 72.0 | +71.0 | 142 / 0 | 4e-43 |
| warehouse | L2 | p0 | 1.0 | 32.5 | +31.5 | 63 / 0 | 2e-19 |
| warehouse | L2 | p1 | 1.0 | 60.0 | +59.0 | 118 / 0 | 6e-36 |
| warehouse | L2 | h1 | 1.0 | 67.5 | +66.5 | 133 / 0 | 2e-40 |
| warehouse | L2 | q14 | 1.0 | 66.0 | +65.0 | 130 / 0 | 1e-39 |
| warehouse | L2 | q7 (100) | 1.0 | 48.0 | +47.0 | 47 / 0 | 1e-14 |
| TSP, n = 200 | L2 | p1 (60) | 28.3 | 63.3 | +35.0 | 21 / 0 | 1e-06 |
| TSP, n = 200 | L2 | q14 (60) | 28.3 | 80.0 | +51.7 | 32 / 1 | 8e-09 |
| TSP, n = 200 | L2 | g12 (20) | 30.0 | 60.0 | +30.0 | 7 / 1 | 0.07 |
| TSP, n = 1000 | L2 | p0 (30) | 16.7 | 36.7 | +20.0 | 6 / 0 | 0.03 |
| TSP, n = 1000 | L2 | p1 (30) | 16.7 | 56.7 | +40.0 | 13 / 1 | 0.002 |
| TSP, n = 1000 | L2 | q14 (30) | 16.7 | 60.0 | +43.3 | 15 / 2 | 0.002 |
| TSP, n = 1000 | L2 | q7 (30) | 16.7 | 33.3 | +16.7 | 5 / 0 | 0.06 |
| TSP, n = 1000 | L2 | g12 (20) | 20.0 | 30.0 | +10.0 | 4 / 2 | 0.69 |

Without the L2 texts the audit found mislabelled, the validity rates move by at most 8 pp, and the verdicts are the same
except TSP n = 1000 p0 (p 0.03 → 0.06) and q14 (0.002 → 0.02). L0/L1 rows and the rest of the arms:
`tsp/results/ablations/a3_*.md`.

**Reading.**
- In the warehouse, doing nothing satisfies the rules in 1 % of episodes. So all the validity in phase C comes from
  translating, and every arm is far above the baseline.
- On the TSP, 17–30 % of the routes are valid with no rule at all, so part of the reported validity is free. The
  differences between arms in phase 5 are unchanged, since they are paired on the same routes.
- At n = 1000 on L2, the weaker LLMs (q7, g12) are not distinguishable from not translating. p1 and q14 are.

## A4: SoftDist heatmaps (Xia et al. 2024) instead of the network

τ = 0.75, chosen by the final length of beam B = 10 windows on 20 new uniform instances (n = 1000). Over the grid
0.01 … ∞, the length falls from 24.125 to 24.051 at τ = 0.75 and rises to 24.221 at τ = ∞ (initial 24.223). With
τ = ∞ the heatmap is uniform on the k-NN graph, so the beam's gain comes from the distances in the heatmap, not from
the graph alone. Windows: w = 20, ≤ 5 passes, all instances. Triage: w = 20, ≤ 3 passes, same x as A1, gaps only;
the Fu-1k triage uses the 64 held-out instances.

| set | pipeline | random | network (TAGCN) | SoftDist | SoftDist − network (pp), Wilcoxon p, instances SoftDist better |
|---|---|---:|---:|---:|---|
| Fu-1k | windows, greedy paths | — | 4.649 ± 0.015 | 4.542 | −0.129, 6e-15, 102/128 |
| Fu-1k | windows, beam B = 10 | — | 4.342 ± 0.027 | 4.142 | −0.237, 1e-20, 118/128 |
| Fu-10k | windows, greedy paths | — | 4.453 ± 0.014 | 4.369 | −0.104, 3e-05, 16/16 |
| Fu-10k | windows, beam B = 10 | — | 4.194 ± 0.013 | 4.030 | −0.177, 3e-05, 16/16 |
| Fu-1k | triage 10 / 20 / 50 %, disagreement | 4.663 / 4.438 / 4.112 | 4.456 / 4.297 / 4.093 | 4.664 / 4.520 / 4.264 | SoftDist ≈ or worse than random |
| Fu-1k | triage 10 / 20 / 50 %, learned selector | 4.663 / 4.438 / 4.112 | 4.416 / 4.259 / 4.082 | 4.558 / 4.389 / 4.113 (retrained); 4.605 / 4.442 / 4.172 (network's selector) | retrained: +0.143 / +0.131 / +0.031, p ≤ 0.003 |
| Fu-10k | triage 10 / 20 / 50 %, disagreement | 4.367 / 4.205 / 3.931 | 4.222 / 4.091 / 3.923 | 4.426 / 4.296 / 4.041 | SoftDist worse than random |
| Fu-10k | triage 10 / 20 / 50 %, learned selector | 4.367 / 4.205 / 3.931 | 4.179 / 4.055 / 3.913 | 4.314 / 4.168 / 3.947 (retrained); 4.372 / 4.226 / 3.989 (network's selector) | retrained: +0.135 / +0.113 / +0.034, 0–1 of 16 better |

Network: windows, mean ± std over TAGCN seeds 0–2 (E3), with the test against seed 0; triage, seed 0 (A1). The first
grid's τ = 0.5 gives the same picture: windows −0.10 to −0.22 pp, retrained selector +0.01 to +0.08 pp worse than the
network's. The windows' 2-opt control improves nothing (4.806 / 4.609, as in phase 3). The AP of the retrained
selector is 0.39 (Fu-10k) against 0.45 for the network's.

**Reading.**
- As a path solver in the windows, a heatmap built from distances alone, with one parameter, beats the trained
  network on every set and decoder by 0.10–0.24 pp. The window gains of phase 3 come from the windows, the beam and the
  ordering by p, not from what the network learned.
- As a triage signal the network does carry information that distances do not. SoftDist's disagreement is no better
  than random, and a selector retrained on SoftDist features stays 0.11–0.14 pp behind the network's at 10–20 %.

## A5: 15 warehouse seeds

The 5 stored seeds were reproduced exactly (0 of 5 differ in any comparison), so only seeds 5–14 were run. Gain of
the second method over the first, mean over seeds with a t-based 95 % CI; Wilcoxon signed-rank.

| comparison | second against first | 5 seeds: gain % [95 % CI], p | 15 seeds: gain % [95 % CI], p | seeds the second wins |
|---|---|---|---|---:|
| T1 medium, cap. 3: walking | set partitioning vs Clarke & Wright | +2.2 [+1.9, +2.5], 0.06 | +2.2 [+2.1, +2.4], 6e-05 | 15/15 |
| T1 medium, cap. 3: makespan | set partitioning vs Clarke & Wright | +1.4 [−1.2, +4.0], 0.19 | +0.5 [−1.2, +2.2], 0.59 | 8/15 |
| T3, 90 robots: makespan | hindsight (o2b3) vs map × γ (o2b2) | +1.0 [+0.1, +1.9], 0.06 | +1.6 [+0.7, +2.5], 0.002 | 13/15 |
| B0c R1, 30 robots: makespan | prioritized + dependencies vs PIBT | +7.7 [−1.6, +17.0], 0.13 | +6.5 [+3.7, +9.3], 0.002 | 13/15 |
| B0c R1, 90 robots: makespan | prioritized + dependencies vs PIBT | +13.4 [+5.1, +21.8], 0.06 | +12.7 [+10.1, +15.4], 6e-05 | 15/15 |
| B0c R1, 30 robots: mean delay | prioritized + dependencies vs PIBT | +5.9 [+4.0, +7.8], 0.06 | +5.6 [+4.4, +6.7], 6e-05 | 15/15 |
| B0c R1, 90 robots: mean delay | prioritized + dependencies vs PIBT | +13.8 [+10.9, +16.6], 0.06 | +11.9 [+10.5, +13.3], 6e-05 | 15/15 |

**Reading.**
- With 5 seeds no Wilcoxon could reach p < 0.05 (the smallest possible is 0.0625). With 15 seeds every gain that was
  claimed is significant, and every interval excludes 3 %, which is the threshold of phase B0.
- The set-partitioning walking gain (2.2 %) and the hindsight on traffic (1.6 %) are real but below 3 %. The makespan
  gain of set partitioning is not there (+0.5 %, n.s.). Prioritized beats PIBT by 6.5 % and 12.7 %.

## Conclusions

**Change.**
1. **LKH on every window costs half of what phase 3 reported.** 49–58 % of its LKH calls repeat a window it already
   solved. With a cache it takes 11.4 s on Fu-10k and 1.14 s on Fu-1k, not 23 s and 2.0 s, and it beats triage at
   50 % without memo on both gap and time. (A1; `tsp/docs/phase3_triage_wsize_report.md`,
   `phase3_benchmark_report.md`, `phase3_window_report.md`.)
2. **w = 50 with the learned signal at top 20 % no longer beats LKH on every w = 20 window in time.** With a memo on
   both, it is 0.23 pp better but 39 % slower (15.1 s against 10.9 s). At about the same time (w = 50 learned top 10 %,
   10.3 s) the two tie (3.861 % against 3.845 %). It still wins at budgets above about 11 s. (A1;
   `phase3_triage_wsize_report.md`.)
3. **At equal time the cheap learned signal beats random at 10–20 % on Fu-10k as well.** The margin is 0.07–0.13 pp
   in 5 of 5 runs and 3 of 3 seeds, so "it is a tie on Fu-10k" holds only at 50 %. With a memo it also wins at 50 %
   (−0.09 pp). "At equal time random selection is as good or better" (benchmark) is no longer true with cheap scoring.
   (A1/A2; `phase3_fast_triage_report.md`, `phase3_learned_triage_report.md`, `phase3_benchmark_report.md`.)
4. **The window gains are not the network's.** SoftDist, with no learning, beats the TAGCN in the same window
   pipelines by 0.10–0.24 pp. So "window refinement is the first use of the TSP-20 model that beats greedy edge +
   2-opt" holds for the pipeline, not for the model. (A4; `phase3_window_report.md`, `phase3_benchmark_report.md`.)

**Hold.**
5. LKH-3 on the whole instance beats all our methods by 30–110× at equal time. The anchors measured alone match, and
   at the new 11.4 s of LKH on every window LKH-3 still gives ≈ 0.11 % against 3.85 %. (A2; `phase3_benchmark_report.md`.)
6. The time orderings of phase 3 between our policies are separated by the IQR of 5 runs. (A2.)
7. The network's signal is worth more than distances as a triage signal: the learned selector beats random, and the
   SoftDist selector loses to it. (A4; `phase3_learned_triage_report.md`.)
8. Phase C and phase 5: every reliable arm is far above not translating, and the comparisons between arms are
   unchanged. (A3; `warehouse/docs/phaseC_language.md`, `tsp/docs/phase5_open_language_report.md`.)
9. Phase B0b/B0c verdicts: no margin reaches 3 %, and prioritized + dependencies beats PIBT by 6.5 % / 12.7 %, now
   with p ≤ 0.002: the margin intervals lie below 3 %, and those of prioritized against PIBT exclude 0. (A5; `warehouse/docs/phaseB0b_margins.md`,
   `phaseB0c_realism.md`.)

**Get weaker.**
10. "The disagreement at w = 50 is worse than random at equal time" holds only without a memo. With a cache or
    exclude it ties random (−0.06 to +0.04 pp). (A1; `phase3_triage_wsize_report.md`.)
11. TSP phase 5 validity: 17–30 % of routes are valid with no translation at all. So "q14 gives 80 % valid routes"
    means +52 pp over doing nothing, and at n = 1000 on L2 q7 and g12 cannot be told apart from not translating. (A3;
    `phase5_open_language_report.md`.)
12. "Hindsight on traffic … all within noise" (B0b): at 90 robots it is +1.6 % [0.7, 2.5], p = 0.002. The gain is
    real, though still below 3 %, so the verdict holds but the wording does not. (A5; `phaseB0b_margins.md`.)

## Corrections to the earlier reports (commit "ablation corrections")

The sentences that stopped being true got a dated note in place; nothing was deleted. The diff, for review:

```diff
diff --git a/tsp/docs/phase3_benchmark_report.md b/tsp/docs/phase3_benchmark_report.md
index 0bcf80c..42adf21 100644
--- a/tsp/docs/phase3_benchmark_report.md
+++ b/tsp/docs/phase3_benchmark_report.md
@@ -11,6 +11,13 @@ help LKH: as LKH's candidate set it is worse than the geometric ones, and as a t
 windows better than random only when few windows go to LKH (20 %), a gain that costs more network time than it
 saves, so at equal time random selection is as good or better.
 
+*Correction 2026-11-03 (ablations A1/A4, `../../docs/ablations_2026-11-03.md`): (a) with the batched scoring of `phase3_fast_triage_report.md`, the
+last clause is no longer true: at 10–20 % the learned signal beats random at equal time by 0.07–0.13 pp (3 of 3 seeds,
+both sets), and the disagreement by 0.05–0.07 pp on Fu-10k in 2 of 3 seeds. (b) Half of the LKH calls of "LKH on all windows" re-solve a window already solved: with a cache
+it takes 1.14 s / 10.9–11.4 s, not 2.02 / 22.92 s, for the same 3.92 / 3.85 %; LKH-3 at that budget still gives about
+0.11 % at 10k, so (2) holds. (c) In (1), the window gains are not the network's: SoftDist heatmaps (distances only, no
+learning) in the same window pipelines give 4.14 / 4.03 % with beam B = 10, against the network's 4.34 / 4.19 %.*
+
 **Key numbers.**
 - Equal time (E1 vs. E3): at the budget of triage top 20 % (0.72 s / 8.4 s) LKH-3 QUADRANT 5 reaches 0.079 % / 0.150 %
   against our 4.34 % / 4.21 %; at the budget of LKH on all windows (1.94 s / 21.5 s), 0.035 % / 0.081 % against 3.92 % /
diff --git a/tsp/docs/phase3_fast_triage_report.md b/tsp/docs/phase3_fast_triage_report.md
index e61d582..91ce3c4 100644
--- a/tsp/docs/phase3_fast_triage_report.md
+++ b/tsp/docs/phase3_fast_triage_report.md
@@ -19,6 +19,10 @@ tie.**
   network processes pay more for their own LKH calls.
 - **The 3-process diagnostic** suggested a win at 10k, but it used only the first 4 instances. The full sequential
   run (16 instances, 3 seeds) does not confirm it.
+- *Correction 2026-11-03 (ablations A1/A2, `../../docs/ablations_2026-11-03.md`): the tie on Fu-10k is at top 50 % only (−0.011 pp, the same sign in
+  5 of 5 repeated runs). At top 10–20 % the batched learned signal beats random at equal time by 0.07–0.13 pp, in all
+  3 seeds and in 5 of 5 runs, as the table "Same time" below already implied (learned 20 % at 8.13 s against random's
+  curve at that time).*
 
 ## The cheap scoring (version A)
 
diff --git a/tsp/docs/phase3_learned_triage_report.md b/tsp/docs/phase3_learned_triage_report.md
index f00bd2a..4952202 100644
--- a/tsp/docs/phase3_learned_triage_report.md
+++ b/tsp/docs/phase3_learned_triage_report.md
@@ -31,6 +31,9 @@ Generated 2026-10-29 by scripts/phase3_learned_triage_report.py (models at commi
    - **Why.** Scoring the windows costs about 4 s per 10k instance: building the window graphs, the network and the
      features. That is more than the LKH time the better ranking saves. The features themselves add only
      0.04 s / 0.4 s (1k / 10k) on top of the disagreement.
+   - *Correction 2026-11-03 (ablation A1, `../../docs/ablations_2026-11-03.md`): the "No" rests on the 4 s scoring of this report. With the batched
+     scoring (`phase3_fast_triage_report.md`) the learned signal beats random at equal time at top 10–20 % (0.07–0.13 pp
+     on Fu-10k), and with a window memo also at 50 %: 3.913 % in 8.7 s against random's 3.931 % in 9.7 s.*
 
 ## What the features say
 
diff --git a/tsp/docs/phase3_triage_wsize_report.md b/tsp/docs/phase3_triage_wsize_report.md
index 9f3954f..b7b2e6b 100644
--- a/tsp/docs/phase3_triage_wsize_report.md
+++ b/tsp/docs/phase3_triage_wsize_report.md
@@ -18,6 +18,8 @@ Generated 2026-10-30 by scripts/phase3_triage_wsize_report.py. Greedy edge + 2-o
      (0.76–0.84 against 0.66–0.73). The network was trained on 20-city windows: in a rescaled 50-city window its p
      still says which windows improve, but not which gain the most. The learned model, retrained for w = 50, fixes
      this.
+     *Correction 2026-11-03 (ablation A1, `../../docs/ablations_2026-11-03.md`): this holds only without a window memo. With the cache or the
+     exclusion of windows already solved, the disagreement at w = 50 ties random at equal time (−0.06 to +0.04 pp).*
 2. **How much of the oracle does the learned signal recover?**
    - **w = 50.** On Fu-10k: 49 % at top 10 %, 62 % at top 20 % and 88 % at top 50 %. On tsp1000: 51 %, 65 % and 89 %.
      Random gets 37 / 56 / 90 % and the disagreement 34 / 50 / 80 %.
@@ -30,6 +32,11 @@ Generated 2026-10-30 by scripts/phase3_triage_wsize_report.py. Greedy edge + 2-o
    - Even random at w = 50, top 20 % (3.697 % in 15.0 s) beats it.
    - LKH on every w = 50 window gives 3.028 % in 63.6 s. Larger windows simply give LKH more to work with, and
      selecting 10–20 % of them is where the time is best spent.
+   - *Correction 2026-11-03 (ablation A1, `../../docs/ablations_2026-11-03.md`): "27 % faster" is no longer true. Half of the LKH calls of LKH on
+     every w = 20 window re-solve a window already solved; with a cache it takes 10.9 s on Fu-10k (all 16 instances,
+     3.845 %), while w = 50 + learned top 20 % with the same cache takes 15.1 s (3.615 %): 0.23 pp better but 39 %
+     slower. At about the same time (w = 50 learned top 10 %, 10.3 s, 3.861 %) the two tie. w = 50 wins only at budgets
+     above about 11 s.*
 
 ## Caveats
 
diff --git a/tsp/docs/phase3_window_report.md b/tsp/docs/phase3_window_report.md
index 10eda22..e6b0e05 100644
--- a/tsp/docs/phase3_window_report.md
+++ b/tsp/docs/phase3_window_report.md
@@ -172,6 +172,13 @@ One CPU process per solver/seed (5 in parallel), network on the GPU in batches.
    - **Cost per pp gained:** LKH windows with w = 20 cost about the same as beam paths or less (28 s/pp against 35 s/pp at n = 10000; 1.9 against 2.7 s/pp at n = 1000), and reach about twice the gain.
    - **Conclusion:** the TSP-20 model works as a local improver, but as a replacement for an exact window solver it is not competitive yet.
 
+*Correction 2026-11-03 (ablations A1/A4, `../../docs/ablations_2026-11-03.md`): (1) and (2) hold for the window pipeline, not for the model. A heatmap
+built from distances alone (SoftDist, one parameter τ = 0.75, no learning) beats the TSP-20 model in the same windows
+by 0.10–0.24 pp on the Fu et al. sets (beam B = 10, w = 20: 4.14 against 4.34 % at 1k and 4.03 against 4.19 % at 10k,
+16/16 instances at 10k), so the gain over greedy edge + 2-opt does not show that the network learned anything useful
+for paths. In (3), about half of the LKH window calls re-solve a window already solved (A1, ranked triage at 100 %): with
+a cache, LKH windows cost about half the time quoted here, so they are cheaper per pp than beam paths.*
+
 ## Missing results
 
 None.
diff --git a/warehouse/docs/phaseB0b_margins.md b/warehouse/docs/phaseB0b_margins.md
index 77b3385..b880158 100644
--- a/warehouse/docs/phaseB0b_margins.md
+++ b/warehouse/docs/phaseB0b_margins.md
@@ -84,6 +84,10 @@ The medium warehouse with corridors of width 1, more robots and proportionally m
 
 **Traffic never reaches 3 % for layer 1.** The baseline loses up to 24 % of its makespan to traffic at 131 % of free flow, yet hindsight on real edge times gains −0.4 to +1.8 % on the best rule without learning, all within noise or close to it, and at most +2.9 % on the Hungarian. Hindsight has the advantage of the best of 3 runs per seed. So no level of congestion in this range makes better distances worth more than 3 %. Congestion here is a problem of the planner: at 5 x 63 % of its calls fall back to PIBT. A model that acts on the planner rather than on layer 1 was not measured at these densities. γ grows from 0.46 to 0.86 with density.
 
+*Correction 2026-11-03 (ablation A5, `../../docs/ablations_2026-11-03.md`): "all within noise" is no longer true at 90
+robots. With 15 seeds, hindsight gains +1.6 % [0.7, 2.5] on map × γ (p = 0.002, 13 of 15 seeds): a real gain, still
+below 3 %, so the conclusion stands.*
+
 | times robots | robots | orders | baseline, % free flow | steps under PIBT % | baseline | Hungarian | static map | map x gamma | hindsight | best not learned | hindsight gain on it % | gain on Hungarian % | seeds |
 |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
 | 1 | 30 | 150 | 105 | 0 | 1007 | 953 | 952 | 944 | 947 | o2b2 | -0.4 ± 1.2 (p=0.438) | +0.6 ± 0.6 (p=0.125) | 5 |
```
