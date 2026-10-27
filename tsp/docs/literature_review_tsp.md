# Literature review extract for the Phase 3 benchmark (sections A and D of the 2026-10-23 review)

Published numbers are gaps to LKH-3 / Concorde as reported by each paper, on their own hardware and test sets. Use them in `phase3_benchmark_report.md` only as "not reproduced" points. Links were verified on 2026-10-23 (title + authors); "nv" = not verified.

## Important correction found in the review

Joshi et al. 2021 use **k = 0.2·n** ("fixed diameter", 20 %) for test-time sparsification, not 0.5·n. Our k = 0.5·n is our own extension. State this in the comparability caveats; a 0.2·n ablation is pending.

## A. Published methods with numbers (gap % 1k / 10k, time per instance, hardware, reference)

| method | trained on target size? | TSP-1000 gap | TSP-10000 gap | time 1k | time 10k | hardware | reference solver | link |
|---|---|---|---|---|---|---|---|---|
| Fu, Qiu & Zha 2021 (Att-GCRN + MCTS) | no (TSP-20/50 subgraphs) | 3.22 % | 4.39 % | 44 s net + 11.7 min MCTS | 4.2 min + 1.69 h | GTX 1080Ti + Xeon Gold 5118 (8 cores) | Concorde / LKH-3 | https://arxiv.org/abs/2012.10658 |
| Cheng et al. 2023 (Select & Optimize, r=50/100, all windows) | trained on TSP-200/500/1000 | 2.80 % | 3.52 % | 2.9 s + 23 s | 40 s + 418 s (LKH-3: 1437 s) | V100; LKH-3/MCTS on Xeon E5-2650 | LKH-3 | https://proceedings.mlr.press/v206/cheng23a.html |
| H-TSP 2023 (hierarchical RL) | trained up to 10k | 6.62 % (4.06 % with LKH-3 lower level) | 7.32 % (5.57 % with LKH-3) | 0.33 s | 3.3 s | V100 16 GB | LKH-3 (4747 s at 10k) | https://arxiv.org/abs/2304.09395 |
| GLOP 2024 (revisers 100/50/20/10) | revisers trained on small SHPP | 3.85 % / 3.11 % | 5.36 % / 4.90 % | 3 min (test set) | 1.8 min (test set) | RTX 3090 + Xeon 12 cores | LKH-3 | https://arxiv.org/abs/2312.08224 |
| DIFUSCO 2023, greedy + 2-opt | yes (trained per size, 8× V100) | 1.90 % | 3.10 % | 24 min (set) | 47 min (set) | 8× V100 train; 1 GPU test | LKH-3 labels | https://arxiv.org/abs/2302.08224 |
| DIFUSCO 2023, + MCTS | yes | 1.17 % | 2.58 % | — | — | idem | LKH-3 | idem |
| SoftDist + MCTS (Xia et al. 2024), no learning | n/a | 2.20 % | 3.13 % | MCTS budget as Fu | idem | EPYC 7V13 (64 threads); SoftDist on A100 | LKH-3 | https://arxiv.org/abs/2406.03503 |
| Xia 2024 re-evaluation: Att-GCN+MCTS / DIMES / DIFUSCO | — | 2.37 / 2.44 / 1.18 % | 3.80 / 3.23 / 2.77 % | — | — | idem | LKH-3 | idem |
| INViT 2024 | no (TSP-100, zero-shot) | 5.99 % | ≈ 6–7.5 % | 4.8 min (set) | — | RTX 4090 | LKH-3 | https://arxiv.org/abs/2402.02317 |
| UDC 2024 (×50) | trained ≤ 2k | 1.78 % | — (App. D.5, nv) | 8 min (set) | — | V100S | LKH-3 | https://arxiv.org/abs/2407.00312 |
| LEHD 2023, greedy / RRC-1000 | no (TSP-100) | 3.17 % / 0.72 % | — | 1.6 min / 7 h | — | RTX 3090 | LKH | https://arxiv.org/abs/2310.07985 |
| BQ-NCO 2023, greedy / beam 16 | no (TSP-100) | 2.29 % / 1.38 % | — | 2 min / 38 min | — | V100-S | LKH | https://arxiv.org/abs/2301.03313 |
| NeuroLKH 2021 (equal time to LKH 1-trial) | TSP-101–500 | 0.09 % (LKH 1.06 %) | — (5k: 0.05 %) | = LKH 1 trial | — | 2080Ti + i9 | LKH 1000 trials | https://arxiv.org/abs/2110.07983 |
| HyperNS 2025 | — | 0.3 % | 0.9 % | 0.89 h | 0.93 h (LKH-3 2.52 h) | V100 32 GB + Xeon Gold | LKH-3 | https://arxiv.org/abs/2510.20169 |
| Pan et al. 2024 (GT-Prior + tuned MCTS) | no learning | — | 2.13 % | — | — | — | LKH-3 | https://arxiv.org/abs/2411.09238 |
| Johnson & McGeoch 1997, greedy / 2-opt / 3-opt / LK | no learning | 16.6 / 5.0 / 3.0 / ~1–2 % over Held-Karp (N=10k) | idem | — | — | 1997 CPU | Held-Karp bound | chapter (nv) |

Notes for the table:
- "time (set)" = time for the whole published test set (DIFUSCO/GLOP/INViT/UDC report set times, Fu/Cheng per instance).
- Standard test sets (Fu 2021, reused by DIMES/DIFUSCO/UDC/GLOP): 128 instances TSP-1000, 16 instances TSP-10000, from https://github.com/Spider-scnu/TSP (files `tsp1000_test_concorde.txt`, `tsp10000_test_concorde.txt`; confirm names/licence).
- Our models are trained on TSP-20 only; our reference at ≥ 2000 cities was a single elkai/LKH run (to be replaced by LKH-3 in E0).

## Closest methods to our Phase 3 (what they do differently)

- **Select & Optimize (Cheng 2023):** same window scheme (consecutive sub-paths, fixed endpoints), r = 50/100, re-optimizer is an RL Transformer, every window is evaluated and the best improvement applied (hill-climbing); initial tour POPMUSIC + LK; no "LKH on the same windows" baseline, no triage.
- **Learning to Delegate (Li, Yan & Wu 2021):** learned regression of the improvement a sub-solver (LKH-3) would achieve per sub-problem; CVRP only; compares learned vs random selection (1.5–2× faster). Code: https://github.com/mit-wu-lab/learning-to-delegate
- **L2Seg (Ouyang et al. 2025, ICLR 2026):** classifier of "unstable" segments trained with solver labels; LKH-3 works only on unstable parts; CVRP/VRPTW; 2–7× speed-up. https://arxiv.org/abs/2507.01037
- **NeuroLKH (Xin et al. 2021):** GNN replaces LKH's candidate set and penalties (guides *how* LKH searches, everywhere). Our heatmap could be LKH's candidate set inside windows.
- **GLOP 2024 / UDC 2024 / LEHD RRC:** decompose by tour segments, not spatial regions; UDC handles seams with a shifted second pass ("Reunion").
- **H-TSP 2023 + LKH-3 lower level (4.06 % / 5.57 %)** is the best published estimate of the loss of a rigid spatial decomposition with a near-exact local solver; consistent with our ≈ 5 pp.
- **Xia 2024, Pan 2024:** search (MCTS tuning) matters more than the heatmap; a zero heatmap with tuned MCTS reaches 1.85 % on TSP-1000.
- No paper found that uses the heatmap's own disagreement with the current tour (1 − mean p over tour edges in a window) as a triage signal for an exact solver.

## D. Our Phase 1–3 conclusions vs the literature

| # | our conclusion | verdict | main evidence |
|---|---|---|---|
| 1 | k ∝ n at test + beam make the difference, not more data/capacity | confirmed (with k = 0.2n in Joshi) | Joshi 2021 Fig. 4; INViT 2024 (8-NN covers > 98 % of optimal next-node choices); Lischka 2024 (sparsification is first-order; dense GAT 15.7 % vs 0.7 % with k=20); Pan 2024 (5-NN > 94 %) |
| 2 | encoders without distance in messages transfer better | confirmed | RsGCN 2025 (mean neighbour distance 0.179 / 0.088 / 0.021 at TSP-20/200/2000; Gated GCN without rescaling collapses from 1k); Fu 2021, INViT, ELG 2024, ICAM 2024 all rescale distances |
| 3 | more epochs on TSP-20 hurts transfer | no paper found | adjacent: Joshi 2021 Fig. 11 (SL stagnates), Manchanda 2022 (AM trained on N=20: 22.6 % at N=100), RsGCN trains 3 epochs |
| 4 | rigid partition costs ≈ 5 pp via "one visit per group"; stitching is not the problem | no published quantification; indirectly confirmed | H-TSP + LKH-3 4.06–5.57 %; Karp 1977 error ∝ t^-1/2; Walshaw 2002 coarsens by edges, not regions; GLOP/UDC/S&O decompose along the tour; HyperNS lets LK cross boundaries |
| 5 | network as triage/guide of LKH, not as solver; disagreement predicts where LKH gains | partly published (where: Li 2021, L2Seg 2025; how: NeuroLKH 2021, Hudson 2022); no paper with heatmap uncertainty as the criterion | Li 2021 (learned vs random selection); L2Seg; NeuroLKH (gap 10.6‰ → 0.9‰ at equal time, TSP-1000); Xia 2024 / Pan 2024 (search dominates); Cheng 2023 (windows, evaluates all) |

## Comparability caveats to include in the report

1. Reference solution: published gaps are vs Concorde or long LKH-3 runs; ours (before E0) vs a single LKH run at ≥ 2000 cities, so our gaps were optimistic.
2. Hardware: published times span 1080Ti to 8× V100 and 64-thread EPYC; ours is a shared T4 (< 5.5 GiB) with `nice -n 10`.
3. Test sets: published numbers use Fu's 128/16 instances; ours used 20 / 3 own instances before E3.
4. Training size: DIFUSCO, UDC, H-TSP, S&O train on or near the target size; our models see only TSP-20.
5. Our k = 0.5n differs from Joshi's 0.2n.
6. Time accounting: some papers report network time and search time separately, some the test-set total; we report wall-clock per instance including preprocessing.

## Links (verified 2026-10-23 unless marked nv)

Joshi 2019 https://arxiv.org/abs/1906.01227 · Joshi 2021 https://arxiv.org/abs/2006.07054 · Fu 2021 https://arxiv.org/abs/2012.10658 · TAGCN https://arxiv.org/abs/1710.10370 · Li, Yan & Wu 2021 https://arxiv.org/abs/2107.04139 · Cheng 2023 https://proceedings.mlr.press/v206/cheng23a.html · Ouyang 2025 https://arxiv.org/abs/2507.01037 · Xia 2024 https://arxiv.org/abs/2406.03503 · NeuroLKH https://arxiv.org/abs/2110.07983 · H-TSP https://arxiv.org/abs/2304.09395 · GLOP https://arxiv.org/abs/2312.08224 · DIFUSCO https://arxiv.org/abs/2302.08224 · INViT https://arxiv.org/abs/2402.02317 · UDC https://arxiv.org/abs/2407.00312 · LEHD https://arxiv.org/abs/2310.07985 · BQ-NCO https://arxiv.org/abs/2301.03313 · HyperNS https://arxiv.org/abs/2510.20169 · Pan 2024 https://arxiv.org/abs/2411.09238 · Lischka 2024 https://arxiv.org/abs/2403.17159 · RsGCN https://arxiv.org/abs/2506.00533 · Hudson 2022 https://arxiv.org/abs/2110.05291 · Karp 1977 https://www2.eecs.berkeley.edu/Pubs/TechRpts/1977/28889.html · Walshaw 2002 https://pubsonline.informs.org/doi/10.1287/opre.50.5.862.373 · Johnson & McGeoch 1997 preprint (nv) https://redmine.iam.upr.si/attachments/download/242/TSPchapter.pdf
