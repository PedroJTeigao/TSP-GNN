# Phase 5 — vision probe: can a vision model see where LKH will gain?

Generated 2026-10-30 by scripts/phase5_vision_probe_report.py. Model gemma4:e4b-it-qat on the private Ollama (port 11435), num_gpu = 999 (43/43 layers on the GPU)

## Summary

**Does the vision model see anything the disagreement does not? No.**
- **Tour only (image a): at the level of random.**
  - AP: 0.501, against 0.475 for random and 0.638 for the disagreement.
  - Spearman with the share of a cell's windows that LKH improves: 0.00.
  - The top-4 cells catch 25 % of the LKH gain, against 23 % at random and 38 % for the disagreement.
- **Tour coloured by the network's p (image b): it reads a little of the colour.**
  - Spearman 0.06; its top-4 cells catch 28 % of the gain.
  - It recovers only a small part of the signal it is handed, and stays far below the disagreement computed from
    the same p.
- **As a policy, vision is no better than random.**
  - "LKH on the windows of the top-4 cells" by vision is no better than 4 random cells (a: +0.012 pp, p = 0.40;
    b: −0.023 pp, p = 0.16).
  - It is clearly worse than the top-4 cells by disagreement (+0.12 / +0.09 pp, p < 1e-3).
  - It also costs 70–110 s of model time per instance, against 0.6 s for the whole cell policy.
- **The answers are well-formed but unstable.**
  - All 300 are valid JSON with the 16 cells.
  - Between repetitions the top-4 cells agree with a Jaccard of 0.22 (a) and 0.36 (b). Two random draws of 4 cells
    out of 16 would give ≈ 0.14.

## Set-up and model

- **Instances.** 50 Fu-1k instances; 6.3 windows per cell on average (w = 20). The 768 × 768 images are legible.
- **Model.** gemma4:e4b-it-qat fitted entirely on the GPU: 43 / 43 layers, 4.3 GiB, under the 5.25 GiB budget, so the
  e2b fallback was not needed.
- **Thinking left on.** The model "thinks" before answering: ~1100–1300 generated tokens per call for a ~50-token
  JSON answer. This was kept on purpose (the quality of the choices is what matters).

## Caveats

- **Latency depends on the load.** The 3 test calls took 26–29 s.
  - Until ~16:19 (97 calls) the mean was 67 s.
  - From ~16:19 another project's Ollama shared the GPU and the CPU (8.5 GiB, ~460 % CPU). Calls then took 170–200 s
    for a while (mean 98 s over the remaining 203).
  - The per-image latencies in the tables mix both periods: image (a), repetition 1 fell mostly in the shared
    period, hence 107 s against 69 s for (b).
  - This does not change any conclusion: vision loses on quality, and even 26 s is 40× the cell policy.
- **The labels are coarse.** Each cell holds ~6 windows, so the cell label is a noisy share. The oracle's AP is 0.83,
  not 1, because the target is "above the instance mean", which is not the same thing as gain.
- **Small sample.** One model, one prompt per image type, one grid size (4 × 4), 50 instances. A different prompt, a
  finer grid or a larger model could do better. Within this probe nothing suggests it.

## Set-up

- 50 Fu-1k instances (the learned-triage window logs, TAGCN seed 0), greedy + 2-opt tour; images 768 × 768 with a numbered 4 × 4 grid: (a) tour only, (b) tour edges coloured by the network's p (green high, red low). At 768 px the 1000 cities, the tour and the cell numbers are legible (checked by eye).
- Windows per cell: 6.2 on average (w = 20, offsets 0 and 10; a window belongs to the cell of its centroid).
- Model fitting: [{"num_gpu": 999, "ollama_pc_gpu_mib": 4344, "seconds": 148.30398022779264, "ok": true, "layers": [43, 43], "error": null}]; latency of 3 test calls: [26.3, 27.4, 29.1] s; GPU used by other processes at tuning: 0 MiB.

Start (waiting for the GPU):

```
2026-10-30 13:54:16  load 1.52 1.34 1.37  GPU used by others 0 MiB  waiting for the larger-windows run
2026-10-30 14:04:17  load 1.10 1.26 1.34  GPU used by others 0 MiB  waiting for the larger-windows run
2026-10-30 14:14:17  load 1.25 1.24 1.27  GPU used by others 0 MiB  waiting for the larger-windows run
2026-10-30 14:24:18  load 1.48 1.07 1.07  GPU used by others 104 MiB  GPU below 6 GiB: start with gemma4:e4b-it-qat
2026-10-30 21:51:32  load 9.77 9.67 9.79  GPU used by others 0 MiB  after the calls
```

## Answers

| Image | Calls | Unusable answers (%) | Cells listed (mean) | Latency per call (s) | Top-4 agreement between repetitions (Jaccard) |
|---|---|---|---|---|---|
| (a) tour only | 150 | 0.0 | 16.0 | 107.4 | 0.22 |
| (b) tour coloured by p | 150 | 0.0 | 16.0 | 69.1 | 0.36 |

## Signal per cell

| Ranking | AP (target: cell above its instance's mean) | Spearman with the cell's improving share | Top-4 cells: share of the LKH gain | Instances ranked |
|---|---|---|---|---|
| Disagreement (mean per cell) | 0.638 | 0.330 | 0.376 | 50 |
| Random | 0.475 ± 0.010 | 0.012 ± 0.039 | 0.234 ± 0.019 | 50 ± 0 |
| Oracle (LKH gain per cell) | 0.828 | 0.768 | 0.652 | 50 |
| Vision (a) tour only | 0.501 ± 0.007 | 0.003 ± 0.024 | 0.252 ± 0.011 | 50 ± 0 |
| Vision (b) tour coloured by p | 0.510 ± 0.020 | 0.064 ± 0.023 | 0.278 ± 0.010 | 50 ± 0 |

Random AP = positive rate = 0.474. Mean ± std over the 3 repetitions (vision: temperature 0.5, seeds 0-2; random: 3 draws). Top-4 capture of a random choice ≈ 0.25.

## Policy: LKH on the windows of the top-4 cells

| Policy | Gap (%) | Gain (pp) | Windows to LKH (%) | Time (s) | of which the model and the image (s) |
|---|---|---|---|---|---|
| P2 top 20 % by disagreement (batched) | 4.216 | 0.571 | 20.0 | 0.63 | 0.00 |
| Top-4 cells by disagreement | 4.437 | 0.350 | 24.1 | 0.61 | 0.00 |
| Top-4 random cells (3 draws) | 4.545 | 0.241 | 25.0 | 0.60 | 0.00 |
| Top-4 cells, oracle (LKH gain) | 4.213 | 0.574 | 25.4 | 0.64 | 0.00 |
| Top-4 cells, vision (a) tour only | 4.557 | 0.230 | 25.2 | 108.15 | 107.54 |
| Top-4 cells, vision (b) tour coloured by p | 4.523 | 0.264 | 25.1 | 69.83 | 69.22 |

Initial gap 4.787 % (greedy + 2-opt, 0.24 s per instance, included in every time). Up to 3 passes, w = 20. Vision time = mean model latency per call + drawing the image (+ the network's edge probabilities for (b)).

| Vision policy | vs. | Better / equal / worse (% inst.) | Mean diff (pp) | p (Wilcoxon) |
|---|---|---|---|---|
| Top-4 cells, vision (a) tour only | Top-4 cells by disagreement | 18 / 0 / 82 | +0.121 | 3.5e-07 |
| Top-4 cells, vision (a) tour only | Top-4 random cells (3 draws) | 48 / 0 / 52 | +0.012 | 4.0e-01 |
| Top-4 cells, vision (b) tour coloured by p | Top-4 cells by disagreement | 30 / 0 / 70 | +0.086 | 2.5e-04 |
| Top-4 cells, vision (b) tour coloured by p | Top-4 random cells (3 draws) | 62 / 0 / 38 | -0.023 | 1.6e-01 |

