# Phase 5, step 3 — open language with a larger local model

Generated 2026-11-01 by scripts/phase5_open_language_report.py. Data in results/phase5/open_language/ (outside git); prompts in src/tsp_gnn/open_dsl.py and src/tsp_gnn/open_gen.py and with every L2 record.

## Conclusion

A larger local LLM beats a fair parser only in part of open language: on the L2 paragraphs qwen3:14b gets 70.9 % of the rules against 54.7 % for p1 at n = 200 (McNemar p = 0.0003), ties at n = 1000 (59.6 % against 60.3 %, p = 1), and loses to p1 on L0 and on the template sentences of L1 (75–80 % against 100 %). Its time cost is 10 s per call on a cool GPU and 25–30 s on the throttled T4, against about 0 s for the parser, and thinking on takes about 16 times longer in the run (438 s against 27 s per L2 text) with no gain on L2 in a small sample. Route validity follows the same split: on L2, q14 gives 80 % valid routes against 63 % for p1 at n = 200 (p = 0.053) and 60 % against 57 % at n = 1000 (p = 1), and with an audited label error of 28 % in L2 (CI 18.5–40.8 %) neither arm is safe to use without checking routes in code.

## Goal

Test the hypothesis of the earlier steps fairly: an LLM only has a place in really open language, and it needs a larger model. Before this step the parser got 99 % and qwen2.5:7b 58–81 %, on paraphrases filtered by the parser itself, which favoured the parser. Here L2 is not filtered by any tested parser or model.

## Design

- **Models.** Private Ollama (port 11435): qwen3:14b (q14), qwen2.5:7b (q7), gemma3:12b (g12). One model loaded at a time, num_ctx at most 8192, 100 % in the GPU.
- **DSL.** The LLMs and the parsers answer with rules over selectors: id, name, `type_is`, `area_is`, `nearest`, `any_of`, `except`. Code expands them to the primitive rules of step 1 (`src/tsp_gnn/open_dsl.py`, 24 tests).
- **Scoring.** On primitives. A rule is exact if all its primitives are predicted and no wrong primitive of the same type touches its customers. An instance is exact if the sets are equal.
- **Levels.** L0 = text_v2b. L1 = composite rules written from sentence templates (2–3 per instance, all solvable by LKH-3). L2 = the same rules rewritten by gemma3:12b at temperature 0.7 into one paragraph with coreferences, double negations and distractors, unfiltered.
- **Legend.** The same for all arms: name, type, area and coordinates of 40 places at n = 200 and 100 at n = 1000. The full table does not fit in 8192 tokens.
- **Prompt.** The same for the LLMs: legend + text + DSL specification + 3 L0 examples, JSON output, one retry if the JSON is invalid, counted in the time.
- **p1.** Regular expressions over the DSL. Written from the L1 templates, then tuned in four rounds on 20 held-out L2 paragraphs at n = 200 only, about 1 h of wall clock in all, under the 2 h limit. On those paragraphs it ended at 43/54 rules. It never saw a test L2 text and was frozen (hash in `p1_frozen.sha256`) before the audit. The n = 1000 dev paragraphs were never written.
- **Part C.** Code check before LKH-3 (precedence cycles, impossible windows, contradictory edges and others): a rejected rule is logged and dropped, and the tour is solved without it. LKH-3 with TIME_LIMIT 60 s / 120 s, external timeout 3 × TL + 60 s, start from the pure route. Routes are judged in code against the label.
- **Sample sizes, reduced for time.** The estimate after step 0 was about 16 h for the plan, because the T4 throttles (see Step 0). What was run, against the plan of at least 300 rules per level and n:

| arm | L0 instances per n | L1, L2 instances per n | L2 label rules |
|---|---|---|---|
| p0, p1, q14 | 30 | 60 | about 150 per n |
| q7 | 20 | 40 | about 100 per n |
| g12 | 20 | 20 | about 50 per n |
| q14t (thinking on) | — | 12, L2 only | about 30 per n |

Also reduced: L2 at n = 1000 has 60 texts, not 150; part C has 60 / 30 instances per level; step-3 formalisation has about 100 rules per n instead of 150; the verifier has 100 records instead of 200. All arms use the same first instances, so the comparisons are paired.

## Step 0: models and speed

| model, mode | size in memory (GB) | in GPU (%) | GPU used (MiB) | tokens/s | mean latency (s) | p95 (s) | rules exact / 20 instances |
|---|---|---|---|---|---|---|---|
| qwen3:14b think_off | 9.9 | 100 | 9565 | 5.6 | 25.3 | 31.5 | 32 / 46 |
| qwen3:14b think_on | 9.9 | 100 | 9565 | 5.1 | 189.3 | 462.9 | 40 / 46 |
| gemma3:12b speed | 8.1 | 100 | 8915 | 8.0 | 26.9 | 36.9 | 19 / 46 |
| qwen2.5:7b speed | 4.9 | 100 | 4849 | 15.2 | 12.2 | 16.1 | 24 / 46 |

**Thinking mode.** On 20 text_v2b instances thinking on gets 40/46 rules against 32/46 off, at 7.5 times the time (189 s against 25 s, p95 463 s). The full sample with thinking was out of budget, so the main arm q14 runs with thinking off, and q14t adds thinking on for 12 L2 instances per n, so that the choice does not bias the test against the LLM.

**Heat.** The T4 is passively cooled. After the first hours it stayed at 83–85 C with the clock near 300 MHz out of 1590, which is why step 0 alone (25.3 s, 5.6 tokens/s) is slower than the earlier run under LKH-3 load on a cold card (14.5 s, about 15 tokens/s). This comes from the server, not from load of mine. The time actually sustained over hours is the "time of the run" column in part B.

**The step-0 times above were measured with no LKH-3 of mine running, on a GPU already throttled by temperature.** An earlier run with LKH-3 in parallel, on a cold GPU, is kept in step0_under_load.json and is not used as a latency. The latencies measured on a cooled GPU are in the next table.

Measured alone, last in the run, with no LKH of mine running (`lkh_running` was checked before every call); each arm started at 70 C or less, and the temperature and clock were read after every call.

| arm | calls | mean (s) | median (s) | p95 (s) | tokens/s | load average at the calls | calls with LKH running | temperature (C) | clock (MHz, min–max) |
|---|---|---|---|---|---|---|---|---|---|
| q7 qwen2.5:7b | 30 | 24.16 | 9.76 | 134.56 | 17.4 | 2.62–7.24 | 0 | 68–84 | 300–1380 |
| q14 qwen3:14b | 30 | 10.00 | 8.56 | 21.68 | 14.0 | 2.18–6.02 | 0 | 40–83 | 600–1560 |
| g12 gemma3:12b | 30 | 26.47 | 15.97 | 31.34 | 9.8 | 2.85–7.30 | 0 | 69–84 | 585–1440 |

GPU heat: 1246 samples (one every 30 s) with the GPU in use, from 19:31:38 on; at most 85 C; clock under 500 MHz (maximum 1590) in 93 % of them, median 315 MHz; software thermal slowdown flagged in 44 %.

GPU guard: 131 checks; free memory at the checks 5340–14913 MiB; 0 pauses, 0 min in total.

## Step 1: the data set

| n | level | instances | label rules | primitive constraints per instance (mean) | prompt tokens (mean, q14) |
|---|---|---|---|---|---|
| 200 | L0 | 176 | 349 | 2.0 | 1805 |
| 200 | L1 | 150 | 373 | 16.0 | 2661 |
| 200 | L2 | 150 | 373 | 16.0 | 2765 |
| 1000 | L0 | 173 | 372 | 2.2 | 2694 |
| 1000 | L1 | 150 | 374 | 35.6 | 4330 |
| 1000 | L2 | 60 | 146 | 34.1 | 4442 |

### Audit of the L2 labels

Sample: 60 L2 texts drawn at random (30 per n, numpy default_rng([20261031, 60]), from the first 60 test instances, the ones the arms use; read after p1 was frozen). I read each one against its formal label.
**Label error rate: 17 / 60 = 28.3 % (95 % Wilson CI 18.5–40.8 %).**

| n | instance | what is wrong |
|---|---|---|
| 200 | 2 | Rua Larga 12: 'not acceptable to skip it without first connecting with Beco Escuro 23' turns the exception into an obligation |
| 200 | 12 | 'not acceptable to skip the east zone unless necessary' says the opposite of the rule, and the text contradicts itself later |
| 200 | 15 | Rua Alta 13 'can only be serviced in conjunction with Beco Escuro 21' makes the allowed link a required one |
| 200 | 34 | the forbidden-edge rule is extended to Avenida Norte 23, which comes from the distractor |
| 200 | 38 | Rua Larga 16 'only serviced in conjunction with' makes the link required; 'all businesses in Campolide' widens the window from shops |
| 200 | 39 | the window is attached to the excepted place Beco Escuro 17 instead of to the houses of Graca |
| 200 | 47 | 'not acceptable to skip the west zone' says the opposite of the soft rule |
| 200 | 53 | the window is attached to the excepted place Rua Azul 11 instead of to the other offices of Restelo |
| 200 | 59 | 'no house next to an office' became 'do not skip a house next to an office': the forbidden-edge rule is lost |
| 1000 | 7 | the excepted place Rua Alta 4 is said to follow the hospitals too, so the exception is lost |
| 1000 | 15 | Largo do Carmo 25 'can only be serviced in connection with' Avenida Central 29 makes the allowed link a required one |
| 1000 | 21 | the window is attached to the excepted place Largo do Carmo 29 instead of to the other pharmacies of Belem |
| 1000 | 24 | 'the places nearest Rua da Paz 24' drops the count (3) |
| 1000 | 26 | after the distractor the text says 'please maintain that order', which reads as a new order rule |
| 1000 | 42 | 'hospital' is lost: the link of Rua da Paz 28 is limited against every place, and it reads as required |
| 1000 | 49 | 'everything else in Campolide' widens the window from houses to all places |
| 1000 | 56 | 'the listed place closest to it' became 'your next delivery point' |

## Step 2B: who formalises best

All texts (no exclusion):

| n | level | arm | instances | rules exact % [95 % CI] | instances exact % [95 % CI] | invalid JSON after retry % | mean time alone (s) | mean time of the run (s), * = under LKH load |
|---|---|---|---|---|---|---|---|---|
| 200 | L0 | p0 old keywords | 30 | 98.2 [90.6, 99.7] | 96.7 [83.3, 99.4] | 0.0 | ~0 | 0.00 |
| 200 | L0 | p1 DSL parser | 30 | 98.2 [90.6, 99.7] | 96.7 [83.3, 99.4] | 0.0 | ~0 | 0.00 |
| 200 | L0 | q7 qwen2.5:7b | 20 | 73.7 [58.0, 85.0] | 35.0 [18.1, 56.7] | 0.0 | 3.06 | 5.35 |
| 200 | L0 | q14 qwen3:14b | 30 | 94.6 [85.4, 98.2] | 90.0 [74.4, 96.5] | 0.0 | 3.60 | 8.88 |
| 200 | L0 | g12 gemma3:12b | 20 | 81.6 [66.6, 90.8] | 70.0 [48.1, 85.5] | 0.0 | 5.53 | 11.25 |
| 200 | L1 | p0 old keywords | 60 | 26.4 [19.9, 34.0] | 1.7 [0.3, 8.9] | 0.0 | ~0 | 0.00 |
| 200 | L1 | p1 DSL parser | 60 | 100.0 [97.5, 100.0] | 100.0 [94.0, 100.0] | 0.0 | ~0 | 0.00 |
| 200 | L1 | q7 qwen2.5:7b | 40 | 48.5 [38.8, 58.3] | 17.5 [8.7, 32.0] | 0.0 | 7.44 | 13.54 |
| 200 | L1 | q14 qwen3:14b | 60 | 80.4 [73.3, 86.0] | 61.7 [49.0, 72.9] | 0.0 | 8.40 | 22.83 |
| 200 | L1 | g12 gemma3:12b | 20 | 60.4 [46.3, 73.0] | 25.0 [11.2, 46.9] | 0.0 | 11.95 | 20.73 |
| 200 | L2 | p0 old keywords | 60 | 22.3 [16.3, 29.7] | 1.7 [0.3, 8.9] | 0.0 | ~0 | 0.00 |
| 200 | L2 | p1 DSL parser | 60 | 54.7 [46.7, 62.5] | 18.3 [10.6, 29.9] | 0.0 | ~0 | 0.00 |
| 200 | L2 | q7 qwen2.5:7b | 40 | 39.2 [30.1, 49.1] | 10.0 [4.0, 23.1] | 0.0 | 35.18 | 22.22 |
| 200 | L2 | q14 qwen3:14b | 60 | 70.9 [63.2, 77.7] | 35.0 [24.2, 47.6] | 0.0 | 9.14 | 24.88 |
| 200 | L2 | q14t qwen3:14b, thinking on | 12 | 69.0 [50.8, 82.7] | 50.0 [25.4, 74.6] | 8.3 | — | 421.32 |
| 200 | L2 | g12 gemma3:12b | 20 | 52.1 [38.3, 65.5] | 10.0 [2.8, 30.1] | 0.0 | 16.08 | 20.94 |
| 1000 | L0 | p0 old keywords | 30 | 97.0 [89.8, 99.2] | 93.3 [78.7, 98.2] | 0.0 | ~0 | 0.00 |
| 1000 | L0 | p1 DSL parser | 30 | 97.0 [89.8, 99.2] | 93.3 [78.7, 98.2] | 0.0 | ~0 | 0.00 |
| 1000 | L0 | q7 qwen2.5:7b | 20 | 58.7 [44.3, 71.7] | 35.0 [18.1, 56.7] | 0.0 | 9.13 | 7.48 |
| 1000 | L0 | q14 qwen3:14b | 30 | 82.1 [71.3, 89.4] | 70.0 [52.1, 83.3] | 0.0 | 6.99 | 15.24 |
| 1000 | L0 | g12 gemma3:12b | 20 | 73.9 [59.7, 84.4] | 45.0 [25.8, 65.8] | 0.0 | 75.42 | 18.55 |
| 1000 | L1 | p0 old keywords | 60 | 32.9 [25.8, 40.9] | 1.7 [0.3, 8.9] | 0.0 | ~0 | 0.00 |
| 1000 | L1 | p1 DSL parser | 60 | 100.0 [97.4, 100.0] | 100.0 [94.0, 100.0] | 0.0 | ~0 | 0.01 |
| 1000 | L1 | q7 qwen2.5:7b | 40 | 44.3 [34.8, 54.2] | 17.5 [8.7, 32.0] | 2.5 | 43.38 | 28.17 |
| 1000 | L1 | q14 qwen3:14b | 60 | 75.3 [67.8, 81.6] | 50.0 [37.7, 62.3] | 0.0 | 12.26 | 28.52 |
| 1000 | L1 | g12 gemma3:12b | 20 | 52.1 [38.3, 65.5] | 20.0 [8.1, 41.6] | 10.0 | 24.07 | 34.03 |
| 1000 | L2 | p0 old keywords | 60 | 30.8 [23.9, 38.7] | 1.7 [0.3, 8.9] | 0.0 | ~0 | 0.00 |
| 1000 | L2 | p1 DSL parser | 60 | 60.3 [52.2, 67.9] | 23.3 [14.4, 35.4] | 0.0 | ~0 | 0.00 |
| 1000 | L2 | q7 qwen2.5:7b | 40 | 27.8 [19.9, 37.5] | 5.0 [1.4, 16.5] | 5.0 | 46.76 | 35.61 |
| 1000 | L2 | q14 qwen3:14b | 60 | 59.6 [51.5, 67.2] | 13.3 [6.9, 24.2] | 0.0 | 19.61 | 29.77 |
| 1000 | L2 | q14t qwen3:14b, thinking on | 12 | 57.1 [39.1, 73.5] | 41.7 [19.3, 68.0] | 25.0 | — | 455.47 |
| 1000 | L2 | g12 gemma3:12b | 20 | 37.5 [25.2, 51.6] | 5.0 [0.9, 23.6] | 5.0 | 25.75 | 43.65 |

Without the L2 texts the audit found wrong:

| n | level | arm | instances | rules exact % [95 % CI] | instances exact % [95 % CI] | invalid JSON after retry % | mean time alone (s) | mean time of the run (s), * = under LKH load |
|---|---|---|---|---|---|---|---|---|
| 200 | L2 | p0 old keywords | 51 | 20.8 [14.6, 28.7] | 0.0 [0.0, 7.0] | 0.0 | ~0 | 0.00 |
| 200 | L2 | p1 DSL parser | 51 | 55.2 [46.5, 63.6] | 19.6 [11.0, 32.5] | 0.0 | ~0 | 0.00 |
| 200 | L2 | q7 qwen2.5:7b | 34 | 38.6 [28.8, 49.3] | 8.8 [3.0, 23.0] | 0.0 | 35.18 | 23.53 |
| 200 | L2 | q14 qwen3:14b | 51 | 72.8 [64.4, 79.8] | 37.3 [25.3, 51.0] | 0.0 | 9.14 | 25.04 |
| 200 | L2 | q14t qwen3:14b, thinking on | 11 | 69.2 [50.0, 83.5] | 54.5 [28.0, 78.7] | 9.1 | — | 428.13 |
| 200 | L2 | g12 gemma3:12b | 17 | 48.8 [34.3, 63.5] | 5.9 [1.0, 27.0] | 0.0 | 16.08 | 21.29 |
| 1000 | L2 | p0 old keywords | 52 | 31.2 [23.7, 39.8] | 1.9 [0.3, 10.1] | 0.0 | ~0 | 0.00 |
| 1000 | L2 | p1 DSL parser | 52 | 61.6 [52.8, 69.7] | 26.9 [16.8, 40.3] | 0.0 | ~0 | 0.00 |
| 1000 | L2 | q7 qwen2.5:7b | 35 | 26.2 [18.0, 36.5] | 5.7 [1.6, 18.6] | 5.7 | 46.76 | 33.47 |
| 1000 | L2 | q14 qwen3:14b | 52 | 60.8 [52.0, 68.9] | 13.5 [6.7, 25.3] | 0.0 | 19.61 | 29.45 |
| 1000 | L2 | q14t qwen3:14b, thinking on | 11 | 52.0 [33.5, 70.0] | 36.4 [15.2, 64.6] | 27.3 | — | 480.78 |
| 1000 | L2 | g12 gemma3:12b | 18 | 37.2 [24.4, 52.1] | 5.6 [1.0, 25.8] | 5.6 | 25.75 | 45.42 |

McNemar, q14 against p1 (rule level, paired, all texts):

| n | level | q14 right, p1 wrong | q14 wrong, p1 right | exact McNemar p |
|---|---|---|---|---|
| 200 | L0 | 0 | 2 | 0.5 |
| 200 | L1 | 0 | 29 | 3.73e-09 |
| 200 | L2 | 33 | 9 | 0.000272 |
| 1000 | L0 | 2 | 12 | 0.0129 |
| 1000 | L1 | 0 | 36 | 2.91e-11 |
| 1000 | L2 | 21 | 22 | 1 |

McNemar, q14 against q7:

| n | level | q14 right, q7 wrong | q14 wrong, q7 right | exact McNemar p |
|---|---|---|---|---|
| 200 | L0 | 7 | 0 | 0.0156 |
| 200 | L1 | 33 | 5 | 4.26e-06 |
| 200 | L2 | 36 | 4 | 1.86e-07 |
| 1000 | L0 | 12 | 2 | 0.0129 |
| 1000 | L1 | 37 | 5 | 4.43e-07 |
| 1000 | L2 | 35 | 2 | 1.02e-08 |

McNemar, q14 with thinking against p1 and against q14 without (same instances):

| n | level | q14t right, p1 wrong | q14t wrong, p1 right | exact McNemar p |
|---|---|---|---|---|
| 200 | L2 | 5 | 3 | 0.727 |
| 1000 | L2 | 4 | 8 | 0.388 |

| n | level | q14t right, q14 wrong | q14t wrong, q14 right | exact McNemar p |
|---|---|---|---|---|
| 200 | L2 | 2 | 4 | 0.688 |
| 1000 | L2 | 5 | 3 | 0.727 |

Error types:

Counts over both n. The first six columns count label rules missed; the last two count primitive constraints predicted that touch no label rule.

| level | arm | rules | missing | wrong selector | exception lost | inverted | wrong type | wrong times | invented, from a distractor | invented, other |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | p0 old keywords | 123 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| L0 | p1 DSL parser | 123 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| L0 | q7 qwen2.5:7b | 84 | 19 | 4 | 0 | 2 | 4 | 0 | 0 | 33 |
| L0 | q14 qwen3:14b | 123 | 4 | 7 | 0 | 3 | 1 | 0 | 0 | 2 |
| L0 | g12 gemma3:12b | 84 | 6 | 5 | 0 | 4 | 4 | 0 | 0 | 5 |
| L1 | p0 old keywords | 294 | 127 | 37 | 32 | 0 | 11 | 0 | 1 | 18 |
| L1 | p1 DSL parser | 294 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | q7 qwen2.5:7b | 194 | 28 | 55 | 6 | 1 | 14 | 0 | 15 | 641 |
| L1 | q14 qwen3:14b | 294 | 5 | 42 | 13 | 3 | 2 | 0 | 1 | 56 |
| L1 | g12 gemma3:12b | 96 | 13 | 18 | 6 | 1 | 4 | 0 | 1 | 25 |
| L2 | p0 old keywords | 294 | 149 | 33 | 11 | 0 | 23 | 0 | 15 | 15 |
| L2 | p1 DSL parser | 294 | 57 | 33 | 4 | 1 | 30 | 0 | 0 | 45 |
| L2 | q7 qwen2.5:7b | 194 | 40 | 57 | 2 | 2 | 28 | 0 | 56 | 1418 |
| L2 | q14 qwen3:14b | 294 | 11 | 62 | 10 | 1 | 18 | 0 | 20 | 86 |
| L2 | q14t qwen3:14b, thinking on | 57 | 12 | 6 | 1 | 0 | 2 | 0 | 1 | 2 |
| L2 | g12 gemma3:12b | 96 | 19 | 21 | 4 | 0 | 9 | 0 | 5 | 62 |

Rules exact (%) by template on L2, both n (rule count in brackets on the first column).

| shape | p0 old keywords | p1 DSL parser | q7 qwen2.5:7b | q14 qwen3:14b | q14t qwen3:14b, thinking on | g12 gemma3:12b |
|---|---|---|---|---|---|---|
| forb_area (23) | 0 | 43 | 12 | 57 | 25 | 0 |
| forb_except (22) | 0 | 0 | 0 | 14 | 33 | 0 |
| forb_group (19) | 0 | 58 | 50 | 89 | 100 | 83 |
| mand_names (30) | 90 | 87 | 33 | 77 | 62 | 46 |
| mand_nearest (36) | 0 | 61 | 27 | 47 | 25 | 0 |
| prec_area (23) | 0 | 30 | 7 | 52 | 40 | 12 |
| prec_except (9) | 0 | 56 | 0 | 22 | 50 | 0 |
| prec_group (8) | 0 | 50 | 14 | 88 | 100 | 100 |
| prec_nearest (18) | 0 | 17 | 0 | 22 | 50 | 83 |
| soft (57) | 89 | 86 | 97 | 100 | 86 | 85 |
| tw_area (14) | 0 | 71 | 0 | 100 | 100 | 83 |
| tw_except (10) | 0 | 50 | 29 | 70 | — | — |
| tw_nearest (10) | 0 | 40 | 0 | 60 | — | — |
| tw_union (15) | 0 | 87 | 80 | 67 | 100 | 33 |

## Step 2C: what it costs in the solver

A rule set equal to the label reuses the reference route, so its LKH-3 time counts as 0 here: the oracle and the exact answers show no solver time, while a real run would pay about 31 s at n = 200 and 161 s at n = 1000. The time column therefore compares the extra cost of formalising, not a full run. LLM time: the latency measured alone; for q14t, its time in the run.

| n | level | arm | routes | valid routes % [95 % CI] | hard rules broken / route | length gap vs oracle (%) | cost gap vs oracle (%) | rules rejected by the code check / route | LKH-3 no route (%) | mean time LLM + LKH (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | L0 | o oracle | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L0 | p0 old keywords | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.02 | +0.00 | 0.00 | 0.0 | 0.3 |
| 200 | L0 | p1 DSL parser | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.02 | +0.00 | 0.00 | 0.0 | 0.3 |
| 200 | L0 | q7 qwen2.5:7b | 20 | 75.0 [53.1, 88.8] | 0.40 | +0.44 | +0.35 | 0.25 | 0.0 | 12.1 |
| 200 | L0 | q14 qwen3:14b | 30 | 90.0 [74.4, 96.5] | 0.10 | +0.17 | +0.14 | 0.03 | 0.0 | 4.8 |
| 200 | L0 | g12 gemma3:12b | 20 | 85.0 [64.0, 94.8] | 0.15 | +0.70 | +0.70 | 0.00 | 0.0 | 10.4 |
| 200 | L1 | o oracle | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L1 | p0 old keywords | 60 | 45.0 [33.1, 57.5] | 3.52 | -0.72 | -0.68 | 0.00 | 0.0 | 5.4 |
| 200 | L1 | p1 DSL parser | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L1 | q7 qwen2.5:7b | 40 | 55.0 [39.8, 69.3] | 1.68 | -0.19 | -0.15 | 0.03 | 0.0 | 28.3 |
| 200 | L1 | q14 qwen3:14b | 60 | 88.3 [77.8, 94.2] | 0.20 | +0.18 | +0.18 | 0.02 | 0.0 | 15.3 |
| 200 | L1 | g12 gemma3:12b | 20 | 80.0 [58.4, 91.9] | 0.30 | +0.34 | +0.29 | 0.00 | 0.0 | 22.4 |
| 200 | L2 | o oracle | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L2 | p0 old keywords | 60 | 41.7 [30.1, 54.3] | 3.82 | -0.34 | -0.26 | 0.08 | 0.0 | 4.5 |
| 200 | L2 | p1 DSL parser | 60 | 63.3 [50.7, 74.4] | 1.95 | -0.25 | -0.16 | 0.08 | 1.7 | 16.8 |
| 200 | L2 | q7 qwen2.5:7b | 40 | 50.0 [35.2, 64.8] | 2.77 | +0.46 | +0.50 | 0.35 | 0.0 | 59.4 |
| 200 | L2 | q14 qwen3:14b | 60 | 80.0 [68.2, 88.2] | 0.33 | +0.59 | +0.61 | 0.20 | 0.0 | 28.9 |
| 200 | L2 | q14t qwen3:14b, thinking on | 12 | 58.3 [32.0, 80.7] | 0.42 | +0.78 | +0.78 | 0.17 | 0.0 | 425.2 |
| 200 | L2 | g12 gemma3:12b | 20 | 60.0 [38.7, 78.1] | 0.55 | +2.35 | +2.38 | 0.30 | 0.0 | 29.5 |
| 1000 | L0 | o oracle | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L0 | p0 old keywords | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.01 | -0.00 | 0.00 | 0.0 | 4.4 |
| 1000 | L0 | p1 DSL parser | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.01 | -0.00 | 0.00 | 0.0 | 4.4 |
| 1000 | L0 | q7 qwen2.5:7b | 20 | 50.0 [29.9, 70.1] | 0.80 | +0.85 | +0.86 | 0.10 | 0.0 | 108.9 |
| 1000 | L0 | q14 qwen3:14b | 30 | 73.3 [55.6, 85.8] | 0.33 | +0.28 | +0.31 | 0.00 | 0.0 | 72.1 |
| 1000 | L0 | g12 gemma3:12b | 20 | 45.0 [25.8, 65.8] | 0.60 | +0.20 | +0.23 | 0.05 | 0.0 | 152.4 |
| 1000 | L1 | o oracle | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L1 | p0 old keywords | 30 | 33.3 [19.2, 51.2] | 12.73 | -1.10 | -1.06 | 0.00 | 0.0 | 47.8 |
| 1000 | L1 | p1 DSL parser | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L1 | q7 qwen2.5:7b | 30 | 33.3 [19.2, 51.2] | 11.13 | +0.52 | +0.59 | 0.23 | 0.0 | 128.3 |
| 1000 | L1 | q14 qwen3:14b | 30 | 73.3 [55.6, 85.8] | 0.30 | +0.19 | +0.20 | 0.00 | 3.3 | 96.2 |
| 1000 | L1 | g12 gemma3:12b | 20 | 50.0 [29.9, 70.1] | 6.45 | -0.06 | -0.04 | 0.05 | 0.0 | 130.3 |
| 1000 | L2 | o oracle | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L2 | p0 old keywords | 30 | 36.7 [21.9, 54.5] | 13.30 | -1.24 | -1.21 | 0.03 | 0.0 | 68.5 |
| 1000 | L2 | p1 DSL parser | 30 | 56.7 [39.2, 72.6] | 5.73 | -0.50 | -0.50 | 0.03 | 0.0 | 70.7 |
| 1000 | L2 | q7 qwen2.5:7b | 30 | 33.3 [19.2, 51.2] | 12.67 | +0.79 | +0.80 | 0.27 | 0.0 | 147.4 |
| 1000 | L2 | q14 qwen3:14b | 30 | 60.0 [42.3, 75.4] | 0.47 | +0.39 | +0.42 | 0.13 | 0.0 | 169.1 |
| 1000 | L2 | q14t qwen3:14b, thinking on | 12 | 66.7 [39.1, 86.2] | 2.42 | -0.29 | -0.23 | 0.00 | 0.0 | 520.0 |
| 1000 | L2 | g12 gemma3:12b | 20 | 30.0 [14.5, 51.9] | 2.10 | -0.03 | +0.04 | 0.20 | 5.0 | 171.9 |

Valid routes, q14 against p1, paired:

| n | level | routes | q14 valid, p1 not | q14 not, p1 valid | exact McNemar p |
|---|---|---|---|---|---|
| 200 | L0 | 30 | 0 | 2 | 0.5 |
| 200 | L1 | 60 | 0 | 7 | 0.0156 |
| 200 | L2 | 60 | 16 | 6 | 0.0525 |
| 1000 | L0 | 30 | 1 | 8 | 0.0391 |
| 1000 | L1 | 30 | 0 | 8 | 0.00781 |
| 1000 | L2 | 30 | 7 | 6 | 1 |

Without the L2 texts the audit found wrong (L2 rows only are affected):

| n | level | arm | routes | valid routes % [95 % CI] | hard rules broken / route | length gap vs oracle (%) | cost gap vs oracle (%) | rules rejected by the code check / route | LKH-3 no route (%) | mean time LLM + LKH (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | L0 | o oracle | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L0 | p0 old keywords | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.02 | +0.00 | 0.00 | 0.0 | 0.3 |
| 200 | L0 | p1 DSL parser | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.02 | +0.00 | 0.00 | 0.0 | 0.3 |
| 200 | L0 | q7 qwen2.5:7b | 20 | 75.0 [53.1, 88.8] | 0.40 | +0.44 | +0.35 | 0.25 | 0.0 | 12.1 |
| 200 | L0 | q14 qwen3:14b | 30 | 90.0 [74.4, 96.5] | 0.10 | +0.17 | +0.14 | 0.03 | 0.0 | 4.8 |
| 200 | L0 | g12 gemma3:12b | 20 | 85.0 [64.0, 94.8] | 0.15 | +0.70 | +0.70 | 0.00 | 0.0 | 10.4 |
| 200 | L1 | o oracle | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L1 | p0 old keywords | 60 | 45.0 [33.1, 57.5] | 3.52 | -0.72 | -0.68 | 0.00 | 0.0 | 5.4 |
| 200 | L1 | p1 DSL parser | 60 | 100.0 [94.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L1 | q7 qwen2.5:7b | 40 | 55.0 [39.8, 69.3] | 1.68 | -0.19 | -0.15 | 0.03 | 0.0 | 28.3 |
| 200 | L1 | q14 qwen3:14b | 60 | 88.3 [77.8, 94.2] | 0.20 | +0.18 | +0.18 | 0.02 | 0.0 | 15.3 |
| 200 | L1 | g12 gemma3:12b | 20 | 80.0 [58.4, 91.9] | 0.30 | +0.34 | +0.29 | 0.00 | 0.0 | 22.4 |
| 200 | L2 | o oracle | 51 | 100.0 [93.0, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 200 | L2 | p0 old keywords | 51 | 41.2 [28.8, 54.8] | 4.14 | -0.43 | -0.35 | 0.08 | 0.0 | 4.0 |
| 200 | L2 | p1 DSL parser | 51 | 60.8 [47.1, 73.0] | 2.24 | -0.29 | -0.20 | 0.10 | 2.0 | 16.5 |
| 200 | L2 | q7 qwen2.5:7b | 34 | 47.1 [31.5, 63.3] | 2.97 | -0.09 | -0.03 | 0.32 | 0.0 | 57.8 |
| 200 | L2 | q14 qwen3:14b | 51 | 80.4 [67.5, 89.0] | 0.33 | +0.28 | +0.29 | 0.16 | 0.0 | 27.8 |
| 200 | L2 | q14t qwen3:14b, thinking on | 11 | 63.6 [35.4, 84.8] | 0.36 | +0.36 | +0.36 | 0.00 | 0.0 | 431.1 |
| 200 | L2 | g12 gemma3:12b | 17 | 58.8 [36.0, 78.4] | 0.59 | +2.14 | +2.17 | 0.24 | 0.0 | 31.1 |
| 1000 | L0 | o oracle | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L0 | p0 old keywords | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.01 | -0.00 | 0.00 | 0.0 | 4.4 |
| 1000 | L0 | p1 DSL parser | 30 | 96.7 [83.3, 99.4] | 0.03 | -0.01 | -0.00 | 0.00 | 0.0 | 4.4 |
| 1000 | L0 | q7 qwen2.5:7b | 20 | 50.0 [29.9, 70.1] | 0.80 | +0.85 | +0.86 | 0.10 | 0.0 | 108.9 |
| 1000 | L0 | q14 qwen3:14b | 30 | 73.3 [55.6, 85.8] | 0.33 | +0.28 | +0.31 | 0.00 | 0.0 | 72.1 |
| 1000 | L0 | g12 gemma3:12b | 20 | 45.0 [25.8, 65.8] | 0.60 | +0.20 | +0.23 | 0.05 | 0.0 | 152.4 |
| 1000 | L1 | o oracle | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L1 | p0 old keywords | 30 | 33.3 [19.2, 51.2] | 12.73 | -1.10 | -1.06 | 0.00 | 0.0 | 47.8 |
| 1000 | L1 | p1 DSL parser | 30 | 100.0 [88.6, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L1 | q7 qwen2.5:7b | 30 | 33.3 [19.2, 51.2] | 11.13 | +0.52 | +0.59 | 0.23 | 0.0 | 128.3 |
| 1000 | L1 | q14 qwen3:14b | 30 | 73.3 [55.6, 85.8] | 0.30 | +0.19 | +0.20 | 0.00 | 3.3 | 96.2 |
| 1000 | L1 | g12 gemma3:12b | 20 | 50.0 [29.9, 70.1] | 6.45 | -0.06 | -0.04 | 0.05 | 0.0 | 130.3 |
| 1000 | L2 | o oracle | 25 | 100.0 [86.7, 100.0] | 0.00 | +0.00 | +0.00 | 0.00 | 0.0 | 0.0 |
| 1000 | L2 | p0 old keywords | 25 | 40.0 [23.4, 59.3] | 13.84 | -1.33 | -1.32 | 0.00 | 0.0 | 75.6 |
| 1000 | L2 | p1 DSL parser | 25 | 60.0 [40.7, 76.6] | 4.80 | -0.52 | -0.54 | 0.04 | 0.0 | 78.2 |
| 1000 | L2 | q7 qwen2.5:7b | 25 | 28.0 [14.3, 47.6] | 13.56 | +0.80 | +0.79 | 0.24 | 0.0 | 151.2 |
| 1000 | L2 | q14 qwen3:14b | 25 | 56.0 [37.1, 73.3] | 0.52 | +0.32 | +0.37 | 0.12 | 0.0 | 187.2 |
| 1000 | L2 | q14t qwen3:14b, thinking on | 11 | 63.6 [35.4, 84.8] | 2.64 | -0.31 | -0.25 | 0.00 | 0.0 | 551.2 |
| 1000 | L2 | g12 gemma3:12b | 18 | 27.8 [12.5, 50.9] | 2.28 | -0.15 | -0.06 | 0.17 | 5.6 | 179.7 |

Rules rejected by the code check, by reason (all levels, both n):

| arm | a customer with more than 2 mandatory edges | an edge both mandatory and forbidden | impossible window (closes before the customer can be reached) | mandatory edges close a sub-tour | precedence cycle | precedence with a time window (LKH-3 cannot take both) | time window with a precedence (LKH-3 cannot take both) | two different windows for one customer |
|---|---|---|---|---|---|---|---|---|
| p0 old keywords | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| p1 DSL parser | 4 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| q7 qwen2.5:7b | 16 | 9 | 3 | 1 | 5 | 3 | 0 | 0 |
| q14 qwen3:14b | 3 | 0 | 0 | 0 | 1 | 4 | 9 | 1 |
| q14t qwen3:14b, thinking on | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| g12 gemma3:12b | 3 | 3 | 1 | 1 | 0 | 0 | 4 | 0 |

## Step 3: the larger model on the older tasks

The 7B column is computed on the same cases from the runs of steps 0b and 2.

| task | metric | 7B | 14B |
|---|---|---|---|
| planner V0 | correct / 40 | 0 | 25 |
| planner V0 | mean latency (s) | 1.17 | 10.48 |
| planner V3 | correct / 40 | 35 | 35 |
| planner V3 | mean latency (s) | 3.43 | 10.19 |
| formalisation text_v2b, n = 200 | rules exact % (101 rules) | 86.1 [78.1, 91.6] | 98.0 [93.1, 99.5] |
| formalisation text_v2b, n = 200 | instances exact % (51) | 47.1 [34.1, 60.5] | 94.1 [84.1, 98.0] |
| formalisation text_v2b, n = 200 | mean latency (s) | 6.91 | 12.68 |
| formalisation text_v2b, n = 1000 | rules exact % (100 rules) | 64.0 [54.2, 72.7] | 86.0 [77.9, 91.5] |
| formalisation text_v2b, n = 1000 | instances exact % (48) | 29.2 [18.2, 43.2] | 75.0 [61.2, 85.1] |
| formalisation text_v2b, n = 1000 | mean latency (s) | 8.59 | 16.63 |
| verifier plain (50 routes) | precision % | 29.0 | 33.3 |
| verifier plain (50 routes) | recall % | 34.6 | 76.9 |
| verifier plain (50 routes) | route verdict right % | 40.0 | 40.0 |
| verifier plain (50 routes) | false alarms on oracle routes % | 50.0 | 70.8 |
| verifier plain (50 routes) | injected error found % | 34.6 | 76.9 |
| verifier plain | mean latency (s) | 10.36 | 28.96 |
| verifier report (50 routes) | precision % | 54.2 | 81.2 |
| verifier report (50 routes) | recall % | 100.0 | 100.0 |
| verifier report (50 routes) | route verdict right % | 68.0 | 88.0 |
| verifier report (50 routes) | false alarms on oracle routes % | 62.5 | 20.8 |
| verifier report (50 routes) | injected error found % | 100.0 | 100.0 |
| verifier report | mean latency (s) | 12.63 | 21.34 |

qwen3:14b is clearly better on the planner V0 (25/40 against 0/40), on formalisation (98 % and 86 % of the rules against 86 % and 64 %) and on the verifier with the code report (88 % of the routes right against 68 %). It ties on planner V3 (35/40). On the plain verifier it finds more of the injected errors (77 % against 35 %) but raises more false alarms (71 % against 50 %), and the route verdict stays at 40 %, so it does not replace the code check. The latency is 2–3 times the 7B one, and about 9 times on planner V0. These 14B times are not comparable with the 7B ones in detail: the planner and formalisation calls ran with LKH-3 of part C in parallel on a hot GPU, the verifier calls ran alone but hot, and the 7B times come from earlier runs on a cooler card.

## Hybrid arms: p1 first, q14 when p1 leaves rules uncovered

Computed offline from the logged answers: no model and no LKH-3 ran. h1 sends a whole text to q14 if p1 leaves a constraint sentence without a valid rule, or has fewer valid rules than constraint sentences; the check uses only the text and the output of p1 (`src/tsp_gnn/open_hybrid.py`). The stock phrase p1 deletes counts as a constraint sentence when it has an order, adjacency, time or zone word. The trigger was fixed on the held-out development texts before the test texts were scored, and p1 stayed frozen. h2 joins the answers of p1 and q14, p1 first, and is scored after the code check, which keeps the first rule on a contradiction.

**h1 keeps what p1 does well and adds part of what q14 does well: it sends no L0 or L1 text to the LLM, so it matches p1 there, and on L2 at n = 200 it gets 64.2 % of the rules against 54.7 % for p1 (p = 0.0066) but stays below q14 at 70.9 % (p = 0.031), while at n = 1000 its 63.0 % ties with both (p ≥ 0.36) at about half of the LLM calls.** The limit is the trigger: 43 of the 95 L2 texts p1 gets wrong are kept with p1, because they have the right number of rules with wrong content. h2 does not beat q14 on exact rules or instances on L2, and its route rate stays open: 68 of its rule sets have no route in the logs and were not run, so its bounds are wide, and the rates over known routes favour simple sets. Times: "LLM s, alone" is the share sent times the mean of q14 measured alone in that cell (5 calls), "LLM s, run" is the logged time on a hot GPU, and the h2 solver time covers known routes only, so it is understated. These tests are not corrected for multiple comparisons.

All texts:

| n | level | arm | texts | rules exact % [95 % CI] | instances exact % [95 % CI] | valid routes % [95 % CI] (routes) | sent to the LLM % | LLM s, alone | LLM s, run | LLM + LKH s |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | L0 | p1 | 30 | 98.2 [90.6, 99.7] | 96.7 [83.3, 99.4] | 96.7 [83.3, 99.4] (30) | 0.0 | 0.0 | 0.0 | 0.3 |
| 200 | L0 | q14 | 30 | 94.6 [85.4, 98.2] | 90.0 [74.4, 96.5] | 90.0 [74.4, 96.5] (30) | 100.0 | 3.6 | 8.9 | 4.8 |
| 200 | L0 | h1 | 30 | 98.2 [90.6, 99.7] | 96.7 [83.3, 99.4] | 96.7 [83.3, 99.4] (30) | 0.0 | 0.0 | 0.0 | 0.3 |
| 200 | L0 | h2 | 30 | 98.2 [90.6, 99.7] | 93.3 [78.7, 98.2] | 100.0 [87.9, 100.0] (28); bounds 93.3–100.0 over 30 | 100.0 | 3.6 | 8.9 | 3.6 |
| 200 | L1 | p1 | 60 | 100.0 [97.5, 100.0] | 100.0 [94.0, 100.0] | 100.0 [94.0, 100.0] (60) | 0.0 | 0.0 | 0.0 | 0.0 |
| 200 | L1 | q14 | 60 | 80.4 [73.3, 86.0] | 61.7 [49.0, 72.9] | 88.3 [77.8, 94.2] (60) | 100.0 | 8.4 | 22.8 | 15.3 |
| 200 | L1 | h1 | 60 | 100.0 [97.5, 100.0] | 100.0 [94.0, 100.0] | 100.0 [94.0, 100.0] (60) | 0.0 | 0.0 | 0.0 | 0.0 |
| 200 | L1 | h2 | 60 | 85.8 [79.3, 90.5] | 66.7 [54.1, 77.3] | 100.0 [92.3, 100.0] (46); bounds 76.7–100.0 over 60 | 100.0 | 8.4 | 22.8 | 11.1 |
| 200 | L2 | p1 | 60 | 54.7 [46.7, 62.5] | 18.3 [10.6, 29.9] | 63.3 [50.7, 74.4] (60) | 0.0 | 0.0 | 0.0 | 16.8 |
| 200 | L2 | q14 | 60 | 70.9 [63.2, 77.7] | 35.0 [24.2, 47.6] | 80.0 [68.2, 88.2] (60) | 100.0 | 9.1 | 24.9 | 28.9 |
| 200 | L2 | h1 | 60 | 64.2 [56.2, 71.5] | 31.7 [21.3, 44.2] | 73.3 [61.0, 82.9] (60) | 48.3 | 4.4 | 11.6 | 25.2 |
| 200 | L2 | h2 | 60 | 70.3 [62.5, 77.0] | 30.0 [19.9, 42.5] | 88.4 [75.5, 94.9] (43); bounds 63.3–91.7 over 60 | 100.0 | 9.1 | 24.9 | 26.3 |
| 1000 | L0 | p1 | 30 | 97.0 [89.8, 99.2] | 93.3 [78.7, 98.2] | 96.7 [83.3, 99.4] (30) | 0.0 | 0.0 | 0.0 | 4.4 |
| 1000 | L0 | q14 | 30 | 82.1 [71.3, 89.4] | 70.0 [52.1, 83.3] | 73.3 [55.6, 85.8] (30) | 100.0 | 7.0 | 15.2 | 72.1 |
| 1000 | L0 | h1 | 30 | 97.0 [89.8, 99.2] | 93.3 [78.7, 98.2] | 96.7 [83.3, 99.4] (30) | 0.0 | 0.0 | 0.0 | 4.4 |
| 1000 | L0 | h2 | 30 | 89.6 [80.0, 94.8] | 76.7 [59.1, 88.2] | 95.8 [79.8, 99.3] (24); bounds 76.7–96.7 over 30 | 100.0 | 7.0 | 15.2 | 7.2 |
| 1000 | L1 | p1 | 60 | 100.0 [97.4, 100.0] | 100.0 [94.0, 100.0] | 100.0 [88.6, 100.0] (30) | 0.0 | 0.0 | 0.0 | 0.0 |
| 1000 | L1 | q14 | 60 | 75.3 [67.8, 81.6] | 50.0 [37.7, 62.3] | 73.3 [55.6, 85.8] (30) | 100.0 | 12.3 | 28.5 | 96.2 |
| 1000 | L1 | h1 | 60 | 100.0 [97.4, 100.0] | 100.0 [94.0, 100.0] | 100.0 [88.6, 100.0] (30) | 0.0 | 0.0 | 0.0 | 0.0 |
| 1000 | L1 | h2 | 60 | 81.5 [74.4, 87.0] | 55.0 [42.5, 66.9] | 100.0 [81.6, 100.0] (17); bounds 56.7–100.0 over 30 | 100.0 | 12.3 | 28.5 | 20.2 |
| 1000 | L2 | p1 | 60 | 60.3 [52.2, 67.9] | 23.3 [14.4, 35.4] | 56.7 [39.2, 72.6] (30) | 0.0 | 0.0 | 0.0 | 70.7 |
| 1000 | L2 | q14 | 60 | 59.6 [51.5, 67.2] | 13.3 [6.9, 24.2] | 60.0 [42.3, 75.4] (30) | 100.0 | 19.6 | 29.8 | 169.1 |
| 1000 | L2 | h1 | 60 | 63.0 [54.9, 70.4] | 16.7 [9.3, 28.0] | 66.7 [48.8, 80.8] (30) | 53.3 | 10.5 | 15.3 | 145.3 |
| 1000 | L2 | h2 | 60 | 59.6 [51.5, 67.2] | 11.7 [5.8, 22.2] | 71.4 [45.4, 88.3] (14); bounds 33.3–86.7 over 30 | 100.0 | 19.6 | 29.8 | 149.6 |

Without the 17 L2 texts the audit found wrong (L2 only):

| n | level | arm | texts | rules exact % [95 % CI] | instances exact % [95 % CI] | valid routes % [95 % CI] (routes) | sent to the LLM % | LLM s, alone | LLM s, run | LLM + LKH s |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | L2 | p1 | 51 | 55.2 [46.5, 63.6] | 19.6 [11.0, 32.5] | 60.8 [47.1, 73.0] (51) | 0.0 | 0.0 | 0.0 | 16.5 |
| 200 | L2 | q14 | 51 | 72.8 [64.4, 79.8] | 37.3 [25.3, 51.0] | 80.4 [67.5, 89.0] (51) | 100.0 | 9.1 | 25.0 | 27.8 |
| 200 | L2 | h1 | 51 | 64.8 [56.1, 72.6] | 33.3 [22.0, 47.0] | 72.5 [59.1, 82.9] (51) | 47.1 | 4.3 | 11.6 | 25.5 |
| 200 | L2 | h2 | 51 | 71.2 [62.7, 78.4] | 31.4 [20.3, 45.0] | 89.2 [75.3, 95.7] (37); bounds 64.7–92.2 over 51 | 100.0 | 9.1 | 25.0 | 24.8 |
| 1000 | L2 | p1 | 52 | 61.6 [52.8, 69.7] | 26.9 [16.8, 40.3] | 60.0 [40.7, 76.6] (25) | 0.0 | 0.0 | 0.0 | 78.2 |
| 1000 | L2 | q14 | 52 | 60.8 [52.0, 68.9] | 13.5 [6.7, 25.3] | 56.0 [37.1, 73.3] (25) | 100.0 | 19.6 | 29.4 | 187.2 |
| 1000 | L2 | h1 | 52 | 65.6 [56.9, 73.4] | 17.3 [9.4, 29.7] | 68.0 [48.4, 82.8] (25) | 55.8 | 10.9 | 15.8 | 166.2 |
| 1000 | L2 | h2 | 52 | 60.0 [51.2, 68.2] | 11.5 [5.4, 23.0] | 72.7 [43.4, 90.3] (11); bounds 32.0–88.0 over 25 | 100.0 | 19.6 | 29.4 | 171.1 |

McNemar, h1 against q14:

| n | level | rules: h1 only / q14 only | p | routes: h1 only / q14 only | p |
|---|---|---|---|---|---|
| 200 | L0 | 2 / 0 | 0.5 | 2 / 0 | 0.5 |
| 200 | L1 | 29 / 0 | 3.73e-09 | 7 / 0 | 0.0156 |
| 200 | L2 | 4 / 14 | 0.0309 | 3 / 7 | 0.344 |
| 1000 | L0 | 12 / 2 | 0.0129 | 8 / 1 | 0.0391 |
| 1000 | L1 | 36 / 0 | 2.91e-11 | 8 / 0 | 0.00781 |
| 1000 | L2 | 12 / 7 | 0.359 | 4 / 2 | 0.688 |

McNemar, h1 against p1:

| n | level | rules: h1 only / p1 only | p | routes: h1 only / p1 only | p |
|---|---|---|---|---|---|
| 200 | L0 | 0 / 0 | 1 | 0 / 0 | 1 |
| 200 | L1 | 0 / 0 | 1 | 0 / 0 | 1 |
| 200 | L2 | 19 / 5 | 0.00661 | 9 / 3 | 0.146 |
| 1000 | L0 | 0 / 0 | 1 | 0 / 0 | 1 |
| 1000 | L1 | 0 / 0 | 1 | 0 / 0 | 1 |
| 1000 | L2 | 14 / 10 | 0.541 | 5 / 2 | 0.453 |

McNemar without the 17 texts, L2, h1 against q14:

| n | level | rules: h1 only / q14 only | p | routes: h1 only / q14 only | p |
|---|---|---|---|---|---|
| 200 | L2 | 3 / 13 | 0.0213 | 3 / 7 | 0.344 |
| 1000 | L2 | 11 / 5 | 0.21 | 4 / 1 | 0.375 |

McNemar without the 17 texts, L2, h1 against p1:

| n | level | rules: h1 only / p1 only | p | routes: h1 only / p1 only | p |
|---|---|---|---|---|---|
| 200 | L2 | 16 / 4 | 0.0118 | 9 / 3 | 0.146 |
| 1000 | L2 | 12 / 7 | 0.359 | 4 / 2 | 0.688 |

What the trigger did (exact instances):

| n | level | texts | sent | p1 exact among the sent | q14 exact among the sent | p1 exact among the kept | p1 wrong and kept |
|---|---|---|---|---|---|---|---|
| 200 | L0 | 30 | 0 | 0 | 0 | 29 | 1 |
| 200 | L1 | 60 | 0 | 0 | 0 | 60 | 0 |
| 200 | L2 | 60 | 29 | 1 | 9 | 10 | 21 |
| 1000 | L0 | 30 | 0 | 0 | 0 | 28 | 2 |
| 1000 | L1 | 60 | 0 | 0 | 0 | 60 | 0 |
| 1000 | L2 | 60 | 32 | 8 | 4 | 6 | 22 |

h2 rule sets with no route in the logs, which would need a new LKH-3 run (not run): 68 (n = 200 L0: 2; n = 200 L1: 14; n = 200 L2: 17; n = 1000 L0: 6; n = 1000 L1: 13; n = 1000 L2: 16).

## Caveats

- **Time budget.** The main run took 9 h 17 min, not the 6 h aimed for (the log clock jumps back an hour at 02:00, when daylight saving ended). The overrun was q14t, whose calls took more than twice the step-0 time on L2, and LKH-3 at n = 1000. The verifier of step 3 crashed on a bug of mine and was rerun alone after the main run, about 35 min more.
- **Sample.** Under the 300 rules per cell asked for. The intervals are wide for q7 and g12, and q14t is a small indication only.
- **L2 label error.** 28 % of the audited texts do not say what the label says, mostly because of the "it is not acceptable to skip" double negation the prompt asked for and because exceptions drift. Only the 17 audited wrong texts can be removed; about as many wrong texts remain in the unaudited part, so accuracy on L2 is a lower bound for every arm. Removing them barely changes the results.
- **Who writes and who reads.** gemma3 writes L2 and qwen reads it; g12 reading its own style is not a clean control. Stock phrases of gemma3 were in the dev paragraphs too, so p1 learned them.
- **p1.** Limited effort; tuned at n = 200 only. L1 is written from the same templates p1 was built on, so its 100 % there is by construction and says nothing about open language. p1 equals p0 on L0 because it reuses that parser.
- **q14t.** Its answers were capped at 2600 tokens to fit 8192 of context, and 8–25 % of them ended with no valid JSON. A larger context could help thinking; the T4 does not have the time or memory for it.
- **Latency table.** 30 calls per arm, 5 per cell, so a cell can hang on one call: runaway answers reach 1024 tokens and then a retry (q7 up to 162 s, g12 313 s). See the median. q14 started colder (40 C) than q7 and g12 (68–69 C), so its 10 s is the best case and the time of the run the sustained case. Cells with mean alone above the run mean come from that noise.
- **Part C.** Valid routes count hard rules only, so a route can be valid with wrong soft rules, and a negative length gap means rules were dropped, not a better route. The time column leaves out LKH-3 for exact rule sets.
- **Other users.** The server load average ran between 2 and 7 at the latency calls. The GPU guard never needed a pause: 0 pauses.
