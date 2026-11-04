# Warehouse, phase C: the LLM as a language interface

1. **Does the LLM beat a fair parser on warehouse requests?** No. On template text p1 gets 200/200 requests right and q14 135/200; on the L2 paragraphs they tie, 84/200 against 81/200 requests exact (p = 0.81), 79.3 % against 81.4 % of rules (p = 0.27) and 60.0 % against 66.0 % of valid episodes (p = 0.17).
2. **Does the hybrid beat both?** Not clearly: it keeps p1's 200/200 on L1 and ties q14 on L2 (41.0 % against 40.5 % of requests exact, 67.5 % against 66.0 % valid, p = 0.63), and its only edge over p1 on L2, 67.5 % against 60.0 % of valid episodes, has p = 0.04 before correcting for three tests and p = 0.21 without the texts the audit found wrong, while exact requests do not improve (41.0 % against 42.0 %).
3. **What does a wrong translation cost?** Mostly validity, little time: on L2, 34 % of q14 episodes and 40 % of p1 episodes break a true rule where the oracle breaks none, those invalid routes look 2–3 % shorter (q14: makespan −2.9 ± 2.9 %, delay −2.3 ± 2.1 %), and the median request costs 0 %, but rules the LLM adds or widens, some of them from label errors in L2 itself, make 6 of 200 requests more than 20 % slower, up to +98 %.

*Note 2026-11-04 (ablation A3, `../../docs/ablations_2026-11-03.md`): with no translation at all (no rule), 2 of 200
episodes (1 %) satisfy the true rules, so nearly all the validity above comes from translating: on L2, p1 is +59 pp
over not translating, q14 +65 pp and h1 +66 pp (McNemar p < 1e-35; q7 +47 pp on its 100 requests). For comparison, on
the TSP of phase 5, 17–30 % of the routes are valid with no translation, and at n = 1000 on L2 q7 and g12 cannot be told
apart from not translating there.*
## Design

The world is written in code. Each shelf block holds one kind of product (normal, fragile, frozen, heavy), and an order has at most one special kind, a named customer and a priority (6 % urgent). Corridors 1–10 run east-west and give access to shelves; corridors 11–20 run north-south, hold no shelves, and only the stretch beside the shelves is closed so that crossings stay open. Stations are A–F. Robot type A carries everything and type B no heavy orders; that is a fact of the warehouse, enforced in every arm, so it is not a rule to translate. Texts are in English, as in phase 5; customer names are Portuguese.

The DSL (`src/warehouse/dsl.py`, on `tsp_gnn.open_dsl`) has the rules `pick_order`, `close`, `no_mix`, `robot_only`, `deadline`, `station`, and the selectors `kind`, `corridor`, `station`, `customer`, `order`, `priority`, `all`, `any_of`, `except`. Scoring is on primitives, so a group and its parts count the same.

| rule | where it goes |
|---|---|
| `pick_order` | sequential ordering problem in LKH-3, exact up to 9 stops |
| `close`, all day | cells leave the grid |
| `close`, with a window | entries forbidden in the planner from 40 steps before it, legs cost 1000, closed shelves left to the end of the trip |
| `no_mix` | infinite penalty in Clarke & Wright |
| `robot_only` | mask in the Hungarian assignment |
| `deadline` | weight in the Hungarian assignment, own trip, TSPTW window in the sequence |
| `station` | the end of the trip of that order |

The validator runs before any solver and rejects, and logs, a rule with a pick order cycle, a closure that isolates a shelf, an order on both sides of `no_mix`, a heavy order on type B, two robot types or two stations for one order, a deadline below the lower bound `fastest`, or a deadline inside a closure. A rejected rule is dropped and the rest of the request runs. The verifier of the simulator checks every rule, and an episode is checked twice, against the rules of the arm and against the true ones.

The pipeline is Clarke & Wright at capacity 3 on a batch of 150 orders (saturation), LKH-3, the Hungarian assignment, prioritized planning with action dependencies, and execution with rotation and R1 delays (p = 0.10, q = 0.002), in the medium warehouse with 30 robots. Each request gets its own world and one episode; each distinct set of accepted rules of a request runs once and serves every arm that gave it.

L1 is built from sentence templates, 2–4 rules per request, with groups, exceptions, windows and names; labels always come from the formal generator. L2 is the same requests rewritten by gemma3:12b into a paragraph with coreference, distracting remarks and one implicit rule, with no filter by any parser. The prompts are in `scripts/phaseC_language.py`, and every seed is fixed.

Arms: (o) the oracle; (p0) the keyword parser; (p1) the extended parser, frozen at the tag `phaseC-parser-frozen` before the audit, after about 1 h of work against L1 and the development requests; (h1) p1 first, and the LLM only when p1 leaves sentences with constraint words uncovered; (q14) qwen3:14b without thinking; (q7) qwen2.5:7b on 100 requests per level. The LLMs share one prompt (legend, DSL, 3 L1 examples), answer in JSON, and get one retry if invalid, counted in the time.
## Data

200 test requests and 30 development requests, one world each; 598 rules, 2-4 per request. Draws needed by the oracle check: 201 for 200 requests. L2 texts missing: 0. Rule shapes: close_all 18, close_from 13, close_two 20, close_until 23, close_window 13, dead_customer 41, dead_except 8, dead_order 47, dead_urgent 16, mix_customers 27, mix_except 21, mix_kind_customer 19, mix_kinds 26, pick_except 28, pick_first 20, pick_last 21, pick_pair 28, robot_customer 28, robot_except 24, robot_kind 37, robot_station 17, st_customer 42, st_except 31, st_kind 30.
## Table B: translation

| level | arm | rules exact [Wilson 95 %] | requests exact [Wilson 95 %] | valid JSON | retries | rules dropped by the schema |
|---|---|---:|---:|---:|---:|---:|
| L1 | p0 | 430/598 = 71.9 % [68.2, 75.4] | 67/200 = 33.5 % [27.3, 40.3] | 100.0 % | 0 | 0 |
| L1 | p1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | h1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | q14 | 521/598 = 87.1 % [84.2, 89.6] | 135/200 = 67.5 % [60.7, 73.6] | 100.0 % | 0 | 14 |
| L1 | q7 | 256/301 = 85.0 % [80.6, 88.6] | 57/100 = 57.0 % [47.2, 66.3] | 100.0 % | 0 | 10 |
| L1 | q14 on the requests of q7 | 262/301 = 87.0 % [82.8, 90.4] | 65/100 = 65.0 % [55.3, 73.6] | | | |
| L2 | p0 | 336/598 = 56.2 % [52.2, 60.1] | 25/200 = 12.5 % [8.6, 17.8] | 100.0 % | 0 | 4 |
| L2 | p1 | 474/598 = 79.3 % [75.8, 82.3] | 84/200 = 42.0 % [35.4, 48.9] | 100.0 % | 0 | 4 |
| L2 | h1 | 489/598 = 81.8 % [78.5, 84.7] | 82/200 = 41.0 % [34.4, 47.9] | 100.0 % | 0 | 49 |
| L2 | q14 | 487/598 = 81.4 % [78.1, 84.4] | 81/200 = 40.5 % [33.9, 47.4] | 100.0 % | 0 | 63 |
| L2 | q7 | 209/301 = 69.4 % [64.0, 74.4] | 18/100 = 18.0 % [11.7, 26.7] | 99.0 % | 2 | 40 |
| L2 | q14 on the requests of q7 | 249/301 = 82.7 % [78.0, 86.6] | 43/100 = 43.0 % [33.7, 52.8] | | | |

Without the L2 texts the audit found wrong:

| level | arm | rules exact [Wilson 95 %] | requests exact [Wilson 95 %] | valid JSON | retries | rules dropped by the schema |
|---|---|---:|---:|---:|---:|---:|
| L1 | p0 | 430/598 = 71.9 % [68.2, 75.4] | 67/200 = 33.5 % [27.3, 40.3] | 100.0 % | 0 | 0 |
| L1 | p1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | h1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | q14 | 521/598 = 87.1 % [84.2, 89.6] | 135/200 = 67.5 % [60.7, 73.6] | 100.0 % | 0 | 14 |
| L1 | q7 | 256/301 = 85.0 % [80.6, 88.6] | 57/100 = 57.0 % [47.2, 66.3] | 100.0 % | 0 | 10 |
| L1 | q14 on the requests of q7 | 262/301 = 87.0 % [82.8, 90.4] | 65/100 = 65.0 % [55.3, 73.6] | | | |
| L2 | p0 | 303/534 = 56.7 % [52.5, 60.9] | 24/179 = 13.4 % [9.2, 19.2] | 100.0 % | 0 | 4 |
| L2 | p1 | 431/534 = 80.7 % [77.2, 83.8] | 81/179 = 45.3 % [38.1, 52.6] | 100.0 % | 0 | 4 |
| L2 | h1 | 442/534 = 82.8 % [79.3, 85.7] | 78/179 = 43.6 % [36.5, 50.9] | 100.0 % | 0 | 39 |
| L2 | q14 | 442/534 = 82.8 % [79.3, 85.7] | 78/179 = 43.6 % [36.5, 50.9] | 100.0 % | 0 | 50 |
| L2 | q7 | 187/261 = 71.6 % [65.9, 76.8] | 18/87 = 20.7 % [13.5, 30.4] | 100.0 % | 1 | 34 |
| L2 | q14 on the requests of q7 | 221/261 = 84.7 % [79.8, 88.5] | 41/87 = 47.1 % [37.0, 57.5] | | | |

q14 against p1:

| level | unit | q14 right, p1 wrong | q14 wrong, p1 right | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | rule | 0 | 77 | 1.32e-23 |
| L1 | request | 0 | 65 | 5.42e-20 |
| L2 | rule | 66 | 53 | 0.271 |
| L2 | request | 32 | 35 | 0.807 |

h1 against q14:

| level | unit | h1 right, q14 wrong | h1 wrong, q14 right | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | rule | 77 | 0 | 1.32e-23 |
| L1 | request | 65 | 0 | 5.42e-20 |
| L2 | rule | 16 | 14 | 0.856 |
| L2 | request | 10 | 9 | 1 |

h1 against p1:

| level | unit | h1 right, p1 wrong | h1 wrong, p1 right | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | rule | 0 | 0 | 1 |
| L1 | request | 0 | 0 | 1 |
| L2 | rule | 52 | 37 | 0.137 |
| L2 | request | 23 | 25 | 0.885 |

The same tests without the L2 texts the audit found wrong, q14 against p1:

| level | unit | q14 right, p1 wrong | q14 wrong, p1 right | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | rule | 0 | 77 | 1.32e-23 |
| L1 | request | 0 | 65 | 5.42e-20 |
| L2 | rule | 56 | 45 | 0.32 |
| L2 | request | 30 | 33 | 0.801 |

h1 against q14:

| level | unit | h1 right, q14 wrong | h1 wrong, q14 right | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | rule | 77 | 0 | 1.32e-23 |
| L1 | request | 65 | 0 | 5.42e-20 |
| L2 | rule | 13 | 13 | 1 |
| L2 | request | 9 | 9 | 1 |

On the 39 audited L2 texts found right, the only L2 sample known to be free of label errors (the L1 rows are unchanged):

| level | arm | rules exact [Wilson 95 %] | requests exact [Wilson 95 %] | valid JSON | retries | rules dropped by the schema |
|---|---|---:|---:|---:|---:|---:|
| L1 | p0 | 430/598 = 71.9 % [68.2, 75.4] | 67/200 = 33.5 % [27.3, 40.3] | 100.0 % | 0 | 0 |
| L1 | p1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | h1 | 598/598 = 100.0 % [99.4, 100.0] | 200/200 = 100.0 % [98.1, 100.0] | 100.0 % | 0 | 0 |
| L1 | q14 | 521/598 = 87.1 % [84.2, 89.6] | 135/200 = 67.5 % [60.7, 73.6] | 100.0 % | 0 | 14 |
| L1 | q7 | 256/301 = 85.0 % [80.6, 88.6] | 57/100 = 57.0 % [47.2, 66.3] | 100.0 % | 0 | 10 |
| L1 | q14 on the requests of q7 | 262/301 = 87.0 % [82.8, 90.4] | 65/100 = 65.0 % [55.3, 73.6] | | | |
| L2 | p0 | 73/120 = 60.8 % [51.9, 69.1] | 5/39 = 12.8 % [5.6, 26.7] | 100.0 % | 0 | 0 |
| L2 | p1 | 98/120 = 81.7 % [73.8, 87.6] | 19/39 = 48.7 % [33.9, 63.8] | 100.0 % | 0 | 0 |
| L2 | h1 | 100/120 = 83.3 % [75.7, 88.9] | 19/39 = 48.7 % [33.9, 63.8] | 100.0 % | 0 | 7 |
| L2 | q14 | 96/120 = 80.0 % [72.0, 86.2] | 15/39 = 38.5 % [24.9, 54.1] | 100.0 % | 0 | 11 |
| L2 | q7 | 35/53 = 66.0 % [52.6, 77.3] | 5/18 = 27.8 % [12.5, 50.9] | 100.0 % | 0 | 5 |
| L2 | q14 on the requests of q7 | 41/53 = 77.4 % [64.5, 86.5] | 7/18 = 38.9 % [20.3, 61.4] | | | |

### Types of error

| level | arm | rules missed | wrong selector | exception lost | inverted | wrong value | missing | rules invented | of them from a distractor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | p0 | 168 | 6 | 30 | 0 | 36 | 96 | 13 | 2 |
| L1 | p1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | h1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L1 | q14 | 77 | 16 | 25 | 10 | 11 | 15 | 0 | 0 |
| L1 | q7 | 45 | 9 | 9 | 5 | 9 | 13 | 4 | 1 |
| L2 | p0 | 262 | 20 | 15 | 5 | 11 | 211 | 85 | 44 |
| L2 | p1 | 124 | 10 | 20 | 5 | 9 | 80 | 25 | 5 |
| L2 | h1 | 109 | 25 | 33 | 10 | 6 | 35 | 51 | 19 |
| L2 | q14 | 111 | 27 | 40 | 10 | 3 | 31 | 50 | 23 |
| L2 | q7 | 92 | 29 | 14 | 7 | 3 | 39 | 52 | 21 |

### Rules exact by type

| level | arm | pick_order | close | no_mix | robot_only | deadline | station |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | p0 | 35 % | 59 % | 44 % | 84 % | 100 % | 100 % |
| L1 | p1 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| L1 | h1 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| L1 | q14 | 63 % | 75 % | 83 % | 99 % | 100 % | 98 % |
| L1 | q7 | 65 % | 79 % | 83 % | 87 % | 98 % | 95 % |
| L2 | p0 | 25 % | 17 % | 24 % | 76 % | 93 % | 87 % |
| L2 | p1 | 54 % | 75 % | 68 % | 92 % | 91 % | 92 % |
| L2 | h1 | 53 % | 75 % | 75 % | 96 % | 93 % | 94 % |
| L2 | q14 | 46 % | 76 % | 74 % | 98 % | 96 % | 93 % |
| L2 | q7 | 31 % | 85 % | 50 % | 79 % | 84 % | 80 % |

### The trigger of the hybrid

| level | sent to the LLM | of the texts p1 got wrong, sent | of the texts p1 got right, sent |
|---|---:|---:|---:|
| L1 | 0/200 | 0/0 | 0/200 |
| L2 | 144/200 | 89/116 | 55/84 |
## Table C: episodes

"0 violations" counts only the true rules, so a rule the arm invented cannot make an episode invalid; it can only slow it. That is why an arm can be valid more often than it is exact, and why a translation that leaves rules out looks faster: read validity and makespan together. Percentages against the oracle are paired on the request, and the means hide a median of 0 %. Violations per episode depend on the kind of rule, since one missed closure counts every step inside it, so they do not rank arms. "Total s" is the translation with the LLM median measured alone, once per request sent, plus the wall time of the pipeline.


| level | arm | episodes | 0 violations of the true rules [Wilson 95 %] | requests with a rule rejected (rules) | makespan | vs oracle % | mean delay | vs oracle % | true rules broken per episode | LLM + parser s, main run | total s, LLM alone | ms/step planner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | oracle | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 | 950 | - | 524 | - | 0.00 | 0 | 9.1 | 9.2 |
| L1 | p0 | 200 | 124/200 = 62.0 % [55.1, 68.4] | 25 (26) | 925 | -2.1 ± 0.9 | 516 | -1.5 ± 0.5 | 25.86 | 0.00 | 8.4 | 8.5 |
| L1 | p1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.00 | 9.1 | 9.2 |
| L1 | h1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.03 | 9.1 | 9.2 |
| L1 | q14 | 200 | 144/200 = 72.0 % [65.4, 77.8] | 27 (28) | 932 | -1.5 ± 0.8 | 518 | -1.2 ± 0.4 | 39.45 | 18.42 | 14.6 | 8.7 |
| L1 | q7 | 100 | 69/100 = 69.0 % [59.4, 77.2] | 15 (16) | 945 | -1.0 ± 1.5 | 522 | -0.5 ± 0.8 | 14.40 | 8.53 | 12.6 | 8.8 |
| L2 | oracle | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 | 950 | - | 524 | - | 0.00 | 0 | 9.1 | 9.2 |
| L2 | p0 | 200 | 65/200 = 32.5 % [26.4, 39.3] | 24 (25) | 928 | -1.6 ± 1.3 | 515 | -1.6 ± 0.7 | 109.81 | 0.00 | 8.5 | 8.5 |
| L2 | p1 | 200 | 120/200 = 60.0 % [53.1, 66.5] | 27 (29) | 939 | -0.7 ± 0.9 | 520 | -0.7 ± 0.5 | 36.47 | 0.00 | 8.9 | 9.0 |
| L2 | h1 | 200 | 135/200 = 67.5 % [60.7, 73.6] | 53 (56) | 952 | +0.6 ± 1.7 | 528 | +0.8 ± 1.8 | 42.20 | 15.76 | 17.2 | 9.1 |
| L2 | q14 | 200 | 132/200 = 66.0 % [59.2, 72.2] | 47 (50) | 951 | +0.5 ± 1.7 | 527 | +0.6 ± 1.8 | 45.45 | 22.08 | 20.3 | 9.0 |
| L2 | q7 | 100 | 48/100 = 48.0 % [38.5, 57.7] | 32 (38) | 949 | -0.3 ± 2.2 | 525 | +0.1 ± 1.3 | 25.04 | 13.72 | 14.1 | 8.6 |

Without the L2 texts the audit found wrong:

| level | arm | episodes | 0 violations of the true rules [Wilson 95 %] | requests with a rule rejected (rules) | makespan | vs oracle % | mean delay | vs oracle % | true rules broken per episode | LLM + parser s, main run | total s, LLM alone | ms/step planner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | oracle | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 | 950 | - | 524 | - | 0.00 | 0 | 9.1 | 9.2 |
| L1 | p0 | 200 | 124/200 = 62.0 % [55.1, 68.4] | 25 (26) | 925 | -2.1 ± 0.9 | 516 | -1.5 ± 0.5 | 25.86 | 0.00 | 8.4 | 8.5 |
| L1 | p1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.00 | 9.1 | 9.2 |
| L1 | h1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.03 | 9.1 | 9.2 |
| L1 | q14 | 200 | 144/200 = 72.0 % [65.4, 77.8] | 27 (28) | 932 | -1.5 ± 0.8 | 518 | -1.2 ± 0.4 | 39.45 | 18.42 | 14.6 | 8.7 |
| L1 | q7 | 100 | 69/100 = 69.0 % [59.4, 77.2] | 15 (16) | 945 | -1.0 ± 1.5 | 522 | -0.5 ± 0.8 | 14.40 | 8.53 | 12.6 | 8.8 |
| L2 | oracle | 179 | 179/179 = 100.0 % [97.9, 100.0] | 0 | 950 | - | 524 | - | 0.00 | 0 | 9.0 | 9.1 |
| L2 | p0 | 179 | 59/179 = 33.0 % [26.5, 40.1] | 19 (20) | 927 | -1.8 ± 1.4 | 514 | -1.7 ± 0.8 | 118.54 | 0.00 | 8.5 | 8.4 |
| L2 | p1 | 179 | 112/179 = 62.6 % [55.3, 69.3] | 24 (25) | 940 | -0.7 ± 1.0 | 521 | -0.7 ± 0.5 | 33.68 | 0.00 | 8.8 | 8.9 |
| L2 | h1 | 179 | 121/179 = 67.6 % [60.4, 74.0] | 45 (46) | 945 | -0.1 ± 1.6 | 524 | +0.1 ± 1.7 | 38.23 | 15.24 | 17.0 | 9.0 |
| L2 | q14 | 179 | 119/179 = 66.5 % [59.3, 73.0] | 38 (40) | 946 | -0.1 ± 1.6 | 523 | -0.0 ± 1.7 | 41.69 | 21.60 | 20.2 | 9.0 |
| L2 | q7 | 87 | 45/87 = 51.7 % [41.4, 61.9] | 23 (25) | 952 | +0.3 ± 2.4 | 526 | +0.3 ± 1.4 | 12.82 | 11.84 | 14.2 | 8.6 |

On the 39 audited L2 texts found right:

| level | arm | episodes | 0 violations of the true rules [Wilson 95 %] | requests with a rule rejected (rules) | makespan | vs oracle % | mean delay | vs oracle % | true rules broken per episode | LLM + parser s, main run | total s, LLM alone | ms/step planner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | oracle | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 | 950 | - | 524 | - | 0.00 | 0 | 9.1 | 9.2 |
| L1 | p0 | 200 | 124/200 = 62.0 % [55.1, 68.4] | 25 (26) | 925 | -2.1 ± 0.9 | 516 | -1.5 ± 0.5 | 25.86 | 0.00 | 8.4 | 8.5 |
| L1 | p1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.00 | 9.1 | 9.2 |
| L1 | h1 | 200 | 200/200 = 100.0 % [98.1, 100.0] | 0 (0) | 950 | +0.0 ± 0.0 | 524 | +0.0 ± 0.0 | 0.00 | 0.03 | 9.1 | 9.2 |
| L1 | q14 | 200 | 144/200 = 72.0 % [65.4, 77.8] | 27 (28) | 932 | -1.5 ± 0.8 | 518 | -1.2 ± 0.4 | 39.45 | 18.42 | 14.6 | 8.7 |
| L1 | q7 | 100 | 69/100 = 69.0 % [59.4, 77.2] | 15 (16) | 945 | -1.0 ± 1.5 | 522 | -0.5 ± 0.8 | 14.40 | 8.53 | 12.6 | 8.8 |
| L2 | oracle | 39 | 39/39 = 100.0 % [91.0, 100.0] | 0 | 946 | - | 521 | - | 0.00 | 0 | 9.0 | 9.2 |
| L2 | p0 | 39 | 16/39 = 41.0 % [27.1, 56.6] | 1 (1) | 913 | -2.4 ± 3.1 | 514 | -1.2 ± 1.2 | 84.28 | 0.00 | 8.7 | 8.7 |
| L2 | p1 | 39 | 27/39 = 69.2 % [53.6, 81.4] | 4 (4) | 925 | -1.5 ± 2.4 | 516 | -0.8 ± 0.9 | 26.10 | 0.00 | 8.7 | 9.0 |
| L2 | h1 | 39 | 28/39 = 71.8 % [56.2, 83.5] | 9 (9) | 929 | -1.2 ± 1.8 | 517 | -0.7 ± 0.8 | 11.15 | 14.69 | 16.3 | 8.8 |
| L2 | q14 | 39 | 25/39 = 64.1 % [48.4, 77.3] | 7 (7) | 936 | -1.0 ± 1.2 | 516 | -0.9 ± 0.8 | 30.82 | 21.96 | 19.8 | 8.7 |
| L2 | q7 | 18 | 10/18 = 55.6 % [33.7, 75.4] | 5 (6) | 951 | +2.9 ± 10.0 | 531 | +3.0 ± 6.3 | 13.17 | 9.72 | 14.1 | 8.6 |

Paired, valid episodes:

| level | arms | first valid, second not | first not, second valid | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | q14 vs p1 | 0 | 56 | 2.78e-17 |
| L1 | h1 vs q14 | 56 | 0 | 2.78e-17 |
| L1 | h1 vs p1 | 0 | 0 | 1 |
| L2 | q14 vs p1 | 38 | 26 | 0.169 |
| L2 | h1 vs q14 | 10 | 7 | 0.629 |
| L2 | h1 vs p1 | 31 | 16 | 0.04 |

The same without the L2 texts the audit found wrong:

| level | arms | first valid, second not | first not, second valid | exact McNemar p |
|---|---|---:|---:|---:|
| L1 | q14 vs p1 | 0 | 56 | 2.78e-17 |
| L1 | h1 vs q14 | 56 | 0 | 2.78e-17 |
| L1 | h1 vs p1 | 0 | 0 | 1 |
| L2 | q14 vs p1 | 32 | 25 | 0.427 |
| L2 | h1 vs q14 | 9 | 7 | 0.804 |
| L2 | h1 vs p1 | 25 | 16 | 0.211 |


12 of 1800 arm episodes broke a rule the arm itself had given the solvers; 0 of 200 oracle episodes did.

### What a wrong translation costs

| level | arm | not exact | of them invalid | makespan vs oracle %, invalid | delay vs oracle %, invalid | makespan vs oracle %, valid | delay vs oracle %, valid | true rule violations per invalid episode | atomic checks broken per invalid episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | p0 | 133 | 76 | -6.0 ± 2.1 | -4.0 ± 1.0 | +0.4 ± 1.1 | +0.2 ± 0.5 | 68.0 | 5.8 |
| L1 | q14 | 65 | 56 | -5.4 ± 2.7 | -4.3 ± 1.1 | +0.0 ± 0.9 | +0.1 ± 0.3 | 140.9 | 1.8 |
| L1 | q7 | 42 | 31 | -3.1 ± 5.1 | -1.5 ± 2.5 | -0.0 ± 0.4 | +0.0 ± 0.2 | 46.5 | 3.5 |
| L2 | p0 | 173 | 135 | -2.6 ± 1.9 | -2.7 ± 1.0 | +1.0 ± 2.0 | +0.9 ± 1.1 | 162.7 | 7.8 |
| L2 | p1 | 105 | 80 | -2.0 ± 2.2 | -2.1 ± 1.0 | +0.5 ± 1.8 | +0.9 ± 0.9 | 91.2 | 5.7 |
| L2 | h1 | 107 | 65 | -2.7 ± 3.2 | -2.1 ± 2.3 | +7.1 ± 6.2 | +7.0 ± 7.6 | 129.8 | 4.3 |
| L2 | q14 | 113 | 68 | -2.9 ± 2.9 | -2.3 ± 2.1 | +6.6 ± 5.8 | +6.3 ± 7.1 | 133.7 | 3.8 |
| L2 | q7 | 78 | 52 | -2.3 ± 3.9 | -1.1 ± 2.3 | +3.2 ± 3.3 | +2.6 ± 2.2 | 48.2 | 7.2 |


| level | arm | makespan vs oracle %, median | requests above +20 % | requests below -5 % | largest % |
|---|---|---:|---:|---:|---:|
| L1 | p0 | +0.0 | 0 | 37 | +18.6 |
| L1 | p1 | +0.0 | 0 | 0 | +0.0 |
| L1 | h1 | +0.0 | 0 | 0 | +0.0 |
| L1 | q14 | +0.0 | 0 | 26 | +12.4 |
| L1 | q7 | +0.0 | 1 | 10 | +52.7 |
| L2 | p0 | +0.0 | 4 | 43 | +34.4 |
| L2 | p1 | +0.0 | 2 | 24 | +31.8 |
| L2 | h1 | +0.0 | 6 | 24 | +98.1 |
| L2 | q14 | +0.0 | 6 | 28 | +98.1 |
| L2 | q7 | +0.0 | 4 | 22 | +64.2 |

### What the validator rejected

| arm | why the validator rejected | rules |
|---|---|---:|
| p0 | closure isolates a shelf | 29 |
| p0 | incompatible with itself: one order is on both sides | 18 |
| p0 | heavy order on a robot that cannot carry it | 2 |
| p0 | two robot types for one order | 1 |
| p0 | pick order cycle | 1 |
| p1 | closure isolates a shelf | 17 |
| p1 | pick order cycle | 6 |
| p1 | incompatible with itself: one order is on both sides | 5 |
| p1 | heavy order on a robot that cannot carry it | 1 |
| h1 | pick order cycle | 25 |
| h1 | closure isolates a shelf | 15 |
| h1 | incompatible with itself: one order is on both sides | 10 |
| h1 | heavy order on a robot that cannot carry it | 3 |
| h1 | deadline inside a closure of its corridor | 2 |
| h1 | two stations for one order | 1 |
| q14 | pick order cycle | 50 |
| q14 | closure isolates a shelf | 14 |
| q14 | incompatible with itself: one order is on both sides | 9 |
| q14 | heavy order on a robot that cannot carry it | 2 |
| q14 | deadline inside a closure of its corridor | 2 |
| q14 | two stations for one order | 1 |
| q7 | pick order cycle | 33 |
| q7 | closure isolates a shelf | 14 |
| q7 | heavy order on a robot that cannot carry it | 3 |
| q7 | two stations for one order | 2 |
| q7 | incompatible with itself: one order is on both sides | 2 |
## Latency

The request is translated once per episode, before it starts. The LLM takes seconds where the planner takes milliseconds per step, so a translation is worth 400–1200 planner steps and must stay out of the real-time loop; set against a whole episode it is about the same size. Calls in the main run took two to three times the latency measured alone: the machine carried a load of 7–10 from other users and the T4 throttled, to about 315 MHz at 84 °C. The 30 calls alone ran L1 first and L2 last with the card warming up, so some of the gap between levels is heat. One q7 L2 call took 85 s with nothing of ours running; the cause is unknown, it is what sets the p95 of 37.3 s, and the median does not move.

Measured alone, 30 calls per arm:

| arm | level | calls | median s | p95 s | tokens/s | GPU C | MHz | calls with LKH of mine running | translation in planner steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q14 | L1 | 15 | 6.1 | 9.8 | 14 | 73 | 1320 | 0 | 659 |
| q14 | L2 | 15 | 11.3 | 18.8 | 10 | 83 | 1215 | 0 | 1228 |
| q7 | L1 | 15 | 3.9 | 6.4 | 24 | 77 | 1380 | 0 | 425 |
| q7 | L2 | 15 | 5.5 | 37.3 | 20 | 83 | 1155 | 0 | 596 |

The planner takes 9.2 ms per step, or 8.8 s over an episode of 950 steps. The request is translated once, before the episode starts, so the LLM never enters the real-time loop.

In the main run, for reference:

| level | arm | mean s in the main run | p95 | of those calls, with LKH of mine running |
|---|---|---:|---:|---:|
| L1 | p0 | 0.00 | 0.00 | 0 |
| L1 | p1 | 0.00 | 0.00 | 0 |
| L1 | h1 | 0.03 | 0.04 | 0 |
| L1 | q14 | 18.42 | 28.58 | 0 |
| L1 | q7 | 8.53 | 13.61 | 0 |
| L2 | p0 | 0.00 | 0.00 | 0 |
| L2 | p1 | 0.00 | 0.00 | 0 |
| L2 | h1 | 15.76 | 33.40 | 0 |
| L2 | q14 | 22.08 | 34.22 | 0 |
| L2 | q7 | 13.72 | 18.32 | 0 |
## GPU

57 checks, 0 pauses, 0.0 min paused in all; the least free memory seen was 4312 MiB.


One model at a time, keep_alive 0 on each swap, num_ctx 8192, nvidia-smi every 5 minutes, and pause below 2 GB free.
## Audit of L2

I read 60 L2 texts drawn from the test requests, after freezing p1, and asked one question: can an attentive reader recover exactly the label? Decoration with no effect was allowed. Most errors are rules gemma invented, above all closed corridors, which the instruction to leave one rule implicit seems to invite, and groups lost to a paraphrase, as in "larger shipments" for heavy orders. Phase 5 had 28 %.


21 of 60 L2 texts carry a wrong label: 35.0 % [24.2, 47.6].

| request | what is wrong |
|---:|---|
| 1 | adds a rule: finish one customer before the next |
| 2 | adds a rule: corridor D closed, which does not exist |
| 5 | pick order turned into an order between trips |
| 20 | contradicts the exception: Quiosque Henriques must use type A |
| 29 | fragile picked first, where the rule says last |
| 30 | loses heavy in the pick order, and fragile in the robot rule |
| 36 | adds an exception for Dona Teresa to the station rule |
| 51 | adds a rule: corridor 2 closed |
| 58 | changes the exception and adds a rule: corridor 7 closed |
| 65 | closure moved from corridor 8 to the corridor of a remark, 7 |
| 79 | heavy becomes 'most of the larger orders' |
| 80 | exception on products of corridor 7 becomes trips through it |
| 91 | adds a rule: corridor 7 closed |
| 122 | exception becomes 'handled first' |
| 123 | only type A becomes 'whenever possible' |
| 124 | loses the customer: every order on type A |
| 125 | the exception of no_mix moves to the pick order; the no_mix rule is lost |
| 156 | heavy becomes 'larger shipments', so the group is lost |
| 158 | station rule inverted: Mercearia Tavares goes to E, the others lose it |
| 175 | adds a rule: corridor 7 closed |
| 177 | heavy becomes 'larger shipments', so the group is lost |
## Budget

| stage | hours |
|---|---:|
| CPU: L1 with its oracle episodes | 0.88 |
| CPU: 746 episodes | 1.86 |
| GPU: L2 | 1.60 |
| GPU: translation | 2.87 |
| GPU: latency alone | 0.15 |
| total | 7.37 |


The total is under the 8 h budget, so no sample was cut. It adds up CPU and GPU stages, and the time of writing p1, about 1 h against the limit of 2 h, is not in it.
## Caveats

- **L2 labels are noisy.** 35 % of the audited texts are wrong, so the best L2 scores cannot reach 100 %. Removing the 21 texts found wrong still leaves about 140 texts never read, and at the same rate about 49 wrong ones. The 39 texts audited and found right are the only clean L2 sample; there p1 and h1 get 19/39 requests and q14 15/39, too few to separate them.
- **p1 was fitted to the same author's grammar.** Its 100 % on L1 says that the templates are regular, not that a parser is enough; the L2 row is the fair one. p1 also saw the 30 development paragraphs.
- **Validity favours an arm that over-constrains.** Inventing a rule costs time, not validity. This probably explains part of the lead of q14 and h1 over p1 in table C despite equal exact requests: p1 misses rules outright (80 on L2 against 31 for q14), the LLM invents them (50 against 25).
- **Tests are not corrected for multiple comparisons.** h1 against p1 on validity, p = 0.04, would be about 0.12 under Holm across the three pairs.
- **Deadlines are not guaranteed.** The validator only checks a lower bound and the planner gives no guarantee, so 12 of 1800 arm episodes broke a rule of their own arm: 11 missed a deadline, and 1 (p0, a wrong closure) left a robot in a closed corridor for 66 steps, which I did not investigate further. None of the 200 oracle episodes did. The TSPTW window changes little in practice; the weight in the assignment does the work.
- **The validator cannot rescue a wrong rule.** Dropping a rejected rule leaves the true rule unenforced, and most q14 rejections (50 of 78 rules) are pick order cycles.
- **My choices, not in the request:** English texts, the 40-step guard on closures, R1 as the delay regime, and the rule that a closure with a window may not cover a corridor with shelves from a step on with no end.
- **Counting bug fixed before the episodes.** Episodes were first keyed on the rule set alone, which would have lost requests that shared a set, mostly the empty one and so the worst translations; the key now includes the request. All 1800 translation rows have an episode, from 746 distinct runs.
- **Times are from shared memos.** Episodes of one request share a distance memo, so the later ones run a little faster; wall times are indicative.
