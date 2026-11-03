## Conclusion

1. **Does the LLM beat a fair parser on warehouse requests?** No. On template text p1 gets 200/200 requests right and q14 135/200; on the L2 paragraphs they tie, 84/200 against 81/200 requests exact (p = 0.81), 79.3 % against 81.4 % of rules (p = 0.27) and 60.0 % against 66.0 % of valid episodes (p = 0.17).
2. **Does the hybrid beat both?** Not clearly: it keeps p1's 200/200 on L1 and ties q14 on L2 (41.0 % against 40.5 % of requests exact, 67.5 % against 66.0 % valid, p = 0.63), and its only edge over p1 on L2, 67.5 % against 60.0 % of valid episodes, has p = 0.04 before correcting for three tests and p = 0.21 without the texts the audit found wrong, while exact requests do not improve (41.0 % against 42.0 %).
3. **What does a wrong translation cost?** Mostly validity, little time: on L2, 34 % of q14 episodes and 40 % of p1 episodes break a true rule where the oracle breaks none, those invalid routes look 2–3 % shorter (q14: makespan −2.9 ± 2.9 %, delay −2.3 ± 2.1 %), and the median request costs 0 %, but rules the LLM adds or widens, some of them from label errors in L2 itself, make 6 of 200 requests more than 20 % slower, up to +98 %.

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

## Reading table C

"0 violations" counts only the true rules, so a rule the arm invented cannot make an episode invalid; it can only slow it. That is why an arm can be valid more often than it is exact, and why a translation that leaves rules out looks faster: read validity and makespan together. Percentages against the oracle are paired on the request, and the means hide a median of 0 %. Violations per episode depend on the kind of rule, since one missed closure counts every step inside it, so they do not rank arms. "Total s" is the translation with the LLM median measured alone, once per request sent, plus the wall time of the pipeline.

## Latency

The request is translated once per episode, before it starts. The LLM takes seconds where the planner takes milliseconds per step, so a translation is worth 400–1200 planner steps and must stay out of the real-time loop; set against a whole episode it is about the same size. Calls in the main run took two to three times the latency measured alone: the machine carried a load of 7–10 from other users and the T4 throttled, to about 315 MHz at 84 °C. The 30 calls alone ran L1 first and L2 last with the card warming up, so some of the gap between levels is heat. One q7 L2 call took 85 s with nothing of ours running; the cause is unknown, it is what sets the p95 of 37.3 s, and the median does not move.

## GPU

One model at a time, keep_alive 0 on each swap, num_ctx 8192, nvidia-smi every 5 minutes, and pause below 2 GB free.

## Audit

I read 60 L2 texts drawn from the test requests, after freezing p1, and asked one question: can an attentive reader recover exactly the label? Decoration with no effect was allowed. Most errors are rules gemma invented, above all closed corridors, which the instruction to leave one rule implicit seems to invite, and groups lost to a paraphrase, as in "larger shipments" for heavy orders. Phase 5 had 28 %.

## Budget

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
