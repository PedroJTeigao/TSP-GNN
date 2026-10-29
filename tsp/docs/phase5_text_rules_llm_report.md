# Phase 5, step 2 — the LLM where the rule is not enough, judged by the code checker

Generated 2026-10-29 by scripts/phase5_text_llm_report.py (runs at commit 63176cd, 938bdea, b7ad9a9). Model: qwen2.5:7b (Q4_K_M) on the private Ollama, context 8192, temperature 0 (paraphrases: 0.8). Data: the step-1 instances, n = 200 and 1000, 200 each.

## Summary

**Read this first: b0's 0 % on text_v2b is by design.** text_v2b was built so that the regex arm b0 fails (no
generator formulation is allowed in a paraphrase). The fair comparison for the LLM is **b0+**, a keyword parser
built from the vocabulary of the part-A meaning filters (no LLM).

1. **Does the LLM beat regex on text_v2b?** It beats b0 (0 %), but only by design. **It does not beat b0+ at either
   n, on any rule type or any sentence kind.**
   - Rules exact, b0+ against the 7B:
     - n = 200: b0+ 99.0 %, b1 77.0 %, b2 80.9 %;
     - n = 1000: b0+ 98.8 %, b1 57.6 %, b2 58.6 %.
   - Instances exact: b0+ 97–98 %, the 7B 25–45 %.
   - Conclusion: on this test set a keyword parser is enough.
   - By sentence kind, the 7B is closest on own sentences with numbers only (87–95 %, against 98–99 % for b0+).
     It loses on merged sentences (64–86 % against 100 %).
   - It collapses on named customers at n = 1000 (15–16 %, against 68 % at n = 200 and 99 % for b0+). The dictionary
     has 100 names there instead of 20. The 7B reads the rule and its times correctly but looks up the wrong
     customer number.
   - By type, the 7B is weakest on mandatory edges (45–48 %) and precedences (53–59 %); soft zones are fine
     (96–99 %).
   - The few-shot examples (b2) help a little at n = 200 (+4 points) and not at n = 1000.
   - On text_v1 (the generator's own sentences) the 7B gets 95–97 % of rules, where b0 and b0+ get 100 %.
2. **How much does a wrong formalisation cost in the tour?** The cost is validity, not length.
   - With an exact translation the pipeline returns the reference route.
   - On text_v2b, the share of routes that meet every hard rule of the correct label falls:
     - from 100 % (oracle) and 98.4–98.5 % (b0+);
     - to 68–74 % (7B, n = 200) and 46–48 % (7B, n = 1000);
     - with 0.3–0.7 broken hard rules per instance.
   - The objective is only +0.5–0.8 % above the reference. Those routes mix valid and invalid ones, and breaking a
     rule usually makes the route shorter: b0, which drops every rule, is −0.2 to −0.3 %.
   - Mandatory edges are the most broken type.
3. **Does the LLM verify better than chance without the code report?** No (n = 200 only).
   - Plain variant (text + route with times): precision 27.7 %, recall 35.0 %. A random checker that flags each rule
     at the base rate gets 29 / 29.
   - It flags a violation on 51.5 % of the correct oracle routes. Its route verdict is right 39.8 % of the time,
     below the 50 % of "never report a violation".
   - With the code checker's report, recall rises to 97.8 %, but precision is only 50.7 %. It still raises false
     alarms on 71.4 % of the oracle routes, partly on the non-binding windows (44.6 %).
   - The LLM adds noise to the code checker; it does not add information.

## Caveats

- **Same model on both sides.** The text_v2b paraphrases were written by the same qwen2.5:7b that translates them in
  b1 / b2, which could favour the LLM. It still loses to b0+.
- **The construction favours b0+.**
  - The text_v2b meaning filters force every accepted sentence to contain an adjacency word for edges and an order
    word compatible with the precedence. Those are exactly the words b0+ looks for.
  - I wrote b0+ after reading about 200 text_v2b sentences in the manual check. I did not tune it on text_v2b, and
    its known failures stay in:
    - "one right after the other, but the order doesn't matter": "n't" is read as a negation, so a mandatory edge
      becomes forbidden (6 rules);
    - zone names written without the word "zone" are missed (3 rules).
  - A free-text test set without these filters would be harder for b0+. It would also need a new manual label check.
- **Label errors.**
  - Rate in the manual check: 3 / 40 random sentences (7.5 %), all of them merged. Across all merged sentences:
    9 / 52 (17 %).
  - Precedences: 0 / 149 inverted, 1 contradictory.
  - The 9 affected instances are left out of B, C and D (196 kept at n = 200, 193 at n = 1000, after also dropping
    the 2 instances with a rule with no accepted sentence).
- **Part D at n = 200 only.** A route written as visits with times is about 10k tokens at n = 1000 and does not fit
  the 8192-token context.
- **Ollama aborts.**
  - Ollama aborted one b1 answer at n = 200 (HTTP 500, "token repeat limit reached": the model looped). It is
    counted as a failed answer, in the same column as invalid JSON.
  - The same abort stopped the first part-B run at n = 1000 before the fix. On resume, the same call (temperature
    0, same seed) answered normally, so the abort is not deterministic.
- **One non-finite time.** One b2 answer (n = 1000, instance 96, text_v2b) gave `"t2": "inf"`, which `float()`
  accepts. The schema now rejects non-finite times. All 7B answers are validated again from the raw answer when
  loaded, and this was the only record that changed.
- **LKH-3 stuck in a loop.**
  - At n = 200, instance 193, arm b1 on text_v2b, the 7B's translation was:
    - a mandatory edge 5–8;
    - precedences 108→169 and 3→108;
    - and both 3→4 and 4→3.
  - The label is precedence 169→186 and mandatory edges 48–33 and 5–8. The named customers were resolved to the
    wrong numbers, and one of the precedences came out in both directions.
  - An SOP with a precedence cycle has no solution. With the mandatory edge as a fixed edge (8–5), LKH-3.0.6 looped
    inside one trial and never checked TIME_LIMIT: 16 h at 100 % CPU, killed by hand. The second direction of the
    fixed edge looped too and was killed after 3.5 min.
  - The case counts as "LKH found no route" and falls back to the pure route: invalid, both mandatory edges of the
    label broken.
  - The pipeline now kills LKH-3 after 3 × TIME_LIMIT + 60 s and keeps a tour it already wrote. At n = 1000, 16
    solves took over 500 s, so the cut probably fired there; the pipeline does not record whether it did.
  - The b2 translation of the same instance also had a precedence cycle (33↔48); LKH-3 declared it infeasible in
    2 s.
  - Translations are not repaired for cycles: the cycle counts as the model's error.
- **Reference tour not always reused.** The key used to recognise an exact translation was not stable for edges
  (the printed order of an unordered pair can change). Some exact translations were therefore solved again by LKH-3
  instead of reusing the reference. Their routes are still valid at cost +0.00 %. The translation column of part C
  is taken from the part-B scoring, and the key is now fixed.
- **b3 not tested.** There is no API key on the server.

## A. The linguistic test set (text_v2b)

| | n = 200 | n = 1000 |
|---|---|---|
| instances | 200 | 200 |
| rules (label) | 401 | 430 |
| rules written as their own paraphrase | 351 | 374 |
| rules inside a merged two-rule sentence | 50 | 54 |
| rules with no accepted sentence (instance left out) | 0 | 2 |
| rules whose customers are named in the dictionary (by design) | 167 | 168 |
| names in the dictionary per instance | 20 | 100 |
| instances with an extra non-binding window | 28 | 27 |
| irrelevant sentences (total) | 293 | 312 |
| instances with a label error (manual check, left out) | 4 | 5 |
| instances with a missing rule (left out) | 0 | 2 |
| **evaluation set** | 196 | 193 |

Manual check of text_v2b (the paraphrases are written by the same 7B that translates them in b1 / b2):
- random sample of 40 sentences (20 per n): 3 with the meaning changed (7.5 %); single-rule sentences 0/31, merged sentences 3/9;
- all 52 merged sentences: 9 wrong (17.3 %): n = 200 #67: forbidden edge inside 'it's preferable to ...' (softened); n = 200 #71: forbidden edge softened ('try not to'); n = 200 #104: mandatory edge given a direction ('right before'); n = 200 #166: the two forbidden pairs are crossed (wrong customers); n = 1000 #10: forbidden pair wrong (Chen's pharmacy with both customers); n = 1000 #14: both mandatory edges given a direction ('right after'); n = 1000 #22: 'and then go to customer 711' adds an order / link; n = 1000 #26: 'and then ... customer 535' adds an order; n = 1000 #65: second precedence followed by 'regardless of the order' (contradictory);
- all precedences (129 single-rule, 20 merged): 0 inverted, 1 contradictory (1000 #65 contradictory).
- These are label errors (the sentence does not say what the label says), not model errors: the instances are left out of B (both texts), C and D. Single-rule sentences passed the keyword meaning filters of text_v2b; merged sentences carry most of the remaining error.

## B. Formalisation: text → formal rules (JSON), exact match against the label

| n | text | arm | rules exact (%) | instances exact (%) | answers with schema errors (%) | invalid JSON or aborted answer (%) | mean latency (s) |
|---|---|---|---|---|---|---|---|
| 200 | text_v1 | b0 regex | 100.0 | 100.0 | 0.0 | 0.0 | 0.00 |
| 200 | text_v1 | b0+ keywords | 100.0 | 100.0 | 0.0 | 0.0 | 0.00 |
| 200 | text_v1 | b1 7B zero-shot | 96.9 | 93.4 | 0.0 | 0.0 | 5.25 |
| 200 | text_v1 | b2 7B few-shot | 96.7 | 93.9 | 0.0 | 0.0 | 5.05 |
| 200 | text_v2b | b0 regex | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| 200 | text_v2b | b0+ keywords | 99.0 | 98.0 | 0.0 | 0.0 | 0.00 |
| 200 | text_v2b | b1 7B zero-shot | 77.0 | 39.8 | 15.3 | 2.0 | 8.66 |
| 200 | text_v2b | b2 7B few-shot | 80.9 | 45.4 | 10.2 | 0.0 | 7.07 |
| 1000 | text_v1 | b0 regex | 100.0 | 100.0 | 0.0 | 0.0 | 0.00 |
| 1000 | text_v1 | b0+ keywords | 100.0 | 100.0 | 0.0 | 0.0 | 0.00 |
| 1000 | text_v1 | b1 7B zero-shot | 95.2 | 89.1 | 0.0 | 0.0 | 5.87 |
| 1000 | text_v1 | b2 7B few-shot | 95.6 | 90.7 | 0.5 | 0.0 | 5.48 |
| 1000 | text_v2b | b0 regex | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| 1000 | text_v2b | b0+ keywords | 98.8 | 97.4 | 0.0 | 0.0 | 0.00 |
| 1000 | text_v2b | b1 7B zero-shot | 57.6 | 32.6 | 17.1 | 1.6 | 10.10 |
| 1000 | text_v2b | b2 7B few-shot | 58.6 | 24.9 | 9.3 | 0.0 | 8.85 |

b3 (API model): not tested, no OPENAI_API_KEY / ANTHROPIC_API_KEY on the server.

Rules exact per type (%), both n together:

| text | arm | precedence | forbidden_edge | mandatory_edge | time_window | soft_zone |
|---|---|---|---|---|---|---|
| text_v1 | b0 regex | 100.0 (147) | 100.0 (149) | 100.0 (148) | 100.0 (195) | 100.0 (166) |
| text_v1 | b0+ keywords | 100.0 (147) | 100.0 (149) | 100.0 (148) | 100.0 (195) | 100.0 (166) |
| text_v1 | b1 7B zero-shot | 88.4 (147) | 98.7 (149) | 92.6 (148) | 99.5 (195) | 99.4 (166) |
| text_v1 | b2 7B few-shot | 90.5 (147) | 99.3 (149) | 89.2 (148) | 100.0 (195) | 100.0 (166) |
| text_v2b | b0 regex | 0.0 (147) | 0.0 (149) | 0.0 (148) | 0.0 (195) | 0.0 (166) |
| text_v2b | b0+ keywords | 100.0 (147) | 100.0 (149) | 95.9 (148) | 100.0 (195) | 98.2 (166) |
| text_v2b | b1 7B zero-shot | 53.1 (147) | 62.4 (149) | 48.0 (148) | 70.8 (195) | 96.4 (166) |
| text_v2b | b2 7B few-shot | 58.5 (147) | 69.1 (149) | 44.6 (148) | 71.3 (195) | 99.4 (166) |

Rules exact on text_v2b by how the rule is written, per n and both together: its own sentence with customer numbers only, its own sentence with at least one customer named, or inside a merged two-rule sentence. Soft zones have no customer and are counted under 'numbers'. The name dictionary has 20 entries at n = 200 and 100 at n = 1000.

| n | arm | own sentence, numbers | own sentence, named | merged sentence |
|---|---|---|---|---|
| 200 | b0 regex | 0.0 (176) | 0.0 (174) | 0.0 (42) |
| 200 | b0+ keywords | 99.4 (176) | 98.3 (174) | 100.0 (42) |
| 200 | b1 7B zero-shot | 86.9 (176) | 68.4 (174) | 71.4 (42) |
| 200 | b2 7B few-shot | 92.6 (176) | 67.8 (174) | 85.7 (42) |
| 1000 | b0 regex | 0.0 (196) | 0.0 (173) | 0.0 (44) |
| 1000 | b0+ keywords | 98.5 (196) | 98.8 (173) | 100.0 (44) |
| 1000 | b1 7B zero-shot | 92.9 (196) | 15.0 (173) | 68.2 (44) |
| 1000 | b2 7B few-shot | 94.9 (196) | 16.2 (173) | 63.6 (44) |
| 200 + 1000 | b0 regex | 0.0 (372) | 0.0 (347) | 0.0 (86) |
| 200 + 1000 | b0+ keywords | 98.9 (372) | 98.6 (347) | 100.0 (86) |
| 200 + 1000 | b1 7B zero-shot | 90.1 (372) | 41.8 (347) | 69.8 (86) |
| 200 + 1000 | b2 7B few-shot | 93.8 (372) | 42.1 (347) | 74.4 (86) |

Confusion of rule types on text_v2b, both n (rows: label type; columns: the type of the prediction matched to it — same key, else same customers / zone, else any shared customer; 'missing' = no prediction; last row: predictions matched to no label rule). Diagonal counts include wrong parameters.

**b0 regex**

| label \ predicted | precedence | forbidden_edge | mandatory_edge | time_window | soft_zone | missing |
|---|---|---|---|---|---|---|
| precedence | · | · | · | · | · | 147 |
| forbidden_edge | · | · | · | · | · | 149 |
| mandatory_edge | · | · | · | · | · | 148 |
| time_window | · | · | · | · | · | 195 |
| soft_zone | · | · | · | · | · | 166 |
| spurious | · | · | · | · | · | · |

**b0+ keywords**

| label \ predicted | precedence | forbidden_edge | mandatory_edge | time_window | soft_zone | missing |
|---|---|---|---|---|---|---|
| precedence | 147 | · | · | · | · | · |
| forbidden_edge | · | 149 | · | · | · | · |
| mandatory_edge | · | 6 | 142 | · | · | · |
| time_window | · | · | · | 195 | · | · |
| soft_zone | · | · | · | · | 163 | 3 |
| spurious | · | · | · | · | · | · |

**b1 7B zero-shot**

| label \ predicted | precedence | forbidden_edge | mandatory_edge | time_window | soft_zone | missing |
|---|---|---|---|---|---|---|
| precedence | 109 | · | 7 | · | · | 31 |
| forbidden_edge | 3 | 123 | 2 | · | · | 21 |
| mandatory_edge | 22 | · | 97 | 1 | · | 28 |
| time_window | 2 | · | · | 138 | · | 55 |
| soft_zone | · | · | · | · | 160 | 6 |
| spurious | 74 | 28 | 56 | 74 | 71 | · |

**b2 7B few-shot**

| label \ predicted | precedence | forbidden_edge | mandatory_edge | time_window | soft_zone | missing |
|---|---|---|---|---|---|---|
| precedence | 117 | · | 7 | · | · | 23 |
| forbidden_edge | 1 | 126 | · | · | · | 22 |
| mandatory_edge | 20 | · | 101 | 2 | · | 25 |
| time_window | 1 | · | · | 139 | · | 55 |
| soft_zone | · | · | · | · | 165 | 1 |
| spurious | 68 | 24 | 50 | 64 | 104 | · |

## C. Full pipeline: arm's translation → LKH-3 → code checker against the CORRECT label

Routes judged against the label. *valid*: meets every hard rule of the label; *violated*: mean number of hard label rules broken per instance; *cost*: objective (length + soft-zone penalty, of the label) against the reference, mean over all instances. Same LKH-3 parameters as the reference (the oracle IS the reference).

| n | text | arm | translation exact (%) | valid routes (%) | violated rules / instance | cost vs reference (%) | LKH found no route for the translation (%) | repaired translations (%) |
|---|---|---|---|---|---|---|---|---|
| 200 | text_v1 | b0 regex | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 200 | text_v1 | b0+ keywords | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 200 | text_v1 | b1 7B zero-shot | 93.4 | 94.4 | 0.06 | +0.01 | 0.0 | 0.5 |
| 200 | text_v1 | b2 7B few-shot | 93.9 | 94.4 | 0.07 | -0.01 | 0.5 | 0.0 |
| 200 | text_v1 | oracle (label) | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 200 | text_v2b | b0 regex | 0.0 | 16.8 | 1.27 | -0.27 | 0.0 | 0.0 |
| 200 | text_v2b | b0+ keywords | 98.0 | 98.5 | 0.02 | +0.00 | 0.0 | 0.0 |
| 200 | text_v2b | b1 7B zero-shot | 39.8 | 68.4 | 0.37 | +0.55 | 3.1 | 11.7 |
| 200 | text_v2b | b2 7B few-shot | 45.4 | 74.0 | 0.31 | +0.71 | 3.6 | 7.1 |
| 200 | text_v2b | oracle (label) | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 1000 | text_v1 | b0 regex | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 1000 | text_v1 | b0+ keywords | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 1000 | text_v1 | b1 7B zero-shot | 89.1 | 90.2 | 0.12 | +0.02 | 1.0 | 1.6 |
| 1000 | text_v1 | b2 7B few-shot | 90.7 | 90.7 | 0.09 | +0.01 | 0.0 | 0.5 |
| 1000 | text_v1 | oracle (label) | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |
| 1000 | text_v2b | b0 regex | 0.0 | 19.2 | 1.39 | -0.19 | 0.0 | 0.0 |
| 1000 | text_v2b | b0+ keywords | 97.4 | 98.4 | 0.02 | -0.00 | 0.0 | 0.0 |
| 1000 | text_v2b | b1 7B zero-shot | 32.6 | 48.2 | 0.71 | +0.74 | 5.2 | 6.7 |
| 1000 | text_v2b | b2 7B few-shot | 24.9 | 46.1 | 0.70 | +0.68 | 3.6 | 6.2 |
| 1000 | text_v2b | oracle (label) | 100.0 | 100.0 | 0.00 | +0.00 | 0.0 | 0.0 |

Violated label rules by type (both n, count over instances):

| text | arm | precedence | forbidden_edge | mandatory_edge | time_window |
|---|---|---|---|---|---|
| text_v1 | b0 regex | 0 | 0 | 0 | 0 |
| text_v1 | b0+ keywords | 0 | 0 | 0 | 0 |
| text_v1 | b1 7B zero-shot | 17 | 5 | 12 | 0 |
| text_v1 | b2 7B few-shot | 13 | 1 | 16 | 2 |
| text_v1 | oracle (label) | 0 | 0 | 0 | 0 |
| text_v2b | b0 regex | 65 | 149 | 148 | 155 |
| text_v2b | b0+ keywords | 0 | 0 | 6 | 0 |
| text_v2b | b1 7B zero-shot | 33 | 58 | 78 | 41 |
| text_v2b | b2 7B few-shot | 29 | 44 | 81 | 41 |
| text_v2b | oracle (label) | 0 | 0 | 0 | 0 |

## D. The 7B as route verifier (n = 200, text_v2b)

n = 1000 left out: a route written as visits with times is ~10k tokens and does not fit the 8192-token context. Routes: the oracle route of each instance (no violation) and the same route with one injected error, verified in code to break exactly one rule. Unit: (route, hard rule of the label); truth: the code checker (100 % by construction). *plain*: text + route; *report*: also the code checker's report.

| variant | routes | invalid JSON (%) | route verdict right (%) | false alarms on oracle routes (%) | injected error found (%) | false alarms on non-binding windows (%) | claims matching no rule | mean latency (s) |
|---|---|---|---|---|---|---|---|---|
| plain | 379 | 0.0 | 39.8 | 51.5 | 35.0 | 14.3 (56) | 192 | 9.8 |
| report | 379 | 1.1 | 60.9 | 71.4 | 97.8 | 44.6 (56) | 91 | 12.4 |

Precision / recall per rule type (unit: route × hard rule; a JSON failure counts as no claim):

| type | violations (truth) | rule units | plain P | plain R | report P | report R | random at base rate P = R |
|---|---|---|---|---|---|---|---|
| precedence | 45 | 144 | 32.1 | 40.0 | 51.8 | 97.8 | 31.2 |
| forbidden_edge | 37 | 130 | 27.4 | 45.9 | 43.2 | 94.6 | 28.5 |
| mandatory_edge | 36 | 146 | 24.3 | 47.2 | 39.1 | 100.0 | 24.7 |
| time_window | 65 | 212 | 27.9 | 18.5 | 67.4 | 98.5 | 30.7 |
| all | 183 | 632 | 27.7 | 35.0 | 50.7 | 97.8 | 29.0 |

Chance baselines: 'never report a violation' has recall 0 and is right on every oracle route (50 % route verdicts if half the routes are injected); a random checker that flags each rule with the base-rate probability has precision = recall = base rate.

