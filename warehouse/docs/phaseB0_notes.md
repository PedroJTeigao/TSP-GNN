## Recommendation

1. **Which component has margin for a learned model?** Only batching with 3 or more orders per trip, and only if a search confirms it: batching gives by far the largest gain (+24–27 % with 2 orders, +33–37 % with 3), but with 2 orders Clarke & Wright is already within 1.3 % / 0.8 % of the exact pairs, and hindsight on traffic stands at −0.4 % / +0.6 % against the best rule without learning, compared with the +4–5 % expected from Yu & Wolf 2023.
2. **What would it have to predict?** Input: a graph of the open orders (shelves, stations, shared products) with the robot positions; output: a score per pair of orders for sharing a trip; label: the trips of LKH-3 in CVRP mode with capacity 3, which has not been run on whole lots and has to show a gap above noise before anything is trained.
3. **Which baseline must it beat?** Clarke & Wright with capacity 3 + LKH-3 + prioritized: 676 steps on the medium warehouse and 920 on the large one, at 4 ms and 13 ms per step.

## Margins

One row per arm and size, over 5 seeds of the E1 lots. The gain is against the baseline, positive is better, with a 95 % interval. With 5 seeds the two-sided Wilcoxon p cannot go below 0.0625, so p = 0.0625 means every seed agrees and is the strongest result possible here; read the interval. "% own free flow" is the arm against its own assignment and routes with each robot alone; "% baseline free flow" uses the same reference for every arm, so batching falls below 100. O4 runs on small and medium only, MCA + LNS and GGO-lite on medium only. Letters follow the plan: O1(a) and O3(a) are the baseline.

## Protocol

- Every arm re-runs the E1 lots with one change. The baseline repeats E1 exactly, and every run is judged by the verifier. The report follows Merschformann 2019: each arm as a percentage of a free-flow limit, and a best/worst factor per component.
- **Free flow:** same assignment and routes, each robot alone. It is not measured for the O2-A and O4 arms, which change only the planner. The free flow stored for `o4pibt5` ran before the arm and repeats the baseline's plan, so it is ignored.
- **O1(b):** "reassignment on each arrival" fires when a robot becomes free, because in E1 every order arrives at step 0 and the literal rule would equal the plain Hungarian.
- **O1(e):** the real path is measured once per episode, as a per-robot scale, not by a simulation per insertion. It stops at 20 iterations, which takes 46 s of the 20 min allowed.
- **O1 costs with traffic:** they include waits and detours; γ is fitted by least squares per edge on other seeds, and comes to 0.46–0.55.
- **O2-B:** it runs on Hungarian + reassignment (`o1b`), and the metric reaches LKH-3 and the Hungarian.
- **PIBT rules:** `far` and `near` only break ties below age, because as real weights `far` blocks and delivers nothing. Only complete runs count.
- **Extra arm:** `o3x` gives the exact pairs of the whole lot to the simulator. The plan limited the exact check to 20 orders; a matching is polynomial, and this run answers the question directly in a few minutes of CPU.

## O3

Grouping orders is by far the largest gain, and the way of grouping matters little once it uses real trip costs: the worst arm with capacity 2 (k-means regions) is 12 % above the exact pairs, and Clarke & Wright is 1.3 % / 0.8 % above in makespan. The seed rule of shelf overlap does not beat its plain control in makespan and loses in delay. A traffic map changes nothing (`o3e`). Capacity 3 was cheap and adds 9–10 points; its exact optimum is not polynomial and no oracle was run for it, so its residual margin is unknown.

The 20-order check of LKH-3 in CVRP mode matches the exact matching in all 40 subsets. On whole lots savings get closer to the optimum than on subsets of 20, since more orders give more good pairs.

## O1

The gain of about 5 % in makespan and 12 % in delay comes almost entirely from serving short orders first: the control restricted to the oldest orders (`o1a_fifo`) gains +0.6 %, and removing the order length from the cost (`o1b0`) halves the delay gain. Iterative reassignment, and costs with traffic, add nothing beyond noise.

The two oracles do not beat the online rules, but they are shallow: 200 swaps and 20 LNS iterations. `o1c` optimises only the makespan and leaves the delay unchanged. So "no margin" here means none found by a cheap search, not a proof, and the gap to the lower bound stays at about 21 % of the baseline; that bound ignores empty walking between orders, so it is loose.

## O2

Traffic costs 4.3–4.6 % of the baseline makespan and 6.2 % of the delay on medium and large (9 % on small), and about 6 % of the makespan once the batching or assignment improves. That is the ceiling of anything that only removes congestion.

**Hindsight − best not learned = −0.4 % on medium, +0.6 % on large, in makespan**, both within noise; in delay it is negative. That is the room for a GNN predicting travel times in layer 1, against the +4–5 % expected from Yu & Wolf 2023. Against the same dispatcher without traffic the oracle gives +0.6 % and +1.8 %. Even so, these numbers lean in its favour: it keeps the best of 3 runs per seed. Iterating does not converge, since new distances move the traffic elsewhere.

No O2-A arm beats the plain planner. Directed highways lose 3–11 %, flow costs at best tie, and GGO-lite overfits: it gains 1.6 % on its two seeds, loses 1.4 % on the five of the report, and pushes its weights towards zero.

## O4

A better planner gives almost nothing where traffic matters. CBS with 3 x the limits gains 3.6 % on small and 0.5 % on medium, at 86 and 428 times the computation per step; on medium 15 % of the steps still fall back to PIBT. No episode reached the budget. PIBT loses 2.6–3.2 % to prioritized even with the best of 5 rules per seed. A window of 10 ties with the window of 20, and a window of 5 loses about 2 %.

## Budget

About 8 h of CPU, at the limit of the budget: 2.1 h of arms, 1.8 h of GGO-lite, 2.8 h of CBS, at most 1.1 h of free flow and some 0.2 h more. Nothing was cut. Not done: the optional O1(f), Liu 2019. Stopped by their own limits or the plan: GGO-lite and MCA + LNS ran on medium only, and O4 on small and medium only.

## Limitations

- Five seeds: a difference below about 1 % is noise, and several arms in O1 and O2-B cannot be told apart.
- Gains do not add up, and only some combinations were measured. The batching arms keep orders by age (`Clusters` hands trips out by order id), so the short-first effect of O1 is not in them; adding it could improve the baseline of the recommendation.
- As noted by Varambally 2022, the model has no rotation and no real service durations.
- The free flow of `o2b3` uses routes in the plain metric, since the record does not keep its metric, so its percentage is approximate.
- E1 is offline: all orders are known at step 0. Lifelong margins were not measured.
- The Yu & Wolf figure is the expectation stated in the plan; it comes from a different setting.
