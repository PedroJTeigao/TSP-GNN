# Assumptions audit

Read-only audit of `tsp/` and `warehouse/` at commit 6aeb1a5, 2026-11-02. No experiment, training or solver was run. The numbers come from code, stored results, and light CPU reads of data files.

The lists asked for are at the end: [(a) ten to tell the professor](#a-the-ten-assumptions-to-tell-the-professor), [(b) what could invalidate a conclusion](#b-assumptions-that-could-invalidate-a-conclusion), [(c) cheap ablations](#c-assumptions-that-a-cheap-ablation-can-test).

## Resolved issue: phase C episode deduplication

At audit time, `warehouse/scripts/phaseC_language.py:main_c` removed duplicate episodes with one `seen` set across all requests, and its key was a hash of the accepted rules, with no request id. Two requests with the same accepted set shared one key, and the usual case is the empty set. Only the first ran. The others had no episode, so `table_c` skipped them without a message, and these are the worst translations.

Stage `b` was running and stage `c` had not started (`warehouse/scripts/run_phaseC.sh`). The fix keys episodes on `(r["idx"], key)` (`phaseC_language.py:251-256`) and was committed in 8b5c294 with the phase C report. Every arm in table C of `warehouse/docs/phaseC_language.md` has its full count of episodes (200, or 100 for q7).

## How to read this file

- **Row format:** where, what it assumes, why, and what changes if it is different, with impact high, medium or low. One row per assumption.
- **"(inferred)"** marks a reason not documented in the code or reports.
- **Discrepancy ids** (M-, L-, R-, W-, S-) point to section 10.
- **Coverage:** `tsp/src`, `tsp/scripts`, `warehouse/src`, `warehouse/scripts` and the reports in `tsp/docs` and `warehouse/docs` were read by five auditing agents, one per area. Warehouse tests were read as assert lists. Warehouse phase C has no report yet.
- **Verification:** the claims behind list (a) and the main verdicts of list (b) were re-checked against code and stored results after the drafts were written (next table). The other rows are the auditors' reading and were not each re-checked. Citations were checked to exist by script, not for meaning.

| Re-checked claim | Result |
|---|---|
| `elkai.solve_int_matrix` is TYPE ATSP | Confirmed in the source and in the installed `_elkai.so` |
| The window model gets no end marker | Confirmed, `hier.py:Heatmaps.__call__` |
| Selected epochs of the phase 3 models | Confirmed from `eval.json`: 12, 1, 4; 1, 1, 10; 5, 3, 1 |
| k = 0.5n restores the training edge scale | Confirmed: directed k-NN mean edge relative to TSP-20 is 0.98 at n = 50, k = 25 against 0.59 at k = 10, and 0.97 at n = 100, k = 50 against 0.40 at k = 10 |
| Fu-10k: about 183 windows improve per instance | Confirmed from `e2/oracle`: 2934 improved over 16 instances (top 50 %); 3000 calls follow from 1000 windows per pass over 3 passes |
| T2 batched arm is not saturated at x2.0 | Confirmed: realised arrivals 1029 / 1896 per h, arm (a) second half 995 / 1888, backlog mean ≈ end. The one-order arm is saturated (604 / 1131, backlog growing) |
| T1 "walking" includes services | Confirmed, `batching.py:47` |
| o4w5 and o4w10 also change `execute` | Confirmed, `phaseB0_margins.py:164` |
| `run_phaseB0.sh` calls a stage that no longer exists | Confirmed: `online` is not among the choices |
| Soft penalty is read as travel time in TSPTW | Confirmed, `rules.py:cost_matrix` and `Penalty_TSPTW.c:20` |
| Hybrid code was committed after the report with test scores | Confirmed: 25dc6cd 05:58, 9920ae0 09:07 on 2026-11-01. Whether the trigger was fixed earlier cannot be shown from git |
| Phase 5 headline at instance level | From the statistics audit: p ≈ 0.0015 against 0.0003 at rule level; it survives |

One estimate was corrected at this stage. The audit first bounded the LKH calls of a window memo at 1366. An improved window also changes itself, so a memo keyed on the exact window order gives at most 1000 + 3 × 183 = 1549 calls. Treating a window that LKH just solved as clean gives 1366. So the saving is about 48–54 %, and this has not been measured.

## Corrections to the premises of the request

| Item in the request | What the code does |
|---|---|
| k = 0.5n | True up to n = 200. Above that k = min(0.5n, 100), and k = 10 in 20-city windows. Training k is always 10 at TSP-20 |
| k = 0.2n ablation | Test time only, on models trained with k = 10 |
| LKH-3 labels | Labels up to n = 100 come from elkai 2.0.1 (modified LKH 3.0.8) through an ATSP path. LKH-3.0.6 is used for E0, E1, phase 5 and the warehouse |
| Time limit on 2-opt | None on window or dense 2-opt. 600 s only in `scale_eval`, `scale_baselines` and `hier_eval` |
| Distance normalization `sqrt_n` | Exists, never trained or evaluated |
| Bootstrap | Not present anywhere. Intervals are Wilson (phase 5, phase C) and Student t over seeds (warehouse). TSP tables have no intervals |
| 3 % criterion | No decision code and no plan file in the repo. It first appears in B0b, after the B0 data. It is a dashed line in figures |
| Versions | LKH 3.0.6 and private Ollama 0.40.4 confirmed. The shared server is 0.34.0. PyG 2.8.1, torch 2.14.1+cu130. No lock file |
| "28 % label error" | L2 of TSP step 3 only (17/60). L0 is 7.5 % on a random sample, L1 was never audited, warehouse L2 is 21/60 = 35 % |
| Retry on invalid JSON | One retry, only in `open_formalize` and `phaseC_language`. None in step 2, the verifier or the probes |
| Warehouse p0 | New code that keeps only the method of the old parser |
| Tardiness, due dates (A–B0c), warm-up, depot, deadlock detection | Not present in the code |
| Free flow as lower bound | It limits one fixed plan, not the problem. The only true bound is `bound()` in `phaseB0_margins.py`, for one order per trip |
| Capacity | Counted in orders, never in products |
| CBS | Plain CBS with a window, next goal only, no heuristics |
| Warehouse phase C report | Does not exist yet; the run was live during the audit |
| Reports in `docs/` | There is no root `docs/`. Reports are in `tsp/docs/` and `warehouse/docs/`; this file is the first in a root `docs/` |

One inspection went beyond the audit's rules: to read the version, an auditor ran `/scratch/pcanaste/tools/ollama/bin/ollama --version`, outside the project folder, and with no `OLLAMA_HOST` it sent one version request to the shared Ollama on port 11434. It changed nothing.

## 1. Data and instances

Paths are relative to `tsp/`; `data.py`, `graph.py`, `hier.py` and similar are in `tsp/src/tsp_gnn/`.

### Instance inventory (from `.npz` metadata, measured)

| Set | Instances | n | Seed | LKH path | RUNS |
|---|---|---|---|---|---|
| tsp20_train / val / test | 12000 / 500 / 1000 | 20 | 1 / 2 / 3 | matrix, ATSP | 10 |
| tsp20_train100k | 100000 | 20 | 8 | matrix, ATSP | 10 |
| tsp50_train / val / test | 12000 / 500 / 500 | 50 | 6 / 7 / 4 | matrix, ATSP | 10 |
| tsp100_train / val / test | 12000 / 500 / 500 | 100 | 19 / 20 / 5 | matrix, ATSP | 10 |
| tsp200 / 500 / 1000 _test | 100 / 50 / 20 | as named | 9 / 10 / 11 | coords, EUC_2D | 10 |
| tsp2000 / 5000 / 10000 _test | 10 / 5 / 3 | as named | 12 / 13 / 14 | coords, EUC_2D | 1 |
| Fu tsp1000 / tsp10000 _test_concorde | 128 / 16 | as named | unknown (not in file) | E0 LKH-3.0.6 reference | 10 / 5 × 696 s |

### Assumptions

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| src/tsp_gnn/data.py:generate_instances:16 | Cities iid uniform in [0,1]², `rng.random((num,n,2))` | Standard benchmark (Joshi, Kool); stated in `fase1_explicacao.md` §1 | High: clustered or real instances are never tested at any phase; every gap and the 1/√n scale argument are conditional on uniformity |
| data.py:generate_instances:18 | Independent sets because each has a distinct seed (1–14, 19, 20) | `fase1_explicacao.md`: "Seeds diferentes ... não há instâncias repetidas" | Low: verified, no shared seed, so no instance-level train/test leak |
| data.py:generate_instances:19 | No filter for duplicate or near-duplicate cities | (inferred) negligible probability | Low: `scale.neighbor_lists:32` handles duplicates anyway |
| scripts/train_model.py:44 | 12,000 training instances suffice (fixed, precomputed, no regeneration) | Phase 1 budget; `README.md:164` | High: 100k gives TSP-20 greedy −1.43 pp but no TSP-50/100 gain (`phase2_datasize_report.md`); Joshi 2019 used far more (inferred from memory of the paper, not in repo) |
| scripts/train_model.py:44-45 | Train and val at one single size; no mixed-size training | Phase 1 scope (memory `phase-scope.md`) | High: generalization to n ≠ 20 rests on one size; mixed-size training never tried |
| scripts/run_phase2_datasize.sh:4 | 25k and 50k are prefixes of the 100k file; 12k is a different file (seed 1) | Nested sets; noted in report decision 1 | Low: 12k→25k changes instances too; noted |
| scripts/run_phase2_datasize.sh:6 | Epochs fixed at 20, so more data = more steps (3,760 → 31,260) | Noted in report | Medium: data and training length confounded; "more data does not fix TSP-50" is not separated (a fixed-step control is proposed in the report, never run) |
| data.py:solve_lkh:28 | Labels = single LKH tour, `runs=10`, defaults (SEED 1, MAX_CANDIDATES 5 alpha, MOVE_TYPE 5, MAX_TRIALS default); no TIME_LIMIT | `third_party/download/elkai-*/LKH-3.0.8/SRC/ReadParameters.c:514-576` | Low at n ≤ 100: near-optimal; MEDIUM as a claim: "optimal" never certified above n = 7 |
| data.py:solve_lkh:31 + elkai `deprecated.py:5` + `LKHmain.c:ElkaiDeprecatedSolve:212,234` | `solve_int_matrix` is treated as a plain TSP solve; in fact it always passes `TYPE: ATSP` (a symmetric problem doubled by LKH's ATSP transform) | Not documented anywhere; `data.py:41` says "Same problem" | Low for label quality; MEDIUM for time: every LKH window uses this path (see Group 2) |
| `_elkai.c:PyDeprecatedSolve:99,114` | Integer distances survive a cast to C `float` (32-bit) | Not documented | Low: exact below 2²⁴ ≈ 1.68e7; max distance ≈ 1.41e6, LKH_BIG = 1e7 both safe. Raising SCALE to 1e8 would silently corrupt labels |
| data.py:13 SCALE | Rounding at 1e-6 is negligible; `solve_lkh` rounds distances | Comment line 11-12 | Low: at n = 10k edges ≈ 7,200 units, ±1e-4 relative per edge, averages out |
| data.py:solve_lkh_coords:38 | Coordinates rounded then EUC_2D nint, called "Same problem as solve_lkh" | Needed above matrix size | Low: optimum equal up to rounding; tested only at n = 30, rtol 1e-6 (`tests/test_data.py:53`); but LKH type (TSP vs ATSP) differs |
| data.py:SCALE/solve_lkh + scripts/make_scale_data.py:25 | Two label paths give comparable references across n (matrix ≤ 100, coords ≥ 200) | Memory limit of matrix | Low: gap curves cross the path change at n = 100→200 |
| data.py:solve_lkh:33 | One tour per instance is the target; rotation irrelevant | Random continuous instances have a unique optimum almost surely | Low |
| scripts/run_phase2_scale.sh:10 | RUNS = 10 up to n = 1000, 1 above; reference "slightly weaker" | LKH time (36 min at 10k); noted in scale report | Low at 10k (E0 agrees to ±0.002 %); UNTESTED at 2000/5000: E0 never recomputed them, the "gaps stand" note covers 1k and 10k only |
| data.py:tour_length:54 | "Optimal length" = float length of the LKH tour; gap to that | Avoids rounding artefacts | Low |
| src/tsp_gnn/graph.py:instance_to_data:83 | `opt_len` stored as float32 | (inferred) oversight; noted in scale report Table 9 ("divided by the LKH length in float32") | Low: up to 1e-4 pp between dense and sparse evaluations |
| graph.py:instance_to_data:73; scripts/scale_eval.py:158 | Models and decoders see float32 coords while `opt` is on float64 coords | "as in the phase 1 decoders" | Low: at most ~1e-5 relative per edge; `scale_baselines.py:40` mirrors it |
| tests/test_data.py:45 | "LKH = exact optimum" evidence is 5 instances with n = 7, plus means vs literature (3.83, 5.70) | `README.md:165`, `fase1_explicacao.md` §2 | Medium: no exact check at 20, 50, 100; TSP-20/50 means can hide per-instance sub-optimality; labels noise cannot be bounded |
| scripts/lkh3_reference.py:solve_one:47 | E0 is a stronger, independent reference: LKH-3.0.6 binary, 1k RUNS=10, 10k RUNS=5, TIME_LIMIT=696 s per run | Benchmark caveat 1 | Medium: same defaults, same SEED; on tsp1000_test 19 of 20 tours are identical to elkai's (measured), so agreement proves determinism, not optimality |
| src/tsp_gnn/lkh3.py:21 + scripts/lkh3_reference.py:32 | LKH-3.0.6 from the cerebis/LKH3 mirror (commit ba346db) is equivalent to official LKH | Official site unreachable; caveat 1 | Low: lengths agree with elkai's 3.0.8 |
| scripts/run_phase3_bench_e0.sh:59-62 | 10k reference near-optimal; run-to-run spread ≈ 1.4e-4 relative (0.014 %, measured from `run_costs`) | Caveat 1 | Low for our 4 % gaps; MEDIUM for E1's 0.08–0.15 % gaps, where 0.014 % is 10–20 % of the value |
| scripts/bench_datasets.py:main:40 | E0 replaces stored lengths even when stored is shorter | Simplicity | Low: on tsp10000_test E0 is longer on 2 of 3 instances (−0.0005, −0.0009 %, measured) |
| scripts/convert_benchmark.py:parse_line:23 | Fu files: `x1 y1 ... output t1..tn t1`, 1-indexed, closed; source Spider-scnu/TSP (MIT) | Module docstring | Low: validity asserted; sha256 stored (1k `605a1508…`, 10k `61c46caa…`, measured) |
| scripts/convert_benchmark.py:51 | 10k file has no reference (identity tour) | Docstring; mean length 5211.7 = identity on random points (measured) | Low: correctly flagged `has_reference = 0` |
| convert_benchmark.py (docstring only) | Upstream commit and generator seeds of Fu files are not recorded | Not present | Low: sha256 fixes content; download command not in repo; seeds "not present" |
| data/benchmark/*.npz | Fu instances are uniform like ours | (inferred) mean/std 0.4996/0.2885 (1k), 0.5001/0.2889 (10k), measured; ideal 0.5/0.2887 | Low: consistent; no formal test |
| phase3_benchmark_report caveat 2 | Fu 1k "concorde" tours are 0.47 % worse than E0 (0.27–0.68 %) | Measured in E0, confirmed from `ref_lkh3.npz` | Medium: any literature gap computed against the file shifts 0.47 pp; published methods use their own references |
| phase3_benchmark_report comparison table | Published rows are comparable though each uses its own reference, hardware, time accounting | Caveats 1, 4, 8 | Medium: "beats Fu 4.39 % at 10k" mixes references and a 1.7 h vs 20 s budget |
| scripts/bench_datasets.py:42 | Output has no `tours`, so `hier_eval.py:85` cannot run on it | E3 needs lengths only | Low: hierarchy never evaluated on Fu sets |
| scripts/run_phase2_scale.sh:8 | 3 instances at n = 10000, 5 at 5000, 10 at 2000 are enough | LKH time | Medium: no tests below 20 instances (reports say so); Fu sets partly fix it at 1k and 10k only |
| docs/fase1_explicacao.md:165 region; scripts/*.sh | Test sets are never used to choose anything | Phase 1 rule | Medium: k = 0.5n, TAGCN K=3, B and w were all chosen after reading test results on those sets (see section 10.1) |

## 2. Graph and model (TSP)

### Measured evidence used below (light NumPy inspection, edge means over undirected pairs)

| n | k | Mean graph edge | Mean tour edge | Tour / graph | Tour coverage | Degree | Positive fraction |
|---|---|---|---|---|---|---|---|
| 20 | 10 | 0.3861 | 0.1915 | 0.496 | 1.0000 | 12.1 | 0.166 |
| 50 | 10 | 0.2209 | 0.1136 | 0.514 | 0.9988 | 11.9 | 0.168 |
| 50 | 25 | 0.3720 | 0.1136 | 0.305 | 1.0000 | 29.9 | 0.067 |
| 100 | 10 | 0.1489 | 0.0781 | 0.525 | 0.9996 | 11.8 | 0.170 |
| 100 | 50 | 0.3649 | 0.0781 | 0.214 | 1.0000 | 59.4 | 0.034 |
| 200 | 100 | 0.3616 | 0.0533 | 0.147 | 1.0000 | 118.6 | 0.017 |
| 1000 | 10 | 0.0429 | 0.0231 | 0.539 | 0.9991 | 11.5 | 0.174 |
| 1000 | 100 | 0.1394 | 0.0231 | 0.166 | 1.0000 | 111.8 | 0.018 |

Windows of 20 cut from LKH tours of tsp1000, after `rescale`, vs uniform TSP-20: short-side span 0.647 (p10 0.356) vs 0.873; consecutive-edge length 0.126 vs tour edge 0.192; nearest-neighbour distance 0.091 vs 0.124; end-to-end distance 0.895 (ends sit at opposite extremes).

### Assumptions

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| graph.py:resolve_k:24 | Training k = 10 at TSP-20 (half the cities); `--k-frac` optional | `README`, phase 1 | Medium: k = 4 (0.2n) training never tried |
| graph.py:resolve_k:32; scale_eval.py:k_of:49; hier.py:Heatmaps:155 | `round()` is Python banker's rounding: round(10.5)=10, round(11.5)=12 | Not considered | Low: only odd short last windows |
| scripts/scale_eval.py:k_of:47 | Test k modes: k50, khalf100 = min(0.5n,100), khalf, k02n, k10 | Cap for memory and time | High: above n = 200 the cap breaks the edge-length match (mean edge 0.139 at n=1000 vs 0.386 in training, measured) |
| scripts/evaluate.py:30 and beam_eval.py:74 | Denser test graph can be applied to a model trained at k = 10 with no retraining | Exp. 2, batch 2 | High: degree 30–119 vs 12; positive fraction falls to 0.017–0.067 vs 0.166; works for ranking only, probabilities not calibrated |
| scripts/run_phase2_k02n.sh:6-8 | k = 0.2n test-time ablation, models fixed at training k = 10, 2-opt lists fixed at khalf100 | Isolates network graph; Joshi 2021 correction in `literature_review_tsp.md:7` | Medium: says nothing about training with k = 0.2n, nor about windows (which always use k = round(0.5w)) |
| hier.py:Heatmaps:155 | k = round(0.5m) hard-coded, ignoring checkpoint `k`, `k_frac` and `edge_norm` | Equals k = 10 at m = 20 | Low in practice (all TAGCN runs edge_norm none); silent if used with a knn_mean model: no assert, unlike `scale_eval.py:172` and `heatcand.py:27` |
| scale.py:knn_graph:47 | Same edges as `graph.knn_edges` | KD-tree for large n; `tests/test_scale.py` | Low: differs only on exact ties |
| graph.py:knn_edges:46-47 | Symmetrized union (i→j and j→i), degree > k | k-NN not symmetric | Low |
| graph.py:tour_labels:50 | Tour edges missing from the graph simply do not exist as targets | Coverage ≈ 1 at k = 10 on TSP-20 | Low: coverage 1.0000 at n=20, 0.9988 at n=50, k=10 (measured) |
| decode.py:greedy_decode:76 | Closing edge forced and unscored, even outside the graph | Guarantees validity; `fase1_explicacao.md` §7 | Low: can be a long edge; 2-opt repairs |
| models.py:EdgeGNN:84 | Node features = raw (x,y) only; no degree, no k-NN indicator, no end marker | Phase 1 design | High: absolute coordinates make the model non-invariant to rotation and reflection; no augmentation exists anywhere (grep: none) |
| models.py:EdgeGNN:85 | Edge feature = length only (1 dim) | Phase 1 | Medium |
| models.py:EdgeGNN:118 | GCN and TAGCN encoders ignore distances; only the head sees them; `e` never updated outside Gated | `tagcn_report` Decisions | Medium: for these, `edge_attr` normalization only touches the head, so knn_mean barely reaches the encoder |
| graph.py:normalize_edges:64 | knn_mean rescales `edge_attr` only; `x` keeps its raw scale | Hypothesis in CLAUDE.md was stated on edge distances | High: neighbour coordinate differences still shrink by √(20/n), so the scale hypothesis was only half tested |
| graph.py:normalize_edges:65 | Mean over the instance's own k-NN edges is a scale proxy | Report decision 3 | High: with k = 0.5n the mean is almost constant in n (0.386, 0.372, 0.365, 0.362), so knn_mean + kfrac is a near no-op by construction |
| scripts/evaluate.py + graph.py:instance_to_data | "k effect" and "distance effect" are separable | Batch 2 summary points 1–2 | High: k = 0.5n restores the training mean edge (ratio 0.96 vs 0.57 at k = 10), so kfrac confounds density with scale |
| graph.py:EDGE_NORMS:21 | `sqrt_n` available | Listed as alternative in `fase1_explicacao.md` | Medium: never run, so "normalization does not help" rests on knn_mean alone |
| models.py:make_norm:33; train.py:eval_loss:46 | BatchNorm eval uses running stats from TSP-20 k = 10 at every n and k | Phase 1 default | High: bn_batch helped 5/6 TSP-100 cells; drift grows with n; never combined with k = 0.5n |
| models.py:with_batch_stats:139 | Batch stats over the 256-graph test batch (transductive), k = 10 only | Report decision 5 | Medium: not deployable as is; per-instance variant never tried |
| models.py:make_norm:37 | LayerNorm is a drop-in replacement, 20 epochs enough | Exp. 3 | Medium: val loss still falling at epoch 20, so LN may be under-trained; memory `phase2-datasize.md` says it did not converge |
| hier.py:Heatmaps:141 and train.py:predict_probs | Batching does not affect output in eval | Report decision 4 | Low: true for BN in eval and LN; timing only (`wsize_eval.py:batch_size:36`) |
| train.py:compute_pos_weight:22; train_model.py:64 | pos_weight = neg/pos on training graphs: 5.048 (TSP-20), 13.922 (TSP-50 k=25), 28.698 (TSP-100 k=50) | Class balance, `fase1_explicacao.md` §6 | Medium: inflates p; never ablated. Beam scores (sum log p) and LKH alphas `round(1000(1−p))` assume p is meaningful |
| train_model.py:95,102 | Selection = lowest greedy val gap, no 2-opt, val at the training n and k | Phase 1 | High: metric differs from every downstream use (beam path, 2-opt, other n); `tagcn_report` shows it is noise |
| train_model.py:105 | `best.pt` used in all phase 3 work: TAGCN K=3 epochs 12, 1, 4; TSP-50 1, 1, 10; TSP-100 5, 3, 1 (read from `eval.json`) | Phase 1 protocol kept | High: seed 1 of the main model has one epoch of training; last.pt differs (val 7.0–7.2 vs 5.7–6.4) |
| train_model.py:29-33 | epochs 20, batch 64, Adam lr 1e-3, cosine T_max 20 (per epoch), no weight decay, dropout or clipping, hidden 64 | Budget and memory cap | High: all conclusions of "the GNN cannot" hold only for this capacity |
| models.py:default_layers:24 | GCN/GAT/Gated 6 layers; TAGCN round(6/K): K=3 → 2 layers, 6 hops | "same reach"; `tagcn_report` Decisions | Medium: at k = 0.5n, 2 hops already span the graph, so hops do not mean equal reach; parameters unequal (TAGCN K=3 50,049, Gated 143,233) |
| models.py:EdgeGNN:97 | TAGConv normalized Â = D^-1/2 A D^-1/2, no self-loops, no edge weights | PyG default; noted | Low |
| models.py:GatedGCNLayer:42 | Called Joshi 2019 residual gated layer | README | Low: sparse k-NN with no edge-type embedding, normalized "soft" aggregation (inferred difference from memory of the paper) |
| train.py + GPU `scatter` | Same seed is not bit-reproducible | `README:226`, `run_tagcn_ckpt.sh` | Medium: 2 of 6 reruns reselect the same epoch; 3 seeds measure this too |
| decode.py:edge_heatmap:21 | Undirected p = mean of two directed probabilities (not logits) | Simple symmetrization | Low |
| decode.py:greedy_decode:45 | Greedy order: p descending, ties by length; outside-graph pairs by distance | `fase1_explicacao.md` §7 | Low |
| decode.py:beam_search_decode:111 | Beam over Kruskal-style edge sets, not node-by-node sequences; ranks by sum of log p | "same edge choices as greedy" | Medium: partial solutions are unions of fragments; differs from Joshi 2019's sequential beam (inferred from memory of the paper) |
| decode.py:beam_search_decode:180 | Final pick = shortest of B tours; closing edge not scored | Stated "as in Joshi 2019" | High: search + length selection, not the network, may carry much of the gain: beam B=10 gives 0.49 pp vs greedy 0.15 pp from the same heatmap at w=20, n=1000 |
| decode.py:89 | Outside-graph log p = log 1e-12 | Floor | Low |
| scale.py:beam_search_sparse:240 | Equivalent to dense beam; B=10 up to n=1000, B=100 only at n=200 | `tests/test_scale.py`, `scale_eval.py:141` | Low: hash change only; sizes differ per n |
| beam_eval.py:decode_dataset:57 | 2-opt is applied only to the selected tour, not to all B | Cost | Medium: 2-opt on all B would give a larger gain |
| decode.py:two_opt:183 | Dense 2-opt: for each i, best j, loop until no change; no iteration or time limit | Phase 1 | Medium: README:202 reports 2.02 / 1.31 / 2.59 by variant, i.e. up to 1.4 pp from the variant alone |
| scale.py:two_opt_neighbors:308 | Sparse 2-opt: k lists, don't-look queue, best of succ/pred per city, 1e-9 tolerance, no Or-opt | O(n·k); scale report | Medium: dense and sparse are different local optima, so phases are not comparable (noted) |
| scale_eval.py:13,131 | 2-opt k = same as graph k (k-mode), 600 s per call; timeouts → NaN, excluded from mean | Scale report; none occurred | Low: k and 2-opt confounded in scale tables; fixed in k02n |
| window.py:initial_tour:32 | Start = greedy edge + 2-opt with k = min(0.5n,100), no time limit | "baseline of the scale experiment" | Medium: k = 100 is generous, inflating its time (0.26 s / 2.82 s) and hence budgets; no 3-opt or Or-opt baseline, so 4.6–4.8 % is a weak start |
| window.py:window_starts:44 | Consecutive disjoint windows, last shorter, skipped if < 4 | Disjoint = batchable | Low |
| window.py:refine:120 | One pass = offsets 0 and w//2; second half-pass sees updated tour | Module docstring | Low |
| window.py:refine:129 | Accept only if strictly shorter by 1e-9 | "tour never gets longer" | High for interpretation: model failure is free, so every model gain is a selection effect; significance tests can only say "better" (report notes this) |
| window.py:refine:117 + scripts | max_passes = 5 (`window_eval`), 2 (`wsize_eval`), ≤ 3 (triage), 1 (triage window logs) | Budget | Medium: gains are not comparable across reports; LKH w=300 gets 1 pass vs model 2 |
| hier.py:rescale:131 | Same factor on both axes, min corner at 0; "the training scale" | Module docstring; window report line "at the scale it was trained on" | High: windows are off-distribution even after rescaling: short side 0.65 vs 0.87, edge 0.126 vs 0.192, NN 0.091 vs 0.124 (measured) |
| hier.py:Heatmaps:155 + window.py:make_solver:95 | Heatmap of a closed-tour model is useful for a fixed-end path; network receives no marker for s and t | Not documented, no report mentions it | High: trained on closed tours of uniform points, used on open paths with far-apart ends; a candidate explanation of the 0–2 % capture at w ≥ 100 |
| hier.py:greedy_path:207, beam_path:248 | Ends enforced by decoder only (cap 1, no s–t merge before last edge) | Docstrings | Low: decoder is sound; beam cannot die (each non-end fragment keeps free ends) |
| hier.py:_path_candidates:185 | `p > 0` identifies graph pairs | Compact | Low: float underflow unlikely |
| window.py:lkh_path:57 | `runs=1`, elkai, dummy node, rescaled window × 1e5 | Module docstring | Medium: heuristic, called "exact optimum" in the window report; goes through the ATSP path (dimension doubled), so `ms_lkh` may overstate LKH cost |
| window.py:28-29 | LKH_SCALE 1e5; LKH_BIG 1e7 "> any path through the window" | Comment | Low: holds for real tours (w=300 path ≈ 1.2e6 units, inferred); assert at line 70 catches violations |
| window.py:two_opt_path:74 | 2-opt on the current path is the "no model" control | Module docstring | High: gains exactly 0 because the tour is already 2-opt optimal, so it does not measure the network; no distance-only heatmap control exists |
| hier.py:Heatmaps:157 | `seconds` counts forward passes only; k-NN building in Python is outside | Timer position | Medium: "Network time: only 0.2–0.3 s" excludes graph construction |
| wsize_eval.py:main:113 | LKH limited to 1 pass if first pass on instance 0 > 5 s per window | Budget | Low: applied only at w = 300 |
| hier.py:balanced_kmeans:34 | Groups 10–30, sklearn n_init=3, seed 0 | Report setup | Low |
| hier.py:kdtree_partition:79 | Groups ≤ 30 (15–30) | Report setup | Low |
| hier.py:Hierarchy:102 | GROUP=20, TOP=60, centroids are points of next level | Model trained at 20 | Medium: centroids not uniform; top level runs on up to 60 points |
| hier.py:hierarchical_tour:405-406 | Top uses `greedy_sparse` + dense `two_opt` | Small | Low |
| hier.py:attraction_matrices:368 | Heatmap attraction on cities only if ≤ 100, else centroids | HEATMAP_MAX_CITIES | Low: n.s. overall |
| hier.py:hierarchical_tour:425 | One visit per group | Design | High: report's own finding (LKH enters a k-means group 2.7–2.8 times) |
| hier_eval.py:76,101 | Final 2-opt k = min(n−1, 0.5n, 100), 600 s | Same as baseline | Low |
| heatcand.py:instance_heatmap:35 + lkh3_e1_prep.py:76 | Candidates from the TSP-20 model on the whole instance, k = min(0.5n,100), raw coordinates, seed 0 only | E1 | High: tests the model where it is known to fail (mean edge 0.36× training at n=1000), one seed |
| heatcand.py:top_candidates:61 | alpha = round(1000(1−p)) preserves order | Module docstring | Medium: ties at 3 decimals; p inflated by pos_weight |
| wsize runs (`run_phase3_wsize.sh:70`) | TSP-50/100 windows models keep 12k instances, 20 epochs, 2 layers, hidden 64 | "phase 1 protocol" | High: val gap 9.5–13.5 %; report calls it "not a weaker model" based on whole-instance + 2-opt |

## 3. Triage and selector

`src/` is `tsp/src/tsp_gnn/`, `scripts/` is `tsp/scripts/`, `docs/` is `tsp/docs/`, `results/` is `tsp/results/`, and Fu-1k / Fu-10k are `tsp1000_test_concorde` / `tsp10000_test_concorde`. "Computed" means read from existing result files.

### 3a. Signals, windows, top-x %

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| src/triage.py:segment_disagreement:38 | Disagreement = 1 − mean of `heat[i,i+1]` over the m−1 edges of the current segment, order as in the tour | docstring: cheap, only needs the network | Medium: other aggregations (min, max) exist only as learned features; `p_mean` carries most of the signal (docs/phase3_learned_triage_report.md, importance 0.052) |
| src/hier.py:Heatmaps.__call__:155-172 | Edge outside the window k-NN graph (k = round(0.5 m) = 10 at w = 20) has p = 0; p of an edge = mean of both directions | documented in triage.py:6-7 | Low-medium: a long edge is punished twice; with k = 10 of 20 few path edges fall out |
| src/hier.py:Heatmaps.__call__:155 | The window is given to the net as a plain closed TSP instance, rescaled to [0,1]²; no feature marks the two fixed ends or that the problem is an open path | (inferred) reuse of the TSP-20 net without retraining | High for signal meaning: p answers "is this edge in the closed tour of these 20 points", not "in the best path between the ends"; AP only 1.8× the base rate may be partly this |
| src/hier.py:rescale:131 | Same scale factor on both axes, so a long thin window keeps its shape and uses only part of [0,1]² | docstring | Medium: distance scale still varies per window; not ablated |
| src/window.py:window_starts:44 | Windows are consecutive, disjoint in a half-pass, last one shorter, kept only if length ≥ 4 | window docstring: solved together | Low: a tail of < 4 cities is skipped in that half-pass |
| src/window.py:refine:120 and src/triage.py:ranked_refine:253 | Fixed offsets 0 and w // 2; windows always cut at the same tour positions in every pass | window docstring | Medium: boundaries never move in later passes |
| src/triage.py:ranked_refine:221, scripts/triage_ablation.py:58 | `max_passes = 3`; stop only if a whole pass gains ≤ 0 | (inferred) cost; window.refine uses 5 | High: see 3c, most calls of passes 2-3 cannot gain |
| src/triage.py:top_fraction:211 | Exactly `max(1, round(frac × W))` windows per half-pass, ties broken by position (stable sort); x % applied per half-pass, not per instance | docstring: same number of LKH calls per policy, comparison "only about the ranking" | Medium: a half-pass with few useful windows still spends x % |
| src/triage.py:ranked_refine:285-299 | Selection has NO memory: a window that LKH already solved and that did not change is eligible again and, for a deterministic score, keeps its score (src/fast_triage.py:WindowBatchScorer.values:152-154 caches exactly that score) | not documented; never mentioned in any report | High: deterministic scores re-send the same dead windows, random does not; see list (a), item 3 |
| src/triage.py:choose_thresholds:117, scripts/triage_eval.py:40-50 | P2 / P3 thresholds = quantiles of disagreement on tsp200 of each seed, fixed and reused at n = 1000, 10000 | triage report: tsp200 is validation | Medium: actual share falls to 14-16 % at top 20 %, report says so; E2 replaced this with exact top-x |
| src/triage.py:choose_thresholds:122-124 | Confidence signal = logp or margin by AP for target (ii), threshold = best F1 | module docstring | Low: P3 is dominated anyway |
| src/triage.py:CATCH:35 | "Model catches" = model gain ≥ 0.8 × LKH gain | documented constant; 0.8 not justified | Low |
| src/triage.py:simple_rule:202 | Segment length / sum of NN distances is the geometric baseline | docstring heuristic | Medium: it beats random in AP (0.288 vs 0.215) but loses to random in gap at both x; likely the re-selection effect |
| src/learned_triage.py:FEATURES:26 | 9 features: p_mean, p_min, p_max, disagreement, seg_len_nn, max_edge_nn, density, position, out_knn5; p_mean and disagreement are exactly collinear | docstring: scale-free, no beam, no LKH | Low: redundant; report notes it |
| src/learned_triage.py:window_features:62 | `position` = start / n is scale-free and meaningful | docstring | Low: depends on arbitrary tour rotation; importance 0.000 |
| src/learned_triage.py:KNN:29, DENSITY_RADIUS:30 | 5 nearest neighbours; density radius = 2 mean NN distances | hard-coded, no justification | Low |
| src/learned_triage.py:instance_context:39 | Column 0 of the KD-tree query is the city itself | comment | Low: false for duplicate points; global_knn_graph handles it (fast_triage.py:180), this one does not |
| src/fast_triage.py:batch_knn_graph:41 | Batched graph equals the per-window graph | docstring; verified: Spearman 1.0000, 108/108 gaps equal (fast report) | None: verified |
| src/fast_triage.py:GlobalScorer:200 | Version B: net on whole instance, k = min(0.5 n, 100), raw coordinates | docstring says it is a different signal | confirmed bad: mean p 0.9-0.99, Spearman 0.02-0.09 |
| scripts/fast_triage_decide.py:THRESHOLD:10 | Version B goes on only if Spearman ≥ 0.5 on 2 instances × 3 seeds | hard-coded gate, no justification | Low: B learned passed at 0.58-0.59 and lost anyway |

### 3b. Label, train / test split, selection of the model

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| src/triage.py:IMPROVE_TOL:34, src/learned_triage.py:labels:66 | "Improvable" = current − LKH length > 1e-9 in instance units | same rule as window.refine | checked, low: computed on seed 0 features, no positive has gain < 1e-3 %, only 4-6 % have gain < 0.1 %. Rounding-noise positives were looked for and not found |
| src/learned_triage.py:labels:69 | Binary label is the target; window gain only for `hgb_reg` | docstring | High: top 20 % of positive windows hold 51-54 % of the gain (computed); AP treats a 0.1 % and a 5 % window alike while the objective is gain |
| scripts/learned_triage_train.py:102,111,119 | Model and hyper-parameters chosen by validation AP of `improves`, also for the regressor | docstring | High: selection metric is not the policy objective (gain captured at x %); wsize report shows AP and policy order disagree at w = 50 |
| scripts/triage_log.py:63, src/wsize_log.py:lkh_log:27 | Labels come only from the first pass on the initial greedy + 2-opt tour, each window alone, nothing replaced, both offsets on the same tour | docstring; report caveat | High: the policy applies the model at passes 2-3 and at offset w/2 on an already changed tour; window populations differ (solved windows do not exist in training) |
| src/window.py:initial_tour:32 | All training and test windows start from greedy edge + 2-opt, k = min(0.5 n, 100) | baseline of the scale experiment | Medium: selector not valid for another start tour; never tested |
| src/learned_triage.py:split_instances:72, scripts/learned_triage_train.py:84-89 | Split by instance, 50/50, seed 0, same for all TAGCN seeds; both offsets of an instance on the same side; assert disjoint | docstring | None: no leak between windows of the same instance. Verified in code |
| scripts/learned_triage_train.py:TRAIN_SETS:40 | Train + validation = tsp200 + Fu-1k; test = tsp1000, tsp10000, Fu-10k. Between sizes: n = 1000 appears in train and test but with different instances | docstring | Low: computed 0 shared instances between tsp1000_test and Fu-1k. n = 10000 is pure extrapolation; log10_n ablation shows no effect |
| scripts/learned_triage_train.py:95-96 | Equal total weight per training set although Fu-1k has 6.4× more windows | docstring | Low |
| scripts/learned_triage_train.py:GRIDS:43 | C in {0.01...10}; chosen C = 0.01 for all seeds and all w, at the grid edge | report says it would not change conclusions, not tested | Low-medium: cheap to extend |
| scripts/run_phase3_learned.sh:49 | Fu-1k is not used as policy test set (half is training) | report caveat | None: correct |
| scripts/run_phase3_triage_wsize.sh:93-98 | w = 50 selector trained on 2960 windows (400 + 2560, metrics.json); chosen type differs by seed (logreg, logreg, hgb_reg) | same protocol | Medium: "Learned" at w = 50 is a mix of model types, report does not say so |
| scripts/triage_eval.py:VALIDATION:35 | tsp200 results of P2 / P3 are in-sample | stated in report | Low: disclosed |
| src/triage.py:targets:83 | LKH path is "the best a window can give" | module docstring | Low: computed, 0.03-0.05 % of windows have LKH longer than current, min −1.30 % (Fu-1k), −1.04 % (Fu-10k) |
| src/heatcand.py and scripts/triage_log.py:33 | Selector inputs always from `best.pt` of tagcn_k3_seed0-2 | triage docstring | Medium: tagcn report says best.pt selection by val gap is noise |

### 3c. Equal-time comparisons and random baseline

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| scripts/triage_ablation.py:102-112 and phase3_*_report.py | Time = `initial__ms` (greedy + 2-opt) + `ms` (ranked_refine, wall by perf_counter), per instance, mean; included in report via `initial__ms + __ms` | docs/phase3_triage_report.md Setup | Low: consistent across reports (grep lines 78, 81, 111, 160, 212, 254) |
| scripts/triage_ablation.py:77-78,113 | NOT included: model load, joblib load, process start, verify_tour, gap computation | (inferred) per-instance cost in a long-lived process | Low at 10k; would matter for a one-shot call |
| scripts/triage_ablation.py:83 | No CUDA warm-up; first call pays it | comment "no warm-up, as for the others" | Low: computed, first instance ms_score 743 ms vs ~300 ms, +30 ms on the mean over 16. E1 does exclude warm-up (lkh3_equal_time.py:132), a small inconsistency in LKH's favour |
| src/hier.py:Heatmaps.__call__:155-166 | `ms_net` covers only forward pass and .cpu(), not graph building in Python | timer starts after graph list | Medium: explains why "network" 0.06 s / 0.27 s while scoring was 0.4 s / 4 s |
| src/triage.py:ranked_refine:258-284 | `ms_score` includes the network for all network scores; oracle time goes to ms_lkh | docstring line 231 | see L-3 |
| src/window.py:lkh_path:57-71 | LKH per window is an in-process C call via elkai: no process start, no file I/O, but a text problem is built and parsed; matrix building in Python is inside ms_lkh | not documented | Medium: compared with E1, which pays fork + file writing, so comparison is conservative for LKH. Cost is 6.2-7.3 ms/window alone |
| src/window.py:lkh_path:66 | `elkai.solve_int_matrix` solves a TSP | module docstring | High for timing: it is `TYPE: ATSP`, doubled to 2(m+1) nodes (elkai/LKH-3.0.8/SRC/LKHmain.c:212-243, ReadProblem.c:1999). The 7 ms is not the cost of a 21-node symmetric solve |
| scripts/triage_ablation.py:76, run scripts | Network on T4 (cuda if available), torch threads 2, OMP 1, LKH single thread, one instance at a time per process | CLAUDE.md, report | High for equal time: GPU-second counted as CPU-second, no CPU-only timing exists; random needs no GPU at all |
| src/fast_triage.py:batch_heat:74-92 | Windows batched: 204 per net call at w = 20 (MAX_NODES 4096), head in 131072-edge chunks, GPU synchronised before stopping the clock | docstring | Low: ms_net varies 78-528 ms per instance because the T4 is shared (computed, seq learned seed 0) |
| scripts/run_phase3_triage.sh, run_phase3_bench_e2.sh:61-68, run_phase3_learned.sh:49-65, run_phase3_fast_triage.sh:93-101 | Parallel protocols differ: triage and E3 4 processes, E2 6, learned 8, fast 7 | CPU reuse | High: time comparisons across reports are not comparable; learned 50 % on Fu-10k is 19.39 s (8 proc), 17.37 s (7 proc), 14.83 s (1 proc) for the same gap |
| scripts/run_phase3_fast_triage_seq.sh, _seq_after.sh | "Clean" measure = one process at a time, policies in fixed order, not interleaved; load went 0.4 → 7-9 after random | report caveat; random repeated at the end | Medium: disclosed; order effect bracketed only for random |
| scripts/run_phase3_triage_wsize.sh:123-140 | w = 20 top 10 % and all ran at load ~1.2, w = 20 top 20 / 50 % reused at load 6-9 | report caveat | Medium: random's interpolation curve mixes two loads; report's "their difference holds" covers same-x pairs only, not the interpolated comparison |
| docs/phase3_triage_wsize_report.md summary, results/phase3_triage_wsize_notes.md | Equal-time = linear interpolation in time along random's curve 10 / 20 / 50 / 100 % | notes | Medium: curve is convex so the chord overstates random's gap, favouring learned; done by hand, no interpolation in any script. Reproduced by hand: −0.147, −0.102, −0.009, −0.148 by hand |
| scripts/phase3_triage_wsize_report.py:NUM_10K:39 | Fu-10k policies on first 8 instances | cost | Medium: min Wilcoxon p = 0.0078; w = 50 top 20 % vs random is n.s. |
| all reports | Time differences are compared as means with no CI or test; Wilcoxon is on gaps only | not stated | High: computed, per-instance time of random top 50 % ranges 9.97-11.37 s (rng 0), up to 12.84 s (rng 1). "+7.5 %" and the 1k "faster" have no error bar. At tsp1000 random 1.43 s > learned 1.36 s with the same LKH calls is noise |
| scripts/triage_ablation.py:108, src/triage.py:260 | Random = uniform `rng.random` per half-pass, `default_rng([s, i])`, same stream for all x; 5 seeds, 3 at wsize | docstring | High: redrawing every half-pass gives coverage: at 50 %, 87.5 % of positions are visited over 3 passes (inferred, 1 − 0.5³). This is why random reaches 88 % of the gain. It is not "no information", it is a coverage heuristic |
| src/triage.py:ranked_refine:278 | Oracle = true LKH gain of the current window, recomputed each half-pass | docstring | Low: equals P1 (182 vs 183 improved at Fu-10k, computed). "Share of oracle" = share of P1 |
| scripts/triage_eval.py:58,92 | P1 "LKH everywhere" means 3 passes over every window, re-solving unchanged windows | window protocol | High: computed, Fu-10k P1 = 3000 LKH calls, 183 improve (6.1 %); Fu-1k 7.9 %. Every budget in E1 and the "LKH on all windows" anchor inherit this |
| docs/phase3_fast_triage_report.md, Diagnostic | Extra LKH time of network policies is "LKH working on harder windows" | report, 7.33 vs 7.76 ms | Medium: supported at seq (6.2-7.0 vs 6.8-7.6 ms/call computed) and by wsize logs (+2-7 %), but in the 7-process run the gap is 7.3 vs 9.3 ms, which is load (mechanism not explained) |
| scripts/run_phase3_triage_wsize.sh:111,132 | "LKH on every window" = random with --fracs 100 | re-use | None |
| docs/phase3_benchmark_report.md caveat 4 | Machine has "8 cores" | see L-5 | Low |

## 4. LKH-3

Same path conventions as section 3. `LKH306/` is `tsp/third_party/LKH-3/SRC/`; `elkai/` is the elkai 2.0.1 source under `tsp/third_party/download/`.

### 4a. Parameters, version, build

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| src/lkh3.py:12, scripts/build_lkh3.sh:12-13 | LKH 3.0.6 (May 2019), mirror cerebis/LKH3 commit ba346db, SHA-256 pinned and verified against the tarball | official site unreachable | Medium: (from memory, not in repo) newer 3.0.x exist; literature uses other versions; TIME_LIMIT semantics may differ |
| third_party/build_LKH-3.log, LKH306/Makefile | `cc` GCC 8.5.0, -O3 -Wall -DTWO_LEVEL_TREE -g, no -march | default Makefile | Low: only warnings; test_pr2392.log 10/10 runs at 378032 |
| src/window.py:lkh_path:66, src/data.py:solve_lkh:31 | elkai 2.0.1 wraps a "highly modified" LKH 3.0.8 (elkai/elkai/_elkai.c comment) | library | Medium: windows use 3.0.8 modified, whole-instance uses 3.0.6, so "LKH on a window" and "LKH-3" are two codes |
| elkai/elkai/_elkai.c:99-117, deprecated.py | `solve_int_matrix` passes ints as 32-bit float, printed `%f` | library | Low: exact below 2²⁴; 1.41e6 and 1e7 are safe. Would break if SCALE grew |
| LKH306/ReadParameters.c:defaults | Defaults everywhere unless set: MAX_CANDIDATES 5, MOVE_TYPE 5, ASCENT_CANDIDATES 50, PATCHING_A 1, PATCHING_C 0, KICKS 1, KICK_TYPE 0, INITIAL_TOUR_ALGORITHM WALK, RECOMBINATION IPT, PRECISION 100, SEED 1, EXCESS 1/n, RESTRICTED_SEARCH 1 (confirmed in ref log) | defaults | Medium: no tuning of LKH beyond candidates; helps conclusion's conservatism |
| LKH306/ReadProblem.c:481, src/lkh3.py:101 | Default RUNS = 10 if unset; all TSP callers set it | code | Low |
| src/lkh3.py:parse_output:60 | TRACE_LEVEL 1 output parsed by regex; missing → NaN or empty | docstring | Low: no NaN found |
| src/window.py:lkh_path:57 | Window solve with `runs=1`, MAX_TRIALS default = Dimension, i.e. (inferred) 2(m+1) = 42 at w = 20 | default | Medium at w = 50, 100: heuristic, not exact; label "improves" and "LKH window gain" are lower bounds of what a window can give |
| scripts/run_phase3_bench_e0.sh:59-62 | 1k: RUNS=10, MAX_TRIALS default 1000, no time limit; 10k: RUNS=5, TIME_LIMIT=696, MAX_TRIALS default 10000 | ~1 h per 10k instance | Medium: at 10k every run stops by time at 3069-3992 of 10000 trials (ref log instance 0) |
| scripts/run_phase3_bench_e1.sh:62-65 | Budgets 0.72 1.25 1.54 1.94 / 8.4 14.2 17.5 21.5 s are hard-coded from own-set P2, P0, P1 | report | Low: Fu-set triage takes 0.80 / 9.07 s, so LKH gets slightly less time. Conservative |

### 4b. Candidates, time limits, seeds, initial tours

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| scripts/lkh3_e1_prep.py:variants:39 | 11 candidate variants: alpha, alpha_p(n/10), alpha_p(n/100), popmusic, nn5, nn8, alpha_nosub, quadrant5, quadrant8, heat5, heat8; plus nnheat5 unprobed | E1 design | Low: QUADRANT 5 picked after looking at all of them on the test sets; margin is huge so selection effect is irrelevant |
| LKH306/CreateCandidateSet.c:47-69 | QUADRANT / NN and candidate file skip the ascent, so Pi = 0 | src/lkh3.py:77 docstring | Medium: at 10k ALPHA needs 120 s and cannot fit any budget; geometric sets are not the LKH default that the literature uses |
| src/heatcand.py:instance_heatmap:35, load_model:27 | Network on the WHOLE instance, raw coordinates, trained on TSP-20; `assert edge_norm == none` | docstring; seed 0 only (HEAT_RUN, lkh3_e1_prep.py:57) | High for "network does not help LKH": fast report later found p saturated on whole instances (mean 0.9-0.99). The test is of a saturated heatmap from one seed, not of the best heat candidates |
| src/heatcand.py:top_candidates:61 | alpha = rint(1000(1−p)); saturated p gives ties | LKH needs ints | Medium: order among candidates lost, LKH tries them in tie order |
| src/heatcand.py:write_candidates:64 | Candidates per city not symmetrised, dad 0 | format | Medium: `MAX_CANDIDATES` not marked SYMMETRIC; (inferred) a WALK start on this sparse graph explains gap1 of 120 % at 10k (e1_prep) |
| src/heatcand.py:hybrid_candidates:78 | nnheat5 alpha = rank by distance, up to 10 candidates; fixed cost estimated, never probed | lkh3_equal_time.py:67-74 | Low: disclosed |
| scripts/lkh3_e1_prep.py:main, results/phase3_benchmark/e1_prep/run_e1_prep.sh | Fixed cost measured on ONE instance (index 0 of own sets), applied to all sets, while E0 used 8 processes | script comment: "absolute times are inflated" | Medium: "does not fit" verdicts and TIME_LIMIT rest on it (ALPHA 1k 1.6 s). Conservative for LKH |
| scripts/lkh3_equal_time.py:162 | TIME_LIMIT = max(0.01, B − fixed − greedy − net) | docstring | Low: computed, floor reached in 0.18 % of Fu-1k cells, none elsewhere; 0 NaN in all fitting cells |
| LKH306/FindTour.c:52, GetTime.c | TIME_LIMIT is per run, checked before each trial, on USER CPU time (ru_utime) | lkh3.py docstring says per run, between trials; does not say CPU | Medium: docs and budgets treat it as seconds of wall; overshoot under load follows; † marks |
| scripts/lkh3_equal_time.py:89 | Compared time is real `total_seconds` = LKH wall + greedy + net | docstring | None: correct, this is the honest figure |
| src/lkh3.py:solve:91,113 | Clock includes writing files, process start, reading output | comment | Low: conservative for LKH; fixed 1.3-1.5 s at 10k (e1_prep quadrant5) |
| src/lkh3.py:solve:71 | `timeout` param exists; no TSP caller passes it (reference, E1, prep, probe) | not stated | Low: `timed_out` path is dead in phase 3; a hang would block forever |
| LKH306/ReadParameters.c:541, LKHmain.c:180 | SEED never set in src/lkh3.py callers, so SEED = 1 for every instance, Seed+1 per run; E1 has a single run per config | default | Medium: E1 cells are one stochastic sample per instance; no seed variance reported at all |
| src/data.py and src/window.py | elkai seeds: same default SEED = 1 (elkai ReadParameters.c:576), so every window solve is deterministic | supports "gaps deterministic" in fast report | positive: makes the dirty-window memo exact |
| scripts/lkh3_equal_time.py:119 | Two starts: LKH's WALK, or our greedy + 2-opt via INITIAL_TOUR_FILE (paying greedy) | E1 design | Low: ours is not better in QUADRANT 5 |
| src/lkh3.py:write_tour:39, src/rules.py:write_initial:331 | Initial tour file is complete; SOP needs end copy; direction with fewer violations chosen | docstring; rules.py:347 says LKH's start is far worse in SOP / TSPTW at n = 1000 | Medium: phase 5 reference depends on pure route as start |

### 4c. Integers, constraints, reference, failures

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| src/data.py:SCALE:13, src/lkh3.py:write_problem:29 | Whole instance: coords × 1e6 rounded, EUC_2D rounds each distance; lengths recomputed in float | "~1e-6 per edge" | Low: gaps of 0.03 % are far above 1e-6 × 1000 edges / 23 |
| src/window.py:LKH_SCALE:28, lkh_path:63 | Window: distances of the rescaled window × 1e5, rounded | comment | Low: resolution 1e-5 of the window span; computed no noise positives |
| src/window.py:LKH_BIG:29 | Dummy edge cost 1e7 > any path | comment | Low: holds at w = 20 (max 2.7e6); theoretical worst case at w = 100 is 1.4e7, real paths far less. Assert at line 70 would fire. ATSP M = INT_MAX/2/100 ≈ 1.07e7, only 7 % above BIG (elkai ReadProblem.c:699) |
| src/rules.py:LKH_SCALE:44, cost_matrix:209 | Phase 5: distances × 1e4, rounded UP | docstring: LKH-feasible is truly feasible | Medium: at n = 1000, mean edge 0.023 is ~230 units, so ±0.5 unit rounding ≈ 0.2 % per edge; reference optimises a distorted metric. Coarser than TSP (1e6) with no stated need |
| src/rules.py:FORBIDDEN:45, cost_matrix:212 | Forbidden edge = big-M 1e6 (soft in LKH), feasibility checked after by check_all | docstring | Medium: at n = 1000 a tour is ~2.4e5, soft zones ×2, so M dominates; tour with M edge is rejected as infeasible, not repaired |
| src/rules.py:write_problem:232-241 | SOP: n+1 nodes, depot copy as end, S[after, before] = −1 → M in Distance_SOP; entry into start = FORBIDDEN | docstring; LKH306/Distance_SOP.c | Low: verified by check_all |
| src/rules.py:solve:359-360, src/open_dsl.py:MAX_DIRECTIONS:41 | Fixed edge in SOP / TSPTW is directed: all 2^k directions solved, best feasible kept; k ≤ 3 | docstring | Medium: cost ×8, each up to TIME_LIMIT |
| src/rules.py:write_problem:254-258 | TSPTW: t1, t2 floored, others [0, 1e9]; ceil distances; LKH penalty, cost = length (not makespan) | docstring | Low: conservative, tiny loss of feasible routes |
| src/rules.py:problem_kind:194, src/open_dsl.py:354-356 | Time window + precedence never combined; in open language such a rule is rejected, logged and dropped, tour solved without it | "LKH-3 has no TSPTW with precedences" (3.0.6 limitation assumed, not shown) | Medium: a real user rule is silently absent from the tour, logged only |
| src/rules.py:solve:356 | Default RUNS=1, MAX_TRIALS=1000, SEED=1; TIME_LIMIT 60 s (n=200) / 120 s (n=1000) | scripts/make_open_language.py:41 | Medium: phase 5 reference is one run, not 10; "reference" is weaker than TSP labels |
| scripts/make_text_rules.py:label:41 | `pure` tour from the same solver with the 1e4 rounding, rules=[] | design | Low: consistent |
| scripts/make_text_rules.py:44-49, make_open_language.py:44-49 | Up to 5 / `--attempts` resamples until feasible; attempts recorded | docstring | Low: computed 200/200 feasible with 0 resamples at both n in text_rules |
| src/rules.py:_run_lkh:290-300 | External timeout kills LKH; a tour already written is still used; result has no timed-out flag, log text dropped from saved records | comment line 295 | Medium: cannot tell afterwards which references came from a killed run. make_text_rules passes no timeout at all despite the hang the comment describes |
| scripts/open_pipeline.py:solve_job:44-50 | LKH exception or infeasible → falls back to the pure tour, marked feasible False, error kept | "one bad rule set must not lose the other solves" | Medium: flagged, so not silent, but a failed solve is scored as the pure route |
| scripts/lkh3_reference.py:solve_one:48, main:98 | No tour → "FAILED" printed, then np.load raises: set fails loudly | code | None: no silent drop |
| src/lkh3.py:solve:107 | Return code of LKH not checked; stale tour detected by ns mtime | docstring; TIME_LIMIT=0 segfault | Low: crash → tour None → NaN in E1; report uses np.mean for gap so a NaN would be visible; 0 found |
| src/window.py:lkh_path | No fallback, no timeout: elkai error or assert aborts the run | code | Low: "0 failures" logged; an in-process hang cannot be stopped |
| scripts/lkh3_reference.py, docs/phase3_benchmark_report.md:E0 | Reference = heuristic, not optimum; RUNS=10 / 5, not certified | caveat 1 | Medium: computed run spread 0.012-0.014 %; best run differs by instance at 10k. HK bound in log: 71246012 vs best 71731837, so proven only within 0.68 % |
| scripts/bench_datasets.py:main:40 | All gaps of E1-E3 use E0 (Fu-10k file had none; Fu-1k file 0.47 % worse) | docstring | Low in pp; high on the ratio "28-110×": it divides by a very small gap, so a reference error δ changes it (with δ = 0.05 pp at 10k: 28× → 21×, 47× → 30×) |
| docs/phase3_benchmark_report.md:E0 table | "tsp1000_test unchanged 0.000 %" taken as confirmation of old reference | caveat 1 | Low: same algorithm, same SEED 1 sequence, not independent evidence |
| src/lkh3.py vs scripts/lkh3_equal_time.py:136 | E0 solved raw float64 coords, E1 and policies use float32 → float64 cast | "as in triage_eval" | Low: effect ~1e-7 relative |
| src/data.py:solve_lkh vs solve_lkh_coords:38 | Labels up to n ~ 200 by ATSP matrix, large sets by EUC_2D TSP; docstring says "same problem" | same objective | Low: same objective, different search space size; tsp10000_test used 1 run at 2070-2340 s |
| warehouse/src/warehouse/sequencing.py:LKH_PARAMS:80, lkh:83 | LKH-3 binary per order: RUNS 1, MAX_TRIALS 50, SEED 1, explicit symmetric matrix, dummy with big = 1 + 2m·max; below 4 stops exact; timeout 60, check=True; default workdir under TMPDIR or /tmp | docstring | Low: failure raises; 50 trials not proven optimal; "reference" for NN / 2-opt gaps |
| warehouse/src/warehouse/traffic.py:52 | Traffic-penalty distances symmetrised because LKH reads symmetric matrix | docstring | Low-medium: real traffic is directional |
| warehouse/src/warehouse/batching.py:cvrp_lkh:122 | CVRP surrogate: pairwise edge = max(0, 2·cost(a,b) − c_a − c_b); SPECIAL, MTSP_MIN_SIZE 0, TIME_LIMIT limit, timeout limit + 280, check=True | docstring | Medium: clipped surrogate of pair cost, only exact for capacity 2 |

## 5. Text rules and DSL

`T/` is `tsp/src/tsp_gnn/`, `TS/` is `tsp/scripts/`, `W/` is `warehouse/src/warehouse/`, `WS/` is `warehouse/scripts/`. "Logs" means read from `tsp/data/phase5`, `tsp/results/phase5` and `warehouse/results/warehouse/phaseC`.

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| T/rules.py:module docstring:4-7, PREAMBLE:49 | Depot = customer 0, start t = 0, speed 1, travel time = Euclidean distance, waiting free, visit time = start of service | Convention of step 1, stated in every text | Medium: service times or waiting cost change every window label and the checker |
| T/rules.py:RULE_TYPES:40 | Exactly 5 rule types: precedence, forbidden_edge, mandatory_edge, time_window, soft_zone | (inferred) what LKH-3 encodes directly | High: the "open language" of step 3 only groups these 5, so openness is in the selectors, not in the semantics |
| T/rules.py:ZONES:41, SOFT_WEIGHT:43 | 5 fixed rectangles (bands 0.25, central 0.25 square), weight 1.0 (zone costs double) | "avoid if possible" = double cost (comment :43) | Low for validity (soft is never broken), medium for cost_gap |
| T/rules.py:sample_types:445, problem_kind:194 | Time window and precedence never in one instance | "LKH-3 has no TSPTW with precedences" (:26) | Medium: excludes realistic mixes; arms writing both are repaired or rejected |
| T/rules.py:sample_rule:402-408 | Each customer in at most one precedence; pair at least n//4 apart in the pure route; violated in a random direction only | Binding on one direction; single precedence is free by reversal (:11-12) | Medium: no chains, so precedence sets are trivially acyclic. Logs: pure breaks only 35 % (n=200) and 52 % (n=1000) |
| T/rules.py:sample_rule:409-414 | Forbidden edge = an edge of the pure route away from the depot | Binding by construction (logs: 100 %) | Low in steps 1-2; step 3 does not do this (next rows) |
| T/rules.py:sample_rule:415-424 | Mandatory edge = one of 5 nearest neighbours, not adjacent in pure, each customer in at most 1 | Binding and cheap | Medium: "must be consecutive" never asks a long detour, so route cost effect is understated |
| T/rules.py:sample_rule:425-436 | Window width 10 % of pure length, t1 >= 0.1 L, closes before pure route arrives in either direction | A later window costs only free waiting (:20) | Medium: forces early service; extra_window (T/textv2.py:138) is the only non-binding case |
| T/rules.py:sample_rule:401,442 | At most 1000 rejection draws, else RuntimeError | (inferred) guard | Low |
| T/rules.py:cost_matrix:201, write_problem:246-248 | Soft-zone penalty is added to the one matrix LKH-3 reads, also in TSPTW, so it is treated as travel time too | Only rounding is discussed (:21-22) | Medium: see R-2; 33 of 400 instances, travel time inflated about 22 % in LKH |
| T/rules.py:FORBIDDEN:45, LKH_SCALE:44 | Forbidden edge = cost 1e6 scaled = 100 true units, above any tour (2.4e5 at n=1000) | Comment :45 | Low at n <= 1000; breaks near n = 10 000 if ever reused |
| T/rules.py:write_problem:255, cost_matrix:209 | Times ceil, window limits floor, so an LKH-feasible route is truly feasible | :21-22 | Low: tightens windows by up to n × 1e-4 |
| T/rules.py:solve:359-360 | Mandatory edge in SOP/TSPTW has a direction, so 2^m runs (logs: up to 4) | LKH-3 asymmetric form | Medium: cost of LKH grows; MAX_DIRECTIONS = 3 cap (T/open_dsl.py:41) |
| T/rules.py:solve:356; TS/make_text_rules.py:44-50 | "Feasible" = LKH found a route in MAX_TRIALS 1000, TIME_LIMIT 60/120 s, 5 redraws, then redraw | Step-1 design | Low so far: logs 200/200 feasible at attempt 1 at both n |
| T/rules.py:_run_lkh:294 | On timeout keep any tour already written; else no route | LKH loops in one trial (:295) | Low; step 1 passes no timeout (TS/make_text_rules.py:48) |
| T/rules.py:check_all:155-172 | Valid route = correct permutation from the depot + every hard rule; soft zones ignored; window checks only the upper limit (waiting makes the lower one automatic) | Definition | High: "valid" ignores soft rules and any non-binding rule; see pure baseline row |
| T/rules.py:check_rule:143-146 | Adjacency = position gap 1 or n-1 (the n-1 case only involves the depot) | Closed tour | Low |
| T/rules.py:check_rule:148 vs check_all:166-171 | check_rule times one window alone; check_all and violated_rules take all windows together | Comment :157 | Low: only pure_breaks uses the single form |
| T/formalize.py:rule_key:69-80 | Edges unordered, times rounded to 2 decimals, zone by name, weight ignored | "canonical form" (:7-8) | Low |
| T/formalize.py:score:165-192 | Rule correct = same key; instance exact = all correct and no leftover. Near matches are for the confusion table only | Report B | Medium: one extra spurious rule fails the instance, but not the rule-level rate |
| T/formalize.py:validate:106-139 | Per-item acceptance: bad items dropped, the rest kept; customers 1..n-1, distinct, t1 <= t2, finite. Lets through infeasible sets, cycles, wrong-but-valid customers | "items that do not fit are dropped and counted" | Medium: a partly wrong answer reaches LKH-3 with no signal |
| T/textv2.py:check_paraphrase:211-240 | Paraphrase kept only if: no template, ratio <= 0.8 to any template, 1 sentence, exact numbers, names, zones (plus check_meaning in v2b) | Keep v2 away from generator formulations | High: filter vocabulary = b0+ vocabulary, so step 2 favours the parser (report caveat :49-52) |
| T/textv2.py:check_meaning:286, ADJACENT:247 | Meaning preserved if keywords exist; in a merged sentence each rule is checked against the whole sentence | Comment :243-246: 15 % of mandatory paraphrases had lost adjacency | Medium: 17 % merged label errors survive |
| T/textv2.py:extra_window:138; TS/make_text_v2.py:121 | 20 % of no-precedence instances get a non-binding window; reference not re-solved | Measure verifier false positives (:13-14) | Low |
| TS/make_text_v2.py:generate:69, :149 | v2b: up to 10 calls per rule, T = 0.8, a rule with no sentence means the instance is left out (2 at n=1000) | Docstring :14-18 | Low |
| text_v2b_label_errors.json via T/open_eval.py:_l0:19 | Label check covers 40 random sentences, all 52 merged, all 149 precedences; only 9 instances removed | Report :59-63 | Medium: single-rule non-precedence sentences have only 0/31 evidence (upper bound about 11 %) |
| T/keyword_parser.py:parse_keywords:93 | Bullets, "Named places:" line, clause split at ", and"; b0+ uses filter vocabulary | "fair non-LLM baseline" (:3-4) | High: written after reading 200 text_v2b sentences (report :52); 99 % is not a held-out result |
| T/open_gen.py:LEGEND_SIZE:22, make_legend:49 | Legend holds 40/100 customers; groups mean legend places only, closed world | Full table would not fit 8192 tokens (:10-11) | High: "every hospital" silently excludes customers with no type; a real text would not say so. Legend coords print with 2 decimals (T/open_dsl.py:444), nearest uses true coords |
| T/open_gen.py:make_legend:59 | Area = cell of a 3×3 grid; names are street addresses, so type only from the legend | :11-12 | Low |
| T/open_gen.py:_window:242 | L1 windows: width 0.4 L, centre in 0.3-0.7 L; not built to bind | (inferred) so that a group of places is feasible | High: logs: only 14/23 and 12/26 window rules are broken by pure at n=200/1000 |
| T/open_gen.py:SHAPES:292, forb_* | Group forbidden edges are random, not edges of the pure route | (inferred) group rules | High: logs: pure breaks only 6/43 and 3/21 forbidden rules, so dropping them rarely invalidates a route |
| T/open_gen.py:sample_rules:299-333 | Rules kept if validate_for_solver accepts; primitives <= MAX_PAIRS 150; <= 2 mandatory; attempt 200 swaps the slot to forbidden_edge | Comment :308 | Medium: type mix shifts silently (not logged) |
| T/open_gen.py:docstring:7; TS/make_open_language.py:44 | L2 is unfiltered, labels from L1 rules | Hypothesis: test open language fairly | High: 28 % of L2 labels wrong; labels of L2 are not ground truth |
| T/open_gen.py:L2_PROMPT:362 | Prompt demands a double negation | Harder text | High: main cause of audited errors (report :471); the difficulty and the label error share one cause |
| T/open_dsl.py:expand_selector:90, expand_rule:155 | Selector expansion in code: union, except applied last, nearest over legend never itself, bare list = union, Cartesian for edges, self pairs dropped, empty = error | "what the text means" vs "what the solver gets" (:3-4) | Medium: mandatory_edge between groups becomes Cartesian and is usually rejected for degree > 2 |
| T/open_dsl.py:score:227-240 | Rule exact = all primitives in P and no wrong primitive of the same type touching its customers; instance exact iff P == L | :22-25 | Medium: scored before the code check (TS/phase5_open_language_report.py:score_all:61-62), so a rule later rejected still counts as exact |
| T/open_dsl.py:validate_for_solver:328-384 | Rejects: cycle, prec+window, two windows, t2 < direct distance, mandatory and forbidden, degree > 2, sub-tour, > 3 mandatory in SOP/TSPTW. Lets through: mutually impossible windows, forbidden edges that isolate a customer, mandatory vs precedence or window conflicts | :27-28 | Medium: first come wins (answer order), mandatory last; LKH then fails and the route is pure |
| T/open_eval.py:_l0:21 | L0 uses idx < 180; L1/L2 dev = idx 180-199 | Same selection as step 2 | Low |
| TS/make_open_language.py:main_l1:69 | Test idx 0..149, dev 199-i (20); reference solved at the same limits | :6-8 | Low |
| T/open_parser.py:docstring:3-4,8; p1_frozen.sha256 | p1 tuned on 20 dev L2 at n=200 only, about 1 h, frozen; hash verified identical today (49c29717…) | Report :21 | Medium: dev 43/54 = 80 % against test 54.7 %: 25 points of overfit |
| T/open_parser.py:FILLER:53, normalise:56 | Deleting "it is not acceptable to skip…" is safe | Tuned on dev (:17-19) | Medium: see R-7 |
| T/open_parser.py:parse_l0:310 | p1 on L0 = b0+ | Reuse | Medium: p1's L0 result is the b0+ result, never held out |
| T/open_parser.py:p0_text:321 | p0 gets p1's sentence splitter on L1/L2 | "best chance" (:322-323) | Low |
| l2_audit.json, T/open_eval.py:audit_excluded:62 | 60 of 120 used L2 texts read by one reader; 17 removed; rest stay wrong | Report :471 | High: no second reader; exclusion removes about half of the wrong texts |
| TS/open_pipeline.py:main:103-106, solve_job:50 | Rule set = label reuses reference; nothing left or no LKH route = pure route, judged against label | :3-7 | High: no-translation route is already valid 22-28 % of the time (logs; L0 27/17 %). Report step 3 has no such baseline row |
| TS/pipeline_eval.py:repair:38 vs T/open_dsl.py:validate_for_solver | Step 2: drop precedences if any window, no cycle check. Step 3: first come, cycle rejected | Different steps | Medium: valid-route rates and LKH-fail rates are not comparable across steps; also is_label seconds = step-1 seconds in step 2, 0 in step 3 |
| W/world.py:module docstring, shelf_kinds:70, make_orders:79 | Each block one kind; orders mostly normal, at most one special kind (40 % plain); 6 % urgent; all arrive at step 0; 30 customers from 46; one world per request | "a shelf never mixes kinds" | Medium: no order on both sides of no_mix between kinds, so that rule is always satisfiable by design |
| W/world.py:corridors:54; W/reqgen.py:shelf_corridor:47, cross_corridor:51 | Corridors 1-10 hold access cells, 11-20 none; crossings belong to no corridor; hard-coded 1-10 and 11-20 | "closing one never cuts the other way" | Medium: all-day closure of 1-10 is rejected by the validator, so generator uses 11-20 for those shapes |
| W/world.py:make:101, W/constraints.py:allowed:221 | Robots alternate A/B; B cannot carry heavy; enforced in every arm | Fact of the warehouse | Low: c_job checks true rules only, without world_checks (WS/phaseC_language.py:225) |
| W/dsl.py:expand_rule:193, _times:177, LONG:38 | 6 rule types: pick_order, close, no_mix, robot_only, deadline, station. t2 >= 5000 = all day; missing t1 = 0. `int(float())` truncates | :3-9; MAX_STEPS 6000 | Low |
| W/dsl.py:_picks:119-140 | Keys are ANDed; `{"all": false}` still means everything | Not stated | Low |
| W/dsl.py:select_orders:145 | An order "of a kind" = has at least one product of that kind | SPEC :432 | Medium: in pick_order the same word means shelves; text and label can part here |
| W/dsl.py:score:274, keys_of:245 | Rule exact on primitives, no wrong primitive of the same type touching it; close key includes exact t1, t2 | :24-26 | Medium: "until 300" vs "to 300" counts as wrong; scored before the validator (phaseC_report.py:66) |
| W/dsl.py:check:368-414 | Rejects: pick cycle, same order on both sides, all-day closure isolating a used shelf, heavy on B, two robots, two stations, deadline below lower bound or inside a closure. Passes: load infeasibility, conflicting deadlines, window closures isolating a shelf | :28-29 | Medium: order-dependent; deadlines last; `t1 > 0, no t2` counts as all day (test :118) |
| W/dsl.py:fastest:352 | Lower bound: farthest shelf + 3 × (shelves + 1); all-day distances | Docstring | Low: optimistic, so accepted deadlines may fail |
| W/reqgen.py:sample_rules:219, MAX_DEADLINE_ORDERS:17 | k = 2-4 rules of k different types; deadline at most 8 orders; 400 tries | :220 | Medium: 6 % of 150 = 9 urgent orders, so dead_urgent is thinned: logs 19 against dead_order 56, dead_except 10 |
| W/reqgen.py:docstring:7; WS/phaseC_language.py:gen_job:65 | Label kept only if the oracle episode breaks nothing and delivers 150/150, else redraw (5) | "every label can be met" | Medium: oracle 100 % by selection, by OUR planner not by proof. Logs: 229/230 first draw |
| W/reqgen.py:L2_PROMPT:267 | "State exactly one rule implicitly"; dev prompt was changed once (l2_dev_v1.jsonl differs) | Harder text | High: 6 of 21 audited errors add a rule, 5 of them "corridor N closed"; (inferred) a remark about a corridor meets the "describe the works" hint |
| W/reqparser.py:docstring:6, test_phaseC.py:291 | p1 from L1 templates, test asserts it reads every template | Fair baseline | High: L1 100 % by construction, as in TSP |
| W/reqparser.py:SYNONYMS:42, VAN:43, CLOSE:30 | Regexes mirror the L2 prompt hints (resurfac, van leaves) | Dev tuning | Medium: parser encodes the writer's prompt, not language |
| WS/phaseC_dev.py:main:22 | Only an assert stops reading L2 test | "No test text is read here" | Low |
| W/constraints.py:module docstring:3-10 | Rule to solver: pick_order = SOP or exact; all-day close = cells removed; window = GUARD 40 and forbidden legs; no_mix = merge refused; robot_only = mask; deadline = own trip + weight + TSPTW; station = fixed end | Design | Medium: see fallbacks below |
| W/constraints.py:ordered:165-184, route:281-282 | Silent fallbacks: window dropped if it cannot be met or if time left <= 0; SOP failure = greedy; TSPTW failure = plain LKH; timeout 20 s | "A window that cannot be met is dropped" (:166) | Medium: not logged; shows up only as `broken` |
| W/constraints.py:GUARD:31, FORBIDDEN:32, WEIGHT:33 | 40 steps, 1000, 1e6 hard-coded | "a stop lasts up to 20" | Low |
| W/verify.py:check_rule:46-73 | to = exclusive; deadline <= t inclusive; pick_before end <= start; robot_only passes an order never assigned; station and deadline fail if undelivered | Docstring | Low |
| W/constraints.py:module docstring:12 | Verifier "shares nothing with this file", yet to_checks is in this file and labels use the same expand_rule | Independence claim | Medium: a selector bug reaches label, solver and verifier alike |

## 6. LLM

Same path conventions as section 5.

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| T/llm.py:HOST:11; TS/ollama_pc.sh | Private Ollama, port 11435, own binary, models, HOME; MAX_LOADED 1, NUM_PARALLEL 1, flash attention, KV q8_0, default context 4096 | Never touch shared 11434 | Low; shared Ollama on 11434 still shares the GPU |
| T/llm.py:MODEL:12; TS/open_formalize.py:MODELS:29; WS/phaseC_language.py:MODELS:36 | qwen2.5:7b (6000 MiB), qwen3:14b (11000), gemma3:12b (10000); step 0 planner also qwen2.5:3b; vision gemma4:e4b-it-qat (fallback e2b unused). Tags only; no digest logged per call | (inferred) what fits a T4 | Medium: tag moves are not recorded; q14 exceeds the CLAUDE.md "under 6 GB" rule, which is written for training |
| T/llm.py:chat:16 defaults | temperature 0, seed 0, num_ctx 8192, num_predict 512, format json, keep_alive 10m, timeout 600 s | Reproducible and fits GPU | Medium: 512 truncates (logs: 3 answers per n at step 2) |
| TS/formalize_eval.py:main:70 | Step 2: b1/b2 no retry; invalid JSON = failed answer | Docstring | Low: 0-2 % |
| TS/open_formalize.py:llm_arm:45-54 | Step 3: 1 retry, seed 1, only on invalid JSON or abort; time counted; schema-invalid items never retried | Design | Low: logs: q14 0 retries; overall 17 retries, 7 rescued, 4 byte-identical (T = 0 makes the seed nearly irrelevant) |
| WS/phaseC_language.py:llm_arm:131 | Same retry, num_predict 1024, q14 think False, q7 default; q14 only 200, q7 first 100; no q14t, no g12 reader | Carried over from TSP | Medium: think-off decision comes from TSP data |
| T/llm.py:chat:34-41 | HTTP 500 and timeout = failed answer (content None), other HTTP errors crash | Report :66-70 | Low: abort not deterministic (report :69-70) |
| T/llm.py:extract_json:125 | First-to-last brace, greedy; None if broken | Models wrap JSON | Low |
| T/formalize.py:SCHEMA_TEXT:88, SYSTEM_B1:204, FEW_SHOT:216 | Prompts in code. b1 catalog + schema; b2 + 5 hand-written n=60 examples | :214 | Low: stored in git, not per record (L2 prompts are) |
| T/open_dsl.py:SPEC:389, EXAMPLES:424 | All arms same prompt, 3 examples all L0 style, none with a selector | "same as the first 3 of step 2" | High: LLM must learn groups, except, nearest from the spec alone, while p1 was tuned on dev; favours p1 on L1/L2 (wrong selector is the top q14 error, 62 + 42) |
| W/dsl.py:SPEC:419, EXAMPLES:458 | Similar, but 3 L1-style examples including except; "stated or clearly implied"; explicit "Last" recipe | Hand-written, customers not in worlds | Medium: warehouse LLM arms are better served than TSP's; not comparable |
| TS/open_step0.py:requests:43, run:59 | Think mode chosen on 20 text_v2b at n=1000 with a foreign legend; think on 40/46 against off 32/46, 7.5× time | Time budget (report :43) | High: q14 off is a budget choice; on L0 thinking won by 8 rules. Those instances overlap L0 test (no idx < 180 or label-error filter) |
| TS/open_formalize.py:MODELS:29, :48 | q14t: think True, 2600 tokens, L2 only, 12 per n | Report :43 | Medium: 8-25 % no JSON; 12 is an indication only |
| T/vision_probe.py and TS/vision_probe.py:chat:92 | No think field, so default on; no system prompt; num_predict unset; T = 0.5; seed = repetition; keep_alive 30m; timeout 900 s (no TimeoutError catch, unlike llm.chat) | Report :31-32 | Low |
| TS/open_latency.py:main:49-71 | Latency alone = 30 calls per arm, 5 per (n, level), no LKH of the user's uid running, warm model, each arm starts <= 70 C | Memory llm-latency-alone | Medium: other users unseen (load 2-7); arm order q14, q7, g12 means q14 cool, others hot (report :475); q14 is 5 samples per cell, used as cell means in table C (TS/phase5_open_language_report.py:llm_time:194) |
| T/llm.py:lkh_running:45 | `pgrep -x LKH` for the user's uid counts as load; sampled after the call in open_formalize (:59) | "registar em cada chamada" | Low: a load that ended before the call ended is missed. Python planners are not counted |
| TS/llm_verify_eval.py:75, TS/formalize_eval.py:70 | Step 2 latencies have no load record at all; D ran in parallel with C (TS/run_phase5_text_llm.sh) | Rule came later (2026-10-31) | Medium: D 9.8/12.4 s are under load, unmarked; see R-8 |
| TS/open_formalize.py:llm_arm vs TS/open_latency.py:59 | Main run has no warm-up call: the first call after unload includes load time | "not counted" only in the latency script | Low |
| T/llm.py:wait_cool:73, gpu_state:65; TS/gpu_thermal_log.sh | T4 throttles to 300 MHz: 70 C cool threshold, 30 s sampling, stop by thermal.stop file | Report :45 | High for time claims: same q14 takes 10 s or 25-30 s; "10 s" is best case |
| T/llm.py:GpuGuard:79 | check before calls: if free - need < 2048 MiB wait; never stop anything; nvidia-smi failure = continue | CLAUDE.md | Low: 0 pauses, min free 5340 |
| TS/open_step3.py:planner:60-61 | V0 kept at ctx 4096, unlimited tokens; V3 512, 8192; 7B column from earlier run | Comment :59 | Medium: latency not comparable (report :368) |
| TS/planner_probe.py:CASES:108, chat:166; TS/planner_probe_v2.py:VARIANTS:164 | 8 hand-written cases × 5 repeats at T = 0, seed 0 = "40" trials; no execution of the chosen tool | Calibration | High: logs: 34 of 48 groups byte-identical, the rest 2 versions; effective n = 8. See R-5 |
| TS/planner_probe.py:EFFECTS:48 | Expected effects are hand-copied from E1/E3 and triage reports, gaps on Fu sets | :44-47 | Medium: stale if those change |
| TS/planner_probe_v2.py:rule:86, feasible:65 | R = argmin of the same table used to build the checks: 40/40 in 9 µs | Baseline | High: R's score is close to tautological |
| TS/planner_probe_v2.py:report_v3:139 | V3 lists only actions that fit, unsorted; about 80 % of the rule | :7-8 | Medium: 35/40 is not a planner, it is an arg-min (misses all of case 2) |
| TS/planner_probe.py:lkh:90-93 | time_limit must lie in 0.7-1.0 × time left | Hard-coded in CASES bounds | Low |
| T/llm_verify.py:inject_error:34 | Exactly 1 violation injected, checked in code; 4 types; precedence by swap, forbidden by moving j next to i, window by moving later | :7-12 | Medium: at most 200 tries; no route if none. Base rate 183/632 units |
| T/llm_verify.py:route_text:66, user_prompt:115 | LLM sees text_v2b + list with times + (variant) code report; n=200 only because 10k tokens at n=1000 | Report :64 | Medium: hides any size effect |
| T/llm_verify.py:match_claims:127, parse:153 | Match by type + customers, else customers only; first hit; strict json.loads (not extract_json) | :15-16 | Low |
| TS/phase5_text_llm_report.py:part_d:271 | JSON failure = false alarm on oracle, but "no claim" for P/R (:280) | Stated for P/R | Low; see R-6 |
| T/open_hybrid.py:coverage:43 | h1 trigger uses text and p1 output only; whole text goes to q14 | :3-9 | High: 43 of 95 p1 wrong L2 texts kept; right count with wrong content is invisible to it |
| T/open_hybrid.py:union:61; TS/open_hybrid.py:records:60-82 | h2 = p1 then q14 through code check; first rule wins; offline, 68 sets unsolved | Report :374 | Medium: bounds wide; no LKH |
| W/reqparser.py:send_to_llm:368 | Same design for phase C; h1 from logged answers | Docstring | same as above |
| TS/make_open_language.py:main_l2:104; WS/phaseC_language.py:main_l2:114 | L2 writer: gemma3:12b, T = 0.7, seed = idx, num_predict 700, system "You rewrite text faithfully." | Report :18 | Low: logs: no paragraph reached 700 tokens (warehouse max 244) |
| TS/make_text_v2.py:generate:74 | Paraphrases by the same 7B, T = 0.8, num_predict 256, seed (n·1000+i)·100+10k+call | Docstring | Medium: same model on both sides (report :47) |
| T/vision_probe.py:parse_ranking:105, ranking_scores:123 | Distinct cells 1..16 kept, rest tied at random; AP pooled over cells | :124-125 | Low |
| T/vision_probe.py:GRID:17, TS/vision_probe.py:step_images:140 | 4×4 grid, window belongs to the centroid cell, labels from first-pass log (nothing replaced, verified in TS/triage_log.py:4), 6.2 windows per cell | Report :43 | Medium: coarse labels; oracle AP only 0.83 |
| TS/run_phase5_vision.sh:48-53 | GPU others < 6 GiB, else after 6 h e2b; budget 5376 MiB | :4-5 | Low: another Ollama arrived at 16:19, latency mixed (report :36-42) |
| TS/phase5_vision_probe_report.py:table_policy:113 | Vision time = mean latency + drawing, added afterwards | Report :94 | Low |
| TS/open_hybrid.py:table:89, phase5_open_language_report.py:pc:47, mcnemar:145 | Wilson and McNemar treat rules as independent, though 2-3 rules share one text; no multiple-comparison correction (report notes the second only) | Not stated | High: p-values are optimistic, see list (b) |

## 7. Warehouse

Paths are relative to `warehouse/`; a bare `env.py`, `sim.py` or `mapf.py` is in `warehouse/src/warehouse/`, and a bare `phase*.py` is in `warehouse/scripts/`. "(computed)" means derived from existing JSON in `results/warehouse/` or from the `env.py` layouts.

### 7.1 Grid and layout

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `src/warehouse/env.py:70-75` `Warehouse.__init__` | 4-connected grid, no diagonals | Docstring: "Kiva style" | Low: same topology as most MAPF benchmarks |
| `env.py:36-40` `SIZES` | Three layouts only: 20x30 / 40x60 / 60x100, 160 / 810 / 1820 shelves | (inferred) spread of scale | Medium: "fixed layout parameters" is an admitted limitation (phaseA_report:252) |
| `env.py:28,37-39` `Layout.corridor` | Aisle width 1 in every experiment; width 2 exists only in tests (`test_env.py:18`, `test_mapf.py:156`) | Not documented beyond "one corridor width" | High: a robot picking stands IN the aisle and blocks it for 3 steps (`env.py:80`), so width 1 sets most of the congestion; head-on traffic needs a cross-aisle |
| `env.py:31-33` | Blocks 2x5, cross-aisle every 6 columns, 2 parking rows | Not documented | Medium: longer blocks give longer blocked stretches |
| `env.py:37-39` `robots` | Baseline density 10/440, 30/1590, 80/4180 = 2.3 / 1.9 / 1.9 robots per 100 free cells (computed); B0b T3 top is 150/1590 = 9.4 | Not documented | High: "congestion is modest", "CBS gains nothing" and "traffic ceiling 4–6 %" are statements about this density |
| `env.py:64-66` | Stations only on the bottom row, evenly spaced, 3 / 6 / 10; one robot at a time per station, no queue area, no lanes | (inferred) simplicity | Medium: station queues spill along the row; delivering robot blocks a through row |
| `env.py:49,152-154` | Parking on the top rows, the far end from stations: mean 28 / 61 / 100 steps to a station (computed) | Docstring only | Medium: first trip and every return to rest are long deadheads, inflating free flow |
| `env.py:76-81` | Each shelf has exactly one access cell, above or below; facing shelves can share one (450 access cells for 810 shelves, medium, computed) | Report A "may share" | Low to medium: choice of side is not a decision, so routing has less freedom |
| `env.py:18,67` | Shelves are permanent obstacles, robots stop beside them; shelves never move | Despite the "Kiva style" label | High: this is robot-at-shelf picking, not goods-to-person; see W-9 |
| `phaseB0b_margins.py:159` `t3_world` | At 4x, 120 robots on 120 parking cells (100 % full); at 5x a third row shifts the blocks one row | Comment "2 rows park 120 robots"; report notes the shift only | Medium: full parking may feed the PIBT fallback jump from 1 % to 35 % of calls at 4x, seed 0 (computed; cause inferred) |

### 7.2 Time and motion

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `env.py:10` | Discrete time, 1 cell per step, wait is the only other action; no acceleration | Report A limitation | Medium: stop-and-go costs nothing in nominal model |
| `env.py:10`, `verify.py:76` `steps_per_hour=3600` | 1 step = 1 s: every orders/h is that assumption; cell size is never stated | "One step counts as one second in the rates" | Medium: absolute rates only; ratios survive |
| `sim.py:21` `SERVICE=3` (`dsl.py:39` repeats it) | 3 steps at a shelf and at a station, constant | No source; B0 limitation says "no real service durations" | High: relative weight of walking against service decides batching gains and aisle blocking; 3 s is short for a real pick (inferred) |
| `env.py:132`, `sim.py:72` | Service is per shelf, not per product: 2 products on one shelf cost one service | Not documented | Medium: favours orders and trips with shared shelves |
| `sim.py:73` | Each order delivered in a trip costs its own 3-step service, also at the same station | Implied by `TripCost` | Low |
| `sim.py:136` | A shelf shared by two orders of a trip costs one service for both | Comment | Medium: extra batching gain from free picks |
| `sim.py:53-62` | Entering a next-stop cell always starts service at once; no approach or alignment | Docstring of mapf | Low |

### 7.3 Execution model (B0c, `exec.py`)

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `exec.py:3,214-219` `Execution.step` | A turn costs 1 step whatever the angle, 180° included | "as asked" (B0c report) | Medium: 2 steps for a U-turn raises rotation cost |
| `exec.py:1` `mapf.py` planners | Planners plan on the nominal model: no heading in PIBT moves or A*; rotation reaches layer 1 only | Report: a heading field sends PIBT on detours | High: the +22 % rotation cost and "knowing it is worth nothing" hold for rotation-blind planners only |
| `exec.py:146` | All robots start with heading 0 (up) | Not documented | Low |
| `exec.py:206-207` | No turning, stop or slip during a service; a robot cannot pre-rotate while picking | Report "never during a service" | Low to medium: pre-rotation would save one step per stop |
| `phaseB0c_realism.py:47` `R1` | p = 0.10, q = 0.002, chosen for this phase | "the plan gave none" | High: only one intensity run; report concedes stronger delays could change it |
| `exec.py:29,208-213` `BRAKE` | Stop length uniform 5–20, mean 12.5 | Not documented | Medium: fat-tailed failures would hurt plans more |
| `exec.py:208,224` vs `:229-236` | p and q are drawn before the dependency check, so a blocked robot also draws them; a slip while blocked costs nothing and a stop can start in a queue | Not documented | Medium: R1 is not really flat, slips per move rise 0.115 to 0.123 with density |
| `exec.py:9-13,29-36` `StateDelay` | p = min(0.9, s·speed·(1+0.25n)·(1+near)); types 0.5/1/1.5 by index mod 3; radius 2; reach 6; bad floor 20 %, patch radius 3 | Report: values chosen here | High: the function is invented; "R2 opens no margin" is conditional on this form |
| `phaseB0c_realism.py:236-268` | s, q_bad calibrated to match R1 slips and stops per move, on the PIBT arm with plain trips, seeds 100–105 | Report | Low to medium: prioritized at 90 robots lands 8 % lower |
| `exec.py:83-91,100` | Bad floor fixed per layout (`floor_seed=0`), so static maps can learn it | Report limitation | Medium: a changing floor would favour a live predictor |
| `exec.py:163-164` | A stalled robot looks like a service to the planner: known instantly and with exact remaining length | "seen, not predicted" | Medium: real faults reveal neither; slips are invisible by contrast |
| `exec.py:166-183` | Hönig-style dependencies over one window only; plan dropped at nominal cadence, a late robot loses the rest | Docstring | Medium: under 5-step windows late robots lose plan content every replan |
| `exec.py:145,190` | Delay draws indexed by (tick, robot), so pairing holds only until trajectories diverge | Report "as far as their trajectories allow" | Medium: CI half-widths of 3–7 % come from this |
| `phaseB0c_realism.py:49` | 5 orders per robot, 3 cells (m30, m90, l60); prioritized never on large | Budget cut | Medium |

### 7.4 Collisions

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `verify.py:20-30`, `mapf.py:283-304` | Vertex and edge (swap) conflicts forbidden, nothing else | Report A | Low |
| `test_verify.py:45`, `mapf.py:174,275` | Following at full speed is allowed: a robot may enter a cell vacated in the same step; so may a full cycle | Test asserts it | High for realism: a safety gap or follow-delay would cut aisle throughput; `exec.py` keeps this ("or are leaving in this same step") |
| `mapf.py:275` | Swap only blocked along the same edge; no footprint, no corner cutting | Not documented | Low |
| `exec.py:181` | Same-step entry behind a leaver is safe even under slips, since the leaver is confirmed in `want` first | Docstring | Low; 0 collisions in all reported runs |

### 7.5 Assignment

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `dispatch.py:35-42` `Nearest` | FIFO queue; oldest order to the free robot nearest any of its shelves, ties by lowest id | "simple on purpose" | Medium: base case O1 beats by 5 % / 12 % |
| `sim.py:84` | Free = no undelivered order and not in service; a robot walking home is free | Not documented | Low |
| `sim.py:110` | Orders are given out only between plans: every step for PIBT, every 3 for CBS, every 5 for prioritized, every 1 / 2 for o4w5 / o4w10 | Comment: orders "would change" plans | Medium: planner comparisons also compare assignment latency, up to 4 steps per trip against prioritized |
| `sim.py:86-92`, `dispatch.py:8` | Reassignment fires on any arrival or delivery, for robots that have not started picking; committed after first pick | Report O1(b) | Low to medium |
| `sim.py:76` | `give` resets `age` to 0, so each reassignment also resets priority | Not documented | Low |
| `sequencing.py:185-190` | Rule ignores congestion and other orders | Docstring "to be improved" | Medium |
| `env.py:131` | The station is fixed by the order, uniform over stations, never chosen by the system | Not documented | High: access-to-random-station 37 steps against 23 to nearest, medium (computed); choosing stations is a free decision left out |

### 7.6 Capacity and Clarke & Wright

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `sim.py:71`, `verify.py:141` | Capacity counts orders, not products, weight or volume: a trip of 3 may hold 36 products | Not documented | High: the +33–37 % batching headline has no physical limit behind it |
| `phaseB0_margins.py:136-139`, `phaseB0b_margins.py:36` | Capacity values 1, 2, 3, 4 (4 medium only) | Plan | Medium |
| `sequencing.py:152-176` `route` | All shelves first, then stations in the cheaper order; no mixed pick-and-drop | Docstring | Medium: a mid-trip delivery can be shorter |
| `batching.py:51-55` `TripCost.way` | Trip cost is a closed walk from the first station back to it + walk between stations + services; the robot's real start is ignored | Docstring | Medium: first trips start at parking; ending at a different station is not credited |
| `batching.py:47` | Cost = walk + `service × (shelves + orders)`; all in steps | Docstring | Medium: "walk" in T1 is not pure walking; see W-3 |
| `batching.py:54` | Cost priced by 2-opt, and on the empty grid; trip later sequenced by LKH-3 from the actual start | "cheap enough for every pair" | Low: 2-opt gap 1.6 % on orders |
| `batching.py:58-81` `savings` | Merge whenever saving > 0, greedy by largest saving; no time, due date, load or queue-length term; no robot balance | C&W standard | High: online, merging at ρ ≈ 0.68 loses to one order per trip (B0b T2) |
| `dispatch.py:231-241` `Clusters` | Trips leave in order of lowest oid, to nearest free robot; so no SPT in batching arms | Report limitation | Medium: baseline of the recommendation could improve cheaply |
| `dispatch.py:249-251` `LiveBatches` | Online: window of 200 oldest orders; rebuilt at each free robot; `wait` holds by oldest order's age | Docstring | Medium: `window=200` touches large at x2.0 |
| `dispatch.py:251,284` | Clairvoyant arm sees orders in full, cannot pre-position or add to a trip under way | Report limitation | Medium: understates perfect knowledge |

### 7.7 Hungarian

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `dispatch.py:61-67` `costs` | Cost = distance robot → first cell of the order's open path + LKH-3 open-path length; no services, no congestion, no order age | Docstring | Medium: objective is total walking, so it implies SPT |
| `sequencing.py:135-149` `open_path` | The first shelf comes from a free-start tour independent of the robot, so robot-specific entry is not optimised | Docstring | Low to medium |
| `dispatch.py:72` | Over the whole queue by default; so long orders can starve | Report: gain "comes almost entirely from short orders first" | High online: never run in lifelong mode; mean delay hides tails, T2 uses Nearest only |
| `dispatch.py:108-118` `TripHungarian` | Over fixed trips, reassign on, cost = distance + trip open path; used for all of B0c | Docstring | Medium: B0c baseline stack differs from the B0 / B0b baseline (Clusters + Nearest) |
| `dispatch.py:169,186` `LiveCosts` | Rebuild every 50 steps, decay 0.9, radius 2, gamma capped at 10 | Not documented | Low |

### 7.8 Prioritized planning

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `mapf.py:208` | horizon 20, execute 5, retries 3 (4 tries); robots beyond the window ignored | Silver 2005 | Medium: o4w10 ties, o4w5 −2 %, but see W-4 |
| `mapf.py:257` | Priority = steps since last goal, ties by id; a pick resets it; idle or homing robots stay at age 0 | Docstring | Medium: never order age or lateness; age is per stop |
| `mapf.py:23-70,239` `search` | Space-time A*, unit cost, heuristic = empty-grid BFS; goal must stay free for the dwell (`lock_goal`) | Docstring | Low |
| `mapf.py:50-57` | Ends at horizon wherever f is lowest; no terminal safety, so a window can end in a trap | Windowed design | Medium: the fallback takes it |
| `mapf.py:228-249` | Plans through several stops inside the window, including dwells | Docstring | Low |
| `mapf.py:258-280` | Failure: failed robots go first, up to 3 retries, then one step of whole-fleet PIBT with no metric | Docstring | Medium: at 5x 65 % of calls, 27 % of steps (seed 0, computed) |
| `phaseB0c_realism.py:161` | Prioritized reads one table for all robot types | Report limitation | Low |

### 7.9 PIBT

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `mapf.py:156-158` | Priority = age + rule bonus + eps, eps redrawn each step | Not documented | Low: Okumura fixes eps per agent, so this is a variant |
| `sim.py:124` | Age grows only with a queue, no hold and an open order | Comment | Low |
| `mapf.py:171-172` | Candidates sorted by distance, then free before occupied, then fewest turns, then random | Report: straight-tie rule added | Low to medium: tie rule cut turns by about a third on seed 0 |
| `mapf.py:162-165` | Robots in service are immovable | Report A | Medium: a service in an aisle blocks PIBT's push |
| `mapf.py:138-151` | "far" / "near" weight < 0.5 and eps < 1, so they barely act; `load` 5 per stop; `fixed` uniform 0–50 | Docstring | Low: best of 5 is mostly seed selection |
| `mapf.py:166` | One goal at a time; no lookahead | Okumura 2022 | Medium: 6–14 % behind prioritized in B0c |

### 7.10 CBS

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `mapf.py:367` | horizon 8, execute 3, cooldown 10 | Not documented | High: compared with prioritized at 20 / 5, so not same-window |
| `phaseA_baselines.py:33`, `phaseB0_margins.py:52` | 30 s per call, 900 s per episode (A); 90 s, 2700 s (B0 "3x"); large not run | Budget | High: A medium is 9–62 % PIBT by episode (computed, matches report) |
| `mapf.py:380-381` | Only the next goal per robot; then it stays until the horizon | Docstring | Medium: ignores follow-up stops that prioritized sees |
| `mapf.py:376-378,392-396` | Fallback PIBT for 10 steps; budget spent means PIBT for the rest of the episode; no-solution and time-out are the same branch | Docstring | Medium: see W-10 |
| `mapf.py:334,357-359` | Plain CBS, tie on conflict flag, pure Python | Not documented | High: "exact" is limited by speed; ECBS or CBSH could change the medium result |
| `mapf.py:336,383` | Wall-clock deadlines, so results depend on machine load | Report "Timing" limitation | Medium: CBS rows are not reproducible, unlike the rest |

### 7.11 Free flow and bounds

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `mapf.py:402-417`, `phaseB0_margins.py:321-328` | Free flow = same trips per robot, each alone, cells shared; one step per plan so assignment every step | Merschformann 2019 | Medium: a limit of a plan, not a bound; "% baseline free flow" goes below 100 for batching |
| `phaseB0_margins.py:307-318` | Visit order is re-solved from the free-flow position; o2b3 in plain metric | Report notes o2b3 only | Low |
| `phaseB0c_realism.py:169-176` | B0c free flow has rotation, no delays, so "% free flow" mixes delay and traffic | Report states it | Medium: 123–177 % not comparable with B0's 105 % |
| `phaseB0_margins.py:398-409` `bound` | max(total work / R, longest order); LKH-3 lengths "tight rather than proved"; one order per trip | Docstring | Low: batching beats it (676 < 734), by design |

### 7.12 Lifelong orders and load

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `env.py:141-149` `poisson_orders` | Homogeneous Poisson, no peaks or batches of arrivals; arrival truncated to integer step; stops at the horizon | Report A | High: real warehouses have waves; wave picking changes T2 |
| `phaseA_baselines.py:34-35` | Rates: 0.5, 0.8, 1.2 and sweep 0.9–2.0 × "capacity" | Plan | Medium |
| `phaseA_baselines.py:143` `e2_jobs` | "Capacity" = E1 LKH + PIBT delivered/makespan: 0.084 / 0.142 / 0.260 per step, below sustained 0.097 / 0.168 / 0.314 | Report limitation | High: "medium" is ρ ≈ 0.66–0.69 of prioritized one-order capacity (computed), and x2.0 is not 2× anything sustained |
| `phaseB0b_margins.py:113-115` `capacity_rate` | T2 reuses the PIBT one-order rate for prioritized and for batched arms | "as in E2" | High: see W-1 |
| `phaseA_report.py:152-154` `keeps_up` | Keeps up = backlog slope in second half < 5 % of the rate; "first rate that fails with every higher one" | Fixed before run, reading added later (report) | Medium: noisy at low rates |

### 7.13 Metrics and "completed"

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `verify.py:131,164` | Makespan = latest delivery end, return to parking excluded | Not documented | Medium: dominated by the last trip, noisy (B0c limitation) |
| `verify.py:131,159` | Completed = delivery service finished at the right station by the assigned robot | Docstring | Low |
| `verify.py:159` | Delay = completion − arrival, delivered orders only; E2's column is survivor-biased, T2 `delay_all` censors open orders at the horizon | B0b report | Medium: not the same metric in A and B0b |
| `verify.py:166`, `phaseA_baselines.py:36` | Orders/h = delivered / T × 3600 over the whole 2000 steps; no warm-up | Not documented | Medium: includes the cold start; the second-half figures (`phaseA_report.py:157`, `phaseB0b_margins.py:135`) are the informal fix |
| `phaseA_baselines.py:36`, `sim.py:107` | Horizon 2000 steps, no drain | Not documented | Medium: only 2–5 cycle times in the second half |
| `phaseA_report.py:140-144`, `phaseB0b_margins.py:133` | Backlog = arrived and undelivered, including orders in progress (up to 3 per robot in T2) | Report notes "one per robot" for E2 only | High: T2 backlog 100 / 222 is about WIP; see W-1 |
| Not present (`grep -i tardi` over src, scripts, docs: no hit) | No tardiness metric anywhere | - | High for a logistics reader |
| `phaseB0_report.py:62-67` | 5 seeds, t interval, Wilcoxon floor p = 0.0625 | Report states it | High: below about 1 % is noise nominal, 3–7 % in B0c |

### 7.14 Pick locations, order sizes, due dates

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `env.py:125` `Catalogue` | Each product on one shelf, uniform; no ABC skew, no slotting, no zones | Not documented | High: skew concentrates traffic and raises shared-shelf batching, against "conflicts only moderately concentrated" |
| `env.py:123`, `phaseA_baselines.py:47` | 2 products per shelf on average; infinite stock; no replenishment | Report A | Medium |
| `env.py:130` | Products in an order uniform, independent; no co-purchase | Not documented | High: co-purchase is what a learned batcher would exploit; absent by construction, so the "no margin" on grouping is partly built in |
| `env.py:129` | Order size uniform 3–12, mean 7.5; no single-line orders | Not documented | High: e-commerce is mostly 1–2 lines (inferred); batching and sequencing margins scale with stops |
| `env.py:109-117` | No due dates in A–B0c; phase C `deadline` is a rule of a request, checked by `verify.py:58` | - | Medium |
| `world.py:79-92` | Phase C orders differ: normal products drawn from normal blocks only, plus at most 1 special kind | Docstring | Medium: B0c margins do not transfer exactly to phase C geography |

### 7.15 Depot, parking, idle robots, deadlock

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `sim.py:36,73` | No depot: the "depot" of `cvrp_lkh` is abstract; each robot has its own parking cell, its start | env docstring | Low |
| `sim.py:73,100,142-143` | Idle robots walk all the way back to their own cell, dwell 0; no staging near stations | Not documented | High online: at ρ < 1 this is the empty-travel cost; loose rate delay of 165 on medium |
| `sim.py:142-143` | A pushed idle robot is sent home at once | Comment | Low |
| `mapf.py:254-280` | Idle robots are planned last, can dodge; they sit in parking rows, away from aisles, so rarely block | Not documented | Low at 1x; medium at 4x (full rows) |
| Whole of `sim.py` | No batteries, charging, faults outside B0c | Report A limitation | Medium |
| `sim.py:24,104`, `episode.py:18` | No deadlock detector. Only priority by age, PIBT fallback, and `max_steps` 30000 (6000 in phase C) that ends silently | Not documented | Medium: caught by asserts in B0 to B0c, by no assert in `phaseA_baselines.run_e1` or `episode.run` |

### 7.16 Verifier

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `verify.py:20-37` | Checks free cells, step size, vertex and edge conflicts | Docstring | Low |
| `verify.py:96-131` | Checks dwell = service exact, pick adjacent to its shelf, after arrival and assignment, before delivery, same robot, right station, all shelves visited | Docstring | Low |
| `verify.py:87-91,96-99` | Trusts `events` for what happened at what time, then confirms against traj | Docstring "shares nothing" | Low: independent code, not independent data |
| `verify.py:132-142` | Capacity by overlap of [assign, end] per robot | Docstring still says "two orders" | Low |
| `verify.py:40-43` | Rotation check: only that two moves in different directions are not adjacent in time; any wait passes as the turn | Docstring | Low: `exec.py` is stricter than its check |
| `verify.py:163` | `delivered < orders` does not fail `ok`; callers must assert | Not documented | Medium: done in B0, B0c; not in `t1_sim` free run or `t3_free` |
| Not checked | Start cells in parking, rest behaviour, computing time, optimality, that the order of `arrival` was respected by FIFO, capacity in products | - | Low |

### 7.17 The 3 % margin and hindsight

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `docs/phaseB0b_notes.md:3,13`, `phaseB0b_report.py:135`, `phaseB0c_report.py:160` | 3 % threshold is "the rule of the plan"; only a dashed line in figures, no decision code; absent from B0's report, which cites +4–5 % (Yu & Wolf) | No source given | High: point estimate against a threshold with half-widths of 1–7 %; not a test |
| `phaseB0_report.py:40,107` | B0 rival: best of o1a, o1b, o1it, o1dw, o1dg, o2b1, o2b2, post hoc per metric and size | Report | Medium: rival set differs by phase |
| `phaseB0b_report.py:27,114` | T3 rival: best of base, o1b, o2b1, o2b2 | Report | Medium |
| `phaseB0c_report.py:31,89` | B0c rival: best of b, c, d; c and d are maps learned from other seeds, so "rules" include a statistical model | Report | Medium: margin is of an oracle above a table |
| `batching.py:47`, `phaseB0b_report.py:55` | T1: compare best partition with C&W, on cost with services, and on makespan | Report | High: W-3 |
| `phaseB0b_report.py:87-97` | T2: best (c), per metric and best λ, against best of (a, b) | Report | Medium: favours oracle by selection, but capped; see W-1 |
| `phaseB0_margins.py:269-282`, `traffic.py:126-128` | Hindsight = whole-episode mean extra per edge, same seed, symmetrised, 3 rounds, best by makespan, starting from o1b | Report | High: time-blind, no per-time or per-robot information, not a ceiling for a live predictor |
| `phaseB0_margins.py:277`, `phaseB0b_margins.py:205` | In B0 and B0b hindsight reaches LKH-3 and the Hungarian only; the planner stays plain | Report: "in layer 1" | High: says nothing about path planning; B0c does pass it to the planner (`phaseB0c_realism.py:140,162`) |
| `phaseB0c_realism.py:201-207` | B0c hindsight starts from (b) of the same seed; report also gives the mean of 3 | Report | Low: good practice |
| `phaseB0_margins.py:112`, `traffic.py:157` | Static map: an edge unseen on other seeds costs 0; map comes from the Nearest baseline, applied to Hungarian | Not documented | Low |
| `sequencing.py:80` | LKH-3 RUNS 1, MAX_TRIALS 50, SEED 1, treated as optimal | Test against Held-Karp | Low |

## 8. Statistics

Paths are relative to the repository root, or to `tsp/scripts/`, `tsp/docs/`, `warehouse/scripts/` and `warehouse/docs/` when given by bare file name.

### 8.0 Seeds and sample units per experiment

| Phase / report | Stochastic unit | Count | Test instances / sample | Where |
|---|---|---|---|---|
| Phase 1 (`results/table.md`) | training seed | 3 (0,1,2) × 3 models | TSP-20 1000, TSP-50 500 | `tsp/results/runs/{gcn,gat,gated}_seed*` |
| Phase 2 exp1, batch2 | training seed | 3 per configuration | + TSP-100 500 | `tsp/results/runs/*_knnmean`, `*_layernorm`, `*_train50*` |
| Phase 2 checks | training seed | 3 × 3 models (9 LN reruns) | same | `*_layernorm_ckpt` |
| Phase 2 datasize | training seed | 3 at 100k; **1 at 25k and 50k** | same | `gat_data25k` etc. (1 dir each) |
| Phase 2 beam, TAGCN, k02n | training seed | 3 | same | `tagcn_k{2,3}` + `_ckpt` reruns, `_layernorm` |
| Phase 2 scale | training seed | 3 | 1000/500/500/100/50/20/10/5/3 for n = 20…10000 | `tsp/data/*_test.npz` shapes |
| Phase 3 hier, window, triage, wsize | training seed | 3 TAGCN K=3 | 100/50/20/10/5/3 | `tsp/data` |
| Phase 3 benchmark | 3 TAGCN + 5 rng (random) + 1 LKH-3 run per config | | Fu-1k 128, Fu-10k 16 | `run_phase3_bench_e2.sh:7` |
| Phase 3 learned | 3 TAGCN + 5 rng | | tsp1000 20, tsp10000 **3**, Fu-10k 16 | `docs/phase3_learned_triage_report.md:141` |
| Phase 3 fast triage | 3 + 5 rng; diagnostic 1 seed, 4 instances | | tsp1000 20, Fu-10k 16 | `docs/phase3_fast_triage_report.md:20` |
| Phase 3 triage wsize | 3 + **3 rng** | | Fu-10k **first 8**, tsp1000 20 | `docs/phase3_triage_wsize_report.md:35,44` |
| Phase 5 text rules | one LLM draw, seed 0, temp 0 (paraphrases 0.8) | | 2 × 200 | `tsp/src/tsp_gnn/llm.py:chat:16` |
| Phase 5 open language | one draw | | L0 30, L1/L2 60 per n; q7 20/40, g12 20, q14t 12 | `docs/phase5_open_language_report.md:25-30` |
| Phase 5 vision | 3 reps (temp 0.5, seed = rep), 3 random draws | | 50 Fu-1k | `tsp/scripts/vision_probe.py:12-14,286` |
| Warehouse A | seed (orders + catalogue) | 5 | E1 240 episodes, E2b 660 | `warehouse/docs/phaseA_notes.md:31` |
| Warehouse B0 | seed | 5; GGO trained on **2** other seeds | one lot per seed/size | `phaseB0_report.py:series:55`, `table_ggo:175` |
| Warehouse B0b | seed | 5; **3 at 5×** robots; T1 check on **1** lot (`t1all`) | | `phaseB0b_notes.md:37,41` |
| Warehouse B0c | seed | 5; calibration seeds 100+; pilot rotation **seed 0 only** | 10 decision cells | `phaseB0c_notes.md:21,43` |
| Warehouse C (running) | request, one world each | 200 test, 30 dev | levels L1, L2 | `phaseC_language.py:12`, `requests.jsonl` (230 lines) |

### 8.1 Tests and intervals

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `tsp/scripts/summarize.py:paired_compare:90` | The unit of test is the instance; two-sided Wilcoxon on per-instance gap differences, one test per training seed | Gaps are skewed, instances are shared across methods (docstring l.91-96) | Medium: it says "this network on these instances", not "this architecture"; the instance p never sees training variance |
| `summarize.py:paired_compare:98-99` | `abs(d) ≤ 1e-6` pp (`TIE_TOL`, l.28) counts as a tie and is removed before scipy; scipy zeros then cannot occur | Numerical ties; report text "ties dropped" | Low: 1e-6 is far below any effect |
| `summarize.py:99` | scipy defaults: `method="auto"`, `correction=False`, `zero_method="wilcox"`. n > 50 gives asymptotic; n ≤ 50 exact with no ties; ties n ≤ 13 permutation; ties n 14–50 asymptotic (`_wilcoxon.py:128-129,230-242`) | Not documented: "(inferred)" default | Low for n ≥ 100; medium at n = 20 with ties (2-opt tours often tie), where the normal approximation is used without being reported |
| `summarize.py:table_paired:144` | Significant only if **max p over seeds** < Bonferroni threshold **and** medians agree in sign | Text l.279-280 | Medium: doubly conservative; an "n.s." at small n is low power, not equivalence |
| `summarize.py:main:258-259` | Bonferroni family = tests in that table only: 12 | l.259 comment | Medium: see next row |
| Families across reports: batch2 "all Table 2", checks 252 and 180, beam 54, TAGCN 16/12/10, scale 48 per n, hier 32 per n, window 12 per n, triage 15 per n, wsize 27, k02n 8 per size, benchmark 4 per set, learned 4 per set, fast 5 per set, triage wsize 6 per (w, set) | The correction is per table / per size, never across reports, though all reuse the same test files | Stated per report | Medium: the same 500 TSP-50 instances carry several hundred tests over Phases 2-3; no global correction, no untouched holdout |
| `phase2_checks_report.py:paired:148` vs `phase2_beam_report.py:paired:75` | checks: strict `same_sign`; beam: a zero median is neutral. `summarize`, exp1, batch2, datasize use the strict rule; beam, TAGCN, scale, hier, window, triage, wsize, k02n the neutral one | Docstring l.78-81: "corrected" rule | Medium: with many ties (TSP-20 + 2-opt) one rule gives n.s. and the other a verdict; earlier reports were not regenerated |
| `phase3_learned_triage_report.py:table_tests:170`, `phase3_fast_triage_report.py:table_tests:216`, `phase3_triage_wsize_report.py:table_tests:206`, `phase3_benchmark_report.py:wilcoxon_e2:218` | Policies are **averaged over seeds first**, then one Wilcoxon; no max p, no sign agreement | Report notes l.190-192 | Medium: a rule change inside Phase 3 (`phase3_triage_report.py:run_tests:280` still uses max p). Averaging removes seed variance, so these p are smaller than the earlier rule would give |
| same | 3 TAGCN seeds are compared with 5 random rng seeds (3 in wsize) | Random was cheap | Low: different noise on each side |
| `MIN_TEST = 20` in scale, hier, triage, window, wsize; `< 5` in benchmark and learned; none in fast and triage wsize | Below the gate only means and counts | "too few to test" | Low to medium: thresholds differ; n = 16 tested, n = 10 not |
| `phase3_triage_wsize_report.py:table_tests:220` | With n = 8, min exact p 0.0078 < threshold 0.0083 | Stated l.220 | Medium: a verdict needs 8/8 with no tie; nothing else can pass, so n.s. carries no information |
| `warehouse/scripts/phaseB0_report.py:paired:62`, `phaseB0b_report.py:paired:30`, `phaseB0c_report.py:paired:36` | Interval: Student t on per-seed % gains, df = n−1 (2.776 at 5), half = t·s/√n; the unit is the seed, paired | Notes l.9 | Medium: normality on n = 5 unchecked; a gain near a skewed tail gives a misleading ± |
| same | `wilcoxon(base, x, method="exact" if no zeros else "auto")`, two-sided, on raw differences while the interval is on % | l.66 | Low |
| same | "exact" with tied abs(d) is not exact: scipy rounds conservatively (`_wilcoxon.py:252-258`). A zero sends "auto" to a deterministic permutation | Not documented | Low: integer makespans make ties likely, p only moves up |
| `phaseB0_notes.md:9` | p floor 0.0625 at n = 5; 0.25 at n = 3 (`phaseB0b_notes.md:41`) | Correct | High for reading: **100 of 124 B0 cells sit on the floor**, so p ranks nothing, and no warehouse comparison can reach 0.05 |
| `warehouse/scripts/phaseA_report.py:table_paired:103` | Phase A uses **paired t** (`ttest_rel`) on mean delay, % as ratio of means | l.104 "paired by seed" | Low to medium: B0 onwards switched to Wilcoxon and mean of ratios, so A and B0 are not on one rule |
| `phaseA_report.py:ci:37`, `table_e1:85` | E1 rows give an **unpaired** per-arm ± though the design is paired | l.31 notes | Medium: overlapping bars can mislead; the paired table sits below |
| All warehouse reports | **No multiple-comparison correction**: 124 + 30 + 92 cells | Not documented; "read the interval" | Medium: with 92 cells of 95 % intervals about 5 exclude 0 by chance |
| `phase5_open_language_report.py:wilson:37`, `warehouse/scripts/phaseC_report.py:wilson:28` | Wilson at z = 1.96, with **rule** as Bernoulli unit, or instance | Standard | Medium: rules of a text are not independent; rule-level intervals are too narrow |
| `phase5_open_language_report.py:mcnemar:145`, `mcnemar_routes:223`, `tsp/scripts/open_hybrid.py:mcnemar:113`, `phaseC_report.py:mcnemar:112` | Exact two-sided binomial on discordant **rules** (TSP); rule and request (Phase C); discordant 0 gives p = 1 | Text "exact McNemar p" | Medium: see the pseudoreplication table. Phase C reports both units, TSP only rule and route |
| `docs/phase5_open_language_report.md` McNemar tables | **54 tests, no correction** (9 tables); headline p = 0.0003, h1 p = 0.0066 and p = 0.031 are all quoted raw | Not documented | Medium: at 0.05/54 = 9.3e-4 the 0.0003 survives; 0.0066 and 0.031 do not |
| `phase3_triage_report.py:136-163`, `phase5_vision_probe_report.py:54` | AP and Spearman pooled over all windows, ± std over seeds, **no CI and no test** | Descriptive | Medium: "AP +0.04-0.05" has no uncertainty attached |
| `docs/phase5_text_rules_llm_report.md:36-40` | "Better than chance? No": precision 27.7 % / recall 35.0 % against random 29 / 29, compared by eye | n = 200 only | Low: direction is safe, no test exists in the generator |
| TSP gap tables, all | No CI at all: mean ± std over 3 seeds | "run-to-run variation" (`summarize.py:269`) | Medium: a 3-seed std is itself very uncertain; model rankings (Gated < GATv2 < TAGCN < GCN) are never tested between models |
| `phase2_scale_report`, `hier`, `window`, `beam`, `summarize` use `ddof=1`; `phase2_k02n_report.py:70`, `phase3_benchmark_report.py:244,388`, `phase3_learned_triage_report.py:47,156`, `phase3_fast_triage_report.py:89,162`, `phase3_triage_wsize_report.py:56,193`, `phase5_vision_probe_report.py:40` use `np.std` (`ddof=0`) | Same "±" symbol, different estimator | Not documented | Low to medium: with 3 seeds ddof 0 is 0.82× ddof 1, so these std look 18 % tighter |
| `lkh3_equal_time`-based and interpolated equal-time claims (`docs/phase3_fast_triage_report.md:14-16`, `phase3_triage_wsize_report.md:9-13`) | Gain "at the same time" is read by **interpolating** random's gap-time curve; a point estimate from one timed run | Interpolation is the method | High for those claims: no CI on time or on the interpolated difference (−0.15 pp, −0.08 pp, ≈ 3.944 %) |

### 8.2 The 3 % criterion

| Where | Assumes | Why | If different (impact) |
|---|---|---|---|
| Origin | Not in the repo as a plan: no plan file, not in `warehouse/README.md`, not in CLAUDE.md. First appears in commit b16ab86 (`phaseB0b_notes.md:3,13`: "The rule of the plan for describing a model") | (inferred) a threshold in the user's phase plan, given in chat | Medium: the source and reason cannot be cited from the repo |
| Timing | Fixed **after** B0 data (`phaseB0_notes.md` says only "gap above noise", and quotes +4-5 % from Yu & Wolf 2023 as the expected size), **before** B0b and B0c data | (inferred) from the commit order: `git log -S"3 %"` shows no bar in B0 | Medium: chosen knowing hindsight was −0.4 / +0.6 % in B0 |
| Baseline | Differs per margin: T1 walk gap against C&W; T2 best clairvoyant against best online; T3 hindsight against **best not learned**; B0c against **best of (b, c, d)** | `phaseB0b_report.py:density:114`, `table_t2_value:87`, `phaseB0c_report.py:gaps:89` | Medium: one number, four denominators; also walking in T1, makespan in T3, orders/h in T2 |
| Decision statistic | The **point estimate** of an oracle against 3 % (`axhline(3)` `phaseB0b_report.py:135`, `phaseB0c_report.py:160`), not an interval bound | "None clears 3 %" | **High:** the correct use for "no margin" is the upper bound. +1.8 ± 3.0 (4.8), +3.1 ± 4.3 (7.4), +2.5 ± 2.9 (5.4), +2.3 ± 7.0 (9.3) all fail to exclude 3 %. B0c says so (`notes:11`), B0b does not |
| Metric | Makespan, driven by the last trip (`phaseB0b_notes.md:13`) | Plan metric | Medium: mean delay half-widths are smaller; the delay reading agrees (`phaseB0c_notes.md:15`), so the direction is stable |
| Oracle logic | An oracle's gain bounds any predictor's gain | Reasoning of B0 | High: holds only if the oracle is a true upper bound. Hindsight re-plans and traffic reacts, so it is one sample, not a bound (`phaseB0b_notes.md:43` admits T2's oracle understates; T1 best is "optimal only over its candidates", l.42) |
| Sum across cells | "Sign pattern across rows is firm" (`phaseB0c_notes.md:11`) | Replaces per-row power | Medium: rows share seeds, layouts and trips, so they are not independent evidence |

### 8.3 What "within noise" means, report by report

| Report | Wording | Actual basis | Problem |
|---|---|---|---|
| `tsp/docs/tagcn_report.md:8` | "close to noise" | Selected epochs change between same-seed reruns | Empirical, the best-founded one |
| `phase2_checks_report.md:29` | "0 of 36 cells differ significantly; mean abs. diff. 0.53 pp" | Rerun with the same seed | Gives a noise floor, never used elsewhere |
| `phase3_fast_triage_report.md:7` | "not significant", p = 0.19, n = 20 | Wilcoxon | Stated as "yes (not significant)": the wording treats n.s. as a weak yes |
| `warehouse/docs/phaseB0_notes.md:52` | "a difference below about 1 % is noise" | Median half-width 1.1, range 0.2-5.0 | Single figure for a wide range; post hoc |
| `phaseB0b_notes.md:41` | "about 1-2 % are noise" | Median 1.95, range 0.3-6.3 | Same; "+1.0 ± 0.9 (p = 0.0625)" at 3× is 5/5 seeds and called "within noise or close to it" |
| `phaseB0c_notes.md:11,57` | "3-7 % are noise in many rows" | Median 3.4, range 0.5-11.1 | The honest version: states the power problem |
| `phaseA_notes.md:45` | "one failure below that is noise of the arrivals" | Poisson arrivals about 12 % of the rate | Added after seeing the data, and says so |

### 8.4 Pseudoreplication and pairing

| Where | Assumes | Why | If different (impact) |
|---|---|---|---|
| `phase3_triage_report.py:table_tests`, `phase3_window_report.py` | Windows are collapsed to one gap per instance before the test | Correct unit | None: windows are not pseudoreplicated in the policy tests |
| `docs/phase3_triage_report.md:15-16`, `phase3_window_report.md:117-122`, `phase3_wsize_report.md:153-157`, `phase3_hier_report.md:136` | "9/9 instances at n = 10000", "30/30", "15/15" | Count of (instance, seed) pairs; the files hold **3, 10 and 5 instances** | **Medium: pseudoreplication, and the wording is wrong:** 9/9 is 3 instances × 3 seeds, in a report that says "instances" |
| `phase3_triage_report.py:136-163` | Windows at offsets 0 and w/2 share cities; AP treats them as independent | Descriptive | Low: only the AP precision is affected |
| `docs/phase3_learned_triage_report.md` Data | Train/validation split **by instance** | "halves by instance" | Correct: no window leakage across the split |
| `phase5_open_language_report.py:mcnemar:145` | Rules (2.4-2.5 per text) independent | Rule level | **Checked from `hybrid.json`:** L2 n = 200, q14 vs p1, instance sign test 28 up / 8 down, p = 0.0012; instance-exact McNemar 18 / 3, p = 0.0015; rule-level was 33 / 9, p = 0.00027. So p is about 4× optimistic, **the conclusion survives**. n = 1000 stays null (11 / 15) |
| `warehouse/scripts/phaseB0_report.py:paired` | Seed is the unit, episode metrics are one number per seed | Correct level | None for gains; `table_o3:146` pools subsets over seeds (mean, worst), descriptive only |
| `phaseA_report.py:e3:224` | Leave-one-seed-out for the hot cells; per-episode shares called inflated (`notes:53`) | Handled | None |
| `warehouse/scripts/phaseB0_report.py:series` | Pairing by seed fixes the orders and catalogue | `phaseA_notes.md:31` | True for A-B0b, which are deterministic given the seed except wall-clock limits (see 9) |
| B0c arms | Delays are random per seed; a seed pairs orders and trips, but (inferred, not verified) draws are not coupled once paths diverge | Half-widths 3-7 % fit | Medium: pairing removes less variance than in B0 |
| `docs/phase2_batch2_report.md:192` | For retrained configs seed pairing is **arbitrary**, the instances are what is paired | Stated (note 9) | Low to medium: seed s of A and of B share nothing; max p over 3 arbitrary pairs |
| `summarize.py:138` and all baseline comparisons | One deterministic baseline against 3 seeds, three p values on the same baseline | Baseline deterministic | Low |

### 8.5 Selection, stopping, post-hoc thresholds

| Where | Assumes | Why | If different (impact) |
|---|---|---|---|
| `phaseB0_report.py:best_rival:107`, `phaseB0b_report.py:density:114`, `phaseB0c_report.py:gaps:89` | Rival = lowest mean of 7 / 4 / 3 arms **on the same 5 seeds** used for the gain | "leans towards the rules" (`phaseB0c_notes.md:9`) | **High for the direction of the conclusion:** winner's curse lowers the oracle's margin, in the direction of "no margin". B0 and B0b do not mention it |
| `phaseB0_margins.py:o2b3:269-282`, `phaseB0c_report.py:third3:93` | Hindsight keeps the best of 3 runs per seed; B0c adds mean of 3 as the fair number; B0 and B0b report only best of 3 | Stated | Medium: the two biases oppose each other; size unknown in B0/B0b |
| `phaseB0b_report.py:t2_series:68-74` | Oracle arm takes the best λ per cell by mean on the same seeds | Not stated | Low: helps the oracle, which still stays below 3 % |
| `phaseB0_report.py:load:47` | `o4pibt` first version dropped, a rule that livelocked could win | Comment | Low |
| `phaseA_notes.md:43,45` | 5 % slope fixed before the run; sweep extended after the first run did not break "large"; "first rate that fails with every higher one" added after the data | Stated | Low: disclosed |
| `phaseB0b_notes.md:37` | 5× ran 3 seeds due to budget (stopping by time, not by evidence); `t1all` on 1 lot | Stated | Medium: the densest cell, the one most likely to show a margin, has the least evidence |
| `phaseB0c_notes.md:53` | Prioritized not run on large; no `e0` at 90 robots; a first version was discarded (`results/warehouse/phaseB0c_v1`) | Stated | Low to medium: v1 data were seen before the design changed (the defect was caught at the first cell, memory note) |
| `docs/phase5_open_language_report.md:23-30` | Samples cut after step 0 (16 h estimate): below the plan of 300 rules per cell | Stated | Medium: cuts came from time, equal for arms, no optional stopping on results |
| `docs/phase5_open_language_report.md:21` | p1 tuned in 4 rounds on 20 held-out paragraphs, hash frozen before the audit; but dev L2 at n = 200 only | Stated | Medium: p1 tuned for n = 200, while q14 wins only at n = 200, so the bias runs against the headline |
| `docs/phase2_batch2_report.md:193` (note 11) | The automatic rule "≥ 75 % of the TSP-20 advantage recovered → capacity" was overridden: **Summary hand-rewritten after reading the results**; rerunning the generator overwrites it | Stated | Medium: post-hoc change of a decision rule, and report not reproducible from its generator |
| `phase3_triage_report.py:474` | Thresholds set on tsp200, so tsp200 is in-sample; "top-q" grid chosen in advance by plan | Stated | Low |
| `phase3_benchmark` † mark: > 25 % of instances over 1.1 × budget | Post-hoc cut-offs | Report note | Low |
| `docs/phase3_learned_triage_report.md:54` | Fu-1k changed role from test set to training after the benchmark | Stated | Low: Fu-10k stays a test set, but several reports have read it |
| All Phase 3 → 5 | Sequential design on the same test sets (w = 20 from wsize, K = 3 from TAGCN, k = 0.5 n from batch2) | Pragmatic | Medium: forking paths; all gaps are slightly optimistic, none is a clean holdout estimate |
| `phaseC_language.py:12`, tag `phaseC-parser-frozen` | Parsers frozen on 30 dev requests before the test run; `DEV = 1000` splits ids | Tag at 6aeb1a5 | Good practice, stronger than Phase 5 |

## 9. Infrastructure

### 9.1 GPU, CPU, load

| Where (file:function) | Assumes | Why | If different (impact) |
|---|---|---|---|
| `tsp/src/tsp_gnn/utils.py:limit_gpu_memory:27` | 5.5 GiB per process through `set_per_process_memory_fraction`, so < 6 GiB with the CUDA context | CLAUDE.md rule; docstring | Low: enforced by the allocator in training and evaluation. Real peak over 104 training runs: 2.29 GiB (`history.json`, `tagcn_k3_tsp100_seed1`) |
| Callers: `train_model.py:59`, `evaluate.py:39`, `beam_eval.py:83`, `scale_eval.py:167`, `window_eval.py:57`, `hier_eval.py:67`, `triage_log.py:32`, `wsize_eval.py:80`, `heatcand.py:24` | Every GPU script is capped (grep finds none uncapped) | Rule | Low; the cap is **per process**, 8 processes could sum to 44 GiB in principle. Peak seen 4787 MiB total (`phase3_wsize.log`) |
| `llm.py:GpuGuard:79` | LLM rule is different: keep ≥ 2048 MiB free, poll every 300 s, wait and never stop | Docstring l.80-82 | Medium: the guard is on free memory, not the 6 GiB per job limit. CLAUDE.md says "cada treino", so literally not a violation, but no written LLM limit exists |
| `phaseC_language.py:MODELS:35` | q14 needs 11000 MiB, q7 6000, gemma3 10000 | Measured earlier | At audit time the private llama-server held 9562 MiB at nice 10, 5348 MiB free in the last guard check |
| No guard for torch runs except `run_phase5_vision.sh:48` (waits for others < 6 GiB) | Other users leave memory | Rule of thumb | Medium: `run_phase3_fast_triage_b10k.sh:4` documents a CUDA OOM when another project held 12.6-13.2 GiB; a failed run, not a wrong number |
| CLAUDE.md | "~7 GB livres" | Written early | Low, stale: the T4 has 15360 MiB; another project was seen holding 12.6-13.2 GiB |
| `run_*.sh` | Self-renice to 10 (`exec nice -n $((10 - $(nice)))`, all of `tsp/scripts/run_*.sh`); warehouse uses `nice -n 10` per step | Rule | Low: niceness loses CPU under load, so timing depends on other users' load |
| `env.sh:5` | `OMP_NUM_THREADS=8` | "não ocupar os 24 CPUs" | Low |
| Run scripts | OMP = 1 in all phase 3 policy, phase 5 and warehouse runs; 2 in beam, scale, hier, k02n; 4 for sklearn training; **unset override (8 from env.sh)** in batch2, checks, datasize, exp1, TAGCN main | Parallel lanes | Low: gaps unaffected |
| `torch.set_num_threads` in eval scripts (1 or 2: `scale_eval.py:56`, `triage_eval.py:69`, `triage_ablation.py:76`) | Overrides OMP | | Low. Fast triage tested threads as a cause: not it (`phase3_fast_triage_report`, diag table) |
| Process counts | ≤ 8 (`MAX_PROCS` in `make_dataset`, `baselines`, `lkh3_equal_time`, `lkh3_reference`); window 5; triage 4; bench E2 6 per set, E3 4; learned 8; fast 7 then 1; scale 4 + 3 data; hier 3; beam 3 lanes; warehouse 4; phase 5 pipeline 7-8 | Rule | Medium for times, see 9.2 |
| Everywhere | **No taskset / CPU pinning** (grep empty); two sockets, 12 cores, 24 threads, no NUMA binding | Not considered | Medium: processes may share a physical core; per-window LKH times depend on the scheduler |
| `gpu_thermal_log.sh` | Logs every 30 s, only in the Phase 5 open language run (1364 lines) | "a hot T4 drops to 300 MHz" | Medium: no thermal log for earlier LLM runs (steps 0-2, vision) or Phase C; Phase C records temp and clock on the latency sample only |
| `llm.py:wait_cool:73` | ≤ 70 C before a latency | Memory `llm-latency-alone` | Low: applied to the alone sample |
| `docs/phase5_open_language_report.md:45,57` | The T4 stays at 83-85 C, 300 MHz; median 315 MHz, under 500 MHz in 93 % of samples | Passive cooling | Medium: LLM throughput in run tables is the throttled state. At audit time, during phase C: 84 C and 420 MHz |

### 9.2 Times: which clock, under which load

| Where | Measure | Load during measurement | If different (impact) |
|---|---|---|---|
| All of tsp and warehouse | `time.perf_counter` (wall), `time.time` in `make_dataset`, `train_model`. **`process_time` is used nowhere** (grep) | | High for any comparison under load: wall clock counts other users' work |
| `torch.cuda.synchronize` before reading the clock: `train.py:70`, `scale_eval.py:97`, `hier.py:165`, `heatcand.py:44`, `fast_triage.py:90,217` | GPU timing is honest | | Low |
| `tsp/src/tsp_gnn/lkh3.py:solve:91,113` | Wall, **includes writing files**, on `/scratch` (local LVM, not tmpfs or AFS) | | Low to medium at w = 20: file I/O is a real share of a 7 ms call |
| `tsp/third_party/LKH-3/SRC/GetTime.c:1`, `FindTour.c:52` | LKH `TIME_LIMIT` is **user CPU time** (`HAVE_GETRUSAGE`), checked between trials | | Medium: E1 sets budgets in wall time and gives them to a CPU-time limit. Overrun is blamed only on "between trials" (`phase3_benchmark_report.md:98`); contention adds to it. Final comparison uses real time, so the table is valid; fewer trials fit under load |
| `phase2_beam_report.md:125` | Decode, 3 evaluations in parallel, nice 10 | Own + others, unlogged | Low: times are secondary there |
| `phase2_scale_report.md:251` | Network alone on GPU; decode 4 workers | Unlogged | Medium: time columns compare methods run on different days |
| `phase3_window_report.md:132`, `phase3_hier_report.md:193` | 5 and 3 processes | Unlogged | Low to medium |
| `phase3_learned_triage_report.md:57` | 8 processes, run together so comparable | Unlogged outside load | Medium: equal-time verdict ("random at 50 % wins") hangs on this; the fast report shows it flips under contention: 7 processes give learned LKH +22 %, 17.4 vs 13.9 s |
| `phase3_fast_triage_report.md:63`, `:165` table | Sequential, one process; load rose 0.4 → 6-9 from another project | Measured before and after only | Medium: random repeated (13.67 → 13.79 s), so load seems balanced; **one repetition, no CI**, and load is not sampled during runs |
| `phase3_triage_wsize_report.md:39-41` | w = 50 quiet (1.1-1.5); w = 20 top 20/50 % at load 6-9 | **Mixed, stated** | Medium for w = 20 at 20/50 %. The headline (w = 50 top 20 %, 15.6 s, against LKH on all w = 20, 21.3 s) compares two quiet runs: fine |
| `phase3_benchmark_report.md:281-282` | E3 4 processes; E1 8 LKH runs | Unlogged | High for "LKH-3 28-110× at equal time": both sides contended, but differently (4 vs 8). Size of the effect is large, so the conclusion survives |
| `run_phase5_*`, `llm.py:lkh_running:45` | `lkh_running` per call; alone sample of 30 per arm | Load 2-7 at the latency calls (`phase5_open_language_report.md:477`) | Medium: "alone" means no LKH of mine; other users are not controlled |
| `phase5_vision_probe.md:38-41` | Another Ollama shared GPU from 16:19 | Mixed periods | Low: vision loses on quality |
| Warehouse `sim.py:109-119` `step_ms`, `phaseA_baselines.py:71`, `phaseB0_margins.py:172` | Wall per step, up to 4 processes, unlogged load | Unlogged | Medium: "86 and 428 times the computation per step" (`phaseB0_notes.md:44`) depends on this; large ratio survives |
| `warehouse/src/warehouse/mapf.py:cbs:336,382-390` | **CBS results depend on the wall clock**: a time-out sends a step to PIBT; `budget`, `cooldown` | | **Medium to high:** CBS makespan is not a function of the seed alone; under load more steps fall to PIBT. Phase A: 23 % of steps, 3 of 15 episodes at 900 s (`phaseA_notes.md:35`). The conclusion "CBS barely beats prioritized" leans towards PIBT dilution |
| `phaseB0_margins.py:249` (LNS 1200 s), `phaseB0b_margins.py` T1 300 s HiGHS, LKH 120/300 s CVRP | Wall limits on search | | Low to medium: `large` stops at the limit with bound 1.3 % lower (`notes:13`) |
| `phaseB0c_report.py:time_spent:179-181`, B0/B0b Budget notes | "h of CPU" is the **sum of wall `elapsed` per process** | | Low: label wrong; notes partly admit (`phaseB0c_notes.md:53`) |

### 9.3 Environment, git, backup, tests

| Where | Assumes | Why | If different (impact) |
|---|---|---|---|
| `.gitignore` | Not in git: `.venv`, `tsp/data`, `tsp/results`, `milp/data`, `milp/results`, `warehouse/data`, `warehouse/results`, `tsp/third_party`, `*.pt`, `tmp/` | Size; LKH licence (`build_lkh3.sh:7`) | **High for reproducibility:** every number in every report rests on files only on `/scratch`, which has **no backup** (CLAUDE.md). Git holds 305 files: code and reports only |
| Sizes | `tsp/data` 405 MB, `tsp/results` 213 MB, `warehouse/results` 211 MB, `third_party` 23 MB | | Cheap to copy: under 1 GB; no backup exists at any size |
| `tsp/results/*_notes.md` | The hand-written parts of Phase 2-3 reports live in `results/`, **outside git** (`phase3_triage_report.py:435`, `phase3_hier_report.py:359`, `phase3_benchmark_report.py:131`); warehouse notes are in `docs/`, in git | History | Medium: regenerating a TSP report needs an unversioned file; if missing, it prints "(notes missing)" and the report silently loses its conclusions |
| Reports in `tsp/docs/` | Copies of `tsp/results/*_report.md` | Run scripts copy | Low: committed copy is the record |
| Conclusions not reproducible from git alone | All. Data can be regenerated from seeds (`make_dataset`, `make_scale_data`), LKH-3 from the pinned hash; **checkpoints cannot** (not bit-reproducible, 9.4); **LLM answers and manual audits cannot** (`l2_audit.json`, models outside repo at `/scratch/pcanaste/tools/ollama_models`, no digests recorded in the repo) | | High for Phase 5 and C |
| `/scratch/pcanaste/tools`, `env.sh:3` (`/scratch/pcanaste/node`) | Ollama and models live outside the project folder | Private instance | Low to medium: conflicts with "Trabalhar SÓ dentro desta pasta" as written; also unbacked |
| No lock file | `pip freeze` state is only in `.venv` | Not done | Medium: on rebuild, scipy's `method="auto"` behaviour and PyG scatter may change |
| Fu benchmark files `tsp/data/benchmark` | Downloaded, not in git | | Medium: source recorded by `bench_datasets.py` / `convert_benchmark.py` only |
| Meta: `git_info` in `tsp/src/tsp_gnn/utils.py:18` | Commit + dirty flag per run | Traceability | **151 result JSONs have `git_dirty: true`**, including Phase 1 seeds 1-2 (`table.md:3`: "5d14168 (uncommitted changes)"), all phase 2 datasize (15), all phase 3 benchmark (36). Dirty may mean just an untracked report, so the commit does not pin code there. Warehouse stores a commit, no dirty flag |
| Memory `phase5-step2.md:16`, `phase2-scale.md:16` | `git push` fails with "Host key verification failed" when AFS tokens expire; `kinit && aklog` from the user | HOME on AFS | Medium: the push at session end is the only off-site copy, and it fails by token expiry |
| `run_phase2_batch2.sh:31-32` and later | Long runs keep `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `GIT_CONFIG_GLOBAL=/dev/null` off AFS | "AFS tokens may expire after logout" | Low: mitigated. Warehouse runs do not set them (`run_phaseB0c.sh`); `commit()` could return empty after expiry (inferred, unverified) |
| Memory `phase5-step2.md:15` | The permission system blocks `kill`; the user runs it | Rule | Low |
| `env.sh:4` | `TMPDIR` = project `tmp/` (3.0 GB, git-ignored, on `/scratch`); warehouse LKH work dirs use it (`sequencing.py:89`, `constraints.py:123`), TSP uses `results/*_work` | `/tmp` is small | Low; falls back to `/tmp` if `env.sh` was not sourced, all `run_*.sh` source it |
| LKH work dirs | One dir per pid | `sequencing.py:89` | Low |

### 9.4 Determinism

| Where | Assumes | Why | If different (impact) |
|---|---|---|---|
| `utils.py:set_seed:10` | random, numpy, torch, cuda seeded; `train_model.py:66` seeds batch order | Reproducible | **Not bit-reproducible, and the code never tries:** `torch.backends.cudnn.deterministic`, `use_deterministic_algorithms`: grep empty |
| `docs/tagcn_report.md:184` Table 8 | Same seed reruns differ: atomic `scatter` sums | Measured | **High:** same code, seed and data: selected epoch 1 → 19, TSP-50 greedy **13.91 → 19.93 %**; identical per-instance gaps only 103-638 of 1500. So a seed fixes nothing, and the rerun spread is as large as many reported differences |
| `docs/phase2_checks_report.md:29` | LN rerun: mean abs. diff. 0.53 pp, 0 of 36 cells significant | Measured | High: a measured floor of about 0.5 pp at greedy, never used as a threshold in Phases 2-3 |
| `phase2_beam_report.md:135`, `phase2_k02n_report.md:140` | Even inference differs slightly (GPU sums flip near-ties) | Stated | Low |
| `DataLoader` (`train_model.py:67-68`) | `num_workers` default 0, no worker seeds needed; grep finds no `num_workers` | | None |
| CPU / GPU | Training GPU only; decoding NumPy on CPU; sklearn on CPU | | Low: no CPU-GPU comparison of the same model |
| `llm.py:chat:21` | seed + temperature 0 repeatable | Option sent | Medium: `phase5_text_rules_llm_report.md:66-70`: an abort did **not** repeat with the same seed; Ollama is not deterministic |
| Warehouse | `random.Random(seed)`, `default_rng`; deterministic apart from the wall-clock limits above; `PYTHONHASHSEED` unset, a narrow grep found no iteration over sets of strings in `warehouse/src` (not exhaustive) | | Low. A test covers exact repetition (`test_margins.py:test_fixed_plan_repeats_the_episode:101`) |
| `make_dataset.py`, `make_text_rules.py` | LKH / elkai labels fixed by data seed; `MAX_PROCS` | | Low |

### 9.5 Tests

| Where | Covers | Does not cover | Impact |
|---|---|---|---|
| `tsp/tests/` 183 tests, 38 files | Decode, graph, models, 2-opt, rules, triage, window, scale, report helpers | | |
| `tsp/tests/test_summarize.py` | `paired_compare` by hand, all-ties p = 1, formatting | Exact vs asymptotic path | Low |
| Report stats tests | `test_phase2_{beam,k02n,datasize}_report`, `test_phase3_benchmark_report`, `test_summarize` | **No test file for** `phase3_fast_triage_report`, `phase3_learned_triage_report`, `phase3_triage_wsize_report`, `phase5_open_language_report`, `phase5_text_llm_report`, `phase5_vision_probe_report`; **wilson and McNemar untested** (grep) | Medium |
| `warehouse/tests/` 126 tests, 7 files | Env, exec 28, mapf 9, margins 41, phaseC 25, sequencing 5, verifier 13 | **No test of any report generator**: `paired`, Student half-width, `gaps`, `wilson`, `mcnemar` | Medium: the numbers that decide B0-B0c are untested |
| None | | No test runs in CI; no CI exists | Low |

## 10. Where code and reports disagree

Found while auditing. Ids are used in the tables above.

### 10.1 TSP data and model (M)

Certainty: high = read in code and data; medium = supported, mechanism inferred.

- **M-1.** **Closed-tour model used for fixed-end paths, undocumented (high).** `hier.py:Heatmaps:155` builds `knn_graph(rescale(pts), ...)`, with `x` = coords only (`scale.py:knn_graph:63`, `models.py:84`). Labels are closed tours (`graph.py:tour_labels:53`). No report mentions this: grep for it in `docs/` finds nothing, and `phase3_wsize_report.md`'s "Likely reason" omits it.
- **M-2.** **Report recommends changing selection, code and later phases do not (high).** `tagcn_report.md` note 2: "A different selection rule (val loss or last epoch) should be used for TAGCN"; `README.md:59`: "the selection rule will be revised". Yet `train_model.py:102-107`, `window_eval.py:58`, `wsize_eval.py:81`, `hier_eval.py:68`, `heatcand.py:20` all use `best.pt`. Epochs: 12, 1, 4 (`results/runs/tagcn_k3_seed*/eval.json`); TSP-50 1, 1, 10; TSP-100 5, 3, 1.
- **M-3.** **"Training scale" claim false in distribution (high).** `hier.py:12` and `phase3_window_report.md` vs measured windows above.
- **M-4.** **kfrac and scale confounded; report treats them as separate (high on the numbers, medium on the conclusion).** `phase2_batch2_report.md` points 1–2 vs table above; the k-mean ratio is 0.96 at k = 0.5n. `fase1_explicacao.md:~388` says the graph/tour ratio stays ~2, but at k = 0.5n it becomes 3.3 and 4.7.
- **M-5.** **Normalization hypothesis only half implemented (high).** CLAUDE.md goal is "normalizar as distâncias"; `graph.py:normalize_edges` leaves `x` raw while the encoder of GCN/TAGCN sees only `x`-derived features (`models.py:112,118`).
- **M-6.** **"Same problem" vs ATSP path (high).** `data.py:41` says "Same problem as solve_lkh"; `solve_lkh` goes through `solve_int_matrix`, which is hard-coded `TYPE: ATSP` (`LKHmain.c:234`). `window.py:66` too. elkai's own `DistanceMatrix.solve_tsp` (types.py:27) picks TSP when symmetric. Optimum the same; solver path and time differ. No doc mentions it.
- **M-7.** **E0 agreement oversold (high).** `phase3_benchmark_report.md` caveat 1 uses "identical" as validation; 19/20 tours are identical edge sets (measured), same defaults and SEED 1 in both (`ReadParameters.c:576`). Two versions (3.0.8 vs 3.0.6) give the same search here, so this is not an independent check. On tsp10000_test E0 is longer on 2 of 3 instances.
- **M-8.** **"Exact optimum" (high).** `phase3_window_report.md`: "(the exact optimum of each window)" vs `window.py:lkh_path:57` `runs=1`. The module docstring calls it "an upper bound of a window".
- **M-9.** **"Test is never used to choose" (medium).** `fase1_explicacao.md` §6 vs k = 0.5n, TAGCN K=3, B and w chosen after test results on TSP-20/50/100 and tsp1000/10000 (`tagcn_report.md` note 1, then `hier_eval.py:48` default). Phase 3 uses other instances and Fu sets are fresh, so bias is limited, but tsp1000/10000 are reused for w and policies.
- **M-10.** **Unequal checkpoint rule inside comparison tables (high).** `run_phase2_beam.sh:6-8`: phase1 and data100k use best.pt; ln_last uses last.pt. `tagcn_report` Table 1 likewise. Differences mix selection with normalization.
- **M-11.** **Network time understated (medium).** `hier.py:157` starts the clock after `knn_graph`; `phase3_window_report.md`: "Network time: only 0.2–0.3 s ... Most of it is the Python path decoder". Graph building is neither model nor decoder.
- **M-12.** **"Same reach" across encoders (medium).** `models.py:default_layers` and `tagcn_report` assume 6 hops is comparable; at k = 0.5n graph diameter ≈ 2.
- **M-13.** **Docstring vs code (low).** `scale_baselines.py:6` says two k-modes; `K_MODES` now has 5 and `lists` builds all of them at line 47, so rerunning at n = 10000 would build k = 5000 lists. Existing results unaffected.
- **M-14.** **2-opt variants table not reproducible (medium).** `README.md:202` and `fase1_explicacao.md:286` report variants, but `decode.py` has only one and no script or commit message contains the others (grep and `git log --grep`).
- **M-15.** **Comment "LKH only accepts integer distances" (low).** `data.py:11` vs elkai `deprecated.py:12` ("The C library accepts floats natively now!"). Harmless.
- **M-16.** **0.58 vs √(20/50) (low).** CLAUDE.md writes "~0,58× ... (≈ √(20/50))", which is 0.63. `fase1_explicacao.md:384` explains boundary effects, but 0.57–0.59 is a k=10 graph-edge and tour-edge ratio, not nearest-neighbour spacing.
- **M-17.** **Silent lack of assert (low).** `hier.py:Heatmaps`, `window_eval.py`, `wsize_eval.py`, `hier_eval.py` have no `edge_norm == none` check, unlike `scale_eval.py:172`. No wrong run found.

### 10.2 Triage and LKH-3 (L)

- **L-1.** **Undocumented ATSP.** `src/window.py:7,15,57` and `src/data.py:41` describe a TSP; `elkai/elkai/deprecated.py` calls `_deprecated_solve` → `elkai/LKH-3.0.8/SRC/LKHmain.c:ElkaiDeprecatedSolve:212` writes `TYPE: ATSP`. `grep ATSP` finds nothing in tsp/src, scripts or docs. Sure, assuming the installed 2.0.1 wheel matches the 2.0.1 source here (version matches).
- **L-2.** **Report's w = 50 explanation vs counts.** docs/phase3_triage_wsize_report.md summary vs computed `__improved` in results/phase3_triage_wsize/seq/w50/*: random 111.5, learned 96.5, disagreement 78.4 (seed 0, top 20 %). Medium sure about the cause, sure about the counts.
- **L-3.** **"Score outside the network".** docs/phase3_learned_triage_report.md footnote "*score* = time to compute the scores outside the network" vs `src/triage.py:258-284` (timer wraps the network) and docstring line 231; `scripts/phase3_learned_triage_report.py:81-82` does not subtract. Same overlap in the fast report's "of which" columns. Sure.
- **L-4.** **"The network costs ≈ 0.4 s at 1k and ≈ 4 s at 10k"** (docs/phase3_benchmark_report.md Key numbers) vs ms_net 0.06 / 0.27 s (learned report table) and `src/hier.py:155-157`. It was Python graph building. Sure. No correction note in that report.
- **L-5.** **Hardware.** Report: "Xeon E5-2620 v3 (8 cores)", "8 runs shared the 8 cores". `lscpu`: 2 sockets × 6 cores × 2 threads = 24 logical CPUs, Cpus_allowed 0-23, no cgroup quota; 8 is `OMP_NUM_THREADS=8` in env.sh:5 and `MAX_PROCS`. Sure. So the † overshoot is not explained by 8 runs filling the machine.
- **L-6.** **TIME_LIMIT is user CPU time, not wall.** `LKH306/GetTime.c`, `FindTour.c:52`. `scripts/lkh3_e1_prep.py:105` subtracts CPU trial time from wall. Report and `src/lkh3.py:13` do not say it. Sure.
- **L-7.** **E1 fixed cost "with a single process"** (docs/phase3_benchmark_report.md caveat 5) omits that E0 was running 8 processes (`e1_prep/run_e1_prep.sh:2`). Sure.
- **L-8.** **"Tests never used to fit or choose"** (learned report, `scripts/learned_triage_train.py:5`) vs `scripts/fast_triage_decide.py:9` gating version B on tsp1000_test and Fu-10k, and (w, x) chosen on Fu-10k in the wsize report. Same tsp1000 / tsp10000 sets were read through all of phase 2-3 before the features were designed. Sure for the gate; the choice of features is (inferred).
- **L-9.** **Equal-time numbers are not script output.** The summary numbers of docs/phase3_triage_wsize_report.md and "≈ 3.944 %" of the fast report come from notes (`results/*_notes.md`); no `interp` in scripts. They are reproducible by hand. Sure.
- **L-10.** **Learned at w = 50 is not one model type.** `results/phase3_triage_wsize/models/w50/tagcn_k3_seed2/metrics.json` chose `hgb_reg`, seeds 0-1 `logreg`. Report row says "Learned (chosen)", and the earlier text says the logistic was chosen. Sure, low impact.
- **L-11.** **Comment vs code on hang protection.** `src/rules.py:295` documents that LKH can spin forever; `scripts/make_text_rules.py:41,48` passes no `timeout`. `scripts/pipeline_eval.py:71`, `make_open_language.py:47`, `open_pipeline.py:46` do. Sure, no harm observed (max 510 s).
- **L-12.** **"LKH improves" as upper bound.** `src/triage.py:11`, `src/window.py:15`, triage report line "LKH is at least as good": computed negative gains down to −1.30 %. Sure, low impact.
- **L-13.** **Benchmark does not state windows and E1 use different LKH codes.** `src/window.py` (elkai, 3.0.8 modified) vs `src/lkh3.py` (3.0.6). Report only names 3.0.8 in the "Stored tours" column. Sure, low impact.
- **L-14.** **`git_dirty: True`** in results/phase3_benchmark/e1/tsp10000_test_concorde.json (commit 101b9d7). Exact code of E1 not pinned. Sure, low impact.

### 10.3 Text rules and LLM (R)

- **R-1.** (bug, high confidence, live.) WS/phaseC_language.py:main_c:242,251-253: `seen` is one set for all requests and is keyed by `key` alone; `key` hashes the accepted primitives (:205-206) and has no request id, while the output file is `{idx}_{key}.json` (:215). Two requests with the same accepted set, above all the empty set, run only once, so `episode(idx, key)` returns None (phaseC_report.py:80) and `table_c` skips them (:168). Worst translations disappear without a message. Primitives of orders and shelves are world-specific, so collisions need an empty set or a set of `close` primitives only. Logs so far: p0 L2 has 5 empty answers; any answer fully dropped or rejected adds to them. Stage c has not started, so it can still be fixed; a rerun only recovers one per key.
- **R-2.** (high confidence on the mechanism, effect unmeasured.) T/rules.py:cost_matrix:205-208 adds the soft penalty to W; write_problem:246-248 writes W for TSPTW; third_party/LKH-3/SRC/Penalty_TSPTW.c:20 uses C(N, NextN) as time. So driving in a zone counts double as TIME in the windows. Docstring :21-25 and phase5_text_rules_report.py:33 do not say so. Logs: 16 + 17 instances, mean penalty/length 0.22-0.23. Routes stay truly feasible (check_all uses true distances), but windows are tighter than the text, in the reference and in every arm that keeps a soft zone with a window.
- **R-3.** (medium confidence.) Report :372 "fixed on the held-out development texts before the test texts were scored". p1/q14 test answers were scored in the report committed in 25dc6cd (2026-11-01 05:58); open_hybrid first appears in 9920ae0 (09:07), hybrid.json 09:04. No dev evaluation exists in TS/open_hybrid.py or tests/test_open_hybrid.py (grep "dev": nothing). Only true if read as "before the hybrid arms were scored".
- **R-4.** (high.) TS/planner_probe.py:CASES:131-133: case 7 has 0.04 s left and accepts `triage frac 20`, which costs 0.11 s (EFFECTS:78). SYSTEM :31 says such an action "gives no tour at all". The scorer rewards it.
- **R-5.** (high.) "correct / 40": repeats at T = 0, seed 0 are not independent. Logs: planner_probe_calls 11 of 16 groups identical, v2 34 of 48, step 3 11 of 16; never more than 2 versions.
- **R-6.** (high, low impact.) Verifier JSON failures: phase5_text_llm_report.py:268,271 count them wrong / false alarm; phase5_open_language_report.py:326,330 use `claimed or []`, so a failure on an oracle route is "right". Same label, two measures.
- **R-7.** (medium.) Report :471 says L2 accuracy is "a lower bound for every arm"; T/open_parser.py:FILLER:53 removes exactly the phrases behind audited errors #12 and #47, so p1 can match a label the text contradicts.
- **R-8.** (high.) TS/run_phase5_text_llm.sh runs C in background with D; llm_verify_eval.py has no lkh_running; report :267-268 latencies are unmarked (the resumed part of D ran alone, so a mix). Against memory rule llm-latency-alone, though that rule is later.
- **R-9.** (high, low impact.) Column "hard rules broken / route" (report :233, table_c:214) counts primitives (12.73 at n=1000 with 2-3 rules). Phase C "true rules broken per episode" (phaseC_report.py:159) is worse: forbidden_cells counts robot-steps (W/verify.py:51).
- **R-10.** (medium, not yet reported.) W/reqparser.py:6 "about 2 h from the L1 templates and the held-out development paragraphs". Files: requests.jsonl 14:14:04, l2_dev_v1 14:19:13, l2_dev.log 14:31:09, reqparser.py 14:32, commit 14:32:46, test L2 start 14:32:47. So at most 18 min with any dev text and about 1.5 min with the complete final dev set. The freeze itself is real (no later change).
- **R-11.** (low.) T/open_gen.py:sample_rules:301 says at most 2 mandatory "with a precedence or window"; :327 applies it always.
- **R-12.** (low.) phase5_open_language_report.py:74 stores schema_errors, never shown. Logs: q7 L2 n=1000 drops 14 items in 40 answers; g12 up to 6 per cell. Warehouse report does show them (table_b).
- **R-13.** (low.) `_int` (T/formalize.py:99) raises OverflowError on inf; not caught at :137, T/open_dsl.py:205 or W/dsl.py:238 (these catch ValueError, TypeError, etc.). JSON `1e999` as a customer would stop a run. Not observed.
- **R-14.** (low.) results/phase5/text_rules_data.md (phase5_text_rules_report.py:96-97) says "either direction for precedences and windows"; TS/make_text_rules.py:56 checks the listed direction only.
- **R-15.** (low.) TS/open_formalize.py:35 passes p0 primitives unnormalised, so `new_forb & mand` (T/open_dsl.py:361) can miss a contradiction for p0.
- **R-16.** (note.) l2_audit.json (warehouse) was written 15:58:46, 102 s after the last L2 paragraph (15:57:04), and before q14. It is blind to the LLM arms but not to p1 (L2 done 15:57). 60 texts in that gap suggests it was prepared while L2 ran; the file alone does not show which.

### 10.4 Warehouse (W)

- **W-1.** **T2 "saturation" is not saturation for batched arms.** Sure (arithmetic on `t2_*.json`). `phaseB0b_margins.py:37,113-115` sets x2.0 from the PIBT one-order E1 rate: 0.2839 / 0.520 per step = 1022 / 1872 per h. Second-half throughput of (a) = 995 / 1888; backlog mean / end = 102.6 / 101.4 and 221.6 / 222.8 (flat). `docs/phaseB0b_margins.md:31` states x2.0 was added because at the medium rate "arrivals fix the throughput", which remains true at x2.0 for batched arms. The one-order arm is really saturated (backlog 206 → 258).
- **W-2.** **T3 column mislabelled.** Sure. `phaseB0b_report.py:119,124` calls `fallbacks/calls` "steps under PIBT %"; `phaseA_report.py:94`, `phaseB0_report.py:196` use `fallbacks/steps`. Fallback calls last 1 step, others 5. Seed 0 at 5x: 373/573 = 65 % of calls, 373/1369 = 27 % of steps; at 4x 35 % vs 10 %. Prose at `phaseB0b_margins.md:85` says "calls": right.
- **W-3.** **"Walking" in T1 includes services.** Sure on the definition, medium on the size. `batching.py:47` adds services; `phaseB0b_margins.py:89` stores it as `walk`; `phaseB0b_margins.md:15` says "2.2 % of the walking". With about 3799 service steps of 14121 on medium, 2.25 % becomes at most 3.08 % on pure walking (computed; if shared shelves cut services, part of the saving is service and it is lower). Large 0.57 → 0.72, capacity 4 1.03 → 1.48.
- **W-4.** **Window arms change two things.** Sure. `phaseB0_margins.py:164` sets `execute = max(1, w // 4)`: 1 at w=5, 2 at w=10 (1015 and 498 calls, medium seed 0), baseline 5. Not stated in `phaseB0_margins.md:179`.
- **W-5.** **"Hindsight" wording.** Sure. Time-blind mean (`traffic.py:128`, `decayed=False`), layer 1 only (`phaseB0_margins.py:277`: no `planner_args`). The report does say "layer 1", but "oracle" and "ceiling" readings go further than the code.
- **W-6.** **Not every run is judged with an assertion.** Sure. Free-flow runs in `phaseB0b_margins.py:66-68` and `:176-177` read `makespan` with no check of `problems` or `delivered`; B0's `free_flow` (`phaseB0_margins.py:326-327`) and B0c's do check. Low impact.
- **W-7.** **`run_phaseB0.sh` cannot run.** Sure. It calls stage `online`, absent from `STAGES` and `ARMS` (`phaseB0_margins.py:61-64,449`), so argparse exits and the script prints FAILED. o1a, o1a_fifo, o1b appear only under the `ff` stage, which runs their free flow, not the arms. README says to run this script first. Results on disk were made by an earlier version. Reproducibility only.
- **W-8.** **R1 comment against measurement.** Sure. `phaseB0c_realism.py:47` says 2.5 % stops; measured 0.035–0.038 per move, slips 0.114–0.123 against p/(1−p) = 0.111 (`phaseB0c_realism.md:128-132`). Cause: draws happen on turn ticks and on blocked ticks (`exec.py:208,224`, before `:229`). So "the same p and q for all" is true per try, not per move.
- **W-9.** **"Kiva style" is a misnomer.** Sure. `env.py:1`, `phaseA_report.md:11`: robots stop beside fixed shelves and pick; Kiva carries pods to stations. Matters for literature comparisons, including rates from Kiva-style MAPF work.
- **W-10.** **CBS "time-outs" include "no solution".** Sure on code, unknown on frequency. `mapf.py:308,392-396`: `None` for either, then `timeouts += 1`. Medium spent 677–785 s against 450–570 s in full time-outs (three episodes), so most are real.
- **W-11.** **ms/step omits batching cost.** Sure. `phaseB0_report.py:78-82` reports `step_ms_mean`; `OFFLINE` lacks o3*. `setup_seconds` of o3c3 = 5.8 s / 42.1 s (computed) are recorded but unused. On large that is about 46 ms per step of makespan on top of the "13 ms" of Recommendation 3. T2 includes it online.
- **W-12.** **PIBT rule labels.** Medium. `mapf.py:150-158`: "far" / "near" add < 0.5 to a sum with eps ∈ [0, 1), so random tie-breaks swamp them, while the docstring says they "only break ties of age". The "best of 5" is mostly seed selection.
- **W-13.** **Verifier docstring lags.** Sure, low. `verify.py:9` "does not carry two orders at once" against `capacity` param; `verify.py:12` rotation is checked only between adjacent moves.
- **W-14.** **`open` counts fewer orders than "started".** Sure, low. `verify.py:102,124`: an order with no finished pick has no event and is not open; phaseA_report:53 calls them "started".
- **W-15.** **"Not at rest" in conflicts.** Sure, low. `sim.py:120` gives the home cell as target, so homing robots count in `conflicts.analyse` and `EdgeStats` despite `conflicts.py:3`. Small in E1; larger at low load, where no map was used.
- **W-16.** **Delay row of hindsight is chosen by makespan.** Sure, low. `phaseB0_margins.py:281` keeps the best-makespan run; `table_hindsight` reads its delay. Unstated in the report; B0c reports the difference properly.
- **W-17.** **Unverified, flagged only.** The +22 % rotation pilot (682 → 835, `phaseB0c_notes.md:21`): no result file located, only the note.

### 10.5 Statistics and infrastructure (S)

- **S-1.** **"8-core CPU" is wrong: the machine has 24 logical / 12 physical cores.** `tsp/docs/phase3_learned_triage_report.md:57` ("shared 8-core CPU"), `phase3_benchmark_report.md:37-38,98` ("8 cores"), against `lscpu` (24 CPUs, 6 cores per socket, 2 threads per core) and `env.sh:5`. `nproc` printed 8 because `OMP_NUM_THREADS=8` changes its output; with the variable unset it prints 24. Sure.
- **S-2.** **"9/9 instances" means instance × seed.** `tsp/docs/phase3_triage_report.md:15-16`; `tsp/data/tsp10000_test.npz` has shape (3, 10000, 2). Same in window, wsize, hier (10, 5, 3 instances shown as 30, 15, 9). Sure.
- **S-3.** **Same "±", different estimator.** `ddof=1` in `summarize.py:87`, `phase2_beam_report.py:72`; `ddof=0` in `phase2_k02n_report.py:70`, `phase3_benchmark_report.py:244,388`, `phase3_learned_triage_report.py:47,156`, `phase3_fast_triage_report.py:89,162`, `phase3_triage_wsize_report.py:56,193`, `phase5_vision_probe_report.py:40`. Sure.
- **S-4.** **The significance rule differs inside TSP.** Max p and sign per seed (`phase3_triage_report.py:297`) against test after averaging (`phase3_learned_triage_report.py:178`); the learned report does not say the rule changed. The strict/neutral difference is documented only in the docstring of `phase2_beam_report.py:78-81`. Sure.
- **S-5.** **"h of CPU" is wall clock.** `phaseB0c_report.py:179-181` sums `perf_counter` `elapsed`. Sure; low impact.
- **S-6.** **LKH `TIME_LIMIT` is CPU time, not wall.** `third_party/LKH-3/SRC/GetTime.c:1,12-14`, `FindTour.c:52`. `tsp/docs/phase3_benchmark_report.md:98` explains overruns only by the between-trials check; budgets come from wall measures. Sure about the timer; contribution to the overrun not measured.
- **S-7.** **Phase A uses a t-test, later phases Wilcoxon**, on the same design (`phaseA_report.py:113`, `phaseB0_report.py:66`). Sure.
- **S-8.** **"Within noise" has three thresholds** (1 %, 1-2 %, 3-7 %), while the measured half-widths have medians 1.1, 1.95, 3.4 and ranges up to 5.0, 6.3, 11.1. The B0 sentence "below about 1 %" fits the median, not the arms with ±5.0. Sure on the numbers (parsed from the three reports).
- **S-9.** **The 3 % rule is cited as "the rule of the plan"** (`phaseB0b_notes.md:13`) but no plan is in the repo. Sure that no file holds it; the reason is unknown.
- **S-10.** **Rule-level McNemar and instance-level headline in Phase 5.** `phase5_open_language_report.md:7` gives only the rule-level p. Recomputed at instance level from `hybrid.json`: p = 0.0012-0.0015, conclusion stands. Sure.
- **S-11.** **`phase3_fast_triage_report.md:7` "yes (not significant)"**: a report headline says yes on p = 0.19. Sure; wording.
- **S-12.** **CLAUDE.md "GPU ~7 GB livres"** is stale: T4 has 15360 MiB, about 9.5 GiB held by our own private LLM at audit time, 12.6-13.2 GiB seen held by another project. Sure.
- **S-13.** **GPU rule scope.** CLAUDE.md limits "cada treino" to < 6 GB; LLM runs hold about 9.5 GiB under a different rule (`GpuGuard.min_free = 2048`). Sure on the numbers; whether it is a violation depends on the reading.
- **S-14.** **Rule "Trabalhar SÓ dentro desta pasta" vs `/scratch/pcanaste/tools`** (`ollama_pc.sh:9`). Sure; likely authorized, but not written in CLAUDE.md.
- **S-15.** **`docs/phaseC_notes.md` missing**, so `phaseC_report.py:300` raises until written. Expected at this stage (phase running), not a defect.
- **S-16.** **`phaseB0b_notes.md:33` "all within noise or close to it"** next to +1.0 ± 0.9 (5/5 seeds same sign, interval excludes zero). Minor; the "or close to it" covers it.

## (a) The ten assumptions to tell the professor

Ordered by how much they change what the project can claim.

1. **The window model was never trained for the task it does.** It is trained on closed tours of uniform points, then used on open paths with fixed ends and gets no marker for those ends (`hier.py:Heatmaps`). After rescaling, windows are not at the training scale either: edges are 0.66× and the short side 0.74× those of uniform TSP-20. Phase 3 rests on this, and no report mentions it. It also limits "training at the window size does not help".
2. **The network's own contribution to the gains has not been measured.** Windows accept only shorter tours, and beam keeps the shortest of B tours, so part of the gain is selection (beam 0.49 pp against greedy 0.15 pp from the same heatmap at n = 1000, w = 20, one seed). No control uses a distance-only heatmap. The 2-opt control gains exactly 0 by construction, so it cannot show this.
3. **The triage protocol solves the same windows again.** At Fu-10k, "LKH everywhere" makes 3000 calls per instance and about 183 improve (6.1 %). LKH is deterministic there, so a memo of unchanged windows should give the same tour with about half the calls (estimate, not measured). Every equal-time anchor inherits this. Selection also has no memory, while random redraws: it is a coverage heuristic, so "random ties at equal time" is not yet a verdict on the signal.
4. **k = 0.5n confounds graph density with distance scale.** It restores the training mean edge (ratio 0.98 against 0.59 at n = 50), so the batch 2 conclusion that scale is not the main cause does not follow. The normalization was also only half tested: it changes `edge_attr` but not the coordinates, it is nearly a no-op with k = 0.5n, and `sqrt_n` was never run.
5. **"The GNN cannot" means this small budget, and seeds do not fix results.** The budget is 12k instances, 20 epochs, hidden size 64, about 50k parameters. Phase 3 uses `best.pt`, which the TAGCN report itself calls noise: selected epochs are 12, 1, 4 for the main model and 1, 1, 10 and 5, 3, 1 for the window-size models. Same-seed reruns moved TSP-50 greedy from 13.91 to 19.93 %, and every model claim has 3 seeds.
6. **Equal-time verdicts are smaller than the measurement noise.** Times are one wall-clock run per point on a shared, unpinned 24-thread machine, with no intervals. The same policy measured 14.8–19.4 s, and the decisive differences are 5–10 %. Process counts changed between reports (4, 6, 7, 8, 1), GPU time counts as CPU time, `TIME_LIMIT` is CPU time while budgets are wall time, and the 7 ms per LKH call is an ATSP solve on doubled nodes. "LKH-3 wins at equal time" is safe. Rankings among our policies are not.
7. **"No margin" in the warehouse means "not detected", and the oracles are not ceilings.** The 3 % bar is applied to point estimates while upper bounds reach 5–9 %, and it has no written source. With 5 seeds p cannot go below 0.0625. The rival is the best arm on the same seeds, which biases margins down. Hindsight is a time-blind mean per edge, and in B0 and B0b it reaches layer 1 only. T2 at x2.0 is not saturated for batched arms, so batched capacity was never measured.
8. **The warehouse world is unsourced, and it omits what a model could exploit.** Density is 1.9–2.3 robots per 100 free cells, aisles are 1 cell wide with the picker blocking them, service is 3 s, orders have 3–12 lines, capacity counts orders (a trip can hold 36 products), robots follow at full speed, and the delay functions are invented. Products are uniform and independent, with no ABC skew or co-purchase, and arrivals have no waves. "CBS gains nothing", "congestion is modest" and the small margins are statements about this world.
9. **The language results rest on weak ground truth and a weak metric.** L2 labels are wrong in 28 % (TSP) and 35 % (warehouse) of audited texts, largely because of devices the prompt asks for. With no translation at all, 22–28 % of routes are already "valid" on L1 and L2, and the report has no such baseline. The parser is 100 % on L1 by construction and fell from 80 % on dev to 55 % on test on L2. In TSP the LLM prompt has no selector example, and thinking was turned off for budget, against the only evidence. The closed-world legend also makes "every hospital" easier than real text.
10. **TSP instances are synthetic, and "optimal" is a heuristic reference.** Nothing outside uniform points in the unit square was tested, and raw coordinates with no augmentation leave the symmetry unused. Exactness was checked only at n = 7. E0 is not independent evidence: same algorithm and seed, 19 of 20 tours identical, and n = 200–5000 were never rechecked. For gaps of about 4 % this is minor. For ratios like "28–110×" over gaps of 0.08–0.15 % it matters.

## (b) Assumptions that could invalidate a conclusion

Verdicts: **does not follow** (the evidence does not support the sentence), **weakened** (true only under a narrower condition), **wording** (a stated fact is wrong, the conclusion stands), **survives** (checked and holds).

Report names are in `tsp/docs/` or `warehouse/docs/`.

| Assumption | Report and conclusion | Mechanism | Verdict |
|---|---|---|---|
| k and distance scale are separable | `phase2_batch2_report.md`: the 0.58× shift "is not the main cause of the failure" | k = 0.5n restores the training mean edge, so its win may be a scale fix; `knn_mean` left `x` untouched | Does not follow |
| `knn_mean` is a valid test on top of k = 0.5n | `phase2_batch2_report.md`: "On top of kfrac it adds nothing" | At k = 0.5n the mean is nearly constant in n, so the division is almost constant | Does not follow |
| An automatic rule decides the summary | `phase2_batch2_report.md` Summary | The "≥ 75 %" rule was overridden by hand after reading the results; regenerating overwrites it (S section 8.5) | Weakened |
| The ablation covers the question | `phase2_k02n_report.md`: 0.5n is better or equal, never worse | Test time only, on k = 10 models; at n = 1000 "our" k = 100 is below 0.2n = 200 | Weakened |
| Greedy baselines are the fair reference for beam | `phase2_beam_report.md`: B = 100 + 2-opt beats both baselines | Baselines have no beam or length selection; a distance-only beam is the fair one | Weakened |
| A 3-seed mean of selected epochs is stable | `tagcn_report.md`: TAGCN `best.pt` has the lowest greedy gap in every TSP-50/100 column | Same-seed reruns change the epoch and the gap; part of the win comes from early checkpoints | Weakened |
| Windows are at the training scale | `phase3_window_report.md`: "at the scale it was trained on" | Measured: edge 0.66×, short side 0.74× | Wording, and a likely cause of weak capture |
| The 2-opt control isolates the model | `phase3_window_report.md`: every model gain is a re-ordering 2-opt cannot find | True, but it does not show the network is the source | Weakened |
| LKH is exact on windows | `phase3_window_report.md`: "the exact optimum of each window" | `runs=1` heuristic; LKH was longer than the current path in 0.03–0.05 % of windows | Wording |
| `best.pt` is a sound model | Phase 3 window, triage and benchmark reports: gaps as means over 3 seeds | Seed 1 of the main model is epoch 1 | Weakened |
| Training at w is a fair test | `phase3_wsize_report.md`: training at the window size does not help; "not a weaker model" | Epochs 1, 1, 10 and 5, 3, 1, closed-tour labels, no end marker, val gap 9.5–13.5 %; whole-instance 2-opt hides model quality | Does not follow |
| The model is at its best and the LKH cost is right | `phase3_wsize_report.md`: no w where the model beats LKH in gain per second | Depends on this model and on the ATSP path. LKH would likely still win, by an unknown margin | Weakened |
| "LKH everywhere" is an efficient anchor | `phase3_triage_wsize_report.md`: w = 50 + learned is 0.19 pp better and 27 % faster than LKH on every w = 20 window | About half of the anchor's calls are redundant. "0.19 pp better" stands; "faster" may flip (estimate) | Weakened |
| Ranked policies are fairly compared with random | `phase3_fast_triage_report.md`: tie at equal time on Fu-10k; `phase3_learned_triage_report.md`: random at 50 % wins | No memory penalises ranked scores; one timed run per policy; the same policy measures 14.8–19.4 s | Does not follow |
| The explanation of w = 50 is right | `phase3_triage_wsize_report.md`: p tells which windows improve, not which gain most | Untested; improved counts (78 against 111 of 240, seed 0) fit re-selection at least as well | Does not follow |
| Candidates tested the network fairly | `phase3_benchmark_report.md`: the network does not help LKH | Whole-instance heatmap, one seed, p saturated at 0.9–0.99 | Weakened |
| Network cost is correctly attributed | `phase3_benchmark_report.md`: network 0.4 / 4 s, costs more than it saves | Mostly Python graph building; superseded later, with no note in that report | Does not follow |
| E0 is a safe reference for ratios | `phase3_benchmark_report.md`: LKH-3 beats us 28–110× | The win is safe. The ratio divides by very small gaps: δ = 0.05 pp turns 28× into 21× and 47× into 30× | Weakened |
| E0 validates old references | `phase3_benchmark_report.md`: earlier reports stay valid | Conclusion likely right, reasoning weak; n = 200, 500, 2000, 5000 unchecked | Weakened |
| Counts are instances | `phase3_triage_report.md`: "9/9 instances at n = 10000" | 3 instances × 3 seeds; same in window, wsize and hier reports | Wording |
| Rules are independent | `phase5_open_language_report.md`: q14 beats p1 at n = 200 (p = 0.0003) | Clustered by text; instance-level p ≈ 0.0015 | **Survives**, with a weaker p |
| The 54 McNemar tests stand alone | `phase5_open_language_report.md`: h1 below q14 (p = 0.031) | Fails any correction over the report's 54 tests | Weakened |
| Hybrid trigger set before test scores | `phase5_open_language_report.md`: fixed on dev before the test texts were scored | Test scores committed 05:58, hybrid code 09:07, no dev evaluation in the repo. If post hoc, "64.2 % against 54.7 %" is optimistic | Weakened (medium confidence) |
| L2 accuracy bounds every arm from below | `phase5_open_language_report.md`: "a lower bound for every arm" | False for p1 where its filter reproduces a label the text contradicts | Wording, with an effect on the gap |
| "Valid route" means translation worked | `phase5_open_language_report.md`: q14 80 % against 63 % | No-translation baseline is 28 %; soft and non-binding rules are ignored; labels 28 % wrong | Weakened |
| The filters are neutral | `phase5_text_rules_llm_report.md`: a keyword parser is enough | Filters require the parser's own words, and it was written after reading the texts; L2 shows 55–60 % once filters go | Weakened (partly admitted) |
| The probe has 40 trials | `planner_probe_v2.md` in `tsp/results/phase5/`: rule 40/40, LLM 35/40 | Effective n = 8, case 7 rewards an action that does not fit, and the rule comes from the checks' own table | Does not follow |
| The reference optimises the stated objective | Step 1 data and cost gaps, 33 of 400 instances | Soft penalty counts as travel time in TSPTW (about 22 %). Routes stay feasible; the windows are tighter than the text | Weakened (effect unmeasured) |
| x2.0 saturates batched arms | `phaseB0b_margins.md`: "+0.8 %" under saturation; "the queue already offers partners" | Throughput is capped by arrivals and backlog is flat and near work in progress, so the queue is nearly empty, which contradicts the explanation | **Does not follow** |
| "Walking" is pure walking | `phaseB0b_margins.md`: below 3 % in every case, 2.2 % of the walking | The cost includes services; on walking alone it reaches at most 3.08 %, on the threshold | Weakened |
| Point estimate against 3 % | `phaseB0b_margins.md`: "None clears 3 %" | Upper bounds reach 4.8–9.3 %, and picking the rival biases margins down | Weakened: "not detected", not "none" |
| Hindsight is a ceiling | `phaseB0_margins.md`: the room for a GNN predicting travel times; `phaseB0b_margins.md`: no level of congestion makes better distances worth 3 % | Static per-edge mean, layer 1 only; it says nothing about a time-varying or live predictor, or about the planner | Weakened |
| Window arms change only the window | `phaseB0_margins.md`: window 10 ties with 20, window 5 loses about 2 % | `execute` also changes (2 and 1 against 5) | Does not follow |
| Capacity in orders is physical | `phaseB0_margins.md`: +33–37 % with capacity 3 | Up to 36 products per trip; the size of the gain depends on it | Weakened |
| CBS is a comparable exact method | `phaseA_report.md`: CBS gains 4.2 % and nothing on medium | Window 8 against 20, next goal only, 9–62 % PIBT, wall-clock limits that depend on load | Weakened |
| Conflicts are structural | `phaseA_report.md` conclusion 2 and E3 | Measured under uniform picks and about 2 % density; skew would raise concentration | Weakened |
| R2 is representative | `phaseB0c_realism.md`: R2 is not clearly above R1 | Invented functions, one intensity, intervals wider than the effect; R1 is also mildly state-dependent | Weakened (partly admitted) |
| Rotation falls on every assignment alike | `phaseB0c_realism.md` | One seed, rotation-blind planners; the pilot has no result file | Weakened |
| All episodes reach table C | Phase C (no report yet) | The `seen` bug drops repeated rule sets, including the worst case | Would invalidate table C if not fixed |

## (c) Assumptions that a cheap ablation can test

The first table is the recommended order. The rest is the full list by area. Costs are estimates; nothing here was run.

| Order | Test | Why first | Cost |
|---|---|---|---|
| 1 | Statistics only: upper bound against 3 %, rival chosen on other seeds, instance-level McNemar | No reruns, and it changes the wording of several conclusions | Minutes to 30 min |
| 2 | Pure-route baseline row, and T1 with service 0 | Stored data; one row to add to phase 5, up to 30 min for T1 | Seconds to 30 min |
| 3 | Distance-only heatmap in windows and beam | Measures the network's own contribution (list (a), item 2) | Minutes, CPU |
| 4 | Memo of unchanged windows in triage; the P1 gap must match exactly | Resets every equal-time anchor (item 3) | About 15 lines, 2 min for P1, about 30 min for all policies |
| 5 | Phase 1 weights, k = 10, both `x` and `edge_attr` multiplied by √(n/20) | Tests the scale hypothesis without retraining (item 4) | Minutes on GPU |
| 6 | Windows with `last.pt` | Checks checkpoint dependence (item 5) | Minutes |
| 7 | T2 arm (a) at x3.0 and x4.0 | Gives the batched capacity (item 7) | 20 min CPU |
| 8 | Time windows as a symmetric TSP in elkai | Corrects the LKH cost of a window | Seconds to 1 min |
| 9 | o4w5 and o4w10 with `execute` = 5 | Removes a confounded comparison | Under 5 min per size |
| 10 | Uniform to Zipf products, and aisle width 2 | First tests of the world assumptions (item 8) | 10–15 min each |

### TSP data and model

| Assumption tested | Change | Rough cost | Reuse |
|---|---|---|---|
| Network adds value in windows | In `make_solver`, replace heatmap with distance-only pseudo-probabilities (e.g. exp(−d/τ) on the same k-NN) | Minutes, CPU only (no GPU): ≈ 1.4 s per 1k instance | `window_eval.py`, tsp1000/10000 test, initial tours |
| Network adds value in beam | Same pseudo-heatmap in `beam_search_decode` on TSP-50/100 | Minutes, CPU | `beam_eval.py`, phase 1 test sets |
| Path task | Train TAGCN K=3 on windows from greedy+2-opt tours with `lkh_path` labels and an s/t node flag | 12k windows at 7 ms ≈ 1.5 min on 8 procs; training ≈ TSP-20 run | `lkh_log`, `train_model.py` needs a 3-dim node input |
| Checkpoint selection | Run windows with `tagcn_k3_ckpt_seed*/last.pt` | Minutes; 1-line `--checkpoint` in `window_eval.py` | Already trained reruns |
| Scale hypothesis, no retraining | Phase 1 weights, k = 10, multiply both `x` and `edge_attr` by √(n/20) | Minutes on GPU | `evaluate.py` + 2 lines; TSP-50/100 test |
| `sqrt_n` | Train with `--edge-norm sqrt_n` | One TSP-20 run per model, seeds 0–2 | Flag exists |
| Confounding k/scale | k = 25 on TSP-50 with edges rescaled to TSP-50 k=10 mean | Evaluation only | `evaluate.py` |
| BN + dense graph | `--bn-batch-stats --k-frac 0.5` together | Minutes | `evaluate.py`, already supports both |
| Augmentation at test | 8 dihedral copies of each window, take the shortest | 8× network time (network < 1 % of cost, but 8× decode) | `Heatmaps`, TAGCN best.pt |
| Window shape | Add a uniform-in-square rescale variant, or centre the short axis | Minutes | `hier.rescale` |
| Reference independence | Run LKH with SEED=2 on 5 tsp1000 and 50 TSP-20 instances, or an exact MILP at n = 20 with HiGHS | ~6 min and minutes | `lkh3.solve`; HiGHS already in `milp/` |
| References at n = 200–5000 | E0 RUNS=10 on tsp2000 and tsp5000 | ~1 h at 8 procs (inferred from 2000: 43 s/1 run) | `lkh3_reference.py` |
| 2-opt list size | `two_opt_neighbors` with k = 10 vs 100 | Minutes, CPU | `initial_tour` |
| Stronger baseline | Add Or-opt after 2-opt, then rerun windows | ~1 h coding, minutes to run | `initial_tour`, tsp1000 |
| Beam on all B tours | 2-opt each of B tours, pick shortest | TSP-50: B × 2-opt time | `beam_eval.py` |
| Calibration | Temperature on logits before beam (T = 0.5, 2) | Minutes | `Heatmaps` |

### Triage and LKH-3

| Change | Cost | Reuses |
|---|---|---|
| Memo in `ranked_refine` / `triage_refine`: skip LKH if (start, cities) already solved unchanged; for ranked policies either count or exclude them | ~15 lines. P1 w = 20, Fu-10k 8 instances: ~2 min; all policies ~30 min one process | models, data, same seeds; P1 gap must match exactly as sanity test |
| `--max-passes 1` and 2 with `--score random --fracs 100` | 1-2 min | gives gain per pass, missing today |
| Compute "gain captured at top x %" on first-pass windows for each signal | seconds, CPU | results/phase3_learned_triage/features/*.npz |
| Choose logreg vs hgb_reg by gain captured; extend C to 1e-4, 1e-3; drop `position` | minutes, sklearn CPU | same features |
| Train with passes 2-3 windows (log windows during a run) | ~1 h with LKH logging | existing policy code |
| Solve windows as symmetric TSP through `_elkai.solve_problem` and time them | < 1 min, 1000 windows | window logs give coordinates and expected lengths |
| `CUDA_VISIBLE_DEVICES=""` scoring time | 5 min on 8 Fu-10k instances | fast_triage_check.py |
| Repeat seq comparison 3 × interleaved random / learned to get a time CI | ~1 h | same scripts |
| E1 QUADRANT 5 with SEED 1, 2, 3 at one budget | 16 × 21 s / 8 × 3 ≈ 2 min | lkh3_equal_time.py |
| Heat candidates from window-level p merged over the tour (or rescaled subgraphs) | ~10 min plus E1 cell | fast_triage.batch_heat |
| LKH window with RUNS 3 at w = 50, compare gain | ~30 min | wsize logs |
| rules.LKH_SCALE = 1e5 on 20 instances, compare true objective | ~20 min | data/phase5 |

### Text rules and LLM

| Test | Change | Cost | Reuses |
|---|---|---|---|
| Pure baseline in part C | check_all(label, pure) per instance | seconds (already computed: L0 26.7/16.7, L1 = L2 28.3/21.7 % at n=200/1000) | text_v3_open, npz |
| Cluster bootstrap by text | resample instances for q14-p1 | seconds | formalize_tsp*.jsonl |
| b0+ on unfiltered text_v2 | run parse_keywords | seconds, plus a label check by hand | text_rules_v2_tsp*.jsonl |
| Rescore step-3 verifier with one JSON rule; rescore case 7 | report code only | seconds | step3_verify, planner calls |
| p1 on dev vs test (warehouse) | phaseC_dev.py on dev only, compare | 1 min | l2.jsonl |
| L1-style examples for q14 (TSP) | 3 examples with selectors, 60 L1 + 60 L2 | about 40 min | same instances, p1 logs |
| L2 without double negation / implicit rule | remove the line, 60 texts, audit | 20 min GPU + audit | L1 rules |
| Second reader on the audits | independent read, kappa | 1 h | l2_audit.json |
| Planner with real variance | 20 new cases or T > 0 | 10-20 min | probes |
| Hybrid trigger on dev | coverage() on 20 dev texts + 20 q14 calls | 10 min | l2_tsp200.jsonl |
| TSPTW + soft | solve the 33 instances without soft in the matrix, compare feasibility and objective | 10-30 min with 8 workers | text_rules_tsp*.npz |
| Validator ablation | send rejected rules to LKH-3 anyway | about 30 min (about 80 rejected rules in the logs) | pipeline logs |

### Warehouse

| Assumption | Change | Rough cost | Data reusable |
|---|---|---|---|
| Batched capacity at x2.0 | T2 arm `a` at x3.0, x4.0, medium and large | 20 min CPU | Same seeds, `capacity_rate` |
| Walking vs service in T1 | Recompute `TripCost` with service 0 on saved groups (groups not saved today: rerun `t1` partition from seeds, or compute on C&W only) | Minutes to 30 min | `t1_*.json` for totals |
| window vs execute | o4w10 with execute 5 and o4w5 with execute 5 | Under 5 min per size | `base` |
| Aisle width | `corridor=2` with same orders; test already exists | 10 min | `test_env.py:18`, needs larger grid or fewer blocks |
| Order capacity in products | Add a cap such as ≤ 12 or 20 products to `savings` | 10 min | `o3c3` as reference |
| SPT on batches | `TripHungarian` on o3c3 trips, no rotation | 5 min | `t1` groups; B0c shows the code |
| Uniform to ABC | Zipf in `Catalogue.shelf_of` | 15 min for base + o3c3 + E3 | E3 scripts |
| Co-purchase | Draw some orders from product clusters | 15 min | Same |
| Station chosen nearest | `make_order` station = nearest last shelf | 10 min | Same |
| Parking near stations | `home` = cell near the last station at ρ 0.68 | 10 min | `t2 one`, E2 |
| Full parking at 4x | `parking_rows=3` at 4x | 25 min | `t3` |
| Follow gap | In `exec.clear`, forbid entry in the leaving step | 15 min PIBT, 30 min prioritized | B0c arm (a) |
| Service time | `service=10` | 10 min | `base`, `o3c3` |
| CBS same window | CBS 20 / 5, or prioritized 8 / 3 | Prioritized 8 / 3: 2 min | `base`, E1 CBS |
| Second intensity | p = 0.2, q = 0.005 on m30, arms a, b, e | 30 min | Trips already fixed per seed |
| Time-varying hindsight | Edge cost per 100-step bin in the planner (needs code) | 2 h | None: `edges_*` have no time bins, reuse `targets` in E1 `.npz` |

### Statistics and infrastructure

| Assumption | Change | Cost | Reuse |
|---|---|---|---|
| 3 % as point estimate | Report the t upper bound in `paired`; add one column | Minutes, no reruns | All B0-B0c JSON |
| Noise of 5 seeds, unbiased rival | Rival chosen on 2 seeds, gain measured on 3 (or leave-one-out) | Minutes, no reruns | B0, B0b, B0c JSON |
| Power | Compute seeds needed for 3 % from the observed s (s ≈ 2.7 at half 3.4; half 1.5 needs about 15-20 seeds) | Minutes | B0c JSON |
| Rule level vs instance level | Instance-level McNemar and cluster bootstrap for all 54 tests (done here for one) | 30 min | `hybrid.json`, `formalize_*.jsonl` |
| Averaging before test | Rerun Phase 3 learned/fast/wsize tests with max p per seed | Under 1 h, no reruns | `per_inst` arrays in `results/phase3_*` |
| Strict vs neutral sign | Regenerate exp1/batch2/datasize with `beam.paired` | Under 1 h | Existing npz |
| ddof | Use `ddof=1` in the 6 generators | Minutes | None |
| Same-seed noise floor | Use Table 8 and Check 1 reruns as a threshold in Phase 2 tables | Under 1 h | `*_ckpt` runs |
| Timing noise | Repeat `seq` Fu-10k random and learned 5× each, alone, with `process_time` beside wall and a 30 s load sampler | About 1-2 h CPU, one process | Same 8 instances |
| CBS wall-clock dependence | Rerun `o4cbs` on one medium seed with an expansion limit instead of a time limit, or twice under different load | 1-2 h, one process | B0 baseline |
| Thread/core effect | `taskset` the 8-process run to 8 physical cores on one instance set | About 30 min | Fu-1k subset |
