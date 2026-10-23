# MILP: fixed-charge network flow

Fixed-charge network flow: classical MILP solver labels small instances, a GNN learns which arcs are open in the optimal solution, the heatmap guides the solver on larger ones.

## Layout

```
milp/
├── src/fcnf_gnn/      solver.py (HiGHS MILP), generate.py (random instances), data.py (.npz format),
│                      graph.py (PyG graphs), models.py (GCN / TAGCN arc classifier), train.py, utils.py
├── scripts/           make_dataset.py, data_stats.py, reference_rerun.py, train_model.py, run_phase1_*.sh
├── tests/             pytest
├── docs/              fcnf_formulation.md (the MILP), reports
├── data/              labelled instances (not in git)
└── results/           logs, statistics, runs, reports (not in git)
```

`fcnf_gnn` does not import `tsp_gnn`: code reused from the TSP project is copied and adapted.
Commands run from `milp/` after `source ../env.sh` (solver: `highspy` 1.15.1, installed in `.venv`).

## Phase 1, data

```bash
source ../env.sh
pytest -q tests/
tmux new-session -d -s fcnfdata "nohup bash scripts/run_phase1_data.sh > results/phase1_data.nohup 2>&1"
```

Sets: fcnf20 train / val / test (12000 / 1000 / 1000, seeds 31 / 32 / 33, solved to optimality),
fcnf50 and fcnf100 test (500 / 200, seeds 34 / 35, 120 s limit). Statistics: `results/phase1_fcnf_data_stats.md`.

## Phase 1, training

```bash
tmux new-session -d -s fcnftrain "nohup bash scripts/run_phase1_train.sh > results/phase1_train.nohup 2>&1"
```

GCN and TAGCN K=3 (64 channels, 3 layers, residual + LayerNorm), 3 seeds each, best epoch by validation
F1 at p >= 0.5 -> `results/runs/<model>_seed<s>/`. In parallel, the unproven fcnf100 test instances are
re-solved with 600 s -> `data/fcnf100_test_ref600.npz` (the reference cost is the best of the two runs).

## Phase 1, decoding and report

```bash
tmux new-session -d -s fcnfdecode "nohup bash scripts/run_phase1_decode.sh > results/phase1_decode.nohup 2>&1"
```

Pruning, warm start, delegation and the baselines on fcnf20 / fcnf50 / fcnf100 -> `results/phase1_decode/`;
report `docs/phase1_fcnf_report.md` (`python scripts/phase1_fcnf_report.py`).

## Phase 1b: equal-time baseline, trust region, feasible warm start

```bash
tmux new-session -d -s fcnf1b "nohup bash scripts/run_phase1b.sh small > results/phase1b_small.nohup 2>&1"
# after "ALL DONE small" in results/phase1b.log:
tmux new-session -d -s fcnf1b100 "nohup bash scripts/run_phase1b_fcnf100.sh > results/phase1b_fcnf100.nohup 2>&1"
python scripts/phase1b_fcnf_report.py
```

Full-MILP improving-solution trajectories (`scripts/milp_trajectory.py`, also with a slope scaling or
trust-region warm start), the trust-region decoder (`decode_eval.py --group trust`) and runs with the time
limit itself (`scripts/equal_time_check.py`) -> `results/phase1b/`; report `docs/phase1b_fcnf_report.md`.
