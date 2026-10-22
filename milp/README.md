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
