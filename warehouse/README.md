# Warehouse with many robots

Order picking plus multi-agent path finding on a grid. Phase A is a simulator with classic baselines and a verifier in code; phase B0 measures the margin of each component before training anything. There is no learning yet. Phase B will add a GNN that predicts where conflicts happen, and phase C an LLM that translates requests into rules.

## Layout

| path | what it holds |
|---|---|
| `src/warehouse/env.py` | grid, catalogue, orders |
| `src/warehouse/sequencing.py` | layer 1: visit order (nearest neighbour, 2-opt, LKH-3) and the assignment rule |
| `src/warehouse/mapf.py` | layer 2: PIBT, prioritized planning, CBS |
| `src/warehouse/sim.py` | the simulator |
| `src/warehouse/verify.py` | the verifier, which only reads the saved trajectory and events |
| `src/warehouse/conflicts.py` | waits, detours and interactions, read from the trajectory |
| `src/warehouse/viz.py` | heat maps and the GIF |
| `src/warehouse/dispatch.py` | which robot takes which order: Hungarian, reassignment, costs with traffic, fixed plans |
| `src/warehouse/batching.py` | which orders share a trip: Clarke & Wright, regions, exact pairs, LKH-3 in CVRP mode |
| `src/warehouse/insertion.py` | MCA with real paths, for the LNS oracle |
| `src/warehouse/traffic.py` | Dijkstra with edge weights, edge statistics, highways |
| `scripts/phaseA_baselines.py` | experiments, with resume |
| `scripts/phaseA_report.py` | builds `docs/phaseA_report.md` and its figures |
| `docs/phaseA_notes.md` | the hand-written parts of the report |
| `scripts/phaseB0_margins.py`, `scripts/phaseB0_report.py` | experiments of phase B0, and `docs/phaseB0_margins.md` with its figure |
| `docs/phaseB0_notes.md` | the hand-written parts of that report |

Results go to `results/warehouse/phaseA/` and `results/warehouse/phaseB0/`, which is outside git. Figures are in `docs/figures/`.

## Commands

Run from `warehouse/`, after `source ../env.sh`:

```bash
pytest -q tests                                  # or, from the repository root: pytest -q warehouse/tests

scripts/run_phaseA.sh                            # every experiment, at most 4 processes, nice -n 10; use tmux
python scripts/phaseA_baselines.py seq           # layer 1: gap and time against LKH-3
python scripts/phaseA_baselines.py e1            # offline lots
python scripts/phaseA_baselines.py e2            # lifelong; its rates come from E1
python scripts/phaseA_baselines.py e3            # conflicts of the E1 episodes
python scripts/phaseA_baselines.py gif           # short GIF of a small episode

python scripts/phaseA_report.py                  # report and figures

scripts/run_phaseB0_revised.sh                   # phase B0: O3, O1, O2, free flow, O4, CBS last; run scripts/run_phaseB0.sh first
python scripts/phaseB0_margins.py o3x            # one arm or stage; see the top of the script
python scripts/phaseB0_margins.py ggo_base       # the baseline on the two seeds of GGO-lite
python scripts/phaseB0_report.py                 # report and figure
```

A stopped command resumes: finished runs are skipped.

## Dependencies

The project's `.venv` is enough, with numpy, scipy, matplotlib, networkx and cma, and needs only the CPU. The LKH-3 binary comes from `tsp/third_party/LKH-3`. `pogema` is not used: it asks for numpy 1.26 or older and pydantic 1.9 or older, so the grid is written by hand.
