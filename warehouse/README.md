# Warehouse with many robots

Order picking plus multi-agent path finding on a grid. Phase A is a simulator with classic baselines and a verifier in code; there is no learning yet. Phase B will add a GNN that predicts where conflicts happen, and phase C an LLM that translates requests into rules.

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
| `scripts/phaseA_baselines.py` | experiments, with resume |
| `scripts/phaseA_report.py` | builds `docs/phaseA_report.md` and its figures |
| `docs/phaseA_notes.md` | the hand-written parts of the report |

Results go to `results/warehouse/phaseA/`, which is outside git. Figures are in `docs/figures/`.

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
```

A stopped command resumes: finished runs are skipped.

## Dependencies

The project's `.venv` is enough, with numpy, scipy and matplotlib, and needs only the CPU. The LKH-3 binary comes from `tsp/third_party/LKH-3`. `pogema` is not used: it asks for numpy 1.26 or older and pydantic 1.9 or older, so the grid is written by hand.
