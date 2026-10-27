"""E1 preparation (phase 3 benchmark): LKH-3 preprocessing cost of cheaper candidate-set variants.

For each variant, with and without INITIAL_TOUR_FILE = our greedy edge + 2-opt tour (window.initial_tour,
k = min(0.5 n, 100), as in the triage experiment), LKH runs with RUNS = 1 and MAX_TRIALS = 1: one
Lin-Kernighan trial from the starting tour. fixed_seconds = wall − time of that trial is the cost paid before
any search (writing the files, reading the problem, ascent, candidate set, initial tour); one_trial_gap_pct
is the gap after that single trial. (TIME_LIMIT = 0 is not usable: LKH-3.0.6 then records no tour and
crashes with a segmentation fault.)
Variants:
    alpha              CANDIDATE_SET_TYPE = ALPHA (default; INITIAL_PERIOD = n / 2)
    alpha_p<k>         ALPHA with INITIAL_PERIOD = k (shorter subgradient ascent)
    popmusic           CANDIDATE_SET_TYPE = POPMUSIC
    nn5, nn8           CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR, MAX_CANDIDATES = 5 / 8
The time of our greedy + 2-opt tour is measured too: a variant with INITIAL_TOUR_FILE must pay it as well.

Example (1 process, to leave the CPU to E0):
    python scripts/lkh3_e1_prep.py --data data/tsp1000_test.npz --index 0 --out results/phase3_benchmark/e1_prep/tsp1000.json
"""

import argparse
import json
import os
import re
import time

import numpy as np

from tsp_gnn import lkh3
from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.window import initial_tour


def variants(n):
    """{name: LKH parameters} of the candidate-set variants for n cities."""
    out = {"alpha": {}}
    for period in (n // 10, n // 100):
        out[f"alpha_p{period}"] = {"INITIAL_PERIOD": period}
    out["popmusic"] = {"CANDIDATE_SET_TYPE": "POPMUSIC"}
    out["nn5"] = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 5}
    out["nn8"] = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 8}
    return out


def main():
    parser = argparse.ArgumentParser(description="E1 preparation: LKH-3 preprocessing per variant.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists: not overwritten")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    workdir = os.path.splitext(args.out)[0] + "_work"

    d = load_dataset(args.data)
    coords = d["coords"][args.index].astype(np.float32).astype(np.float64)   # as in triage_eval
    n, stored = len(coords), float(d["lengths"][args.index])
    k = min(n - 1, round(0.5 * n), 100)
    start = time.perf_counter()
    t0 = initial_tour(coords, k)
    init_seconds = time.perf_counter() - start
    init_gap = 100 * (tour_length(coords, t0) / stored - 1)
    print(f"n = {n}: greedy + 2-opt {init_seconds:.2f} s, gap {init_gap:.2f} % (vs stored tour)", flush=True)

    records = {"data": args.data, "index": args.index, "n": n, "greedy_2opt_seconds": init_seconds,
               "greedy_2opt_gap_pct": init_gap, "variants": []}
    for name, params in variants(n).items():
        for with_init in (False, True):
            r = lkh3.solve(coords, workdir, {"RUNS": 1, "MAX_TRIALS": 1, **params},
                           initial_tour=t0 if with_init else None)
            ascent = re.search(r"Ascent time = ([\d.]+) sec", r["log"])
            rec = {"variant": name, "params": params, "initial_tour_file": with_init,
                   "wall_seconds": r["wall_seconds"], "preprocessing_seconds": r["preprocessing_seconds"],
                   "ascent_seconds": float(ascent.group(1)) if ascent else 0.0,
                   "trial_seconds": r["search_seconds"],
                   "fixed_seconds": r["wall_seconds"] - r["search_seconds"],
                   "one_trial_gap_pct": 100 * (r["length"] / stored - 1)}
            records["variants"].append(rec)
            print(f"  {name:<12} init={'ours' if with_init else 'LKH ':<4}  wall {rec['wall_seconds']:7.2f} s  "
                  f"preprocessing {rec['preprocessing_seconds']:7.2f} s  ascent {rec['ascent_seconds']:7.2f} s  "
                  f"1 trial {rec['trial_seconds']:6.2f} s  fixed {rec['fixed_seconds']:7.2f} s  "
                  f"gap after 1 trial {rec['one_trial_gap_pct']:6.2f} %", flush=True)
            with open(args.out, "w") as f:      # rewritten after each variant: partial results survive
                json.dump(records, f, indent=1)
    print(f"saved to: {args.out}")


if __name__ == "__main__":
    main()
