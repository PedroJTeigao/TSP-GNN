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
    alpha_nosub        ALPHA with SUBGRADIENT = NO (alpha-nearness on the plain minimum 1-tree, no ascent)
    quadrant5, 8       CANDIDATE_SET_TYPE = QUADRANT, MAX_CANDIDATES = 5 / 8
    heat5, heat8       the 5 / 8 edges of highest p of the network for each city (heatcand.py), read from a
                       CANDIDATE_FILE with CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR (no ascent); the k-NN graph
                       and the network (TAGCN K=3 seed 0, best.pt, k = min(0.5 n, 100)) are part of the fixed
                       cost (net_seconds); the CUDA warm-up is not
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

from tsp_gnn import heatcand, lkh3
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
    out["alpha_nosub"] = {"SUBGRADIENT": "NO"}
    out["quadrant5"] = {"CANDIDATE_SET_TYPE": "QUADRANT", "MAX_CANDIDATES": 5}
    out["quadrant8"] = {"CANDIDATE_SET_TYPE": "QUADRANT", "MAX_CANDIDATES": 8}
    out["heat5"] = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 5}
    out["heat8"] = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 8}
    # Union of nn5 and heat5 (up to 10 per city), used by lkh3_equal_time.py only (not probed here).
    out["nnheat5"] = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 10}
    return out


HEAT_RUN = "tagcn_k3_seed0"
NET_VARIANTS = ("heat5", "heat8", "nnheat5")     # variants whose candidates come from the network


def main():
    parser = argparse.ArgumentParser(description="E1 preparation: LKH-3 preprocessing per variant.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="variants to run (default: all)")
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
    todo = {v: p for v, p in variants(n).items()
            if v != "nnheat5" and (args.only is None or v in args.only)}
    if any(v.startswith("heat") for v in todo):
        model, device = heatcand.load_model(HEAT_RUN)
        heatcand.instance_heatmap(model, device, coords, k)          # warm-up (CUDA kernels), not timed
    for name, params in todo.items():
        for with_init in (False, True):
            candidates, net_seconds = None, 0.0
            if name.startswith("heat"):
                start = time.perf_counter()
                pi, pj, prob, _, _ = heatcand.instance_heatmap(model, device, coords, k)
                candidates = heatcand.top_candidates(pi, pj, prob, n, params["MAX_CANDIDATES"])
                net_seconds = time.perf_counter() - start         # graph + network + top-c selection
            r = lkh3.solve(coords, workdir, {"RUNS": 1, "MAX_TRIALS": 1, **params},
                           initial_tour=t0 if with_init else None, candidates=candidates)
            ascent = re.search(r"Ascent time = ([\d.]+) sec", r["log"])
            rec = {"variant": name, "params": params, "initial_tour_file": with_init,
                   "wall_seconds": r["wall_seconds"], "preprocessing_seconds": r["preprocessing_seconds"],
                   "ascent_seconds": float(ascent.group(1)) if ascent else 0.0,
                   "trial_seconds": r["search_seconds"], "net_seconds": net_seconds,
                   "fixed_seconds": net_seconds + r["wall_seconds"] - r["search_seconds"],
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
