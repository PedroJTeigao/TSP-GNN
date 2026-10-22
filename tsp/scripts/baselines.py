"""Measure the gap to LKH of the classical heuristics (greedy and Christofides, with and without 2-opt).

Saves the summary to results/baselines.json and the per-instance gaps to
results/baselines_per_instance.npz.
Example:
    python scripts/baselines.py --data data/tsp20_test.npz data/tsp50_test.npz
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.baselines import christofides_tour, greedy_distance_tour
from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, two_opt, verify_tour
from tsp_gnn.utils import git_info

MAX_PROCS = 8  # shared server: at most 8 processes
METHODS = ["greedy", "greedy+2opt", "christofides", "christofides+2opt"]


def run_instance(args):
    """Run the 4 methods on one instance. Returns {method: (gap %, time in ms)}."""
    coords, opt_len = args
    n = len(coords)
    out = {}
    for name, heuristic in [("greedy", greedy_distance_tour), ("christofides", christofides_tour)]:
        start = time.perf_counter()
        tour = heuristic(coords)
        t_build = time.perf_counter() - start
        out[name] = (optimality_gap(tour_length(coords, tour), opt_len), 1000 * t_build)

        start = time.perf_counter()
        improved = verify_tour(two_opt(coords, tour), n)
        t_2opt = time.perf_counter() - start
        # the "+2opt" time includes building the initial tour
        out[name + "+2opt"] = (optimality_gap(tour_length(coords, improved), opt_len),
                               1000 * (t_build + t_2opt))
    return out


def main():
    parser = argparse.ArgumentParser(description="Baselines without learning: greedy and Christofides.")
    parser.add_argument("--data", nargs="+", default=["data/tsp20_test.npz", "data/tsp50_test.npz"])
    parser.add_argument("--out", default="results/baselines.json")
    args = parser.parse_args()

    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty,
                        "date": datetime.date.today().isoformat()}}
    per_instance = {}

    procs = min(MAX_PROCS, os.cpu_count())
    for path in args.data:
        name = os.path.splitext(os.path.basename(path))[0]   # e.g. tsp20_test
        d = load_dataset(path)
        with Pool(procs) as pool:
            results = pool.map(run_instance, zip(d["coords"], d["lengths"]), chunksize=8)

        entry = {"n": int(d["coords"].shape[1]), "num": len(results),
                 "lkh_mean_len": float(d["lengths"].mean())}
        for m in METHODS:
            gaps = np.array([r[m][0] for r in results])
            ms = np.array([r[m][1] for r in results])
            lengths = d["lengths"] * (1 + gaps / 100)        # tour length of the method
            entry[m] = {"mean_len": float(lengths.mean()), "gap_mean": float(gaps.mean()),
                        "gap_std": float(gaps.std(ddof=1)), "gap_max": float(gaps.max()),
                        "ms_per_inst": float(ms.mean())}
            per_instance[f"{name}__{m}"] = gaps
        summary[name] = entry

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    npz_path = os.path.splitext(args.out)[0] + "_per_instance.npz"
    np.savez_compressed(npz_path, **per_instance)

    # Table: one row per method, one column group per dataset.
    names = [k for k in summary if k != "meta"]
    print(f"{'method':<18}" + "".join(f"{n + ' (mean / max / ms)':>34}" for n in names))
    for m in METHODS:
        cells = "".join(f"{summary[n][m]['gap_mean']:>12.2f}% /{summary[n][m]['gap_max']:>7.2f}% /"
                        f"{summary[n][m]['ms_per_inst']:>8.1f}  " for n in names)
        print(f"{m:<18}{cells}")
    print(f"\ncommit {commit}{' (uncommitted changes)' if dirty else ''}"
          f"  |  saved to: {args.out} and {npz_path}")


if __name__ == "__main__":
    main()
