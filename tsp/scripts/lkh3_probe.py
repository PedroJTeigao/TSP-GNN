"""Time the LKH-3 binary on single instances (phase 3 benchmark, before launching E0 / E1).

Prints and saves, per instance: the float length, its gap to the dataset's stored tour, the wall-clock,
preprocessing and search times, and the cost / time of every run.

Example:
    python scripts/lkh3_probe.py --data data/tsp10000_test.npz --index 0 --params RUNS=1 \
        --out results/phase3_benchmark/probe/tsp10000_runs1.json
"""

import argparse
import json
import os

import numpy as np

from tsp_gnn import lkh3
from tsp_gnn.data import load_dataset


def parse_params(text):
    """"RUNS=1,TIME_LIMIT=8.4" -> {"RUNS": "1", "TIME_LIMIT": "8.4"}"""
    return dict(item.split("=", 1) for item in text.split(",") if item) if text else {}


def main():
    parser = argparse.ArgumentParser(description="Time LKH-3 on a few instances.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, nargs="+", default=[0])
    parser.add_argument("--params", default="RUNS=1", help="comma-separated LKH parameters")
    parser.add_argument("--out", required=True, help=".json with one record per instance")
    parser.add_argument("--workdir", default=None, help="default: <out without .json>_work")
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists: not overwritten")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    workdir = args.workdir or os.path.splitext(args.out)[0] + "_work"

    data = load_dataset(args.data)
    records = []
    for i in args.index:
        r = lkh3.solve(data["coords"][i], workdir, parse_params(args.params))
        stored = float(data["lengths"][i])
        rec = {"data": args.data, "index": i, "params": args.params, "length": r["length"],
               "stored_length": stored, "gap_to_stored_pct": 100 * (r["length"] / stored - 1),
               **{k: r[k] for k in ("wall_seconds", "preprocessing_seconds", "search_seconds",
                                    "run_costs", "run_seconds", "timed_out")}}
        records.append(rec)
        print(f"instance {i}: length {rec['length']:.4f} (stored {stored:.4f}, {rec['gap_to_stored_pct']:+.3f} %)"
              f"  wall {rec['wall_seconds']:.1f} s = preprocessing {rec['preprocessing_seconds']:.1f} s"
              f" + search {rec['search_seconds']:.1f} s  ({len(rec['run_costs'])} runs)", flush=True)
        with open(args.out, "w") as f:      # rewritten after each instance: partial results survive
            json.dump(records, f, indent=1)
        with open(os.path.join(workdir, f"lkh_output_{i}.txt"), "w") as f:
            f.write(r["log"])
    print(f"saved to: {args.out}")


if __name__ == "__main__":
    main()
