"""Generate a large TSP test set with LKH tours, timing LKH on each instance (phase 2, scale experiment).

Like make_dataset.py, but LKH reads the coordinates (data.solve_lkh_coords) instead of an (n, n)
matrix, which would not fit in memory for n = 10000. The .npz also stores, per instance, the LKH
time in seconds (lkh_seconds) and the number of LKH runs (lkh_runs).

Example:
    python scripts/make_scale_data.py --n 1000 --num 20 --seed 11 --runs 10 --procs 3
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.data import generate_instances, is_valid_tour, save_dataset, solve_lkh_coords, tour_length


def solve_timed(job):
    """(tour, seconds) for one instance; job = (coords, runs)."""
    coords, runs = job
    start = time.perf_counter()
    tour = solve_lkh_coords(coords, runs)
    return tour, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Large TSP test set + LKH tours and LKH times.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10, help="LKH RUNS (10 as in the phase 1 sets)")
    parser.add_argument("--procs", type=int, default=3, help="instances solved in parallel (≤ 8)")
    parser.add_argument("--out", default=None, help="default: data/tsp{n}_test.npz")
    args = parser.parse_args()
    out = args.out or f"data/tsp{args.n}_test.npz"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: not overwritten")

    coords = generate_instances(args.num, args.n, args.seed)
    tours, seconds = [], []
    with Pool(min(args.procs, 8, args.num)) as pool:
        for i, (tour, sec) in enumerate(pool.imap(solve_timed, [(c, args.runs) for c in coords])):
            assert is_valid_tour(tour, args.n), f"LKH returned an invalid tour (instance {i})"
            tours.append(tour); seconds.append(sec)
            print(f"  instance {i + 1}/{args.num}: LKH {sec:.1f} s", flush=True)
    tours, seconds = np.stack(tours), np.array(seconds)
    lengths = np.array([tour_length(c, t) for c, t in zip(coords, tours)])
    save_dataset(out, coords, tours, lengths, n=args.n, seed=args.seed, lkh_seconds=seconds, lkh_runs=args.runs)
    print(f"instances: {args.num}  |  cities: {args.n}  |  LKH runs: {args.runs}")
    print(f"mean length: {lengths.mean():.4f}  |  LKH time per instance: mean {seconds.mean():.1f} s, "
          f"max {seconds.max():.1f} s")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
