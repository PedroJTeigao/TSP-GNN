"""Generate a set of TSP instances with LKH tours and save it to an .npz.

Example:
    python scripts/make_dataset.py --n 20 --num 100 --seed 1234
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.data import generate_instances, is_valid_tour, save_dataset, solve_lkh, tour_length

MAX_PROCS = 8  # shared server: at most 8 processes


def main():
    parser = argparse.ArgumentParser(description="Generate TSP instances + LKH tours.")
    parser.add_argument("--n", type=int, required=True, help="number of cities per instance")
    parser.add_argument("--num", type=int, required=True, help="number of instances")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the random generator")
    parser.add_argument("--out", default=None, help=".npz file (default: data/tsp{n}_seed{seed}.npz)")
    args = parser.parse_args()
    out = args.out or f"data/tsp{args.n}_seed{args.seed}.npz"

    coords = generate_instances(args.num, args.n, args.seed)

    # Each instance is independent: we solve several in parallel.
    procs = min(MAX_PROCS, os.cpu_count())
    start = time.time()
    with Pool(procs) as pool:
        tours = pool.map(solve_lkh, coords, chunksize=max(1, args.num // (procs * 4)))
    elapsed = time.time() - start

    tours = np.stack(tours)
    assert all(is_valid_tour(t, args.n) for t in tours), "LKH returned an invalid tour"
    lengths = np.array([tour_length(c, t) for c, t in zip(coords, tours)])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_dataset(out, coords, tours, lengths, n=args.n, seed=args.seed)

    sem = lengths.std(ddof=1) / np.sqrt(len(lengths))  # standard error of the mean
    print(f"instances: {args.num}  |  cities: {args.n}  |  processes: {procs}")
    print(f"mean length: {lengths.mean():.4f} ± {sem:.4f} (standard error)  |  std: {lengths.std(ddof=1):.4f}")
    print(f"min / max: {lengths.min():.4f} / {lengths.max():.4f}")
    print(f"LKH time: {elapsed:.1f} s  ({1000 * elapsed / args.num:.1f} ms per instance)")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
