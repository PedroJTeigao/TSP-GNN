"""Decoding sanity check: uses the true labels as probabilities.

With the "perfect network", graphs with full coverage must give the LKH tour (gap 0%).
Example:
    python scripts/check_decode.py --data data/tsp20_test.npz data/tsp50_test.npz
"""

import argparse
import time

import numpy as np

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import (edge_heatmap, greedy_decode, optimality_gap, tour_edges,
                            two_opt, verify_tour)
from tsp_gnn.graph import instance_to_data


def check(path, k):
    """Decode each instance of `path` with probs = y and return one row per instance."""
    d = load_dataset(path)
    rows = []
    for coords, lkh_tour, opt_len in zip(d["coords"], d["tours"], d["lengths"]):
        n = len(coords)
        g = instance_to_data(coords, lkh_tour, opt_len, k)
        start = time.perf_counter()
        heat, in_graph = edge_heatmap(g.edge_index, g.y, n)   # probs = true labels
        greedy = verify_tour(greedy_decode(heat, in_graph, coords), n)
        improved = verify_tour(two_opt(coords, greedy), n)
        elapsed = time.perf_counter() - start
        rows.append({
            "full": int(g.y.sum()) == 2 * n,                  # full tour coverage in the k-NN
            "gap_greedy": optimality_gap(tour_length(coords, greedy), opt_len),
            "gap_2opt": optimality_gap(tour_length(coords, improved), opt_len),
            "same_greedy": tour_edges(greedy) == tour_edges(lkh_tour),  # same edges as LKH
            "same_2opt": tour_edges(improved) == tour_edges(lkh_tour),
            "ms": 1000 * elapsed,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Decoding sanity check with the true labels.")
    parser.add_argument("--data", nargs="+", required=True, help=".npz files from make_dataset.py")
    parser.add_argument("--k", type=int, default=10, help="number of neighbours in the k-NN graph")
    args = parser.parse_args()

    header = (f"{'group':<12}{'graphs':>7}  {'greedy gap (mean / max)':>24}  "
              f"{'+2-opt gap (mean / max)':>24}  {'same as LKH (gr / 2opt)':>23}  {'ms/inst':>8}")
    for path in args.data:
        rows = check(path, args.k)
        print(f"=== {path} ===")
        print(header)
        for name, full in [("full cov.", True), ("incomplete", False)]:
            group = [r for r in rows if r["full"] == full]
            if not group:
                print(f"{name:<12}{0:>7}")
                continue
            col = lambda key: np.array([r[key] for r in group])
            print(f"{name:<12}{len(group):>7}  "
                  f"{col('gap_greedy').mean():>10.4f}% / {col('gap_greedy').max():>8.4f}%  "
                  f"{col('gap_2opt').mean():>10.4f}% / {col('gap_2opt').max():>8.4f}%  "
                  f"{col('same_greedy').sum():>10} / {col('same_2opt').sum():<10}  "
                  f"{col('ms').mean():>8.2f}")
        print()


if __name__ == "__main__":
    main()
