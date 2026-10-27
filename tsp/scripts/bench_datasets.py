"""Phase 3 benchmark, E3: test sets whose "lengths" are the LKH-3 reference of E0 (ref_lkh3.npz).

The evaluation scripts (window_eval.py, triage_eval.py, triage_ablation.py) compute the gap against the
dataset's "lengths". The Fu et al. files carry non-optimal tours (1k) or no tour at all (10k: the stored
length is that of the identity tour), so they are re-written with the E0 reference:
    <out-dir>/<set>.npz   coords, lengths (= E0 LKH-3 lengths), source_lengths (the original ones),
                          reference ("LKH-3 E0, <params>"), source (the original file)
The original files are not changed. A file that already exists is not overwritten.

Example:
    python scripts/bench_datasets.py --sets tsp1000_test_concorde tsp10000_test_concorde
"""

import argparse
import os

import numpy as np

from tsp_gnn.data import load_dataset

SOURCES = {"tsp1000_test": "data/tsp1000_test.npz", "tsp10000_test": "data/tsp10000_test.npz",
           "tsp1000_test_concorde": "data/benchmark/tsp1000_test_concorde.npz",
           "tsp10000_test_concorde": "data/benchmark/tsp10000_test_concorde.npz"}


def main():
    parser = argparse.ArgumentParser(description="E3: test sets with the E0 LKH-3 reference.")
    parser.add_argument("--sets", nargs="+", required=True, choices=list(SOURCES))
    parser.add_argument("--ref", default="results/phase3_benchmark/ref_lkh3.npz")
    parser.add_argument("--out-dir", default="results/phase3_benchmark/data")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with np.load(args.ref) as ref:
        for name in args.sets:
            out = os.path.join(args.out_dir, f"{name}.npz")
            if os.path.exists(out):
                print(f"{out} already exists: not overwritten")
                continue
            d = load_dataset(SOURCES[name])
            lengths = ref[f"{name}/lengths"].astype(np.float64)
            assert len(lengths) == len(d["coords"]), "reference and dataset differ in size"
            np.savez(out, coords=d["coords"], lengths=lengths, source_lengths=d["lengths"],
                     reference=f"LKH-3 E0, {ref[f'{name}/params']}", source=SOURCES[name])
            print(f"{name}: {len(lengths)} instances, mean reference {lengths.mean():.4f} "
                  f"(source {d['lengths'].mean():.4f})  ->  {out}")


if __name__ == "__main__":
    main()
