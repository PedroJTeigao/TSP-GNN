"""Convert a standard TSP test file of Fu et al. 2021 / DIMES / DIFUSCO to our .npz format (phase 3 benchmark).

Source: https://github.com/Spider-scnu/TSP (MIT licence), files MCTS/tsp-*/tsp{n}_test_concorde.txt.
One instance per line: "x1 y1 x2 y2 ... xn yn output t1 t2 ... tn t1" (tour 1-indexed, closed).
In tsp10000_test_concorde.txt the "tour" is only 1, 2, ..., n: there is no reference tour. The .npz keeps
the file's tour and length anyway and records whether it is a real tour (has_reference = 0 or 1); the
LKH-3 reference of the benchmark is computed separately (E0).

Example:
    python scripts/convert_benchmark.py --txt data/benchmark/tsp1000_test_concorde.txt
"""

import argparse
import hashlib
import os

import numpy as np

from tsp_gnn.data import is_valid_tour, save_dataset, tour_length


def parse_line(line):
    """(coords (n, 2), tour (n,) 0-indexed) of one line of the file."""
    fields = line.split()
    cut = fields.index("output")
    coords = np.array(fields[:cut], dtype=np.float64).reshape(-1, 2)
    tour = np.array(fields[cut + 1:], dtype=np.int64) - 1
    assert tour[0] == tour[-1], "the tour in the file should be closed"
    return coords, tour[:-1]


def main():
    parser = argparse.ArgumentParser(description="Standard test file (.txt) -> .npz")
    parser.add_argument("--txt", required=True)
    parser.add_argument("--out", default=None, help="default: same name with .npz")
    args = parser.parse_args()
    out = args.out or os.path.splitext(args.txt)[0] + ".npz"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: not overwritten")

    with open(args.txt, "rb") as f:
        raw = f.read()
    sha256 = hashlib.sha256(raw).hexdigest()
    pairs = [parse_line(line) for line in raw.decode().splitlines() if line.strip()]
    coords = np.stack([c for c, _ in pairs])
    tours = np.stack([t for _, t in pairs])
    n = coords.shape[1]
    assert all(is_valid_tour(t, n) for t in tours), "invalid tour in the file"
    lengths = np.array([tour_length(c, t) for c, t in zip(coords, tours)])
    # A placeholder tour is the identity 0, 1, ..., n-1 on every line.
    has_reference = int(not all(np.array_equal(t, np.arange(n)) for t in tours))
    save_dataset(out, coords, tours, lengths, n=n, source=os.path.basename(args.txt), sha256=sha256,
                 has_reference=has_reference)
    print(f"instances: {len(coords)}  |  cities: {n}  |  reference tour in the file: {bool(has_reference)}")
    print(f"mean length of the file's tours: {lengths.mean():.4f}")
    print(f"sha256: {sha256}\nsaved to: {out}")


if __name__ == "__main__":
    main()
