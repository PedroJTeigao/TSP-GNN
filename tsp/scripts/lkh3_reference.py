"""E0 (phase 3 benchmark): recompute the reference tours of a test set with the stand-alone LKH-3 binary.

One .npz per instance in <out-dir>/<set>/inst_<i>.npz (an instance already done is skipped, so a stopped
run can be resumed), then all of them in <out-dir>/<set>.npz with:
    lengths, tours                         new reference (float length, LKH-3.0.6)
    old_lengths                            the dataset's stored tours (elkai / Concorde / placeholder)
    improvement_pct                        100 × (old − new) / old  (> 0: the new reference is shorter)
    wall_seconds, preprocessing_seconds, search_seconds, run_costs (instances × runs, -1 if not run)
    params, lkh_version, has_reference (0 for a file without a reference tour)

LKH-3.0.6 has no total time limit: TIME_LIMIT is per run (checked between trials). To cap a 10k instance
at about 1 h we use RUNS = 5 and TIME_LIMIT = 696 s per run ((3600 − ~120 s of preprocessing) / 5).

With --combine, the per-set files are gathered into one .npz (keys "<set>/<field>"), e.g.
results/phase3_benchmark/ref_lkh3.npz.

Examples:
    python scripts/lkh3_reference.py --data data/tsp1000_test.npz --params RUNS=10 --procs 8
    python scripts/lkh3_reference.py --combine results/phase3_benchmark/ref_lkh3.npz \
        --sets tsp1000_test tsp10000_test
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

from tsp_gnn import lkh3
from tsp_gnn.data import load_dataset

LKH_VERSION = "LKH-3.0.6 (GitHub mirror cerebis/LKH3, commit ba346db)"
MAX_PROCS = 8


def parse_params(text):
    """"RUNS=5,TIME_LIMIT=696" -> {"RUNS": "5", "TIME_LIMIT": "696"}"""
    return dict(item.split("=", 1) for item in text.split(",") if item)


def solve_one(job):
    """Solve instance i and save it; job = (i, coords, params, set_dir)."""
    i, coords, params, set_dir = job
    path = os.path.join(set_dir, f"inst_{i}.npz")
    if os.path.exists(path):
        return i, "skipped"
    r = lkh3.solve(coords, os.path.join(set_dir, f"work_{i}"), params)
    if r["tour"] is None:
        return i, "FAILED (no tour)"
    np.savez(path, params=",".join(f"{k}={v}" for k, v in params.items()), tour=r["tour"], length=r["length"],
             wall_seconds=r["wall_seconds"],
             preprocessing_seconds=r["preprocessing_seconds"], search_seconds=r["search_seconds"],
             run_costs=np.array(r["run_costs"]), run_seconds=np.array(r["run_seconds"]))
    with open(os.path.join(set_dir, f"work_{i}", "lkh_output.txt"), "w") as f:
        f.write(r["log"])
    return i, f"length {r['length']:.4f}, wall {r['wall_seconds']:.0f} s"


def combine(out, out_dir, sets):
    """Gather <out_dir>/<set>.npz of every set into one .npz with keys "<set>/<field>"."""
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: not overwritten")
    merged = {}
    for name in sets:
        with np.load(os.path.join(out_dir, f"{name}.npz")) as f:
            merged.update({f"{name}/{k}": f[k] for k in f.files})
    np.savez(out, **merged)
    print(f"saved to: {out} ({len(sets)} sets)")


def main():
    parser = argparse.ArgumentParser(description="E0: LKH-3 reference of a test set.")
    parser.add_argument("--data")
    parser.add_argument("--params", help='e.g. "RUNS=10" or "RUNS=5,TIME_LIMIT=696"')
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--out-dir", default="results/phase3_benchmark/ref")
    parser.add_argument("--combine", default=None, help="output .npz: gather --sets instead of solving")
    parser.add_argument("--sets", nargs="+", default=[])
    args = parser.parse_args()
    if args.combine:
        return combine(args.combine, args.out_dir, args.sets)
    if not (args.data and args.params):
        parser.error("--data and --params are required unless --combine is given")
    name = os.path.splitext(os.path.basename(args.data))[0]
    set_dir = os.path.join(args.out_dir, name)
    out = os.path.join(args.out_dir, f"{name}.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists: not overwritten")
    os.makedirs(set_dir, exist_ok=True)

    data = load_dataset(args.data)
    params = parse_params(args.params)
    jobs = [(i, c, params, set_dir) for i, c in enumerate(data["coords"])]
    with Pool(min(args.procs, MAX_PROCS, len(jobs))) as pool:
        for i, msg in pool.imap_unordered(solve_one, jobs):
            print(f"  {name} instance {i}: {msg}", flush=True)

    done = [np.load(os.path.join(set_dir, f"inst_{i}.npz")) for i in range(len(jobs))]  # fails if one is missing
    assert all(str(d["params"]) == args.params for d in done), "instances solved with other parameters"
    lengths = np.array([float(d["length"]) for d in done])
    old = data["lengths"].astype(float)
    runs = max(len(d["run_costs"]) for d in done)
    run_costs = np.full((len(done), runs), -1, dtype=np.int64)
    for k, d in enumerate(done):
        run_costs[k, :len(d["run_costs"])] = d["run_costs"]
    improvement = 100 * (old - lengths) / old
    np.savez(out, lengths=lengths, tours=np.stack([d["tour"] for d in done]), old_lengths=old,
             improvement_pct=improvement,
             wall_seconds=np.array([float(d["wall_seconds"]) for d in done]),
             preprocessing_seconds=np.array([float(d["preprocessing_seconds"]) for d in done]),
             search_seconds=np.array([float(d["search_seconds"]) for d in done]), run_costs=run_costs,
             params=args.params, lkh_version=LKH_VERSION, source=args.data,
             has_reference=int(data.get("has_reference", 1)),
             time_limit_note="TIME_LIMIT is per run (LKH-3.0.6), not a total limit")
    print(f"{name}: {len(done)} instances, mean length {lengths.mean():.4f} (old {old.mean():.4f}), "
          f"improvement vs old: mean {improvement.mean():+.4f} %, max {improvement.max():+.4f} %")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
