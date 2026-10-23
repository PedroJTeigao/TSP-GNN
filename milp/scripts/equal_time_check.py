"""Equal-time baseline run literally (phase 1b): the full MILP with time_limit = the time a decoder used on that
instance, and with the fixed limits of the curve. The report reads the same quantities from the improving-
solution trajectory of one unlimited run (scripts/milp_trajectory.py); this script checks on a sample that both
give the same cost, so the trajectory can stand for the 13 decoders × 6 runs × all instances.

Decoder times come from results/phase1_decode/<source>/<set>__guided.npz (prune_q20/30/50, delegate) and
<decode-root>/<source>/<set>__trust.npz (tr_q*_d*). Writes <out-root>/<source>/<set>.npz with "<method>__limit",
"<method>__cost" (nan = no solution by then), "fixed_<L>__cost" and "index" (the instances), then .json.

Example:
    python scripts/equal_time_check.py --data data/fcnf50_test.npz --source tagcn_k3_seed0 --max-instances 100
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fcnf_gnn.data import get_instance, load_dataset, num_instances
from fcnf_gnn.solver import solve_fcnf
from fcnf_gnn.utils import git_info

FIXED = (1, 2, 5, 10, 25, 50)


def run_job(job):
    k, key, inst, limit = job
    r = solve_fcnf(inst, time_limit=limit)
    return k, key, np.nan if r["cost"] is None else r["cost"]


def main():
    parser = argparse.ArgumentParser(description="FCNF phase 1b: equal-time MILP, run with the time limit itself.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--phase1-root", default="results/phase1_decode")
    parser.add_argument("--decode-root", default="results/phase1b/decode")
    parser.add_argument("--max-instances", type=int, default=100)
    parser.add_argument("--fixed-max", type=float, default=50, help="largest fixed limit to run")
    parser.add_argument("--methods", default=None, help="comma-separated decoders to check (default: all)")
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--out-root", default="results/phase1b/equal_time_check")
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = os.path.join(args.out_root, args.source)
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return

    d = load_dataset(args.data)
    num = min(args.max_instances, num_instances(d))
    index = np.arange(num)                              # the first instances (the sets are i.i.d.)
    limits = {}
    for path in (os.path.join(args.phase1_root, args.source, f"{name}__guided.npz"),
                 os.path.join(args.decode_root, args.source, f"{name}__trust.npz")):
        z = np.load(path)
        for key in z.files:
            if key.endswith("__seconds") and (args.methods is None or key[:-len("__seconds")] in args.methods.split(",")):
                limits[key[:-len("__seconds")]] = z[key][:num]
    for L in FIXED:
        if L <= args.fixed_max:
            limits[f"fixed_{L}"] = np.full(num, float(L))
    insts = [get_instance(d, i) for i in index]
    jobs = [(k, key, insts[k], float(limits[key][k])) for key in limits for k in range(num)]
    jobs.sort(key=lambda j: -j[3])                      # longest first: better load balance
    print(f"{name}: {num} instances, {len(limits)} limits, {len(jobs)} MILPs, "
          f"{sum(j[3] for j in jobs) / args.procs / 60:.0f} min at most", flush=True)

    cost = {key: np.full(num, np.nan) for key in limits}
    start = time.time()
    with Pool(args.procs) as pool:
        for n_done, (k, key, c) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            cost[key][k] = c
            if n_done % max(1, len(jobs) // 20) == 0:
                print(f"  {n_done}/{len(jobs)}  ({time.time() - start:.0f} s)", flush=True)
    arrays = {"index": index}
    for key in limits:
        arrays[f"{key}__limit"] = limits[key]
        arrays[f"{key}__cost"] = cost[key]

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                       "dataset": name, "num": num, "source": args.source, "procs": args.procs,
                       "wall_seconds": time.time() - start}, "limits": sorted(limits)}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **arrays)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
