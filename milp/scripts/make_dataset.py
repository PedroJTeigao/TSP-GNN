"""Generate FCNF instances and label them with the MILP (HiGHS), saved to one .npz (+ .json summary).

Labels: y = 1 for the arcs open in the best solution found (optimal unless --time-limit stops the
solver first; then status = 1 and mip_gap says how far from proven optimal it is).

Example:
    python scripts/make_dataset.py --n 20 --num 1000 --seed 33 --out data/fcnf20_test.npz
    python scripts/make_dataset.py --n 100 --num 200 --seed 35 --time-limit 120 --out data/fcnf100_test.npz
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fcnf_gnn.data import STATUS_CODE, save_dataset
from fcnf_gnn.generate import generate_instances
from fcnf_gnn.solver import is_feasible, solve_fcnf
from fcnf_gnn.utils import git_info

MAX_PROCS = 8  # shared server: at most 8 processes, one HiGHS thread each


def label(job):
    inst, time_limit = job
    r = solve_fcnf(inst, time_limit=time_limit)
    m = len(inst["src"])
    if r["y"] is None:                                   # no feasible solution within the limit
        out = {"x": np.zeros(m), "y": np.zeros(m, np.int8), "cost": np.nan, "status": 2, "mip_gap": np.nan}
    else:
        assert is_feasible(inst, r["x"], r["y"]), "the solver returned an infeasible solution"
        out = {"x": r["x"], "y": r["y"], "cost": r["cost"], "status": STATUS_CODE.get(r["status"], 2),
               "mip_gap": r["mip_gap"]}
    out.update({"bound": r["bound"], "nodes": r["nodes"], "seconds": r["seconds"]})
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate FCNF instances + MILP labels.")
    parser.add_argument("--n", type=int, required=True, help="number of nodes per instance")
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit", type=float, default=None, help="seconds per instance (default: none)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} already exists: skipped")
        return

    start = time.time()
    instances = generate_instances(args.num, args.n, args.seed)
    print(f"generated {args.num} instances (n = {args.n}) in {time.time() - start:.1f} s", flush=True)

    procs = min(MAX_PROCS, os.cpu_count())
    start = time.time()
    with Pool(procs) as pool:
        for i, out in enumerate(pool.imap(label, [(inst, args.time_limit) for inst in instances], chunksize=1)):
            instances[i].update(out)
            if (i + 1) % max(1, args.num // 20) == 0 or i + 1 == args.num:
                print(f"  {i + 1}/{args.num} labelled  ({time.time() - start:.0f} s)", flush=True)
    wall = time.time() - start

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_dataset(args.out, instances, n=args.n, seed=args.seed,
                 time_limit=np.nan if args.time_limit is None else args.time_limit)
    status = np.array([inst["status"] for inst in instances])
    secs = np.array([inst["seconds"] for inst in instances])
    open_frac = np.concatenate([inst["y"] for inst in instances]).mean()
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "n": args.n, "num": args.num, "seed": args.seed, "time_limit": args.time_limit,
                        "processes": procs},
               "arcs_mean": float(np.mean([len(inst["src"]) for inst in instances])),
               "milp_seconds_mean": float(secs.mean()), "milp_seconds_max": float(secs.max()),
               "wall_seconds": wall, "open_frac": float(open_frac),
               "optimal": int((status == 0).sum()), "time_limit_hit": int((status == 1).sum()),
               "no_solution": int((status == 2).sum())}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
