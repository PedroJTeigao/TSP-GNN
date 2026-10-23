"""Full MILP on one test set, recording every improving solution (phase 1b).

One run per instance gives the whole time profile of the full MILP: the best solution found by time t is
the last improving solution with time <= t. The equal-time baseline (the MILP stopped at the time a decoder
used) and the curve at fixed limits are read from it; scripts/equal_time_check.py re-runs a sample with the
time limit itself to check that reading the trajectory gives the same answer.

    --warm none    no warm start
    --warm slope   warm start = slope scaling solution (its time counts as heuristic time)
    --warm trust   warm start = trust-region solution tr_q50_d20 of --source, read from
                   <decode-root>/<source>/<set>__trust.npz (warm_x); its time counts as heuristic time.
                   An instance where it found no solution runs without warm start (warm_used = 0).

Writes <out-root>/<tag>/<set>.npz (tag = none, slope or trust_<source>) with, per instance, seconds (MILP
only), heur_seconds, warm_cost (nan without a warm start), warm_used, status (0 optimal, 1 time limit with a
solution, 2 no solution), cost, bound, and the trajectory flat in traj (rows: seconds since the start of
the MILP, cost, bound) with traj_ptr; then .json (summary, written last).

Example:
    python scripts/milp_trajectory.py --data data/fcnf50_test.npz --warm slope --time-limit 120
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fcnf_gnn.data import get_instance, load_dataset, num_instances
from fcnf_gnn.decode import slope_scaling
from fcnf_gnn.solver import is_feasible, solve_fcnf
from fcnf_gnn.utils import git_info

STATUS = {"optimal": 0, "time_limit": 1}


def run_instance(job):
    i, inst, mode, warm_x, warm_seconds, time_limit = job
    start, heur_seconds = None, 0.0
    if mode == "slope":
        heur = slope_scaling(inst)
        start, heur_seconds = (heur["x"], heur["y"]), heur["seconds"]
    elif mode == "trust":
        heur_seconds = warm_seconds
        if not np.isnan(warm_x).any():
            start = (warm_x, (warm_x > 0).astype(float))
    warm_cost = np.nan
    if start is not None:
        assert is_feasible(inst, *start), f"infeasible warm start on instance {i}"
        warm_cost = float(np.dot(inst["c"], start[0]) + np.dot(inst["F"], start[1]))
    r = solve_fcnf(inst, time_limit=time_limit, warm_start=start, trajectory=True)
    status = STATUS.get(r["status"], 1) if r["x"] is not None else 2
    return {"seconds": r["seconds"], "heur_seconds": heur_seconds, "warm_cost": warm_cost,
            "warm_used": start is not None, "status": status, "cost": np.nan if r["cost"] is None else r["cost"],
            "bound": r["bound"], "traj": r["trajectory"]}


def main():
    parser = argparse.ArgumentParser(description="FCNF phase 1b: full MILP with its improving-solution trajectory.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--warm", choices=("none", "slope", "trust"), default="none")
    parser.add_argument("--source", default=None, help="run name for --warm trust")
    parser.add_argument("--decode-root", default="results/phase1b/decode")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--out-root", default="results/phase1b/traj")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    tag = f"trust_{args.source}" if args.warm == "trust" else args.warm
    out_dir = os.path.join(args.out_root, tag)
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return

    d = load_dataset(args.data)
    num = num_instances(d) if args.max_instances is None else min(args.max_instances, num_instances(d))
    ptr = d["arc_ptr"]
    warm_x, warm_seconds = [None] * num, [0.0] * num
    if args.warm == "trust":
        tr = np.load(os.path.join(args.decode_root, args.source, f"{name}__trust.npz"))
        assert len(tr["tr_q50_d20__seconds"]) >= num, "the trust-region results cover fewer instances"
        warm_x = [tr["warm_x"][ptr[i]:ptr[i + 1]] for i in range(num)]
        warm_seconds = tr["tr_q50_d20__seconds"][:num].tolist()
    jobs = [(i, get_instance(d, i), args.warm, warm_x[i], warm_seconds[i], args.time_limit) for i in range(num)]
    print(f"{name}: {num} instances, warm start {tag}, time limit {args.time_limit}, {args.procs} processes", flush=True)

    rows = []
    start = time.time()
    with Pool(args.procs) as pool:
        for k, row in enumerate(pool.imap(run_instance, jobs, chunksize=1)):
            rows.append(row)
            if (k + 1) % max(1, num // 20) == 0 or k + 1 == num:
                print(f"  {k + 1}/{num}  ({time.time() - start:.0f} s)", flush=True)
    arrays = {key: np.array([r[key] for r in rows], dtype=float) for key in rows[0] if key != "traj"}
    arrays["traj"] = np.concatenate([r["traj"] for r in rows])
    arrays["traj_ptr"] = np.concatenate([[0], np.cumsum([len(r["traj"]) for r in rows])])
    summary = {"seconds_mean": float(arrays["seconds"].mean()), "heur_seconds_mean": float(arrays["heur_seconds"].mean()),
               "proven": int((arrays["status"] == 0).sum()), "no_solution": int((arrays["status"] == 2).sum()),
               "warm_used": int(arrays["warm_used"].sum()),
               "gap_vs_dataset_pct": float(np.nanmean(100 * (arrays["cost"] / d["cost"][:num] - 1)))}
    print(summary)

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                       "dataset": name, "num": num, "warm": args.warm, "source": args.source,
                       "time_limit": args.time_limit, "procs": args.procs, "wall_seconds": time.time() - start},
              **summary}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **arrays)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
