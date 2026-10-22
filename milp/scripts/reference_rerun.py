"""Re-solve the instances of a test set that stopped at the time limit, with a longer limit, and keep
the best known solution. The original .npz is not modified; the result goes to a new .npz.

Each unproven instance is solved again from scratch with --time-limit, with the first run's y as a
warm start (so the second run is never worse). The reference cost of every instance becomes the best
of the two runs, and the best lower bound is the larger of the two.

Output (.npz, + .json summary):
    rerun_idx                    indices of the re-solved instances
    rerun_status, rerun_cost, rerun_mip_gap, rerun_bound, rerun_seconds    (per re-solved instance)
    rerun_arc_ptr, rerun_x, rerun_y                                       (their solutions, flat)
    ref_cost, ref_bound, ref_gap (%), ref_proven (0/1), ref_source (0 = first run, 1 = rerun)  (all instances)
    ref_y (flat, same order as the original arcs): the open arcs of the reference solution

Example:
    python scripts/reference_rerun.py --data data/fcnf100_test.npz --time-limit 600 --procs 6 \
        --out data/fcnf100_test_ref600.npz
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fcnf_gnn.data import get_instance, load_dataset, num_instances
from fcnf_gnn.solver import is_feasible, solve_fcnf
from fcnf_gnn.utils import git_info


def resolve(job):
    inst, time_limit = job
    r = solve_fcnf(inst, time_limit=time_limit, warm_start=inst["y"])
    if r["y"] is not None:
        assert is_feasible(inst, r["x"], r["y"])
    return r


def main():
    parser = argparse.ArgumentParser(description="Re-solve unproven instances with a longer limit.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--procs", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rerun", type=int, default=None, help="only the first N unproven instances (smoke test)")
    args = parser.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} already exists: skipped")
        return

    d = load_dataset(args.data)
    num = num_instances(d)
    idx = np.where(d["status"] == 1)[0][:args.max_rerun]
    print(f"{args.data}: {len(idx)} of {num} instances without a proof of optimality; "
          f"re-solving with {args.time_limit:.0f} s, {args.procs} processes", flush=True)
    insts = [get_instance(d, int(i)) for i in idx]
    results = []
    start = time.time()
    with Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap(resolve, [(inst, args.time_limit) for inst in insts], chunksize=1)):
            results.append(r)
            print(f"  {k + 1}/{len(idx)}  instance {idx[k]}: {r['status']}, cost {r['cost']:.2f} "
                  f"(first run {d['cost'][idx[k]]:.2f}), gap {100 * (r['mip_gap'] or 0):.2f} %  "
                  f"({time.time() - start:.0f} s)", flush=True)

    ref_cost, ref_bound = d["cost"].copy(), d["bound"].copy()
    ref_source = np.zeros(num, dtype=np.int8)
    ref_y = d["y"].copy()
    ptr = d["arc_ptr"]
    for i, r in zip(idx, results):
        ref_bound[i] = max(ref_bound[i], r["bound"])
        if r["cost"] is not None and r["cost"] < ref_cost[i]:
            ref_cost[i] = r["cost"]
            ref_source[i] = 1
            ref_y[ptr[i]:ptr[i + 1]] = r["y"]
    ref_gap = 100 * np.maximum(ref_cost - ref_bound, 0) / ref_cost
    proven = (d["status"] == 0).astype(np.int8)
    proven[idx] = [r["status"] == "optimal" for r in results]
    ref_gap[proven == 1] = 0.0

    m_ptr = np.concatenate([[0], np.cumsum([len(inst["src"]) for inst in insts])])
    ok = [r["y"] is not None for r in results]
    np.savez_compressed(
        args.out, rerun_idx=idx, time_limit=args.time_limit,
        rerun_status=np.array([{"optimal": 0, "time_limit": 1}.get(r["status"], 2) for r in results]),
        rerun_cost=np.array([r["cost"] if r["cost"] is not None else np.nan for r in results]),
        rerun_mip_gap=np.array([r["mip_gap"] if r["mip_gap"] is not None else np.nan for r in results]),
        rerun_bound=np.array([r["bound"] for r in results]), rerun_seconds=np.array([r["seconds"] for r in results]),
        rerun_arc_ptr=m_ptr,
        rerun_x=np.concatenate([r["x"] if o else np.zeros(len(i["src"])) for r, i, o in zip(results, insts, ok)]),
        rerun_y=np.concatenate([r["y"] if o else np.zeros(len(i["src"]), np.int8) for r, i, o in zip(results, insts, ok)]),
        ref_cost=ref_cost, ref_bound=ref_bound, ref_gap=ref_gap, ref_proven=proven, ref_source=ref_source, ref_y=ref_y)

    first_gap = 100 * d["mip_gap"][idx]
    improvement = 100 * (d["cost"][idx] - ref_cost[idx]) / d["cost"][idx]
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "data": args.data, "time_limit": args.time_limit, "procs": args.procs},
               "rerun": int(len(idx)), "now_proven": int(sum(r["status"] == "optimal" for r in results)),
               "improved_cost": int((ref_source[idx] == 1).sum()),
               "cost_improvement_pct_mean": float(improvement.mean()),
               "cost_improvement_pct_max": float(improvement.max()),
               "gap_first_run_mean_pct": float(first_gap.mean()),
               "gap_after_mean_pct": float(ref_gap[idx].mean()),
               "still_unproven": int(len(idx) - (proven[idx] == 1).sum()),
               "wall_seconds": time.time() - start}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
