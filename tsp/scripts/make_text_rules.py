"""Phase 5, step 1: label TSP instances with text rules (tsp_gnn.rules) using the LKH-3 reference solver.

For each instance (uniform points in the unit square, customer 0 = depot, new seeds [20261027, n, i]):
    1. pure TSP with the same pipeline (explicit matrix, rounded-up distances, same LKH parameters);
    2. 1-3 rules (types drawn by rules.sample_types; each made to bind against the pure route where possible);
    3. the reference solution of the formal translation, LKH starting from the pure route (in the direction that
       breaks fewer rules); if LKH finds no feasible route, the rules are re-drawn
       (at most --attempts times; the number of attempts is recorded).
Runs --workers instances in parallel (one LKH process each).

Writes <out>.jsonl (one record per instance: rules with text and formal form, the instance text, pure and reference
lengths, objective, LKH kind, runs and seconds, which rules the pure route breaks) and <out>.npz (coords, pure and
reference tours).

Example:
    python scripts/make_text_rules.py --n 200 --num 200 --workers 8
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

SEED = 20261027
ARGS = None


def label(i):
    n = ARGS.n
    rng = np.random.default_rng([SEED, n, i])
    coords = rng.random((n, 2))
    params = {"MAX_TRIALS": ARGS.max_trials, "TIME_LIMIT": ARGS.time_limit}
    wd = os.path.join(ARGS.work, f"tsp{n}_w{os.getpid()}")
    pure = R.solve(coords, [], os.path.join(wd, "pure"), params)
    assert pure["feasible"]
    k = int(rng.integers(1, 4))
    for attempt in range(1, ARGS.attempts + 1):
        types = R.sample_types(rng, k)
        taken = set()
        rules = [R.sample_rule(t, rng, coords, pure["tour"], pure["length"], taken) for t in types]
        res = R.solve(coords, rules, os.path.join(wd, "rules"), params, initial_tour=pure["tour"])
        if res["feasible"]:
            break
    rec = {"idx": i, "n": n, "num_rules": k, "attempts": attempt, "rules": rules, "text": R.instance_text(rules),
           "kind": res["kind"], "feasible": res["feasible"], "lkh_runs": res["lkh_runs"],
           "seconds": res["wall_seconds"], "pure_seconds": pure["wall_seconds"],
           "pure_length": pure["length"], "length": res["length"], "objective": res["objective"],
           "pure_objective": R.objective(coords, pure["tour"], rules),
           "pure_breaks": [not R.check_rule(r, coords, pure["tour"])[0] for r in rules],
           "zone_length": [R.check_rule(r, coords, res["tour"])[1] if res["feasible"] else None
                           for r in rules if r["type"] == "soft_zone"],
           "pure_zone_length": [R.check_rule(r, coords, pure["tour"])[1] for r in rules if r["type"] == "soft_zone"]}
    return rec, coords, pure["tour"], res["tour"]


def init(args):
    global ARGS
    ARGS = args


def main():
    parser = argparse.ArgumentParser(description="Label TSP instances with text rules (LKH-3 reference).")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=60.0, help="LKH TIME_LIMIT per run (s)")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--out", default=None, help="default data/phase5/text_rules_tsp<n>")
    parser.add_argument("--work", default="data/phase5/work")
    args = parser.parse_args()
    out = args.out or f"data/phase5/text_rules_tsp{args.n}"
    if os.path.exists(out + ".jsonl"):
        print(f"{out}.jsonl already exists: not overwritten")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    commit, dirty = git_info()
    start = time.perf_counter()
    results = {}
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for rec, coords, pure_tour, tour in pool.imap_unordered(label, range(args.num)):
            results[rec["idx"]] = (rec, coords, pure_tour, tour)
            if len(results) % 10 == 0 or len(results) == args.num:
                print(f"  {len(results)}/{args.num} instances  ({time.perf_counter() - start:.0f} s)", flush=True)
    meta = {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(), "seed": SEED,
            "max_trials": args.max_trials, "time_limit": args.time_limit, "lkh_scale": R.LKH_SCALE,
            "total_seconds": time.perf_counter() - start}
    order = sorted(results)
    with open(out + ".jsonl", "w") as f:
        for i in order:
            f.write(json.dumps({**results[i][0], "meta": meta}) + "\n")
    n = args.n
    tours = np.stack([results[i][3] if results[i][3] is not None else -np.ones(n, dtype=np.int64) for i in order])
    np.savez_compressed(out + ".npz", coords=np.stack([results[i][1] for i in order]),
                        pure_tours=np.stack([results[i][2] for i in order]), tours=tours)
    bad = sum(not results[i][0]["feasible"] for i in order)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  {bad} infeasible  |  saved to {out}.jsonl")


if __name__ == "__main__":
    main()
