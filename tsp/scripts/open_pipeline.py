"""Phase 5, step 3, part 2C: what a formalisation costs in the solver.

For every record of open_formalize.py (and the oracle): the code check (open_dsl.validate_for_solver) rejects the rules
that cannot be solved and records them; LKH-3 solves the rest with the parameters of step 1 (MAX_TRIALS 1000, TIME_LIMIT
60 s at n = 200, 120 s at n = 1000, start from the pure route, external timeout 3 x TIME_LIMIT + 60 s); if no rule is left,
or LKH-3 finds no route, the route is the pure one. The route is judged in code against the LABEL.
A rule set equal to the label reuses the reference route; other sets are solved once per (instance, set).

Solves are written to pipeline_solves_tsp<n>.jsonl as they finish, so a stopped run resumes. Final records:
results/phase5/open_language/pipeline_tsp<n>.jsonl.

Example:
    python scripts/open_pipeline.py --n 200 --workers 6 --limit 50
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import open_dsl as D
from tsp_gnn import open_eval as E
from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

OUT = "results/phase5/open_language"
ARGS = None


def set_key(prims):
    return json.dumps(sorted(repr(tuple(sorted(x) if isinstance(x, frozenset) else x for x in F.rule_key(p)))
                             for p in prims))


def solve_job(job):
    n, idx, key, prims = job
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    coords, pure = arr["coords"][idx], arr["pure_tours"][idx]
    tl = 60.0 if n <= 200 else 120.0
    wd = os.path.join(ARGS.work, f"openpipe{n}_w{os.getpid()}")
    try:
        res = R.solve(coords, prims, wd, {"MAX_TRIALS": 1000, "TIME_LIMIT": tl}, timeout=3 * tl + 60, initial_tour=pure)
    except Exception as e:                     # one bad rule set must not lose the other solves
        return {"idx": idx, "key": key, "tour": np.asarray(pure).tolist(), "feasible": False, "seconds": 0.0,
                "error": repr(e)}
    tour = res["tour"] if res["feasible"] else pure
    return {"idx": idx, "key": key, "tour": np.asarray(tour).tolist(), "feasible": bool(res["feasible"]),
            "seconds": res["wall_seconds"], "error": None}


def init(args):
    global ARGS
    ARGS = args


def main():
    parser = argparse.ArgumentParser(description="Open-language pipeline: formalisation -> code check -> LKH-3.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="only the first instances of each level")
    parser.add_argument("--work", default="data/phase5/work")
    args = parser.parse_args()
    n = args.n
    insts = {(i["idx"], lv): i for lv in E.LEVELS for i in E.load_level(n, lv, args.limit)}
    recs = [e for e in map(json.loads, open(f"{OUT}/formalize_tsp{n}.jsonl")) if (e["idx"], e["level"]) in insts]
    cases = [{"idx": i["idx"], "level": i["level"], "arm": "oracle", "pred": i["label"], "seconds": 0.0}
             for i in insts.values()] + recs
    jobs = {}
    for c in cases:
        inst = insts[(c["idx"], c["level"])]
        prims, rejected = D.validate_for_solver(c["pred"], inst["coords"])
        c["prims"], c["rejected"] = prims, rejected
        c["key"] = set_key(prims)
        c["is_label"] = c["key"] == set_key(E.label_prims(inst))
        if not c["is_label"] and prims and (c["idx"], c["key"]) not in jobs:
            jobs[(c["idx"], c["key"])] = (n, c["idx"], c["key"], prims)
    cache = f"{OUT}/pipeline_solves_tsp{n}.jsonl"
    solved = {}
    if os.path.exists(cache):
        solved = {(e["idx"], e["key"]): e for e in map(json.loads, open(cache))}
    todo = [j for k, j in jobs.items() if k not in solved]
    print(f"{len(cases)} cases, {len(jobs)} LKH-3 solves needed, {len(todo)} to run", flush=True)
    os.makedirs(args.work, exist_ok=True)
    start = time.perf_counter()
    with open(cache, "a") as f, Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for k, e in enumerate(pool.imap_unordered(solve_job, todo), 1):
            solved[(e["idx"], e["key"])] = e
            f.write(json.dumps(e) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} solves  ({time.perf_counter() - start:.0f} s)", flush=True)
    commit, _ = git_info()
    with open(f"{OUT}/pipeline_tsp{n}.jsonl", "w") as f:
        for c in cases:
            inst = insts[(c["idx"], c["level"])]
            coords, ref = inst["coords"], inst["reference"]
            label = E.label_prims(inst)
            s = solved.get((c["idx"], c["key"]))
            if c["is_label"]:
                tour, feas, sec = ref, True, 0.0
            elif s is None:                        # nothing left after the check: the pure route
                tour, feas, sec = inst["pure"], True, 0.0
            else:
                tour, feas, sec = np.array(s["tour"]), s["feasible"], s["seconds"]
            ref_len, ref_obj = R.route_length(coords, ref), R.objective(coords, ref, label)
            f.write(json.dumps({"idx": c["idx"], "n": n, "level": c["level"], "arm": c["arm"],
                                "reused_reference": c["is_label"], "rejected": c["rejected"], "solver_feasible": feas,
                                "lkh_seconds": sec, "llm_seconds": c.get("seconds", 0.0),
                                "valid": bool(R.check_all(label, coords, tour)),
                                "violated": [label[k]["type"] for k in R.violated_rules(label, coords, tour)],
                                "length_gap": R.route_length(coords, tour) / ref_len - 1,
                                "cost_gap": R.objective(coords, tour, label) / ref_obj - 1,
                                "git_commit": commit}) + "\n")
    print(f"saved to {OUT}/pipeline_tsp{n}.jsonl")


if __name__ == "__main__":
    main()
