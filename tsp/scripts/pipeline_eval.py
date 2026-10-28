"""Phase 5, step 2, part C: the full pipeline per formalisation arm, judged against the CORRECT translation.

For every instance, text (v1 / v2b) and arm (b0, b1, b2 from scripts/formalize_eval.py, and the oracle = the label):
    1. the arm's formal rules go through the same numeric rule as the step-1 reference solver: rules.problem_kind
       picks TSP / SOP / TSPTW and the LKH-3 parameters are the same (MAX_TRIALS 1000, TIME_LIMIT 60 s at n = 200,
       120 s at n = 1000, LKH started from the pure route). Translations LKH-3 cannot take are repaired and the
       repair recorded: precedences dropped when a time window is present; a second window of the same customer
       dropped;
    2. LKH-3 returns a route; if it finds none that meets the arm's rules, the pipeline falls back to the pure-TSP
       route (recorded as solver_infeasible);
    3. the code checker (rules.check_rule / check_all) judges the route against the LABEL: valid or not, which
       rules are violated, and the objective (length + soft-zone penalty of the label) against the reference.
A rule set equal to the label reuses the step-1 reference route (same solver, same parameters), so the oracle is the
reference itself; any other rule set is solved once and shared by the arms that predicted it.

Reads data/phase5/text_rules_v2b_tsp<n>.jsonl, data/phase5/formalize_tsp<n>.jsonl and the step-1 .npz.
Writes data/phase5/pipeline_tsp<n>.jsonl (one record per instance, text and arm). At most --workers LKH processes.

Example:
    python scripts/pipeline_eval.py --n 200 --workers 8
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

ARGS = None


def repair(pred):
    """Make a predicted rule set acceptable to LKH-3; returns (rules, list of repairs)."""
    rules, notes, windows = [], [], set()
    has_tw = any(r["type"] == "time_window" for r in pred)
    for r in pred:
        if has_tw and r["type"] == "precedence":
            notes.append("precedence dropped (time window present)")
            continue
        if r["type"] == "time_window":
            c = r["formal"]["customer"]
            if c in windows:
                notes.append("second window of a customer dropped")
                continue
            windows.add(c)
        rules.append(r)
    return rules, notes


def rule_set_key(rules):
    return tuple(sorted(repr(F.rule_key(r)) for r in rules))


def solve_job(job):
    """job = (n, idx, key, rules): solve one rule set with the reference parameters."""
    n, idx, key, rules = job
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    coords, pure = arr["coords"][idx], arr["pure_tours"][idx]
    wd = os.path.join(ARGS.work, f"pipe{n}_w{os.getpid()}")
    params = {"MAX_TRIALS": 1000, "TIME_LIMIT": ARGS.time_limit}
    res = R.solve(coords, rules, wd, params, initial_tour=pure)
    tour = res["tour"] if res["feasible"] else pure
    return (idx, key), {"tour": np.asarray(tour).tolist(), "solver_feasible": bool(res["feasible"]),
                        "kind": res["kind"], "seconds": res["wall_seconds"]}


def init(args):
    global ARGS
    ARGS = args


def label_rules(row):
    return [{"type": r["type"], "formal": r["formal"]} for r in row["rules"]]


def main():
    parser = argparse.ArgumentParser(description="Full pipeline per arm: translation -> LKH-3 -> code checker.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=None, help="default 60 s at n <= 200, else 120 s")
    parser.add_argument("--out", default=None, help="default data/phase5/pipeline_tsp<n>.jsonl")
    parser.add_argument("--work", default="data/phase5/work")
    args = parser.parse_args()
    n = args.n
    args.time_limit = args.time_limit or (60.0 if n <= 200 else 120.0)
    out = args.out or f"data/phase5/pipeline_tsp{n}.jsonl"
    if os.path.exists(out):
        print(f"{out} already exists: not overwritten")
        return
    rows = {r["idx"]: r for r in map(json.loads, open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl"))}
    step1 = {r["idx"]: r for r in map(json.loads, open(f"data/phase5/text_rules_tsp{n}.jsonl"))}
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    preds = [json.loads(line) for line in open(f"data/phase5/formalize_tsp{n}.jsonl")]
    for idx, row in rows.items():
        for text in ("v1", "v2"):
            preds.append({"idx": idx, "text": text, "arm": "oracle", "pred": label_rules(row)})

    # the rule sets to solve: one job per distinct (instance, rule set) different from the label
    jobs, cases = {}, []
    for p in preds:
        row = rows[p["idx"]]
        rules, notes = repair(p["pred"])
        key = rule_set_key(rules)
        is_label = key == rule_set_key(label_rules(row))
        if not is_label and (p["idx"], key) not in jobs:
            jobs[(p["idx"], key)] = (n, p["idx"], key, rules)
        cases.append((p, rules, notes, key, is_label))
    print(f"{len(cases)} cases, {len(jobs)} LKH-3 solves needed", flush=True)

    commit, dirty = git_info()
    start = time.perf_counter()
    os.makedirs(args.work, exist_ok=True)
    solved = {}
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for k, (jk, res) in enumerate(pool.imap_unordered(solve_job, jobs.values()), 1):
            solved[jk] = res
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} solves  ({time.perf_counter() - start:.0f} s)", flush=True)

    with open(out, "w") as f:
        for p, rules, notes, key, is_label in cases:
            idx = p["idx"]
            coords = arr["coords"][idx]
            label = label_rules(rows[idx])
            if is_label:
                tour, feas, seconds = arr["tours"][idx], True, step1[idx]["seconds"]
            else:
                s = solved[(idx, key)]
                tour, feas, seconds = np.array(s["tour"]), s["solver_feasible"], s["seconds"]
            violated = [label[k]["type"] for k in R.violated_rules(label, coords, tour)]
            valid = R.check_all(label, coords, tour)
            obj = R.objective(coords, tour, label)
            ref = R.objective(coords, arr["tours"][idx], label)
            f.write(json.dumps({"idx": idx, "n": n, "text": p["text"], "arm": p["arm"], "reused_reference": is_label,
                                "repairs": notes, "solver_feasible": feas, "kind": R.problem_kind(rules),
                                "lkh_seconds": seconds, "valid": bool(valid), "violated": violated,
                                "objective": obj, "reference_objective": ref,
                                "cost_vs_reference": obj / ref - 1, "git_commit": commit}) + "\n")
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out}")


if __name__ == "__main__":
    main()
