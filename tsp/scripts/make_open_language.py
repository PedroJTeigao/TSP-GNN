"""Phase 5, step 3, part 1: the text_v3_open data set (tsp_gnn.open_gen).

    python scripts/make_open_language.py l1 --n 200 --num 150   # L1 rules, sentences and the LKH-3 reference (CPU)
    python scripts/make_open_language.py l2 --n 200             # L2 rewrite by gemma3:12b (needs ollama_pc)

Instances: the step-1 coordinates and pure routes, idx 0..num-1 for the test set and the last --dev instances (idx 199,
198, ...) held out as a development set for the p1 parser. Rules: 2-3 composite rules per instance; if LKH-3 finds no
feasible route the rules are drawn again (at most 5 times), so every label is solvable.

Writes results/phase5/open_language/text_v3_open_tsp<n>.jsonl (+ .npz with the reference tours) and
l2_tsp<n>.jsonl (one record per instance: paragraph, prompt, latency). The L2 step can be resumed.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import llm
from tsp_gnn import open_gen as G
from tsp_gnn import rules as R
from tsp_gnn.utils import git_info

OUT = "results/phase5/open_language"
L2_MODEL = "gemma3:12b"
L2_MIB = 10000
ARGS = None


def label(job):
    n, idx, split = job
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    coords, pure = arr["coords"][idx], arr["pure_tours"][idx]
    rng = np.random.default_rng([G.SEED, n, idx, 2])
    legend = G.make_legend(n, idx, coords)
    pure_len = R.route_length(coords, pure)
    wd = os.path.join(ARGS.work, f"open{n}_w{os.getpid()}")
    params = {"MAX_TRIALS": 1000, "TIME_LIMIT": 60.0 if n <= 200 else 120.0}
    k = int(rng.integers(2, 4))
    res, rules = None, None
    for attempt in range(1, 6):
        rules = G.sample_rules(rng, legend, coords, n, pure_len, k)
        prims = [p for r in rules for p in r["prims"]]
        res = R.solve(coords, prims, wd, params, timeout=3 * params["TIME_LIMIT"] + 60, initial_tour=pure)
        if res["feasible"]:
            break
    background = G.distractors(rng, legend, int(rng.integers(1, 3)))
    rec = {"idx": idx, "n": n, "split": split, "legend": legend, "rules": rules, "background": background,
           "text_l1": R.PREAMBLE + "\n" + "\n".join(f"- {r['sentence']}" for r in rules),
           "kind": res["kind"], "feasible": bool(res["feasible"]), "attempts": attempt, "lkh_runs": res["lkh_runs"],
           "seconds": res["wall_seconds"], "objective": res["objective"], "length": res["length"]}
    return rec, res["tour"]


def init(args):
    global ARGS
    ARGS = args


def main_l1(args):
    path = f"{OUT}/text_v3_open_tsp{args.n}.jsonl"
    if os.path.exists(path):
        print(f"{path} already exists: not overwritten")
        return
    n = args.n
    jobs = [(n, i, "test") for i in range(args.num)] + [(n, 199 - i, "dev") for i in range(args.dev)]
    os.makedirs(args.work, exist_ok=True)
    commit, dirty = git_info()
    start = time.perf_counter()
    res = {}
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool:
        for k, (rec, tour) in enumerate(pool.imap_unordered(label, jobs), 1):
            res[rec["idx"]] = (rec, tour)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  ({time.perf_counter() - start:.0f} s)", flush=True)
    order = sorted(res)
    with open(path, "w") as f:
        for i in order:
            f.write(json.dumps({**res[i][0], "git_commit": commit, "git_dirty": dirty}) + "\n")
    tours = np.stack([res[i][1] if res[i][1] is not None else -np.ones(n, dtype=np.int64) for i in order])
    np.savez_compressed(path.replace(".jsonl", ".npz"), idx=np.array(order), tours=tours)
    print(f"commit {commit}  |  {sum(not res[i][0]['feasible'] for i in order)} infeasible  |  saved to {path}")


def main_l2(args):
    n = args.n
    rows = [json.loads(line) for line in open(f"{OUT}/text_v3_open_tsp{n}.jsonl")]
    if args.limit:                                # the first test instances only (time budget)
        rows = [r for r in rows if r["split"] == "test"][:args.limit]
    path = f"{OUT}/l2_tsp{n}.jsonl"
    done = {json.loads(line)["idx"] for line in open(path)} if os.path.exists(path) else set()
    guard = llm.GpuGuard(f"{OUT}/gpu_guard.jsonl")
    commit, _ = git_info()
    start, calls = time.perf_counter(), 0
    with open(path, "a") as f:
        for row in rows:
            if row["idx"] in done:
                continue
            prompt = G.l2_prompt([r["sentence"] for r in row["rules"]], row["background"])
            guard.check(L2_MODEL, L2_MIB, label=f"L2 n={n}")
            content, sec, raw = llm.chat("You rewrite text faithfully.", prompt, temperature=0.7,
                                         seed=row["idx"], num_predict=700, fmt_json=False, model=L2_MODEL)
            calls += 1
            f.write(json.dumps({"idx": row["idx"], "n": n, "paragraph": (content or "").strip(), "seconds": sec, "lkh_running": llm.lkh_running(),
                                "prompt": prompt, "prompt_tokens": raw.get("prompt_eval_count"),
                                "eval_tokens": raw.get("eval_count"), "error": raw.get("error"),
                                "model": L2_MODEL, "git_commit": commit}) + "\n")
            f.flush()
            if calls % 10 == 0:
                print(f"  {calls} L2 calls  ({time.perf_counter() - start:.0f} s)", flush=True)
    llm.unload(L2_MODEL)
    print(f"L2 n = {n}: {calls} calls, saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="The text_v3_open data set.")
    parser.add_argument("what", choices=("l1", "l2"))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num", type=int, default=150, help="test instances")
    parser.add_argument("--dev", type=int, default=20, help="held-out development instances for the p1 parser")
    parser.add_argument("--limit", type=int, default=None, help="l2: only the first test instances")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--work", default="data/phase5/work")
    args = parser.parse_args()
    (main_l1 if args.what == "l1" else main_l2)(args)


if __name__ == "__main__":
    main()
