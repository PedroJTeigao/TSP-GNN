"""Evaluate the heatmap-guided decoders (with the probabilities of one trained run) or the baselines without
a network on one test set. Costs are stored per instance; gaps are computed by the report against the
reference cost (best known solution), so a better reference found later changes no file here.

    --source <run name>   prune_q20, prune_q30, prune_q50, warm, delegate with results/runs/<run>/eval_probs.npz
    --source baselines    slope, lp_round, random_q20, random_q30, random_q50 (prune with p uniform at random,
                          seed = instance index)
Every MILP gets --time-limit (none on fcnf20; 120 s on fcnf50 / fcnf100, like the reference MILP).

Writes <out-root>/<source>/<set>.npz with "<method>__<field>" per instance (field: cost, seconds,
first_feasible, kept_frac, repairs, solved, plus bfs_added for prune) and .json (summary, written last).

Example:
    python scripts/decode_eval.py --data data/fcnf20_test.npz --source tagcn_k3_seed0
    python scripts/decode_eval.py --data data/fcnf100_test.npz --source baselines --time-limit 120
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from fcnf_gnn.data import get_instance, load_dataset, num_instances
from fcnf_gnn.decode import delegate, lp_round, prune, slope_scaling, warm
from fcnf_gnn.solver import is_feasible
from fcnf_gnn.utils import git_info

QS = (20, 30, 50)
FIELDS = ("cost", "seconds", "first_feasible", "kept_frac", "repairs", "bfs_added")


def run_instance(job):
    i, inst, p, time_limit = job
    out = {}
    if p is None:                                         # baselines
        out["slope"] = slope_scaling(inst)
        out["lp_round"] = lp_round(inst)
        rand = np.random.default_rng(i).random(len(inst["src"]))
        for q in QS:
            out[f"random_q{q}"] = prune(inst, rand, q, time_limit)
    else:
        for q in QS:
            out[f"prune_q{q}"] = prune(inst, p, q, time_limit)
        out["warm"] = warm(inst, p, time_limit)
        out["delegate"] = delegate(inst, p, time_limit, heuristic=slope_scaling(inst))
    row = {}
    for method, r in out.items():
        if r["x"] is not None:
            assert is_feasible(inst, r["x"], r["y"]), f"{method}: infeasible solution on instance {i}"
        row[f"{method}__solved"] = r["x"] is not None
        for f in FIELDS:
            if f in r:
                row[f"{method}__{f}"] = float(r[f])
    return row


def main():
    parser = argparse.ArgumentParser(description="FCNF phase 1: heatmap-guided decoders / baselines.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", required=True, help="run name in results/runs, or 'baselines'")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--out-root", default="results/phase1_decode")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = os.path.join(args.out_root, args.source)
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return

    d = load_dataset(args.data)
    num = num_instances(d) if args.max_instances is None else min(args.max_instances, num_instances(d))
    probs, run_meta = [None] * num, None
    if args.source != "baselines":
        flat = np.load(os.path.join("results", "runs", args.source, "eval_probs.npz"))[name]
        assert len(flat) == d["arc_ptr"][-1], "probabilities and arcs do not match"
        probs = [flat[d["arc_ptr"][i]:d["arc_ptr"][i + 1]] for i in range(num)]
        with open(os.path.join("results", "runs", args.source, "eval.json")) as f:
            run_meta = {"best_epoch": json.load(f)["meta"]["best_epoch"]}
    jobs = [(i, get_instance(d, i), probs[i], args.time_limit) for i in range(num)]
    print(f"{name}: {num} instances, source {args.source}, time limit {args.time_limit}, {args.procs} processes",
          flush=True)

    rows = []
    start = time.time()
    with Pool(args.procs) as pool:
        for k, row in enumerate(pool.imap(run_instance, jobs, chunksize=1)):
            rows.append(row)
            if (k + 1) % max(1, num // 20) == 0 or k + 1 == num:
                print(f"  {k + 1}/{num}  ({time.time() - start:.0f} s)", flush=True)
    arrays = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    methods = sorted({key.split("__")[0] for key in arrays})
    ref = d["cost"][:num]
    summary = {m: {"gap_vs_dataset_pct": float(np.nanmean(100 * (arrays[f"{m}__cost"] / ref - 1))),
                   "seconds_mean": float(arrays[f"{m}__seconds"].mean()),
                   "first_feasible": float(arrays[f"{m}__first_feasible"].mean()),
                   "kept_frac": float(arrays[f"{m}__kept_frac"].mean()),
                   "solved": float(arrays[f"{m}__solved"].mean())} for m in methods}
    print(f"{'method':<12}{'gap vs data':>12}{'seconds':>9}{'1st feas.':>10}{'kept':>7}")
    for m, s in summary.items():
        print(f"{m:<12}{s['gap_vs_dataset_pct']:>11.2f}%{s['seconds_mean']:>9.2f}{s['first_feasible']:>10.2f}"
              f"{s['kept_frac']:>7.2f}")

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                       "dataset": name, "num": num, "source": args.source, "run": run_meta,
                       "time_limit": args.time_limit, "procs": args.procs, "wall_seconds": time.time() - start},
              **summary}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **arrays)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
