"""E1 (phase 3 benchmark): LKH-3 on the whole instance at the same time budgets as our policies.

For each instance, variant (candidate set, see lkh3_e1_prep.variants), start (LKH's own initial tour or our
greedy edge + 2-opt tour as INITIAL_TOUR_FILE) and budget B (seconds), LKH runs once (RUNS = 1) with
    TIME_LIMIT = B − (fixed cost of the variant, measured by lkh3_e1_prep.py on one instance)
                   − (our greedy + 2-opt time on this instance, if it is the start)
                   − (graph + network time on this instance, for the heatmap variants)
and at least 0.01 s (TIME_LIMIT = 0 crashes LKH-3.0.6). TIME_LIMIT is checked between trials, so the
real time can exceed B: what is recorded and compared is the real total time:
    total_seconds = LKH wall-clock + greedy + 2-opt (if used) + graph + network (if used)
A (variant, start, budget) whose measured fixed cost + one trial already exceeds B is not run ("does not
fit"); its fields are NaN. The hybrid nnheat5 (union of the 5 nearest neighbours and the 5 edges of highest
p, up to 10 candidates per city, ordered by distance) was not probed: its fixed cost is estimated from heat5
and its trial time from the slower of nn8 / heat8 (see fixed_costs).

Per run also: LKH preprocessing and search times, the number of trials done, the cost after the first trial
(the gap after one trial is computed by the report) and the final float length.

Output: <out-root>/<set>.npz with arrays (instances × configs) and the config list, plus <set>.json
(meta). A set already done is skipped.

Example:
    python scripts/lkh3_equal_time.py --data data/tsp1000_test.npz --budgets 0.72 1.25 1.54 1.94 \
        --prep results/phase3_benchmark/e1_prep/tsp1000.json results/phase3_benchmark/e1_prep/tsp1000_extra.json
"""

import argparse
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn import heatcand, lkh3
from tsp_gnn.data import SCALE, load_dataset
from tsp_gnn.scale import neighbor_lists
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour
from lkh3_e1_prep import HEAT_RUN, NET_VARIANTS, variants

MAX_PROCS = 8
MIN_TIME_LIMIT = 0.01
FIELDS = ("total_seconds", "lkh_wall_seconds", "preprocessing_seconds", "search_seconds", "trials",
          "first_trial_length", "length", "time_limit")


def fixed_costs(prep_files):
    """{(variant, uses_our_tour): (LKH fixed s, one trial s, minimum total s)} from lkh3_e1_prep.py outputs.

    'fixed_seconds' there includes the network for the heatmap variants but not our greedy + 2-opt. Here the
    LKH-only fixed cost is used for TIME_LIMIT (greedy and network are measured again on each instance), and
    the minimum total = LKH fixed + one trial + network + greedy (if our tour is the start) decides whether a
    (variant, start, budget) fits.
    """
    out = {}
    for path in prep_files:
        with open(path) as f:
            rec = json.load(f)
        for v in rec["variants"]:
            net = v.get("net_seconds", 0.0)
            lkh_fixed = v["fixed_seconds"] - net
            greedy = rec["greedy_2opt_seconds"] if v["initial_tour_file"] else 0.0
            out[(v["variant"], v["initial_tour_file"])] = (lkh_fixed, v["trial_seconds"],
                                                           lkh_fixed + v["trial_seconds"] + net + greedy)
    # Hybrid nnheat5 (not probed): LKH fixed cost and network of heat5, one trial as the slower of nn8 /
    # heat8 (up to 10 candidates per city). An estimate, used only for TIME_LIMIT and the "fits" check.
    for ours in (False, True):
        if all((v, ours) in out for v in ("heat5", "heat8", "nn8")):
            lkh_fixed, _, total5 = out[("heat5", ours)]
            trial5 = out[("heat5", ours)][1]
            trial = max(out[("nn8", ours)][1], out[("heat8", ours)][1])
            out[("nnheat5", ours)] = (lkh_fixed, trial, total5 - trial5 + trial)
    return out


def run_job(job):
    """One LKH run; job = (instance, config index, coords, params, tour or None, candidates or None,
    time limit, extra seconds already spent (greedy / network), work root). Each worker process has its own
    work dir (one run at a time per process), so concurrent runs never share files."""
    i, c, coords, params, tour, candidates, limit, extra, work_root = job
    r = lkh3.solve(coords, os.path.join(work_root, f"pid_{os.getpid()}"),
                   {"RUNS": 1, "TIME_LIMIT": f"{limit:.3f}", **params}, initial_tour=tour, candidates=candidates)
    trials = re.search(r"Trials\.min = (\d+)", r["log"])
    first = re.search(r"^\* 1: Cost = (\d+)", r["log"], flags=re.M)
    # The cost of LKH is an integer sum of rounded scaled distances: / SCALE gives the length to ~1e-6.
    first_len = int(first.group(1)) / SCALE if first else float("nan")
    return i, c, {"total_seconds": r["wall_seconds"] + extra, "lkh_wall_seconds": r["wall_seconds"],
                  "preprocessing_seconds": r["preprocessing_seconds"], "search_seconds": r["search_seconds"],
                  "trials": float(trials.group(1)) if trials else float("nan"),
                  "first_trial_length": first_len, "length": r["length"], "time_limit": limit}


def main():
    parser = argparse.ArgumentParser(description="E1: LKH-3 at equal time.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", required=True)
    parser.add_argument("--prep", nargs="+", required=True, help="lkh3_e1_prep.py outputs of this size")
    parser.add_argument("--variants", nargs="*", default=None, help="default: every variant in --prep")
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--out-root", default="results/phase3_benchmark/e1")
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    out_npz, out_json = (os.path.join(args.out_root, f"{name}.{ext}") for ext in ("npz", "json"))
    if os.path.exists(out_json):
        print(f"{out_json} already exists: skipped")
        return
    os.makedirs(args.out_root, exist_ok=True)

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    costs = fixed_costs(args.prep)
    all_variants = variants(n)
    names = [v for v in all_variants if (v, False) in costs and (args.variants is None or v in args.variants)]
    configs = [(v, ours, b) for v in names for ours in (False, True) for b in args.budgets]
    fits = [costs[(v, ours)][2] <= b for v, ours, b in configs]
    print(f"{name}: n = {n}, {num} instances, {len(configs)} configs, {sum(fits)} fit their budget", flush=True)
    for (v, ours, b), ok in zip(configs, fits):
        if not ok:
            print(f"  does not fit: {v} start={'ours' if ours else 'LKH'} budget {b} s "
                  f"(minimum total {costs[(v, ours)][2]:.2f} s)", flush=True)

    # The pool is created before CUDA starts (as in scale_eval.py): the workers only use the CPU.
    pool = Pool(min(args.procs, MAX_PROCS))
    model = device = None
    if any(v in NET_VARIANTS for v in names):
        model, device = heatcand.load_model(HEAT_RUN)
        heatcand.instance_heatmap(model, device, d["coords"][0].astype(np.float64), k)   # warm-up, not timed

    jobs, prep = [], {"greedy_seconds": [], "net_seconds": []}
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)   # as in triage_eval
        start = time.perf_counter()
        t0 = initial_tour(coords, k)
        greedy_s = time.perf_counter() - start
        # Graph + network once per instance (shared), then each variant's own selection, timed separately.
        heat, own_s, net_s = {}, {}, 0.0
        if model is not None:
            start = time.perf_counter()
            pi, pj, prob, _, _ = heatcand.instance_heatmap(model, device, coords, k)
            net_s = time.perf_counter() - start
            for v in names:
                if v not in NET_VARIANTS:
                    continue
                start = time.perf_counter()
                if v == "nnheat5":
                    top5, _ = heatcand.top_candidates(pi, pj, prob, n, 5)
                    heat[v] = heatcand.hybrid_candidates(coords, top5, neighbor_lists(coords, 5))
                else:
                    heat[v] = heatcand.top_candidates(pi, pj, prob, n, all_variants[v]["MAX_CANDIDATES"])
                own_s[v] = time.perf_counter() - start
        prep["greedy_seconds"].append(greedy_s)
        prep["net_seconds"].append(net_s)
        for c, ((v, ours, b), ok) in enumerate(zip(configs, fits)):
            if not ok:
                continue
            extra = (greedy_s if ours else 0.0) + (net_s + own_s[v] if v in heat else 0.0)
            limit = max(MIN_TIME_LIMIT, b - costs[(v, ours)][0] - extra)
            jobs.append((i, c, coords, all_variants[v], t0 if ours else None, heat.get(v), limit, extra,
                         os.path.join(args.out_root, f"{name}_work")))
    print(f"  {len(jobs)} LKH runs", flush=True)

    res = {f: np.full((num, len(configs)), np.nan) for f in FIELDS}
    with pool:
        for done, (i, c, rec) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=4), 1):
            for f in FIELDS:
                res[f][i, c] = rec[f]
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} runs", flush=True)

    commit, dirty = git_info()
    np.savez_compressed(out_npz, **res, greedy_seconds=np.array(prep["greedy_seconds"]),
                        net_seconds=np.array(prep["net_seconds"]), stored_lengths=d["lengths"][:num],
                        config_variant=np.array([v for v, _, _ in configs]),
                        config_ours=np.array([o for _, o, _ in configs]),
                        config_budget=np.array([b for _, _, b in configs]), config_fits=np.array(fits))
    meta = {"git_commit": commit, "git_dirty": dirty, "data": args.data, "n": n, "num": num, "k": k,
            "budgets": args.budgets, "variants": {v: all_variants[v] for v in names}, "prep": args.prep,
            "heat_run": HEAT_RUN, "lkh_version": "LKH-3.0.6 (cerebis/LKH3 mirror)",
            "note": "TIME_LIMIT per run, checked between trials; compare total_seconds with the budget"}
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to: {out_npz}")


if __name__ == "__main__":
    main()
