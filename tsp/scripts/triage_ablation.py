"""E2 (phase 3 benchmark): which windows should go to LKH? Ranked triage policies on one test set.

In each half-pass the top x % of the windows by a score go to LKH, the rest is left as it is (triage.ranked_refine,
w = 20, offsets 0 and w/2, at most 3 passes, from the greedy edge + 2-opt tour). Scores:
    random         uniform; --seeds are the random-number seeds (5 by default)
    simple         segment length / sum of the nearest-neighbour distances of its cities (no model)
    disagreement   1 − mean network probability of the segment's edges; --seeds is ONE TAGCN K=3 seed
    oracle         the real LKH gain of the window (an upper bound: LKH runs on every window to know it)
Unlike P2 of the triage experiment (fixed thresholds from tsp200), every score sends exactly the same share of
windows to LKH, so the comparison is only about the ranking.

Writes <out-root>/<score>[_seed<s>]/<dataset>.{json,npz}; the .npz has per-instance arrays "initial__gap",
"initial__ms" and "<policy>__<field>" with policy "<score>_s<seed>_top<x>" (random) or "<score>_top<x>" and
fields gap, gain, ms (refinement time), ms_score, ms_net, ms_lkh, windows, to_lkh, improved, passes.

Example:
    python scripts/triage_ablation.py --data results/phase3_benchmark/data/tsp1000_test.npz --score simple
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, verify_tour
from tsp_gnn.triage import RANKED_SCORES, ranked_refine
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour

from triage_log import load_heatmaps

FIELDS = ("ms_score", "ms_net", "ms_lkh", "windows", "to_lkh", "improved", "passes")


def main():
    parser = argparse.ArgumentParser(description="E2: ranked triage policies (random / simple / disagreement / oracle).")
    parser.add_argument("--data", required=True)
    parser.add_argument("--score", choices=RANKED_SCORES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="random: rng seeds (default 0-4); disagreement: one TAGCN K=3 seed")
    parser.add_argument("--fracs", type=int, nargs="+", default=[20, 50], help="x, in % of the windows")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--out-root", default="results/phase3_benchmark/e2")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    if args.score == "disagreement" and (args.seeds is None or len(args.seeds) != 1):
        parser.error("disagreement needs exactly one --seeds (the TAGCN K=3 seed)")
    seeds = args.seeds if args.seeds is not None else ([0, 1, 2, 3, 4] if args.score == "random" else [None])
    folder = f"disagreement_seed{seeds[0]}" if args.score == "disagreement" else args.score
    out_dir = os.path.join(args.out_root, folder)
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    heatmaps, epoch = load_heatmaps(seeds[0]) if args.score == "disagreement" else (None, None)

    # One policy per (rng seed, x); the random generator of each (instance, policy) is seeded from both.
    policies = {}
    for s in seeds:
        for x in args.fracs:
            key = f"{args.score}_s{s}_top{x}" if args.score == "random" else f"{args.score}_top{x}"
            policies[key] = (s, x)

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    print(f"{name}: n = {n}, {num} instances, {folder}, policies {list(policies)}", flush=True)
    out = {}
    add = lambda key, v: out.setdefault(key, []).append(v)
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)   # as in the triage experiment
        opt = float(d["lengths"][i])
        start = time.perf_counter()
        t0 = initial_tour(coords, k)
        add("initial__ms", 1000 * (time.perf_counter() - start))
        g0 = optimality_gap(tour_length(coords, t0), opt)
        add("initial__gap", g0)
        for p, (s, x) in policies.items():
            rng = np.random.default_rng([s, i]) if args.score == "random" else None
            start = time.perf_counter()
            tour, stats = ranked_refine(coords, t0, args.width, args.score, x / 100, heatmaps, rng, args.max_passes)
            add(f"{p}__ms", 1000 * (time.perf_counter() - start))
            verify_tour(tour, n)
            g = optimality_gap(tour_length(coords, tour), opt)
            add(f"{p}__gap", g)
            add(f"{p}__gain", g0 - g)
            for f in FIELDS:
                add(f"{p}__{f}", stats[f])
        if (i + 1) % 10 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num} instances", flush=True)

    per_inst = {key: np.array(v, dtype=float) for key, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "score": args.score, "seeds": seeds, "epoch": epoch, "dataset": name, "data": args.data,
                        "n": n, "num": num, "two_opt_k": k, "width": args.width, "max_passes": args.max_passes,
                        "fracs": args.fracs},
               "initial": {"gap_mean": float(per_inst["initial__gap"].mean())}}
    for p in policies:
        summary[p] = {"gap_mean": float(per_inst[f"{p}__gap"].mean()),
                      "gain_mean": float(per_inst[f"{p}__gain"].mean()),
                      "lkh_frac": float(per_inst[f"{p}__to_lkh"].sum() / per_inst[f"{p}__windows"].sum()),
                      "ms_mean": float(per_inst[f"{p}__ms"].mean())}
        s = summary[p]
        print(f"  {p:<22} gap {s['gap_mean']:6.2f}%  gain {s['gain_mean']:5.2f} pp  "
              f"to LKH {100 * s['lkh_frac']:5.1f}%  {s['ms_mean']:9.1f} ms", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
