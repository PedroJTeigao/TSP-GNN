"""Triage policies on one test set (phase 3, src/tsp_gnn/triage.py): final gap and time of each policy.

--seed s     TAGCN K=3 seed s: P0 (beam everywhere), P2 (LKH on the top 10/20/30/50 % disagreement
             windows, nothing elsewhere) and P3 (LKH where disagreement is high and confidence low,
             beam where confidence is high, nothing elsewhere; disagreement at top 10/20/30 %).
             The thresholds come from the validation log <log-root>/tagcn_k3_seed<s>/tsp200_test.npz
             (triage_log.py) and are fixed numbers, also used on the other sizes.
--seed lkh   P1 (LKH everywhere; no model, no seed).
All start from the greedy edge + 2-opt tour, w = 20, at most --max-passes (3) passes.
Writes <out-root>/{tagcn_k3_seed<s>,lkh}/<dataset>.json (summary + thresholds, written last) and .npz with
per-instance arrays "initial__gap", "initial__ms" and "<policy>__<field>", fields: gap, gain (pp), ms
(refinement time), ms_net, ms_beam, ms_lkh, windows, to_model, to_lkh, improved, passes.

Example:
    python scripts/triage_eval.py --data data/tsp1000_test.npz --seed 0
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
from tsp_gnn.triage import choose_thresholds, triage_refine
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour

from triage_log import load_heatmaps

VALIDATION = "tsp200_test"
P2_TOP = (10, 20, 30, 50)
P3_TOP = (10, 20, 30)


def policies_for(seed, log_root):
    """{policy name: spec} and the thresholds used (None for P1)."""
    if seed == "lkh":
        return {"P1": {"kind": "lkh"}}, None
    with np.load(os.path.join(log_root, f"tagcn_k3_seed{seed}", f"{VALIDATION}.npz")) as z:
        th = choose_thresholds({k: z[k] for k in z.files}, P2_TOP)
    pol = {"P0": {"kind": "model"}}
    pol.update({f"P2_top{q}": {"kind": "dis", "dis": th["dis"][str(q)]} for q in P2_TOP})
    pol.update({f"P3_top{q}": {"kind": "mixed", "dis": th["dis"][str(q)], "conf": th["conf"],
                               "conf_signal": th["conf_signal"]} for q in P3_TOP})
    return pol, th


def main():
    parser = argparse.ArgumentParser(description="Phase 3 triage: policies nothing / model / LKH per window.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", required=True, help="TAGCN K=3 seed, or 'lkh' for P1")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--log-root", default="results/phase3_triage/log")
    parser.add_argument("--out-root", default="results/phase3_triage/policies")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    folder = "lkh" if args.seed == "lkh" else f"tagcn_k3_seed{args.seed}"
    out_dir = os.path.join(args.out_root, folder)
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    policies, thresholds = policies_for(args.seed, args.log_root)
    heatmaps, epoch = (None, None) if args.seed == "lkh" else load_heatmaps(int(args.seed))
    if thresholds:
        print(f"thresholds (from {VALIDATION}): {json.dumps(thresholds)}", flush=True)

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    print(f"{name}: n = {n}, {num} instances, {folder}, policies {list(policies)}", flush=True)
    out = {}
    add = lambda key, v: out.setdefault(key, []).append(v)
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)   # as in the scale experiment
        opt = float(d["lengths"][i])
        start = time.perf_counter()
        t0 = initial_tour(coords, k)
        add("initial__ms", 1000 * (time.perf_counter() - start))
        g0 = optimality_gap(tour_length(coords, t0), opt)
        add("initial__gap", g0)
        for p, spec in policies.items():
            start = time.perf_counter()
            tour, stats = triage_refine(coords, t0, args.width, spec, heatmaps, args.max_passes)
            add(f"{p}__ms", 1000 * (time.perf_counter() - start))
            verify_tour(tour, n)
            g = optimality_gap(tour_length(coords, tour), opt)
            add(f"{p}__gap", g)
            add(f"{p}__gain", g0 - g)
            for f in ("ms_net", "ms_beam", "ms_lkh", "windows", "to_model", "to_lkh", "improved", "passes"):
                add(f"{p}__{f}", stats[f])
        if (i + 1) % 10 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num} instances", flush=True)

    per_inst = {key: np.array(v, dtype=float) for key, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "seed": args.seed, "epoch": epoch, "dataset": name, "n": n, "num": num, "two_opt_k": k,
                        "width": args.width, "max_passes": args.max_passes, "validation": VALIDATION,
                        "policies": policies, "thresholds": thresholds},
               "initial": {"gap_mean": float(per_inst["initial__gap"].mean())}}
    for p in policies:
        windows = per_inst[f"{p}__windows"].sum()
        summary[p] = {"gap_mean": float(per_inst[f"{p}__gap"].mean()),
                      "gain_mean": float(per_inst[f"{p}__gain"].mean()),
                      "lkh_frac": float(per_inst[f"{p}__to_lkh"].sum() / windows),
                      "model_frac": float(per_inst[f"{p}__to_model"].sum() / windows),
                      "ms_mean": float(per_inst[f"{p}__ms"].mean())}
        s = summary[p]
        print(f"  {p:<9} gap {s['gap_mean']:6.2f}%  gain {s['gain_mean']:5.2f} pp  to LKH {100 * s['lkh_frac']:5.1f}%  "
              f"to model {100 * s['model_frac']:5.1f}%  {s['ms_mean']:9.1f} ms", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
