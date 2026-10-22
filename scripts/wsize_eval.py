"""Window size vs. solver (phase 3): window refinement with w = 20 ... 300 on one test set.

--solver lkh             LKH on each window (1 run)
--solver model --run R   TAGCN checkpoint results/runs/<R>/best.pt, beam B=10 paths
Starts from the greedy edge + 2-opt tour (window.initial_tour, k = min(0.5 n, 100)) and runs at most
--max-passes passes (2) for each width. A pass is run only if the previous one improved the tour. For
LKH, if the first pass on the first instance of the set takes more than --lkh-limit-ms per window (5 s),
that width gets 1 pass on every instance of the set ("<key>__limited" = 1; the decision is kept in meta).
Writes <out-root>/{lkh,<R>}/<dataset>.json (summary, written last) and .npz with per-instance arrays
"initial__gap", "initial__ms" and "<solver>_w<w>__<field>", fields: gap, gain (pp), ms (refinement
time), ms_model (network time inside it), windows, improved, passes, limited.

Example:
    python scripts/wsize_eval.py --data data/tsp1000_test.npz --solver model --run tagcn_k3_tsp100_seed0 --widths 100 200
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
from tsp_gnn.hier import Heatmaps
from tsp_gnn.models import build_model
from tsp_gnn.utils import git_info, limit_gpu_memory
from tsp_gnn.window import initial_tour, make_solver, refine

EDGES_PER_BATCH = 256 * 20 * 10     # model batch of windows: as many k-NN edges as 256 windows of 20 cities


def batch_size(w):
    return max(1, min(256, EDGES_PER_BATCH // (w * round(0.5 * w))))


def refine_passes(coords, tour, w, solver, max_passes, limit_ms=None):
    """window.refine one pass at a time (same result as refine(..., max_passes)); stops after the first
    pass if it took more than limit_ms per window. Returns (tour, stats, ms per window of the first pass)."""
    total = {"windows": 0, "improved": 0, "passes": 0, "gain": 0.0}
    first_ms = None
    for _ in range(max_passes):
        start = time.perf_counter()
        tour, stats = refine(coords, tour, w, solver, max_passes=1)
        if first_ms is None:
            first_ms = 1000 * (time.perf_counter() - start) / stats["windows"]
        for key in total:
            total[key] += stats[key]
        if stats["gain"] <= 0 or (limit_ms is not None and first_ms > limit_ms):
            break
    return tour, total, first_ms


def main():
    parser = argparse.ArgumentParser(description="Phase 3: window size vs. solver.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--solver", choices=["model", "lkh"], required=True)
    parser.add_argument("--run", default=None, help="run folder in results/runs (solver model)")
    parser.add_argument("--widths", type=int, nargs="+", default=[20, 50, 100, 200, 300])
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--lkh-limit-ms", type=float, default=5000.0)
    parser.add_argument("--out-root", default="results/phase3_wsize")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    folder = args.run if args.solver == "model" else "lkh"
    out_dir = os.path.join(args.out_root, folder)
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)

    heatmaps, epoch, train_set = None, None, None
    if args.solver == "model":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            limit_gpu_memory(device)
        checkpoint = torch.load(os.path.join("results/runs", args.run, "best.pt"), map_location=device)
        model = build_model(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        heatmaps, epoch, train_set = Heatmaps(model, device), checkpoint["epoch"], checkpoint["config"]["train"]
        solver = make_solver("beam", heatmaps, 10)
    else:
        solver = make_solver("lkh")
    method = "beam10" if args.solver == "model" else "lkh"
    limit = args.lkh_limit_ms if args.solver == "lkh" else None

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    print(f"{name}: n = {n}, {num} instances, {folder}, widths {args.widths}", flush=True)
    limited = {}                     # w -> True if LKH is limited to 1 pass (decided on the first instance)
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
        for w in args.widths:
            key = f"{method}_w{w}"
            if heatmaps:
                heatmaps.batch_size = batch_size(w)
            model_before = heatmaps.seconds if heatmaps else 0.0
            start = time.perf_counter()
            passes = 1 if limited.get(w) else args.max_passes
            tour, stats, first_ms = refine_passes(coords, t0, w, solver, passes, limit if i == 0 else None)
            if i == 0 and limit is not None and first_ms > limit and args.max_passes > 1:
                limited[w] = True
                print(f"  LKH w = {w}: {first_ms:.0f} ms per window on the first pass > {limit:.0f} ms: "
                      f"1 pass on every instance", flush=True)
            add(f"{key}__ms", 1000 * (time.perf_counter() - start))
            add(f"{key}__ms_model", 1000 * ((heatmaps.seconds if heatmaps else 0.0) - model_before))
            verify_tour(tour, n)
            g = optimality_gap(tour_length(coords, tour), opt)
            add(f"{key}__gap", g)
            add(f"{key}__gain", g0 - g)
            for f in ("windows", "improved", "passes"):
                add(f"{key}__{f}", stats[f])
            add(f"{key}__limited", float(limited.get(w, False)))
        print(f"  {i + 1}/{num} instances", flush=True)

    per_inst = {key: np.array(v, dtype=float) for key, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "solver": args.solver, "run": args.run, "epoch": epoch, "train_set": train_set,
                        "dataset": name, "n": n, "num": num, "two_opt_k": k, "widths": args.widths,
                        "max_passes": args.max_passes, "lkh_limit_ms": limit,
                        "limited_widths": sorted(limited)},
               "initial": {"gap_mean": float(per_inst["initial__gap"].mean())}}
    for w in args.widths:
        key = f"{method}_w{w}"
        windows = per_inst[f"{key}__windows"].sum()
        summary[key] = {"gap_mean": float(per_inst[f"{key}__gap"].mean()),
                        "gain_mean": float(per_inst[f"{key}__gain"].mean()),
                        "improved_frac": float(per_inst[f"{key}__improved"].sum() / windows),
                        "ms_mean": float(per_inst[f"{key}__ms"].mean()),
                        "ms_per_window": float(per_inst[f"{key}__ms"].sum() / windows),
                        "limited_instances": int(per_inst[f"{key}__limited"].sum())}
        s = summary[key]
        print(f"  {key:<12} gap {s['gap_mean']:6.2f}%  gain {s['gain_mean']:5.2f} pp  improved {100 * s['improved_frac']:5.1f}%  "
              f"{s['ms_mean']:10.1f} ms  {s['ms_per_window']:8.1f} ms/window  limited {s['limited_instances']}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
