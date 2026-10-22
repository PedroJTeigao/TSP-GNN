"""Window refinement of the greedy edge + 2-opt tour on one test set (phase 3, src/tsp_gnn/window.py).

--solver model  TAGCN K=3 (best.pt of --seed), greedy and beam B=10 paths
--solver lkh    LKH on each window (upper bound of what a window can give; no model, no seed)
--solver 2opt   2-opt on each window alone (control without model)
for window sizes --widths (20, 30, 50), step w/2, at most 5 passes.

The initial tour is greedy edge + neighbour-list 2-opt with k = min(0.5 n, 100): the same tour as the
baseline of the scale experiment (checked in the report).
Writes <out-dir>/<dataset>.json (summary, written last) and .npz with per-instance arrays
"initial__gap", "initial__ms" and "<method>_w<w>__<field>", fields: gap (final), gain (initial − final, pp),
ms (refinement time), ms_model (network time inside it), windows, improved, passes.

Example:
    python scripts/window_eval.py --data data/tsp1000_test.npz --solver model --seed 0
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


def main():
    parser = argparse.ArgumentParser(description="Phase 3: window refinement of the greedy edge + 2-opt tour.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--solver", choices=["model", "lkh", "2opt"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="TAGCN K=3 seed (solver model)")
    parser.add_argument("--widths", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--out-root", default="results/phase3_window")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    folder = f"tagcn_k3_seed{args.seed}" if args.solver == "model" else args.solver
    out_dir = os.path.join(args.out_root, folder)
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)

    heatmaps, epoch = None, None
    if args.solver == "model":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            limit_gpu_memory(device)
        checkpoint = torch.load(os.path.join("results/runs", f"tagcn_k3_seed{args.seed}", "best.pt"), map_location=device)
        model = build_model(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        heatmaps, epoch = Heatmaps(model, device), checkpoint["epoch"]
        methods = {"greedy": make_solver("greedy", heatmaps), "beam10": make_solver("beam", heatmaps, 10)}
    else:
        methods = {args.solver: make_solver(args.solver)}

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    print(f"{name}: n = {n}, {num} instances, {folder}, widths {args.widths}", flush=True)
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
        for m, solver in methods.items():
            for w in args.widths:
                key = f"{m}_w{w}"
                model_before = heatmaps.seconds if heatmaps else 0.0
                start = time.perf_counter()
                tour, stats = refine(coords, t0, w, solver, args.max_passes)
                add(f"{key}__ms", 1000 * (time.perf_counter() - start))
                add(f"{key}__ms_model", 1000 * ((heatmaps.seconds if heatmaps else 0.0) - model_before))
                verify_tour(tour, n)
                g = optimality_gap(tour_length(coords, tour), opt)
                add(f"{key}__gap", g)
                add(f"{key}__gain", g0 - g)
                for f in ("windows", "improved", "passes"):
                    add(f"{key}__{f}", stats[f])
        if (i + 1) % 10 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num} instances", flush=True)

    per_inst = {key: np.array(v, dtype=float) for key, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "solver": args.solver, "seed": args.seed, "epoch": epoch, "dataset": name, "n": n,
                        "num": num, "two_opt_k": k, "widths": args.widths, "max_passes": args.max_passes},
               "initial": {"gap_mean": float(per_inst["initial__gap"].mean())}}
    for m in methods:
        for w in args.widths:
            key = f"{m}_w{w}"
            summary[key] = {"gap_mean": float(per_inst[f"{key}__gap"].mean()),
                            "gain_mean": float(per_inst[f"{key}__gain"].mean()),
                            "improved_frac": float(per_inst[f"{key}__improved"].sum() / per_inst[f"{key}__windows"].sum()),
                            "ms_mean": float(per_inst[f"{key}__ms"].mean())}
            s = summary[key]
            print(f"  {key:<12} gap {s['gap_mean']:6.2f}%  gain {s['gain_mean']:5.2f} pp  "
                  f"improved {100 * s['improved_frac']:5.1f}% of windows  {s['ms_mean']:9.1f} ms", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
