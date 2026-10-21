"""Evaluate the hierarchical TSP (src/tsp_gnn/hier.py) with one trained model on one test set (phase 3).

Nothing is retrained. For each instance and partition (kmeans, kdtree) the hierarchy is built once;
then every variant {attract distance, heatmap} × {cands 1, 3} × {greedy, beam10 paths} gives a tour,
which is improved with the neighbour-list 2-opt of the scale experiment (k = min(0.5 n, 100), the
k of the greedy edge + 2-opt baseline it is compared with).

Writes <out-dir>/<dataset>.json (summary, written last) and .npz with per-instance arrays
"<partition>/<attract>/c<c>/<decoder>__<field>", fields:
    before, after                 gap (%) before / after the final 2-opt
    ms_top, ms_stitch, ms_paths   time of each step (ms); ms_model = network time inside them
    ms_2opt                       final 2-opt
    boundary_edges, boundary_wrong, interior_edges, interior_wrong
                                  tour edges (before 2-opt) between / inside level-0 groups, and how
                                  many of them are not in the LKH tour
    moved                         cities whose neighbours in other level-0 groups changed with the 2-opt
    c_used                        candidates per link actually needed (> c if the constraint forced it)
and per partition "<partition>__ms_partition" plus the hierarchy in the .json.

Example:
    python scripts/hier_eval.py --data data/tsp1000_test.npz --seed 0
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
from tsp_gnn.hier import (ATTRACTS, PARTITIONS, Hierarchy, Heatmaps, boundary_stats, hierarchical_tour,
                          moved_between_groups)
from tsp_gnn.models import build_model
from tsp_gnn.scale import neighbor_lists, two_opt_neighbors
from tsp_gnn.utils import git_info, limit_gpu_memory

DECODERS = ("greedy", "beam10")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: hierarchical TSP with a TSP-20 model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True, help="seed of the TAGCN K=3 run (best.pt)")
    parser.add_argument("--run", default=None, help="default: tagcn_k3_seed<seed>")
    parser.add_argument("--out-dir", default=None, help="default: results/phase3_hier/<run>")
    parser.add_argument("--partitions", nargs="+", default=list(PARTITIONS))
    parser.add_argument("--attracts", nargs="+", default=list(ATTRACTS))
    parser.add_argument("--cands", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--decoders", nargs="+", default=list(DECODERS))
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--two-opt-limit", type=float, default=600.0)
    args = parser.parse_args()
    run = args.run or f"tagcn_k3_seed{args.seed}"
    out_dir = args.out_dir or os.path.join("results/phase3_hier", run)
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)
    checkpoint = torch.load(os.path.join("results/runs", run, "best.pt"), map_location=device)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k2opt = min(n - 1, round(0.5 * n), 100)
    variants = [(p, a, c, dec) for p in args.partitions for a in args.attracts for c in args.cands
                for dec in args.decoders]
    print(f"{name}: n = {n}, {num} instances, {run}, {len(variants)} variants, 2-opt k = {k2opt}", flush=True)

    out, hier_info = {}, {p: [] for p in args.partitions}
    add = lambda key, v: out.setdefault(key, []).append(v)
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)   # what the models see (as in phase 2)
        opt, ref = float(d["lengths"][i]), d["tours"][i]
        nbrs = neighbor_lists(coords, k2opt)
        for part in args.partitions:
            start = time.perf_counter()
            H = Hierarchy(coords, part, seed=0)
            add(f"{part}__ms_partition", 1000 * (time.perf_counter() - start))
            hier_info[part].append({"points_per_level": [len(p) for p in H.points],
                                    "group_sizes": [H.group_sizes(l).tolist() for l in range(1, H.top + 1)]})
            for p, a, c, dec in variants:
                if p != part:
                    continue
                key = f"{p}/{a}/c{c}/{dec}"
                heatmaps = Heatmaps(model, device)
                tour, times, stats = hierarchical_tour(H, heatmaps, a, c, "greedy" if dec == "greedy" else "beam", 10)
                verify_tour(tour, n)
                start = time.perf_counter()
                improved, finished = two_opt_neighbors(coords, tour, nbrs, time_limit=args.two_opt_limit)
                add(f"{key}__ms_2opt", 1000 * (time.perf_counter() - start))
                verify_tour(improved, n)
                add(f"{key}__before", optimality_gap(tour_length(coords, tour), opt))
                add(f"{key}__after", optimality_gap(tour_length(coords, improved), opt) if finished else np.nan)
                for step in ("top", "stitch", "paths", "model"):
                    add(f"{key}__ms_{step}", 1000 * times[step])
                for field, v in boundary_stats(H, tour, ref).items():
                    add(f"{key}__{field}", v)
                add(f"{key}__moved", moved_between_groups(H, tour, improved))
                add(f"{key}__c_used", stats["c_used_max"])
        if (i + 1) % 10 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num} instances", flush=True)

    per_inst = {k: np.array(v, dtype=float) for k, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "run": run, "epoch": checkpoint["epoch"], "dataset": name, "n": n, "num": num,
                        "two_opt_k": k2opt, "two_opt_limit_s": args.two_opt_limit, "hierarchy": hier_info}}
    for p, a, c, dec in variants:
        key = f"{p}/{a}/c{c}/{dec}"
        entry = {}
        for field in ("before", "after"):
            v = per_inst[f"{key}__{field}"]
            entry[f"gap_{field}"] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else None
        entry["timeouts"] = int(np.isnan(per_inst[f"{key}__after"]).sum())
        entry["ms_total"] = float(sum(per_inst[f"{key}__ms_{s}"].mean() for s in ("top", "stitch", "paths", "2opt"))
                                  + per_inst[f"{p}__ms_partition"].mean())
        summary[key] = entry
        print(f"  {key:<30} before {entry['gap_before']:6.2f}%  after {entry['gap_after']:6.2f}%  "
              f"{entry['ms_total']:9.1f} ms", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
