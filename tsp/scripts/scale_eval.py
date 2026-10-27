"""Evaluate trained models on one (possibly very large) test set, with timings (phase 2, scale experiment).

Nothing is retrained. For each instance of --data:
    1) k-NN graph with a KD-tree (scale.knn_graph); k from --k-mode:
         k50        k = 50 (never more than n-1)
         khalf100   k = min(round(0.5 n), 100)
         khalf      k = round(0.5 n) (uncapped)
         k02n       k = round(0.2 n) (Joshi et al. 2021, "fixed diameter" 20 %)
         k10        k = 10 (the training graphs of TSP-20)
    2) the network, one instance at a time (GPU; CPU if it does not fit in the 5.5 GiB cap)
    3) decoding with the sparse decoders (same tours as the dense ones, see tests/test_scale.py):
         greedy always; beam B=10 if n <= --beam10-max-n; beam B=100 if n is in --beam100-n
    4) 2-opt with neighbour lists (the same k neighbours), applied to each decoder's tour, stopped
       after --two-opt-limit seconds (then the "+2opt" gap is NaN and the timeout is recorded).
       With --two-opt-k-mode the 2-opt lists use that k-mode instead (to compare graphs with the same 2-opt).
Steps 3-4 run in --workers CPU processes while the GPU computes the next model.

Writes, for each run, results/phase2_scale/<run>/<k-mode>/<dataset>.json (summary) and .npz
(per-instance gaps "<method>" and times in ms "<method>__ms", plus "graph__ms" and "gnn__ms";
the "+2opt__ms" time is the 2-opt alone). The .json is written last: it marks the result as done.

Example:
    python scripts/scale_eval.py --data data/tsp200_test.npz --k-mode k50 --runs gcn_seed0 gat_seed0
"""

import argparse
import datetime
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Data

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, verify_tour
from tsp_gnn.models import build_model
from tsp_gnn.scale import (beam_search_sparse, edge_pairs, greedy_sparse, knn_graph, neighbor_lists,
                           two_opt_neighbors)
from tsp_gnn.utils import git_info, limit_gpu_memory

K_MODES = ("k50", "khalf100", "khalf", "k02n", "k10")


def k_of(mode, n):
    """Number of neighbours of the test graph for a k-mode (never more than n-1)."""
    k = {"k50": 50, "khalf100": min(round(0.5 * n), 100), "khalf": round(0.5 * n), "k02n": round(0.2 * n),
         "k10": 10}[mode]
    return min(k, n - 1)


def decode_job(job):
    """Decode one instance with every decoder, then 2-opt each tour. Returns {key: value}."""
    torch.set_num_threads(1)
    coords, opt, edge_index, probs, nbrs, widths, limit = job
    n = len(coords)
    pairs = edge_pairs(edge_index, probs, n)
    decoders = {"greedy": lambda: greedy_sparse(*pairs, coords)}
    for b in widths:
        decoders[f"beam{b}"] = lambda b=b: beam_search_sparse(*pairs, coords, b)
    out = {}
    for name, fn in decoders.items():
        start = time.perf_counter()
        tour = verify_tour(fn(), n)
        out[f"{name}__ms"] = 1000 * (time.perf_counter() - start)
        out[name] = optimality_gap(tour_length(coords, tour), opt)
        start = time.perf_counter()
        improved, finished = two_opt_neighbors(coords, tour, nbrs, time_limit=limit)
        out[f"{name}+2opt__ms"] = 1000 * (time.perf_counter() - start)
        verify_tour(improved, n)
        out[f"{name}+2opt"] = optimality_gap(tour_length(coords, improved), opt) if finished else np.nan
        out[f"{name}+2opt__timeout"] = float(not finished)
    return out


@torch.no_grad()
def run_network(model, graphs, device):
    """Edge probabilities of each graph (one at a time) + per-instance time (ms), device used, peak GiB.

    If a graph does not fit on the GPU (out of memory), that graph and every later one run on the CPU.
    """
    model.eval()
    probs, ms, used = [], [], device.type
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    on = lambda g, dev: Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr).to(dev)  # g stays on the CPU
    if device.type == "cuda":
        model(on(graphs[0], device))                # warm-up (CUDA kernels), not timed
    for g in graphs:
        start = time.perf_counter()
        try:
            p = torch.sigmoid(model(on(g, torch.device(used)))).cpu()
            if used == "cuda":
                torch.cuda.synchronize(device)
        except torch.OutOfMemoryError:
            print(f"    out of GPU memory at n = {g.num_nodes}, {g.num_edges} edges: moving to the CPU", flush=True)
            torch.cuda.empty_cache()
            used = "cpu"
            model = model.cpu()
            start = time.perf_counter()
            p = torch.sigmoid(model(on(g, torch.device("cpu")))).cpu()
        ms.append(1000 * (time.perf_counter() - start))
        probs.append(p)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    return probs, np.array(ms), used, peak


def summarize(values):
    """Mean / std / max of a per-instance array (NaNs = 2-opt timeouts are left out)."""
    v = np.asarray(values, dtype=float)
    ok = v[~np.isnan(v)]
    if len(ok) == 0:
        return {"gap_mean": None, "gap_std": None, "gap_max": None, "num_valid": 0}
    return {"gap_mean": float(ok.mean()), "gap_std": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0,
            "gap_max": float(ok.max()), "num_valid": int(len(ok))}


def main():
    parser = argparse.ArgumentParser(description="Scale experiment: evaluate trained models on one test set.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--k-mode", choices=K_MODES, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="folders in results/runs/ (best.pt is used)")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--out-root", default="results/phase2_scale")
    parser.add_argument("--workers", type=int, default=4, help="CPU processes for decoding + 2-opt")
    parser.add_argument("--beam10-max-n", type=int, default=1000)
    parser.add_argument("--beam100-n", type=int, nargs="*", default=[200])
    parser.add_argument("--two-opt-limit", type=float, default=600.0, help="seconds per 2-opt call")
    parser.add_argument("--two-opt-k-mode", choices=K_MODES, default=None,
                        help="k-mode of the 2-opt neighbour lists (default: --k-mode)")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.data))[0]
    d = load_dataset(args.data)
    n, num = int(d["coords"].shape[1]), len(d["coords"])
    k = k_of(args.k_mode, n)
    k2 = k_of(args.two_opt_k_mode, n) if args.two_opt_k_mode else k
    widths = ([10] if n <= args.beam10_max_n else []) + ([100] if n in args.beam100_n else [])
    todo = [r for r in args.runs if not os.path.exists(os.path.join(args.out_root, r, args.k_mode, f"{name}.json"))]
    print(f"{name}: n = {n}, {num} instances, k = {k} ({args.k_mode}), beam widths {widths}, "
          f"{len(todo)}/{len(args.runs)} runs to do", flush=True)
    if not todo:
        return

    # 1) graphs (built once, shared by every model)
    graphs, lists, graph_ms = [], [], []
    for c in d["coords"]:
        start = time.perf_counter()
        g, nb = knn_graph(c, k, return_lists=True)
        graph_ms.append(1000 * (time.perf_counter() - start))
        if k2 != k:
            nb = neighbor_lists(torch.as_tensor(c, dtype=torch.float32).double().numpy(), k2)   # as knn_graph
        graphs.append(g); lists.append(nb)
    graph_ms = np.array(graph_ms)
    coords = [g.x.numpy().astype(np.float64) for g in graphs]   # as in the phase 1 decoders
    print(f"  graphs: {graph_ms.mean():.1f} ms per instance, {np.mean([g.num_edges for g in graphs]):.0f} "
          f"edges on average", flush=True)

    commit, dirty = git_info()
    pending = []
    with Pool(args.workers) as pool:              # created before CUDA starts: the workers only use the CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            limit_gpu_memory(device)
        # 2) network on the GPU, run after run; 3-4) decoding of each run starts right away on the CPU
        for run in todo:
            checkpoint = torch.load(os.path.join("results/runs", run, args.checkpoint), map_location=device)
            cfg = checkpoint["config"]
            assert cfg.get("edge_norm", "none") == "none", "the scale experiment uses raw distances (phase 1)"
            model = build_model(cfg).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            probs, gnn_ms, used, peak = run_network(model, graphs, device)
            del model
            print(f"  {run}: network {gnn_ms.mean():.1f} ms per instance on {used}, peak GPU {peak:.2f} GiB",
                  flush=True)
            jobs = [(coords[i], float(d["lengths"][i]), graphs[i].edge_index.numpy(), probs[i].numpy(), lists[i],
                     widths, args.two_opt_limit) for i in range(num)]
            pending.append((run, checkpoint["epoch"], cfg, gnn_ms, used, peak, pool.map_async(decode_job, jobs)))

        for run, epoch, cfg, gnn_ms, used, peak, async_result in pending:
            res = async_result.get()
            per_inst = {key: np.array([r[key] for r in res]) for key in res[0]}
            per_inst["graph__ms"], per_inst["gnn__ms"] = graph_ms, gnn_ms
            methods = [m for m in res[0] if "__" not in m]
            summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                                "run": run, "checkpoint": args.checkpoint, "epoch": epoch, "model": cfg["model"],
                                "dataset": name, "n": n, "num": num, "k_mode": args.k_mode, "k": k,
                                "beam_widths": widths, "two_opt": "neighbour lists (k), don't-look bits", "two_opt_k": k2,
                                "two_opt_limit_s": args.two_opt_limit, "gnn_device": used, "peak_gpu_gib": peak,
                                "graph_ms_mean": float(graph_ms.mean()), "gnn_ms_mean": float(gnn_ms.mean())}}
            for m in methods:
                summary[m] = {**summarize(per_inst[m]), "ms_mean": float(per_inst[f"{m}__ms"].mean())}
                if m.endswith("+2opt"):
                    summary[m]["timeouts"] = int(per_inst[f"{m}__timeout"].sum())
            out_dir = os.path.join(args.out_root, run, args.k_mode)
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **per_inst)
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump(summary, f, indent=2)
            cells = "  ".join(f"{m} {summary[m]['gap_mean']:.2f}%" if summary[m]["gap_mean"] is not None
                              else f"{m} timeout" for m in methods)
            print(f"  {run}: {cells}", flush=True)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {args.out_root}/<run>/{args.k_mode}/")


if __name__ == "__main__":
    main()
