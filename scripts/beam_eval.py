"""Evaluate a trained model with greedy and beam search decoding (phase 2, beam search step).

The network runs once per test set; the same probabilities are then decoded with greedy and with
beam search for each beam width, each with and without 2-opt (applied to the selected tour).
Nothing is retrained and the checkpoint's own folder is not touched: results go to --out-dir.

Writes to --out-dir:
    eval.json               mean/std/max gap and mean decoding time per dataset and method
    eval_per_instance.npz   per-instance gaps "<dataset>__<method>" and decoding times
                            "<dataset>__<method>__ms" (methods: greedy, beam<B>, each also "+2opt")

Example:
    python scripts/beam_eval.py --run-dir results/runs/gated_seed0 --k-frac 0.5 \
        --beam-widths 10 100 --out-dir results/phase2_beam/phase1/gated_seed0
"""

import argparse
import datetime
import json
import os
import time

import numpy as np
import torch

from tsp_gnn.data import tour_length
from tsp_gnn.decode import (beam_search_decode, edge_heatmap, greedy_decode, optimality_gap, two_opt,
                            verify_tour)
from tsp_gnn.graph import load_graphs
from tsp_gnn.models import build_model
from tsp_gnn.train import predict_probs
from tsp_gnn.utils import git_info, limit_gpu_memory


def timed(fn, *args):
    """(result of fn(*args), time taken in ms)."""
    start = time.perf_counter()
    out = fn(*args)
    return out, 1000 * (time.perf_counter() - start)


def decode_dataset(graphs, probs, beam_widths):
    """Per-instance gaps and decoding times of every method. Returns {key: array}.

    Times: "<method>__ms" is the decoder alone; "<method>+2opt__ms" is decoder + 2-opt.
    """
    decoders = {"greedy": lambda h, g, c: greedy_decode(h, g, c)}
    for b in beam_widths:
        decoders[f"beam{b}"] = lambda h, g, c, b=b: beam_search_decode(h, g, c, b)
    out = {k: [] for name in decoders for k in (name, f"{name}__ms", f"{name}+2opt", f"{name}+2opt__ms")}
    for i, (data, p) in enumerate(zip(graphs, probs)):
        coords = data.x.numpy().astype(np.float64)
        n, opt = len(coords), float(data.opt_len)
        heat, in_graph = edge_heatmap(data.edge_index, p, n)
        for name, decoder in decoders.items():
            tour, ms = timed(decoder, heat, in_graph, coords)
            improved, ms_2opt = timed(two_opt, coords, tour)
            for key, t, t_ms in ((name, tour, ms), (f"{name}+2opt", improved, ms + ms_2opt)):
                verify_tour(t, n)                 # raises on an invalid tour
                out[key].append(optimality_gap(tour_length(coords, t), opt))
                out[f"{key}__ms"].append(t_ms)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(graphs)} instances", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description="Greedy vs. beam search decoding of a trained EdgeGNN.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt", help="best.pt, best_loss.pt or last.pt")
    parser.add_argument("--test", nargs="+",
                        default=["data/tsp20_test.npz", "data/tsp50_test.npz", "data/tsp100_test.npz"])
    parser.add_argument("--out-dir", required=True, help="new folder for eval.json and eval_per_instance.npz")
    parser.add_argument("--k-frac", type=float, default=None,
                        help="test graphs with k = round(k_frac × n) (default: as in training)")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--batch-size", type=int, default=64, help="graphs per forward pass")
    parser.add_argument("--max-instances", type=int, default=None, help="only the first N of each test set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)  # shared GPU: same 5.5 GiB cap as training
    checkpoint = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device)
    cfg = checkpoint["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    edge_norm = cfg.get("edge_norm", "none")
    k_frac = args.k_frac if args.k_frac is not None else cfg.get("k_frac")
    print(f"{args.run_dir}/{args.checkpoint}: {cfg['model']}, norm {cfg.get('norm', 'batchnorm')}, "
          f"epoch {checkpoint['epoch']}  |  k_frac {k_frac}, beam widths {args.beam_widths}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary, per_instance = {}, {}
    for path in args.test:
        name = os.path.splitext(os.path.basename(path))[0]
        graphs = load_graphs(path, cfg["k"], edge_norm, k_frac)[:args.max_instances]
        probs, t_inf = predict_probs(model, graphs, args.batch_size)
        print(f"{name}: {len(graphs)} instances", flush=True)
        res = decode_dataset(graphs, probs, args.beam_widths)
        entry = {"n": graphs[0].num_nodes, "num": len(graphs), "inference_ms_per_inst": 1000 * t_inf / len(graphs)}
        for key, values in res.items():
            per_instance[f"{name}__{key}"] = values
            if not key.endswith("__ms"):
                entry[key] = {"gap_mean": float(values.mean()), "gap_std": float(values.std(ddof=1)),
                              "gap_max": float(values.max()), "decode_ms_per_inst": float(res[f"{key}__ms"].mean())}
                print(f"  {key:<14}{values.mean():>7.2f}%  {res[key + '__ms'].mean():>9.1f} ms/inst", flush=True)
        summary[name] = entry

    commit, dirty = git_info()
    result = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                       "run_dir": args.run_dir, "checkpoint": args.checkpoint, "epoch": checkpoint["epoch"],
                       "config": cfg, "k_frac": k_frac, "beam_widths": args.beam_widths,
                       "batch_size": args.batch_size},
              **summary}
    # npz first, eval.json last: eval.json marks the folder as complete for the runner.
    np.savez_compressed(os.path.join(args.out_dir, "eval_per_instance.npz"), **per_instance)
    with open(os.path.join(args.out_dir, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {args.out_dir}")


if __name__ == "__main__":
    main()
