"""Per-window record for the triage experiment (phase 3, src/tsp_gnn/triage.py) on one test set.

Starts from the greedy edge + 2-opt tour (window.initial_tour, k = min(0.5 n, 100)) and records every
window of w = 20 cities at offsets 0 and w/2 of that tour (first pass, nothing replaced): disagreement,
beam B=10 confidence (logp, margin), current / beam / LKH path lengths and times (TAGCN K=3, best.pt of
--seed). Writes <out-root>/tagcn_k3_seed<s>/<dataset>.npz (one row per window, "instance" says which)
and .json (summary, written last).

Example:
    python scripts/triage_log.py --data data/tsp200_test.npz --seed 0
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

from tsp_gnn.data import load_dataset
from tsp_gnn.hier import Heatmaps
from tsp_gnn.models import build_model
from tsp_gnn.triage import average_precision, targets, window_log
from tsp_gnn.utils import git_info, limit_gpu_memory
from tsp_gnn.window import initial_tour


def load_heatmaps(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_gpu_memory(device)
    checkpoint = torch.load(os.path.join("results/runs", f"tagcn_k3_seed{seed}", "best.pt"), map_location=device)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return Heatmaps(model, device), checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 triage: per-window signals and gains.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True, help="TAGCN K=3 seed")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--out-root", default="results/phase3_triage/log")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    out_dir = os.path.join(args.out_root, f"tagcn_k3_seed{args.seed}")
    name = os.path.splitext(os.path.basename(args.data))[0]
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    heatmaps, epoch = load_heatmaps(args.seed)

    d = load_dataset(args.data)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    print(f"{name}: n = {n}, {num} instances, seed {args.seed}, w = {args.width}", flush=True)
    logs = []
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)   # as in the scale experiment
        log = window_log(coords, initial_tour(coords, k), args.width, heatmaps)
        log["instance"] = np.full(len(log["start"]), i, dtype=float)
        logs.append(log)
        if (i + 1) % 10 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num} instances", flush=True)
    log = {key: np.concatenate([x[key] for x in logs]) for key in logs[0]}

    lkh_improves, catches = targets(log)
    summary = {"windows": int(len(log["start"])),
               "lkh_improves_frac": float(lkh_improves.mean()),
               "model_improves_frac": float((log["cur_len"] - log["model_len"] > 1e-9).mean()),
               "catches_frac_of_lkh_improves": float(catches[lkh_improves].mean()) if lkh_improves.any() else None,
               "ap_disagreement_i": average_precision(log["disagreement"], lkh_improves),
               "ms_per_window": {f: float(log[f].mean()) for f in ("ms_net", "ms_beam", "ms_lkh")}}
    print(json.dumps(summary, indent=2), flush=True)
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "seed": args.seed, "epoch": epoch, "dataset": name, "n": n, "num": num,
                        "two_opt_k": k, "width": args.width, "beam_width": 10}, **summary}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **log)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
