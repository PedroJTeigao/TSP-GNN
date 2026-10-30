"""Triage with larger windows, step 1: window logs for one (set, w), and the features per TAGCN seed.

--part lkh       LKH on every window of the first pass (tsp_gnn.wsize_log.lkh_log); writes
                 results/phase3_triage_wsize/log/w<w>/<dataset>.{npz,json} (LKH time per window, share that improves).
--part features  features of the same windows for one seed (wsize_log.seed_features) plus labels; writes
                 results/phase3_triage_wsize/features/w<w>/tagcn_k3_seed<s>/<dataset>.{npz,json} (the format of the
                 learned-triage features, without the beam's logp).

Example:
    python scripts/wsize_log.py --part lkh --data data/tsp200_test.npz --width 50
    python scripts/wsize_log.py --part features --data data/tsp200_test.npz --width 50 --seed 0
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

from tsp_gnn import heatcand
from tsp_gnn import learned_triage as L
from tsp_gnn import wsize_log as W
from tsp_gnn.data import load_dataset
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour

ROOT = "results/phase3_triage_wsize"


def main():
    parser = argparse.ArgumentParser(description="Window logs (LKH) and features for a window size w.")
    parser.add_argument("--part", choices=("lkh", "features"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None, help="TAGCN K=3 seed (features)")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    log_dir = os.path.join(ROOT, "log", f"w{args.width}")
    out_dir = log_dir if args.part == "lkh" else os.path.join(ROOT, "features", f"w{args.width}",
                                                               f"tagcn_k3_seed{args.seed}")
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    d = load_dataset(args.data)
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    num = len(d["coords"]) if args.max_instances is None else min(args.max_instances, len(d["coords"]))
    commit, dirty = git_info()
    meta = {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(), "dataset": name,
            "n": n, "num": num, "width": args.width, "two_opt_k": k}

    if args.part == "lkh":
        logs = []
        for i in range(num):
            coords = d["coords"][i].astype(np.float32).astype(np.float64)
            log = W.lkh_log(coords, initial_tour(coords, k), args.width)
            log["instance"] = np.full(len(log["start"]), i, dtype=float)
            logs.append(log)
            print(f"  {i + 1}/{num} instances", flush=True)
        out = {key: np.concatenate([x[key] for x in logs]) for key in logs[0]}
        improves, gain = L.labels(out["cur_len"], out["lkh_len"])
        summary = {"meta": meta, "windows": int(len(improves)), "improves_frac": float(improves.mean()),
                   "mean_gain_rel_pct": float(gain.mean()), "ms_lkh_per_window": float(out["ms_lkh"].mean()),
                   "ms_lkh_per_window_improves": float(out["ms_lkh"][improves].mean()) if improves.any() else None,
                   "ms_lkh_per_window_not": float(out["ms_lkh"][~improves].mean()) if (~improves).any() else None}
    else:
        log = dict(np.load(os.path.join(log_dir, f"{name}.npz")))
        model, device = heatcand.load_model(f"tagcn_k3_seed{args.seed}")
        inst = log["instance"].astype(int)
        cols = {f: np.zeros(len(inst)) for f in L.FEATURES}
        for i in np.unique(inst):
            coords = d["coords"][i].astype(np.float32).astype(np.float64)
            rows = np.flatnonzero(inst == i)
            f = W.seed_features(model, device, coords, initial_tour(coords, k), {c: log[c][rows] for c in log})
            for c in L.FEATURES:
                cols[c][rows] = f[c]
        improves, gain = L.labels(log["cur_len"], log["lkh_len"])
        out = {**cols, "log10_n": np.full(len(inst), np.log10(n)), "improves": improves, "gain_rel": gain,
               "instance": inst, **{c: log[c] for c in ("offset", "start", "size", "ms_lkh")}}
        meta["seed"] = args.seed
        meta["instances"] = int(len(np.unique(inst)))
        summary = {"meta": meta, "windows": int(len(inst)), "improves_frac": float(improves.mean()),
                   "mean_gain_rel_pct": float(gain.mean()),
                   "feature_means": {c: float(cols[c].mean()) for c in L.FEATURES}}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **out)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
