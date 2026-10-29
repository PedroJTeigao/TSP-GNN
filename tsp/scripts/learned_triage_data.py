"""Learned triage, step 1: per-window features and labels from the window logs of the triage experiment.

For each window of a log (scripts/triage_log.py: first pass on the greedy edge + 2-opt tour, offsets 0 and w/2),
rebuilds the window from (instance, start, size) on the same deterministic initial tour, runs the TAGCN K=3 of the
log's seed on it again (for the edge probabilities) and computes tsp_gnn.learned_triage.FEATURES, plus the extras
log10_n and logp and the labels. Check: the recomputed disagreement must equal the logged one (same windows).

Writes results/phase3_learned_triage/features/tagcn_k3_seed<s>/<dataset>.npz (one row per window, columns =
features, extras, labels, instance, offset, start, size) and .json (summary).

Example:
    python scripts/learned_triage_data.py --log results/phase3_triage/log/tagcn_k3_seed0/tsp200_test.npz \
        --data data/tsp200_test.npz --seed 0
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

from tsp_gnn import learned_triage as L
from tsp_gnn.data import load_dataset
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour

from triage_log import load_heatmaps


def main():
    parser = argparse.ArgumentParser(description="Learned triage: features and labels per window.")
    parser.add_argument("--log", required=True, help="window log .npz of scripts/triage_log.py")
    parser.add_argument("--data", required=True, help="the test set the log was made on")
    parser.add_argument("--seed", type=int, required=True, help="TAGCN K=3 seed of the log")
    parser.add_argument("--out-root", default="results/phase3_learned_triage/features")
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = os.path.join(args.out_root, f"tagcn_k3_seed{args.seed}")
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    heatmaps, _ = load_heatmaps(args.seed)
    log = dict(np.load(args.log))
    d = load_dataset(args.data)
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    inst = log["instance"].astype(int)
    rows = {f: np.zeros(len(inst)) for f in L.FEATURES}
    for i in np.unique(inst):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)      # as in triage_log.py
        tour = initial_tour(coords, k)
        ctx = L.instance_context(coords)
        where = np.flatnonzero(inst == i)
        cities = [tour[(int(log["start"][j]) + np.arange(int(log["size"][j]))) % n] for j in where]
        heats = heatmaps([coords[c] for c in cities])
        for j, c, h in zip(where, cities, heats):
            for f, v in L.window_features(ctx, coords, c, h, int(log["start"][j])).items():
                rows[f][j] = v
        if (i + 1) % 10 == 0:
            print(f"  instance {i + 1}", flush=True)
    diff = float(np.abs(rows["disagreement"] - log["disagreement"]).max())
    assert diff < 1e-4, f"recomputed disagreement differs from the log by {diff}: not the same windows"
    improves, gain_rel = L.labels(log["cur_len"], log["lkh_len"])
    out = {**rows, "log10_n": np.full(len(inst), np.log10(n)), "logp": log["logp"],
           "improves": improves, "gain_rel": gain_rel, "instance": inst,
           **{f: log[f] for f in ("offset", "start", "size", "ms_net")}}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "seed": args.seed, "dataset": name, "n": n, "instances": int(len(np.unique(inst))),
                        "log": args.log},
               "windows": int(len(inst)), "improves_frac": float(improves.mean()),
               "mean_gain_rel_pct": float(gain_rel.mean()), "max_disagreement_diff": diff,
               "feature_means": {f: float(rows[f].mean()) for f in L.FEATURES}}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **out)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_dir}/{name}.json")


if __name__ == "__main__":
    main()
