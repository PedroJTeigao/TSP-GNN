"""Fast triage, version B: window features with the edge probabilities taken from ONE network pass on the whole
instance (tsp_gnn.fast_triage.GlobalScorer), for retraining the learned signal on them.

Starts from the step-1 features of the learned triage (results/phase3_learned_triage/features/...: same windows,
labels and geometric features) and replaces the probability columns (p_mean, p_min, p_max, disagreement) by their
global-p values on the same initial-tour windows. Records the share of tour edges outside the global graph (p = 0).

Writes results/phase3_fast_triage/global_features/tagcn_k3_seed<s>/<dataset>.{npz,json}.

Example:
    python scripts/fast_triage_global_features.py --data data/tsp200_test.npz --seed 0
"""

import argparse
import json
import os

import numpy as np
import torch

from tsp_gnn import heatcand
from tsp_gnn.data import load_dataset
from tsp_gnn.fast_triage import GlobalScorer, batch_features
from tsp_gnn.learned_triage import instance_context
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour

P_COLS = ("p_mean", "p_min", "p_max", "disagreement")


def main():
    parser = argparse.ArgumentParser(description="Fast triage B: window features from the global heatmap.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features-root", default="results/phase3_learned_triage/features")
    parser.add_argument("--out-root", default="results/phase3_fast_triage/global_features")
    args = parser.parse_args()
    name = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = os.path.join(args.out_root, f"tagcn_k3_seed{args.seed}")
    if os.path.exists(os.path.join(out_dir, f"{name}.json")):
        print(f"{out_dir}/{name}.json already exists: skipped")
        return
    torch.set_num_threads(2)
    model, device = heatcand.load_model(f"tagcn_k3_seed{args.seed}")
    old = dict(np.load(os.path.join(args.features_root, f"tagcn_k3_seed{args.seed}", f"{name}.npz")))
    d = load_dataset(args.data)
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    inst = old["instance"].astype(int)
    new = {c: np.zeros(len(inst)) for c in P_COLS}
    geo_diff = 0.0
    scorer = GlobalScorer(model, device)
    for i in np.unique(inst):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)
        tour = initial_tour(coords, k)
        scorer.prepare(coords)
        where = np.flatnonzero(inst == i)
        for size in np.unique(old["size"][where]).astype(int):
            rows = where[old["size"][where] == size]
            starts = old["start"][rows].astype(int)
            cities = tour[(starts[:, None] + np.arange(size)[None, :]) % n]
            p_seg = scorer.edge_p(cities[:, :-1], cities[:, 1:])
            f = batch_features(instance_context(coords), coords, cities, p_seg, starts)
            for c in P_COLS:
                new[c][rows] = f[c]
            geo_diff = max(geo_diff, float(np.abs(f["seg_len_nn"] - old["seg_len_nn"][rows]).max()))
    assert geo_diff < 1e-9, f"geometric features differ from step 1 by {geo_diff}: not the same windows"
    out = {**old, **new}
    missing = scorer.missing[0] / max(scorer.missing[1], 1)
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "seed": args.seed, "dataset": name, "n": n,
                        "global_k": k},
               "windows": int(len(inst)), "tour_edges_outside_global_graph": missing,
               "mean_p_window": float(old["p_mean"].mean()), "mean_p_global": float(new["p_mean"].mean())}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **out)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
