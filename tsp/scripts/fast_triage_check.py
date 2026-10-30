"""Fast triage: are the batched scores the same ranking as the per-window ones? And how long does each take?

On a sample (the first --instances instances of a test set, the 3 TAGCN seeds, the initial greedy + 2-opt tour,
both half-passes), compares with the current per-window scores (hier.Heatmaps + segment_disagreement, and
learned_triage.LearnedScore with the chosen model of the learned triage):
    A  fast_triage.WindowBatchScorer  (same windows, batched)       disagreement and learned (same model)
    B  fast_triage.GlobalScorer       (one pass on the whole graph)  disagreement, and learned retrained on global
                                                                    features (if --global-models has it)
Per (instance, half-pass): Spearman correlation, overlap of the top 20 % (triage.top_fraction, as the policy picks)
and the largest absolute difference; per instance, the scoring time of both half-passes (network included).

Writes results/phase3_fast_triage/check/<dataset>.json.

Example:
    python scripts/fast_triage_check.py --data results/phase3_benchmark/data/tsp1000_test.npz
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr

from tsp_gnn import fast_triage as F
from tsp_gnn import heatcand
from tsp_gnn.data import load_dataset
from tsp_gnn.hier import Heatmaps
from tsp_gnn.learned_triage import LearnedScore
from tsp_gnn.triage import segment_disagreement, top_fraction
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour, window_starts

SEEDS = (0, 1, 2)


def compare(a, b):
    ma, mb = top_fraction(a, 0.2), top_fraction(b, 0.2)
    return {"spearman": float(spearmanr(a, b)[0]), "top20_overlap": float((ma & mb).sum() / ma.sum()),
            "max_abs_diff": float(np.abs(a - b).max())}


def main():
    parser = argparse.ArgumentParser(description="Fast triage: equivalence and time of the batched scores.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--models", default="results/phase3_learned_triage/models")
    parser.add_argument("--global-models", default="results/phase3_fast_triage/global_models")
    parser.add_argument("--out-root", default="results/phase3_fast_triage/check")
    args = parser.parse_args()
    torch.set_num_threads(2)
    name = os.path.splitext(os.path.basename(args.data))[0]
    d = load_dataset(args.data)
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    rows = []
    for seed in SEEDS:
        model, device = heatcand.load_model(f"tagcn_k3_seed{seed}")
        heatmaps = Heatmaps(model, device)
        learned = joblib.load(os.path.join(args.models, f"tagcn_k3_seed{seed}", "chosen.joblib"))
        gpath = os.path.join(args.global_models, f"tagcn_k3_seed{seed}", "chosen.joblib")
        glearned = joblib.load(gpath) if os.path.exists(gpath) else None
        scorers = {"A disagreement": F.WindowBatchScorer(model, device),
                   "A learned": F.WindowBatchScorer(model, device, learned),
                   "B disagreement": F.GlobalScorer(model, device)}
        if glearned is not None:
            scorers["B learned"] = F.GlobalScorer(model, device, glearned)
        warm = d["coords"][0].astype(np.float32).astype(np.float64)
        heatmaps([warm[:20]])
        F.batch_heat(model, device, warm[None, :20])                       # CUDA warm-up, not timed
        for i in range(min(args.instances, len(d["coords"]))):
            coords = d["coords"][i].astype(np.float32).astype(np.float64)
            tour = initial_tour(coords, k)
            ref_l = LearnedScore(learned)
            times = {}
            start = time.perf_counter()
            ref_l.prepare(coords)
            ref = {"disagreement": [], "learned": []}
            for offset in (0, 10):
                wins = window_starts(n, 20, offset)
                idx = [(s + np.arange(m)) % n for s, m in wins]
                heats = heatmaps([coords[tour[j]] for j in idx])
                ref["disagreement"].append(np.array([segment_disagreement(h) for h in heats]))
                ref["learned"].append(ref_l.values(coords, tour, idx, heats, [s for s, _ in wins]))
            times["per-window (both scores)"] = time.perf_counter() - start
            for label, sc in scorers.items():
                start = time.perf_counter()
                sc.prepare(coords)
                vals = [sc.values(coords, tour, window_starts(n, 20, offset)) for offset in (0, 10)]
                times[label] = time.perf_counter() - start
                kind = label.split()[1]
                for off, (a, b) in enumerate(zip(vals, ref[kind])):
                    rows.append({"seed": seed, "instance": i, "half_pass": off, "version": label, **compare(a, b)})
            rows.append({"seed": seed, "instance": i, "times_s": times,
                         "global_missing": scorers["B disagreement"].missing[0] / max(scorers["B disagreement"].missing[1], 1)})
            print(f"  seed {seed} instance {i}: {times}", flush=True)

    summary = {}
    for label in sorted({r["version"] for r in rows if "version" in r}):
        rs = [r for r in rows if r.get("version") == label]
        summary[label] = {m: {"mean": float(np.mean([r[m] for r in rs])), "min": float(np.min([r[m] for r in rs])),
                              "max": float(np.max([r[m] for r in rs]))}
                          for m in ("spearman", "top20_overlap", "max_abs_diff")}
    tr = [r for r in rows if "times_s" in r]
    summary["times_s"] = {k2: float(np.mean([r["times_s"][k2] for r in tr if k2 in r["times_s"]]))
                          for k2 in tr[0]["times_s"]}
    summary["global_missing"] = float(np.mean([r["global_missing"] for r in tr]))
    commit, dirty = git_info()
    out = {"meta": {"git_commit": commit, "git_dirty": dirty, "dataset": name, "n": n,
                    "instances": args.instances, "seeds": SEEDS}, "summary": summary, "rows": rows}
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
