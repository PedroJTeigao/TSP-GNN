"""Ablation A4 (2026-11-03): how much of the window gains is the network's, and how much is selection by beam / 2-opt / LKH?

The network's heatmap is replaced by SoftDist (tsp_gnn.softdist: softmax of −d / τ over the same k-NN graph of the
rescaled window) in the same pipelines, w = 20, on Fu-1k (128) and Fu-10k (16), gap against the E0 LKH-3 reference:

    tau       τ by grid search: final length of beam B = 10 windows (≤ 5 passes) on 20 NEW uniform instances, n = 1000
              (seed 20261103; no test set is used to choose τ). Writes results/ablations/a4/tau.json.
    window    model windows as E3 (window.refine, ≤ 5 passes), greedy and beam B = 10 paths, plus the 2-opt control
    triage    ranked triage as A1 (≤ 3 passes, top 10 / 20 / 50 %, no memo): SoftDist disagreement; the learned selector of
              phase 3 (TAGCN seed 0, unchanged) on SoftDist features ("learned_old"); the selector retrained on SoftDist
              features ("learned_new", after the features and train stages)
    features  the learned-triage feature files with p_mean / p_min / p_max / disagreement recomputed from SoftDist (the
              geometric features, labels and windows are those of phase 3); then scripts/learned_triage_train.py with
              --features-root results/ablations/a4/features (folder name tagcn_k3_seed0 kept for that script)

Outputs results/ablations/a4/<stage>/... (.json + .npz per set); a finished output is skipped.

    python scripts/ablation_a4.py tau
    python scripts/ablation_a4.py window --set fu1k --method beam10
    python scripts/ablation_a4.py triage --set fu10k --score dis
"""

import argparse
import datetime
import json
import os
import time

import joblib
import numpy as np

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, verify_tour
from tsp_gnn.hier import beam_path, greedy_path
from tsp_gnn.softdist import SoftDistBatchScorer, SoftDistHeatmaps, batch_softdist
from tsp_gnn.triage import ranked_refine
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour, make_solver, refine

OUT = "results/ablations/a4"
BENCH = "results/phase3_benchmark/data"
SETS = {"fu1k": f"{BENCH}/tsp1000_test_concorde.npz", "fu10k": f"{BENCH}/tsp10000_test_concorde.npz"}
TAUS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
TAU_SEED, TAU_N, TAU_NUM = 20261103, 1000, 20
FEATURE_SETS = {"tsp200_test": "data/tsp200_test.npz", "tsp1000_test_concorde": SETS["fu1k"],
                "tsp1000_test": "data/tsp1000_test.npz", "tsp10000_test_concorde": SETS["fu10k"]}
OLD_FEATURES = "results/phase3_learned_triage/features/tagcn_k3_seed0"
OLD_MODEL = "results/phase3_learned_triage/models/tagcn_k3_seed0/chosen.joblib"
NEW_MODEL = f"{OUT}/models/tagcn_k3_seed0/chosen.joblib"


def solver(kind, tau):
    if kind == "2opt":
        return make_solver("2opt"), None
    heat = SoftDistHeatmaps(tau)
    decode = (lambda h, p: greedy_path(h, p, 0, len(p) - 1)) if kind == "greedy" else \
        (lambda h, p: beam_path(h, p, 0, len(p) - 1, 10))
    return (lambda sets: [decode(h, p) for h, p in zip(heat(sets), sets)]), heat


def chosen_tau():
    return json.load(open(f"{OUT}/tau.json"))["tau"]


def stage_tau():
    path = f"{OUT}/tau.json"
    if os.path.exists(path):
        print(f"{path} exists: skipped")
        return
    rng = np.random.default_rng(TAU_SEED)
    insts = [rng.random((TAU_N, 2)).astype(np.float32).astype(np.float64) for _ in range(TAU_NUM)]
    k = min(TAU_N - 1, round(0.5 * TAU_N), 100)
    starts = [initial_tour(c, k) for c in insts]
    lengths = {}
    for tau in TAUS:
        sol, _ = solver("beam10", tau)
        lengths[tau] = [tour_length(c, refine(c, t, 20, sol, 5)[0]) for c, t in zip(insts, starts)]
        print(f"tau {tau}: mean length {np.mean(lengths[tau]):.5f}", flush=True)
    base = float(np.mean([tour_length(c, t) for c, t in zip(insts, starts)]))
    best = min(TAUS, key=lambda t: np.mean(lengths[t]))
    rec = {"tau": best, "grid": {str(t): float(np.mean(v)) for t, v in lengths.items()}, "initial_mean_length": base,
           "instances": TAU_NUM, "n": TAU_N, "seed": TAU_SEED, "criterion": "mean length after beam B=10 windows, w=20, <=5 passes"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(path, "w"), indent=1)
    print(json.dumps(rec, indent=1))


def run_set(name, folder, policies, extra_meta):
    """policies: {key: fn(coords, t0) -> (tour, stats)}; one record per instance as triage_ablation / window_eval."""
    data = SETS[name]
    out_dir = os.path.join(OUT, folder)
    base = os.path.splitext(os.path.basename(data))[0]
    if os.path.exists(os.path.join(out_dir, f"{base}.json")):
        print(f"{out_dir}/{base}.json exists: skipped")
        return
    d = load_dataset(data)
    num, n = len(d["coords"]), int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    out = {}
    add = lambda key, v: out.setdefault(key, []).append(v)
    for i in range(num):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)
        opt = float(d["lengths"][i])
        t0 = initial_tour(coords, k)
        g0 = optimality_gap(tour_length(coords, t0), opt)
        add("initial__gap", g0)
        for key, fn in policies.items():
            start = time.perf_counter()
            tour, stats = fn(coords, t0)
            add(f"{key}__ms", 1000 * (time.perf_counter() - start))
            verify_tour(tour, n)
            g = optimality_gap(tour_length(coords, tour), opt)
            add(f"{key}__gap", g)
            add(f"{key}__gain", g0 - g)
            for f in ("windows", "improved", "passes", "to_lkh", "lkh_calls", "ms_net", "ms_lkh"):
                if f in stats:
                    add(f"{key}__{f}", stats[f])
        if (i + 1) % 8 == 0 or i + 1 == num:
            print(f"  {i + 1}/{num}", flush=True)
    per = {k_: np.array(v, dtype=float) for k_, v in out.items()}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(), "dataset": base,
                        "n": n, "num": num, **extra_meta}, "initial": {"gap_mean": float(per["initial__gap"].mean())}}
    for key in policies:
        summary[key] = {"gap_mean": float(per[f"{key}__gap"].mean()), "gain_mean": float(per[f"{key}__gain"].mean()),
                        "ms_mean": float(per[f"{key}__ms"].mean())}
        print(f"  {key:<24} gap {summary[key]['gap_mean']:.3f} %  gain {summary[key]['gain_mean']:.3f} pp  "
              f"{summary[key]['ms_mean']:.0f} ms", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{base}.npz"), **per)
    json.dump(summary, open(os.path.join(out_dir, f"{base}.json"), "w"), indent=1)


def stage_window(name, method):
    tau = chosen_tau()
    sol, _ = solver(method, tau)
    run_set(name, f"window/{method}", {f"{method}_w20": lambda c, t: refine(c, t, 20, sol, 5)},
            {"method": method, "tau": tau if method != "2opt" else None, "max_passes": 5, "width": 20})


def stage_triage(name, score):
    tau = chosen_tau()
    bundle = None if score == "dis" else joblib.load(OLD_MODEL if score == "learned_old" else NEW_MODEL)
    policies = {}
    for x in (10, 20, 50):
        def fn(c, t, x=x):
            return ranked_refine(c, t, 20, "scorer", x / 100, scorer=SoftDistBatchScorer(tau, bundle), max_passes=3)
        policies[f"softdist_{score}_top{x}"] = fn
    run_set(name, f"triage/{score}", policies, {"score": score, "tau": tau, "max_passes": 3, "width": 20,
                                                 "model": None if score == "dis" else (OLD_MODEL if score == "learned_old" else NEW_MODEL)})


def stage_features():
    tau = chosen_tau()
    out_dir = f"{OUT}/features/tagcn_k3_seed0"
    os.makedirs(out_dir, exist_ok=True)
    for name, data in FEATURE_SETS.items():
        path = f"{out_dir}/{name}.npz"
        if os.path.exists(path):
            print(f"{path} exists: skipped")
            continue
        old = dict(np.load(f"{OLD_FEATURES}/{name}.npz"))
        d = load_dataset(data)
        n = int(d["coords"].shape[1])
        k = min(n - 1, round(0.5 * n), 100)
        inst = old["instance"].astype(int)
        p = {f: np.zeros(len(inst)) for f in ("p_mean", "p_min", "p_max")}
        for i in np.unique(inst):
            coords = d["coords"][i].astype(np.float32).astype(np.float64)       # as learned_triage_data.py
            tour = initial_tour(coords, k)
            where = np.flatnonzero(inst == i)
            for size in np.unique(old["size"][where]).astype(int):
                rows = where[old["size"][where] == size]
                cities = np.stack([tour[(int(old["start"][j]) + np.arange(size)) % n] for j in rows])
                heat = batch_softdist(coords[cities], tau)
                seg = heat[:, np.arange(size - 1), np.arange(1, size)]
                p["p_mean"][rows], p["p_min"][rows], p["p_max"][rows] = seg.mean(1), seg.min(1), seg.max(1)
        new = {**old, **p, "disagreement": 1.0 - p["p_mean"]}
        np.savez_compressed(path, **new)
        json.dump({"tau": tau, "source": f"{OLD_FEATURES}/{name}.npz", "windows": int(len(inst)),
                   "p_mean_mean": float(p["p_mean"].mean())}, open(f"{out_dir}/{name}.json", "w"), indent=1)
        print(f"{name}: {len(inst)} windows, mean p {p['p_mean'].mean():.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Ablation A4: SoftDist heatmaps in the window pipelines.")
    p.add_argument("stage", choices=("tau", "window", "triage", "features"))
    p.add_argument("--set", choices=tuple(SETS))
    p.add_argument("--method", choices=("greedy", "beam10", "2opt"))
    p.add_argument("--score", choices=("dis", "learned_old", "learned_new"))
    a = p.parse_args()
    {"tau": stage_tau, "features": stage_features}.get(a.stage, lambda: None)()
    if a.stage == "window":
        stage_window(a.set, a.method)
    elif a.stage == "triage":
        stage_triage(a.set, a.score)


if __name__ == "__main__":
    main()
