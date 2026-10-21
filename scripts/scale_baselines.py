"""Classical baselines on one (possibly very large) test set, with timings (phase 2, scale experiment).

For each instance:
    greedy        greedy edge (scale.greedy_sparse with no graph pairs: same tour as the dense version)
    christofides  networkx Christofides, only if n <= --christofides-max-n (it needs the full graph)
and each of them followed by 2-opt with neighbour lists, once per k-mode of scale_eval.py
(k = 50 and k = min(0.5 n, 100)), so that the comparison with the models uses the same 2-opt:
    greedy+2opt@k50, greedy+2opt@khalf100, christofides+2opt@k50, christofides+2opt@khalf100

Writes results/phase2_scale/baselines/<dataset>.json (summary) and .npz (per-instance gaps and
times in ms "<method>__ms"; for "+2opt" methods the time is the 2-opt alone).

Example:
    python scripts/scale_baselines.py --data data/tsp200_test.npz
"""

import argparse
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from tsp_gnn.baselines import christofides_tour
from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap, verify_tour
from tsp_gnn.scale import greedy_sparse, neighbor_lists, two_opt_neighbors
from tsp_gnn.utils import git_info

sys.path.insert(0, os.path.dirname(__file__))
from scale_eval import K_MODES, k_of, summarize  # noqa: E402  (same k and summary as the models)


def run_instance(job):
    """Baselines on one instance. Returns {key: value} (gaps in %, times in ms)."""
    coords, opt, with_christofides, limit = job
    coords = coords.astype(np.float32).astype(np.float64)   # the coordinates the models see
    n = len(coords)
    empty = np.zeros(0, dtype=np.int64)
    builders = {"greedy": lambda: greedy_sparse(empty, empty, np.zeros(0), coords)}
    if with_christofides:
        builders["christofides"] = lambda: christofides_tour(coords)
    out = {}
    lists = {mode: neighbor_lists(coords, k_of(mode, n)) for mode in K_MODES}
    for name, build in builders.items():
        start = time.perf_counter()
        tour = verify_tour(build(), n)
        out[f"{name}__ms"] = 1000 * (time.perf_counter() - start)
        out[name] = optimality_gap(tour_length(coords, tour), opt)
        for mode in K_MODES:
            key = f"{name}+2opt@{mode}"
            start = time.perf_counter()
            improved, finished = two_opt_neighbors(coords, tour, lists[mode], time_limit=limit)
            out[f"{key}__ms"] = 1000 * (time.perf_counter() - start)
            verify_tour(improved, n)
            out[key] = optimality_gap(tour_length(coords, improved), opt) if finished else np.nan
            out[f"{key}__timeout"] = float(not finished)
    return out


def main():
    parser = argparse.ArgumentParser(description="Scale experiment: greedy edge and Christofides baselines.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="results/phase2_scale/baselines")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--christofides-max-n", type=int, default=1000)
    parser.add_argument("--two-opt-limit", type=float, default=600.0)
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.data))[0]
    out_json = os.path.join(args.out_dir, f"{name}.json")
    if os.path.exists(out_json):
        print(f"{out_json} already exists: skipped")
        return
    d = load_dataset(args.data)
    n, num = int(d["coords"].shape[1]), len(d["coords"])
    with_chr = n <= args.christofides_max_n
    print(f"{name}: n = {n}, {num} instances, Christofides {'yes' if with_chr else 'no'}", flush=True)
    jobs = [(c, float(l), with_chr, args.two_opt_limit) for c, l in zip(d["coords"], d["lengths"])]
    with Pool(min(args.workers, num)) as pool:
        res = pool.map(run_instance, jobs, chunksize=1)

    per_inst = {key: np.array([r[key] for r in res]) for key in res[0]}
    commit, dirty = git_info()
    summary = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(),
                        "dataset": name, "n": n, "num": num, "k": {m: k_of(m, n) for m in K_MODES},
                        "two_opt_limit_s": args.two_opt_limit,
                        "lkh_seconds_mean": float(d["lkh_seconds"].mean()) if "lkh_seconds" in d else None,
                        "lkh_runs": int(d["lkh_runs"]) if "lkh_runs" in d else 10}}
    for m in [k for k in res[0] if "__" not in k]:
        summary[m] = {**summarize(per_inst[m]), "ms_mean": float(per_inst[f"{m}__ms"].mean())}
        if "+2opt" in m:
            summary[m]["timeouts"] = int(per_inst[f"{m}__timeout"].sum())
        print(f"  {m:<26}{summary[m]['gap_mean'] if summary[m]['gap_mean'] is not None else float('nan'):>7.2f}%"
              f"  {summary[m]['ms_mean']:>10.1f} ms", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, f"{name}.npz"), **per_inst)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"commit {commit}{' (uncommitted changes)' if dirty else ''}  |  saved to {out_json}")


if __name__ == "__main__":
    main()
