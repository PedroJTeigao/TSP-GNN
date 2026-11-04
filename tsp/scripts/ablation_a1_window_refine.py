"""Ablation A1, check (2026-11-04): the window memo in the original window.refine with LKH (≤ 5 passes), not the ranked triage.

Fu-10k (16 instances), w = 20, one run, one process, CPU. The same refine is run twice per instance: with the plain LKH solver,
and with that solver behind a cache keyed by the window's points in path order (a window with the same cities in the same
order gets the order LKH gave it before; LKH is deterministic, so the tour must be the same, which is asserted). Time is
the refinement only (the greedy + 2-opt tour is built once, outside the timing).

Writes results/ablations/a1_window_refine.json.

    python scripts/ablation_a1_window_refine.py
"""

import datetime
import json
import os
import time

import numpy as np

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import optimality_gap
from tsp_gnn.utils import git_info
from tsp_gnn.window import initial_tour, lkh_path, refine

DATA = "results/phase3_benchmark/data/tsp10000_test_concorde.npz"
OUT = "results/ablations/a1_window_refine.json"


class CachedLKH:
    def __init__(self):
        self.solved, self.calls, self.hits = {}, 0, 0

    def __call__(self, sets):
        out = []
        for p in sets:
            key = (len(p), np.ascontiguousarray(p).tobytes())
            if key in self.solved:
                self.hits += 1
            else:
                self.calls += 1
                self.solved[key] = lkh_path(p)
            out.append(self.solved[key])
        return out


def main():
    d = load_dataset(DATA)
    n = int(d["coords"].shape[1])
    k = min(n - 1, round(0.5 * n), 100)
    load_before = os.getloadavg()
    rows = []
    for i in range(len(d["coords"])):
        coords = d["coords"][i].astype(np.float32).astype(np.float64)
        opt = float(d["lengths"][i])
        t0 = initial_tour(coords, k)
        start = time.perf_counter()
        plain, st = refine(coords, t0, 20, lambda sets: [lkh_path(p) for p in sets], 5)
        ms_plain = 1000 * (time.perf_counter() - start)
        cache = CachedLKH()
        start = time.perf_counter()
        cached, st_c = refine(coords, t0, 20, cache, 5)
        ms_cache = 1000 * (time.perf_counter() - start)
        assert np.array_equal(plain, cached), f"instance {i}: the cache changed the tour"
        rows.append({"gap": optimality_gap(tour_length(coords, plain), opt), "ms_none": ms_plain, "ms_cache": ms_cache,
                     "windows": st["windows"], "passes": st["passes"], "calls_cache": cache.calls, "hits": cache.hits})
        print(f"{i + 1}/{len(d['coords'])}: gap {rows[-1]['gap']:.3f} %  none {ms_plain / 1000:.1f} s  cache "
              f"{ms_cache / 1000:.1f} s  repeats {cache.hits / st['windows']:.0%}", flush=True)
    mean = lambda f: float(np.mean([r[f] for r in rows]))
    commit, dirty = git_info()
    rec = {"meta": {"git_commit": commit, "git_dirty": dirty, "date": datetime.date.today().isoformat(), "data": DATA,
                    "width": 20, "max_passes": 5, "load_before": load_before, "load_after": os.getloadavg(),
                    "device": "cpu"},
           "gap_mean": mean("gap"), "s_none_mean": mean("ms_none") / 1000, "s_cache_mean": mean("ms_cache") / 1000,
           "windows_mean": mean("windows"), "passes_mean": mean("passes"), "calls_cache_mean": mean("calls_cache"),
           "repeat_share": float(np.sum([r["hits"] for r in rows]) / np.sum([r["windows"] for r in rows])), "rows": rows}
    json.dump(rec, open(OUT, "w"), indent=1)
    print(json.dumps({k_: v for k_, v in rec.items() if k_ != "rows"}, indent=1))


if __name__ == "__main__":
    main()
