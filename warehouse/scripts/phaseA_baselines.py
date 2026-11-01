"""Phase A experiments. Every run writes one JSON (and one .npz of the episode) as it ends, so a stopped command resumes.

    python scripts/phaseA_baselines.py seq              # layer 1: gap and time of NN and 2-opt against LKH-3
    python scripts/phaseA_baselines.py e1               # offline lots
    python scripts/phaseA_baselines.py e2               # lifelong (rates come from the throughput of E1)
    python scripts/phaseA_baselines.py e3               # conflicts of the E1 episodes
    python scripts/phaseA_baselines.py gif              # a short GIF of a small episode with PIBT

Results in results/warehouse/phaseA/. At most --workers processes; run it with nice -n 10.
"""

import argparse
import glob
import json
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np

from warehouse import conflicts as C
from warehouse import env as W
from warehouse import sequencing as S
from warehouse import sim
from warehouse import verify as V
from warehouse import viz

OUT = "results/warehouse/phaseA"
SEEDS = range(5)
METHODS = ("nn", "2opt", "lkh")
PLANNERS = ("pibt", "prioritized", "cbs")
CBS_ARGS = {"time_limit": 30.0, "budget": 900.0}
RATES = {"loose": 0.5, "medium": 0.8, "saturated": 1.2}
SWEEP = (0.9, 1.0, 1.1, 1.3, 1.4, 1.5, 1.7, 2.0)   # extra rates, to place the point where a method stops keeping up
LIFELONG_STEPS = 2000
E2_DIR = "e2b"                                  # e2/ is the first run, whose verifier took orders cut by the horizon for errors
SEQ_ORDERS = 200


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def world(size, seed):
    wh = W.Warehouse(W.SIZES[size])
    return wh, W.Catalogue(wh, 2 * len(wh.shelves), seed)


def as_dict(orders):
    return [{k: getattr(o, k) for k in ("oid", "products", "station", "arrival")} for o in orders]


def save(path, ep, rep, extra):
    np.savez_compressed(path + ".npz", traj=ep["traj"], targets=ep["targets"], step_seconds=ep["step_seconds"])
    rec = {**rep, **extra, "planner_stats": ep["planner"], "seq_seconds": ep["seq_seconds"], "wall": ep["seconds"],
           "pending_end": ep["pending_end"], "commit": commit()}
    rec["deliveries"] = sorted(e["end"] for e in ep["events"] if e["kind"] == "deliver")
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, path + ".json")


def run_e1(job):
    size, method, planner, seed = job
    path = f"{OUT}/e1/{size}_{method}_{planner}_{seed}"
    if os.path.exists(path + ".json"):
        return path
    wh, cat = world(size, seed)
    orders = W.batch_orders(wh, cat, W.BATCH[size], seed)
    t0 = time.perf_counter()
    ep = sim.run_episode(wh, cat, orders, method, planner, seed, planner_args=CBS_ARGS if planner == "cbs" else None)
    rep = V.verify(ep["traj"], wh.grid, as_dict(orders), cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"])
    save(path, ep, rep, {"size": size, "method": method, "planner": planner, "seed": seed, "elapsed": time.perf_counter() - t0})
    return path


def run_e2(job):
    size, name, rate, planner, seed = job
    path = f"{OUT}/{E2_DIR}/{size}_{name}_{planner}_{seed}"
    if os.path.exists(path + ".json"):
        return path
    wh, cat = world(size, seed)
    orders = W.poisson_orders(wh, cat, rate, LIFELONG_STEPS, seed)
    t0 = time.perf_counter()
    ep = sim.run_episode(wh, cat, orders, "lkh", planner, seed, stop_at=LIFELONG_STEPS)
    rep = V.verify(ep["traj"], wh.grid, as_dict(orders), cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"], cut=True)
    save(path, ep, rep, {"size": size, "rate_name": name, "rate": rate, "planner": planner, "seed": seed,
                         "arrivals": [o.arrival for o in orders], "elapsed": time.perf_counter() - t0})
    return path


def run_seq(size):
    path = f"{OUT}/seq_{size}.json"
    if os.path.exists(path):
        return path
    wh, cat = world(size, 0)
    rng = np.random.default_rng([0, 7])
    orders = W.batch_orders(wh, cat, SEQ_ORDERS, 0)
    rows = []
    for o in orders:
        start = int(rng.choice(wh.free))
        stops = [wh.access[s] for s in o.shelves]
        d = S.matrix(wh, start, stops, o.station)
        row = {"oid": o.oid, "products": len(o.products), "stops": len(stops)}
        end = len(d) - 1
        for name, fn in (("nn", S.nearest), ("2opt", S.two_opt), ("lkh", S.lkh)):
            t0 = time.perf_counter()
            order = fn(d)
            row[f"{name}_s"] = time.perf_counter() - t0
            row[name] = S.path_length(d, [0] + order + [end])
        rows.append(row)
    json.dump(rows, open(path, "w"))
    return path


def run_e3(path):
    """path: an E1 run. Writes conflicts_<run>.npz."""
    name = os.path.basename(path)[:-5]
    out = f"{OUT}/conflicts_{name}.npz"
    if os.path.exists(out):
        return out
    size = name.split("_")[0]
    wh = W.Warehouse(W.SIZES[size])
    arr = np.load(path.replace(".json", ".npz"))
    res = C.analyse(wh, arr["traj"], arr["targets"])
    np.savez_compressed(out, **res)
    return out


def pool_map(fn, jobs, workers):
    done = 0
    with Pool(workers) as pool:
        for r in pool.imap_unordered(fn, jobs):
            done += 1
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(jobs)} {r}", flush=True)


def e2_jobs():
    jobs = []
    for size in W.SIZES:
        runs = [json.load(open(p)) for p in glob.glob(f"{OUT}/e1/{size}_lkh_pibt_*.json")]
        capacity = float(np.mean([r["delivered"] / r["makespan"] for r in runs]))      # orders per step of LKH + PIBT in E1
        rates = {**RATES, **{f"x{x:.1f}": x for x in SWEEP}}
        for name, x in rates.items():
            for planner in ("pibt", "prioritized"):
                for seed in SEEDS:
                    jobs.append((size, name, x * capacity, planner, seed))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("seq", "e1", "e2", "e3", "gif"))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    for d in ("e1", E2_DIR):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)
    if args.what == "seq":
        pool_map(run_seq, list(W.SIZES), min(args.workers, 3))
    elif args.what == "e1":
        jobs = [(s, m, q, seed) for q in PLANNERS for s in W.SIZES for m in METHODS for seed in SEEDS
                if not (q == "cbs" and s == "large")]
        pool_map(run_e1, jobs, args.workers)
    elif args.what == "e2":
        pool_map(run_e2, e2_jobs(), args.workers)
    elif args.what == "e3":
        pool_map(run_e3, sorted(glob.glob(f"{OUT}/e1/*.json")), args.workers)
    else:
        wh = W.Warehouse(W.SIZES["small"])
        arr = np.load(f"{OUT}/e1/small_lkh_pibt_0.npz")
        viz.gif(wh, arr["traj"], arr["targets"], "docs/figures/phaseA_small_pibt.gif")
        print(os.path.getsize("docs/figures/phaseA_small_pibt.gif"))


if __name__ == "__main__":
    main()
