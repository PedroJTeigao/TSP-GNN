"""Phase B0: how much is there to gain, before training anything. Every arm is a re-run of the E1 lots (LKH-3 + prioritized)
with one thing changed, and every run is judged by the verifier.

    base       the baseline
    o1a        minimum-cost assignment over the queue; o1a_fifo the same on the oldest orders only
    o1b        o1a, with robots that have not started to pick back in the pool whenever a robot comes free
    o1c        a posteriori: 200 swaps of orders between robots, each simulated, kept only if the makespan drops
    o2a        traffic in hindsight: the losses of the previous run of this seed set the distances of layer 1, 3 times
    o2b        traffic from a fixed map, the losses of the other seeds
    o3a        two orders per trip, the largest shelf overlap; o3b the same, the partner that adds the least walking
    o4cbs      windowed CBS with 3 x the time limit and the budget of E1
    o4pibt5    PIBT with 5 priority rules, best makespan of the 5 complete and verified runs; o4pibt/ was the first version,
               where a rule that livelocked could win, and is not used
    bound      a lower bound on the makespan of any assignment with one order per trip

Results in results/warehouse/phaseB0/; a finished run is skipped, and o1c resumes from its last checkpoint.

    python scripts/phaseB0_margins.py base --workers 4
"""

import argparse
import glob
import json
import os
import random
import subprocess
import time
from multiprocessing import Pool

import numpy as np

from warehouse import conflicts as C
from warehouse import dispatch as D
from warehouse import env as W
from warehouse import sequencing as S
from warehouse import sim
from warehouse import traffic as T
from warehouse import verify as V

OUT = "results/warehouse/phaseB0"
SEEDS = range(5)
SIZES = ("medium", "large")
SWAPS = 200
CBS_ARGS = {"time_limit": 90.0, "budget": 2700.0}
RULES = ("age", "far", "near", "load", "fixed")
ARMS = {"base": ("small", "medium", "large"), "o1a": SIZES, "o1a_fifo": SIZES, "o1b": SIZES, "o1c": SIZES, "o2a": SIZES, "o2b": SIZES,
        "o3a": SIZES, "o3b": SIZES, "o4cbs": ("small", "medium"), "o4pibt5": ("small", "medium"), "bound": SIZES}
STAGES = {"base": ("base",), "online": ("o1a", "o1a_fifo", "o1b", "o2a", "o2b", "o3a", "o3b", "o4pibt5", "bound"), "o1c": ("o1c",), "o4cbs": ("o4cbs",)}


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def world(size, seed):
    wh = W.Warehouse(W.SIZES[size])
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    return wh, cat, W.batch_orders(wh, cat, W.BATCH[size], seed)


def as_dict(orders):
    return [{k: getattr(o, k) for k in ("oid", "products", "station", "arrival")} for o in orders]


def judged(wh, cat, orders, ep, capacity=1):
    return V.verify(ep["traj"], wh.grid, as_dict(orders), cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"], capacity=capacity)


def record(rep, ep, **extra):
    return {**rep, **extra, "planner_stats": ep["planner"], "seq_seconds": ep["seq_seconds"], "commit": commit()}


def write(path, rec):
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, path)


def run_simple(size, seed, method="lkh", planner="prioritized", capacity=1, args=None, make=None):
    wh, cat, orders = world(size, seed)
    t0 = time.perf_counter()
    ep = sim.run_episode(wh, cat, orders, method, planner, seed, planner_args=args, **(make(wh) if make else {}))
    return judged(wh, cat, orders, ep, capacity), ep, wh, time.perf_counter() - t0


def base_penalty(wh, size, seeds):
    return T.lost_per_pass([{k: v for k, v in np.load(f"{OUT}/base_{size}_{s}.npz").items()} for s in seeds])


def plan_of(events):
    plan = {}
    for e in events:
        if e["kind"] == "assign":
            plan.setdefault(e["robot"], []).append(e["order"])
    return plan


def last_robot(events):
    return max((e for e in events if e["kind"] == "deliver"), key=lambda e: e["end"])["robot"]


def o1c(size, seed, path):
    """Each try swaps an order of the robot that delivers last with one of another robot: a swap elsewhere cannot lower the
    makespan of this episode."""
    wh, cat, orders = world(size, seed)
    cache, rng = {}, random.Random(seed + 11)
    ck = path + ".ck"
    t0 = time.perf_counter()

    def simulate(plan):
        return sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.FixedPlan(plan), cache=cache)

    if os.path.exists(ck):
        st = json.load(open(ck))
        st["plan"] = {int(r): v for r, v in st["plan"].items()}
        rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
    else:
        first = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, cache=cache)
        rep = judged(wh, cat, orders, first)
        st = {"plan": plan_of(first["events"]), "makespan": rep["makespan"], "last": last_robot(first["events"]), "done": 0, "kept": 0,
              "history": [rep["makespan"]], "spent": 0.0}
    while st["done"] < SWAPS:
        plan = {r: list(v) for r, v in st["plan"].items()}
        a = st["last"]
        b = rng.choice(sorted(r for r in plan if r != a and plan[r]))
        i, j = rng.randrange(len(plan[a])), rng.randrange(len(plan[b]))
        plan[a][i], plan[b][j] = plan[b][j], plan[a][i]
        ep = simulate(plan)
        m = float(max(e["end"] for e in ep["events"] if e["kind"] == "deliver"))
        st["done"] += 1
        if m < st["makespan"]:
            st.update(plan=plan, makespan=m, last=last_robot(ep["events"]), kept=st["kept"] + 1)
        st["history"].append(st["makespan"])
        if st["done"] % 5 == 0:
            s = rng.getstate()
            write(ck, {**st, "rng": [s[0], list(s[1]), s[2]], "spent": st["spent"] + time.perf_counter() - t0})
            st["spent"] += time.perf_counter() - t0
            t0 = time.perf_counter()
    ep = simulate(st["plan"])
    rep = judged(wh, cat, orders, ep)
    assert rep["makespan"] == st["makespan"], "the kept plan does not repeat"
    return record(rep, ep, swaps=st["done"], kept=st["kept"], history=st["history"], elapsed=st["spent"] + time.perf_counter() - t0)


def o2a(size, seed):
    wh, cat, orders = world(size, seed)
    runs, t0 = [], time.perf_counter()
    penalty = base_penalty(wh, size, [seed])
    for it in range(3):
        m = T.Traffic(wh, penalty)
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, metric=m)
        rep = judged(wh, cat, orders, ep)
        runs.append(record(rep, ep, iteration=it + 1, mean_penalty=float(penalty[penalty > 0].mean())))
        penalty = T.lost_per_pass([C.analyse(wh, ep["traj"], ep["targets"])])
    best = min(runs, key=lambda r: r["makespan"])
    return {**best, "iterations": runs, "elapsed": time.perf_counter() - t0}


def o4pibt5(size, seed):
    runs, t0 = [], time.perf_counter()
    for rule in RULES:
        rep, ep, _, _ = run_simple(size, seed, planner="pibt", args={"rule": rule})
        runs.append(record(rep, ep, rule=rule))
    good = [r for r in runs if r["ok"] and r["delivered"] == r["orders"]]
    best = min(good, key=lambda r: r["makespan"])
    return {**best, "all_rules": runs, "elapsed": time.perf_counter() - t0}


def bound(size, seed):
    """Total work over the number of robots, or the longest order. The work of an order: the shortest way in from a station
    or a parking cell, its walk from the first shelf to the station (LKH-3, so tight rather than proved), and its services."""
    wh, cat, orders = world(size, seed)
    entry = list(wh.stations) + W.start_cells(wh, wh.layout.robots, seed)
    work = []
    for o in orders:
        stops = [wh.access[s] for s in o.shelves]
        _, length = S.open_path(wh, stops, o.station, "lkh")
        into = min(wh.dist_to(c)[e] for c in stops for e in entry)
        work.append(into + length + sim.SERVICE * (len(o.shelves) + 1))
    return {"makespan_bound": max(sum(work) / wh.layout.robots, max(work)), "mean_work": float(np.mean(work)), "elapsed": 0.0}


def job(j):
    arm, size, seed = j
    path = f"{OUT}/{arm}_{size}_{seed}.json"
    if os.path.exists(path):
        return path
    meta = {"arm": arm, "size": size, "seed": seed}
    if arm == "base":
        rep, ep, wh, sec = run_simple(size, seed)
        res = C.analyse(wh, ep["traj"], ep["targets"])
        np.savez_compressed(f"{OUT}/base_{size}_{seed}.npz", waits=res["waits"], detours=res["detours"], visits=res["visits"])
        rec = record(rep, ep, elapsed=sec)
    elif arm in ("o1a", "o1a_fifo", "o1b"):
        rep, ep, _, sec = run_simple(size, seed, make=lambda wh: {"dispatcher": D.Hungarian(wh, reassign=arm == "o1b", fifo=arm == "o1a_fifo")})
        rec = record(rep, ep, elapsed=sec)
    elif arm == "o1c":
        rec = o1c(size, seed, path)
    elif arm == "o2a":
        rec = o2a(size, seed)
    elif arm == "o2b":
        rep, ep, _, sec = run_simple(size, seed, make=lambda wh: {"metric": T.Traffic(wh, base_penalty(wh, size, [s for s in SEEDS if s != seed]))})
        rec = record(rep, ep, elapsed=sec)
    elif arm in ("o3a", "o3b"):
        rule = "overlap" if arm == "o3a" else "near"
        rep, ep, _, sec = run_simple(size, seed, capacity=2, make=lambda wh: {"dispatcher": D.Batcher(wh, rule=rule)})
        rec = record(rep, ep, elapsed=sec)
    elif arm == "o4cbs":
        rep, ep, _, sec = run_simple(size, seed, planner="cbs", args=CBS_ARGS)
        rec = record(rep, ep, elapsed=sec)
    elif arm == "o4pibt5":
        rec = o4pibt5(size, seed)
    else:
        rec = bound(size, seed)
    write(path, {**rec, **meta})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=list(STAGES) + list(ARMS))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.what == "o4cbs" and os.environ.get("B0_RUN_CBS") != "1":      # last in the queue: the revised plan runs it after the new arms
        print("o4cbs deferred: set B0_RUN_CBS=1", flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    arms = STAGES.get(args.what, (args.what,))
    jobs = [(a, s, seed) for a in arms for s in ARMS[a] for seed in SEEDS]
    done = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            done += 1
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(jobs)} {r}", flush=True)


if __name__ == "__main__":
    main()
