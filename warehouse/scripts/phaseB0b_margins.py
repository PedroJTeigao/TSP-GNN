"""Phase B0b: three margins left open by B0. Same protocol: LKH-3 + prioritized, 5 seeds, every run judged by the verifier.

    t1   capacity: Clarke & Wright against LKH-3 in CVRP mode and set partitioning, on whole lots, capacity 3 and, medium only, 4
    t1all  set partitioning over every trip of 3, on one medium lot: how much the list of candidates of t1 leaves out
    t2   arrivals: lifelong, trips of 3 at once (a), after a wait of 20 or 50 steps (b), or knowing the arrivals (c)
    t3   density: the medium warehouse with more robots and orders; baseline, Hungarian, static map, map x gamma and hindsight

Results in results/warehouse/phaseB0b/; a finished run is skipped.

    python scripts/phaseB0b_margins.py t1 --workers 4
"""

import argparse
import dataclasses
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import phaseB0_margins as P
from warehouse import batching as B
from warehouse import dispatch as D
from warehouse import env as W
from warehouse import sim
from warehouse import traffic as T
from warehouse import verify as V

OUT = "results/warehouse/phaseB0b"
P.OUT = OUT                                     # the edge files of T3 live here
SEEDS = range(5)
STEPS = 2000
LKH_LIMIT = {"medium": 120, "large": 300}
PARTNERS = {3: 25, 4: 12}
T1_JOBS = [("medium", 3), ("large", 3), ("medium", 4)]
RATES = {"medium": 0.8, "x2.0": 2.0}            # multiples of the E1 throughput of LKH-3 + PIBT, as in E2
WAITS = (0, 20, 50)
HORIZONS = {"c100": 100, "cinf": STEPS}
LAMBDAS = (0.0, 0.5, 2.0)
DENSITIES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
T3_ARMS = ("base", "o1b", "o2b1", "o2b2", "o2b3")
T3_SEEDS = {5.0: range(3)}                      # the CPU budget: 3 seeds at the highest density, where one episode costs most
T1ALL_SEEDS = (0,)


def judged(wh, cat, orders, ep, capacity, cut=False):
    return V.verify(ep["traj"], wh.grid, P.as_dict(orders), cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"], capacity=capacity, cut=cut)


def done(path):
    return os.path.exists(path)


def save(path, rec):
    rec = {**rec, "commit": P.commit()}
    P.write(path, rec)
    return path


def t1_sim(wh, cat, orders, groups, seed, cap):
    ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.Clusters(wh, groups))
    rep = judged(wh, cat, orders, ep, cap)
    assert rep["ok"] and rep["delivered"] == rep["orders"], rep["problems"]
    plan = D.plan_of(ep["events"])
    ff = sim.run_episode(wh, cat, orders, "lkh", "free", seed, dispatcher=D.FixedPlan(plan))
    free = judged(wh, cat, orders, ff, cap)
    return {"makespan": rep["makespan"], "mean_delay": rep["mean_delay"], "step_ms": rep["step_ms_mean"], "free_flow": free["makespan"]}


def t1(job):
    size, cap, seed = job
    path = f"{OUT}/t1_{size}_{cap}_{seed}.json"
    if done(path):
        return path
    wh, cat, orders = P.world(size, seed)
    cost = B.TripCost(wh, wh, sim.SERVICE)
    t0 = time.perf_counter()
    cw = B.savings(orders, cost, cap)
    t1_ = time.perf_counter()
    work = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wh_b0b_{os.getpid()}")
    lkh = B.cvrp_lkh(orders, cost, work, seed=seed + 1, limit=LKH_LIMIT[size], capacity=cap, trips=True, max_trials=10 ** 6)
    t2_ = time.perf_counter()
    best, info = B.partition(orders, cost, cap, PARTNERS[cap], seeds=cw + lkh)
    t3_ = time.perf_counter()
    rec = {"size": size, "cap": cap, "seed": seed, "cw_seconds": t1_ - t0, "lkh_seconds": t2_ - t1_, "partition_seconds": t3_ - t2_, **info, "elapsed": t3_ - t0}
    for name, groups in (("cw", cw), ("lkh", lkh), ("best", best)):
        assert sorted(o.oid for g in groups for o in g) == [o.oid for o in orders] and max(map(len, groups)) <= cap
        rec[name] = {"walk": B.total(groups, cost), "trips": len(groups), **t1_sim(wh, cat, orders, groups, seed, cap)}
    return save(path, rec)


def t1all(seed):
    path = f"{OUT}/t1all_{seed}.json"
    if done(path):
        return path
    wh, cat, orders = P.world("medium", seed)
    cost, t0 = B.TripCost(wh, wh, sim.SERVICE), time.perf_counter()
    cw = B.savings(orders, cost, 3)
    best, info = B.partition(orders, cost, 3, len(orders), seeds=cw, limit=1200)
    return save(path, {"seed": seed, "cw": B.total(cw, cost), "best": B.total(best, cost), **info, "elapsed": time.perf_counter() - t0})


def best_lambdas():
    """Per horizon, the weight with the lowest delay on the medium warehouse: the large one runs only that one."""
    out = []
    for h in HORIZONS:
        delay = {v: np.mean([json.load(open(f"{OUT}/t2_medium_{r}_{h}_{v}_{k}.json"))["delay_all"] for r in RATES for k in SEEDS]) for v in LAMBDAS}
        out.append(f"{h}_{min(delay, key=delay.get)}")
    return out


def capacity_rate(size, x):
    runs = [json.load(open(f"results/warehouse/phaseA/e1/{size}_lkh_pibt_{s}.json")) for s in SEEDS]
    return x * float(np.mean([r["delivered"] / r["makespan"] for r in runs]))


def t2_config(arm, wh, orders):
    cost = B.TripCost(wh, wh, sim.SERVICE)
    if arm == "a":
        return D.LiveBatches(wh, cost, 3)
    if arm.startswith("b"):
        return D.LiveBatches(wh, cost, 3, wait=int(arm[1:]))
    name, lam = arm.split("_")
    return D.LiveBatches(wh, cost, 3, future=orders, horizon=HORIZONS[name], lam=float(lam))


def t2_metrics(orders, events, rep):
    ends = {e["order"]: e["end"] for e in events if e["kind"] == "deliver"}
    arrivals = np.array([o.arrival for o in orders])
    finish = np.array([ends.get(o.oid, STEPS) for o in orders])
    moments = np.arange(STEPS // 2, STEPS)
    backlog = [int(((arrivals <= t) & (finish > t)).sum()) for t in moments]
    late = sum(e >= STEPS // 2 for e in ends.values())
    return {"orders": len(orders), "delivered": len(ends), "orders_per_hour": rep["orders_per_hour"], "orders_per_hour_late": late / (STEPS // 2) * 3600,
            "delay_all": float((finish - arrivals).mean()), "delay_delivered": rep["mean_delay"], "backlog": float(np.mean(backlog)), "backlog_end": backlog[-1],
            "step_ms": rep["step_ms_mean"]}


def t2(job):
    size, rate, arm, seed = job
    path = f"{OUT}/t2_{size}_{rate}_{arm}_{seed}.json"
    if done(path):
        return path
    wh = W.Warehouse(W.SIZES[size])
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    orders = W.poisson_orders(wh, cat, capacity_rate(size, RATES[rate]), STEPS, seed)
    t0 = time.perf_counter()
    cap = 1 if arm == "one" else 3
    ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, stop_at=STEPS, dispatcher=None if arm == "one" else t2_config(arm, wh, orders))
    rep = judged(wh, cat, orders, ep, cap, cut=True)
    assert rep["ok"], rep["problems"]
    return save(path, {"size": size, "rate": rate, "arm": arm, "seed": seed, "ok": rep["ok"], **t2_metrics(orders, ep["events"], rep), "elapsed": time.perf_counter() - t0})


def t3_world(x, seed):
    lay = W.SIZES["medium"]
    robots = int(lay.robots * x)
    wh = W.Warehouse(dataclasses.replace(lay, robots=robots, parking_rows=2 if robots <= 120 else 3))   # 2 rows park 120 robots
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    return wh, cat, W.batch_orders(wh, cat, int(W.BATCH["medium"] * x), seed)


def t3_edges(job):
    x, seed = job
    path = f"{OUT}/edges_d{x}_{seed}.npz"
    if done(path):
        return path
    wh, cat, orders = t3_world(x, seed)
    ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed)
    np.savez_compressed(path, **T.edge_stats(wh, ep["traj"], ep["targets"]).export())
    return path


def t3_free(wh, cat, orders, seed, ep, metric=None):
    ff = sim.run_episode(wh, cat, orders, "lkh", "free", seed, dispatcher=D.FixedPlan(D.plan_of(ep["events"])), metric=metric)
    return judged(wh, cat, orders, ff, 1)["makespan"]


def t3(job):
    x, arm, seed = job
    path = f"{OUT}/t3_{x}_{arm}_{seed}.json"
    if done(path):
        return path
    wh, cat, orders = t3_world(x, seed)
    size, t0 = f"d{x}", time.perf_counter()
    extra = {}
    if arm == "base":
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed)
        m = None
    elif arm == "o1b":
        m = None
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.Hungarian(wh, reassign=True))
    elif arm in ("o2b1", "o2b2"):
        gamma = P.fit_gamma(size, seed) if arm == "o2b2" else 1.0
        m = T.EdgeMetric(wh, {k: gamma * v for k, v in P.static_extra(wh, size, P.others(seed)).items()}, symmetric=True)
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, metric=m, dispatcher=D.Hungarian(wh, metric=m, reassign=True))
        extra["gamma"] = gamma
    else:
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.Hungarian(wh, reassign=True))
        stats = T.edge_stats(wh, ep["traj"], ep["targets"]).extra()
        runs = []
        for _ in range(3):
            m = T.EdgeMetric(wh, stats, symmetric=True)
            e = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, metric=m, dispatcher=D.Hungarian(wh, metric=m, reassign=True))
            r = judged(wh, cat, orders, e, 1)
            runs.append((r["makespan"], len(runs), e, m))
            stats = T.edge_stats(wh, e["traj"], e["targets"]).extra()
        _, _, ep, m = min(runs, key=lambda q: q[:2])
        extra["iterations"] = [q[0] for q in runs]
    rep = judged(wh, cat, orders, ep, 1)
    assert rep["ok"] and rep["delivered"] == rep["orders"], rep["problems"]
    free = t3_free(wh, cat, orders, seed, ep, m)
    return save(path, {"x": x, "arm": arm, "seed": seed, "ok": True, "robots": wh.layout.robots, "orders": len(orders), "makespan": rep["makespan"], "mean_delay": rep["mean_delay"],
                       "free_flow": free, "step_ms": rep["step_ms_mean"], "fallbacks": ep["planner"]["fallbacks"], "calls": ep["planner"]["calls"], **extra, "elapsed": time.perf_counter() - t0})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("t1", "t1all", "t2", "t3edges", "t3"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--densities", type=float, nargs="*", default=list(DENSITIES))
    p.add_argument("--sizes", nargs="*", default=["medium", "large"])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.what == "t1":
        fn, jobs = t1, [(s, c, k) for s, c in T1_JOBS for k in SEEDS if s in args.sizes]
    elif args.what == "t1all":
        fn, jobs = t1all, list(T1ALL_SEEDS)
    elif args.what == "t2":
        arms = {"medium": [f"{h}_{v}" for h in HORIZONS for v in LAMBDAS]}
        arms = {s: ["one", "a", "b20", "b50"] + (arms[s] if s in arms else best_lambdas()) for s in args.sizes}
        fn, jobs = t2, [(s, r, a, k) for s in args.sizes for r in RATES for a in arms[s] for k in SEEDS]
    elif args.what == "t3edges":
        fn, jobs = t3_edges, [(x, k) for x in args.densities for k in SEEDS]
    else:
        fn, jobs = t3, [(x, a, k) for x in args.densities for a in T3_ARMS for k in T3_SEEDS.get(x, SEEDS)]
    n = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(fn, jobs):
            n += 1
            print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} {r}", flush=True)


if __name__ == "__main__":
    main()
