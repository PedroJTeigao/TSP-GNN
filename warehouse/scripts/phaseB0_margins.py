"""Phase B0: how much is there to gain, before training anything. Every arm is a re-run of the E1 lots (LKH-3 + prioritized)
with one thing changed, and every run is judged by the verifier. Arms by component, baseline first:

    base       LKH-3 + prioritized + nearest robot: also O3(a), no batching, and O1(a); edges repeats it and keeps the edge statistics
    O3         o3a seed (shelf overlap); o3b control, least added walking; o3c Clarke & Wright; o3c3 the same with 3 orders;
               o3d regions (k-means); o3e savings on the static traffic map; o3f savings against the exact optimum on 20 orders;
               o3x the exact pairs of the whole lot, also 2 orders per trip
    O1         o1a Hungarian over the queue; o1a_fifo control, oldest orders only; o1b Hungarian with reassignment; o1b0 same, cost
               without the length of the order; o1it greedy + iterative reassignment; o1dw Hungarian on decayed waits per edge;
               o1dg Hungarian on distance + gamma x delay near the first goal; o1e MCA + LNS, medium only;
               o1c a posteriori, 200 simulated swaps
    O2-A       planner only: o2ah1, o2ah2 crisscross highways, wrong way +1, +2; o2ag CMA-ES on 6 weights (ggo fits them);
               o2af1, o2af2 flow and contraflow costs
    O2-B       upstream distances, in LKH-3 and the Hungarian: o2b1 static map; o2b2 the same x gamma; o2b3 hindsight per edge, 3 runs.
               o2b and o2a are the first version: nearest robot, per cell
    O4         small and medium: o4pibt5 best of 5 priority rules, complete runs only; o4w5, o4w10 windows (20 is the baseline);
               o4cbs windowed CBS with 3 x the limits of E1, only with B0_RUN_CBS=1, last
    bound      a lower bound on the makespan with one order per trip
    ff_<arm>   the free-flow limit of an arm: same assignment and routes, each robot alone

Results in results/warehouse/phaseB0/; a finished run is skipped, and o1c, o1e and ggo resume from a checkpoint.

    python scripts/phaseB0_margins.py o3 --workers 4
"""

import argparse
import glob
import json
import os
import pickle
import random
import subprocess
import time
from multiprocessing import Pool

import numpy as np

from warehouse import batching as B
from warehouse import conflicts as C
from warehouse import dispatch as D
from warehouse import env as W
from warehouse import insertion as I
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
WORK = ("small", "medium")
ARMS = {"base": ("small", "medium", "large"), "edges": ("small", "medium", "large"), "bound": SIZES,
        "o3a": SIZES, "o3b": SIZES, "o3c": SIZES, "o3c3": SIZES, "o3d": SIZES, "o3e": SIZES, "o3f": SIZES, "o3x": SIZES,
        "o1a": SIZES, "o1a_fifo": SIZES, "o1b": SIZES, "o1b0": SIZES, "o1it": SIZES, "o1dw": SIZES, "o1dg": SIZES, "o1e": ("medium",), "o1c": SIZES,
        "o2a": SIZES, "o2b": SIZES, "o2ah1": SIZES, "o2ah2": SIZES, "o2ag": ("medium",), "o2af1": SIZES, "o2af2": SIZES,
        "o2b1": SIZES, "o2b2": SIZES, "o2b3": SIZES,
        "o4pibt5": WORK, "o4w5": WORK, "o4w10": WORK, "o4cbs": WORK}
STAGES = {"edges": ("edges",), "o3": ("o3a", "o3b", "o3c", "o3c3", "o3d", "o3e", "o3f"), "o1": ("o1b0", "o1it", "o1dw", "o1dg", "o1e"),
          "o2": ("o2ah1", "o2ah2", "o2af1", "o2af2", "o2b1", "o2b2", "o2b3"), "ggo": (), "ggo_base": (), "o2g": ("o2ag",), "o4": ("o4w5", "o4w10", "o4pibt5"),
          "o4cbs": ("o4cbs",), "o3x": ("o3x",), "base": ("base",), "bound": ("bound",),
          "ff": ("base", "o3a", "o3b", "o3c", "o3c3", "o3d", "o3e", "o3x", "o1a", "o1a_fifo", "o1b", "o1b0", "o1it", "o1dw", "o1dg", "o1e", "o1c", "o2b1", "o2b2", "o2b3", "o4pibt5")}
GGO_SEEDS = (100, 101)                          # kept apart from the 5 seeds of the report
GGO_GENERATIONS, GGO_POP = 40, 8


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


def static_extra(wh, size, seeds):
    return T.pooled_extra([dict(np.load(f"{OUT}/edges_{size}_{s}.npz")) for s in seeds])


def others(seed):
    return [s for s in SEEDS if s != seed]


def fit_gamma(size, seed):
    """Least squares through 0, weighted by passes: the real loss of each edge in a seed against the map of the other seeds,
    over the seeds other than this one."""
    xy = xx = 0.0
    for s in others(seed):
        x = dict(np.load(f"{OUT}/edges_{size}_{s}.npz"))
        table = T.pooled_extra([dict(np.load(f"{OUT}/edges_{size}_{q}.npz")) for q in others(seed) if q != s])
        for (u, v), n, y in zip(x["edges"].tolist(), x["edge_passes"].tolist(), x["edge_delta"].tolist()):
            m = table.get((u, v), 0.0)
            xy += n * m * y
            xx += n * m * m
    return xy / xx if xx else 0.0


def ggo_extra(wh, x):
    return T.crisscross(wh, [float(v) for v in x[:3]], [float(v) for v in x[3:]])


def config(arm, size, seed, wh, cat, orders):
    """What an arm changes: keyword arguments of run_episode, with capacity and method."""
    cfg = {"planner": "prioritized", "capacity": 1}
    cost = lambda metric=wh: B.TripCost(wh, metric, sim.SERVICE)
    if arm in ("o1a", "o1a_fifo", "o1b", "o1b0"):
        cfg["dispatcher"] = D.Hungarian(wh, reassign=arm in ("o1b", "o1b0"), fifo=arm == "o1a_fifo", use_length=arm != "o1b0")
    elif arm == "o1it":
        cfg["dispatcher"] = D.Iterative(wh)
    elif arm in ("o1dw", "o1dg"):
        cfg["dispatcher"] = D.LiveCosts(wh, "edge" if arm == "o1dw" else "near")
    elif arm == "o2b":
        cfg["metric"] = T.Traffic(wh, base_penalty(wh, size, others(seed)))
    elif arm in ("o3a", "o3b"):
        cfg.update(capacity=2, dispatcher=D.Batcher(wh, rule="overlap" if arm == "o3a" else "near"))
    elif arm in ("o3c", "o3c3"):
        cap = 3 if arm == "o3c3" else 2
        cfg.update(capacity=cap, dispatcher=D.Clusters(wh, B.savings(orders, cost(), cap)))
    elif arm == "o3x":
        c = cost()
        groups = B.exact_clusters(orders, c)
        cfg.update(capacity=2, dispatcher=D.Clusters(wh, groups), walk=B.total(groups, c), walk_savings=B.total(B.savings(orders, c, 2), c))
    elif arm == "o3d":
        cfg.update(capacity=2, dispatcher=D.Clusters(wh, B.regions(wh, orders, 2, wh.layout.robots, seed)))
    elif arm == "o3e":
        m = T.EdgeMetric(wh, static_extra(wh, size, others(seed)), symmetric=True)
        cfg.update(capacity=2, metric=m, dispatcher=D.Clusters(wh, B.savings(orders, cost(m), 2), m))
    elif arm in ("o2ah1", "o2ah2"):
        p = 1.0 if arm == "o2ah1" else 2.0
        cfg["planner_args"] = {"metric": T.EdgeMetric(wh, T.crisscross(wh, [p], [p]))}
    elif arm == "o2ag":
        cfg["planner_args"] = {"metric": T.EdgeMetric(wh, ggo_extra(wh, json.load(open(f"{OUT}/ggo.json"))["best"]))}
    elif arm == "o2af1":
        cfg["planner_args"] = {"with_flow": 0.1, "against": 0.5}
    elif arm == "o2af2":
        cfg["planner_args"] = {"with_flow": 0.3, "against": 1.0}
    elif arm in ("o2b1", "o2b2"):
        gamma = fit_gamma(size, seed) if arm == "o2b2" else 1.0
        m = T.EdgeMetric(wh, {k: gamma * v for k, v in static_extra(wh, size, others(seed)).items()}, symmetric=True)
        cfg.update(metric=m, dispatcher=D.Hungarian(wh, metric=m, reassign=True), gamma=gamma)
    elif arm in ("o4w5", "o4w10"):
        w = 5 if arm == "o4w5" else 10
        cfg["planner_args"] = {"horizon": w, "execute": max(1, w // 4)}
    elif arm == "o4cbs":
        cfg.update(planner="cbs", planner_args=CBS_ARGS)
    return cfg


def run_arm(arm, size, seed):
    wh, cat, orders = world(size, seed)
    t0 = time.perf_counter()
    cfg = config(arm, size, seed, wh, cat, orders)
    setup = time.perf_counter() - t0
    capacity, planner, gamma = cfg.pop("capacity"), cfg.pop("planner"), cfg.pop("gamma", None)
    walk = {k: cfg.pop(k) for k in ("walk", "walk_savings") if k in cfg}
    ep = sim.run_episode(wh, cat, orders, "lkh", planner, seed, **cfg)
    rep = judged(wh, cat, orders, ep, capacity)
    return record(rep, ep, elapsed=time.perf_counter() - t0, setup_seconds=setup, gamma=gamma, trips=D.plan_of(ep["events"]), **walk), ep, cfg.get("metric")


def base_penalty(wh, size, seeds):
    return T.lost_per_pass([{k: v for k, v in np.load(f"{OUT}/base_{size}_{s}.npz").items()} for s in seeds])


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
        st = {"plan": D.plan_of(first["events"]), "makespan": rep["makespan"], "last": last_robot(first["events"]), "done": 0, "kept": 0,
              "history": [rep["makespan"]], "spent": 0.0}
    while st["done"] < SWAPS:
        plan = {r: [x if isinstance(x, int) else x[0] for x in v] for r, v in st["plan"].items()}
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


def o1e(size, seed, path):
    """MCA, then LNS: destroy the robot that ends last, or 10 % of the orders at random; reinsert; keep a lower makespan, or the
    same one with less delay. Stops at 20 iterations or 20 minutes."""
    wh, cat, orders = world(size, seed)
    cache, rng, t0 = {}, random.Random(seed + 13), time.perf_counter()
    ins = I.Insertion(wh, orders, W.start_cells(wh, wh.layout.robots, seed), sim.SERVICE, cache)

    def simulate(plan):
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.FixedPlan(plan), cache=cache)
        return ep, judged(wh, cat, orders, ep)

    best = ins.build()
    ep, rep = simulate(best)
    ins.measure(best, ep["events"])
    start, history = rep["makespan"], [rep["makespan"]]
    for it in range(20):
        if time.perf_counter() - t0 > 1200:
            break
        plan = {r: list(v) for r, v in best.items()}
        if it % 2 == 0:
            r = last_robot(ep["events"])
            out, plan[r] = plan[r], []
        else:
            every = [o for v in plan.values() for o in v]
            out = rng.sample(every, max(1, len(every) // 10))
            plan = {r: [o for o in v if o not in out] for r, v in plan.items()}
        for o in sorted(out, key=lambda k: (-ins.trip(ins.order[k].station, k), k)):
            ins.insert(plan, o)
        e2, r2 = simulate(plan)
        if (r2["makespan"], r2["mean_delay"]) < (rep["makespan"], rep["mean_delay"]):
            best, ep, rep = plan, e2, r2
            ins.measure(best, ep["events"])
        history.append(rep["makespan"])
    return record(rep, ep, elapsed=time.perf_counter() - t0, start=start, history=history, trips=D.plan_of(ep["events"]))


def o2b3(size, seed):
    """Hindsight per edge: the real times of the last run set the distances of LKH-3 and the Hungarian."""
    wh, cat, orders = world(size, seed)
    t0, runs = time.perf_counter(), []
    ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, dispatcher=D.Hungarian(wh, reassign=True))
    extra = T.edge_stats(wh, ep["traj"], ep["targets"]).extra()
    for it in range(3):
        m = T.EdgeMetric(wh, extra, symmetric=True)
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, metric=m, dispatcher=D.Hungarian(wh, metric=m, reassign=True))
        rep = judged(wh, cat, orders, ep)
        runs.append(record(rep, ep, iteration=it + 1, trips=D.plan_of(ep["events"])))
        extra = T.edge_stats(wh, ep["traj"], ep["targets"]).extra()
    best = min(runs, key=lambda r: r["makespan"])
    return {**best, "iterations": [{k: v for k, v in r.items() if k != "trips"} for r in runs], "elapsed": time.perf_counter() - t0}


def o3f(size, seed):
    """Savings against the optimum on 20 orders, 4 subsets per seed: the exact matching, and LKH-3 in CVRP mode."""
    wh, cat, orders = world(size, seed)
    cost = B.TripCost(wh, wh, sim.SERVICE)
    rng, rows = np.random.default_rng([seed, 20]), []
    for k in range(4):
        sub = [orders[int(i)] for i in sorted(rng.choice(len(orders), 20, replace=False))]
        rows.append({"alone": sum(cost([o]) for o in sub), "savings": B.total(B.savings(sub, cost, 2), cost), "exact": B.exact_pairs(sub, cost),
                     "cvrp": B.cvrp_lkh(sub, cost, os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wh_cvrp_{os.getpid()}"))})
    return {"subsets": rows, "elapsed": 0.0}


def edges_job(size, seed):
    wh, cat, orders = world(size, seed)
    ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed)
    rep = judged(wh, cat, orders, ep)
    old = json.load(open(f"{OUT}/base_{size}_{seed}.json"))
    assert rep["makespan"] == old["makespan"], "the baseline changed"
    np.savez_compressed(f"{OUT}/edges_{size}_{seed}.npz", **T.edge_stats(wh, ep["traj"], ep["targets"]).export())
    return {"makespan": rep["makespan"], "elapsed": 0.0}


def plan_for_ff(arm, size, seed):
    """(trips of the arm, the metric of its layer 1, its capacity)."""
    wh, cat, orders = world(size, seed)
    rec = json.load(open(f"{OUT}/{arm}_{size}_{seed}.json")) if os.path.exists(f"{OUT}/{arm}_{size}_{seed}.json") else {}
    cfg = config(arm, size, seed, wh, cat, orders)
    if "trips" in rec:
        return {int(r): v for r, v in rec["trips"].items()}, cfg.get("metric"), cfg["capacity"]
    if arm == "o1c":
        st = json.load(open(f"{OUT}/o1c_{size}_{seed}.json.ck"))
        return {int(r): v for r, v in st["plan"].items()}, None, 1
    r, _, m = run_arm(arm, size, seed)
    return {int(k): v for k, v in r["trips"].items()}, m, cfg["capacity"]


def free_flow(arm, size, seed):
    wh, cat, orders = world(size, seed)
    plan, metric, capacity = plan_for_ff(arm, size, seed)
    ep = sim.run_episode(wh, cat, orders, "lkh", "free", seed, dispatcher=D.FixedPlan(plan), metric=metric)
    rep = judged(wh, cat, orders, ep, capacity)
    bad = {k: v for k, v in rep["problems"].items() if v and k not in ("vertex_conflicts", "edge_conflicts")}
    assert not bad and rep["delivered"] == rep["orders"], f"free flow of {arm}: {bad}"
    return {"makespan": rep["makespan"], "mean_delay": rep["mean_delay"], "orders_per_hour": rep["orders_per_hour"], "elapsed": 0.0}


def ggo_eval(x):
    size = "medium"
    out = []
    for seed in GGO_SEEDS:
        wh, cat, orders = world(size, seed)
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed, planner_args={"metric": T.EdgeMetric(wh, ggo_extra(wh, x))})
        out.append(float(max(e["end"] for e in ep["events"] if e["kind"] == "deliver")))
    return float(np.mean(out))


def ggo_base():
    """The baseline on the two seeds of the search: what the search has to beat."""
    out = []
    for seed in GGO_SEEDS:
        wh, cat, orders = world("medium", seed)
        ep = sim.run_episode(wh, cat, orders, "lkh", "prioritized", seed)
        out.append(float(max(e["end"] for e in ep["events"] if e["kind"] == "deliver")))
    write(f"{OUT}/ggo_base.json", {"makespan": out})


def ggo(workers):
    """CMA-ES on 6 wrong-way weights, 3 bands of rows and 3 of columns: the mean makespan on two seeds of its own."""
    import cma
    ck = f"{OUT}/ggo.ck"
    if os.path.exists(f"{OUT}/ggo.json"):
        return
    if os.path.exists(ck):
        es, log = pickle.load(open(ck, "rb"))
    else:
        es, log = cma.CMAEvolutionStrategy([1.0] * 6, 0.7, {"bounds": [0, 5], "popsize": GGO_POP, "seed": 7, "verbose": -9}), []
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        while len(log) < GGO_GENERATIONS:
            xs = es.ask()
            ys = pool.map(ggo_eval, [x.tolist() for x in xs])
            es.tell(xs, ys)
            log.append({"best": float(min(ys)), "mean": float(np.mean(ys)), "sofar": float(es.result.fbest), "seconds": time.perf_counter() - t0})
            pickle.dump((es, log), open(ck, "wb"))
            print(f"{time.strftime('%H:%M:%S')} ggo {len(log)} {log[-1]['sofar']:.1f}", flush=True)
    write(f"{OUT}/ggo.json", {"best": es.result.xbest.tolist(), "train_makespan": float(es.result.fbest), "log": log, "commit": commit()})


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
    wh, cat, orders = world(size, seed)
    for rule in RULES:
        ep = sim.run_episode(wh, cat, orders, "lkh", "pibt", seed, planner_args={"rule": rule})
        runs.append(record(judged(wh, cat, orders, ep), ep, rule=rule, trips=D.plan_of(ep["events"])))
    good = [r for r in runs if r["ok"] and r["delivered"] == r["orders"]]
    best = min(good, key=lambda r: r["makespan"])
    return {**best, "all_rules": [{k: v for k, v in r.items() if k != "trips"} for r in runs], "elapsed": time.perf_counter() - t0}


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
        wh = world(size, seed)[0]
        rec, ep, _ = run_arm(arm, size, seed)
        res = C.analyse(wh, ep["traj"], ep["targets"])
        np.savez_compressed(f"{OUT}/base_{size}_{seed}.npz", waits=res["waits"], detours=res["detours"], visits=res["visits"])
    elif arm.startswith("ff_"):
        rec = free_flow(arm[3:], size, seed)
    elif arm == "o1c":
        rec = o1c(size, seed, path)
    elif arm == "o1e":
        rec = o1e(size, seed, path)
    elif arm == "o2a":
        rec = o2a(size, seed)
    elif arm == "o2b3":
        rec = o2b3(size, seed)
    elif arm == "o3f":
        rec = o3f(size, seed)
    elif arm == "edges":
        rec = edges_job(size, seed)
    elif arm == "o4pibt5":
        rec = o4pibt5(size, seed)
    elif arm == "bound":
        rec = bound(size, seed)
    else:
        rec = run_arm(arm, size, seed)[0]
    write(path, {**rec, **meta})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=list(dict.fromkeys(list(STAGES) + list(ARMS))))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.what == "o4cbs" and os.environ.get("B0_RUN_CBS") != "1":      # last in the queue: the revised plan runs it after the new arms
        print("o4cbs deferred: set B0_RUN_CBS=1", flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    if args.what == "ggo":
        ggo(args.workers)
        return
    if args.what == "ggo_base":
        ggo_base()
        return
    arms = STAGES.get(args.what, (args.what,))
    jobs = [(a, s, seed) for a in arms for s in ARMS[a] for seed in SEEDS]
    if args.what == "ff":
        jobs = [(f"ff_{a}", s, seed) for a, s, seed in jobs]
    done = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            done += 1
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(jobs)} {r}", flush=True)


if __name__ == "__main__":
    main()
