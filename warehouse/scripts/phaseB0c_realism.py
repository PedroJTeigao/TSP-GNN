"""Phase B0c: is there margin for a predictor once execution is imperfect? Clarke & Wright with 3 orders + LKH-3 + Hungarian, on PIBT and
on prioritized under action dependencies, 5 seeds, every run judged by the verifier. Rotation and delays happen in exec.py.

    regimes    r1 the same p and q for all; r2 the same mean, from the state (exec.StateDelay)
    arms       a  blind to turns and delays
               b  turns known: Dijkstra on cell + heading
               c  b + static map of what delays cost, from the a runs of the other seeds
               d  c x gamma, fitted on the other seeds
               e  oracle, the function known: b + the traffic map of c without the robots' own delays + the expected delay per robot type
               e0 control, the same without the robot type
               f  oracle, hindsight: b + the real times of the last run of this seed, 3 runs
    calibrate  s and q_bad of r2, per cell of the design, on seeds of their own

The distances of an arm reach Clarke & Wright, the Hungarian, the visit order and the paths of the planner. Results in
results/warehouse/phaseB0c/; a finished run is skipped.

    python scripts/phaseB0c_realism.py calibrate --workers 4
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
from warehouse import exec as E
from warehouse import sim
from warehouse import traffic as T
from warehouse import verify as V

OUT = "results/warehouse/phaseB0c"
CELLS = {"m30": ("medium", 30), "m90": ("medium", 90), "l60": ("large", 60)}
SEEDS = range(5)
CAL_SEEDS = (100, 101)
CAL_ROUNDS = 5
R1 = (0.10, 0.002)                              # p, q: about 11 % of steps lost to slips, 2.5 % to stops
FIRST = {"p": 0.04, "q": 0.01}                  # where the calibration of r2 starts
ORDERS_PER_ROBOT = 5
PLANNERS = ("pibt", "prioritized")
ARMS = ("a", "b", "c", "d", "e", "e0", "f")
CAP = 3


def path(*parts):
    return f"{OUT}/" + "_".join(str(p) for p in parts)


def world(cell, seed):
    size, robots = CELLS[cell]
    wh = W.Warehouse(dataclasses.replace(W.SIZES[size], robots=robots))
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    return wh, cat, W.batch_orders(wh, cat, ORDERS_PER_ROBOT * robots, seed)


def regime_of(name, cell, wh):
    if name == "r1":
        return E.Flat(*R1)
    if name == "none":
        return E.Flat()
    cal = json.load(open(path("cal", cell) + ".json"))
    return E.StateDelay(wh, cal["s"], cal["q_bad"])


def judged(wh, cat, orders, ep, rotation=True):
    return V.verify(ep["traj"], wh.grid, P.as_dict(orders), cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"], capacity=CAP, rotation=rotation)


def edges_of(wh, ep):
    """What the robots lost per edge, apart from the turns: everything, and traffic alone, without their own delays."""
    log = ep["log"]
    return T.edge_stats(wh, ep["traj"], ep["targets"], skip=log == E.Execution.TURN).export(), T.edge_stats(wh, ep["traj"], ep["targets"], skip=log > 0).export()


def load_edges(cell, regime, planner, arm, seed, part):
    return dict(np.load(path("edges", cell, regime, planner, arm, seed, part) + ".npz"))


def static_map(cell, regime, planner, seeds, part="all"):
    return T.pooled_extra([load_edges(cell, regime, planner, "a", s, part) for s in seeds])


def fit_gamma(cell, regime, planner, seed):
    """As phaseB0_margins.fit_gamma: the real loss per edge in one seed against the map of the others, over the seeds other than this."""
    rest = [s for s in SEEDS if s != seed]
    xy = xx = 0.0
    for s in rest:
        x = load_edges(cell, regime, planner, "a", s, "all")
        table = static_map(cell, regime, planner, [q for q in rest if q != s])
        for (u, v), n, y in zip(x["edges"].tolist(), x["edge_passes"].tolist(), x["edge_delta"].tolist()):
            m = table.get((u, v), 0.0)
            xy += n * m * y
            xx += n * m * m
    return xy / xx if xx else 0.0


def known(wh, regime, traffic, kinds, tries):
    """extra[(u, v)] per kind: the traffic map plus the expected delay of the function, charged to the cell left."""
    cost = regime.expected(wh, tries)
    out = []
    for k in kinds:
        by_cell = np.mean([cost[j] for j in k], axis=0)
        out.append({(u, v): traffic.get((u, v), 0.0) + float(by_cell[u]) for u in wh.free for v in wh.neighbours[u]})
    return out


def distances(arm, cell, regime_name, regime, planner, seed, wh, extra=None):
    """(metric of layer 1 and of PIBT, the same for prioritized, the settings worth recording)."""
    if arm == "a":
        return None, None, {}
    if arm == "b":
        return E.TurnMetric(wh), None, {}
    rest = [s for s in SEEDS if s != seed]
    if arm == "f":
        return E.TurnMetric(wh, extra), T.EdgeMetric(wh, extra, symmetric=True), {}
    if arm in ("c", "d"):
        gamma = fit_gamma(cell, regime_name, planner, seed) if arm == "d" else 1.0
        extra = {k: gamma * v for k, v in static_map(cell, regime_name, planner, rest).items()}
        return E.TurnMetric(wh, extra), T.EdgeMetric(wh, extra, symmetric=True), {"gamma": gamma}
    traffic = static_map(cell, regime_name, planner, rest, "traffic")
    tries = sum(load_edges(cell, regime_name, planner, "a", s, "all")["tries"] for s in rest) if regime.counts else None
    every = list(range(regime.kinds))
    shared = known(wh, regime, traffic, [every], tries)[0]
    fixed = T.EdgeMetric(wh, shared, symmetric=True)
    if arm == "e0" or regime.kinds == 1:
        return E.TurnMetric(wh, shared), fixed, {}
    each = [E.TurnMetric(wh, x) for x in known(wh, regime, traffic, [[k] for k in every], tries)]
    return E.Typed(each, E.TurnMetric(wh, shared), regime.kind), fixed, {}


def play(cell, regime_name, planner, seed, metric, route_metric):
    wh, cat, orders = world(cell, seed)
    regime = regime_of(regime_name, cell, wh)
    t0 = time.perf_counter()
    groups = B.savings(orders, B.TripCost(wh, metric or wh, sim.SERVICE), CAP)
    cw = time.perf_counter() - t0
    ex = E.Execution(wh, wh.layout.robots, regime, seed)
    args = {"metric": metric} if planner == "pibt" else ({"metric": route_metric} if route_metric else {})
    ep = sim.run_episode(wh, cat, orders, "lkh", planner, seed, metric=metric, dispatcher=D.TripHungarian(wh, groups, metric=metric), planner_args=args, execution=ex)
    rep = judged(wh, cat, orders, ep)
    assert rep["ok"] and rep["delivered"] == rep["orders"], (cell, regime_name, planner, seed, rep["problems"], rep["delivered"])
    return wh, cat, orders, ex, ep, rep, cw


def free_flow(wh, cat, orders, seed, ep, metric):
    """The same trips, each robot alone, with rotation and no delays."""
    ex = E.Execution(wh, wh.layout.robots, E.Flat(), seed, ghost=True)
    ff = sim.run_episode(wh, cat, orders, "lkh", E.TurnFree(wh), seed, dispatcher=D.FixedPlan(D.plan_of(ep["events"])), metric=metric, execution=ex)
    rep = judged(wh, cat, orders, ff)
    bad = {k: v for k, v in rep["problems"].items() if v and k not in ("vertex_conflicts", "edge_conflicts")}
    assert not bad and rep["delivered"] == rep["orders"], bad
    return rep["makespan"]


def summary(rep, ex, ep, cw, **extra):
    return {"makespan": rep["makespan"], "mean_delay": rep["mean_delay"], "step_ms": rep["step_ms_mean"], "cw_seconds": cw, **ex.stats(),
            "fallbacks": ep["planner"]["fallbacks"], "calls": ep["planner"]["calls"], "seconds": ep["seconds"], "seq_seconds": ep["seq_seconds"], **extra}


def keep_edges(wh, ex, ep, cell, regime, planner, arm, seed):
    everything, traffic = edges_of(wh, ep)
    if ex.tries is not None:
        everything["tries"] = ex.tries
    np.savez_compressed(path("edges", cell, regime, planner, arm, seed, "all") + ".npz", **everything)
    np.savez_compressed(path("edges", cell, regime, planner, arm, seed, "traffic") + ".npz", **traffic)


def run(job):
    cell, regime, planner, arm, seed = job
    out = path(cell, regime, planner, arm, seed) + ".json"
    if os.path.exists(out):
        return out
    t0 = time.perf_counter()
    wh, cat, orders = world(cell, seed)
    if arm == "f":
        runs = []
        extra = T.pooled_extra([load_edges(cell, regime, planner, "b", seed, "all")])
        for _ in range(3):
            metric, route, _ = distances("f", cell, regime, None, planner, seed, wh, extra)
            w, c, o, ex, ep, rep, cw = play(cell, regime, planner, seed, metric, route)
            runs.append((rep["makespan"], len(runs), rep["mean_delay"], ex, ep, rep, cw, metric))
            extra = T.pooled_extra([edges_of(w, ep)[0]])
        _, _, _, ex, ep, rep, cw, metric = min(runs, key=lambda q: q[:2])
        wh, cat, orders = w, c, o
        info = {"iterations": [q[0] for q in runs], "iteration_delays": [q[2] for q in runs]}
    else:
        metric, route, info = distances(arm, cell, regime, regime_of(regime, cell, wh), planner, seed, wh)
        wh, cat, orders, ex, ep, rep, cw = play(cell, regime, planner, seed, metric, route)
        if arm in ("a", "b"):
            keep_edges(wh, ex, ep, cell, regime, planner, arm, seed)
    rec = {"cell": cell, "regime": regime, "planner": planner, "arm": arm, "seed": seed, "ok": True, "orders": rep["orders"], "robots": wh.layout.robots,
           **summary(rep, ex, ep, cw, **info), "free_flow": free_flow(wh, cat, orders, seed, ep, metric), "elapsed": time.perf_counter() - t0, "commit": P.commit()}
    P.write(out, rec)
    return out


def rates(rows):
    moves = sum(r["moves"] for r in rows)
    return sum(r["slips"] for r in rows) / moves, sum(r["stops"] for r in rows) / moves


def cal_run(job):
    cell, name, s, q, seed = job
    wh, cat, orders = world(cell, seed)
    regime = E.Flat(*R1) if name == "r1" else E.StateDelay(wh, s, q)
    ex = E.Execution(wh, wh.layout.robots, regime, seed)
    groups = B.savings(orders, B.TripCost(wh, wh, sim.SERVICE), CAP)
    ep = sim.run_episode(wh, cat, orders, "lkh", "pibt", seed, dispatcher=D.TripHungarian(wh, groups), execution=ex)
    return {**ex.stats(), "makespan": float(max(e["end"] for e in ep["events"] if e["kind"] == "deliver"))}


def calibrate(pool, cell):
    """s and q_bad, so that slips and stops per move match r1 on the PIBT baseline."""
    out = path("cal", cell) + ".json"
    if os.path.exists(out):
        return
    goal = rates(pool.map(cal_run, [(cell, "r1", 0, 0, k) for k in CAL_SEEDS]))
    s, q, history = FIRST["p"], FIRST["q"], []
    for _ in range(CAL_ROUNDS):
        got = rates(pool.map(cal_run, [(cell, "r2", s, q, k) for k in CAL_SEEDS]))
        history.append({"s": s, "q_bad": q, "slip": got[0], "stop": got[1]})
        s, q = s * goal[0] / got[0], q * goal[1] / got[1]
        print(f"{time.strftime('%H:%M:%S')} {cell} {history[-1]} goal {goal}", flush=True)
    best = min(history, key=lambda h: abs(h["slip"] / goal[0] - 1) + abs(h["stop"] / goal[1] - 1))
    P.write(out, {"s": best["s"], "q_bad": best["q_bad"], "goal": goal, "history": history, "commit": P.commit()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("calibrate", "run"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cells", nargs="*", default=list(CELLS))
    p.add_argument("--regimes", nargs="*", default=["r1", "r2"])
    p.add_argument("--planners", nargs="*", default=["pibt"])
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with Pool(args.workers) as pool:
        if args.what == "calibrate":
            for c in args.cells:
                calibrate(pool, c)
            return
        jobs = [(c, r, q, a, k) for c in args.cells for r in args.regimes for q in args.planners for a in args.arms for k in args.seeds if not (a == "e0" and r == "r1")]    # r1 has no types
        for n, r in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} {r}", flush=True)


if __name__ == "__main__":
    main()
