"""Heatmap-guided solving of the FCNF (phase 1, step 5) and the baselines without a network.

Every decoder returns a dict with
    cost            c·x + F·1[x > 0] of the final solution (arcs opened without flow are closed again,
                    which only lowers the cost); nan if no solution was found
    x, y            the solution (y = 1[x > 0])
    seconds         total time of the decoder (LPs + MILPs + heuristic), without the network
    first_feasible  True if no repair step was needed
    kept_frac       fraction of the arcs the MILP was allowed to open (1 for the full MILP)
    repairs         number of re-insertion rounds after an infeasible restricted problem
    milp_status     status of the last MILP / LP solved

Decoders with a heatmap p (one probability per arc):
    prune       keep the top q % arcs by p, plus the arcs needed for every supply to reach a demand
                and every demand to be reached from a supply (BFS check; a missing connection is
                added as the path of largest product of p, i.e. shortest in −log p); solve the
                restricted MILP; if it has no solution, put back the pruned arcs in blocks of 10 %
                of the arcs by decreasing p and solve again
    warm        full MILP with the warm start y = 1[p >= 0.5]
    delegate    slope scaling first; disagreement of each node = mean |p − y_heur| over its adjacent
                arcs; the MILP is free only on the arcs adjacent to the 30 % nodes with the largest
                disagreement and every other arc is fixed to the heuristic (closed or open); the
                heuristic solution is the warm start, so the restricted MILP is always feasible
    trust_region  trust region (Han et al. 2023, predict-and-search): the top q % arcs by p are free and
                at most delta of the other arcs may open (Σ y <= delta on them); no repair, so it can
                be infeasible (a terminal that needs more than delta arcs outside the prediction)
Baselines without a network:
    slope       slope scaling alone (Kim & Pardalos 1999)
    lp_round    LP relaxation, y = 1[y_lp >= 0.5], min-cost flow on those arcs, repaired as in prune
                by decreasing y_lp
    prune with random p (same q) is the control for prune.
"""

import time

import networkx as nx
import numpy as np

from fcnf_gnn.solver import OPEN_TOL, solve_fcnf

REPAIR_BLOCK = 0.10
DELEGATE_FRAC = 0.30


def _finish(inst, x, seconds, **extra):
    """Result dict from a flow x (y = 1[x > 0]); x None = no solution."""
    if x is None:
        return {"cost": np.nan, "x": None, "y": None, "seconds": seconds, **extra}
    x = np.where(np.asarray(x) > OPEN_TOL, x, 0.0)
    y = (x > 0).astype(np.int8)
    return {"cost": float(np.dot(inst["c"], x) + np.dot(inst["F"], y)), "x": x, "y": y, "seconds": seconds, **extra}


def min_cost_flow(inst, cost, open_mask=None):
    """Min Σ cost·x s.t. conservation and 0 <= x <= u, only on the arcs of open_mask (LP, HiGHS).
    Returns x or None if infeasible."""
    lin = dict(inst, c=np.asarray(cost, float), F=np.zeros(len(inst["src"])))
    closed = None if open_mask is None else np.where(~open_mask)[0]
    r = solve_fcnf(lin, fixed_zero=closed, relax=True)
    return r["x"] if r["status"] == "optimal" else None


# ---------------------------------------------------------------- connectivity repair

def terminals_connected(inst, keep):
    """(supplies that cannot reach any demand, demands not reachable from any supply) using arcs in keep."""
    b, src, dst = inst["b"], inst["src"], inst["dst"]
    g = nx.DiGraph()
    g.add_nodes_from(range(len(b)))
    g.add_edges_from(zip(src[keep].tolist(), dst[keep].tolist()))
    sup, dem = set(np.where(b > 0)[0].tolist()), set(np.where(b < 0)[0].tolist())
    bad_sup = [s for s in sup if not (nx.descendants(g, s) & dem)]
    bad_dem = [d for d in dem if not (nx.ancestors(g, d) & sup)]
    return bad_sup, bad_dem


def connect_terminals(inst, p, keep):
    """Add to keep (bool mask) the most likely paths so that every terminal is connected (see prune).
    Returns the new mask and the number of arcs added."""
    keep = keep.copy()
    src, dst = inst["src"], inst["dst"]
    weight = -np.log(np.clip(p, 1e-9, 1.0))
    full = nx.DiGraph()
    for a, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
        full.add_edge(i, j, w=0.0 if keep[a] else float(weight[a]), arc=a)  # kept arcs are free
    added = 0
    for _ in range(len(inst["b"])):
        bad_sup, bad_dem = terminals_connected(inst, keep)
        if not bad_sup and not bad_dem:
            break
        sup, dem = np.where(inst["b"] > 0)[0].tolist(), np.where(inst["b"] < 0)[0].tolist()
        if bad_sup:                                  # path from this supply to the closest demand
            dist, paths = nx.single_source_dijkstra(full, bad_sup[0], weight="w")
            target = min((d for d in dem if d in dist), key=dist.get)
            path = paths[target]
        else:                                        # path from the closest supply to this demand
            dist, paths = nx.single_source_dijkstra(full.reverse(copy=False), bad_dem[0], weight="w")
            target = min((s for s in sup if s in dist), key=dist.get)
            path = paths[target][::-1]
        for i, j in zip(path[:-1], path[1:]):
            a = full[i][j]["arc"]
            if not keep[a]:
                keep[a] = True
                full[i][j]["w"] = 0.0
                added += 1
    return keep, added


# ---------------------------------------------------------------- decoders

def prune(inst, p, q, time_limit=None):
    """Restricted MILP on the top q % arcs (q in percent) plus the connectivity repair."""
    start = time.perf_counter()
    m = len(inst["src"])
    order = np.argsort(-np.asarray(p), kind="stable")
    keep = np.zeros(m, dtype=bool)
    keep[order[:int(np.ceil(q / 100 * m))]] = True
    keep, bfs_added = connect_terminals(inst, p, keep)
    repairs = 0
    while True:
        r = solve_fcnf(inst, time_limit=time_limit, fixed_zero=np.where(~keep)[0])
        if r["x"] is not None or keep.all():
            break
        repairs += 1                                 # no solution: put back the next block by p
        missing = [a for a in order if not keep[a]]
        keep[missing[:int(np.ceil(REPAIR_BLOCK * m))]] = True
    return _finish(inst, r["x"], time.perf_counter() - start, first_feasible=repairs == 0,
                   kept_frac=float(keep.mean()), repairs=repairs, bfs_added=bfs_added, milp_status=r["status"])


def trust_region(inst, p, q, delta, time_limit=None):
    """MILP with the top q % arcs free and at most delta open arcs among the others.
    outside_used = open arcs of the final solution outside the top q % (-1 without a solution);
    proven = the restricted MILP was solved to optimality (not stopped by the time limit)."""
    start = time.perf_counter()
    m = len(inst["src"])
    order = np.argsort(-np.asarray(p), kind="stable")
    outside = order[int(np.ceil(q / 100 * m)):]
    r = solve_fcnf(inst, time_limit=time_limit, budget=(outside, delta))
    out = _finish(inst, r["x"], time.perf_counter() - start)
    used = -1 if out["x"] is None else int(out["y"][outside].sum())
    out.update(first_feasible=r["x"] is not None, kept_frac=1 - len(outside) / m, repairs=0, outside_used=used,
               proven=r["status"] == "optimal", milp_status=r["status"])
    return out


def warm(inst, p, time_limit=None, threshold=0.5):
    """Full MILP with warm start y = 1[p >= threshold]. first_feasible = that y admits a flow."""
    start = time.perf_counter()
    y0 = (np.asarray(p) >= threshold).astype(float)
    r = solve_fcnf(inst, time_limit=time_limit, warm_start=y0)
    seconds = time.perf_counter() - start
    feasible_start = min_cost_flow(inst, inst["c"], y0 > 0) is not None      # checked outside the timing
    return _finish(inst, r["x"], seconds, first_feasible=feasible_start, kept_frac=1.0, repairs=0,
                   milp_status=r["status"])


def slope_scaling(inst, max_iter=50, tol=1e-6):
    """Dynamic slope scaling (Kim & Pardalos 1999): min-cost flow with cost c + F / x̄ (x̄ = previous
    flow; c + F / u at the start, unchanged where x̄ = 0); returns the best true-cost solution."""
    start = time.perf_counter()
    c, F, u = (np.asarray(inst[k], float) for k in ("c", "F", "u"))
    slope = c + F / u
    best, prev, iters = None, None, 0
    for iters in range(1, max_iter + 1):
        x = min_cost_flow(inst, slope)
        if x is None:
            break
        sol = _finish(inst, x, 0.0)
        if best is None or sol["cost"] < best["cost"]:
            best = sol
        if prev is not None and np.abs(x - prev).max() <= tol * max(1.0, u.max()):
            break
        prev = x
        pos = x > OPEN_TOL
        slope[pos] = c[pos] + F[pos] / x[pos]
    out = _finish(inst, None if best is None else best["x"], time.perf_counter() - start)
    out.update(first_feasible=best is not None, kept_frac=1.0, repairs=0, iterations=iters, milp_status="lp")
    return out


def delegate(inst, p, time_limit=None, frac=DELEGATE_FRAC, heuristic=None):
    """MILP free only around the nodes where the heatmap disagrees most with slope scaling."""
    start = time.perf_counter()
    heur = heuristic if heuristic is not None else slope_scaling(inst)
    heur_seconds = heur["seconds"] if heuristic is not None else 0.0     # reused: add its time
    y_h = heur["y"].astype(float)
    n, m = len(inst["b"]), len(inst["src"])
    diff = np.abs(np.asarray(p) - y_h)
    total = np.bincount(inst["src"], diff, n) + np.bincount(inst["dst"], diff, n)
    count = np.bincount(inst["src"], minlength=n) + np.bincount(inst["dst"], minlength=n)
    disagreement = total / np.maximum(count, 1)
    hot = np.zeros(n, dtype=bool)
    hot[np.argsort(-disagreement, kind="stable")[:int(np.ceil(frac * n))]] = True
    free = hot[inst["src"]] | hot[inst["dst"]]
    fixed_zero = np.where(~free & (y_h == 0))[0]
    fixed_one = np.where(~free & (y_h == 1))[0]
    r = solve_fcnf(inst, time_limit=time_limit, fixed_zero=fixed_zero, fixed_one=fixed_one, warm_start=y_h)
    x = r["x"] if r["x"] is not None else heur["x"]
    out = _finish(inst, x, time.perf_counter() - start + heur_seconds)
    if out["cost"] > heur["cost"]:                    # never worse than the heuristic it starts from
        out = _finish(inst, heur["x"], out["seconds"])
    out.update(first_feasible=True, kept_frac=float(free.mean()), repairs=0, milp_status=r["status"])
    return out


def lp_round(inst):
    """LP relaxation -> y = 1[y_lp >= 0.5] -> min-cost flow on those arcs, repaired by decreasing y_lp."""
    start = time.perf_counter()
    m = len(inst["src"])
    lp = solve_fcnf(inst, relax=True)
    y_lp = np.asarray(lp["y"])
    order = np.lexsort((-np.asarray(lp["x"]), -y_lp))        # by y_lp, ties by flow
    keep = y_lp >= 0.5
    repairs = 0
    while True:
        x = min_cost_flow(inst, inst["c"], keep)
        if x is not None or keep.all():
            break
        repairs += 1
        missing = [a for a in order if not keep[a]]
        keep[missing[:int(np.ceil(REPAIR_BLOCK * m))]] = True
    return _finish(inst, x, time.perf_counter() - start, first_feasible=repairs == 0,
                   kept_frac=float(keep.mean()), repairs=repairs, milp_status="lp")
