"""Layer 1: in which order a robot visits the shelves of its order, and which robot takes which order.

The visit order is a TSP with fixed ends: it starts at the robot's cell and ends at the station. Distances are shortest
paths on the empty grid (BFS, cached in Warehouse.dist_to). Three solvers:
    nearest   nearest neighbour;
    two_opt   nearest neighbour, then 2-opt on the open path (the ends never move);
    lkh       LKH-3 on an explicit matrix, with a dummy city joined only to the two ends, so that the closed tour read
              from the dummy is the open path wanted. tsp_gnn.lkh3.solve only takes coordinates, so the problem file is
              written here; the binary and the tour reader are the ones of tsp_gnn.lkh3.
exact (Held-Karp) is the reference of the tests.
"""

import itertools
import os
import subprocess
import time

from tsp_gnn import lkh3


def path_length(dist, path):
    return sum(dist[a][b] for a, b in zip(path, path[1:]))


def matrix(wh, start, cells, end):
    """Distances among [start, *cells, end], as a list of lists."""
    pts = [start] + list(cells) + [end]
    maps = [wh.dist_to(p) for p in pts]
    return [[m[q] for q in pts] for m in maps]


def nearest(dist):
    """Visit order of nodes 1..m-2 of a matrix where 0 is the start and m-1 the end."""
    todo, at, out = set(range(1, len(dist) - 1)), 0, []
    while todo:
        at = min(todo, key=lambda k: (dist[at][k], k))
        todo.remove(at)
        out.append(at)
    return out


def two_opt(dist):
    path = [0] + nearest(dist) + [len(dist) - 1]
    improved = True
    while improved:
        improved = False
        for i in range(len(path) - 2):
            for j in range(i + 2, len(path) - 1):
                a, b, c, e = path[i], path[i + 1], path[j], path[j + 1]
                if dist[a][c] + dist[b][e] < dist[a][b] + dist[c][e]:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    improved = True
    return path[1:-1]


def exact(dist):
    """Held-Karp on the open path: the optimum, for the tests and for orders too small for LKH-3."""
    m = len(dist) - 2
    if m == 0:
        return []
    best = {(1 << k, k): (dist[0][k + 1], None) for k in range(m)}
    for size in range(2, m + 1):
        for sub in itertools.combinations(range(m), size):
            mask = sum(1 << k for k in sub)
            for k in sub:
                prev = mask ^ (1 << k)
                best[(mask, k)] = min(((best[(prev, q)][0] + dist[q + 1][k + 1], q) for q in sub if q != k),
                                      key=lambda x: (x[0], x[1]))
    full = (1 << m) - 1
    k = min(range(m), key=lambda q: (best[(full, q)][0] + dist[q + 1][m + 1], q))
    out, mask = [], full
    while k is not None:
        out.append(k + 1)
        prev = best[(mask, k)][1]
        mask ^= 1 << k
        k = prev
    return out[::-1]


LKH_PARAMS = {"RUNS": 1, "MAX_TRIALS": 50, "SEED": 1, "TRACE_LEVEL": 0}


def lkh(dist, workdir=None):
    """Open path by LKH-3. Below 4 stops the exact solver is used: LKH-3 needs a few cities to start, and there the
    optimum costs nothing."""
    m = len(dist)
    if m - 2 < 4:
        return exact(dist)
    workdir = workdir or os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wh_lkh_{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    problem, par, out = (os.path.join(workdir, x) for x in ("p.atsp", "p.par", "p.tour"))
    # the dummy (node m) is joined to the start (0) and the end (m - 1) at cost 0, and to the rest at a cost no tour pays
    big = 1 + 2 * m * max(map(max, dist))
    full = [row + [big] for row in dist] + [[big] * (m + 1)]
    full[0][m] = full[m][0] = full[m - 1][m] = full[m][m - 1] = 0
    full[0][m - 1] = full[m - 1][0] = big                 # the ends meet only through the dummy
    with open(problem, "w") as f:
        f.write(f"NAME : order\nTYPE : TSP\nDIMENSION : {m + 1}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                "EDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        f.write("".join(" ".join(map(str, row)) + "\n" for row in full) + "EOF\n")
    with open(par, "w") as f:
        f.write(f"PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {out}\n")
        f.write("".join(f"{k} = {v}\n" for k, v in LKH_PARAMS.items()))
    if os.path.exists(out):
        os.remove(out)
    subprocess.run([lkh3.LKH_BIN, par], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=True)
    tour = lkh3.read_tour(out).tolist()                   # starts at node 0
    k = tour.index(m)
    assert tour[k - 1] == m - 1 or tour[(k + 1) % len(tour)] == m - 1, "the dummy is not between the ends"
    if tour[1] == m:                                      # read the other way round: 0, dummy, end, ...
        tour = [0] + tour[:0:-1]
    assert tour[-1] == m and tour[-2] == m - 1
    return tour[1:-2]


SOLVERS = {"nn": nearest, "2opt": two_opt, "lkh": lkh, "exact": exact}


def sequence(wh, start, stops, station, method, cache=None):
    """(stops in visit order, seconds). stops: cells; repeated cells are kept (two shelves can share an access cell).
    The cache is keyed by the problem, so a repeated problem costs nothing."""
    key = (start, tuple(stops), station, method)
    if cache is not None and key in cache:
        return cache[key], 0.0
    t0 = time.perf_counter()
    dist = matrix(wh, start, stops, station)
    order = SOLVERS[method](dist)
    out = [k - 1 for k in order]                         # positions in stops
    sec = time.perf_counter() - t0
    if cache is not None:
        cache[key] = out
    return out, sec


def open_path(metric, stops, station, method="lkh", cache=None):
    """The order on its own, from no particular start: (first cell, length to the station). Node 0 is a free start."""
    key = ("open", tuple(stops), station, method)
    if cache is not None and key in cache:
        return cache[key]
    pts = list(stops) + [station]
    maps = [metric.dist_to(p) for p in pts]
    dist = [[0] * (len(pts) + 1)] + [[0] + [m[q] for q in pts] for m in maps]
    last = max(map(max, dist))
    dist[0][-1] = dist[-1][0] = 2 * last + 1              # the free start does not jump to the station
    order = SOLVERS[method](dist)
    out = (stops[order[0] - 1], path_length(dist, [0] + order + [len(dist) - 1]))
    if cache is not None:
        cache[key] = out
    return out


def route(metric, wh, start, group, method, cache=None):
    """The trip of a robot that takes the orders of group: ([(shelf, owner)], [(station, order id)], seconds). The owner is
    an order id, or a tuple for a shelf of both orders. With two orders the shelves come first, then both stations, in the
    cheaper of the two ways, so no delivery precedes a pick."""
    if len(group) == 1:
        o = group[0]
        idx, sec = sequence(metric, start, [wh.access[s] for s in o.shelves], o.station, method, cache)
        return [(o.shelves[k], o.oid) for k in idx], [(o.station, o.oid)], sec
    a, b = group
    shelves = sorted(set(a.shelves) | set(b.shelves))
    owner = {s: (a.oid, b.oid) if s in a.shelves and s in b.shelves else (a.oid if s in a.shelves else b.oid) for s in shelves}
    cells = [wh.access[s] for s in shelves]
    best, spent = None, 0.0
    for x, y in ((a, b), (b, a)) if a.station != b.station else ((a, b),):
        idx, sec = sequence(metric, start, cells, x.station, method, cache)
        spent += sec
        d = matrix(metric, start, cells, x.station)
        size = path_length(d, [0] + [k + 1 for k in idx] + [len(d) - 1])
        if best is None or size < best[0]:
            best = (size, idx, x, y)
    _, idx, x, y = best
    return [(shelves[k], owner[shelves[k]]) for k in idx], [(x.station, x.oid), (y.station, y.oid)], spent


def assign_robot(wh, free_robots, positions, stop_cells):
    """The documented rule: the free robot closest to the order's shelf closest to it (ties: lowest robot id). The first
    shelf of the visit depends on the robot, so "the first shelf" is read as the shelf nearest that robot. To be improved:
    the rule does not look at the other orders or at congestion."""
    maps = [wh.dist_to(c) for c in stop_cells]
    return min(free_robots, key=lambda r: (min(m[positions[r]] for m in maps), r))
