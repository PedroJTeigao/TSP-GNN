"""What the rules of a request do to the solvers. Each primitive of dsl.py has one way in:

    pick_order   SOP in LKH-3 for the trip (start and station fixed, precedences between its shelves)
    close        all day: the cells leave the grid of every layer, so LKH-3 only sees legs around them. With a window: entries are
                 forbidden in the planner from GUARD steps before it until it ends, and in a trip that meets it the legs into the corridor
                 cost FORBIDDEN and its shelves are visited last
    no_mix       a merge of Clarke & Wright that would join the two orders is refused
    robot_only   a mask in the Hungarian, on top of the fact that type B carries nothing heavy
    deadline     a weight in the Hungarian, a trip of its own, and a time window (TSPTW) in the visit order
    station      the order's station changes, so it is the fixed end of the trip

to_checks turns the same primitives into plain data for the verifier, which shares nothing with this file.
"""

import copy
import dataclasses
import os
import subprocess
import time

import numpy as np

from tsp_gnn import lkh3
from tsp_gnn import open_dsl as OD
from warehouse import dispatch as D
from warehouse import dsl as L
from warehouse import mapf as M
from warehouse import sequencing as S
from warehouse import traffic as T

GUARD = 40                                      # steps before a closure from which nobody enters: a stop lasts up to 20
FORBIDDEN = 1000                                # steps
WEIGHT = 10 ** 6                                # of a deadline in the Hungarian, above any distance
BLOCKED = 10 ** 9
EXACT_STOPS = 9                                 # up to here the order is exact, and LKH-3 would need more nodes anyway
BOTH_STOPS = 14                                 # precedences and a window together: LKH-3 takes one, so exact
SOP_FORBIDDEN = 10 ** 6


def closed_grid(wh, cells):
    """A copy of the warehouse without those cells, with its own distances."""
    out = copy.copy(wh)
    cells = set(cells)
    out.neighbours = [[] if c in cells else [u for u in nb if u not in cells] for c, nb in enumerate(wh.neighbours)]
    out.free = [c for c in wh.free if c not in cells]
    out.parking = [c for c in wh.parking if c not in cells]
    out._dist = {}
    return out


def feasible(path, pairs):
    pos = {v: k for k, v in enumerate(path)}
    return all(pos[a] < pos[b] for a, b in pairs if a in pos and b in pos)


def exact_rules(dist, pairs, latest):
    """Held-Karp on the open path with precedences (nodes) and latest arrivals (node -> time). None if nothing fits."""
    m = len(dist) - 2
    need = {k: 0 for k in range(m)}
    for a, b in pairs:
        need[b - 1] |= 1 << (a - 1)
    best = {}
    for k in range(m):
        t = dist[0][k + 1]
        if not need[k] and t <= latest.get(k + 1, BLOCKED):
            best[(1 << k, k)] = (t, None)
    for mask in range(1, 1 << m):
        for k in range(m):
            cur = best.get((mask, k))
            if cur is None:
                continue
            for q in range(m):
                bit = 1 << q
                if mask & bit or need[q] & ~mask:
                    continue
                t = cur[0] + dist[k + 1][q + 1]
                if t > latest.get(q + 1, BLOCKED):
                    continue
                if (mask | bit, q) not in best or t < best[(mask | bit, q)][0]:
                    best[(mask | bit, q)] = (t, k)
    full = (1 << m) - 1
    ends = [(best[(full, k)][0] + dist[k + 1][m + 1], k) for k in range(m) if (full, k) in best]
    ends = [e for e in ends if e[0] <= latest.get(m + 1, BLOCKED)]
    if not ends:
        return None
    k, mask, out = min(ends)[1], full, []
    while k is not None:
        out.append(k + 1)
        prev = best[(mask, k)][1]
        mask ^= 1 << k
        k = prev
    return out[::-1]


def greedy_rules(dist, pairs):
    """Nearest stop among those whose predecessors are done: the fallback of an SOP."""
    todo, at, out = set(range(1, len(dist) - 1)), 0, []
    while todo:
        ok = [k for k in todo if not any(a in todo and b == k for a, b in pairs)]
        at = min(ok or todo, key=lambda k: (dist[at][k], k))
        todo.remove(at)
        out.append(at)
    return out


def _lkh(text, m, workdir, timeout):
    os.makedirs(workdir, exist_ok=True)
    problem, par, out = (os.path.join(workdir, x) for x in ("r.lkh", "r.par", "r.tour"))
    open(problem, "w").write(text)
    with open(par, "w") as f:
        f.write(f"PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {out}\n" + "".join(f"{k} = {v}\n" for k, v in S.LKH_PARAMS.items()))
    if os.path.exists(out):
        os.remove(out)
    try:
        subprocess.run([lkh3.LKH_BIN, par], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:                   # LKH-3 can spin inside a trial, as in phase 5
        return None
    return lkh3.read_tour(out).tolist() if os.path.exists(out) else None


def sop(dist, pairs, workdir=None):
    """The open path by LKH-3 in SOP mode: node 0 first, the last node last; entry (b, a) = -1 says a comes before b."""
    workdir = workdir or os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wh_sop_{os.getpid()}")
    n = len(dist)
    W = np.full((n, n), SOP_FORBIDDEN, dtype=np.int64)
    W[:n - 1, 1:] = np.asarray(dist, dtype=np.int64)[:n - 1, 1:]
    W[:, 0] = SOP_FORBIDDEN
    W[n - 1, 0] = 0
    W[0, n - 1] = SOP_FORBIDDEN
    np.fill_diagonal(W, 0)
    for a, b in pairs:
        W[b, a] = -1
    W[1:n - 1, 0] = -1                                  # the start precedes all: the form LKH-3 itself writes
    W[n - 1, 1:n - 1] = -1
    W[n - 1, 0] = -1
    body = "\n".join(" ".join(map(str, row)) for row in W.tolist())
    text = f"NAME : trip\nTYPE : SOP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n{n}\n{body}\nEOF\n"
    tour = _lkh(text, n, workdir, 20)
    if tour is None or tour[-1] != n - 1 or not feasible(tour, pairs):
        return None
    return tour[1:-1]


def tsptw(dist, latest, workdir=None):
    """The open path by LKH-3 in TSPTW mode. Closing the tour costs nothing from the station and too much from anywhere else."""
    workdir = workdir or os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wh_tw_{os.getpid()}")
    n = len(dist)
    W = np.asarray(dist, dtype=np.int64).copy()
    top = int(W.max()) * n + 1
    W[:, 0] = top
    W[n - 1, 0] = 0
    W[0, n - 1] = top
    W[n - 1, 1:n - 1] = top
    np.fill_diagonal(W, 0)
    body = "\n".join(" ".join(map(str, row)) for row in W.tolist())
    wins = "".join(f"{v + 1} 0 {int(latest.get(v, BLOCKED))}\n" for v in range(n))
    text = (f"NAME : trip\nTYPE : TSPTW\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n{body}\n"
            f"TIME_WINDOW_SECTION\n{wins}DEPOT_SECTION\n1\n-1\nEOF\n")
    tour = _lkh(text, n, workdir, 20)
    if tour is None or tour[-1] != n - 1:
        return None
    return tour[1:-1]


def ordered(dist, pairs, latest, cache, key):
    """Nodes 1..m-2 in visit order, under precedences between nodes and latest arrivals. A window that cannot be met is dropped."""
    if key in cache:
        return cache[key]
    m = len(dist) - 2
    pairs = sorted(pairs)
    if m == 0:
        out = []
    elif not pairs and not latest:
        out = S.SOLVERS["lkh"](dist)
    elif m <= EXACT_STOPS or (pairs and latest and m <= BOTH_STOPS):
        out = exact_rules(dist, pairs, latest) or exact_rules(dist, pairs, {})
    elif latest and not pairs:
        out = tsptw(dist, latest) or S.SOLVERS["lkh"](dist)
    else:
        out = sop(dist, pairs) or greedy_rules(dist, pairs)
    if out is None:                                     # no order fits the precedences: the validator should have stopped this
        out = greedy_rules(dist, pairs)
    cache[key] = out
    return out


class Rules:
    """The accepted primitives of one request, in the form the solvers read."""

    def __init__(self, world, prims):
        self.w = world
        self.wh = wh = world.wh
        self.prims = prims
        by = lambda t: [p for p in prims if p["type"] == t]
        self.before = set()
        for p in by("pick_order"):
            self.before |= {(a, b) for a in p["first"] for b in p["then"] if a != b}
        self.shut = set()
        self.windows = []                               # (cells, t1, t2)
        for p in by("close"):
            cells = world.corridors[p["corridor"]]
            if p["t1"] == 0 and p["t2"] == L.FOREVER:
                self.shut.update(cells)
            else:
                self.windows.append((set(cells), p["t1"], p["t2"]))
        self.apart = {tuple(p["pair"]) for p in by("no_mix")}
        self.only = {p["order"]: p["robot"] for p in by("robot_only")}
        self.deadline = {}
        for p in by("deadline"):
            self.deadline[p["order"]] = min(p["t"], self.deadline.get(p["order"], L.FOREVER))
        self.station = {p["order"]: world.stations[p["station"]] for p in by("station")}
        self.grid = closed_grid(wh, self.shut) if self.shut else wh
        self.mouths = [[(u, v) for v in cells for u in wh.neighbours[v] if u not in cells] for cells, _, _ in self.windows]
        self.costly = {}
        self.spare = {}

    def orders(self):
        """The orders of the episode, with their stations changed by the rules."""
        return [dataclasses.replace(o, station=self.station.get(o.oid, o.station)) for o in self.w.orders]

    def allowed(self, group):
        """The robot types that may take this trip."""
        out = {"A", "B"}
        for o in group:
            if any(self.w.shelf_kind[s] == "heavy" for s in o.shelves):
                out.discard("B")
            if o.oid in self.only:
                out &= {self.only[o.oid]}
        return out

    def penalty(self, a, b):
        """What a merge costs in Clarke & Wright: everything if the rules refuse it."""
        if any(o.oid in self.deadline for o in a + b) or not self.allowed(a + b):
            return float("inf")
        if any((min(x.oid, y.oid), max(x.oid, y.oid)) in self.apart for x in a for y in b):
            return float("inf")
        return 0.0

    def meeting(self, t, span):
        """The windows that a trip from t, of about span steps, runs into."""
        return [k for k, (_, t1, t2) in enumerate(self.windows) if t < t2 and t + span >= t1 - GUARD]

    def metric(self, keys):
        """The grid with the legs into those corridors priced out."""
        keys = tuple(sorted(keys))
        out = self.costly.get(keys)
        if out is None:
            extra = {e: FORBIDDEN for k in keys for e in self.mouths[k]}
            out = self.costly[keys] = T.EdgeMetric(self.grid, extra)
        return out

    def route(self, metric, wh, start, group, method, cache, t):
        """As sequencing.route, with the rules. Stops first, then the stations, in the cheaper way."""
        tick = time.perf_counter()
        shelves = sorted({s for o in group for s in o.shelves})
        owner = {}
        for s in shelves:
            who = tuple(o.oid for o in group if s in o.shelves)
            owner[s] = who if len(who) > 1 else who[0]
        cells = [wh.access[s] for s in shelves]
        base = S.matrix(metric, start, cells, group[0].station)
        span = S.path_length(base, list(range(len(base)))) + L.SERVICE * len(cells)
        keys = self.meeting(t, span)
        table = self.metric(keys) if keys else metric
        unit = T.SCALE if keys else 1
        late = [s for s in shelves if any(wh.access[s] in self.windows[k][0] for k in keys)]
        rule_pairs = [(a, b) for a in shelves for b in shelves if (a, b) in self.before]
        extra = [(a, b) for a in shelves if a not in late for b in late]
        best = None
        for way in S.delivery_orders(metric, group):
            d = S.matrix(table, start, cells, way[0].station)
            d = [[x + L.SERVICE * unit * (0 < i < len(d) - 1) for x in row] for i, row in enumerate(d)]
            index = {s: k + 1 for k, s in enumerate(shelves)}
            pairs = {(index[a], index[b]) for a, b in rule_pairs}
            soft = pairs | {(index[a], index[b]) for a, b in extra}
            if not OD._has_cycle(soft):
                pairs = soft
            latest = {}
            dead = [self.deadline[o.oid] for o in way if o.oid in self.deadline]
            if dead:
                latest[len(d) - 1] = unit * (min(dead) - t - L.SERVICE)
                latest = {k: v for k, v in latest.items() if v > 0}
            key = ("rules", id(table), start, tuple(cells), way[0].station, tuple(sorted(pairs)), tuple(sorted(latest.items())))
            order = ordered(d, pairs, latest, cache, key)
            size = S.path_length(d, [0] + order + [len(d) - 1])
            size += sum(table.dist_to(y.station)[x.station] for x, y in zip(way, way[1:]))
            if best is None or size < best[0]:
                best = (size, order, way)
        _, order, way = best
        return [(shelves[k - 1], owner[shelves[k - 1]]) for k in order], [(o.station, o.oid) for o in way], time.perf_counter() - tick


class RuleHungarian(D.TripHungarian):
    """Deadlines go first, and a mask keeps a trip away from robots that may not take it."""

    def __init__(self, wh, clusters, rules, types, metric=None, cache=None):
        super().__init__(wh, clusters, metric=metric, cache=cache)
        self.rules, self.types = rules, types

    def costs(self, free, trips, pos):
        out = super().costs(free, trips, pos).astype(np.float64)
        for j, group in enumerate(trips):
            ok = self.rules.allowed(group)
            for i, r in enumerate(free):
                if self.types[r] not in ok:
                    out[i, j] = BLOCKED
            dead = [self.rules.deadline[o.oid] for o in group if o.oid in self.rules.deadline]
            if dead:
                out[:, j] = np.where(out[:, j] < BLOCKED, out[:, j] - WEIGHT + 100 * min(dead), BLOCKED)
        return out

    def choose(self, t, free, pending, pos):
        here = {o.oid for o in pending}
        trips = [g for g in self.trips if all(o.oid in here for o in g)]
        if not free or not trips:
            return []
        table = self.costs(free, trips, pos)
        rows, cols = D.linear_sum_assignment(table)
        out = [(free[i], trips[j]) for i, j in zip(rows, cols) if table[i, j] < BLOCKED]
        for _, group in out:
            for o in group:
                pending.remove(o)
        return out


class RulePlanner(M.Prioritized):
    """Prioritized on the grid of the rules, with the mouths of a closing corridor forbidden from GUARD steps before it."""

    def __init__(self, rules, seed=0):
        super().__init__(rules.grid, seed=seed)
        self.rules = rules
        self.gates = self.entries

    def entries(self, t):
        out, shut = set(), set()
        for (cells, t1, t2), mouth in zip(self.rules.windows, self.rules.mouths):
            a, b = max(t1 - GUARD, t), min(t2, t + self.horizon + 1)
            out.update((u, v, k - t) for u, v in mouth for k in range(a, b))
            if t1 - GUARD <= t < t2:
                shut |= cells
        self.pibt.shut = shut
        return out


def to_checks(world, prims):
    """Plain data for verify.check_rule, primitive by primitive."""
    out = []
    for p in prims:
        t = p["type"]
        if t == "pick_order":
            out.append({"type": "pick_before", "first": p["first"], "then": p["then"]})
        elif t == "close":
            rule = {"type": "forbidden_cells", "cells": world.corridors[p["corridor"]], "from": p["t1"]}
            out.append(rule if p["t2"] == L.FOREVER else {**rule, "to": p["t2"]})
        elif t == "no_mix":
            out.append({"type": "no_mix", "a": p["pair"][0], "b": p["pair"][1]})
        elif t == "robot_only":
            out.append({"type": "robot_only", "order": p["order"], "robots": [i for i, x in enumerate(world.robots) if x == p["robot"]]})
        elif t == "deadline":
            out.append({"type": "deadline", "order": p["order"], "t": p["t"]})
        elif t == "station":
            out.append({"type": "station", "order": p["order"], "cell": world.stations[p["station"]]})
    return out


def world_checks(world):
    """The fact of the warehouse: type B carries nothing heavy."""
    ctx = L.Context(world)
    return [{"type": "robot_only", "order": o, "robots": [i for i, x in enumerate(world.robots) if x == "A"]} for o, v in ctx.kinds.items() if "heavy" in v]

