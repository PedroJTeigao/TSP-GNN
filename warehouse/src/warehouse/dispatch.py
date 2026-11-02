"""Who takes which orders. Every rule answers choose(t, free, pending, pos) with [(robot, (order, ...))], and takes the
orders it gives out from pending.

    Nearest     the baseline: the oldest order goes to the free robot closest to one of its shelves;
    Hungarian   minimum-cost assignment of free robots to orders; the cost is the distance to the first shelf plus the
                length of the order on its own (LKH-3). Over the whole queue it also picks which orders go first; with
                fifo=True only the oldest orders enter, which isolates the matching. With reassign=True a robot that has
                not started to pick returns to the pool whenever a robot comes free or an order arrives;
    FixedPlan   a list of orders per robot, decided beforehand (the a posteriori search);
    Batcher     up to two orders per trip: the oldest one and the order that overlaps it most (rule "overlap"), or the one
                that adds the least walking (rule "near");
    Clusters    trips decided before the episode (batching.py), the first ones to the nearest free robot;
    Iterative   the nearest rule, then a robot changes order while that lowers its cost, or two robots swap;
    LiveCosts   Hungarian on the traffic seen so far in the episode: distance plus the decayed wait per edge, or distance plus
                gamma times the delay near the first goal, with gamma fitted on the trips already done.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from warehouse import sequencing as S
from warehouse import traffic as T


class Nearest:
    reassign = False

    def __init__(self, wh, metric=None):
        self.wh, self.metric = wh, metric or wh

    def choose(self, t, free, pending, pos):
        out = []
        while pending and free:
            o = pending.popleft()
            r = S.assign_robot(self.metric, free, pos, [self.wh.access[s] for s in o.shelves])
            free.remove(r)
            out.append((r, (o,)))
        return out


class Hungarian:
    def __init__(self, wh, metric=None, method="lkh", cache=None, reassign=False, fifo=False, use_length=True):
        self.wh, self.metric, self.method = wh, metric or wh, method
        self.cache = {} if cache is None else cache
        self.reassign, self.fifo, self.use_length = reassign, fifo, use_length
        self.maps = {}

    def profile(self, o):
        return S.open_path(self.metric, [self.wh.access[s] for s in o.shelves], o.station, self.method, self.cache)

    def to(self, cell):
        m = self.maps.get(cell)
        if m is None:
            m = self.maps[cell] = np.asarray(self.metric.dist_to(cell))
        return m

    def costs(self, free, orders, pos):
        at = np.asarray([pos[r] for r in free])
        out = np.empty((len(free), len(orders)), dtype=np.int64)
        for j, o in enumerate(orders):
            first, length = self.profile(o)
            out[:, j] = self.to(first)[at] + (length if self.use_length else 0)
        return out

    def choose(self, t, free, pending, pos):
        if not free or not pending:
            return []
        orders = list(pending)[:len(free)] if self.fifo else list(pending)
        rows, cols = linear_sum_assignment(self.costs(free, orders, pos))
        out = [(free[i], (orders[j],)) for i, j in zip(rows, cols)]
        for _, (o,) in out:
            pending.remove(o)
        return out


class Iterative(Hungarian):
    def __init__(self, wh, **kw):
        super().__init__(wh, reassign=True, **kw)

    def choose(self, t, free, pending, pos):
        if not free or not pending:
            return []
        orders = list(pending)
        cost = self.costs(free, orders, pos)
        have = {}                                     # row -> column
        spare = set(range(len(orders)))
        for j in range(min(len(free), len(orders))):
            i = min((i for i in range(len(free)) if i not in have), key=lambda i: (cost[i, j], i))
            have[i] = j
            spare.remove(j)
        moved = True
        while moved:
            moved = False
            for i in sorted(have):
                j = min(spare, key=lambda q: (cost[i, q], q), default=None)
                if j is not None and cost[i, j] < cost[i, have[i]]:
                    spare.add(have[i])
                    spare.remove(j)
                    have[i] = j
                    moved = True
                for k in sorted(have):
                    if k != i and cost[i, have[k]] + cost[k, have[i]] < cost[i, have[i]] + cost[k, have[k]]:
                        have[i], have[k] = have[k], have[i]
                        moved = True
        out = [(free[i], (orders[j],)) for i, j in sorted(have.items())]
        for _, (o,) in out:
            pending.remove(o)
        return out


class LiveCosts(Hungarian):
    REFRESH = 50                                      # steps between two rebuilds of the distances

    def __init__(self, wh, mode, decay=0.9, radius=2, **kw):
        super().__init__(wh, reassign=True, **kw)
        self.mode, self.radius = mode, radius
        self.stats = T.EdgeStats(wh, decay)
        self.live, self.built = None, -1
        self.xy, self.xx, self.gamma = 0.0, 0.0, 1.0
        self.trips = {}                               # robot -> (first cell, start, free distance, estimate)
        self.near = {}

    def observe(self, t, now, nxt, goals):
        self.stats.observe(now, nxt, goals)
        for r, (first, start, free, est) in list(self.trips.items()):
            if nxt[r] == first:
                self.xy += est * max(t + 1 - start - free, 0)
                self.xx += est * est
                self.gamma = min(self.xy / self.xx, 10.0) if self.xx else 1.0
                del self.trips[r]

    def around(self, cell):
        c = self.near.get(cell)
        if c is None:
            d = self.wh.dist_to(cell)
            c = self.near[cell] = [x for x in self.wh.free if 0 <= d[x] <= self.radius]
        return c

    def delay(self, cell):
        cells = self.stats.cells
        return sum(cells[c][2] for c in self.around(cell) if c in cells)

    def costs(self, free, orders, pos):
        at = np.asarray([pos[r] for r in free])
        out = np.empty((len(free), len(orders)), dtype=np.int64)
        for j, o in enumerate(orders):
            first, length = self.profile(o)
            if self.mode == "edge":
                out[:, j] = self.scaled(first)[at] + T.SCALE * length
            else:
                out[:, j] = T.SCALE * (self.to(first)[at] + length + self.gamma * self.delay(first))
        return out

    def scaled(self, cell):
        return np.asarray(self.live.dist_to(cell))

    def choose(self, t, free, pending, pos):
        if self.mode == "edge" and (self.live is None or t - self.built >= self.REFRESH):
            self.live, self.built = T.EdgeMetric(self.wh, self.stats.extra(decayed=True), symmetric=True), t
        got = super().choose(t, free, pending, pos)
        for r, (o,) in got:
            first, _ = self.profile(o)
            self.trips[r] = (first, t, self.wh.dist_to(first)[pos[r]], self.delay(first))
        return got


class Clusters:
    reassign = False

    def __init__(self, wh, clusters, metric=None):
        self.wh, self.metric = wh, metric or wh
        self.todo = [[o.oid for o in c] for c in clusters]

    def choose(self, t, free, pending, pos):
        out = []
        by_id = {o.oid: o for o in pending}
        while self.todo and free:
            group = tuple(by_id[k] for k in self.todo.pop(0))
            for o in group:
                pending.remove(o)
            r = S.assign_robot(self.metric, free, pos, [self.wh.access[s] for o in group for s in o.shelves])
            free.remove(r)
            out.append((r, group))
        return out


class FixedPlan:
    reassign = False

    def __init__(self, plan):
        # robot -> its trips, in order; a trip is an order id or a list of them
        self.plan = {r: [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in v] for r, v in plan.items()}

    def choose(self, t, free, pending, pos):
        out = []
        by_id = {o.oid: o for o in pending}
        for r in list(free):
            todo = self.plan.get(r)
            if todo:
                group = tuple(by_id[k] for k in todo.pop(0))
                for o in group:
                    pending.remove(o)
                out.append((r, group))
        return out


def plan_of(events):
    """The plan a finished episode followed: robot -> trips, from the last assignment of each order."""
    final = {}
    for e in events:
        if e["kind"] == "assign":
            final[e["order"]] = (e["t"], e["robot"])
    trips = {}
    for oid, key in final.items():
        trips.setdefault(key, []).append(oid)
    plan = {}
    for (t, r), oids in sorted(trips.items()):
        plan.setdefault(r, []).append(sorted(oids))
    return plan


class Batcher:
    reassign = False

    def __init__(self, wh, metric=None, rule="overlap"):
        self.wh, self.metric, self.rule = wh, metric or wh, rule

    def added(self, a, b):
        """Steps b adds to a trip for a: each shelf of b from the nearest shelf of a, and the walk between the stations."""
        maps = [self.metric.dist_to(self.wh.access[s]) for s in a.shelves]
        walk = sum(min(m[self.wh.access[s]] for m in maps) for s in b.shelves if s not in a.shelves)
        return walk + self.metric.dist_to(a.station)[b.station]

    def partner(self, o, pending):
        if self.rule == "near":
            return min(pending, key=lambda p: (self.added(o, p), p.oid), default=None)
        mine = set(o.shelves)
        return min(pending, key=lambda p: (-len(mine & set(p.shelves)), p.station != o.station, p.oid), default=None)

    def choose(self, t, free, pending, pos):
        out = []
        while pending and free:
            o = pending.popleft()
            p = self.partner(o, pending)
            group = (o,) if p is None else (o, p)
            if p is not None:
                pending.remove(p)
            cells = [self.wh.access[s] for g in group for s in g.shelves]
            r = S.assign_robot(self.metric, free, pos, cells)
            free.remove(r)
            out.append((r, group))
        return out
