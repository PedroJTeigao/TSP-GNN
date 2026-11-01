"""Who takes which orders. Every rule answers choose(t, free, pending, pos) with [(robot, (order, ...))], and takes the
orders it gives out from pending.

    Nearest     the baseline: the oldest order goes to the free robot closest to one of its shelves;
    Hungarian   minimum-cost assignment of free robots to orders; the cost is the distance to the first shelf plus the
                length of the order on its own (LKH-3). Over the whole queue it also picks which orders go first; with
                fifo=True only the oldest orders enter, which isolates the matching. With reassign=True a robot that has
                not started to pick returns to the pool whenever a robot comes free or an order arrives;
    FixedPlan   a list of orders per robot, decided beforehand (the a posteriori search);
    Batcher     up to two orders per trip: the oldest one and the order that overlaps it most (rule "overlap"), or the one
                that adds the least walking (rule "near").
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from warehouse import sequencing as S


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
    def __init__(self, wh, metric=None, method="lkh", cache=None, reassign=False, fifo=False):
        self.wh, self.metric, self.method = wh, metric or wh, method
        self.cache = {} if cache is None else cache
        self.reassign, self.fifo = reassign, fifo
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
            out[:, j] = self.to(first)[at] + length
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


class FixedPlan:
    reassign = False

    def __init__(self, plan):
        self.plan = {r: list(v) for r, v in plan.items()}             # robot -> order ids, in order

    def choose(self, t, free, pending, pos):
        out = []
        by_id = {o.oid: o for o in pending}
        for r in list(free):
            todo = self.plan.get(r)
            if todo:
                o = by_id[todo.pop(0)]
                pending.remove(o)
                out.append((r, (o,)))
        return out


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
