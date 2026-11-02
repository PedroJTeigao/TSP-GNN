"""The a posteriori assignment of phase B0: MCA / RMCA insertion, improved by a large neighbourhood search.

An order goes where it adds the least to the sum of completion times of a robot's list, and the
estimate of a robot's trips is scaled by what the last simulated episode measured for that robot, so the marginal delays
are read from the real paths episode by episode rather than from a simulation per insertion, which would not fit in the time.
"""

from warehouse import sequencing as S


class Insertion:
    def __init__(self, wh, orders, homes, service, cache):
        self.wh, self.homes = wh, homes
        self.order = {o.oid: o for o in orders}
        self.info = {o.oid: S.open_path(wh, [wh.access[s] for s in o.shelves], o.station, "lkh", cache) for o in orders}
        self.fixed = {o.oid: service * (len(o.shelves) + 1) for o in orders}
        self.scale = {r: 1.0 for r in range(len(homes))}

    def trip(self, at, oid):
        first, length = self.info[oid]
        return self.wh.dist_to(first)[at] + length + self.fixed[oid]

    def times(self, r, todo):
        """(finish, sum of completions) of robot r with the list todo, in estimated steps."""
        t, total, at = 0.0, 0.0, self.homes[r]
        for oid in todo:
            t += self.scale[r] * self.trip(at, oid)
            total += t
            at = self.order[oid].station
        return t, total

    def insert(self, plan, oid):
        best = None
        for r, todo in plan.items():
            _, before = self.times(r, todo)
            for k in range(len(todo) + 1):
                new = todo[:k] + [oid] + todo[k:]
                finish, after = self.times(r, new)
                key = (after - before, finish, r, k)
                if best is None or key < best[0]:
                    best = (key, r, new)
        plan[best[1]] = best[2]

    def build(self):
        plan = {r: [] for r in range(len(self.homes))}
        for oid in sorted(self.order, key=lambda k: (-self.trip(self.order[k].station, k), k)):
            self.insert(plan, oid)
        return plan

    def measure(self, plan, events):
        """Scale each robot's estimate by its real finish."""
        last = {}
        for e in events:
            if e["kind"] == "deliver":
                last[e["robot"]] = max(last.get(e["robot"], 0), e["end"])
        saved = dict(self.scale)
        self.scale = {r: 1.0 for r in plan}
        for r, todo in plan.items():
            if todo and r in last:
                self.scale[r] = last[r] / max(self.times(r, todo)[0], 1.0)
        return saved
