"""Layer 2: paths without collisions.

Every planner gets the same state and answers with the next cells of every robot:
    pos[i]    cell of robot i;
    hold[i]   steps it must still stay there (service in progress);
    queue[i]  its next stops [(cell, dwell), ...], the last one its rest cell, dwell 0; empty if it is at rest;
    age[i]    steps since it last reached a goal, the priority of PIBT and of prioritized planning.
A robot that enters the cell of its next stop starts its service there and stays: the planners below must respect that.

    PIBT         Okumura et al. 2022: one step at a time, in order of priority, pushing lower robots out of the way.
    Prioritized  A* in space-time, one robot after the other, against a table of reservations; a rolling window: plan
                 horizon steps, execute the first execute ones (windowed cooperative A*, Silver 2005).
    WindowedCBS  conflict-based search (Sharon et al. 2015) on the next goal of every robot, conflicts counted inside the
                 window only; on a time-out, PIBT.
cbs() is the exact solver: with no horizon it finds the minimum sum of costs for one goal per robot.
"""

import heapq
import random
import time


def search(nb, start, t0, goal, h, vertex, edge, horizon=None, stay=None, lock_goal=False, max_t=None):
    """A* in (cell, time) from start at time t0; wait is a move. Returns the cells from t0 to the end, or None.

    vertex: set of (cell, t) forbidden; edge: set of (u, v, t), moves u -> v arriving at t forbidden.
    The search ends at goal if the robot can then stay there: for stay steps, or up to the horizon if stay is None and
    there is a horizon. With no horizon it must be free of all later constraints (the classic "stay at the goal").
    With a horizon it may also end at time horizon anywhere, valued by the heuristic.
    lock_goal: the robot may not enter goal unless it can stop there (a robot that enters its stop starts its service).
    """
    def free_after(t):
        if stay is not None:
            return all((goal, t + k) not in vertex for k in range(1, stay + 1))
        if horizon is not None:
            return all((goal, k) not in vertex for k in range(t + 1, horizon + 1))
        return True

    if (start, t0) in vertex:
        return None
    last = None
    if horizon is None:                                 # classic mode: the end time must pass the last constraint on goal
        last = max((t for c, t in vertex if c == goal), default=-1)
    limit = horizon if horizon is not None else max_t
    tie = 0
    heap = [(t0 + h[start], 0, t0, start)]
    parent = {(start, t0): None}
    while heap:
        _, _, t, v = heapq.heappop(heap)
        done = v == goal and (t > last if last is not None else True) and free_after(t)
        if done or t == horizon:
            path = []
            node = (v, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]
        if limit is not None and t >= limit:
            continue
        t1 = t + 1
        for u in nb[v] + [v]:
            state = (u, t1)
            if state in parent or state in vertex or (v, u, t1) in edge:
                continue
            if lock_goal and u == goal and v != goal and not free_after(t1):
                continue
            parent[state] = (v, t)
            tie += 1
            heapq.heappush(heap, (t1 + h[u], -tie, t1, u))      # ties: the most recent, which goes deeper
    return None


def search_weighted(nb, start, t0, goal, h, vertex, edge, cost, horizon, stay=None, lock_goal=False):
    """search with a horizon, for the cheapest path rather than the fastest: a move u -> v costs 1 + cost(u, v), a wait 1.
    h: lower bound on that cost to goal. A move into a robot's own stop is locked as in search."""
    def free_after(t):
        if stay is not None:
            return all((goal, t + k) not in vertex for k in range(1, stay + 1))
        return all((goal, k) not in vertex for k in range(t + 1, horizon + 1))

    if (start, t0) in vertex:
        return None
    tie = 0
    heap = [(h[start], 0, 0.0, t0, start)]
    best, parent, closed = {(start, t0): 0.0}, {(start, t0): None}, set()
    while heap:
        _, _, g, t, v = heapq.heappop(heap)
        if (v, t) in closed:
            continue
        closed.add((v, t))
        if (v == goal and free_after(t)) or t == horizon:
            path, node = [], (v, t)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]
        t1 = t + 1
        for u in nb[v] + [v]:
            state = (u, t1)
            if state in vertex or (v, u, t1) in edge:
                continue
            if lock_goal and u == goal and v != goal and not free_after(t1):
                continue
            g1 = g + 1 + (cost(v, u) if u != v else 0.0)
            if g1 < best.get(state, float("inf")):
                best[state] = g1
                parent[state] = (v, t)
                tie += 1
                heapq.heappush(heap, (g1 + h[u], -tie, g1, t1, u))
    return None


class PIBT:
    """Priority inheritance with backtracking: h is the distance to the goal on the empty grid, or in metric, which may have one
    table per robot (of). heading: filled in by an Execution; between equal options a robot then keeps going straight."""

    execute = 1

    def __init__(self, wh, seed=0, rule="age", metric=None):
        self.wh, self.rule, self.metric = wh, rule, metric
        self.rng = random.Random(seed)
        self.fixed = None
        self.heading = None
        self.shut = ()                                  # cells nobody may enter now (phase C)

    def field(self, i, cell):
        m = self.metric or self.wh
        return (m.of(i) if hasattr(m, "of") else m).dist_to(cell)

    def bent(self, i, v):
        """1 if the move of robot i to v needs a turn."""
        if self.heading is None:
            return 0
        step = v - self._pos[i]
        return 0 if step == 0 else int(step != (-self.wh.width, 1, self.wh.width, -1)[self.heading[i]])

    def bonus(self, pos, queue):
        """What rule adds to the age. The distance rules only break ties of age: as a real weight "far" livelocks, even at
        half a point per cell (0 of 12 orders on a tiny grid)."""
        R = len(pos)
        if self.rule == "age":
            return [0] * R
        if self.rule == "fixed":
            if self.fixed is None:
                self.fixed = [self.rng.uniform(0, 50) for _ in range(R)]
            return self.fixed
        if self.rule == "load":
            return [5 * len(q) for q in queue]
        d = [self.wh.dist_to(q[0][0])[p] if q else 0 for q, p in zip(queue, pos)]
        top = 2 * (max(d) + 1)                                     # below 1, with the eps of the caller
        return [x / top for x in d] if self.rule == "far" else [-x / top for x in d]        # "near"

    def step(self, pos, hold, queue, age):
        wh, rng = self.wh, self.rng
        R = len(pos)
        eps = [rng.random() for _ in range(R)]
        bonus = self.bonus(pos, queue)
        order = sorted(range(R), key=lambda i: -(age[i] + bonus[i] + eps[i]))
        here = {p: i for i, p in enumerate(pos)}
        nxt = [None] * R
        taken = {}
        for i in range(R):                              # a robot in service does not move
            if hold[i] > 0:
                nxt[i] = pos[i]
                taken[pos[i]] = i
        maps = [self.field(i, q[0][0] if q else p) for i, (q, p) in enumerate(zip(queue, pos))]
        self._pos = pos

        def pibt(i, j):
            h = maps[i]
            options = [v for v in wh.neighbours[pos[i]] if v not in self.shut] + [pos[i]]
            options = sorted(options, key=lambda v: (h[v], v in here and v != pos[i], self.bent(i, v), rng.random()))
            for v in options:
                if v in taken or (j is not None and v == pos[j]):
                    continue
                nxt[i] = v
                taken[v] = i
                k = here.get(v)
                if k is not None and k != i and nxt[k] is None and not pibt(k, i):
                    del taken[v]
                    nxt[i] = None
                    continue
                return True
            nxt[i] = pos[i]
            taken[pos[i]] = i
            return False

        for i in order:
            if nxt[i] is None:
                pibt(i, None)
        return nxt

    def plan(self, t, pos, hold, queue, age):
        return [[c] for c in self.step(pos, hold, queue, age)]


def _goals(q, p):
    return list(q) if q else [(p, 0)]


class Prioritized:
    """Plans horizon steps for every robot, by age, and executes the first execute; a failed robot goes first in the next
    try, and after retries the step falls back on PIBT.

    Costs on the paths, all optional: metric is a directed traffic.EdgeMetric (highways); with, against charge a move for
    each robot already planned in this call over the same edge, or over it the other way (flow, in the style of Guided PIBT)."""

    def __init__(self, wh, horizon=20, execute=5, retries=3, seed=0, metric=None, with_flow=0.0, against=0.0):
        self.wh, self.horizon, self.execute, self.retries = wh, horizon, execute, retries
        self.metric, self.with_flow, self.against = metric, with_flow, against
        self.flow = {}
        self.pibt = PIBT(wh, seed)
        self.fallbacks = self.calls = 0
        self.gates = None                               # t -> moves (u, v, k) forbidden k steps from now (phase C)

    def bound(self, cell):
        return self.metric.steps_to(cell) if self.metric is not None else self.wh.dist_to(cell)

    def move_cost(self, u, v):
        x = self.metric.extra.get((u, v), 0.0) if self.metric is not None else 0.0
        return x + self.with_flow * self.flow.get((u, v), 0) + self.against * self.flow.get((v, u), 0)

    def route(self, i, pos, hold, queue, vertex, edge):
        """The cells of robot i for times 0..horizon, through its stops, or None."""
        wh, w = self.wh, self.horizon
        path = [pos[i]] * (min(hold[i], w) + 1)
        stops = _goals(queue[i], pos[i])
        for k, (cell, dwell) in enumerate(stops):
            if len(path) - 1 >= w:
                break
            if path[-1] == cell and dwell > 0:          # the next stop shares this cell: the service follows at once
                t = len(path) - 1
                if any((cell, t + j) in vertex for j in range(1, dwell + 1)):
                    return None
                path += [cell] * min(dwell, w - t)
                continue
            end = k == len(stops) - 1
            if self.metric is None and not (self.with_flow or self.against):
                seg = search(wh.neighbours, path[-1], len(path) - 1, cell, wh.dist_to(cell), vertex, edge, horizon=w,
                             stay=None if end else dwell, lock_goal=True)
            else:
                seg = search_weighted(wh.neighbours, path[-1], len(path) - 1, cell, self.bound(cell), vertex, edge, self.move_cost, w,
                                      stay=None if end else dwell, lock_goal=True)
            if seg is None:
                return None
            path += seg[1:]
            if path[-1] != cell:
                break
            path += [cell] * min(dwell, w - (len(path) - 1))
        if len(path) < w + 1:                           # the last stop was checked free up to the horizon
            path += [path[-1]] * (w + 1 - len(path))
        return path

    def plan(self, t, pos, hold, queue, age):
        self.calls += 1
        R = len(pos)
        ranked = sorted(range(R), key=lambda i: (-age[i], i))
        for _ in range(self.retries + 1):
            vertex, edge, paths, failed = set(), set(self.gates(t)) if self.gates else set(), [None] * R, []
            self.flow = {}
            for i in range(R):                          # the robots in service are fixed
                vertex.update((pos[i], k) for k in range(hold[i] + 1) if hold[i] > 0)
            for i in ranked:
                free = {x for x in vertex if not (hold[i] > 0 and x[0] == pos[i] and x[1] <= hold[i])}
                p = self.route(i, pos, hold, queue, free if hold[i] > 0 else vertex, edge)
                if p is None:
                    failed.append(i)
                    continue
                paths[i] = p
                if self.with_flow or self.against:
                    for a, b in zip(p, p[1:]):
                        if a != b:
                            self.flow[(a, b)] = self.flow.get((a, b), 0) + 1
                vertex.update((c, k) for k, c in enumerate(p))
                edge.update((b, a, k + 1) for k, (a, b) in enumerate(zip(p, p[1:])))
            if not failed:
                return [p[1:self.execute + 1] for p in paths]
            ranked = failed + [i for i in ranked if i not in failed]
        self.fallbacks += 1
        return self.pibt.plan(t, pos, hold, queue, age)


def first_conflict(paths, horizon=None):
    """(a, b, kind, data, t) of the earliest conflict, or None. A path that ended stays at its last cell."""
    T = max(map(len, paths))
    if horizon is not None:
        T = min(T, horizon + 1)
    at = lambda p, t: p[t] if t < len(p) else p[-1]
    for t in range(T):
        seen = {}
        for i, p in enumerate(paths):
            c = at(p, t)
            if c in seen:
                return seen[c], i, "vertex", c, t
            seen[c] = i
        if t:
            moved = {}
            for i, p in enumerate(paths):
                a, b = at(p, t - 1), at(p, t)
                if a != b:
                    if (b, a) in moved:
                        return moved[(b, a)], i, "edge", (b, a), t
                    moved[(a, b)] = i
    return None


def cbs(nb, starts, goals, maps, horizon=None, stays=None, holds=None, lock=None, deadline=None, max_t=None):
    """Conflict-based search. Returns (paths, cost, expanded) or None if the deadline passed or there is no solution.

    horizon None: the classic problem, each robot stops at its goal; the optimum of the sum of costs (arrival times).
    With a horizon, conflicts count up to it only. holds[i]: steps robot i must first stay in place (fixed);
    lock[i]: True if entering the goal starts a service (see search).
    """
    R = len(starts)
    holds = holds or [0] * R
    lock = lock or [False] * R
    limit = max_t if max_t is not None else 4 * len(nb)

    def low(i, vertex, edge):
        if any((starts[i], k) in vertex for k in range(holds[i] + 1)):
            return None
        tail = search(nb, starts[i], holds[i], goals[i], maps[i], vertex, edge, horizon=horizon, lock_goal=lock[i],
                      max_t=limit)
        return None if tail is None else [starts[i]] * holds[i] + tail

    def value(i, p):
        return len(p) - 1 + maps[i][p[-1]]

    empty_v, empty_e = [set() for _ in range(R)], [set() for _ in range(R)]
    paths = [low(i, empty_v[i], empty_e[i]) for i in range(R)]
    if any(p is None for p in paths):
        return None
    tie, expanded = 0, 0
    heap = [(sum(value(i, p) for i, p in enumerate(paths)), 0, 0, paths, empty_v, empty_e)]
    while heap:
        if deadline is not None and time.perf_counter() > deadline:
            return None
        cost, _, _, paths, vs, es = heapq.heappop(heap)
        expanded += 1
        c = first_conflict(paths, horizon)
        if c is None:
            return paths, cost, expanded
        a, b, kind, data, t = c
        for agent, other in ((a, b), (b, a)):
            nv, ne = list(vs), list(es)
            if kind == "vertex":
                nv[agent] = vs[agent] | {(data, t)}
            else:
                u, w = data if agent == a else data[::-1]
                ne[agent] = es[agent] | {(u, w, t)}
            p = low(agent, nv[agent], ne[agent])
            if p is None:
                continue
            child = list(paths)
            child[agent] = p
            tie += 1
            total = sum(value(i, q) for i, q in enumerate(child))
            conflicts = 0 if first_conflict(child, horizon) is None else 1
            heapq.heappush(heap, (total, conflicts, tie, child, nv, ne))
    return None


class WindowedCBS:
    """CBS on the next goal of every robot, with conflicts up to horizon; the first execute steps are used. After a
    time-out, PIBT for cooldown steps. A budget in seconds for the episode ends the use of CBS, also in favour of PIBT."""

    def __init__(self, wh, horizon=8, execute=3, time_limit=30.0, cooldown=10, budget=None, seed=0):
        self.wh, self.horizon, self.execute, self.time_limit = wh, horizon, execute, time_limit
        self.cooldown, self.budget = cooldown, budget
        self.pibt = PIBT(wh, seed)
        self.calls = self.fallbacks = self.timeouts = self.expanded = 0
        self.spent = 0.0
        self.wake = 0

    def plan(self, t, pos, hold, queue, age):
        if t < self.wake or (self.budget is not None and self.spent >= self.budget):
            self.fallbacks += 1
            return self.pibt.plan(t, pos, hold, queue, age)
        wh = self.wh
        goals = [q[0][0] if q else p for q, p in zip(queue, pos)]
        locks = [bool(q) and q[0][1] > 0 for q in queue]
        t0 = time.perf_counter()
        limit = self.time_limit
        if self.budget is not None:
            limit = min(limit, self.budget - self.spent)
        # a next stop on the same cell (two shelves share an access cell): that service follows at once, so it is fixed too
        holds = [h + q[0][1] if h and q and q[0][0] == p else h for h, q, p in zip(hold, queue, pos)]
        res = cbs(wh.neighbours, pos, goals, [wh.dist_to(g) for g in goals], horizon=self.horizon, holds=holds, lock=locks,
                  deadline=t0 + limit)
        self.spent += time.perf_counter() - t0
        self.calls += 1
        if res is None:
            self.timeouts += 1
            self.fallbacks += 1
            self.wake = t + self.cooldown
            return self.pibt.plan(t, pos, hold, queue, age)
        paths, _, expanded = res
        self.expanded += expanded
        return [[(p[k] if k < len(p) else p[-1]) for k in range(1, self.execute + 1)] for p in paths]


class FreeFlow:
    """Not a planner of real robots: each one walks a shortest path as if it were alone, so robots may share cells. It gives
    the free-flow limit of an assignment, and its episodes only pass the verifier without the collision checks."""

    def __init__(self, wh, seed=0):
        self.wh = wh

    def plan(self, t, pos, hold, queue, age):
        out = []
        for p, h, q in zip(pos, hold, queue):
            if h > 0 or not q or q[0][0] == p:
                out.append([p])
            else:
                d = self.wh.dist_to(q[0][0])
                out.append([min(self.wh.neighbours[p], key=lambda v: (d[v], v))])
        return out


def make_planner(name, wh, seed=0, **kw):
    return {"pibt": PIBT, "prioritized": Prioritized, "cbs": WindowedCBS, "free": FreeFlow}[name](wh, seed=seed, **kw)
