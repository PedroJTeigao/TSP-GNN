"""Imperfect execution: what the robots really do with the plan. Planners still plan on the nominal model.

    rotation    a robot has a heading; moving another way first costs one step to turn, whatever the angle;
    slip        with probability p a robot that would advance does not;
    stop        with probability q per step of trying to move, a robot stops for d steps, d uniform in [5, 20].

A regime gives p and q. Flat is the same p and q for everyone (R1). StateDelay (R2) has the same mean, but depends on the state:

    p(robot, cell, n) = min(0.9, s * speed[type] * (1 + OCCUPANCY * n) * (1 + STATION * near(cell)))
    q(cell)           = q_bad on the bad floor, 0 elsewhere

n is the number of other robots within 2 cells along the grid; near(cell) falls from 1 at a station to 0 at STATION_REACH cells; the
type of a robot is its index modulo 3, with speed 0.5, 1, 1.5; the bad floor is 20 % of the free cells in patches, fixed per
layout. s and q_bad are calibrated so that each kind of lost step per move matches R1 (see calibrate in phaseB0c_realism.py).

Nothing collides, because every plan is executed under the action dependencies of Hönig et al. 2019. A plan fixes who goes through
each cell in what order, and a robot may enter a cell only when all earlier visitors have left it, or are leaving in this same
step. PIBT is the case of a plan one step long. Plans are replaced at the nominal cadence from the real positions, so a robot that
fell behind drops the rest of its window. A stopped robot shows up to the planner as a robot in service: that is seen, not predicted.
"""

import heapq

import numpy as np

from warehouse import mapf as M
from warehouse import traffic as T

BRAKE = (5, 20)                                 # steps of a stop, both ends included
SPEED = (0.5, 1.0, 1.5)                         # what a type multiplies p by
OCCUPANCY, STATION = 0.25, 1.0
STATION_REACH = 6
RADIUS = 2
TOP = 0.9
BAD_SHARE, PATCH = 0.2, 3
MAX_N = 13                                      # the most robots within 2 cells of a cell on a 4-connected grid: 12


def stop_mean():
    return sum(BRAKE) / 2


def turned_to(wh, a, b):
    """The heading of a move from cell a to the next cell b: 0 up, 1 right, 2 down, 3 left."""
    return {-wh.width: 0, 1: 1, wh.width: 2, -1: 3}[b - a]


def around(wh):
    """For each free cell, the other free cells within RADIUS along the grid."""
    out = {}
    for c in wh.free:
        near, edge = {c}, [c]
        for _ in range(RADIUS):
            edge = [u for v in edge for u in wh.neighbours[v] if u not in near]
            near.update(edge)
        out[c] = sorted(near - {c})
    return out


class Flat:
    """R1: the same p and q everywhere, for every robot."""

    kinds = 1
    counts = False                              # p does not read the robots around

    def __init__(self, p=0.0, q=0.0):
        self.p, self.q = p, q

    def kind(self, robot):
        return 0

    def slip(self, robot, cell, n):
        return self.p

    def stop(self, cell):
        return self.q

    def expected(self, wh, table):
        """kind -> extra[cell]: the steps above 1 that one cell costs, on average; table is unused here."""
        return {0: np.full(wh.height * wh.width, (1 + self.q * stop_mean()) / (1 - self.p) - 1)}


def bad_floor(wh, seed):
    """BAD_SHARE of the free cells, in patches of radius PATCH around random cells."""
    rng = np.random.default_rng([seed, 31])
    need, bad = round(BAD_SHARE * len(wh.free)), set()
    while len(bad) < need:
        c = int(rng.choice(wh.free))
        near = [x for x in wh.free if 0 <= wh.dist_to(c)[x] <= PATCH and x not in bad]
        bad.update(near[:need - len(bad)])
    return bad


class StateDelay:
    """R2. The function is at the top of the file."""

    kinds = len(SPEED)
    counts = True

    def __init__(self, wh, s, q_bad, floor_seed=0):
        self.s, self.q_bad = s, q_bad
        self.bad = bad_floor(wh, floor_seed)
        near = [0.0] * (wh.height * wh.width)
        for c in wh.free:
            d = min(wh.dist_to(st)[c] for st in wh.stations)
            near[c] = max(0.0, 1 - d / STATION_REACH)
        self.near = near

    def kind(self, robot):
        return robot % self.kinds

    def base(self, kind, cell, n):
        return min(TOP, self.s * SPEED[kind] * (1 + OCCUPANCY * n) * (1 + STATION * self.near[cell]))

    def slip(self, robot, cell, n):
        return self.base(self.kind(robot), cell, n)

    def stop(self, cell):
        return self.q_bad if cell in self.bad else 0.0

    def expected(self, wh, table):
        """kind -> extra[cell], with the function known and n distributed as in table (cells, MAX_N), counts of tries from other
        seeds. A cell with no tries takes the distribution of all cells."""
        table = np.asarray(table, dtype=np.float64)
        mean = table.sum(0) / max(table.sum(), 1)
        out = {}
        for k in range(self.kinds):
            extra = np.zeros(wh.height * wh.width)
            for c in wh.free:
                w = table[c] / table[c].sum() if table[c].sum() else mean
                value = sum(x / (1 - self.base(k, c, n)) for n, x in enumerate(w) if x)
                extra[c] = (1 + self.stop(c) * stop_mean()) * value - 1
            out[k] = extra
        return out


class Execution:
    """Between the planner and the cells. load takes a plan, step gives the real next cells. log[t][i]: 0, or TURN or OWN if that step
    went to turning or to the robot's own delay. ghost: robots do not see each other, for the free-flow limit."""

    TURN, OWN = 1, 2

    def __init__(self, wh, robots, regime, seed, turns=True, ghost=False):
        self.wh, self.R, self.regime, self.turns, self.ghost = wh, robots, regime, turns, ghost
        self.rng = np.random.default_rng([seed, 41])
        self.heading = [0] * robots
        self.stalled = [0] * robots
        self.paths, self.prog, self.visits, self.left = [], [], {}, 0
        self.log = []
        self.tries = np.zeros((wh.height * wh.width, MAX_N), dtype=np.int64) if regime.counts else None
        self.near = around(wh) if regime.counts else None
        self.moves = self.slips = self.stops = self.turned = 0

    def bind(self, planner):
        """A PIBT planner reads the headings to break its ties; so does the PIBT inside a prioritized planner."""
        planner.heading = self.heading if self.turns else None
        if hasattr(planner, "pibt"):
            planner.pibt.heading = planner.heading

    def idle(self):
        return self.left == 0

    def view(self, hold):
        return [max(h, s) for h, s in zip(hold, self.stalled)]

    def load(self, pos, paths):
        self.paths = [[p] + list(q) for p, q in zip(pos, paths)]
        self.prog = [0] * self.R
        self.left = len(paths[0])
        self.visits = {}
        for i, path in enumerate(self.paths):
            start = 0
            for k in range(1, len(path) + 1):
                if k == len(path) or path[k] != path[start]:
                    self.visits.setdefault(path[start], []).append((start, k - 1, i))
                    start = k

    def clear(self, i, cell, entry, leaving):
        """Every earlier visit of cell is over, or ends in this step, so robot i may enter it at plan step entry."""
        for first, last, r in self.visits[cell]:
            if first < entry and r != i and self.prog[r] <= last and not (self.prog[r] == last and r in leaving):
                return False
        return True

    def step(self, pos, hold):
        here = {}
        if self.regime.counts:
            for p in pos:
                here[p] = here.get(p, 0) + 1
        draws = self.rng.random((self.R, 3))    # drawn for every robot, so the delays of a seed do not depend on the arm
        mark = [0] * self.R
        want = set()
        for i in range(self.R):
            if self.stalled[i] > 0:
                self.stalled[i] -= 1
                self.stops += 1
                mark[i] = self.OWN
                continue
            k = self.prog[i]
            if k + 1 >= len(self.paths[i]):
                continue
            a, b = self.paths[i][k], self.paths[i][k + 1]
            if a == b:                          # a planned wait passes with time, also during a service
                self.prog[i] += 1
                continue
            if hold[i] > 0:
                continue
            if draws[i, 1] < self.regime.stop(a):
                low, high = BRAKE
                self.stalled[i] = low + int(draws[i, 2] * (high - low + 1)) - 1
                self.stops += 1
                mark[i] = self.OWN
                continue
            way = turned_to(self.wh, a, b)
            if self.turns and self.heading[i] != way:
                self.heading[i] = way
                self.turned += 1
                mark[i] = self.TURN
                continue
            n = 0
            if self.regime.counts:
                n = sum(here.get(c, 0) for c in self.near[a])
                self.tries[a, min(n, MAX_N - 1)] += 1
            if draws[i, 0] < self.regime.slip(i, a, n):
                self.slips += 1
                mark[i] = self.OWN
                continue
            want.add(i)
        changed = not self.ghost
        while changed:
            changed = False
            for i in list(want):
                k = self.prog[i]
                if not self.clear(i, self.paths[i][k + 1], k + 1, want):
                    want.remove(i)
                    changed = True
        nxt = list(pos)
        for i in want:
            self.prog[i] += 1
            nxt[i] = self.paths[i][self.prog[i]]
            self.moves += 1
        self.left -= 1
        self.log.append(mark)
        return nxt

    def stats(self):
        return {"moves": self.moves, "slips": self.slips, "stops": self.stops, "turns": self.turned}


class TurnMetric(T.EdgeMetric):
    """Shortest times with headings: a Dijkstra on (cell, heading). extra is as in EdgeMetric, plus turn steps for each change of
    heading. dist_to(goal) gives, for each cell, the cost from there with the best first heading, which is the same both ways and so
    suits LKH-3; table(goal) gives it per heading, (4, cells). Costs are integers in hundredths of a step."""

    def __init__(self, wh, extra=None, turn=1.0, symmetric=True, keep=False):
        super().__init__(wh, extra or {}, symmetric=symmetric)
        self.turn, self.keep = round(T.SCALE * turn), keep
        self._table = {}

    def table(self, goal):
        out = self._table.get(goal)
        if out is None:
            out = self.solve(goal)
            if self.keep:
                self._table[goal] = out
        return out

    def solve(self, goal):
        wh = self.wh
        steps = (-wh.width, 1, wh.width, -1)
        size = wh.height * wh.width
        d = [[-1] * size for _ in range(4)]
        turned = [False] * size
        todo = [(0, goal, h) for h in range(4)]
        nb = wh.neighbours
        while todo:
            x, v, h = heapq.heappop(todo)
            if d[h][v] >= 0:
                continue
            d[h][v] = x
            if not turned[v]:                   # the first heading out of v is the best: the others turn into it
                turned[v] = True
                for g in range(4):
                    if d[g][v] < 0:
                        heapq.heappush(todo, (x + self.turn, v, g))
            u = v - steps[h]                    # u -> v goes with heading h
            if 0 <= u < size and v in nb[u] and d[h][u] < 0:
                heapq.heappush(todo, (x + self.edge(u, v), u, h))
        return np.array(d, dtype=np.int32)

    def dist_to(self, goal):
        out = self._dist.get(goal)
        if out is None:
            t = self.table(goal)
            free = t[0] >= 0
            out = self._dist[goal] = np.where(free, t.min(axis=0), -1).tolist()
        return out


class Typed:
    """One metric per type of robot, which of(robot) picks; shared answers for what belongs to no robot, such as the trips."""

    def __init__(self, metrics, shared, kind):
        self.metrics, self.shared, self.kind = metrics, shared, kind

    def of(self, robot):
        return self.metrics[self.kind(robot)]

    def edge(self, a, b):
        return self.shared.edge(a, b)

    def dist_to(self, goal):
        return self.shared.dist_to(goal)


class TurnFree(M.FreeFlow):
    """The free-flow limit with rotation: each robot alone, on a fastest path given its heading."""

    def __init__(self, wh, seed=0):
        super().__init__(wh, seed)
        self.metric = TurnMetric(wh, keep=True)
        self.heading = None

    def plan(self, t, pos, hold, queue, age):
        out = []
        for i, (p, h, q) in enumerate(zip(pos, hold, queue)):
            if h > 0 or not q or q[0][0] == p:
                out.append([p])
                continue
            d = self.metric.table(q[0][0])
            value = lambda v: (d[turned_to(self.wh, p, v), v] + (0 if turned_to(self.wh, p, v) == self.heading[i] else self.metric.turn), v)
            out.append([min(self.wh.neighbours[p], key=value)])
        return out
