"""Layer 2: no collisions, every goal reached, and CBS optimal against brute force."""

import heapq
import itertools

import numpy as np
import pytest

from warehouse import env as W
from warehouse import mapf as M
from warehouse import sim
from warehouse import verify as V

TINY = W.Layout("tiny", 12, 16, 2, 2, corridor=1, stations=2, robots=4)


def grid_graph(rows, cols, walls=()):
    """A small grid with the Warehouse interface the search uses."""
    class G:
        pass
    g = G()
    g.neighbours = [[] for _ in range(rows * cols)]
    for r, c in itertools.product(range(rows), range(cols)):
        if (r, c) in walls:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = r + dr, c + dc
            if 0 <= a < rows and 0 <= b < cols and (a, b) not in walls:
                g.neighbours[r * cols + c].append(a * cols + b)
    return g


def bfs(nb, goal):
    d = [10 ** 6] * len(nb)
    d[goal] = 0
    todo = [goal]
    for v in todo:
        for u in nb[v]:
            if d[u] > d[v] + 1:
                d[u] = d[v] + 1
                todo.append(u)
    return d


def brute_force(nb, starts, goals):
    """Minimum sum of costs by Dijkstra on the joint state: positions + which robots have stopped for good. A robot pays 1
    a step until it stops, which it can only do at its goal."""
    R = len(starts)
    start = (tuple(starts), (False,) * R)
    best = {start: 0}
    heap = [(0, 0, start)]
    tie = 0
    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > best[state]:
            continue
        pos, stopped = state
        if all(stopped):
            return cost
        choices = [[(pos[i], True)] if stopped[i] else
                   [(u, False) for u in nb[pos[i]] + [pos[i]]] + ([(pos[i], True)] if pos[i] == goals[i] else [])
                   for i in range(R)]
        for pick in itertools.product(*choices):
            cells = tuple(x[0] for x in pick)
            if len(set(cells)) < R:
                continue
            if any(cells[i] == pos[j] and cells[j] == pos[i] and pos[i] != pos[j] for i in range(R) for j in range(i)):
                continue
            flags = tuple(x[1] for x in pick)
            paying = sum(not s for s in stopped) - sum(a and not b for a, b in zip(flags, stopped))
            new = (cells, flags)
            c = cost + paying
            if c < best.get(new, 10 ** 9):
                best[new] = c
                tie += 1
                heapq.heappush(heap, (c, tie, new))
    return None


@pytest.mark.parametrize("seed", range(20))
def test_cbs_is_optimal_on_tiny_instances(seed):
    rng = np.random.default_rng(seed)
    rows, cols = 3, 4
    walls = {(1, 1)} if seed % 2 else set()
    g = grid_graph(rows, cols, walls)
    free = [c for c in range(rows * cols) if divmod(c, cols) not in walls]
    R = 2 + seed % 2 + (seed % 5 == 0)
    cells = rng.choice(free, 2 * R, replace=False).tolist()
    starts, goals = cells[:R], cells[R:]
    res = M.cbs(g.neighbours, starts, goals, [bfs(g.neighbours, x) for x in goals], max_t=40)
    assert res is not None
    paths, cost, _ = res
    assert M.first_conflict(paths) is None
    assert all(p[0] == s and p[-1] == x for p, s, x in zip(paths, starts, goals))
    assert cost == brute_force(g.neighbours, starts, goals)


def test_cbs_swap_in_a_corridor_needs_the_side_pocket():
    # a corridor with one pocket: two robots must swap ends
    g = grid_graph(2, 3, {(1, 0), (1, 2)})
    a, b = 0, 2
    res = M.cbs(g.neighbours, [a, b], [b, a], [bfs(g.neighbours, b), bfs(g.neighbours, a)], max_t=20)
    assert res is not None and res[1] == brute_force(g.neighbours, [a, b], [b, a])


def test_search_respects_constraints():
    g = grid_graph(1, 5)
    h = bfs(g.neighbours, 4)
    assert M.search(g.neighbours, 0, 0, 4, h, set(), set(), max_t=20) == [0, 1, 2, 3, 4]
    path = M.search(g.neighbours, 0, 0, 4, h, {(2, 2)}, set(), max_t=20)
    assert len(path) == 6 and path[2] != 2
    path = M.search(g.neighbours, 0, 0, 4, h, {(4, 6)}, set(), max_t=20)    # the goal is taken later: arrive after it
    assert len(path) - 1 == 7 and path[-1] == 4
    assert M.search(g.neighbours, 0, 0, 4, h, {(0, 0)}, set(), max_t=20) is None


def test_window_ends_at_the_horizon_with_the_best_cell():
    g = grid_graph(1, 8)
    h = bfs(g.neighbours, 7)
    assert M.search(g.neighbours, 0, 0, 7, h, set(), set(), horizon=3) == [0, 1, 2, 3]


def test_first_conflict():
    assert M.first_conflict([[0, 1], [2, 1]])[2:] == ("vertex", 1, 1)
    assert M.first_conflict([[0, 1], [1, 0]])[2] == "edge"
    assert M.first_conflict([[0, 1], [1, 2]]) is None
    assert M.first_conflict([[0], [1, 0]])[2:] == ("vertex", 0, 1)              # a finished robot stays
    assert M.first_conflict([[0, 1, 2], [2, 3, 3, 2]], horizon=1) is None


def episode(planner, seed, method="2opt", layout=TINY, k=6, args=None):
    wh = W.Warehouse(layout)
    cat = W.Catalogue(wh, 200, seed)
    orders = W.batch_orders(wh, cat, k, seed)
    ep = sim.run_episode(wh, cat, orders, method, planner, seed, max_steps=4000, planner_args=args)
    rep = V.verify(ep["traj"], wh.grid, [o.__dict__ for o in orders], cat.shelf_of, wh.shelves, ep["events"], ep["service"],
                   ep["step_seconds"])
    return ep, rep, orders


@pytest.mark.parametrize("planner", ["pibt", "prioritized"])
def test_hundred_random_episodes(planner):
    for seed in range(100):
        ep, rep, orders = episode(planner, seed)
        assert rep["ok"], (seed, rep["problems"])
        assert rep["delivered"] == len(orders), seed


def test_hundred_random_episodes_cbs_window():
    for seed in range(100):
        ep, rep, orders = episode("cbs", seed, args={"time_limit": 2.0})
        assert rep["ok"], (seed, rep["problems"])
        assert rep["delivered"] == len(orders), seed


def test_two_wide_corridors_and_all_solvers():
    lay = W.Layout("wide", 24, 40, 4, 4, corridor=2, stations=2, robots=6)
    for method in ("nn", "2opt", "lkh"):
        for planner in ("pibt", "prioritized"):
            _, rep, orders = episode(planner, 1, method, lay, 10)
            assert rep["ok"] and rep["delivered"] == len(orders)


def test_shared_access_cell_is_served_twice():
    # shelves of two blocks that face each other share the corridor cell between them
    wh = W.Warehouse(TINY)
    shared = [k for k in wh.access for q in wh.access if k < q and wh.access[k] == wh.access[q]]
    assert shared
