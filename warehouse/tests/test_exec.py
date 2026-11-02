"""The execution model: nominal runs repeat the old simulator, delays never collide, and distances with headings are right."""

import dataclasses
import itertools

import numpy as np
import pytest

from warehouse import batching as B
from warehouse import dispatch as D
from warehouse import env as W
from warehouse import exec as E
from warehouse import sim
from warehouse import traffic as T
from warehouse import verify as V


def world(size="small", seed=0, k=12, robots=None):
    lay = W.SIZES[size]
    wh = W.Warehouse(dataclasses.replace(lay, robots=robots or lay.robots))
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    return wh, cat, W.batch_orders(wh, cat, k, seed)


def judged(wh, cat, orders, ep, capacity=1, rotation=False):
    rows = [{k: getattr(o, k) for k in ("oid", "products", "station", "arrival")} for o in orders]
    return V.verify(ep["traj"], wh.grid, rows, cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"], capacity=capacity, rotation=rotation)


class Grid:
    """A bare open grid, for the unit tests of Execution."""

    def __init__(self, h=4, w=5, blocked=()):
        self.height, self.width = h, w
        self.free = [c for c in range(h * w) if c not in blocked]
        self.neighbours = [[] for _ in range(h * w)]
        for c in self.free:
            r, q = divmod(c, w)
            for a, b in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)):
                n = a * w + b
                if 0 <= a < h and 0 <= b < w and n in self.free:
                    self.neighbours[c].append(n)


class Scripted(E.Flat):
    """Robot in slipping never advances."""

    def __init__(self, slipping=()):
        super().__init__()
        self.slipping = set(slipping)

    def slip(self, robot, cell, n):
        return 1.0 if robot in self.slipping else 0.0


@pytest.mark.parametrize("planner", ["pibt", "prioritized"])
def test_no_delays_no_turns_repeats_the_old_simulator(planner):
    wh, cat, orders = world(k=30)
    old = sim.run_episode(wh, cat, orders, "2opt", planner, 3)
    new = sim.run_episode(wh, cat, orders, "2opt", planner, 3, execution=E.Execution(wh, wh.layout.robots, E.Flat(), 3, turns=False))
    assert np.array_equal(old["traj"], new["traj"])
    assert old["events"] == new["events"]


def test_a_stopped_chain_stops_behind_its_head():
    wh = Grid()
    ex = E.Execution(wh, 3, Scripted([2]), 0, turns=False)
    ex.load([0, 1, 2], [[1], [2], [3]])
    assert ex.step([0, 1, 2], [0, 0, 0]) == [0, 1, 2]


def test_a_chain_moves_together():
    wh = Grid()
    ex = E.Execution(wh, 3, E.Flat(), 0, turns=False)
    ex.load([0, 1, 2], [[1], [2], [3]])
    assert ex.step([0, 1, 2], [0, 0, 0]) == [1, 2, 3]


def test_a_cycle_moves_together_and_stops_together():
    wh = Grid()
    cells = [0, 1, 6, 5]
    plan = [[1], [6], [5], [0]]
    ex = E.Execution(wh, 4, E.Flat(), 0, turns=False)
    ex.load(cells, plan)
    assert ex.step(cells, [0] * 4) == [1, 6, 5, 0]
    ex = E.Execution(wh, 4, Scripted([3]), 0, turns=False)
    ex.load(cells, plan)
    assert ex.step(cells, [0] * 4) == cells


def test_dependency_keeps_the_order_of_the_plan_when_the_first_is_late():
    """Robot 0 goes through cell 2 first and robot 1 after it. With 0 late, 1 waits at the door rather than go first."""
    wh = Grid()
    ex = E.Execution(wh, 2, Scripted([0]), 0, turns=False)
    ex.load([1, 3], [[2, 7], [3, 2]])
    pos = [1, 3]
    for _ in range(2):
        pos = ex.step(pos, [0, 0])
    assert pos == [1, 3]


def test_dependency_lets_the_second_go_once_the_first_has_left():
    wh = Grid()
    ex = E.Execution(wh, 2, E.Flat(), 0, turns=False)
    ex.load([1, 3], [[2, 7, 7], [3, 2, 1]])
    pos = [1, 3]
    seen = []
    for _ in range(3):
        pos = ex.step(pos, [0, 0])
        seen.append(tuple(pos))
    assert seen == [(2, 3), (7, 2), (7, 1)]


def test_a_turn_costs_one_step_whatever_the_angle():
    wh = Grid()
    ex = E.Execution(wh, 1, E.Flat(), 0)                     # it faces up, and goes right
    ex.load([6], [[7]])
    assert ex.step([6], [0]) == [6]
    assert ex.log[-1][0] == ex.TURN
    ex.load([6], [[7]])
    assert ex.step([6], [0]) == [7]
    ex.load([7], [[6]])                                       # a half turn is one step too
    assert ex.step([7], [0]) == [7]
    ex.load([7], [[6]])
    assert ex.step([7], [0]) == [6]


def test_a_stop_lasts_its_steps_and_shows_as_a_hold():
    wh = Grid()
    ex = E.Execution(wh, 1, E.Flat(q=1.0), 1, turns=False)
    ex.regime.stop = lambda cell: 1.0 if not ex.stalled[0] and not ex.stops else 0.0
    pos = [0]
    ex.load(pos, [[1]])
    ex.step(pos, [0])
    left = ex.stalled[0]
    assert E.BRAKE[0] - 1 <= left <= E.BRAKE[1] - 1
    assert ex.view([0]) == [left]
    for _ in range(left):
        ex.load(pos, [[0]])
        assert ex.step(pos, [0]) == [0]
    assert ex.stalled[0] == 0 and ex.stops == left + 1


def test_breakdown_lengths_cover_the_range():
    wh = Grid()
    ex = E.Execution(wh, 1, E.Flat(q=1.0), 1, turns=False)
    seen = set()
    for _ in range(2000):
        ex.stalled[0] = 0
        ex.load([0], [[1]])
        ex.step([0], [0])
        seen.add(ex.stalled[0] + 1)
    assert seen == set(range(E.BRAKE[0], E.BRAKE[1] + 1))


@pytest.mark.parametrize("planner", ["pibt", "prioritized"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delays_and_turns_never_collide(planner, seed):
    wh, cat, orders = world(seed=seed, k=30)
    ex = E.Execution(wh, wh.layout.robots, E.Flat(0.2, 0.01), seed)
    ep = sim.run_episode(wh, cat, orders, "2opt", planner, seed, execution=ex)
    rep = judged(wh, cat, orders, ep, rotation=True)
    assert rep["ok"], rep["problems"]
    assert rep["delivered"] == len(orders)
    assert ex.slips and ex.stops and ex.turned


@pytest.mark.parametrize("planner", ["pibt", "prioritized"])
def test_state_delays_never_collide(planner):
    wh, cat, orders = world(k=30)
    ep = sim.run_episode(wh, cat, orders, "2opt", planner, 0, execution=E.Execution(wh, wh.layout.robots, E.StateDelay(wh, 0.05, 0.01), 0))
    rep = judged(wh, cat, orders, ep, rotation=True)
    assert rep["ok"] and rep["delivered"] == len(orders), rep["problems"]


def test_verifier_catches_a_turn_without_a_step():
    assert V.sharp_turns(np.array([[0], [1], [6]])) == 1
    assert V.sharp_turns(np.array([[0], [1], [1], [6]])) == 0
    assert V.sharp_turns(np.array([[0], [1], [2]])) == 0


def test_verifier_catches_a_collision_from_a_broken_dependency():
    assert V.collisions(np.array([[1, 3], [2, 2]]))[0] == 1


def test_the_same_seed_gives_the_same_delays():
    wh, cat, orders = world(k=20)
    a = sim.run_episode(wh, cat, orders, "2opt", "pibt", 0, execution=E.Execution(wh, wh.layout.robots, E.Flat(0.1, 0.002), 5))
    b = sim.run_episode(wh, cat, orders, "2opt", "pibt", 0, execution=E.Execution(wh, wh.layout.robots, E.Flat(0.1, 0.002), 5))
    assert np.array_equal(a["traj"], b["traj"])


def brute(wh, metric, start, heading, goal):
    """Bellman-Ford on (cell, heading): the reference of the Dijkstra."""
    steps = (-wh.width, 1, wh.width, -1)
    d = {(c, h): (0 if c == goal else float("inf")) for c in wh.free for h in range(4)}
    for _ in range(4 * len(wh.free)):
        for c, h in itertools.product(wh.free, range(4)):
            best = min(d[c, g] + metric.turn for g in range(4) if g != h)
            n = c + steps[h]
            if n in wh.neighbours[c]:
                best = min(best, d[n, h] + metric.edge(c, n))
            if c != goal:
                d[c, h] = min(d[c, h], best)
    return d[start, heading]


@pytest.mark.parametrize("extra", [{}, {(6, 7): 2.0, (7, 6): 2.0, (11, 12): 0.5, (12, 11): 0.5}])
def test_dijkstra_with_headings_against_bellman_ford(extra):
    wh = Grid(blocked=(8,))
    m = E.TurnMetric(wh, extra)
    for goal in (0, 13, 19):
        t = m.table(goal)
        for c, h in itertools.product(wh.free, range(4)):
            assert t[h, c] == brute(wh, m, c, h, goal)


def test_heading_distances_count_the_turns():
    wh = Grid()
    m = E.TurnMetric(wh)
    t = m.table(19)                                           # the far corner, 3 down and 4 right
    assert t[1, 0] == 800                                     # facing right: 7 moves and a turn
    assert t[0, 0] == 900                                     # facing up: one more turn
    assert m.dist_to(19)[0] == 800


def test_heading_distances_are_symmetric():
    wh = Grid(blocked=(7, 8, 12))
    m = E.TurnMetric(wh, {(0, 1): 1.5, (6, 11): 0.5})
    for a, b in itertools.combinations(wh.free, 2):
        assert m.dist_to(a)[b] == m.dist_to(b)[a]


def test_no_turn_cost_gives_the_plain_distances():
    wh, _, _ = world()
    m = E.TurnMetric(wh, turn=0.0)
    g = wh.free[17]
    assert [x // T.SCALE if x > 0 else x for x in m.dist_to(g)] == wh.dist_to(g)


def test_free_flow_with_turns_takes_the_fewest_turns():
    wh = Grid()
    ex = E.Execution(wh, 1, E.Flat(), 0, ghost=True)
    plan = E.TurnFree(wh)
    ex.bind(plan)
    pos, steps = [0], 0
    while pos != [19]:
        ex.load(pos, plan.plan(steps, pos, [0], [[(19, 0)]], [0]))
        pos = ex.step(pos, [0])
        steps += 1
    assert steps == 9                                         # 7 moves, the first turn and the turn in the middle


def test_ghosts_pass_through_each_other():
    wh = Grid()
    ex = E.Execution(wh, 2, E.Flat(), 0, turns=False, ghost=True)
    ex.load([1, 2], [[2], [1]])
    assert ex.step([1, 2], [0, 0]) == [2, 1]


def test_bad_floor_is_a_fifth_of_the_cells_and_fixed():
    wh, _, _ = world("medium")
    a, b = E.bad_floor(wh, 0), E.bad_floor(wh, 0)
    assert a == b and len(a) == round(0.2 * len(wh.free))


def test_the_state_function():
    wh, _, _ = world("medium")
    r = E.StateDelay(wh, 0.1, 0.02)
    corner = wh.cell(0, 0)
    station = wh.stations[0]
    assert r.slip(0, corner, 0) == pytest.approx(0.05)         # type 0: half the speed factor
    assert r.slip(1, corner, 0) == pytest.approx(0.1)
    assert r.slip(2, corner, 0) == pytest.approx(0.15)
    assert r.slip(1, corner, 4) == pytest.approx(0.2)
    assert r.slip(1, station, 0) == pytest.approx(0.2)
    assert r.slip(2, station, 100) == E.TOP
    assert {r.stop(c) for c in wh.free} == {0.0, 0.02}


def test_expected_delay_of_the_function_matches_a_simulation_of_it():
    wh, _, _ = world("medium")
    r = E.StateDelay(wh, 0.1, 0.02)
    table = np.zeros((wh.height * wh.width, E.MAX_N))
    cell = next(c for c in wh.free if c in r.bad)
    table[cell, 2] = 1
    extra = r.expected(wh, table)
    p = r.base(1, cell, 2)
    assert extra[1][cell] == pytest.approx((1 + 0.02 * 12.5) / (1 - p) - 1)
    assert extra[0][cell] < extra[1][cell] < extra[2][cell]


def test_flat_expected_is_the_same_everywhere():
    wh, _, _ = world()
    extra = E.Flat(0.2, 0.004).expected(wh, None)
    assert extra[0][5] == pytest.approx(1.05 / 0.8 - 1)


def test_edge_stats_leave_out_the_skipped_steps():
    wh, _, _ = world()
    a = wh.free[40]
    b = next(c for c in wh.neighbours[a])
    goal = b
    traj = np.array([[a], [a], [a], [b]])
    targets = np.array([[goal], [goal], [goal]])
    lost = T.edge_stats(wh, traj, targets).extra()[(a, b)]
    kept = T.edge_stats(wh, traj, targets, skip=np.array([[1], [0], [0]])).extra()[(a, b)]
    assert (lost, kept) == (2.0, 1.0)


def test_trip_hungarian_gives_every_trip_once():
    wh, cat, orders = world(k=24)
    groups = B.savings(orders, B.TripCost(wh, wh, sim.SERVICE), 3)
    ep = sim.run_episode(wh, cat, orders, "2opt", "pibt", 0, dispatcher=D.TripHungarian(wh, groups, method="2opt"))
    rep = judged(wh, cat, orders, ep, capacity=3)
    assert rep["ok"] and rep["delivered"] == len(orders), rep["problems"]


def test_trip_hungarian_prefers_the_cheap_trip():
    wh, _, orders = world(k=12)
    groups = [[o] for o in orders]
    d = D.TripHungarian(wh, groups, method="2opt", reassign=False)
    pending = list(orders)
    pos = [wh.parking[0]]
    (r, group), = d.choose(0, [0], pending, pos)
    costs = d.costs([0], d.trips, pos)[0]
    assert group == d.trips[int(np.argmin(costs))] and len(pending) == len(orders) - 1


def test_typed_metrics_route_by_robot_type():
    wh, cat, orders = world(k=18)
    plain = E.TurnMetric(wh)
    heavy = E.TurnMetric(wh, {(a, b): 3.0 for a in wh.free for b in wh.neighbours[a]})
    m = E.Typed([plain, heavy], plain, lambda r: r % 2)
    groups = B.savings(orders, B.TripCost(wh, m, sim.SERVICE), 3)
    d = D.TripHungarian(wh, groups, metric=m, method="2opt")
    costs = d.costs([0, 1], d.trips, [wh.parking[0], wh.parking[0]])
    assert (costs[1] > costs[0]).all()
    ep = sim.run_episode(wh, cat, orders, "2opt", "pibt", 0, dispatcher=d, metric=m, planner_args={"metric": m}, execution=E.Execution(wh, wh.layout.robots, E.Flat(0.1, 0.002), 0))
    rep = judged(wh, cat, orders, ep, capacity=3, rotation=True)
    assert rep["ok"] and rep["delivered"] == len(orders), rep["problems"]
