"""Assignment rules, distances with traffic and trips with two orders (phase B0)."""

import itertools
from collections import deque

import numpy as np
import pytest

from warehouse import batching as B
from warehouse import conflicts as C
from warehouse import dispatch as D
from warehouse import env as W
from warehouse import insertion as I
from warehouse import sequencing as S
from warehouse import sim
from warehouse import traffic as T
from warehouse import verify as V

LAY = W.Layout("tiny", 12, 16, 2, 2, corridor=1, stations=2, robots=4)


@pytest.fixture(scope="module")
def world():
    wh = W.Warehouse(LAY)
    cat = W.Catalogue(wh, 200, 3)
    return wh, cat, W.batch_orders(wh, cat, 12, 3)


def judge(wh, cat, orders, ep, capacity=1):
    return V.verify(ep["traj"], wh.grid, [o.__dict__ for o in orders], cat.shelf_of, wh.shelves, ep["events"], ep["service"],
                    ep["step_seconds"], capacity=capacity)


def brute(cost):
    """Minimum total cost of giving each row its own column, trying everything."""
    n, m = cost.shape
    return min(sum(cost[i, j] for i, j in enumerate(p)) for p in itertools.permutations(range(m), n))


@pytest.mark.parametrize("seed", range(10))
def test_hungarian_matches_brute_force(world, seed):
    wh, cat, orders = world
    rng = np.random.default_rng(seed)
    free = rng.choice(4, 3, replace=False).tolist()
    pos = rng.choice(wh.free, 4, replace=False).tolist()
    h = D.Hungarian(wh, method="exact")
    pool = deque(orders[:6])
    cost = h.costs(free, list(pool), pos)
    got = h.choose(0, list(free), pool, pos)
    assert len(got) == 3 and len(pool) == 3 and len({r for r, _ in got}) == 3
    index = {o.oid: j for j, o in enumerate(orders[:6])}
    assert sum(cost[free.index(r), index[g[0].oid]] for r, g in got) == brute(cost)


def test_hungarian_cost_is_distance_plus_length(world):
    wh, cat, orders = world
    h = D.Hungarian(wh, method="exact")
    o = orders[0]
    first, length = h.profile(o)
    pos = {2: wh.free[5]}
    assert h.costs([2], [o], pos)[0, 0] == wh.dist_to(first)[wh.free[5]] + length


def test_open_path_is_the_best_free_start(world):
    wh, cat, orders = world
    o = next(o for o in orders if 3 <= len(o.shelves) <= 6)
    stops = [wh.access[s] for s in o.shelves]
    first, length = S.open_path(wh, stops, o.station, "lkh")
    assert first in stops
    best = min(sum(wh.dist_to(a)[b] for a, b in zip(p, p[1:] + (o.station,))) for p in itertools.permutations(stops))
    assert length == best


def test_fifo_only_looks_at_the_oldest(world):
    wh, cat, orders = world
    pool = deque(orders[:6])
    got = D.Hungarian(wh, method="exact", fifo=True).choose(0, [0, 1], pool, list(wh.free[:4]))
    assert {g[0].oid for _, g in got} == {orders[0].oid, orders[1].oid}


@pytest.mark.parametrize("reassign", [False, True])
def test_hungarian_episode_passes_the_verifier(world, reassign):
    wh, cat, orders = world
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=D.Hungarian(wh, method="2opt", reassign=reassign))
    rep = judge(wh, cat, orders, ep)
    assert rep["ok"] and rep["delivered"] == 12


def test_reassigning_changes_robots_that_have_not_started():
    wh = W.Warehouse(LAY)
    cat = W.Catalogue(wh, 200, 5)
    orders = W.batch_orders(wh, cat, 30, 5)
    plain = sim.run_episode(wh, cat, orders, "2opt", "pibt", 5, dispatcher=D.Hungarian(wh, method="2opt"))
    again = sim.run_episode(wh, cat, orders, "2opt", "pibt", 5, dispatcher=D.Hungarian(wh, method="2opt", reassign=True))
    count = lambda ep: sum(e["kind"] == "assign" for e in ep["events"])
    assert count(plain) == 30 and count(again) > 30
    assert judge(wh, cat, orders, again)["ok"]


def test_fixed_plan_repeats_the_episode(world):
    wh, cat, orders = world
    a = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3)
    plan = {}
    for e in a["events"]:
        if e["kind"] == "assign":
            plan.setdefault(e["robot"], []).append(e["order"])
    b = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=D.FixedPlan(plan))
    assert np.array_equal(a["traj"], b["traj"])


def test_traffic_without_penalty_is_the_empty_grid(world):
    wh = world[0]
    m = T.Traffic(wh, np.zeros((wh.height, wh.width)))
    for c in wh.free[::37]:
        assert m.dist_to(c) == [T.SCALE * x if x > 0 else x for x in wh.dist_to(c)]


def test_traffic_distances_are_symmetric(world):
    wh = world[0]
    m = T.Traffic(wh, np.random.default_rng(1).exponential(0.5, (wh.height, wh.width)))
    cells = wh.free[::23]
    for a, b in itertools.combinations(cells, 2):
        assert m.dist_to(a)[b] == m.dist_to(b)[a]
        assert m.dist_to(a)[b] >= T.SCALE * wh.dist_to(a)[b]


def test_traffic_goes_around_a_jam():
    grid = W.Warehouse(W.Layout("open", 12, 16, 1, 1, corridor=1, stations=1, robots=2))
    a, b = grid.cell(0, 0), grid.cell(0, 6)                   # the top row is free
    penalty = np.zeros((grid.height, grid.width))
    penalty[0, 3] = 50.0
    m = T.Traffic(grid, penalty)
    assert grid.dist_to(a)[b] == 6
    assert m.dist_to(a)[b] == T.SCALE * 8                     # two steps more, through row 1


def test_lost_per_pass():
    z = np.zeros((1, 3), dtype=np.int32)
    res = {"waits": z + [2, 0, 0], "detours": z + [1, 0, 0], "visits": z + [6, 4, 0]}
    assert T.lost_per_pass([res]).tolist() == [[1.0, 0.0, 0.0]]              # 3 lost in 3 passes


def test_traffic_changes_the_visit_order(world):
    wh, cat, orders = world
    o = max(orders, key=lambda o: len(o.shelves))
    stops = [wh.access[s] for s in o.shelves]
    penalty = np.zeros((wh.height, wh.width))
    r, c = wh.xy(stops[0])
    penalty[r, :] = 30.0
    m = T.Traffic(wh, penalty)
    d = S.matrix(m, wh.free[0], stops, o.station)
    assert all(isinstance(x, int) for row in d for x in row)
    idx, _ = S.sequence(m, wh.free[0], stops, o.station, "lkh")
    assert sorted(idx) == list(range(len(stops)))
    best = min(S.path_length(d, [0, *p, len(d) - 1]) for p in itertools.permutations(range(1, len(d) - 1))) if len(stops) <= 7 else None
    if best is not None:
        assert S.path_length(d, [0] + [k + 1 for k in idx] + [len(d) - 1]) == best


def test_episode_with_traffic_passes_the_verifier(world):
    wh, cat, orders = world
    base = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3)
    m = T.Traffic(wh, T.lost_per_pass([C.analyse(wh, base["traj"], base["targets"])]))
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, metric=m)
    assert judge(wh, cat, orders, ep)["ok"]


def test_batcher_takes_the_largest_overlap(world):
    wh, cat, orders = world
    a = W.Order(0, [], wh.stations[0], 0, [1, 2, 3])
    b = W.Order(1, [], wh.stations[0], 0, [7, 8, 9])
    c = W.Order(2, [], wh.stations[1], 0, [2, 3, 9])
    d = W.Order(3, [], wh.stations[0], 0, [1, 9, 10])
    pool = deque([a, b, c, d])
    got = D.Batcher(wh).choose(0, [0], pool, list(wh.free[:4]))
    assert [o.oid for o in got[0][1]] == [0, 2] and [o.oid for o in pool] == [1, 3]


def test_two_orders_share_a_trip(world):
    wh, cat, orders = world
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=D.Batcher(wh))
    rep = judge(wh, cat, orders, ep, capacity=2)
    assert rep["ok"] and rep["delivered"] == 12
    assert judge(wh, cat, orders, ep)["problems"]["wrong_overlap"] > 0


def test_a_shared_shelf_serves_both_orders(world):
    wh, cat, orders = world
    shelf = orders[0].shelves[0]
    twin = W.Order(1, orders[1].products, orders[1].station, 0, sorted({*orders[1].shelves, shelf}))
    stops, deliveries, _ = S.route(wh, wh, wh.free[0], (orders[0], twin), "2opt", {})
    assert dict(stops)[shelf] == (orders[0].oid, 1)
    assert sorted(o for _, o in deliveries) == sorted((orders[0].oid, 1))
    assert sorted(s for s, _ in stops) == sorted({*orders[0].shelves, *twin.shelves})


@pytest.mark.parametrize("rule", ["far", "near", "load", "fixed"])
def test_priority_rules(world, rule):
    wh, cat, orders = world
    ep = sim.run_episode(wh, cat, orders, "2opt", "pibt", 3, planner_args={"rule": rule})
    assert judge(wh, cat, orders, ep)["ok"]


def test_edge_metric_is_directed():
    wh = W.Warehouse(LAY)
    a, b = wh.cell(0, 0), wh.cell(0, 1)
    m = T.EdgeMetric(wh, {(a, b): 7.0})
    assert m.dist_to(b)[a] == 3 * T.SCALE                     # around, by row 1, is cheaper than 8 steps
    assert m.dist_to(a)[b] == T.SCALE


def test_edge_metric_without_extra_is_the_empty_grid():
    wh = W.Warehouse(LAY)
    m = T.EdgeMetric(wh, {})
    for c in wh.free[::37]:
        assert m.dist_to(c) == [T.SCALE * x if x > 0 else x for x in wh.dist_to(c)]


def test_edge_metric_symmetric_for_lkh():
    wh = W.Warehouse(LAY)
    rng = np.random.default_rng(2)
    extra = {(c, n): float(rng.exponential(1.0)) for c in wh.free[::3] for n in wh.neighbours[c]}
    m = T.EdgeMetric(wh, extra, symmetric=True)
    for a, b in itertools.combinations(wh.free[::29], 2):
        assert m.dist_to(a)[b] == m.dist_to(b)[a]


def test_edge_metric_matches_dijkstra_by_hand():
    wh = W.Warehouse(LAY)
    rng = np.random.default_rng(3)
    extra = {(c, n): float(rng.integers(0, 4)) for c in wh.free for n in wh.neighbours[c]}
    m = T.EdgeMetric(wh, extra)
    goal = wh.free[40]
    d = m.dist_to(goal)
    for c in wh.free:                                         # Bellman's equation at every cell
        if c != goal:
            assert d[c] == min(m.edge(c, n) + d[n] for n in wh.neighbours[c])


def test_edge_stats_charge_the_next_pass():
    wh = W.Warehouse(LAY)
    a, b, c = wh.cell(0, 0), wh.cell(0, 1), wh.cell(0, 2)
    s = T.EdgeStats(wh, decay=0.9)
    s.observe([a], [a], [c])                                  # waits twice, then passes
    s.observe([a], [a], [c])
    s.observe([a], [b], [c])
    s.observe([b], [c], [c])
    assert s.edges[(a, b)] == [1, 2, 2.0] and s.edges[(b, c)] == [1, 0, 0.0]
    assert s.cells[a] == [1, 2, 2.0]


def test_edge_stats_decay():
    wh = W.Warehouse(LAY)
    s = T.EdgeStats(wh, decay=0.9)
    for lost in (2, 0, 1):
        s.add(s.edges, (1, 2), lost)
    assert s.edges[(1, 2)][:2] == [3, 3]
    assert s.edges[(1, 2)][2] == pytest.approx(0.9 * (0.9 * 2) + 0.1)


def test_edge_stats_add_up_to_the_losses(world):
    wh, cat, orders = world
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3)
    x = T.edge_stats(wh, ep["traj"], ep["targets"]).export()
    a = C.analyse(wh, ep["traj"], ep["targets"])
    assert 0 < x["edge_lost"].sum() <= a["lost"]              # losses after the last pass of a leg stay unpaid
    assert (x["edge_delta"] >= 0).all()
    assert x["edge_lost"].sum() == x["cell_lost"].sum()


def test_free_flow_is_the_trip_alone(world):
    wh, cat, orders = world
    real = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3)
    plan = D.plan_of(real["events"])
    ff = sim.run_episode(wh, cat, orders, "2opt", "free", 3, dispatcher=D.FixedPlan(plan))
    rep = judge(wh, cat, orders, ff)
    assert {k for k, v in rep["problems"].items() if v} <= {"vertex_conflicts", "edge_conflicts"}
    assert rep["delivered"] == 12
    assert rep["makespan"] <= judge(wh, cat, orders, real)["makespan"]


def test_plan_of_keeps_trips_of_two():
    ev = [{"kind": "assign", "robot": 1, "order": 3, "t": 0}, {"kind": "assign", "robot": 1, "order": 2, "t": 0},
          {"kind": "assign", "robot": 0, "order": 5, "t": 0}, {"kind": "assign", "robot": 1, "order": 5, "t": 4},
          {"kind": "pick", "robot": 1, "order": 5, "t": 9}]
    assert D.plan_of(ev) == {1: [[2, 3], [5]]}


def test_iterative_reaches_a_local_optimum(world):
    wh, cat, orders = world
    pool = deque(orders[:6])
    it = D.Iterative(wh, method="exact")
    free, pos = [0, 1, 2], list(wh.free[:4])
    cost = it.costs(free, list(pool), pos)
    got = it.choose(0, list(free), pool, pos)
    index = {o.oid: j for j, o in enumerate(orders[:6])}
    pairs = [(free.index(r), index[g[0].oid]) for r, g in got]
    spare = set(range(6)) - {j for _, j in pairs}
    for i, j in pairs:
        assert all(cost[i, q] >= cost[i, j] for q in spare)
        assert all(cost[i, j] + cost[k, q] <= cost[i, q] + cost[k, j] for k, q in pairs)


@pytest.mark.parametrize("mode", ["edge", "near"])
def test_live_costs_episode(world, mode):
    wh, cat, orders = world
    d = D.LiveCosts(wh, mode, method="2opt")
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=d)
    assert judge(wh, cat, orders, ep)["ok"]
    assert d.stats.edges


def test_savings_merges_what_saves_most(world):
    wh, cat, orders = world
    cost = B.TripCost(wh, wh, 3)
    groups = B.savings(orders, cost, 2)
    assert sorted(o.oid for g in groups for o in g) == sorted(o.oid for o in orders)
    assert all(len(g) <= 2 for g in groups)
    assert B.total(groups, cost) <= sum(cost([o]) for o in orders)


def test_savings_against_matching_and_cvrp(world, tmp_path):
    wh, cat, orders = world
    cost = B.TripCost(wh, wh, 3)
    best = B.exact_pairs(orders, cost)
    assert best <= B.total(B.savings(orders, cost, 2), cost)
    assert B.cvrp_lkh(orders, cost, str(tmp_path)) == best


def test_exact_clusters_cover_every_order_once(world):
    wh, cat, orders = world
    groups = B.exact_clusters(orders, B.TripCost(wh, wh, 3))
    assert sorted(o.oid for g in groups for o in g) == sorted(o.oid for o in orders) and max(map(len, groups)) <= 2


@pytest.mark.parametrize("cap", [2, 3])
def test_clusters_pass_the_verifier(world, cap):
    wh, cat, orders = world
    groups = B.savings(orders, B.TripCost(wh, wh, 3), cap)
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=D.Clusters(wh, groups))
    assert judge(wh, cat, orders, ep, capacity=cap)["ok"]


def test_regions_cover_every_order(world):
    wh, cat, orders = world
    groups = B.regions(wh, orders, 2, 3, 0)
    assert sorted(o.oid for g in groups for o in g) == sorted(o.oid for o in orders)
    assert all(len(g) <= 2 for g in groups)


def test_highways_and_flow_are_verified(world):
    wh, cat, orders = world
    m = T.EdgeMetric(wh, T.crisscross(wh, [1.0], [1.0]))
    assert m.extra
    for args in ({"metric": m}, {"with_flow": 0.1, "against": 0.5}):
        ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, planner_args=args)
        assert judge(wh, cat, orders, ep)["ok"]


def test_insertion_gives_every_order(world):
    wh, cat, orders = world
    ins = I.Insertion(wh, orders, W.start_cells(wh, 4, 3), 3, {})
    plan = ins.build()
    assert sorted(o for v in plan.values() for o in v) == sorted(o.oid for o in orders)
    ep = sim.run_episode(wh, cat, orders, "2opt", "prioritized", 3, dispatcher=D.FixedPlan(plan))
    assert judge(wh, cat, orders, ep)["ok"]
