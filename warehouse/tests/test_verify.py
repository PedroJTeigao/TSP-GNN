"""The verifier must catch what it claims to catch, on episodes altered by hand."""

import copy

import numpy as np
import pytest

from warehouse import env as W
from warehouse import sim
from warehouse import verify as V

LAY = W.Layout("tiny", 12, 16, 2, 2, corridor=1, stations=2, robots=4)


@pytest.fixture(scope="module")
def run():
    wh = W.Warehouse(LAY)
    cat = W.Catalogue(wh, 200, 3)
    orders = W.batch_orders(wh, cat, 6, 3)
    ep = sim.run_episode(wh, cat, orders, "2opt", "pibt", 3)
    return wh, cat, [o.__dict__ for o in orders], ep


def check(run, traj=None, events=None, orders=None, rules=()):
    wh, cat, o, ep = run
    return V.verify(ep["traj"] if traj is None else traj, wh.grid, o if orders is None else orders, cat.shelf_of, wh.shelves,
                    ep["events"] if events is None else events, ep["service"], ep["step_seconds"], rules=rules)


def test_clean_episode_passes(run):
    rep = check(run)
    assert rep["ok"] and rep["delivered"] == 6
    assert rep["makespan"] == max(e["end"] for e in run[3]["events"] if e["kind"] == "deliver")
    assert rep["orders_per_hour"] == pytest.approx(6 / rep["steps"] * 3600)


def test_vertex_conflict():
    assert V.collisions(np.array([[1, 2], [3, 3]])) == (1, 0)


def test_edge_conflict():
    assert V.collisions(np.array([[1, 2], [2, 1]])) == (0, 1)


def test_three_robots_following_is_no_conflict():
    assert V.collisions(np.array([[1, 2, 3], [2, 3, 4]])) == (0, 0)


def test_teleport_and_shelf(run):
    wh = run[0]
    traj = run[3]["traj"].copy()
    traj[5, 0] = wh.cell(*wh.shelves[0])
    rep = check(run, traj=traj)
    assert not rep["ok"] and rep["problems"]["on_shelf"] == 1 and rep["problems"]["long_moves"] > 0


def test_missing_pick(run):
    ev = copy.deepcopy(run[3]["events"])
    k = next(i for i, e in enumerate(ev) if e["kind"] == "pick")
    del ev[k]
    assert check(run, events=ev)["problems"]["wrong_pick"] > 0


def test_pick_after_delivery(run):
    ev = copy.deepcopy(run[3]["events"])
    d = next(e for e in ev if e["kind"] == "deliver")
    p = next(e for e in ev if e["kind"] == "pick" and e["order"] == d["order"])
    p["start"], p["end"] = d["start"] + 1, d["start"] + 1 + run[3]["service"]
    rep = check(run, events=ev)
    assert rep["problems"]["wrong_pick"] > 0 or rep["problems"]["wrong_dwell"] > 0


def test_short_service(run):
    ev = copy.deepcopy(run[3]["events"])
    next(e for e in ev if e["kind"] == "pick")["end"] -= 1
    assert check(run, events=ev)["problems"]["wrong_dwell"] == 1


def test_wrong_station(run):
    wh, cat, orders, ep = run
    d = next(e for e in ep["events"] if e["kind"] == "deliver")
    other = copy.deepcopy(orders)
    o = next(x for x in other if x["oid"] == d["order"])
    o["station"] = next(s for s in wh.stations if s != o["station"])
    assert check(run, orders=other)["problems"]["wrong_delivery"] == 1


def test_rules(run):
    ep = run[3]
    start = int(ep["traj"][0, 0])
    assert check(run, rules=[{"type": "forbidden_cells", "cells": [start], "from": 0, "to": 1}])["problems"]["rules_broken"] == 1
    assert check(run, rules=[{"type": "deadline", "order": 0, "t": 1}])["problems"]["rules_broken"] == 1
    o = run[2][0]
    a, b = o["products"][:2]
    both = check(run, rules=[{"type": "product_order", "first": a, "then": b}, {"type": "product_order", "first": b, "then": a}])
    assert both["problems"]["rules_broken"] >= 1


def test_it_does_not_import_the_simulator():
    src = open(V.__file__).read()
    assert "import sim" not in src and "from warehouse" not in src and "mapf" not in src.split('"""')[2]
