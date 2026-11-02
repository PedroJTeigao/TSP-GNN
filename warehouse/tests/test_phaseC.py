import numpy as np
import pytest

from warehouse import batching as B
from warehouse import constraints as C
from warehouse import dsl as L
from warehouse import episode as P
from warehouse import exec as E
from warehouse import mapf as M
from warehouse import reqgen as G
from warehouse import reqparser as R
from warehouse import sequencing as S
from warehouse import sim
from warehouse import verify as V
from warehouse import world as Y


@pytest.fixture(scope="module")
def w():
    return Y.make(0)


@pytest.fixture(scope="module")
def ctx(w):
    return L.Context(w)


def customer_with(w, ctx, kind):
    return next(o.customer for o in w.orders if kind in ctx.kinds[o.oid])


def test_world_kinds_corridors_and_robots(w, ctx):
    assert all(len({w.shelf_kind[s] for s in range(k, k + Y.BLOCK)}) == 1 for k in range(0, len(w.shelf_kind), Y.BLOCK))
    assert all(len(v - {"normal"}) <= 1 for v in ctx.kinds.values())
    assert len(w.corridors) == 20 and all(c not in w.wh.access.values() for k in range(11, 21) for c in w.corridors[k])
    assert all(w.shelves_in(k) for k in range(1, 11))
    assert not set(w.corridors[1]) & set(w.corridors[11])          # the crossings belong to no corridor
    assert set(w.robots) == {"A", "B"}


def test_selectors_by_each_key(w, ctx):
    assert L.select_orders({"order": 7}, ctx) == {7}
    frozen = L.select_orders({"kind": "frozen"}, ctx)
    assert frozen == {o for o, v in ctx.kinds.items() if "frozen" in v}
    assert L.select_orders({"station": "B"}, ctx) == {o.oid for o in w.orders if o.station == w.stations["B"]}
    name = w.orders[0].customer
    both = L.select_orders({"kind": "frozen", "except": [{"customer": name}]}, ctx)
    assert both == frozen - {o.oid for o in w.orders if o.customer == name}
    assert L.select_orders({"any_of": [{"order": 1}, {"order": 2}]}, ctx) == {1, 2}
    assert L.select_orders([1, 2], ctx) == {1, 2}
    shelves = L.select_shelves({"corridor": 3}, ctx)
    assert shelves == set(w.shelves_in(3)) & set(ctx.used)
    assert L.select_shelves({"all": True, "except": [{"kind": "frozen"}]}, ctx) == {s for s in ctx.used if w.shelf_kind[s] != "frozen"}


@pytest.mark.parametrize("bad", [{"kind": "wet"}, {"corridor": 99}, {"station": "Z"}, {"customer": "Nobody"}, {"order": 999}, {"color": "red"}, {"any_of": []}])
def test_bad_selectors_are_errors(ctx, bad):
    with pytest.raises(L.DslError):
        L.select_orders(bad, ctx)


def test_expansion_of_each_rule_type(w, ctx):
    p = L.expand_rule({"type": "pick_order", "first": {"kind": "heavy"}, "then": {"kind": "frozen"}}, ctx)
    assert len(p) == 1 and all(w.shelf_kind[s] == "heavy" for s in p[0]["first"])
    p = L.expand_rule({"type": "close", "corridors": [5, 14], "t1": 100, "t2": 200}, ctx)
    assert [(x["corridor"], x["t1"], x["t2"]) for x in p] == [(5, 100, 200), (14, 100, 200)]
    p = L.expand_rule({"type": "close", "corridor": "corridor 14"}, ctx)
    assert (p[0]["t1"], p[0]["t2"]) == (0, L.FOREVER)
    p = L.expand_rule({"type": "no_mix", "a": {"order": 3}, "b": {"order": 1}}, ctx)
    assert p == [{"type": "no_mix", "pair": [1, 3]}]
    assert L.expand_rule({"type": "robot_only", "who": {"order": 2}, "robot": "b"}, ctx)[0]["robot"] == "B"
    assert L.expand_rule({"type": "deadline", "who": {"order": 2}, "t": 300}, ctx) == [{"type": "deadline", "order": 2, "t": 300}]
    assert L.expand_rule({"type": "station", "who": {"order": 2}, "station": "station c"}, ctx)[0]["station"] == "C"


def test_score_treats_a_group_and_its_parts_alike(ctx):
    rule = {"type": "deadline", "who": {"any_of": [{"order": 1}, {"order": 2}]}, "t": 300}
    label = [{"dsl": rule, "prims": L.expand_rule(rule, ctx)}]
    parts = L.expand_rule({"type": "deadline", "who": {"order": 1}, "t": 300}, ctx) + L.expand_rule({"type": "deadline", "who": {"order": 2}, "t": 300}, ctx)
    assert L.score(label, parts) == {"correct": [True], "exact": True, "wrong": 0}
    assert L.score(label, parts[:1])["correct"] == [False]
    extra = parts + L.expand_rule({"type": "deadline", "who": {"order": 3}, "t": 300}, ctx)
    s = L.score(label, extra)
    assert s["correct"] == [True] and not s["exact"]


def test_error_classes(w, ctx):
    rule = {"type": "pick_order", "first": {"kind": "heavy"}, "then": {"kind": "frozen"}}
    label = {"dsl": rule, "prims": L.expand_rule(rule, ctx)}
    flipped = L.expand_rule({"type": "pick_order", "first": {"kind": "frozen"}, "then": {"kind": "heavy"}}, ctx)
    assert L.classify_error(label, flipped, ctx) == "inverted"
    assert L.classify_error(label, [], ctx) == "missing"
    name = next(o.customer for o in w.orders if "frozen" in ctx.kinds[o.oid])
    rule = {"type": "deadline", "who": {"kind": "frozen", "except": [{"customer": name}]}, "t": 300}
    label = {"dsl": rule, "prims": L.expand_rule(rule, ctx)}
    assert L.classify_error(label, L.expand_rule({"type": "deadline", "who": {"kind": "frozen"}, "t": 300}, ctx), ctx) == "exception lost"
    assert L.classify_error(label, L.expand_rule({**rule, "t": 400}, ctx), ctx) == "wrong value"
    close = {"type": "close", "corridors": [5], "t1": 10, "t2": 20}
    assert L.classify_error({"dsl": close, "prims": L.expand_rule(close, ctx)}, L.expand_rule({"type": "close", "corridors": [6], "t1": 10, "t2": 20}, ctx), ctx) == "missing"


def rejected(rules, ctx):
    return [x["reason"] for x in L.check([{"prims": L.expand_rule(r, ctx)} for r in rules], ctx)[1]]


def test_validator_pick_order_cycle(ctx):
    a = {"type": "pick_order", "first": {"kind": "heavy"}, "then": {"kind": "frozen"}}
    b = {"type": "pick_order", "first": {"kind": "frozen"}, "then": {"kind": "fragile"}}
    c = {"type": "pick_order", "first": {"kind": "fragile"}, "then": {"kind": "heavy"}}
    assert rejected([a, b], ctx) == []
    assert rejected([a, b, c], ctx) == ["pick order cycle"]


def test_validator_closure_isolating_a_shelf(ctx):
    assert rejected([{"type": "close", "corridors": [5]}], ctx) == ["closure isolates a shelf"]
    assert rejected([{"type": "close", "corridors": [14]}], ctx) == []
    assert rejected([{"type": "close", "corridors": [5], "t1": 100, "t2": 200}], ctx) == []
    assert rejected([{"type": "close", "corridors": [5], "t1": 100}], ctx) == ["closure isolates a shelf"]     # 100 steps do not serve them all
    assert rejected([{"type": "close", "corridors": [14], "t1": 100}], ctx) == []


def test_validator_impossible_order(w, ctx):
    name = customer_with(w, ctx, "frozen")
    assert rejected([{"type": "no_mix", "a": {"kind": "frozen"}, "b": {"customer": name}}], ctx) == ["incompatible with itself: one order is on both sides"]
    assert rejected([{"type": "no_mix", "a": {"kind": "frozen"}, "b": {"kind": "heavy"}}], ctx) == []


def test_validator_robots_stations_deadlines(w, ctx):
    assert rejected([{"type": "robot_only", "who": {"kind": "heavy"}, "robot": "B"}], ctx) == ["heavy order on a robot that cannot carry it"]
    a = {"type": "robot_only", "who": {"order": 1}, "robot": "A"}
    b = {"type": "robot_only", "who": {"order": 1}, "robot": "B"}
    assert "two robot types for one order" in rejected([a, b], ctx) or ctx.kinds[1] & {"heavy"}
    a = {"type": "station", "who": {"order": 1}, "station": "A"}
    assert rejected([a, {"type": "station", "who": {"order": 1}, "station": "B"}], ctx) == ["two stations for one order"]
    assert rejected([{"type": "deadline", "who": {"order": 1}, "t": 5}], ctx) == ["deadline out of reach"]
    assert rejected([{"type": "deadline", "who": {"order": 1}, "t": 500}], ctx) == []
    corridor = next(k for k in range(1, 11) if set(w.orders[1].shelves) & set(w.shelves_in(k)))
    assert rejected([{"type": "close", "corridors": [corridor], "t2": 600}, {"type": "deadline", "who": {"order": 1}, "t": 500}], ctx) == ["deadline inside a closure of its corridor"]


def test_exact_rules_and_lkh_sop_and_tsptw_agree():
    rng = np.random.default_rng(1)
    xy = rng.integers(0, 60, (12, 2))
    d = np.abs(xy[:, None] - xy[None]).sum(-1).tolist()
    pairs = [(1, 5), (2, 5), (5, 7), (3, 8)]
    ex = C.exact_rules(d, pairs, {})
    lk = C.sop(d, pairs)
    assert C.feasible(ex, pairs) and C.feasible(lk, pairs) and sorted(lk) == list(range(1, 11))
    assert S.path_length(d, [0] + lk + [11]) == S.path_length(d, [0] + ex + [11])
    free = S.path_length(d, [0] + C.exact_rules(d, [], {}) + [11])
    tw = C.tsptw(d, {11: free + 5})
    assert S.path_length(d, [0] + tw + [11]) <= free + 5
    assert C.exact_rules(d, [], {11: free - 1}) is None


def test_exact_rules_keeps_a_window_inside_the_path():
    d = [[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 1], [9, 9, 1, 0]]
    assert C.exact_rules(d, [], {2: 10})[0] == 1 or C.exact_rules(d, [], {2: 10}) == [2, 1]
    assert C.exact_rules(d, [(2, 1)], {}) == [2, 1]


def test_rules_refuse_merges_and_give_each_trip_its_robots(w, ctx):
    a, b = w.orders[0], w.orders[1]
    r = C.Rules(w, L.expand_rule({"type": "no_mix", "a": {"order": a.oid}, "b": {"order": b.oid}}, ctx))
    assert r.penalty([a], [b]) == float("inf")
    assert r.penalty([a], [w.orders[2]]) == 0.0
    r = C.Rules(w, L.expand_rule({"type": "deadline", "who": {"order": a.oid}, "t": 500}, ctx))
    assert r.penalty([a], [b]) == float("inf")
    heavy = next(o for o in w.orders if "heavy" in ctx.kinds[o.oid])
    plain = next(o for o in w.orders if ctx.kinds[o.oid] == {"normal"})
    r = C.Rules(w, L.expand_rule({"type": "robot_only", "who": {"order": plain.oid}, "robot": "B"}, ctx))
    assert r.allowed([heavy]) == {"A"} and r.allowed([plain]) == {"B"}
    assert r.penalty([heavy], [plain]) == float("inf")


def test_station_rule_changes_the_order_only_in_the_copy(w, ctx):
    r = C.Rules(w, L.expand_rule({"type": "station", "who": {"order": 3}, "station": "C"}, ctx))
    assert r.orders()[3].station == w.stations["C"]
    assert w.orders[3].station == Y.make(0).orders[3].station


def test_closed_grid_drops_cells_and_paths_go_round(w):
    cells = w.corridors[14]
    g = C.closed_grid(w.wh, cells)
    assert all(not g.neighbours[c] for c in cells) and w.wh.neighbours[cells[0]]
    above, below = cells[0] - w.wh.width, cells[1] + w.wh.width
    assert g.dist_to(above)[below] > w.wh.dist_to(above)[below]


def test_planner_gate_forbids_entry_before_a_closure(w):
    r = C.Rules(w, [{"type": "close", "corridor": 14, "t1": 100, "t2": 200}])
    plan = C.RulePlanner(r)
    assert plan.entries(0) == set()
    u, v = r.mouths[0][0]
    assert (u, v, 20 - 0) not in plan.entries(10) and (u, v, 0) in plan.entries(100 - C.GUARD)
    assert plan.entries(200) == set()
    plan.entries(100)
    assert plan.pibt.shut == r.windows[0][0]
    plan.entries(300)
    assert plan.pibt.shut == set()


def test_pibt_does_not_enter_a_shut_cell(w):
    wh = w.wh
    a = w.corridors[14][0]
    start = next(u for u in wh.neighbours[a] if u not in w.corridors[14])
    p = M.PIBT(wh)
    assert p.step([start], [0], [[(a, 3)]], [0]) == [a]
    p.shut = {a}
    assert p.step([start], [0], [[(a, 3)]], [0]) != [a]


def episode(w, prims):
    return P.run(w, prims, C.to_checks(w, prims) + C.world_checks(w), w.seed)[0]


@pytest.mark.parametrize("rule", [
    {"type": "pick_order", "first": {"all": True, "except": [{"kind": "frozen"}]}, "then": {"kind": "frozen"}},
    {"type": "close", "corridors": [5], "t1": 60, "t2": 200},
    {"type": "close", "corridors": [14]},
    {"type": "no_mix", "a": {"kind": "frozen"}, "b": {"kind": "heavy"}},
    {"type": "robot_only", "who": {"kind": "frozen"}, "robot": "A"},
    {"type": "deadline", "who": {"priority": "urgent"}, "t": 600},
    {"type": "station", "who": {"kind": "frozen"}, "station": "B"},
])
def test_episode_keeps_each_rule_on_a_small_batch(rule):
    w = Y.make(1, robots=10, count=25)
    prims = L.expand_rule(rule, L.Context(w))
    rep = episode(w, prims)
    assert rep["ok"], rep["problems"]
    assert rep["delivered"] == 25


@pytest.fixture(scope="module")
def blind():
    w = Y.make(1, robots=10, count=25)
    r = C.Rules(w, [])
    orders = r.orders()
    groups = B.savings(orders, P.Costs(w.wh, w.wh, sim.SERVICE), P.CAP, r.penalty)
    ep = sim.run_episode(w.wh, w.cat, orders, "lkh", C.RulePlanner(r, 1), 1, dispatcher=C.RuleHungarian(w.wh, groups, r, w.robots), execution=E.Execution(w.wh, 10, E.Flat(*P.R1), 1), router=r.route, max_steps=P.MAX_STEPS)
    return w, orders, ep


def judged(blind, checks):
    w, orders, ep = blind
    return V.verify(ep["traj"], w.wh.grid, P.data(orders), w.cat.shelf_of, w.wh.shelves, ep["events"], ep["service"], ep["step_seconds"], rules=checks, capacity=P.CAP, rotation=True)


def test_verifier_sees_each_kind_of_violation(blind):
    w, orders, ep = blind
    picks = [e for e in ep["events"] if e["kind"] == "pick"]
    a, b = picks[0], next(e for e in picks[1:] if e["robot"] == picks[0]["robot"] and e["shelf"] != picks[0]["shelf"] and e["start"] >= picks[0]["end"])
    assert judged(blind, [{"type": "pick_before", "first": [b["shelf"]], "then": [a["shelf"]]}])["problems"]["rules_broken"] == 1
    assert judged(blind, [{"type": "pick_before", "first": [a["shelf"]], "then": [b["shelf"]]}])["problems"]["rules_broken"] == 0
    cell = int(ep["traj"][50, 0])
    assert judged(blind, [{"type": "forbidden_cells", "cells": [cell], "from": 50, "to": 51}])["problems"]["rules_broken"] == 1
    assert judged(blind, [{"type": "forbidden_cells", "cells": [cell], "from": 0, "to": 1}])["problems"]["rules_broken"] == (cell == int(ep["traj"][0, 0]))
    done = next(e for e in ep["events"] if e["kind"] == "deliver")
    assert judged(blind, [{"type": "deadline", "order": done["order"], "t": done["end"] - 1}])["problems"]["rules_broken"] == 1
    assert judged(blind, [{"type": "deadline", "order": done["order"], "t": done["end"]}])["problems"]["rules_broken"] == 0
    assert judged(blind, [{"type": "robot_only", "order": done["order"], "robots": [done["robot"]]}])["problems"]["rules_broken"] == 0
    assert judged(blind, [{"type": "robot_only", "order": done["order"], "robots": [done["robot"] + 1]}])["problems"]["rules_broken"] == 1
    assert judged(blind, [{"type": "station", "order": done["order"], "cell": done["cell"]}])["problems"]["rules_broken"] == 0
    assert judged(blind, [{"type": "station", "order": done["order"], "cell": done["cell"] + 1}])["problems"]["rules_broken"] == 1
    together = {}
    for e in ep["events"]:
        if e["kind"] == "assign":
            together.setdefault((e["robot"], e["t"]), []).append(e["order"])
    final = {e["order"]: (e["robot"], e["t"]) for e in ep["events"] if e["kind"] == "assign"}
    trips = {}
    for o, key in final.items():
        trips.setdefault(key, []).append(o)
    x, y = next(g[:2] for g in trips.values() if len(g) >= 2)
    assert judged(blind, [{"type": "no_mix", "a": x, "b": y}])["problems"]["rules_broken"] == 1


def test_generator_labels_pass_the_validator(ctx):
    for k in range(10):
        rng = np.random.default_rng([G.SEED, 9, k])
        rules = G.sample_rules(rng, ctx, int(rng.integers(2, 5)))
        assert len({r["dsl"]["type"] for r in rules}) == len(rules)
        assert not L.check(rules, ctx)[1]


def test_prompts_fit_and_hold_the_legend(w):
    assert "Dona" in L.user_prompt(w, "x") or w.customers[0] in L.user_prompt(w, "x")
    assert "pick_order" in L.system_prompt()
    assert len(L.system_prompt()) + len(L.user_prompt(w, "x" * 2000)) < 4 * 8192


def test_p1_reads_every_template(w, ctx):
    for name, shapes in G.SHAPES.items():
        for shape in shapes:
            rng = np.random.default_rng([G.SEED, 77])
            for _ in range(20):
                dsl, sentence = shape(rng, w)
                try:
                    want = L.keyset(L.expand_rule(dsl, ctx))
                except L.DslError:
                    continue
                got, errors = L.validate(R.parse_p1("- " + sentence, w), ctx)
                assert L.keyset([p for r in got for p in r["prims"]]) == want, (sentence, errors)


def test_p0_has_no_exceptions(w):
    got = R.parse_p0("- Every frozen order except those of Dona Maria must go to station B.", w)["rules"]
    assert got == [{"type": "station", "who": {"kind": "frozen"}, "station": "B"}]


def test_p1_drops_remarks_and_follows_references(w):
    name = w.customers[0]
    assert R.parse_p1(f"Last week a robot happened to pick heavy products before frozen products, which was only a coincidence. {name} paid the last invoice on time.", w) == {"rules": []}
    text = f"{name} is very busy; the orders of {name} go to station B, and they must be delivered by step 400."
    assert {"type": "deadline", "who": {"customer": name}, "t": 400} in R.parse_p1(text, w)["rules"]


def test_trigger_of_the_hybrid(w):
    assert not R.send_to_llm("- Corridor 14 is closed.\n- All urgent orders must be delivered by step 500.", w)
    assert R.send_to_llm("- Corridor 14 is closed.\n- The robots must never hurry.", w)
