"""Tests of the phase 5, step 3 DSL: selector expansion, rule expansion, scoring and the code check before LKH-3."""

import numpy as np
import pytest

from tsp_gnn import llm
from tsp_gnn import open_dsl as D
from tsp_gnn import open_gen as G
from tsp_gnn import rules as R


def place(c, name, kind, area):
    return {"id": c, "name": name, "kind": kind, "area": area, "x": 0.0, "y": 0.0}


@pytest.fixture
def ctx():
    coords = np.zeros((10, 2))
    coords[:, 0] = np.arange(10) / 10                     # customers on a line: distance = |i - j| / 10
    legend = [place(1, "Rua Azul 3", "hospital", "Alfama"), place(2, "Rua Verde 1", "hospital", "Belem"),
              place(3, "Rua Nova 2", "school", "Alfama"), place(4, "Rua Alta 9", "school", "Belem"),
              place(5, "Rua do Sol 4", "school", "Belem"), place(7, "Beco Escuro 8", "shop", "Alfama")]
    return D.Context(legend, coords, 10)


def test_selectors(ctx):
    assert D.expand_selector(5, ctx) == [5]
    assert D.expand_selector("8", ctx) == [8]                  # customers outside the legend are allowed by number
    assert D.expand_selector("rua azul 3", ctx) == [1]
    assert D.expand_selector({"name": "Rua Nova 2"}, ctx) == [3]
    assert D.expand_selector({"type_is": "school"}, ctx) == [3, 4, 5]
    assert D.expand_selector({"type_is": "schools"}, ctx) == [3, 4, 5]
    assert D.expand_selector({"area_is": "Alfama"}, ctx) == [1, 3, 7]
    assert D.expand_selector({"type_is": "school", "area_is": "Belem"}, ctx) == [4, 5]
    assert D.expand_selector({"type_is": "school", "except": ["Rua Alta 9"]}, ctx) == [3, 5]
    assert D.expand_selector({"any_of": ["Rua Azul 3", {"type_is": "shop"}]}, ctx) == [1, 7]


def test_nearest(ctx):
    assert D.expand_selector({"nearest": {"k": 2, "of": "Rua Nova 2"}}, ctx) == [2, 4]        # never itself
    assert D.expand_selector({"nearest": {"k": 1, "of": "Rua Azul 3"}, "type_is": "school"}, ctx) == [3]
    # the exception is removed last: "the 2 closest except Y" leaves one place
    assert D.expand_selector({"nearest": {"k": 2, "of": "Rua Nova 2"}, "except": ["Rua Verde 1"]}, ctx) == [4]


@pytest.mark.parametrize("sel", [{"type_is": "castle"}, {"area_is": "Mars"}, "Nowhere 1", 0, 10, None, True, {},
                                 {"foo": 1}, {"nearest": {"k": 2}}, {"nearest": {"k": 0, "of": "Rua Nova 2"}},
                                 {"type_is": "shop", "except": ["Beco Escuro 8"]}, {"any_of": []}])
def test_bad_selectors(ctx, sel):
    with pytest.raises(ValueError):
        D.expand_selector(sel, ctx)


def keys(rule, ctx):
    return D.keyset(D.expand_rule(rule, ctx))


def test_rule_expansion(ctx):
    prec = {"type": "precedence", "before": {"type_is": "hospital"},
            "after": {"type_is": "school", "except": ["Rua Nova 2"]}}
    assert keys(prec, ctx) == {("precedence", a, b) for a in (1, 2) for b in (4, 5)}
    forb = {"type": "forbidden_edge", "a": "Rua Azul 3", "b": {"area_is": "Alfama"}}          # the self pair is dropped
    assert keys(forb, ctx) == {("forbidden_edge", frozenset((1, 3))), ("forbidden_edge", frozenset((1, 7)))}
    win = {"type": "time_window", "who": {"type_is": "hospital"}, "t1": 1, "t2": 2.5}
    assert keys(win, ctx) == {("time_window", 1, 1.0, 2.5), ("time_window", 2, 1.0, 2.5)}
    assert keys({"type": "soft_zone", "zone": "North"}, ctx) == {("soft_zone", "north")}
    assert keys({"type": "mandatory_edge", "i": 3, "j": 1}, ctx) == {("mandatory_edge", frozenset((1, 3)))}


def test_validate_keeps_good_items(ctx):
    obj = {"rules": [{"type": "soft_zone", "zone": "east"}, {"type": "soft_zone", "zone": "up"},
                     {"type": "time_window", "who": 1, "t1": 3, "t2": 2}, "junk"]}
    accepted, errors = D.validate(obj, ctx)
    assert len(accepted) == 1 and len(errors) == 3
    assert D.validate([1], ctx) == ([], ["no rules list"])


def test_score_group_or_singles_are_equal(ctx):
    rule = {"type": "precedence", "before": {"type_is": "hospital"}, "after": "Rua Nova 2"}
    label = [{"dsl": rule, "prims": D.expand_rule(rule, ctx)}]
    singles = [p for c in (1, 2) for p in D.expand_rule({"type": "precedence", "before": c, "after": 3}, ctx)]
    assert D.score(label, singles) == {"correct": [True], "exact": True, "wrong": []}
    only_one = singles[:1]
    assert D.score(label, only_one)["correct"] == [False]
    # a wrong extra of the same type touching the rule's customers makes it wrong
    extra = D.expand_rule({"type": "precedence", "before": 7, "after": 3}, ctx)
    assert D.score(label, singles + extra)["correct"] == [False]
    # a wrong extra elsewhere only breaks the instance
    other = D.expand_rule({"type": "precedence", "before": 8, "after": 9}, ctx)
    s = D.score(label, singles + other)
    assert s["correct"] == [True] and not s["exact"]


def test_classify_errors(ctx):
    rule = {"type": "precedence", "before": {"type_is": "hospital"}, "after": {"type_is": "school",
                                                                               "except": ["Rua Nova 2"]}}
    r = {"dsl": rule, "prims": D.expand_rule(rule, ctx)}
    flipped = D.expand_rule({"type": "precedence", "before": {"type_is": "school", "except": ["Rua Nova 2"]},
                             "after": {"type_is": "hospital"}}, ctx)
    assert D.classify_error(r, flipped, ctx) == "inverted"
    lost = D.expand_rule(D.without_except(rule), ctx)
    assert D.classify_error(r, lost, ctx) == "exception lost"
    assert D.classify_error(r, [], ctx) == "missing"
    wrong = D.expand_rule({"type": "precedence", "before": 1, "after": 7}, ctx)
    assert D.classify_error(r, wrong, ctx) == "wrong selector"
    other = D.expand_rule({"type": "forbidden_edge", "a": 1, "b": 7}, ctx)
    assert D.classify_error(r, other, ctx) == "wrong type"


def wrap(*rules):
    return [{"prims": r} for r in rules]


def prim(kind, **f):
    return {"type": kind, "formal": f}


def test_solver_check_cycles():
    coords = np.random.default_rng(0).random((10, 2))
    a = [prim("precedence", before=1, after=2), prim("precedence", before=2, after=3)]
    b = [prim("precedence", before=3, after=1)]
    prims, rejected = D.validate_for_solver(wrap(a, b), coords)
    assert len(prims) == 2 and rejected == [{"rule": 1, "reason": "precedence cycle"}]


def test_solver_check_windows():
    coords = np.zeros((6, 2))
    coords[1] = [1.0, 0.0]
    far = [prim("time_window", customer=1, t1=0.0, t2=0.5)]
    assert D.validate_for_solver(wrap(far), coords)[1][0]["reason"].startswith("impossible window")
    w1, w2 = [prim("time_window", customer=1, t1=2.0, t2=3.0)], [prim("time_window", customer=1, t1=4.0, t2=5.0)]
    assert D.validate_for_solver(wrap(w1, w2), coords)[1][0]["reason"] == "two different windows for one customer"
    p = [prim("precedence", before=2, after=3)]
    prims, rejected = D.validate_for_solver(wrap(w1, p), coords)
    assert len(prims) == 1 and prims[0]["type"] == "time_window" and "precedence" in rejected[0]["reason"]


def test_solver_check_mandatory():
    coords = np.random.default_rng(0).random((10, 2))
    e = lambda i, j: [prim("mandatory_edge", i=i, j=j)]
    assert "more than 2" in D.validate_for_solver(wrap(e(1, 2), e(1, 3), e(1, 4)), coords)[1][0]["reason"]
    assert "sub-tour" in D.validate_for_solver(wrap(e(1, 2), e(2, 3), e(1, 3)), coords)[1][0]["reason"]
    assert D.validate_for_solver(wrap([prim("forbidden_edge", i=1, j=2)], e(1, 2)), coords)[1][0]["reason"] == \
        "an edge both mandatory and forbidden"
    # a full Hamiltonian cycle of 3 mandatory edges is impossible at n = 10 but fine as a path
    p = [prim("precedence", before=7, after=8)]
    assert D.validate_for_solver(wrap(p, e(1, 2), e(2, 3), e(3, 4), e(5, 6)), coords)[1][-1]["rule"] == 4


def test_generated_rules_pass_the_check_and_solver():
    n = 200
    rng = np.random.default_rng(1)
    coords = rng.random((n, 2))
    legend = G.make_legend(n, 0, coords)
    assert len(legend) == G.LEGEND_SIZE[n] and len({p["name"] for p in legend}) == len(legend)
    ctx = D.Context(legend, coords, n)
    for seed in range(20):
        rules = G.sample_rules(np.random.default_rng(seed), legend, coords, n, 10.0, 3)
        assert not D.validate_for_solver(rules, coords)[1]
        for r in rules:
            assert D.keyset(D.expand_rule(r["dsl"], ctx)) == D.keyset(r["prims"])
        R.problem_kind([p for r in rules for p in r["prims"]])         # raises if a window meets a precedence


def test_prompt_never_uses_the_shared_ollama():
    assert llm.HOST.endswith(":11435")
    assert "Rua Azul 3" in D.system_prompt() and "except" in D.SPEC
