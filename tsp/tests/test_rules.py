"""Tests for tsp_gnn.rules: every rule is checkable in code (the LKH-3 reference meets it, a route that breaks it is
caught).  Needs the LKH-3 binary (scripts/build_lkh3.sh).  Run with: pytest tests/"""

import numpy as np
import pytest

from tsp_gnn import rules as R

N = 30


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    rng = np.random.default_rng(7)
    coords = rng.random((N, 2))
    wd = str(tmp_path_factory.mktemp("lkh"))
    pure = R.solve(coords, [], wd + "/pure")
    assert pure["feasible"] and pure["kind"] == "TSP"
    return coords, pure, wd


def solved(inst, kinds, seed=0):
    coords, pure, wd = inst
    rng = np.random.default_rng(seed)
    taken = set()
    rules = [R.sample_rule(k, rng, coords, pure["tour"], pure["length"], taken) for k in kinds]
    res = R.solve(coords, rules, wd + "/" + "_".join(kinds))
    assert res["feasible"], res["log"][-500:]
    assert R.check_all(rules, coords, res["tour"])
    return coords, pure, rules, res


def test_inside_length_by_hand():
    north, central = R.ZONES["north"], R.ZONES["central"]
    assert R.inside_length([0, 0.5], [1, 0.5], north) == pytest.approx(0.0)
    assert R.inside_length([0, 0.8], [1, 0.8], north) == pytest.approx(1.0)
    assert R.inside_length([0, 0.5], [1, 0.5], central) == pytest.approx(0.25)
    assert R.inside_length([0.5, 0.0], [0.5, 1.0], north) == pytest.approx(0.25)
    assert R.inside_length([0.5, 0.9], [0.5, 0.9], north) == pytest.approx(0.0)        # zero-length segment
    p = np.array([[0.0, 0.0], [0.0, 0.0]]); q = np.array([[1.0, 1.0], [0.2, 0.2]])
    assert R.inside_length(p, q, central) == pytest.approx([np.sqrt(2) * 0.25, 0.0])


def test_precedence(inst):
    coords, pure, rules, res = solved(inst, ["precedence", "precedence"])
    assert res["kind"] == "SOP"
    t = res["tour"].copy()
    f = rules[0]["formal"]
    a, b = np.where(t == f["before"])[0][0], np.where(t == f["after"])[0][0]
    t[a], t[b] = t[b], t[a]                                     # b now comes before a
    assert not R.check_rule(rules[0], coords, t)[0] and not R.check_all(rules, coords, t)


def test_forbidden_edge(inst):
    coords, pure, rules, res = solved(inst, ["forbidden_edge"])
    f = rules[0]["formal"]
    assert not R.check_rule(rules[0], coords, pure["tour"])[0]  # made from an edge of the pure route
    t = [v for v in res["tour"].tolist() if v != f["j"]]
    t.insert(t.index(f["i"]) + 1, f["j"])                      # put j right after i
    assert not R.check_all(rules, coords, np.array(t))


def test_mandatory_edge(inst):
    coords, pure, rules, res = solved(inst, ["mandatory_edge"])
    assert not R.check_rule(rules[0], coords, pure["tour"])[0]  # the pure route does not join them
    assert R.check_rule(rules[0], coords, res["tour"])[0]
    assert res["length"] >= pure["length"] - 1e-6


def test_mandatory_edge_in_sop_and_tsptw(inst):
    for kinds in (["precedence", "mandatory_edge"], ["time_window", "mandatory_edge"]):
        coords, pure, rules, res = solved(inst, kinds, seed=3)
        assert res["lkh_runs"] == 2                             # both directions of the fixed edge were tried


def test_time_window(inst):
    coords, pure, rules, res = solved(inst, ["time_window"])
    assert res["kind"] == "TSPTW"
    rev = np.concatenate([pure["tour"][:1], pure["tour"][1:][::-1]])
    for t in (pure["tour"], rev):                               # the window closes before the pure route arrives
        assert not R.check_rule(rules[0], coords, t)[0]
    k = rules[0]["formal"]["customer"]
    t = R.service_times(coords, res["tour"], rules)[np.where(res["tour"] == k)[0][0]]
    assert rules[0]["formal"]["t1"] - 1e-9 <= t <= rules[0]["formal"]["t2"] + 1e-9


def test_soft_zone(inst):
    coords, pure, rules, res = solved(inst, ["soft_zone"])
    rect = rules[0]["formal"]["rect"]
    # Soft: never "violated", but ignoring it costs more than the reference objective.
    assert R.objective(coords, pure["tour"], rules) >= res["objective"] - 1e-3
    assert R.zone_length(coords, res["tour"], rect) <= R.zone_length(coords, pure["tour"], rect) + 1e-9
    assert res["objective"] == pytest.approx(res["length"] + R.zone_length(coords, res["tour"], rect))


def test_invalid_routes_rejected(inst):
    coords, pure, _ = inst
    assert not R.check_all([], coords, pure["tour"][1:])                  # a customer missing
    assert not R.check_all([], coords, np.roll(pure["tour"], 1))          # does not start at the depot


def test_text_and_types():
    rng = np.random.default_rng(0)
    for _ in range(500):
        types = R.sample_types(rng, 3)
        assert not ("time_window" in types and "precedence" in types)
    with pytest.raises(ValueError):
        R.problem_kind([{"type": "time_window"}, {"type": "precedence"}])
    examples = {"precedence": dict(before=3, after=7), "forbidden_edge": dict(i=3, j=7),
                "mandatory_edge": dict(i=3, j=7), "time_window": dict(customer=3, t1=1.5, t2=2.25),
                "soft_zone": dict(zone="north", rect=list(R.ZONES["north"]), weight=1.0)}
    for kind, formal in examples.items():
        assert 3 <= len(R.TEMPLATES[kind]) <= 4
        texts = {R.make_rule(kind, v, **formal)["text"] for v in range(len(R.TEMPLATES[kind]))}
        assert len(texts) == len(R.TEMPLATES[kind])
        for t in texts:
            assert ("north" in t) if kind == "soft_zone" else ("3" in t and ("7" in t or "1.50" in t))


def test_initial_tour(inst):
    coords, pure, wd = inst
    rng = np.random.default_rng(1)
    rule = R.sample_rule("precedence", rng, coords, pure["tour"], pure["length"], set())
    res = R.solve(coords, [rule], wd + "/init", initial_tour=pure["tour"])
    assert res["feasible"]
    assert res["length"] == pytest.approx(pure["length"], rel=1e-6)   # one precedence: the reversed route is free
    with open(wd + "/init/initial.tour") as f:
        nodes = f.read().split("TOUR_SECTION")[1].split()
    assert nodes[0] == "1" and nodes[N] == str(N + 1)                 # SOP: depot first, its copy last
