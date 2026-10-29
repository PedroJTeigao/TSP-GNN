"""Tests for the JSON schema and scoring of the formalisation arms (tsp_gnn.formalize) and for the injected errors
and claim matching of the LLM verifier (tsp_gnn.llm_verify). The injection test needs the LKH-3 binary.
Run with: pytest tests/"""

import numpy as np
import pytest

from tsp_gnn import formalize as F
from tsp_gnn import llm_verify as V
from tsp_gnn import rules as R


def test_validate_schema():
    obj = {"rules": [{"type": "precedence", "before": 3, "after": "9"},
                     {"type": "forbidden_edge", "i": 4, "j": 5.0},
                     {"type": "time_window", "customer": 7, "t1": 1.5, "t2": 2.25},
                     {"type": "soft_zone", "zone": "North"},
                     {"type": "mandatory_edge", "i": 2, "j": 2},             # same customer twice
                     {"type": "time_window", "customer": 8, "t1": 3, "t2": 1},  # t1 > t2
                     {"type": "precedence", "before": 3, "after": 99},     # out of range for n = 50
                     {"type": "speed_limit", "value": 3},                    # unknown type
                     {"type": "soft_zone", "zone": "downtown"},
                     {"type": "forbidden_edge", "i": 4},                     # missing field
                     {"type": "forbidden_edge", "i": 5, "j": 4},             # duplicate (edges are unordered)
                     "not an object"]}
    rules, errors = F.validate(obj, 50)
    assert [r["type"] for r in rules] == ["precedence", "forbidden_edge", "time_window", "soft_zone"]
    assert rules[0]["formal"] == {"before": 3, "after": 9}
    assert rules[3]["formal"]["rect"] == list(R.ZONES["north"])
    assert len(errors) == 7
    assert F.validate({"rules": [{"type": "time_window", "customer": 7, "t1": 0, "t2": "inf"}]}, 50) == \
        ([], ["ValueError: t1 or t2 not finite"])
    assert F.validate({"answer": []}, 50) == ([], ["no rules list"])
    assert F.validate(None, 50) == ([], ["no rules list"])


def test_score():
    lab = [{"type": "precedence", "formal": {"before": 3, "after": 9}},
           {"type": "mandatory_edge", "formal": {"i": 4, "j": 5}},
           {"type": "soft_zone", "formal": {"zone": "east"}}]
    s = F.score(lab, [{"type": "precedence", "formal": {"before": 3, "after": 9}},
                      {"type": "mandatory_edge", "formal": {"i": 5, "j": 4}},
                      {"type": "soft_zone", "formal": {"zone": "east"}}])
    assert s["exact"] and s["correct"] == [True, True, True]
    s = F.score(lab, [{"type": "precedence", "formal": {"before": 9, "after": 3}},     # inverted
                      {"type": "forbidden_edge", "formal": {"i": 4, "j": 5}},        # wrong type
                      {"type": "time_window", "formal": {"customer": 30, "t1": 0, "t2": 1}}])  # spurious
    assert not s["exact"] and s["correct"] == [False, False, False]
    assert s["pairs"] == [("precedence", "precedence"), ("mandatory_edge", "forbidden_edge"),
                          ("soft_zone", "missing")]
    assert s["spurious"] == ["time_window"]
    # a near match must not steal an exact match of another label rule
    lab2 = [{"type": "precedence", "formal": {"before": 3, "after": 9}},
            {"type": "forbidden_edge", "formal": {"i": 3, "j": 9}}]
    s = F.score(lab2, [{"type": "forbidden_edge", "formal": {"i": 3, "j": 9}}])
    assert s["correct"] == [False, True] and s["pairs"][0] == ("precedence", "missing")


def test_few_shot_examples_are_valid():
    for text, answer in F.FEW_SHOT:
        rules, errors = F.validate(answer, 60)
        assert not errors and len(rules) == len(answer["rules"])
        assert F.parse_regex(text) == []                 # not written in the generator's formulations
    assert "Worked examples" in F.system_prompt("b2") and "Worked examples" not in F.system_prompt("b1")


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    n = 40
    rng = np.random.default_rng(3)
    coords = rng.random((n, 2))
    wd = str(tmp_path_factory.mktemp("lkh"))
    pure = R.solve(coords, [], wd + "/pure")
    out = {}
    for kinds in (["precedence", "precedence", "forbidden_edge", "mandatory_edge"],
                  ["time_window", "time_window", "mandatory_edge", "soft_zone"]):
        taken = set()
        rules = [R.sample_rule(k, rng, coords, pure["tour"], pure["length"], taken) for k in kinds]
        res = R.solve(coords, rules, wd + "/" + "_".join(kinds), initial_tour=pure["tour"])
        assert res["feasible"]
        out[tuple(kinds)] = (coords, rules, res["tour"])
    return out


def test_injected_errors_break_exactly_one_rule(solved):
    rng = np.random.default_rng(0)
    seen = set()
    for coords, rules, tour in solved.values():
        assert R.violated_rules(rules, coords, tour) == []
        for kind in sorted({r["type"] for r in rules} & set(V.INJECTABLE)):
            for _ in range(5):
                inj = V.inject_error(rng, coords, tour, rules, kind)
                assert inj is not None, kind
                t, k = inj
                assert sorted(t.tolist()) == list(range(len(coords))) and t[0] == 0
                assert rules[k]["type"] == kind and R.violated_rules(rules, coords, t) == [k]
                assert not R.check_all(rules, coords, t)
                seen.add(kind)
    assert seen == set(V.INJECTABLE)


def test_route_text_and_claims(solved):
    coords, rules, tour = solved[("precedence", "precedence", "forbidden_edge", "mandatory_edge")]
    text = V.route_text(coords, tour, rules)
    assert text.count("\n") == len(coords) and text.startswith("0. depot (customer 0), t = 0.00")
    report = V.code_report(rules, coords, tour)
    assert report.count("satisfied") == 4 and "VIOLATED" not in report
    p = rules[0]["formal"]
    obj = {"violations": [{"type": "precedence", "customers": [p["after"], p["before"]]},
                          {"type": "time_window", "customers": [1]}]}
    assert V.match_claims(obj, rules) == ({0}, 1)
    m = rules[3]["formal"]                                      # right customers, wrong type name
    assert V.match_claims({"violations": [{"type": "forbidden_edge", "customers": [m["i"], m["j"]]}]},
                          rules) == ({3}, 0)
    assert V.match_claims({"violations": []}, rules) == (set(), 0)
    assert V.match_claims({"other": 1}, rules) == (None, 0)


def test_keyword_parser_b0plus():
    from tsp_gnn import keyword_parser as K
    for kind, formal in {"precedence": dict(before=12, after=7), "forbidden_edge": dict(i=12, j=7),
                         "mandatory_edge": dict(i=12, j=7), "time_window": dict(customer=12, t1=1.5, t2=2.25),
                         "soft_zone": dict(zone="central", rect=list(R.ZONES["central"]), weight=1.0)}.items():
        for v in range(len(R.TEMPLATES[kind])):                   # every generator formulation
            rule = R.make_rule(kind, v, **formal)
            got = K.parse_keywords(R.instance_text([rule]), 50)
            assert [F.rule_key(g) for g in got] == [F.rule_key(rule)], rule["text"]
    text = (R.PREAMBLE + "\nNamed places: customer 17 is the hospital; customer 4 is Rosa's café.\n"
            "- Customer 9 has a dog, so ring the bell.\n"
            "- Visit customer 30 only after the hospital has been served.\n"
            "- Never go straight between Rosa's café and stop 22, and keep driving in the north and west zones low.\n"
            "- The hospital and customer 5 must be visited one right after the other.\n"
            "- Service at customer 8 must start between 2.50 and 3.75, and customer 41 accepts it from 0.40 to 1.10.")
    got = {F.rule_key(r) for r in K.parse_keywords(text, 50)}
    assert got == {("precedence", 17, 30), ("forbidden_edge", frozenset((4, 22))), ("soft_zone", "north"),
                   ("soft_zone", "west"), ("mandatory_edge", frozenset((17, 5))),
                   ("time_window", 8, 2.5, 3.75), ("time_window", 41, 0.4, 1.1)}
