"""Quick tests for scripts/planner_probe_v2.py (no LLM needed).  Run with: pytest tests/"""

import importlib.util
import json
import os

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("planner_probe_v2", os.path.join(_dir, "planner_probe_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def test_rule_solves_all_cases():
    for c in v2.CASES:
        assert v2.probe.score(c, json.dumps(v2.rule(c)))["ok"], c["id"]


def test_feasible():
    case = {c["id"]: c for c in v2.CASES}
    assert v2.feasible(case[3]) == [] and v2.feasible(case[4]) == [] and v2.feasible(case[7]) == []
    assert all(a[0] != "lkh_whole" for a in v2.feasible(case[8]))            # unavailable
    assert v2.feasible(case[5])[0][:2] == ("lkh_whole", {"candidates": "NN8", "time_limit": 0.43})
    assert "stop" in v2.report_v3(case[1]) and "QUADRANT5" in v2.report_v3(case[1])


def test_extract_json():
    text = 'Fits: {a lot}.\n{"tool": "x", "args": {}}\nfinal:\n{"tool": "stop", "args": {}, "reason": "r"}'
    assert json.loads(v2.extract_json(text))["tool"] == "stop"
    assert v2.extract_json("no json here") == "no json here"


def test_examples_are_correct_and_not_test_cases():
    tests = {(c["n"], c["budget"], c["used"], tuple(c["unavailable"])) for c in v2.CASES}
    for state, _, answer in v2.EXAMPLES:
        assert (state["n"], state["budget"], state["used"], tuple(state["unavailable"])) not in tests
        r = v2.rule(state)
        assert (r["tool"], r["args"]) == (answer["tool"], answer["args"])
