"""Quick tests for scripts/planner_probe.py (no LLM needed).  Run with: pytest tests/"""

import importlib.util
import json
import os

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("planner_probe", os.path.join(_dir, "planner_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

case = {c["id"]: c for c in probe.CASES}
answer = lambda tool, args: json.dumps({"tool": tool, "args": args, "reason": "r"})


def test_parse_errors():
    assert probe.parse("not json")[3] == "not JSON"
    assert probe.parse('{"tool": "stop"}')[3] == "missing keys"
    assert probe.parse(answer("run_everything", {}))[3] is not None
    assert probe.parse(answer("stop", {}))[3] is None


def test_checks():
    assert probe.score(case[1], answer("lkh_whole", {"candidates": "QUADRANT5", "time_limit": 5}))["ok"]
    assert not probe.score(case[1], answer("lkh_whole", {"candidates": "QUADRANT5", "time_limit": 6}))["ok"]  # over
    assert not probe.score(case[1], answer("lkh_whole", {"candidates": "ALPHA", "time_limit": 5}))["ok"]
    assert probe.score(case[3], answer("stop", {}))["ok"]
    assert probe.score(case[7], answer("triage", {"frac": 20}))["ok"]
    assert probe.score(case[8], answer("windows_model", {"decoder": "beam10"}))["ok"]
    assert not probe.score(case[8], answer("lkh_whole", {"candidates": "QUADRANT5", "time_limit": 12}))["ok"]


def test_reports():
    for c in probe.CASES:
        text = probe.report_text(c)
        assert f"n = {c['n']}" in text and "stop" in text
    assert "[unavailable]" in probe.report_text(case[8])
    assert "none yet" in probe.report_text(case[3])
