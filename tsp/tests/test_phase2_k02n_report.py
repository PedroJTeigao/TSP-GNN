"""Quick tests for scripts/phase2_k02n_report.py.  Run with: pytest tests/"""

import importlib.util
import os

import numpy as np

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("phase2_k02n_report", os.path.join(_dir, "phase2_k02n_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_k_of():
    assert [report.k_of(m, 1000) for m in ("k10", "k02n", "khalf100", "khalf")] == [10, 200, 100, 500]
    assert report.k_of("k02n", 500) == report.k_of("khalf100", 500) == 100


def test_paired_verdicts():
    rng = np.random.default_rng(0)
    base = [rng.random(40) for _ in range(3)]
    assert report.paired(base, base, 0.01)["verdict"] == "identical"
    assert report.paired([b - 1 for b in base], base, 0.01)["verdict"] == "better"
    assert report.paired([b + 1 for b in base], base, 0.01)["verdict"] == "worse"


def test_headline():
    t = lambda v: {"verdict": v}
    tests = {(50, "gcn", "greedy+2opt"): t("better"), (50, "gcn", "beam10+2opt"): t("better"),
             (50, "gcn", "greedy"): t("worse"),                       # without 2-opt: not in the headline
             (500, "gcn", "greedy+2opt"): t("identical"), (1000, "gcn", "greedy+2opt"): t("n.s.")}
    assert report.headline(tests) == "n = 50: better; n = 500: same graph; n = 1000: equal"
