"""Quick tests for the statistics in scripts/summarize.py.  Run with: pytest tests/"""

import importlib.util
import os

import numpy as np

# scripts/ is not a package: we load the file directly.
_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize.py")
_spec = importlib.util.spec_from_file_location("summarize", _path)
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


def test_paired_counts_by_hand():
    base = np.array([1.0, 1.0, 1.0, 1.0])
    model = np.array([0.5, 1.0, 2.0, 0.0])          # better, equal, worse, better
    res = summarize.paired_compare(model, base)
    assert (res["better"], res["equal"], res["worse"]) == (50.0, 25.0, 25.0)
    assert np.isclose(res["mean_diff"], (-0.5 + 0 + 1 - 1) / 4)


def test_wilcoxon_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 5, size=50)
    model = base - rng.uniform(0.1, 0.5, size=50)    # always better
    res = summarize.paired_compare(model, base)
    assert res["better"] == 100.0 and res["p"] < 1e-6


def test_all_ties_gives_p_one():
    gaps = np.array([0.0, 1.5, 2.0])
    res = summarize.paired_compare(gaps, gaps.copy())
    assert res["equal"] == 100.0 and res["p"] == 1.0


def test_mean_std_formatting():
    assert summarize.mean_std([1.0]) == "1.00"
    assert summarize.mean_std([1.0, 3.0]) == "2.00 ± 1.41"
