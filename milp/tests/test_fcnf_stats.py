"""Tests for fcnf_gnn.stats (paired comparisons).  Run with: pytest tests/"""

import numpy as np

from fcnf_gnn.stats import paired, paired_compare


def test_paired_directions():
    rng = np.random.default_rng(0)
    ref = rng.random(200)
    better = [ref - 0.1 - 0.01 * rng.random(200) for _ in range(3)]
    res = paired(better, [ref] * 3, threshold=0.01)
    assert res["verdict"] == "better" and res["seeds"] == 3 and res["better"] == 100
    assert paired([ref + 0.1] * 3, [ref] * 3, 0.01)["verdict"] == "worse"
    mixed = [ref - 0.1, ref + 0.1, ref - 0.1]                                  # seeds disagree
    assert paired(mixed, [ref] * 3, 0.01)["verdict"] == "n.s."


def test_all_ties():
    a = np.ones(50)
    r = paired_compare(a, a)
    assert r["p"] == 1.0 and r["equal"] == 100
