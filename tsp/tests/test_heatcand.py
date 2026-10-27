"""Tests of the heatmap candidate edges for LKH (heatcand.py)."""

import numpy as np

from tsp_gnn.heatcand import top_candidates, write_candidates


def test_top_candidates_order_and_alphas():
    # 4 cities, undirected edges with probabilities; city 0 has edges to 1 (0.9), 2 (0.2), 3 (0.5).
    pi = np.array([0, 0, 0, 1, 2])
    pj = np.array([1, 2, 3, 2, 3])
    prob = np.array([0.9, 0.2, 0.5, 0.7, 0.4])
    cand, alpha = top_candidates(pi, pj, prob, 4, 2)
    assert cand[0].tolist() == [1, 3]                 # highest p first
    assert alpha[0].tolist() == [100, 500]            # round(1000 (1 - p))
    assert cand[1].tolist() == [0, 2]                 # edges seen from both ends
    assert cand[2].tolist() == [1, 3]
    assert cand[3].tolist() == [0, 2]


def test_write_candidates_format(tmp_path):
    cand = np.array([[1, 2], [0, 2], [0, 1]])
    alpha = np.array([[0, 10], [0, 20], [5, 30]])
    path = tmp_path / "c.cand"
    write_candidates(path, cand, alpha)
    lines = path.read_text().splitlines()
    assert lines[0] == "3"
    assert lines[1] == "1 0 2 2 0 3 10"               # city, dad 0, count, then (city, alpha) 1-indexed
    assert lines[-2:] == ["-1", "EOF"]


def test_hybrid_candidates_union_by_distance():
    from tsp_gnn.heatcand import hybrid_candidates
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 0.5]])
    heat = np.array([[3, 1], [0, 3], [0, 4], [1, 0], [0, 2]])
    nn = np.array([[4, 1], [0, 4], [4, 0], [1, 0], [0, 1]])
    cands, alphas = hybrid_candidates(coords, heat, nn)
    assert cands[0].tolist() == [4, 1, 3]              # union {1, 3, 4}, nearest first: 0.5, 1, 3
    assert alphas[0].tolist() == [0, 1, 2]
    assert sorted(cands[2].tolist()) == [0, 4]         # duplicates (4, 0 in both) appear once


def test_write_candidates_variable_counts(tmp_path):
    path = tmp_path / "v.cand"
    write_candidates(path, [np.array([1]), np.array([0, 2]), np.array([1])],
                     [np.array([0]), np.array([0, 1]), np.array([0])])
    lines = path.read_text().splitlines()
    assert lines[1] == "1 0 1 2 0"
    assert lines[2] == "2 0 2 1 0 3 1"
