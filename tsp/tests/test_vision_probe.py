"""Tests for the vision probe helpers (tsp_gnn.vision_probe). Run with: pytest tests/"""

import numpy as np

from tsp_gnn import vision_probe as V
from tsp_gnn.window import initial_tour, window_starts


def test_cell_numbering():
    pts = np.array([[0.1, 0.9], [0.9, 0.9], [0.1, 0.1], [0.9, 0.1], [0.3, 0.8], [1.0, 0.0], [0.0, 1.0]])
    assert V.cell_of(pts).tolist() == [1, 4, 13, 16, 2, 16, 1]       # row by row from the top left


def test_parse_and_scores():
    assert V.parse_ranking({"cells": [3, "7", 3, 17, "x", 0, 16]}) == [3, 7, 16]
    assert V.parse_ranking({"cells": []}) is None and V.parse_ranking([1, 2]) is None
    s = V.ranking_scores([5, 2, 9], np.random.default_rng(0))
    assert V.top_cells(s, 3) == [5, 2, 9]
    assert len(set(V.top_cells(s, 4))) == 4


def test_cell_stats():
    cells = np.array([1, 1, 2, 16])
    st = V.cell_stats(cells, np.array([True, False, True, False]), np.array([1.0, 0.0, 2.0, 0.0]),
                      np.array([0.2, 0.4, 0.1, 0.3]))
    assert st["windows"][[0, 1, 15]].tolist() == [2, 1, 1] and st["windows"].sum() == 4
    assert st["label"][0] == 0.5 and st["gain"][1] == 2.0 and abs(st["disagreement"][0] - 0.3) < 1e-12
    assert np.isnan(st["label"][2])


def test_cell_refine_only_chosen_cells():
    rng = np.random.default_rng(0)
    coords = rng.random((300, 2))
    tour = initial_tour(coords, 20)
    t, st = V.cell_refine(coords, tour, 20, [], max_passes=1)
    assert st["to_lkh"] == 0 and np.array_equal(t, tour)
    t, st = V.cell_refine(coords, tour, 20, range(1, 17), max_passes=1)
    assert st["to_lkh"] == st["windows"] == sum(len(window_starts(300, 20, o)) for o in (0, 10))
    assert sorted(t.tolist()) == list(range(300))
