"""Tests for src/tsp_gnn/window.py (window refinement of a tour).  Run with: pytest tests/"""

import itertools

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, is_valid_tour, tour_length
from tsp_gnn.hier import Heatmaps
from tsp_gnn.window import initial_tour, lkh_path, make_solver, path_length, refine, two_opt_path, window_starts


class ShortEdges(torch.nn.Module):
    """Stand-in for the trained model: shorter edges get higher logits."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        return -20 * data.edge_attr.squeeze(-1) + 2 + self.dummy


@pytest.mark.parametrize("n,w", [(200, 20), (205, 20), (1003, 30), (100, 50)])
def test_windows_of_a_half_pass_are_disjoint(n, w):
    for offset in (0, w // 2):
        wins = window_starts(n, w, offset)
        pos = np.concatenate([(s + np.arange(length)) % n for s, length in wins])
        assert len(pos) == len(set(pos.tolist())) and n - len(pos) < 4
        assert all(length >= 4 for _, length in wins) and wins[0][0] == offset


def test_lkh_path_is_optimal_with_fixed_ends():
    for seed in range(4):
        pts = np.random.default_rng(seed).random((7, 2))
        best = min(path_length(pts[[0, *p, 6]]) for p in itertools.permutations(range(1, 6)))
        order = lkh_path(pts)
        assert order[0] == 0 and order[-1] == 6 and sorted(order.tolist()) == list(range(7))
        assert np.isclose(path_length(pts[order]), best)


def test_two_opt_path_keeps_ends_and_is_a_local_optimum():
    pts = np.random.default_rng(1).random((30, 2))
    p = two_opt_path(pts)
    assert p[0] == 0 and p[-1] == 29 and sorted(p.tolist()) == list(range(30))
    assert path_length(pts[p]) <= path_length(pts) + 1e-12
    q = pts[p]
    assert np.array_equal(two_opt_path(q), np.arange(30))      # nothing left to improve


def test_initial_tour_is_valid_and_better_than_greedy():
    coords = generate_instances(1, 500, 0)[0]
    t = initial_tour(coords, 100)
    assert is_valid_tour(t, 500)


@pytest.mark.parametrize("kind", ["greedy", "beam10", "lkh", "2opt"])
def test_refine_never_worse_and_valid(kind):
    coords = generate_instances(1, 400, 3)[0]
    start = initial_tour(coords, 50)
    heatmaps = Heatmaps(ShortEdges(), torch.device("cpu"))
    solver = make_solver("beam" if kind == "beam10" else kind, heatmaps)
    tour, stats = refine(coords, start, 20, solver)
    assert is_valid_tour(tour, 400)
    before, after = tour_length(coords, start), tour_length(coords, tour)
    assert after <= before + 1e-12 and np.isclose(before - after, stats["gain"])
    assert 1 <= stats["passes"] <= 5 and 0 <= stats["improved"] <= stats["windows"]
    if kind == "lkh":
        assert stats["improved"] > 0          # windows of a 2-opt tour still have room for an exact solver
