"""Tests for src/tsp_gnn/hier.py (phase 3, hierarchical TSP).  Run with: pytest tests/"""

import itertools

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, is_valid_tour
from tsp_gnn.hier import (Hierarchy, Heatmaps, balanced_kmeans, beam_path, choose_links, greedy_path,
                          hierarchical_tour, kdtree_partition, moved_between_groups, boundary_stats)


class ShortEdges(torch.nn.Module):
    """Stand-in for the trained model: shorter edges get higher logits."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        return -20 * data.edge_attr.squeeze(-1) + 2 + self.dummy


@pytest.mark.parametrize("m", [70, 200, 537, 1000])
def test_balanced_kmeans_sizes(m):
    pts = generate_instances(1, m, m)[0]
    labels = balanced_kmeans(pts)
    sizes = np.bincount(labels)
    assert len(sizes) == round(m / 20) and sizes.min() >= 10 and sizes.max() <= 30


@pytest.mark.parametrize("m", [70, 200, 1000])
def test_kdtree_partition_sizes(m):
    sizes = np.bincount(kdtree_partition(generate_instances(1, m, 1)[0]))
    assert sizes.sum() == m and sizes.max() <= 30 and sizes.min() >= 15


@pytest.mark.parametrize("method", ["kmeans", "kdtree"])
def test_hierarchy_tree(method):
    coords = generate_instances(1, 2000, 2)[0]
    H = Hierarchy(coords, method)
    assert len(H.points[H.top]) <= 60 and len(H.points[H.top - 1]) > 60
    for l in range(1, H.top + 1):
        assert sorted(np.concatenate(H.children[l]).tolist()) == list(range(len(H.points[l - 1])))
        assert sorted(np.concatenate(H.cities[l]).tolist()) == list(range(2000))
        for g, ch in enumerate(H.children[l]):   # centroid = mean of the children's points
            assert np.allclose(H.points[l][g], H.points[l - 1][ch].mean(axis=0))


def random_heat(m, seed, density=0.5):
    rng = np.random.default_rng(seed)
    heat = rng.random((m, m)) * (rng.random((m, m)) < density)
    heat = np.triu(heat, 1)
    return heat + heat.T


@pytest.mark.parametrize("decoder", ["greedy", "beam"])
def test_paths_are_valid_with_fixed_ends(decoder):
    for m, seed in itertools.product([1, 2, 3, 4, 7, 20, 30], range(4)):
        rng = np.random.default_rng(seed)
        pts = rng.random((m, 2))
        s, t = (0, 0) if m == 1 else tuple(rng.choice(m, 2, replace=False).tolist())
        for density in (0.0, 0.3, 1.0):
            heat = random_heat(m, seed, density)
            path = greedy_path(heat, pts, s, t) if decoder == "greedy" else beam_path(heat, pts, s, t, 10)
            assert sorted(path.tolist()) == list(range(m))
            assert path[0] == s and path[-1] == t


def test_beam_path_width_one_is_greedy():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pts = rng.random((20, 2))
        heat = random_heat(20, seed)
        g = greedy_path(heat, pts, 3, 11)
        assert np.array_equal(beam_path(heat, pts, 3, 11, 1), g)


def test_greedy_path_follows_a_perfect_heatmap():
    pts = np.random.default_rng(0).random((12, 2))
    true = np.random.default_rng(1).permutation(12)
    heat = np.zeros((12, 12))
    for a, b in zip(true[:-1], true[1:]):
        heat[a, b] = heat[b, a] = 0.9
    assert np.array_equal(greedy_path(heat, pts, true[0], true[-1]), true)


def test_choose_links_constraint_and_optimum():
    rng = np.random.default_rng(0)
    for trial in range(20):
        M = int(rng.integers(2, 6))
        sizes = rng.integers(1, 4, size=M).tolist()
        attrs = [rng.random((sizes[k], sizes[(k + 1) % M])) for k in range(M)]
        dists = [np.zeros_like(a) for a in attrs]
        links, _ = choose_links(attrs, dists, sizes, c=100)
        for k in range(M):   # group k: entered by link k-1, left by link k
            assert sizes[k] < 2 or links[k - 1][1] != links[k][0]
        # brute force over every choice
        best = -1
        for pick in itertools.product(*[itertools.product(range(a.shape[0]), range(a.shape[1])) for a in attrs]):
            if all(sizes[k] < 2 or pick[k - 1][1] != pick[k][0] for k in range(M)):
                best = max(best, sum(attrs[k][pick[k]] for k in range(M)))
        assert np.isclose(sum(attrs[k][links[k]] for k in range(M)), best)
        links1, _ = choose_links(attrs, dists, sizes, c=1)
        assert all(sizes[k] < 2 or links1[k - 1][1] != links1[k][0] for k in range(M))


@pytest.mark.parametrize("method,attract,cands,decoder", [("kmeans", "distance", 1, "greedy"),
                                                          ("kmeans", "heatmap", 3, "beam"),
                                                          ("kdtree", "heatmap", 1, "greedy"),
                                                          ("kdtree", "distance", 3, "beam")])
def test_hierarchical_tour_is_valid(method, attract, cands, decoder):
    coords = generate_instances(1, 1500, 5)[0]
    H = Hierarchy(coords, method)
    assert H.top == 2
    heatmaps = Heatmaps(ShortEdges(), torch.device("cpu"))
    tour, times, stats = hierarchical_tour(H, heatmaps, attract, cands, decoder)
    assert is_valid_tour(tour, 1500)
    assert times["model"] > 0 and stats["c_used_max"] >= cands
    # each level-0 group is one contiguous stretch of the tour: as many boundary edges as groups
    s = boundary_stats(H, tour, tour)
    assert s["boundary_edges"] == len(H.points[1]) and s["boundary_wrong"] == s["interior_wrong"] == 0
    assert moved_between_groups(H, tour, tour) == 0
