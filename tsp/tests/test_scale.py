"""Tests for src/tsp_gnn/scale.py: the sparse code must match the dense phase 1 code.  Run with: pytest tests/"""

import numpy as np
import pytest
import torch

from tsp_gnn.baselines import greedy_distance_tour
from tsp_gnn.data import generate_instances, is_valid_tour, tour_length
from tsp_gnn.decode import beam_search_decode, edge_heatmap, greedy_decode, two_opt
from tsp_gnn.graph import knn_edges
from tsp_gnn.scale import (beam_search_sparse, edge_pairs, greedy_sparse, knn_graph, neighbor_lists,
                           two_opt_neighbors)


def random_case(seed, n, k, peaked=False):
    """Instance, k-NN graph and random edge probabilities (peaked: a few very likely edges)."""
    coords = generate_instances(1, n, seed)[0]
    g = knn_graph(coords, k)
    rng = np.random.default_rng(seed)
    p = rng.random(g.num_edges) ** (8 if peaked else 1)
    return coords, g, torch.tensor(p, dtype=torch.float32)


@pytest.mark.parametrize("n,k", [(20, 10), (50, 25), (60, 3), (100, 50)])
def test_knn_graph_matches_dense_knn_edges(n, k):
    coords = generate_instances(1, n, 7)[0]
    pos = torch.as_tensor(coords, dtype=torch.float32)
    dense = {tuple(e) for e in knn_edges(pos, k).t().tolist()}
    sparse = knn_graph(coords, k)
    assert {tuple(e) for e in sparse.edge_index.t().tolist()} == dense
    s, d = sparse.edge_index
    assert torch.allclose(sparse.edge_attr.squeeze(1), (pos[s] - pos[d]).norm(dim=1))


def test_edge_pairs_match_heatmap():
    coords, g, p = random_case(0, 30, 6)
    pi, pj, prob = edge_pairs(g.edge_index, p, 30)
    heat, in_graph = edge_heatmap(g.edge_index, p, 30)
    assert np.all(pi < pj) and np.all(in_graph[pi, pj]) and len(pi) == in_graph.sum() // 2
    assert np.array_equal(prob, heat[pi, pj])


@pytest.mark.parametrize("n,k", [(20, 10), (50, 25), (50, 3), (100, 10), (120, 50)])
def test_greedy_sparse_equals_dense(n, k):
    for seed in range(4):
        coords, g, p = random_case(seed, n, k, peaked=seed % 2 == 1)
        heat, in_graph = edge_heatmap(g.edge_index, p, n)
        assert np.array_equal(greedy_sparse(*edge_pairs(g.edge_index, p, n), coords),
                              greedy_decode(heat, in_graph, coords))


@pytest.mark.parametrize("n", [5, 20, 100, 300])
def test_greedy_edge_baseline_equals_dense(n):
    for seed in range(3):
        coords = generate_instances(1, n, seed)[0]
        empty = np.zeros(0, dtype=np.int64)
        assert np.array_equal(greedy_sparse(empty, empty, np.zeros(0), coords), greedy_distance_tour(coords))


@pytest.mark.parametrize("n,k,width", [(20, 10, 1), (20, 10, 10), (30, 4, 5), (50, 25, 10), (50, 3, 10),
                                       (40, 20, 100)])
def test_beam_sparse_equals_dense(n, k, width):
    for seed in range(3):
        coords, g, p = random_case(seed, n, k, peaked=seed == 1)
        heat, in_graph = edge_heatmap(g.edge_index, p, n)
        pairs = edge_pairs(g.edge_index, p, n)
        sparse = beam_search_sparse(*pairs, coords, width)
        assert np.array_equal(sparse, beam_search_decode(heat, in_graph, coords, width))
        if width == 1:
            assert np.array_equal(sparse, greedy_sparse(*pairs, coords))


@pytest.mark.parametrize("n,k", [(20, 19), (50, 49), (200, 10), (1000, 20)])
def test_two_opt_neighbors_is_a_local_optimum(n, k):
    coords = generate_instances(1, n, 3)[0]
    empty = np.zeros(0, dtype=np.int64)
    start = greedy_sparse(empty, empty, np.zeros(0), coords)
    nbrs = neighbor_lists(coords, k)
    tour, finished = two_opt_neighbors(coords, start, nbrs)
    assert finished and is_valid_tour(tour, n) and tour[0] == 0
    length = tour_length(coords, tour)
    assert length <= tour_length(coords, start) + 1e-12
    # No improving move is left among the neighbour lists (both orientations of the tour).
    nxt, prv = np.roll(tour, -1), np.roll(tour, 1)
    pos = np.empty(n, dtype=int); pos[tour] = np.arange(n)
    d = lambda a, b: np.linalg.norm(coords[a] - coords[b], axis=-1)
    for a in range(n):
        c = nbrs[a]
        sa, sc = nxt[pos[a]], nxt[pos[c]]
        pa, pc = prv[pos[a]], prv[pos[c]]
        assert np.all(d(a, sa) + d(c, sc) - d(a, c) - d(sa, sc) <= 1e-9)
        assert np.all(d(a, pa) + d(c, pc) - d(a, c) - d(pa, pc) <= 1e-9)
    if k == n - 1:
        # With complete lists it is a full 2-opt local optimum: decode.two_opt finds nothing to improve.
        assert np.isclose(tour_length(coords, two_opt(coords, tour)), length)


def test_two_opt_neighbors_time_limit_gives_valid_tour():
    coords = generate_instances(1, 2000, 4)[0]
    empty = np.zeros(0, dtype=np.int64)
    start = np.random.default_rng(0).permutation(2000)
    tour, finished = two_opt_neighbors(coords, start, neighbor_lists(coords, 10), time_limit=0.0)
    assert not finished and is_valid_tour(tour, 2000) and tour[0] == 0
