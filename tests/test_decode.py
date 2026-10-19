"""Quick tests for src/tsp_gnn/decode.py.  Run with: pytest tests/"""

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, is_valid_tour, solve_lkh, tour_length
from tsp_gnn.decode import (beam_search_decode, decode, edge_heatmap, greedy_decode, optimality_gap,
                            tour_edges, two_opt, verify_tour)
from tsp_gnn.graph import instance_to_data

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def make_graph(seed, n=20):
    coords = generate_instances(num=1, n=n, seed=seed)[0]
    tour = solve_lkh(coords)
    return coords, tour, instance_to_data(coords, tour, tour_length(coords, tour))


def test_heatmap_is_symmetric_average():
    edge_index = torch.tensor([[0, 1], [1, 0]])        # edge 0-1 in both directions
    heat, in_graph = edge_heatmap(edge_index, torch.tensor([0.2, 0.6]), 3)
    assert np.isclose(heat[0, 1], 0.4) and np.isclose(heat[1, 0], 0.4)
    assert in_graph[0, 1] and in_graph[1, 0] and not in_graph[0, 2]


def test_greedy_with_true_labels_recovers_lkh():
    coords, tour, g = make_graph(seed=0)               # instance with full coverage
    assert g.y.sum() == 2 * 20
    heat, in_graph = edge_heatmap(g.edge_index, g.y, 20)
    assert tour_edges(greedy_decode(heat, in_graph, coords)) == tour_edges(tour)


@pytest.mark.parametrize("kind", ["random", "zeros"])
def test_greedy_always_gives_valid_tour(kind):
    for seed in range(5):
        coords, _, g = make_graph(seed=seed, n=30)
        probs = torch.rand(g.num_edges) if kind == "random" else torch.zeros(g.num_edges)
        heat, in_graph = edge_heatmap(g.edge_index, probs, 30)
        assert is_valid_tour(greedy_decode(heat, in_graph, coords), 30)


def test_greedy_fallback_with_sparse_graph():
    # With k=3 several tour edges are missing from the graph: greedy has to use pairs outside
    # the k-NN. Even with the true labels, the tour must be valid and gap >= 0.
    max_missing = 0
    for seed in range(5):
        coords = generate_instances(num=1, n=20, seed=seed)[0]
        tour = solve_lkh(coords)
        opt = tour_length(coords, tour)
        g = instance_to_data(coords, tour, opt, k=3)
        max_missing = max(max_missing, 20 - int(g.y.sum()) // 2)
        heat, in_graph = edge_heatmap(g.edge_index, g.y, 20)
        greedy = greedy_decode(heat, in_graph, coords)
        assert is_valid_tour(greedy, 20)
        assert optimality_gap(tour_length(coords, greedy), opt) >= -1e-9
        assert tour_length(coords, two_opt(coords, greedy)) <= tour_length(coords, greedy) + 1e-12
    assert max_missing >= 2   # makes sure the "several missing edges" case was really tested


def test_two_opt_uncrosses_square():
    # 0-2-1-3 uses both diagonals (length 2 + 2√2); the optimum is the perimeter (4).
    improved = two_opt(SQUARE, [0, 2, 1, 3])
    assert np.isclose(tour_length(SQUARE, improved), 4.0)
    assert improved[0] == 0


def test_two_opt_never_worse():
    rng = np.random.default_rng(0)
    for coords in generate_instances(num=10, n=25, seed=3):
        tour = np.concatenate([[0], rng.permutation(np.arange(1, 25))])
        improved = two_opt(coords, tour)
        assert is_valid_tour(improved, 25)
        assert tour_length(coords, improved) <= tour_length(coords, tour) + 1e-12


def test_verify_tour_rejects_bad_tours():
    verify_tour(np.array([0, 2, 1, 3]), 4)             # valid: does not raise
    with pytest.raises(ValueError):
        verify_tour(np.array([0, 1, 1, 3]), 4)         # repeated city (and 2 missing)
    with pytest.raises(ValueError):
        verify_tour(np.array([0, 1, 2]), 4)            # wrong length


def test_optimality_gap():
    assert np.isclose(optimality_gap(105.0, 100.0), 5.0)
    assert optimality_gap(100.0, 100.0) == 0.0


def test_decode_end_to_end_true_labels():
    _, _, g = make_graph(seed=0)
    result = decode(g, g.y)
    assert abs(result["gap"]) < 1e-4                    # x is float32: small tolerance


def noisy_heatmap(seed, n, k):
    """Heatmap of a fake, imperfect model: probability falls with distance, plus noise."""
    coords = generate_instances(num=1, n=n, seed=seed)[0]
    g = instance_to_data(coords, np.arange(n), 1.0, k=k)
    torch.manual_seed(seed)
    d = torch.norm(g.x[g.edge_index[0]] - g.x[g.edge_index[1]], dim=1)
    probs = torch.sigmoid(-10 * d + 2 * torch.randn(len(d)))
    heat, in_graph = edge_heatmap(g.edge_index, probs, n)
    return coords, heat, in_graph


@pytest.mark.parametrize("n, k", [(20, 10), (50, 25), (30, 3)])   # k=3: many pairs need the fallback
def test_beam_width_1_is_greedy(n, k):
    for seed in range(10):
        coords, heat, in_graph = noisy_heatmap(seed, n, k)
        assert np.array_equal(beam_search_decode(heat, in_graph, coords, 1),
                              greedy_decode(heat, in_graph, coords))
    # Ties everywhere (all probabilities 0): same tie-breaks as greedy.
    heat = np.zeros_like(heat)
    assert np.array_equal(beam_search_decode(heat, in_graph, coords, 1), greedy_decode(heat, in_graph, coords))


@pytest.mark.parametrize("beam_width", [2, 10, 100])
def test_beam_gives_valid_tours(beam_width):
    for seed in range(5):
        coords, heat, in_graph = noisy_heatmap(seed, 30, 5)
        tour = beam_search_decode(heat, in_graph, coords, beam_width)
        assert is_valid_tour(tour, 30) and tour[0] == 0


def test_beam_shorter_on_average():
    # Pure beam search does not guarantee a shorter tour on every instance (the greedy path can
    # be pruned from the beam), so we check the mean over a small sample.
    lengths = {1: [], 10: [], 100: []}
    for seed in range(20):
        coords, heat, in_graph = noisy_heatmap(seed, 30, 15)
        for b in lengths:
            lengths[b].append(tour_length(coords, beam_search_decode(heat, in_graph, coords, b)))
    assert np.mean(lengths[10]) < np.mean(lengths[1])
    assert np.mean(lengths[100]) <= np.mean(lengths[10])


def test_beam_with_true_labels_recovers_lkh():
    coords, tour, g = make_graph(seed=0)
    heat, in_graph = edge_heatmap(g.edge_index, g.y, 20)
    assert tour_edges(beam_search_decode(heat, in_graph, coords, 10)) == tour_edges(tour)
