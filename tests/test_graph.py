"""Quick tests for src/tsp_gnn/graph.py.  Run with: pytest tests/"""

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data, knn_edges, normalize_edges, resolve_k, tour_labels

SQUARE = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)


def edge_set(edge_index):
    return {tuple(e) for e in edge_index.t().tolist()}


def test_square_knn_by_hand():
    # With k=2, each corner links to the two adjacent corners (distance 1), not the diagonal (√2).
    sides = {(0, 1), (1, 2), (2, 3), (3, 0)}
    expected = sides | {(j, i) for i, j in sides}
    assert edge_set(knn_edges(SQUARE, k=2)) == expected


def test_square_labels_by_hand():
    edge_index = knn_edges(SQUARE, k=2)
    # Tour along the sides: all 8 edges (4 sides x 2 directions) belong to the tour.
    assert tour_labels(edge_index, [0, 1, 2, 3], 4).sum() == 8
    # Tour 0-2-1-3: uses 2 diagonals (outside the graph) and 2 sides (1-2 and 3-0) -> 4 positives.
    y = tour_labels(edge_index, [0, 2, 1, 3], 4)
    positives = edge_set(edge_index[:, y == 1])
    assert positives == {(1, 2), (2, 1), (3, 0), (0, 3)}


def test_random_graph_properties():
    coords = generate_instances(num=1, n=20, seed=0)[0]
    tour = solve_lkh(coords)
    g = instance_to_data(coords, tour, tour_length(coords, tour), k=10)
    src, dst = g.edge_index

    assert (src != dst).all()                                   # no self-loops
    assert edge_set(g.edge_index) == edge_set(g.edge_index.flip(0))  # symmetric
    assert g.num_edges == len(edge_set(g.edge_index))           # no duplicate edges
    # edge_attr = real distance between the cities
    real = np.linalg.norm(coords[src.numpy()] - coords[dst.numpy()], axis=1)
    assert np.allclose(g.edge_attr.view(-1).numpy(), real, atol=1e-6)
    # each tour edge that is in the graph appears twice (both directions) with y=1;
    # we count those edges independently, without assuming full coverage
    tour_edges = {(int(a), int(b)) for a, b in zip(tour, np.roll(tour, -1))}
    covered = sum((a, b) in edge_set(g.edge_index) for a, b in tour_edges)
    assert g.y.sum() == 2 * covered
    assert torch.isclose(g.opt_len, torch.tensor([tour_length(coords, tour)]))


# ---------------------------------------------------------------- edge_norm (phase 2)

def phase1_data(coords, tour, k=10):
    """The graph exactly as phase 1 built it (the code before edge_norm existed)."""
    pos = torch.as_tensor(coords, dtype=torch.float32)
    edge_index = knn_edges(pos, k)
    src, dst = edge_index
    return edge_index, (pos[src] - pos[dst]).norm(dim=1, keepdim=True), tour_labels(edge_index, tour, len(pos))


@pytest.mark.parametrize("n", [20, 50])
def test_edge_norm_none_is_phase1(n):
    coords = generate_instances(num=1, n=n, seed=1)[0]
    tour = solve_lkh(coords)
    edge_index, edge_attr, y = phase1_data(coords, tour)
    for g in (instance_to_data(coords, tour, 1.0), instance_to_data(coords, tour, 1.0, edge_norm="none")):
        assert torch.equal(g.edge_index, edge_index)    # bit-for-bit, not just close
        assert torch.equal(g.edge_attr, edge_attr)
        assert torch.equal(g.y, y)


@pytest.mark.parametrize("n", [20, 50, 100])
def test_edge_norm_knn_mean_and_sqrt_n(n):
    coords = generate_instances(num=1, n=n, seed=2)[0]
    tour = np.arange(n)                                 # any valid tour: only edge_attr matters here
    edge_index, edge_attr, y = phase1_data(coords, tour)

    g = instance_to_data(coords, tour, 1.0, edge_norm="knn_mean")
    assert torch.isclose(g.edge_attr.mean(), torch.tensor(1.0), atol=1e-6)   # mean edge length = 1
    assert torch.allclose(g.edge_attr, edge_attr / edge_attr.mean())
    assert torch.equal(g.edge_index, edge_index) and torch.equal(g.y, y)     # only edge_attr changes

    g = instance_to_data(coords, tour, 1.0, edge_norm="sqrt_n")
    assert torch.allclose(g.edge_attr, edge_attr * np.sqrt(n))
    assert torch.equal(g.edge_index, edge_index) and torch.equal(g.y, y)


def test_edge_norm_unknown_raises():
    with pytest.raises(ValueError):
        normalize_edges(torch.ones(3, 1), 3, "mean")



# ---------------------------------------------------------------- k_frac (phase 2)

def test_resolve_k():
    assert [resolve_k(n) for n in (5, 20, 50, 100)] == [4, 10, 10, 10]        # default: fixed k=10
    assert [resolve_k(n, k_frac=0.5) for n in (20, 50, 100)] == [10, 25, 50]
    assert resolve_k(20, k=3, k_frac=0.5) == 10                               # k_frac wins over k
    assert resolve_k(10, k_frac=1.0) == 9                                     # never more than n-1


def test_k_frac_half_on_tsp20_is_phase1_graph():
    coords = generate_instances(num=1, n=20, seed=3)[0]
    tour = solve_lkh(coords)
    g10 = instance_to_data(coords, tour, 1.0)                     # phase 1: k = 10
    gfrac = instance_to_data(coords, tour, 1.0, k_frac=0.5)       # k = round(0.5 × 20) = 10
    for key in ("edge_index", "edge_attr", "y"):
        assert torch.equal(g10[key], gfrac[key])


def test_k_frac_grows_graph_on_tsp50():
    coords = generate_instances(num=1, n=50, seed=3)[0]
    g = instance_to_data(coords, np.arange(50), 1.0, k_frac=0.5)
    out_degree = torch.bincount(g.edge_index[0], minlength=50)
    assert (out_degree >= 25).all()                               # each city keeps its 25 nearest
