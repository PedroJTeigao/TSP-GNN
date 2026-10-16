"""Quick tests for src/tsp_gnn/graph.py.  Run with: pytest tests/"""

import numpy as np
import torch

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data, knn_edges, tour_labels

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
