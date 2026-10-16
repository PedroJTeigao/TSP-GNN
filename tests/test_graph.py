"""Testes rápidos para src/tsp_gnn/graph.py.  Correr com: pytest tests/"""

import numpy as np
import torch

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data, knn_edges, tour_labels

SQUARE = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)


def edge_set(edge_index):
    return {tuple(e) for e in edge_index.t().tolist()}


def test_square_knn_by_hand():
    # Com k=2, cada canto liga aos dois cantos adjacentes (distância 1), não à diagonal (√2).
    sides = {(0, 1), (1, 2), (2, 3), (3, 0)}
    expected = sides | {(j, i) for i, j in sides}
    assert edge_set(knn_edges(SQUARE, k=2)) == expected


def test_square_labels_by_hand():
    edge_index = knn_edges(SQUARE, k=2)
    # Tour pelos lados: as 8 arestas (4 lados x 2 sentidos) pertencem ao tour.
    assert tour_labels(edge_index, [0, 1, 2, 3], 4).sum() == 8
    # Tour 0-2-1-3: usa 2 diagonais (fora do grafo) e 2 lados (1-2 e 3-0) -> 4 positivas.
    y = tour_labels(edge_index, [0, 2, 1, 3], 4)
    positives = edge_set(edge_index[:, y == 1])
    assert positives == {(1, 2), (2, 1), (3, 0), (0, 3)}


def test_random_graph_properties():
    coords = generate_instances(num=1, n=20, seed=0)[0]
    tour = solve_lkh(coords)
    g = instance_to_data(coords, tour, tour_length(coords, tour), k=10)
    src, dst = g.edge_index

    assert (src != dst).all()                                   # sem self-loops
    assert edge_set(g.edge_index) == edge_set(g.edge_index.flip(0))  # simétrico
    assert g.num_edges == len(edge_set(g.edge_index))           # sem arestas repetidas
    # edge_attr = distância real entre as cidades
    real = np.linalg.norm(coords[src.numpy()] - coords[dst.numpy()], axis=1)
    assert np.allclose(g.edge_attr.view(-1).numpy(), real, atol=1e-6)
    # cada aresta do tour que está no grafo aparece 2x (dois sentidos) com y=1;
    # contamos essas arestas de forma independente, sem assumir cobertura total
    tour_edges = {(int(a), int(b)) for a, b in zip(tour, np.roll(tour, -1))}
    covered = sum((a, b) in edge_set(g.edge_index) for a, b in tour_edges)
    assert g.y.sum() == 2 * covered
    assert torch.isclose(g.opt_len, torch.tensor([tour_length(coords, tour)]))
