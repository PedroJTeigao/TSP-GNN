"""Testes rápidos para src/tsp_gnn/train.py (no CPU).  Correr com: pytest tests/"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data
from tsp_gnn.models import EdgeGNN
from tsp_gnn.train import compute_pos_weight, predict_probs, train_epoch


def small_graphs(num=8, n=20):
    graphs = []
    for coords in generate_instances(num=num, n=n, seed=0):
        tour = solve_lkh(coords)
        graphs.append(instance_to_data(coords, tour, tour_length(coords, tour)))
    return graphs


def test_pos_weight_by_hand():
    g1 = Data(y=torch.tensor([1., 0., 0., 0.]), edge_index=torch.zeros(2, 4, dtype=torch.long))
    g2 = Data(y=torch.tensor([1., 1., 0., 0.]), edge_index=torch.zeros(2, 4, dtype=torch.long))
    assert compute_pos_weight([g1, g2]) == 5 / 3        # 5 negativas / 3 positivas


def test_can_overfit_tiny_dataset():
    # Se o modelo nem consegue decorar 8 grafos, há um erro no ciclo de treino.
    torch.manual_seed(0)
    graphs = small_graphs()
    loader = DataLoader(graphs, batch_size=8)
    model = EdgeGNN("gated", hidden=32, layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(compute_pos_weight(graphs)))
    losses = [train_epoch(model, loader, optimizer, loss_fn, "cpu") for _ in range(30)]
    assert losses[-1] < 0.5 * losses[0]


def test_predict_probs_splits_per_graph():
    # Separar o batch por grafo tem de dar o mesmo que correr o modelo grafo a grafo.
    torch.manual_seed(0)
    graphs = small_graphs(num=5)
    model = EdgeGNN("gat").eval()
    probs, _ = predict_probs(model, graphs, batch_size=3)  # 3 não divide 5: testa o último batch
    assert len(probs) == len(graphs)
    with torch.no_grad():
        for g, p in zip(graphs, probs):
            assert p.shape == (g.num_edges,)
            assert torch.allclose(p, torch.sigmoid(model(g)), atol=1e-6)
