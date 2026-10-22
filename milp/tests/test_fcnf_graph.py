"""Tests for fcnf_gnn.graph, fcnf_gnn.models and fcnf_gnn.train.  Run with: pytest tests/"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from fcnf_gnn.generate import generate_instances
from fcnf_gnn.graph import build_graph
from fcnf_gnn.models import ArcGNN
from fcnf_gnn.train import compute_pos_weight, heatmap_metrics, predict_probs


def graphs(num=4, n=20):
    out = []
    for inst in generate_instances(num, n, seed=3):
        inst["y"] = (np.arange(len(inst["src"])) % 9 == 0).astype(np.int8)    # fake labels
        out.append(build_graph(inst))
    return out


def test_features():
    inst = generate_instances(1, 20, seed=4)[0]
    g = build_graph(inst)
    m = len(inst["src"])
    assert g.x.shape == (20, 5) and g.arc_attr.shape == (m, 5) and g.arc_index.shape == (2, m)
    assert torch.allclose(g.x[:, 0].abs().max(), torch.tensor(1.0))
    assert torch.allclose(g.arc_attr[:, :4].mean(0), torch.ones(4), atol=1e-5)        # normalized by the mean
    assert torch.allclose(g.arc_attr[:, 4], torch.full((m,), float(np.log(inst["ratio"]))))
    # message graph: both directions, no duplicates, covers every arc
    e = set(map(tuple, g.edge_index.t().tolist()))
    assert len(e) == g.edge_index.shape[1]
    assert all((i, j) in e and (j, i) in e for i, j in g.arc_index.t().tolist())


def test_batching_and_predictions():
    gs = graphs()
    batch = next(iter(DataLoader(gs, batch_size=4)))
    offsets = np.cumsum([0] + [g.num_nodes for g in gs[:-1]])
    start = 0
    for g, off in zip(gs, offsets):                                  # arc_index shifted per graph
        m = g.arc_index.shape[1]
        assert torch.equal(batch.arc_index[:, start:start + m], g.arc_index + int(off))
        start += m
    for kind in ("gcn", "tagcn"):
        model = ArcGNN(kind)
        probs, _ = predict_probs(model, gs, batch_size=3)
        assert [len(p) for p in probs] == [g.arc_index.shape[1] for g in gs]
        assert all(((p >= 0) & (p <= 1)).all() for p in probs)
    m = heatmap_metrics(gs, probs)
    assert 0 <= m["ap"] <= 1 and abs(m["base_rate"] - np.concatenate([g.y.numpy() for g in gs]).mean()) < 1e-9


def test_pos_weight():
    gs = graphs()
    y = np.concatenate([g.y.numpy() for g in gs])
    assert abs(compute_pos_weight(gs) - (len(y) - y.sum()) / y.sum()) < 1e-9
