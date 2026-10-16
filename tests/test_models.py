"""Quick tests for src/tsp_gnn/models.py.  Run with: pytest tests/"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data
from tsp_gnn.models import EdgeGNN

KINDS = ["gcn", "gat", "gated"]


def small_batch(num=4, n=20):
    graphs = []
    for coords in generate_instances(num=num, n=n, seed=0):
        tour = solve_lkh(coords)
        graphs.append(instance_to_data(coords, tour, tour_length(coords, tour)))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shape(kind):
    torch.manual_seed(0)
    batch = small_batch()
    logits = EdgeGNN(kind)(batch)
    assert logits.shape == (batch.num_edges,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("kind", KINDS)
def test_all_parameters_get_gradient(kind):
    torch.manual_seed(0)
    batch = small_batch()
    model = EdgeGNN(kind)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch.y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient (not used in forward?)"
        assert p.grad.abs().sum() > 0, f"{name} has a zero gradient"


@pytest.mark.parametrize("kind", KINDS)
def test_permutation_equivariance(kind):
    # Renumbering the cities must not change the logit of any edge.
    torch.manual_seed(0)
    g = small_batch(num=1)[0]
    model = EdgeGNN(kind).eval()                       # eval: BatchNorm does not depend on the batch

    perm = torch.randperm(g.num_nodes)                 # new city k = old city perm[k]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(g.num_nodes)              # old city v -> new number inv[v]
    g_perm = Data(x=g.x[perm], edge_index=inv[g.edge_index], edge_attr=g.edge_attr)

    with torch.no_grad():
        assert torch.allclose(model(g), model(g_perm), atol=1e-5)
