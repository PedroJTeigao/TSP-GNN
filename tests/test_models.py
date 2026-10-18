"""Quick tests for src/tsp_gnn/models.py.  Run with: pytest tests/"""

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch, Data

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data
from tsp_gnn.models import NORMS, EdgeGNN, with_batch_stats

KINDS = ["gcn", "gat", "gated"]


def small_batch(num=4, n=20):
    graphs = []
    for coords in generate_instances(num=num, n=n, seed=0):
        tour = solve_lkh(coords)
        graphs.append(instance_to_data(coords, tour, tour_length(coords, tour)))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("kind", KINDS)
def test_forward_shape(kind, norm):
    torch.manual_seed(0)
    batch = small_batch()
    logits = EdgeGNN(kind, norm=norm)(batch)
    assert logits.shape == (batch.num_edges,)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("kind", KINDS)
def test_all_parameters_get_gradient(kind, norm):
    torch.manual_seed(0)
    batch = small_batch()
    model = EdgeGNN(kind, norm=norm)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch.y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient (not used in forward?)"
        assert p.grad.abs().sum() > 0, f"{name} has a zero gradient"


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("kind", KINDS)
def test_permutation_equivariance(kind, norm):
    # Renumbering the cities must not change the logit of any edge.
    torch.manual_seed(0)
    g = small_batch(num=1)[0]
    model = EdgeGNN(kind, norm=norm).eval()            # eval: BatchNorm does not depend on the batch

    perm = torch.randperm(g.num_nodes)                 # new city k = old city perm[k]
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(g.num_nodes)              # old city v -> new number inv[v]
    g_perm = Data(x=g.x[perm], edge_index=inv[g.edge_index], edge_attr=g.edge_attr)

    with torch.no_grad():
        assert torch.allclose(model(g), model(g_perm), atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_default_norm_is_phase1_batchnorm(kind):
    # Same seed -> identical weights and parameter names with and without norm="batchnorm":
    # phase 1 checkpoints load unchanged and default training starts from the same weights.
    torch.manual_seed(0)
    default = EdgeGNN(kind).state_dict()
    torch.manual_seed(0)
    explicit = EdgeGNN(kind, norm="batchnorm").state_dict()
    assert default.keys() == explicit.keys()
    assert all(torch.equal(default[k], explicit[k]) for k in default)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in EdgeGNN(kind).modules())


@pytest.mark.parametrize("kind", KINDS)
def test_with_batch_stats(kind):
    torch.manual_seed(0)
    batch = small_batch()
    model = EdgeGNN(kind)
    with torch.no_grad():                              # a few train-mode passes: running stats move away from 0/1
        for _ in range(3):
            model(batch)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    copy = with_batch_stats(model).eval()
    reference = deepcopy(model).train()                # train mode = normalize with the batch statistics
    with torch.no_grad():
        out_copy, out_ref, out_stored = copy(batch), reference(batch), model(batch)
    assert torch.allclose(out_copy, out_ref, atol=1e-5)
    assert not torch.allclose(out_copy, out_stored, atol=1e-3)   # differs from the stored statistics
    # The original model is untouched: same weights and the same stored statistics.
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_with_batch_stats_needs_batchnorm():
    with pytest.raises(ValueError):
        with_batch_stats(EdgeGNN("gated", norm="layernorm"))
