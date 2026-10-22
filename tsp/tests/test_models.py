"""Quick tests for src/tsp_gnn/models.py.  Run with: pytest tests/"""

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Batch, Data

from tsp_gnn.data import generate_instances, solve_lkh, tour_length
from tsp_gnn.graph import instance_to_data
from tsp_gnn.models import KINDS, NORMS, EdgeGNN, build_model, count_parameters, default_layers, with_batch_stats


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


# ---------------------------------------------------------------- TAGCN

@pytest.mark.parametrize("tag_k", [2, 3])
def test_tagcn_layers_give_six_hops(tag_k):
    # K=2 -> 3 layers, K=3 -> 2 layers: 6 hops of reach, like the 6 one-hop layers of the other models.
    layers = default_layers("tagcn", tag_k)
    assert layers * tag_k == 6
    model = EdgeGNN("tagcn", layers=layers, tag_k=tag_k)
    assert len(model.convs) == layers
    assert all(len(conv.lins) == tag_k + 1 for conv in model.convs)   # one W_k per hop k = 0..K
    assert default_layers("gcn") == default_layers("gat") == default_layers("gated") == 6


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("tag_k", [2, 3])
def test_tagcn_any_number_of_nodes(tag_k, norm):
    # The same weights run on TSP-20 and TSP-100 graphs: one logit per edge, nothing depends on n.
    torch.manual_seed(0)
    model = EdgeGNN("tagcn", layers=default_layers("tagcn", tag_k), norm=norm, tag_k=tag_k).eval()
    with torch.no_grad():
        for n in (20, 100):
            batch = small_batch(num=2, n=n)
            logits = model(batch)
            assert logits.shape == (batch.num_edges,)
            assert torch.isfinite(logits).all()


def test_tagcn_k0_is_a_per_node_mlp():
    # With K=0 there is no propagation: each layer is h <- h + ReLU(BN(W_0 h + b)), node by node.
    torch.manual_seed(0)
    g = small_batch(num=1)[0]
    model = EdgeGNN("tagcn", layers=2, tag_k=0).eval()
    with torch.no_grad():
        h_graph, _ = model.encode(g)
        no_edges = Data(x=g.x, edge_index=torch.empty(2, 0, dtype=torch.long), edge_attr=g.edge_attr[:0])
        h_alone, _ = model.encode(no_edges)                # same nodes, no neighbours at all
        h_mlp = model.node_embed(g.x)                      # the same computation written as an MLP
        for conv, norm in zip(model.convs, model.norms):
            h_mlp = h_mlp + torch.relu(norm(conv.lins[0](h_mlp) + conv.bias))
    assert torch.allclose(h_graph, h_alone, atol=1e-6)
    assert torch.allclose(h_graph, h_mlp, atol=1e-6)


def test_tagcn_parameters_and_build_model():
    # Per layer: (K + 1) weight matrices H×H (no bias) + one bias + the normalization (2H).
    h = 64
    for tag_k in (2, 3):
        layers = default_layers("tagcn", tag_k)
        model = EdgeGNN("tagcn", hidden=h, layers=layers, tag_k=tag_k)
        gcn_encoder = count_parameters(EdgeGNN("gcn", hidden=h, layers=6)) - 6 * (h * h + h + 2 * h)
        assert count_parameters(model) == gcn_encoder + layers * ((tag_k + 1) * h * h + h + 2 * h)
        cfg = {"model": "tagcn", "hidden": h, "layers": layers, "tag_k": tag_k, "norm": "layernorm"}
        rebuilt = build_model(cfg)
        assert rebuilt.tag_k == tag_k and rebuilt.norm == "layernorm" and len(rebuilt.convs) == layers
    # Phase 1 checkpoints (no tag_k, no norm in their config) still rebuild as before.
    old = build_model({"model": "gated", "hidden": h, "layers": 6})
    assert old.norm == "batchnorm" and len(old.convs) == 6
