"""GNNs that predict, for each edge of the k-NN graph, whether it belongs to the tour.

EdgeGNN(kind, norm) with kind in {'gcn', 'gat', 'gated'} and norm in {'batchnorm', 'layernorm'}:
    embeddings (coordinates and distance) -> L encoder layers (residual + normalization)
    -> MLP head([h_i, h_j, e_ij]) -> one logit per edge.

BatchNorm (phase 1) normalizes each feature with statistics of the whole batch, stored at
training time; on larger graphs those stored statistics may no longer fit. LayerNorm normalizes
each node/edge vector on its own, so it does not depend on the batch nor on stored statistics.
PyG convention: edge_index[0] = source j, edge_index[1] = target i (i aggregates from j).
"""

from copy import deepcopy

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import scatter

NORMS = ("batchnorm", "layernorm")


def make_norm(norm, hidden):
    """Normalization layer over the `hidden` features."""
    if norm == "batchnorm":
        return nn.BatchNorm1d(hidden)
    if norm == "layernorm":
        return nn.LayerNorm(hidden)
    raise ValueError(f"unknown norm: {norm!r} (use one of {NORMS})")


class GatedGCNLayer(nn.Module):
    """Residual Gated GCN layer (Bresson & Laurent 2017; Joshi et al. 2019).

        ê_ij = A h_i + B h_j + C e_ij
        η_ij = σ(ê_ij) / (Σ_j' σ(ê_ij') + ε)
        h_i  ← h_i  + ReLU(BN(U h_i + Σ_j η_ij ⊙ V h_j))
        e_ij ← e_ij + ReLU(BN(ê_ij))
    (BN = BatchNorm by default, or LayerNorm with norm="layernorm".)
    """

    def __init__(self, hidden, eps=1e-6, norm="batchnorm"):
        super().__init__()
        self.A, self.B, self.C, self.U, self.V = (nn.Linear(hidden, hidden) for _ in range(5))
        # Names kept as bn_* for every norm so that phase 1 checkpoints load unchanged.
        self.bn_h = make_norm(norm, hidden)
        self.bn_e = make_norm(norm, hidden)
        self.eps = eps

    def forward(self, h, e, edge_index):
        src, dst = edge_index                              # message j=src -> i=dst
        e_hat = self.A(h)[dst] + self.B(h)[src] + self.C(e)  # new vector of each edge
        gate = torch.sigmoid(e_hat)                        # gates in [0,1], one per dimension
        n = h.size(0)
        num = scatter(gate * self.V(h)[src], dst, dim=0, dim_size=n, reduce="sum")
        den = scatter(gate, dst, dim=0, dim_size=n, reduce="sum")
        h_new = self.U(h) + num / (den + self.eps)         # "soft" average of the neighbours
        h = h + torch.relu(self.bn_h(h_new))               # residual + normalization
        e = e + torch.relu(self.bn_e(e_hat))
        return h, e


class EdgeGNN(nn.Module):
    """Edge classifier with a GCN, GATv2 or Gated GCN encoder."""

    def __init__(self, kind, hidden=64, layers=6, heads=4, norm="batchnorm"):
        super().__init__()
        self.kind = kind
        self.norm = norm
        self.node_embed = nn.Linear(2, hidden)             # coordinates -> h⁰
        self.edge_embed = nn.Linear(1, hidden)             # distance    -> e⁰

        if kind == "gcn":      # fixed average weighted by degrees; does not use distances
            self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        elif kind == "gat":    # attention with the distance; 4 heads of hidden/4, concatenated
            self.convs = nn.ModuleList(
                GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=hidden)
                for _ in range(layers))
        elif kind == "gated":  # also updates one vector per edge
            self.convs = nn.ModuleList(GatedGCNLayer(hidden, norm=norm) for _ in range(layers))
        else:
            raise ValueError(f"unknown kind: {kind!r} (use 'gcn', 'gat' or 'gated')")
        if kind != "gated":    # the Gated layer already has its own normalizations
            self.norms = nn.ModuleList(make_norm(norm, hidden) for _ in range(layers))

        # Shared head: [h_i, h_j, e_ij] -> edge logit (the sigmoid lives in the loss).
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, data):
        edge_index = data.edge_index
        h = self.node_embed(data.x)
        e = self.edge_embed(data.edge_attr)
        for layer_idx, conv in enumerate(self.convs):
            if self.kind == "gated":
                h, e = conv(h, e, edge_index)
            else:
                out = conv(h, edge_index) if self.kind == "gcn" else conv(h, edge_index, e)
                h = h + torch.relu(self.norms[layer_idx](out))  # residual + normalization
        src, dst = edge_index
        return self.head(torch.cat([h[src], h[dst], e], dim=-1)).squeeze(-1)  # (E,)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def with_batch_stats(model):
    """Copy of `model` whose BatchNorms use the statistics of the current batch, also in eval mode.

    Normally, in eval mode, BatchNorm uses the running mean/variance stored during training.
    Here the copy drops those stored statistics, so each forward pass normalizes with the mean and
    variance of the batch it receives. The weights are the same; the original model (and its
    files) are not modified.
    """
    batch_norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not batch_norms:
        raise ValueError("the model has no BatchNorm layers (e.g. norm='layernorm')")
    copy = deepcopy(model)
    for m in copy.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False   # with no stored statistics PyTorch uses the batch ones
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return copy
