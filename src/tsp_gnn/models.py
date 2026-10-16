"""GNNs that predict, for each edge of the k-NN graph, whether it belongs to the tour.

EdgeGNN(kind) with kind in {'gcn', 'gat', 'gated'}:
    embeddings (coordinates and distance) -> L encoder layers (residual + BatchNorm)
    -> MLP head([h_i, h_j, e_ij]) -> one logit per edge.
PyG convention: edge_index[0] = source j, edge_index[1] = target i (i aggregates from j).
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import scatter


class GatedGCNLayer(nn.Module):
    """Residual Gated GCN layer (Bresson & Laurent 2017; Joshi et al. 2019).

        ê_ij = A h_i + B h_j + C e_ij
        η_ij = σ(ê_ij) / (Σ_j' σ(ê_ij') + ε)
        h_i  ← h_i  + ReLU(BN(U h_i + Σ_j η_ij ⊙ V h_j))
        e_ij ← e_ij + ReLU(BN(ê_ij))
    """

    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.A, self.B, self.C, self.U, self.V = (nn.Linear(hidden, hidden) for _ in range(5))
        self.bn_h = nn.BatchNorm1d(hidden)
        self.bn_e = nn.BatchNorm1d(hidden)
        self.eps = eps

    def forward(self, h, e, edge_index):
        src, dst = edge_index                              # message j=src -> i=dst
        e_hat = self.A(h)[dst] + self.B(h)[src] + self.C(e)  # new vector of each edge
        gate = torch.sigmoid(e_hat)                        # gates in [0,1], one per dimension
        n = h.size(0)
        num = scatter(gate * self.V(h)[src], dst, dim=0, dim_size=n, reduce="sum")
        den = scatter(gate, dst, dim=0, dim_size=n, reduce="sum")
        h_new = self.U(h) + num / (den + self.eps)         # "soft" average of the neighbours
        h = h + torch.relu(self.bn_h(h_new))               # residual + BatchNorm
        e = e + torch.relu(self.bn_e(e_hat))
        return h, e


class EdgeGNN(nn.Module):
    """Edge classifier with a GCN, GATv2 or Gated GCN encoder."""

    def __init__(self, kind, hidden=64, layers=6, heads=4):
        super().__init__()
        self.kind = kind
        self.node_embed = nn.Linear(2, hidden)             # coordinates -> h⁰
        self.edge_embed = nn.Linear(1, hidden)             # distance    -> e⁰

        if kind == "gcn":      # fixed average weighted by degrees; does not use distances
            self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        elif kind == "gat":    # attention with the distance; 4 heads of hidden/4, concatenated
            self.convs = nn.ModuleList(
                GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=hidden)
                for _ in range(layers))
        elif kind == "gated":  # also updates one vector per edge
            self.convs = nn.ModuleList(GatedGCNLayer(hidden) for _ in range(layers))
        else:
            raise ValueError(f"unknown kind: {kind!r} (use 'gcn', 'gat' or 'gated')")
        if kind != "gated":    # the Gated layer already has its own BatchNorms
            self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))

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
                h = h + torch.relu(self.norms[layer_idx](out))  # residual + BatchNorm
        src, dst = edge_index
        return self.head(torch.cat([h[src], h[dst], e], dim=-1)).squeeze(-1)  # (E,)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
