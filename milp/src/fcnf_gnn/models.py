"""GNN that predicts, for each arc of an FCNF instance, whether it is open in the optimal solution.

Adapted from tsp_gnn.models (EdgeGNN, GCN and TAGCN encoders):
    node embedding -> L encoder layers on the undirected graph (residual + LayerNorm)
    -> MLP head([h_i, h_j, e_ij]) on every arc (i, j) -> one logit "arc open".
As in the TSP project, GCN and TAGCN propagate node vectors over the topology only; the arc data
(c, F, u, length, ratio) enter through the head. h_i (tail) and h_j (head) are concatenated in
order, so the head sees the arc's direction.
PyG convention: edge_index[0] = source j, edge_index[1] = target i (i aggregates from j).
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv, TAGConv

from fcnf_gnn.graph import ARC_FEATURES, NODE_FEATURES

KINDS = ("gcn", "tagcn")


class ArcGNN(nn.Module):
    def __init__(self, kind, hidden=64, layers=3, tag_k=3):
        super().__init__()
        self.kind = kind
        self.node_embed = nn.Linear(NODE_FEATURES, hidden)
        self.arc_embed = nn.Linear(ARC_FEATURES, hidden)
        if kind == "gcn":      # D^-1/2 (A + I) D^-1/2 H W
            self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        elif kind == "tagcn":  # Σ_k Â^k H W_k, k = 0..K (Du et al. 2017)
            self.convs = nn.ModuleList(TAGConv(hidden, hidden, K=tag_k) for _ in range(layers))
        else:
            raise ValueError(f"unknown kind: {kind!r} (use one of {KINDS})")
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, data):
        h = self.node_embed(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(norm(conv(h, data.edge_index)))     # residual + LayerNorm
        src, dst = data.arc_index
        e = self.arc_embed(data.arc_attr)
        return self.head(torch.cat([h[src], h[dst], e], dim=-1)).squeeze(-1)   # (m,)


def build_model(cfg):
    return ArcGNN(cfg["model"], cfg["hidden"], cfg["layers"], cfg.get("tag_k", 3))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
