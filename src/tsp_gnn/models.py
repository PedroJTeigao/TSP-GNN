"""GNNs que preveem, para cada aresta do grafo k-NN, se pertence ao tour.

EdgeGNN(kind) com kind em {'gcn', 'gat', 'gated'}:
    embeddings (coordenadas e distância) -> L camadas do encoder (residual + BatchNorm)
    -> cabeça MLP([h_i, h_j, e_ij]) -> um logit por aresta.
Convenção do PyG: edge_index[0] = origem j, edge_index[1] = destino i (i agrega de j).
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import scatter


class GatedGCNLayer(nn.Module):
    """Camada Gated GCN residual (Bresson & Laurent 2017; Joshi et al. 2019).

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
        src, dst = edge_index                              # mensagem j=src -> i=dst
        e_hat = self.A(h)[dst] + self.B(h)[src] + self.C(e)  # novo vetor de cada aresta
        gate = torch.sigmoid(e_hat)                        # portas em [0,1], uma por dimensão
        n = h.size(0)
        num = scatter(gate * self.V(h)[src], dst, dim=0, dim_size=n, reduce="sum")
        den = scatter(gate, dst, dim=0, dim_size=n, reduce="sum")
        h_new = self.U(h) + num / (den + self.eps)         # média "suave" dos vizinhos
        h = h + torch.relu(self.bn_h(h_new))               # residual + BatchNorm
        e = e + torch.relu(self.bn_e(e_hat))
        return h, e


class EdgeGNN(nn.Module):
    """Classificador de arestas com encoder GCN, GATv2 ou Gated GCN."""

    def __init__(self, kind, hidden=64, layers=6, heads=4):
        super().__init__()
        self.kind = kind
        self.node_embed = nn.Linear(2, hidden)             # coordenadas -> h⁰
        self.edge_embed = nn.Linear(1, hidden)             # distância   -> e⁰

        if kind == "gcn":      # média fixa pelos graus; não usa as distâncias
            self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        elif kind == "gat":    # atenção com a distância; 4 cabeças de hidden/4, concatenadas
            self.convs = nn.ModuleList(
                GATv2Conv(hidden, hidden // heads, heads=heads, concat=True, edge_dim=hidden)
                for _ in range(layers))
        elif kind == "gated":  # atualiza também um vetor por aresta
            self.convs = nn.ModuleList(GatedGCNLayer(hidden) for _ in range(layers))
        else:
            raise ValueError(f"kind desconhecido: {kind!r} (use 'gcn', 'gat' ou 'gated')")
        if kind != "gated":    # a camada Gated já tem as suas BatchNorm
            self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))

        # Cabeça comum: [h_i, h_j, e_ij] -> logit da aresta (a sigmoide fica na perda).
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
    """Número de parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
