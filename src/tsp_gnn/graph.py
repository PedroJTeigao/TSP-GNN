"""Conversion of TSP instances into PyTorch Geometric k-NN graphs.

Each instance becomes a torch_geometric.data.Data with:
    x          (n, 2)  city coordinates
    edge_index (2, E)  edges of the k-NN graph, in both directions
    edge_attr  (E, 1)  length of each edge (optionally rescaled, see EDGE_NORMS)
    y          (E,)    1.0 if the edge belongs to the LKH tour, 0.0 otherwise
    opt_len    (1,)    length of the LKH tour (to compute the gap later)
"""

import torch
from torch_geometric.data import Data

from tsp_gnn.data import load_dataset

# How edge_attr is rescaled. In uniform [0,1]² instances the typical neighbour distance
# shrinks like 1/√n, so a model trained on TSP-20 sees smaller distances on TSP-50.
#   none      raw lengths (phase 1)
#   knn_mean  divide by the mean edge length of the instance's k-NN graph (mean becomes 1)
#   sqrt_n    multiply by √n (fixed rescaling, does not look at the data)
EDGE_NORMS = ("none", "knn_mean", "sqrt_n")


def resolve_k(n, k=10, k_frac=None):
    """Number of neighbours: fixed `k` (phase 1), or round(k_frac × n) if k_frac is given.

    With k_frac = 0.5 each city sees half of the others, as with k=10 on TSP-20:
    k = 10 on TSP-20, 25 on TSP-50, 50 on TSP-100. Never more than n-1.
    """
    if k_frac is None:
        return min(k, n - 1)
    return min(n - 1, round(k_frac * n))


def knn_edges(pos, k=10):
    """Edges of the k-NN graph of `pos` (n, 2), in both directions and without duplicates. Shape: (2, E)."""
    n = pos.size(0)
    k = min(k, n - 1)                          # there are at most n-1 neighbours
    dist = torch.cdist(pos, pos)               # (n, n) distances between all cities
    dist.fill_diagonal_(float("inf"))          # a city is not its own neighbour
    nbrs = dist.topk(k, largest=False).indices # (n, k) the k nearest neighbours of each city

    src = torch.arange(n).repeat_interleave(k) # i, i, ..., i  (k times, for each i)
    dst = nbrs.reshape(-1)                     # the corresponding neighbours
    # k-NN is not symmetric: add the reverse edges and remove duplicates.
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return torch.unique(edge_index, dim=1)


def tour_labels(edge_index, tour, n):
    """Label of each edge: 1.0 if it connects two consecutive cities of the tour, 0.0 otherwise."""
    tour = torch.as_tensor(tour, dtype=torch.long)
    nxt = torch.roll(tour, -1)                 # next city (the last links to the first)
    in_tour = torch.zeros(n, n, dtype=torch.bool)
    in_tour[tour, nxt] = True                  # mark the tour edges in both directions
    in_tour[nxt, tour] = True
    return in_tour[edge_index[0], edge_index[1]].float()


def normalize_edges(edge_attr, n, edge_norm="none"):
    """Rescale the edge lengths according to `edge_norm` (one of EDGE_NORMS)."""
    if edge_norm == "none":
        return edge_attr                       # untouched: bit-for-bit the phase 1 values
    if edge_norm == "knn_mean":
        return edge_attr / edge_attr.mean()
    if edge_norm == "sqrt_n":
        return edge_attr * n ** 0.5
    raise ValueError(f"unknown edge_norm: {edge_norm!r} (use one of {EDGE_NORMS})")


def instance_to_data(coords, tour, length, k=10, edge_norm="none", k_frac=None):
    """Build the Data object of one instance (coordinates, LKH tour and its length)."""
    pos = torch.as_tensor(coords, dtype=torch.float32)
    edge_index = knn_edges(pos, resolve_k(pos.size(0), k, k_frac))
    src, dst = edge_index
    edge_attr = (pos[src] - pos[dst]).norm(dim=1, keepdim=True)  # length of each edge
    edge_attr = normalize_edges(edge_attr, pos.size(0), edge_norm)
    return Data(
        x=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=tour_labels(edge_index, tour, pos.size(0)),
        opt_len=torch.tensor([length], dtype=torch.float32),  # shape (1,): becomes (B,) in a batch
    )


def load_graphs(path, k=10, edge_norm="none", k_frac=None):
    """Read an .npz from data.py and return the list of graphs (built in memory)."""
    d = load_dataset(path)
    return [instance_to_data(c, t, l, k, edge_norm, k_frac)
            for c, t, l in zip(d["coords"], d["tours"], d["lengths"])]
