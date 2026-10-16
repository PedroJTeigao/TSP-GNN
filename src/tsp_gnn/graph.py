"""Conversion of TSP instances into PyTorch Geometric k-NN graphs.

Each instance becomes a torch_geometric.data.Data with:
    x          (n, 2)  city coordinates
    edge_index (2, E)  edges of the k-NN graph, in both directions
    edge_attr  (E, 1)  length of each edge
    y          (E,)    1.0 if the edge belongs to the LKH tour, 0.0 otherwise
    opt_len    (1,)    length of the LKH tour (to compute the gap later)
"""

import torch
from torch_geometric.data import Data

from tsp_gnn.data import load_dataset


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


def instance_to_data(coords, tour, length, k=10):
    """Build the Data object of one instance (coordinates, LKH tour and its length)."""
    pos = torch.as_tensor(coords, dtype=torch.float32)
    edge_index = knn_edges(pos, k)
    src, dst = edge_index
    edge_attr = (pos[src] - pos[dst]).norm(dim=1, keepdim=True)  # length of each edge
    return Data(
        x=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=tour_labels(edge_index, tour, pos.size(0)),
        opt_len=torch.tensor([length], dtype=torch.float32),  # shape (1,): becomes (B,) in a batch
    )


def load_graphs(path, k=10):
    """Read an .npz from data.py and return the list of graphs (built in memory)."""
    d = load_dataset(path)
    return [instance_to_data(c, t, l, k) for c, t, l in zip(d["coords"], d["tours"], d["lengths"])]
