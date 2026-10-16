"""Conversão de instâncias de TSP em grafos k-NN do PyTorch Geometric.

Cada instância vira um torch_geometric.data.Data com:
    x          (n, 2)  coordenadas das cidades
    edge_index (2, E)  arestas do grafo k-NN, nos dois sentidos
    edge_attr  (E, 1)  comprimento de cada aresta
    y          (E,)    1.0 se a aresta pertence ao tour do LKH, 0.0 se não
    opt_len    (1,)    comprimento do tour do LKH (para calcular o gap mais tarde)
"""

import torch
from torch_geometric.data import Data

from tsp_gnn.data import load_dataset


def knn_edges(pos, k=10):
    """Arestas do grafo k-NN de `pos` (n, 2), nos dois sentidos e sem repetidas. Forma: (2, E)."""
    n = pos.size(0)
    k = min(k, n - 1)                          # não há mais de n-1 vizinhos
    dist = torch.cdist(pos, pos)               # (n, n) distâncias entre todas as cidades
    dist.fill_diagonal_(float("inf"))          # uma cidade não é vizinha de si própria
    nbrs = dist.topk(k, largest=False).indices # (n, k) os k vizinhos mais próximos de cada cidade

    src = torch.arange(n).repeat_interleave(k) # i, i, ..., i  (k vezes, para cada i)
    dst = nbrs.reshape(-1)                     # os vizinhos correspondentes
    # O k-NN não é simétrico: juntamos as arestas inversas e tiramos as repetidas.
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return torch.unique(edge_index, dim=1)


def tour_labels(edge_index, tour, n):
    """Rótulo de cada aresta: 1.0 se liga duas cidades consecutivas no tour, 0.0 se não."""
    tour = torch.as_tensor(tour, dtype=torch.long)
    nxt = torch.roll(tour, -1)                 # cidade seguinte (a última liga à primeira)
    in_tour = torch.zeros(n, n, dtype=torch.bool)
    in_tour[tour, nxt] = True                  # marcar as arestas do tour nos dois sentidos
    in_tour[nxt, tour] = True
    return in_tour[edge_index[0], edge_index[1]].float()


def instance_to_data(coords, tour, length, k=10):
    """Constrói o Data de uma instância (coordenadas, tour do LKH e o seu comprimento)."""
    pos = torch.as_tensor(coords, dtype=torch.float32)
    edge_index = knn_edges(pos, k)
    src, dst = edge_index
    edge_attr = (pos[src] - pos[dst]).norm(dim=1, keepdim=True)  # comprimento de cada aresta
    return Data(
        x=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=tour_labels(edge_index, tour, pos.size(0)),
        opt_len=torch.tensor([length], dtype=torch.float32),  # forma (1,): num batch fica (B,)
    )


def load_graphs(path, k=10):
    """Lê um .npz de data.py e devolve a lista de grafos (construídos em memória)."""
    d = load_dataset(path)
    return [instance_to_data(c, t, l, k) for c, t, l in zip(d["coords"], d["tours"], d["lengths"])]
