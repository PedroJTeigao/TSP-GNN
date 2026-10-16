"""Estatísticas dos grafos k-NN de um ou mais conjuntos de dados.

Exemplo:
    python scripts/graph_stats.py --data data/tsp20_test.npz data/tsp50_test.npz --k 10
"""

import argparse

import torch

from tsp_gnn.graph import load_graphs


def stats(graphs):
    """Calcula as estatísticas de uma lista de grafos e devolve um dicionário."""
    num_edges = torch.tensor([g.num_edges for g in graphs], dtype=torch.float)
    num_pos = torch.tensor([g.y.sum().item() for g in graphs])
    n = torch.tensor([g.num_nodes for g in graphs], dtype=torch.float)
    covered = num_pos / 2                       # cada aresta do tour aparece 2x (dois sentidos)

    all_dists = torch.cat([g.edge_attr.view(-1) for g in graphs])
    tour_edge_len = torch.cat([g.opt_len for g in graphs]) / n   # comprimento médio de uma aresta do tour

    return {
        "n": int(n[0]),
        "E_mean": num_edges.mean().item(), "E_min": int(num_edges.min()), "E_max": int(num_edges.max()),
        "pos_frac": (num_pos.sum() / num_edges.sum()).item(),
        "coverage": (covered.sum() / n.sum()).item(),
        "incomplete": int((covered < n).sum()),  # grafos com alguma aresta do tour de fora
        "graph_edge_len": all_dists.mean().item(),
        "tour_edge_len": tour_edge_len.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Estatísticas dos grafos k-NN.")
    parser.add_argument("--data", nargs="+", required=True, help="ficheiros .npz de make_dataset.py")
    parser.add_argument("--k", type=int, default=10, help="número de vizinhos do grafo k-NN")
    args = parser.parse_args()

    for path in args.data:
        graphs = load_graphs(path, args.k)
        s = stats(graphs)
        print(f"=== {path}  ({len(graphs)} grafos, n={s['n']}, k={args.k}) ===")
        print(f"  arestas por grafo:          média {s['E_mean']:.1f}  (mín {s['E_min']}, máx {s['E_max']})")
        print(f"  fração de arestas positivas: {100 * s['pos_frac']:.2f}%")
        print(f"  cobertura do tour no k-NN:   {100 * s['coverage']:.3f}%  "
              f"({s['incomplete']} grafos com alguma aresta do tour de fora)")
        print(f"  distância média: arestas do grafo {s['graph_edge_len']:.4f}  |  "
              f"arestas do tour {s['tour_edge_len']:.4f}")


if __name__ == "__main__":
    main()
