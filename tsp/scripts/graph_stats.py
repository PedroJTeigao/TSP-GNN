"""Statistics of the k-NN graphs of one or more datasets.

Example:
    python scripts/graph_stats.py --data data/tsp20_test.npz data/tsp50_test.npz --k 10
    python scripts/graph_stats.py --data data/tsp20_test.npz data/tsp50_test.npz --edge-norm knn_mean
"""

import argparse

import torch

from tsp_gnn.graph import EDGE_NORMS, load_graphs


def stats(graphs):
    """Compute the statistics of a list of graphs and return a dictionary."""
    num_edges = torch.tensor([g.num_edges for g in graphs], dtype=torch.float)
    num_pos = torch.tensor([g.y.sum().item() for g in graphs])
    n = torch.tensor([g.num_nodes for g in graphs], dtype=torch.float)
    covered = num_pos / 2                       # each tour edge appears twice (both directions)

    all_dists = torch.cat([g.edge_attr.view(-1) for g in graphs])
    tour_edge_len = torch.cat([g.opt_len for g in graphs]) / n   # mean length of one tour edge

    return {
        "n": int(n[0]),
        "E_mean": num_edges.mean().item(), "E_min": int(num_edges.min()), "E_max": int(num_edges.max()),
        "pos_frac": (num_pos.sum() / num_edges.sum()).item(),
        "coverage": (covered.sum() / n.sum()).item(),
        "incomplete": int((covered < n).sum()),  # graphs with some tour edge left out
        "graph_edge_len": all_dists.mean().item(),
        "tour_edge_len": tour_edge_len.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Statistics of the k-NN graphs.")
    parser.add_argument("--data", nargs="+", required=True, help=".npz files from make_dataset.py")
    parser.add_argument("--k", type=int, default=10, help="number of neighbours in the k-NN graph")
    parser.add_argument("--edge-norm", choices=EDGE_NORMS, default="none")
    args = parser.parse_args()

    for path in args.data:
        graphs = load_graphs(path, args.k, args.edge_norm)
        s = stats(graphs)
        print(f"=== {path}  ({len(graphs)} graphs, n={s['n']}, k={args.k}, edge_norm={args.edge_norm}) ===")
        print(f"  edges per graph:            mean {s['E_mean']:.1f}  (min {s['E_min']}, max {s['E_max']})")
        print(f"  fraction of positive edges: {100 * s['pos_frac']:.2f}%")
        print(f"  tour coverage in the k-NN:  {100 * s['coverage']:.3f}%  "
              f"({s['incomplete']} graphs with some tour edge left out)")
        print(f"  mean distance: graph edges {s['graph_edge_len']:.4f}  |  "
              f"tour edges {s['tour_edge_len']:.4f} (raw, never normalized)")


if __name__ == "__main__":
    main()
