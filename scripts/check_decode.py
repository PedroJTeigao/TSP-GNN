"""Teste de sanidade da decodificação: usa os rótulos verdadeiros como probabilidades.

Com a "rede perfeita", os grafos com cobertura total têm de dar o tour do LKH (gap 0%).
Exemplo:
    python scripts/check_decode.py --data data/tsp20_test.npz data/tsp50_test.npz
"""

import argparse
import time

import numpy as np

from tsp_gnn.data import load_dataset, tour_length
from tsp_gnn.decode import (edge_heatmap, greedy_decode, optimality_gap, tour_edges,
                            two_opt, verify_tour)
from tsp_gnn.graph import instance_to_data


def check(path, k):
    """Decodifica cada instância de `path` com probs = y e devolve uma linha por instância."""
    d = load_dataset(path)
    rows = []
    for coords, lkh_tour, opt_len in zip(d["coords"], d["tours"], d["lengths"]):
        n = len(coords)
        g = instance_to_data(coords, lkh_tour, opt_len, k)
        start = time.perf_counter()
        heat, in_graph = edge_heatmap(g.edge_index, g.y, n)   # probs = rótulos verdadeiros
        greedy = verify_tour(greedy_decode(heat, in_graph, coords), n)
        improved = verify_tour(two_opt(coords, greedy), n)
        elapsed = time.perf_counter() - start
        rows.append({
            "full": int(g.y.sum()) == 2 * n,                  # cobertura total do tour no k-NN
            "gap_greedy": optimality_gap(tour_length(coords, greedy), opt_len),
            "gap_2opt": optimality_gap(tour_length(coords, improved), opt_len),
            "same_greedy": tour_edges(greedy) == tour_edges(lkh_tour),  # mesmas arestas que o LKH
            "same_2opt": tour_edges(improved) == tour_edges(lkh_tour),
            "ms": 1000 * elapsed,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sanidade da decodificação com rótulos verdadeiros.")
    parser.add_argument("--data", nargs="+", required=True, help="ficheiros .npz de make_dataset.py")
    parser.add_argument("--k", type=int, default=10, help="número de vizinhos do grafo k-NN")
    args = parser.parse_args()

    header = (f"{'grupo':<12}{'grafos':>7}  {'gap gulosa (méd / máx)':>24}  "
              f"{'gap +2-opt (méd / máx)':>24}  {'igual LKH (gul / 2opt)':>23}  {'ms/inst':>8}")
    for path in args.data:
        rows = check(path, args.k)
        print(f"=== {path} ===")
        print(header)
        for name, full in [("cob. total", True), ("incompleta", False)]:
            group = [r for r in rows if r["full"] == full]
            if not group:
                print(f"{name:<12}{0:>7}")
                continue
            col = lambda key: np.array([r[key] for r in group])
            print(f"{name:<12}{len(group):>7}  "
                  f"{col('gap_greedy').mean():>10.4f}% / {col('gap_greedy').max():>8.4f}%  "
                  f"{col('gap_2opt').mean():>10.4f}% / {col('gap_2opt').max():>8.4f}%  "
                  f"{col('same_greedy').sum():>10} / {col('same_2opt').sum():<10}  "
                  f"{col('ms').mean():>8.2f}")
        print()


if __name__ == "__main__":
    main()
