"""Heurísticas clássicas para o TSP (sem aprendizagem), usadas como baselines.

Ambas devolvem um tour como array (n,) que começa na cidade 0.
"""

import networkx as nx
import numpy as np

from tsp_gnn.data import distance_matrix
from tsp_gnn.decode import greedy_decode, verify_tour


def greedy_distance_tour(coords):
    """Guloso clássico ("greedy edge"): aceita arestas só por ordem de distância.

    Reutiliza greedy_decode com a máscara do k-NN toda a False: todos os pares vão
    para a lista ordenada por distância, e o heatmap (a zeros) não tem nenhum papel.
    """
    n = len(coords)
    tour = greedy_decode(np.zeros((n, n)), np.zeros((n, n), dtype=bool), coords)
    return verify_tour(tour, n)


def christofides_tour(coords):
    """Heurística de Christofides (networkx): garantidamente <= 1,5x o ótimo."""
    n = len(coords)
    dist = distance_matrix(coords)
    i, j = np.triu_indices(n, k=1)                     # todos os pares i < j
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(i.tolist(), j.tolist(), dist[i, j].tolist()))
    cycle = nx.approximation.christofides(graph, weight="weight")
    tour = np.array(cycle[:-1])                        # o networkx repete a 1.ª cidade no fim
    tour = np.roll(tour, -int(np.where(tour == 0)[0][0]))  # começar na cidade 0
    return verify_tour(tour, n)
