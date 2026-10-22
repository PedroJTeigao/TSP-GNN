"""Classical TSP heuristics (no learning), used as baselines.

Both return a tour as an (n,) array that starts at city 0.
"""

import networkx as nx
import numpy as np

from tsp_gnn.data import distance_matrix
from tsp_gnn.decode import greedy_decode, verify_tour


def greedy_distance_tour(coords):
    """Classical greedy ("greedy edge"): accepts edges by distance only.

    Reuses greedy_decode with the k-NN mask all False: every pair goes into the list
    sorted by distance, and the heatmap (all zeros) plays no role.
    """
    n = len(coords)
    tour = greedy_decode(np.zeros((n, n)), np.zeros((n, n), dtype=bool), coords)
    return verify_tour(tour, n)


def christofides_tour(coords):
    """Christofides heuristic (networkx): guaranteed to be <= 1.5x the optimum."""
    n = len(coords)
    dist = distance_matrix(coords)
    i, j = np.triu_indices(n, k=1)                     # all pairs i < j
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(i.tolist(), j.tolist(), dist[i, j].tolist()))
    cycle = nx.approximation.christofides(graph, weight="weight")
    tour = np.array(cycle[:-1])                        # networkx repeats the first city at the end
    tour = np.roll(tour, -int(np.where(tour == 0)[0][0]))  # start at city 0
    return verify_tour(tour, n)
