"""Testes rápidos para src/tsp_gnn/baselines.py.  Correr com: pytest tests/"""

import numpy as np
import pytest

from tsp_gnn.baselines import christofides_tour, greedy_distance_tour
from tsp_gnn.data import generate_instances, is_valid_tour, solve_lkh, tour_length

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


@pytest.mark.parametrize("heuristic", [greedy_distance_tour, christofides_tour])
def test_square_gives_perimeter(heuristic):
    assert np.isclose(tour_length(SQUARE, heuristic(SQUARE)), 4.0)


@pytest.mark.parametrize("heuristic", [greedy_distance_tour, christofides_tour])
def test_valid_tours(heuristic):
    for coords in generate_instances(num=5, n=20, seed=0):
        tour = heuristic(coords)
        assert is_valid_tour(tour, 20) and tour[0] == 0


def test_christofides_within_1_5_of_optimum():
    # Garantia teórica do Christofides em instâncias métricas: <= 1,5x o ótimo.
    for coords in generate_instances(num=20, n=20, seed=5):
        opt = tour_length(coords, solve_lkh(coords))
        assert tour_length(coords, christofides_tour(coords)) <= 1.5 * opt + 1e-9
