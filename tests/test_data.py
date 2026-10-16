"""Testes rápidos para src/tsp_gnn/data.py.  Correr com: pytest tests/"""

import itertools

import numpy as np

from tsp_gnn.data import (generate_instances, is_valid_tour, load_dataset, save_dataset,
                          solve_lkh, tour_length)


def test_tour_length_unit_square():
    # Os 4 cantos do quadrado unitário, percorridos pela ordem: perímetro = 4.
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert np.isclose(tour_length(square, np.array([0, 1, 2, 3])), 4.0)


def test_lkh_tours_are_permutations():
    for coords in generate_instances(num=10, n=20, seed=0):
        tour = solve_lkh(coords)
        assert is_valid_tour(tour, 20)
        assert tour[0] == 0


def test_save_load_roundtrip(tmp_path):
    coords = generate_instances(num=3, n=10, seed=1)
    tours = np.stack([solve_lkh(c) for c in coords])
    lengths = np.array([tour_length(c, t) for c, t in zip(coords, tours)])
    path = tmp_path / "ds.npz"
    save_dataset(path, coords, tours, lengths, n=10, seed=1)

    data = load_dataset(path)
    assert np.array_equal(data["coords"], coords)
    assert np.array_equal(data["tours"], tours)
    assert np.array_equal(data["lengths"], lengths)
    assert data["n"] == 10 and data["seed"] == 1


def brute_force_optimum(coords):
    """Ótimo exato: experimenta todas as ordens (fixando a cidade 0 no início)."""
    n = len(coords)
    return min(tour_length(coords, np.array((0,) + perm))
               for perm in itertools.permutations(range(1, n)))


def test_lkh_matches_brute_force():
    # 7 cidades -> 6! = 720 tours a testar por instância (instantâneo).
    for coords in generate_instances(num=5, n=7, seed=42):
        lkh = tour_length(coords, solve_lkh(coords))
        # Tolerância pequena: o LKH otimiza distâncias arredondadas a 1e-6.
        assert abs(lkh - brute_force_optimum(coords)) < 1e-5
