"""Quick tests for src/tsp_gnn/data.py.  Run with: pytest tests/"""

import itertools

import numpy as np

from tsp_gnn.data import (generate_instances, is_valid_tour, load_dataset, save_dataset,
                          solve_lkh, tour_length)


def test_tour_length_unit_square():
    # The 4 corners of the unit square, visited in order: perimeter = 4.
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
    """Exact optimum: tries every order (with city 0 fixed at the start)."""
    n = len(coords)
    return min(tour_length(coords, np.array((0,) + perm))
               for perm in itertools.permutations(range(1, n)))


def test_lkh_matches_brute_force():
    # 7 cities -> 6! = 720 tours to try per instance (instant).
    for coords in generate_instances(num=5, n=7, seed=42):
        lkh = tour_length(coords, solve_lkh(coords))
        # Small tolerance: LKH optimizes distances rounded to 1e-6.
        assert abs(lkh - brute_force_optimum(coords)) < 1e-5


def test_solve_lkh_coords_matches_matrix_version():
    from tsp_gnn.data import solve_lkh_coords
    for seed in range(3):
        coords = generate_instances(1, 30, seed)[0]
        tour = solve_lkh_coords(coords)
        assert is_valid_tour(tour, 30) and tour[0] == 0
        assert np.isclose(tour_length(coords, tour), tour_length(coords, solve_lkh(coords)), rtol=1e-6)
