"""TSP instance generation and labels from LKH (via elkai).

An instance is an (n, 2) array of coordinates in [0,1]².
A tour is a permutation of 0..n-1 that starts at city 0; the closing edge
(last city -> first) is implicit.
"""

import elkai
import numpy as np

# LKH only accepts integer distances: we multiply by this factor before
# rounding. With 1e6 the rounding error is ~1e-6 per edge (negligible).
SCALE = 1_000_000


def generate_instances(num, n, seed):
    """Generate `num` instances with `n` cities uniform in [0,1]². Shape: (num, n, 2)."""
    rng = np.random.default_rng(seed)  # same seed -> same data
    return rng.random((num, n, 2))


def distance_matrix(coords):
    """(n, n) matrix of Euclidean distances between all cities."""
    diff = coords[:, None, :] - coords[None, :, :]  # (n, n, 2): difference for each pair
    return np.sqrt((diff ** 2).sum(axis=-1))


def solve_lkh(coords, runs=10):
    """Solve one instance with LKH. Returns the tour as an (n,) integer array."""
    dist_int = np.rint(distance_matrix(coords) * SCALE).astype(np.int64)
    tour = elkai.solve_int_matrix(dist_int.tolist(), runs=runs)
    tour = np.asarray(tour, dtype=np.int64)
    # Make sure the tour starts at city 0 (rotating does not change the cycle).
    start = int(np.where(tour == 0)[0][0])
    return np.roll(tour, -start)


def solve_lkh_coords(coords, runs=10):
    """Solve one instance with LKH from its coordinates (no (n, n) matrix, for large n).

    Same problem as solve_lkh: the coordinates are multiplied by SCALE and LKH (EUC_2D) rounds each
    distance to the nearest integer. Returns the tour as an (n,) integer array that starts at city 0.
    """
    from elkai import _elkai   # the low-level call that elkai's own classes use

    xy = np.rint(np.asarray(coords) * SCALE).astype(np.int64)
    problem = (f"TYPE : TSP\nDIMENSION : {len(xy)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
               + "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(xy.tolist())))
    tour = np.asarray(_elkai.solve_problem(f"RUNS = {runs}\nPROBLEM_FILE = :stdin:\n", problem), dtype=np.int64) - 1
    start = int(np.where(tour == 0)[0][0])
    return np.roll(tour, -start)


def tour_length(coords, tour):
    """Real tour length (float distances), including the closing edge."""
    ordered = coords[tour]                    # cities in tour order
    nxt = np.roll(ordered, -1, axis=0)        # the next one for each (the last links to the first)
    return float(np.sqrt(((ordered - nxt) ** 2).sum(axis=-1)).sum())


def is_valid_tour(tour, n):
    """True if the tour visits each of the n cities exactly once."""
    return len(tour) == n and np.array_equal(np.sort(tour), np.arange(n))


def save_dataset(path, coords, tours, lengths, **meta):
    """Save coordinates, tours and lengths to a compressed .npz (+ metadata, e.g. n, seed)."""
    np.savez_compressed(path, coords=coords, tours=tours, lengths=lengths, **meta)


def load_dataset(path):
    """Read an .npz written by save_dataset and return a dictionary of arrays."""
    with np.load(path) as f:
        return {key: f[key] for key in f.files}
