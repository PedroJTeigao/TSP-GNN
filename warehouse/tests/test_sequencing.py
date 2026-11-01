"""Layer 1: the three solvers and the dummy-city trick, against Held-Karp."""

import numpy as np
import pytest

from warehouse import env as W
from warehouse import sequencing as S


@pytest.fixture(scope="module")
def wh():
    return W.Warehouse(W.SIZES["small"])


def problem(wh, rng, m):
    cells = rng.choice(wh.free, m + 2, replace=False).tolist()
    return S.matrix(wh, cells[0], cells[1:-1], cells[-1])


def is_order(order, m):
    return sorted(order) == list(range(1, m + 1))


def test_lkh_is_optimal_and_keeps_the_ends(wh):
    rng = np.random.default_rng(0)
    for m in (4, 5, 6, 8, 10, 12):
        for _ in range(5):
            d = problem(wh, rng, m)
            order = S.lkh(d)
            assert is_order(order, m)
            assert S.path_length(d, [0] + order + [m + 1]) == S.path_length(d, [0] + S.exact(d) + [m + 1])


def test_heuristics_are_valid_and_two_opt_improves(wh):
    rng = np.random.default_rng(1)
    for m in (1, 2, 3, 7, 12):
        for _ in range(5):
            d = problem(wh, rng, m)
            a, b, e = S.nearest(d), S.two_opt(d), S.exact(d)
            assert is_order(a, m) and is_order(b, m) and is_order(e, m)
            la, lb, le = (S.path_length(d, [0] + x + [m + 1]) for x in (a, b, e))
            assert le <= lb <= la


def test_exact_matches_brute_force(wh):
    import itertools
    rng = np.random.default_rng(2)
    d = problem(wh, rng, 6)
    best = min(S.path_length(d, [0, *p, 7]) for p in itertools.permutations(range(1, 7)))
    assert S.path_length(d, [0] + S.exact(d) + [7]) == best


def test_sequence_cache(wh):
    cache = {}
    stops = [wh.access[k] for k in (0, 7, 20)]
    a, _ = S.sequence(wh, wh.parking[0], stops, wh.stations[0], "2opt", cache)
    b, sec = S.sequence(wh, wh.parking[0], stops, wh.stations[0], "2opt", cache)
    assert a == b and sec == 0.0


def test_assignment_takes_the_closest_free_robot(wh):
    target = wh.access[0]
    near = wh.neighbours[target][0]
    far = wh.stations[-1]
    assert S.assign_robot(wh, [0, 1], [far, near], [target]) == 1
    assert S.assign_robot(wh, [0], [far, near], [target]) == 0
