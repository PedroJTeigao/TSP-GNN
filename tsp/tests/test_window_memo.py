"""Ablation A1 (2026-11-03): the window memo of triage.ranked_refine.

The memo is only correct if LKH on a window is deterministic: the same cities in the same order give the same path.
"cache" must then give exactly the tour of "none" with fewer LKH calls, and the shadow record of "none" must predict
those calls; "exclude" must never send a solved window to LKH again.
"""

import numpy as np
import pytest

from tsp_gnn.triage import ranked_refine, window_key
from tsp_gnn.window import initial_tour, lkh_path


def instance(n=300, seed=0):
    coords = np.random.default_rng(seed).random((n, 2)).astype(np.float32).astype(np.float64)
    return coords, initial_tour(coords, min(n - 1, round(0.5 * n), 100))


def test_lkh_window_is_deterministic():
    coords, tour = instance()
    rng = np.random.default_rng(1)
    for w in (20, 50):
        starts = rng.integers(0, len(tour) - w, 10)
        first = [lkh_path(coords[tour[s:s + w]]) for s in starts]
        for s in rng.integers(0, len(tour) - w, 10):        # other windows in between
            lkh_path(coords[tour[s:s + w]])
        again = [lkh_path(coords[tour[s:s + w]]) for s in starts]
        assert all(np.array_equal(a, b) for a, b in zip(first, again))


def test_window_key_depends_on_order():
    assert window_key([1, 2, 3]) == window_key(np.array([1, 2, 3]))
    assert window_key([1, 2, 3]) != window_key([1, 3, 2])


@pytest.mark.parametrize("score,frac", [("simple", 0.2), ("simple", 1.0), ("random", 0.5)])
def test_cache_gives_the_same_tour_with_fewer_calls(score, frac):
    coords, t0 = instance()
    runs = {}
    for memo in ("none", "cache"):
        rng = np.random.default_rng([0, 0])
        runs[memo] = ranked_refine(coords, t0, 20, score, frac, rng=rng, memo=memo)
    (t_none, s_none), (t_cache, s_cache) = runs["none"], runs["cache"]
    assert np.array_equal(t_none, t_cache)
    assert s_none["to_lkh"] == s_cache["to_lkh"]
    assert s_none["lkh_calls"] == s_none["to_lkh"]                          # no memo: every selected window is solved
    assert s_cache["memo_hits"] == s_none["memo_hits"]                      # the shadow record predicts the hits
    assert s_cache["lkh_calls"] == s_none["lkh_calls"] - s_none["memo_hits"]
    if score == "simple":                                                   # a deterministic score re-selects
        assert s_none["memo_hits"] > 0


def test_exclude_never_resolves_a_window():
    coords, t0 = instance()
    tour, s = ranked_refine(coords, t0, 20, "simple", 0.2, memo="exclude")
    assert s["memo_hits"] == 0 and s["lkh_calls"] == s["to_lkh"]
    assert sorted(tour.tolist()) == list(range(len(t0)))


def test_exclude_all_equals_cache_all():
    """At 100 % every window not yet solved goes to LKH in both modes, so "exclude" and "cache" give the same tour."""
    coords, t0 = instance()
    t_cache, s_cache = ranked_refine(coords, t0, 20, "simple", 1.0, memo="cache")
    t_excl, s_excl = ranked_refine(coords, t0, 20, "simple", 1.0, memo="exclude")
    assert np.array_equal(t_cache, t_excl)
    assert s_excl["lkh_calls"] == s_cache["lkh_calls"]
