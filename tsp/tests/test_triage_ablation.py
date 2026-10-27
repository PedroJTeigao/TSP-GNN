"""Tests for the E2 ablation in src/tsp_gnn/triage.py: simple rule, oracle and ranked refinement."""

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, is_valid_tour, tour_length
from tsp_gnn.hier import Heatmaps
from tsp_gnn.triage import (nn_distances, ranked_refine, simple_rule, top_fraction, triage_refine,
                            window_log)
from tsp_gnn.window import initial_tour

from test_triage import ShortEdges


@pytest.fixture(scope="module")
def setup():
    coords = generate_instances(1, 300, 5)[0]
    return coords, initial_tour(coords, 50), Heatmaps(ShortEdges(), torch.device("cpu"))


def test_nn_distances():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert np.allclose(nn_distances(coords), [1.0, 1.0, 2.0, 4.0])


def test_simple_rule_by_hand():
    # Path (0,0) -> (3,0) -> (1,0): length 3 + 2 = 5; nearest-neighbour distances of the three cities
    # (within these points): 1, 2, 1 -> sum 4 -> rule = 5 / 4.
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
    assert np.isclose(simple_rule(pts, np.array([1.0, 2.0, 1.0])), 1.25)
    # The same cities visited in the good order: 1 + 2 = 3 -> 3 / 4; a detour scores higher.
    assert simple_rule(pts[[0, 2, 1]], np.array([1.0, 1.0, 2.0])) < simple_rule(pts, np.array([1.0, 2.0, 1.0]))


def test_simple_rule_scale_free():
    pts = np.random.default_rng(0).random((20, 2))
    nn = np.random.default_rng(1).random(20)
    assert np.isclose(simple_rule(pts, nn), simple_rule(7 * pts, 7 * nn))


def test_top_fraction():
    mask = top_fraction([0.1, 0.9, 0.5, 0.7, 0.3], 0.4)
    assert mask.tolist() == [False, True, False, True, False]
    assert top_fraction([1.0, 2.0], 0.01).sum() == 1                  # at least one window
    assert top_fraction([3.0, 3.0, 3.0], 0.34).tolist() == [True, False, False]   # stable on ties
    assert top_fraction(np.arange(10), 1.0).all()


def test_oracle_picks_the_largest_lkh_gain(setup):
    # First half-pass of the oracle = the windows of window_log (offset 0) with the largest real LKH gain.
    coords, tour, heatmaps = setup
    log = window_log(coords, tour, 20, heatmaps)
    first = log["offset"] == 0
    gains = (log["cur_len"] - log["lkh_len"])[first]
    _, stats = ranked_refine(coords, tour, 20, "oracle", 1 / len(gains), max_passes=1)
    assert stats["to_lkh"] == 2                                         # one window in each half-pass
    # The best window of the first half-pass alone already shortens the tour by max(gains).
    assert gains.max() > 0
    new, _ = ranked_refine(coords, tour, 20, "oracle", 1 / len(gains), max_passes=1)
    assert tour_length(coords, new) <= tour_length(coords, tour) - gains.max() + 1e-9


def test_oracle_everywhere_equals_lkh_everywhere(setup):
    coords, tour, _ = setup
    a, sa = ranked_refine(coords, tour, 20, "oracle", 1.0)
    b, sb = triage_refine(coords, tour, 20, {"kind": "lkh"})
    assert np.array_equal(a, b) and sa["improved"] == sb["improved"]


@pytest.mark.parametrize("score", ["random", "simple", "disagreement", "oracle"])
def test_ranked_refine(setup, score):
    coords, tour, heatmaps = setup
    new, stats = ranked_refine(coords, tour, 20, score, 0.2, heatmaps=heatmaps,
                               rng=np.random.default_rng(0))
    assert is_valid_tour(new, len(coords))
    assert tour_length(coords, new) <= tour_length(coords, tour) + 1e-9
    per_pass = stats["windows"] / stats["passes"]
    assert stats["to_lkh"] == pytest.approx(0.2 * stats["windows"], abs=2 * stats["passes"])
    assert stats["improved"] <= stats["to_lkh"] and per_pass == 30
