"""Tests for the learned triage features, labels and instance split (tsp_gnn.learned_triage).
Run with: pytest tests/"""

import numpy as np
import pytest

from tsp_gnn import learned_triage as L


def test_window_features_by_hand():
    # 12 cities on a line, x = 0..11: every nearest-neighbour distance is 1
    coords = np.stack([np.arange(12.0), np.zeros(12)], axis=1)
    ctx = L.instance_context(coords)
    assert ctx["mean_nn"] == pytest.approx(1.0)
    cities = np.array([0, 1, 11])                       # edges 0-1 (length 1) and 1-11 (length 10)
    heat = np.zeros((3, 3))
    heat[0, 1] = heat[1, 0] = 0.8
    heat[1, 2] = heat[2, 1] = 0.2
    f = L.window_features(ctx, coords, cities, heat, start=3)
    assert f["p_mean"] == pytest.approx(0.5) and f["p_min"] == pytest.approx(0.2) and f["p_max"] == pytest.approx(0.8)
    assert f["disagreement"] == pytest.approx(0.5)
    assert f["seg_len_nn"] == pytest.approx(11 / 3)     # length 11 / three nearest-neighbour distances of 1
    assert f["max_edge_nn"] == pytest.approx(10.0)
    assert f["density"] == pytest.approx((2 + 3 + 2) / 3)   # cities within 2: {1,2}, {0,2,3}, {9,10}
    assert f["position"] == pytest.approx(3 / 12)
    assert f["out_knn5"] == 1                            # 1-11: neither is among the other's 5 nearest
    assert set(f) == set(L.FEATURES)


def test_labels():
    improves, gain = L.labels([10.0, 10.0, 5.0], [9.0, 10.0, 5.0 - 1e-12])
    assert improves.tolist() == [True, False, False]     # below triage.IMPROVE_TOL is not an improvement
    assert gain[0] == pytest.approx(10.0) and gain[1] == 0.0


def test_split_by_instance():
    a, b = L.split_instances(100, seed=0), L.split_instances(100, seed=0)
    assert (a == b).all() and a.sum() == 50               # same split for every TAGCN seed
    assert (L.split_instances(100, seed=1) != a).any()
    # windows of one instance never fall on both sides
    inst = np.repeat(np.arange(100), 20)
    train = a[inst]
    assert set(inst[train]).isdisjoint(set(inst[~train]))
    assert len(set(inst[train])) == 50 and len(set(inst[~train])) == 50
