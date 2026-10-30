"""Tests for the window logs with a variable window size (tsp_gnn.wsize_log). Run with: pytest tests/"""

import numpy as np
import torch

from tsp_gnn import learned_triage as L
from tsp_gnn import wsize_log as W
from tsp_gnn.hier import Heatmaps
from tsp_gnn.models import build_model
from tsp_gnn.window import initial_tour, lkh_path, path_length


def test_window_cities_any_width():
    tour = np.arange(120)
    for w, sizes in ((50, [50, 50, 20, 50, 50, 20]), (100, [100, 20, 100, 20]), (20, [20] * 12)):
        wins = W.window_cities(tour, w)
        assert [len(c) for _, _, c in wins] == sizes
        assert [o for o, _, _ in wins] == [0] * (len(sizes) // 2) + [w // 2] * (len(sizes) // 2)
        for _, s, c in wins:                                   # consecutive tour cities, wrapping around
            assert c.tolist() == [(s + j) % 120 for j in range(len(c))]
    # a last piece shorter than 4 cities is dropped
    assert [len(c) for o, _, c in W.window_cities(np.arange(102), 50) if o == 0] == [50, 50]


def test_lkh_log_and_labels():
    rng = np.random.default_rng(0)
    coords = rng.random((130, 2))
    tour = initial_tour(coords, 20)
    log = W.lkh_log(coords, tour, 50)
    assert len(log["start"]) == len(W.window_cities(tour, 50))
    for j, (_, s, c) in enumerate(W.window_cities(tour, 50)):
        pts = coords[c]
        assert log["cur_len"][j] == path_length(pts)
        assert log["lkh_len"][j] == path_length(pts[lkh_path(pts)])   # LKH (seeded) is deterministic
        assert log["lkh_len"][j] <= log["cur_len"][j] + 1e-9
    improves, gain = L.labels(log["cur_len"], log["lkh_len"])
    assert ((gain > 0) == improves).all() or (gain[improves] > 0).all()


def test_seed_features_match_per_window():
    torch.manual_seed(0)
    model = build_model({"model": "tagcn", "hidden": 16, "layers": 2, "tag_k": 3}).eval()
    device = torch.device("cpu")
    rng = np.random.default_rng(1)
    coords = rng.random((230, 2))
    tour = initial_tour(coords, 20)
    wins = W.window_cities(tour, 100)
    log = {"start": np.array([s for _, s, _ in wins], float), "size": np.array([len(c) for _, _, c in wins], float)}
    got = W.seed_features(model, device, coords, tour, log)
    heats = Heatmaps(model, device)([coords[c] for _, _, c in wins])
    ctx = L.instance_context(coords)
    for j, ((_, s, c), h) in enumerate(zip(wins, heats)):
        ref = L.window_features(ctx, coords, c, h, s)
        for f in L.FEATURES:
            assert abs(got[f][j] - ref[f]) < 1e-6, (f, j)
