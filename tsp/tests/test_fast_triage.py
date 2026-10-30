"""Equivalence tests for the batched window scoring (tsp_gnn.fast_triage) against the per-window version
(hier.Heatmaps, triage.segment_disagreement, learned_triage.window_features) and scale.knn_graph.
Run with: pytest tests/"""

import numpy as np
import pytest
import torch

from tsp_gnn import fast_triage as F
from tsp_gnn import learned_triage as L
from tsp_gnn.hier import Heatmaps
from tsp_gnn.models import build_model
from tsp_gnn.scale import edge_pairs, knn_graph
from tsp_gnn.triage import ranked_refine, segment_disagreement
from tsp_gnn.window import initial_tour, window_starts


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    model = build_model({"model": "tagcn", "hidden": 16, "layers": 2, "tag_k": 3}).eval()
    return model, torch.device("cpu")


class Dummy:                                     # "probability" = p_mean − 0.1 · out_knn5
    def predict_proba(self, X):
        v = X[:, 0] - 0.1 * X[:, 1]
        return np.stack([1 - v, v], axis=1)


def test_batch_heat_equals_per_window(net):
    model, device = net
    rng = np.random.default_rng(1)
    heatmaps = Heatmaps(model, device)
    for m in (20, 7):
        pts = rng.random((30, m, 2)) * rng.random() * 3       # any scale: windows are rescaled
        heat, _ = F.batch_heat(model, device, pts, max_windows=11)          # also across chunks
        ref = heatmaps(list(pts))
        assert np.abs(heat - np.stack(ref)).max() < 1e-5


def test_window_scorer_matches_per_window(net):
    model, device = net
    rng = np.random.default_rng(2)
    coords = rng.random((300, 2))
    tour = initial_tour(coords, 20)
    heatmaps = Heatmaps(model, device)
    bundle = {"model": Dummy(), "kind": "hgb", "features": ("p_mean", "out_knn5")}
    ctx = L.instance_context(coords)
    for offset in (0, 10):
        wins = window_starts(len(tour), 20, offset)
        idx = [(s + np.arange(m)) % len(tour) for s, m in wins]
        heats = heatmaps([coords[tour[i]] for i in idx])
        dis = np.array([segment_disagreement(h) for h in heats])
        sc = F.WindowBatchScorer(model, device)
        sc.prepare(coords)
        assert np.abs(sc.values(coords, tour, wins) - dis).max() < 1e-6
        sl = F.WindowBatchScorer(model, device, bundle)
        sl.prepare(coords)
        got = sl.values(coords, tour, wins)
        ref = []
        for i, h, (s, _) in zip(idx, heats, wins):
            f = L.window_features(ctx, coords, tour[i], h, s)
            ref.append(f["p_mean"] - 0.1 * f["out_knn5"])
        assert np.abs(got - np.array(ref)).max() < 1e-6
        # unchanged windows are not scored again
        before = sl.net_seconds
        assert np.array_equal(sl.values(coords, tour, wins), got) and sl.net_seconds == before


def test_batch_features_match(net):
    rng = np.random.default_rng(3)
    coords = rng.random((80, 2))
    ctx = L.instance_context(coords)
    cities = np.stack([rng.permutation(80)[:20] for _ in range(5)])
    heat = rng.random((5, 20, 20))
    heat = (heat + heat.transpose(0, 2, 1)) / 2
    p_seg = heat[:, np.arange(19), np.arange(1, 20)]
    got = F.batch_features(ctx, coords, cities, p_seg, [3, 5, 7, 9, 11])
    for w in range(5):
        ref = L.window_features(ctx, coords, cities[w], heat[w], [3, 5, 7, 9, 11][w])
        for f in L.FEATURES:
            assert got[f][w] == pytest.approx(ref[f])


def test_global_graph_and_lookup(net):
    model, device = net
    rng = np.random.default_rng(4)
    coords = rng.random((200, 2))
    g, keys = F.global_knn_graph(coords, 30)
    ref = knn_graph(coords, 30)
    assert torch.equal(g.edge_index, ref.edge_index) and torch.allclose(g.edge_attr, ref.edge_attr)
    sc = F.GlobalScorer(model, device, k_max=30, chunk=1000)                # head in several chunks
    sc.prepare(coords)
    with torch.no_grad():
        p = torch.sigmoid(model(ref))
    pi, pj, prob = edge_pairs(ref.edge_index, p, 200)
    assert np.abs(sc.edge_p(pi, pj) - prob).max() < 1e-6 and sc.missing == [0, len(pi)]
    # a pair far apart is outside the 30-NN graph: p = 0, counted as missing
    far = np.argmax(((coords - coords[0]) ** 2).sum(1))
    assert sc.edge_p(np.array([0]), np.array([far]))[0] == 0.0 and sc.missing[0] == 1


def test_ranked_refine_with_scorer(net):
    model, device = net
    rng = np.random.default_rng(5)
    coords = rng.random((120, 2))
    tour = initial_tour(coords, 20)
    heatmaps = Heatmaps(model, device)
    t_ref, st_ref = ranked_refine(coords, tour, 20, "disagreement", 0.2, heatmaps=heatmaps)
    t_new, st_new = ranked_refine(coords, tour, 20, "scorer", 0.2, scorer=F.WindowBatchScorer(model, device))
    assert np.array_equal(t_ref, t_new) and st_ref["to_lkh"] == st_new["to_lkh"]
