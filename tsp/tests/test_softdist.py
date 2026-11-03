"""Ablation A4 (2026-11-03): the SoftDist heatmap must look like the network's to the decoders and the scorers:
symmetric, zero outside the window's k-NN graph, positive on it, and the batched scorer must agree with the per-window one."""

import numpy as np

from tsp_gnn.fast_triage import batch_knn_graph
from tsp_gnn.softdist import SoftDistBatchScorer, SoftDistHeatmaps, batch_softdist
from tsp_gnn.triage import ranked_refine, segment_disagreement
from tsp_gnn.window import initial_tour, window_starts


def windows(num=6, m=20, seed=0):
    return np.random.default_rng(seed).random((num, m, 2))


def test_symmetric_and_on_the_knn_graph():
    pts = windows()
    heat = batch_softdist(pts, 0.05)
    assert np.allclose(heat, heat.transpose(0, 2, 1))
    _, (w, s, t) = batch_knn_graph(pts)                          # the network's graph
    graph = np.zeros(heat.shape, dtype=bool)
    graph[w, s, t] = True
    assert np.all(heat[graph] > 0) and np.all(heat[~graph] == 0)
    assert np.allclose(heat.sum(axis=2).mean(), 1.0)             # rows of S sum to 1, so (S + S^T)/2 on average too


def test_small_tau_prefers_the_nearest_neighbour():
    pts = windows(1, 20, 3)[0]
    heat = batch_softdist(pts[None], 1e-3)[0]
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nearest = d.argmin(axis=1)
    # S_i,nn(i) -> 1 as tau -> 0 (unless a second neighbour is almost as close), so (S + S^T)/2 is about 1/2 or more
    assert np.all(heat[np.arange(20), nearest] >= 0.45)


def test_heatmaps_match_batch_and_rescale():
    pts = windows(3, 20, 5)
    per = SoftDistHeatmaps(0.1)([p * 7 + 3 for p in pts])       # translation and scale do not matter
    assert all(np.allclose(a, b) for a, b in zip(per, batch_softdist(pts, 0.1)))


def test_scorer_disagreement_matches_per_window():
    rng = np.random.default_rng(1)
    coords = rng.random((200, 2)).astype(np.float32).astype(np.float64)
    tour = initial_tour(coords, 100)
    wins = window_starts(len(tour), 20, 0)
    sc = SoftDistBatchScorer(0.05)
    sc.prepare(coords)
    batched = sc.values(coords, tour, wins)
    heats = SoftDistHeatmaps(0.05)([coords[tour[(s + np.arange(m)) % len(tour)]] for s, m in wins])
    assert np.allclose(batched, [segment_disagreement(h) for h in heats])
    t, st = ranked_refine(coords, tour, 20, "scorer", 0.2, scorer=SoftDistBatchScorer(0.05))
    assert sorted(t.tolist()) == list(range(200))
