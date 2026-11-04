"""SoftDist heatmaps (Xia et al. 2024): a heatmap made from distances alone, as a control for the network (ablation A4).

On each window, rescaled to [0, 1]² and with the same k-NN graph as the network (k = round(0.5 m), both directions), row i
is a softmax of −d_ij / τ over the neighbours j of i; the heatmap is (S + Sᵀ) / 2, so it is symmetric and 0 outside the
graph, like the network's (p_ij + p_ji) / 2. Nothing is learned: τ is the only parameter, chosen by a grid search.

    SoftDistHeatmaps(tau)        drop-in for hier.Heatmaps (window.make_solver, triage.ranked_refine)
    SoftDistBatchScorer(tau, learned=None)
                                 drop-in for fast_triage.WindowBatchScorer: disagreement, or a learned model on the same
                                 features with p from SoftDist
"""

import time

import numpy as np

from tsp_gnn.fast_triage import WindowBatchScorer, batch_features, rescale_batch

P_FLOOR = 1e-12        # a graph pair keeps p > 0 even if exp underflows, so the decoders still see it as a graph pair


def batch_softdist(pts, tau):
    """(W, m, m) SoftDist heatmaps of W windows of m points each (in any order; only the positions matter)."""
    W, m, _ = pts.shape
    if m < 3:
        return np.zeros((W, m, m))
    k = min(m - 1, round(0.5 * m))
    pos = rescale_batch(pts).astype(np.float32).astype(np.float64)      # as fast_triage.batch_knn_graph
    d = np.sqrt(((pos[:, :, None, :] - pos[:, None, :, :]) ** 2).sum(-1))
    d[:, np.arange(m), np.arange(m)] = np.inf
    nbr = np.argsort(d, axis=2, kind="stable")[:, :, :k]
    adj = np.zeros((W, m, m), dtype=bool)
    adj[np.arange(W)[:, None, None], np.arange(m)[None, :, None], nbr] = True
    adj |= adj.transpose(0, 2, 1)
    with np.errstate(invalid="ignore"):          # tau = inf: inf / inf on the diagonal, which is masked out
        logits = np.where(adj, -d / tau, -np.inf)
    logits -= logits.max(axis=2, keepdims=True)
    s = np.exp(logits)
    s /= s.sum(axis=2, keepdims=True)
    s = np.where(adj, np.maximum(s, P_FLOOR), 0.0)
    return (s + s.transpose(0, 2, 1)) / 2


class SoftDistHeatmaps:
    """hier.Heatmaps with SoftDist instead of the network: a list of (m, m) matrices for a list of point sets."""

    def __init__(self, tau):
        self.tau = tau
        self.seconds = 0.0          # time to build the heatmaps (no network)
        self.device = "cpu"

    def __call__(self, point_sets):
        start = time.perf_counter()
        out = [None] * len(point_sets)
        by_size = {}
        for i, pts in enumerate(point_sets):
            by_size.setdefault(len(pts), []).append(i)
        for m, idx in by_size.items():
            heats = batch_softdist(np.stack([point_sets[i] for i in idx]), self.tau)
            for i, h in zip(idx, heats):
                out[i] = h
        self.seconds += time.perf_counter() - start
        return out


class SoftDistBatchScorer(WindowBatchScorer):
    """WindowBatchScorer with SoftDist heatmaps; net_seconds counts the time to build them."""

    def __init__(self, tau, learned=None):
        super().__init__(None, None, learned)
        self.tau = tau

    def values(self, coords, tour, wins):
        out = np.zeros(len(wins))
        todo = {}
        for k, (s, length) in enumerate(wins):
            cities = tour[(s + np.arange(length)) % len(tour)]
            key = (s, cities.tobytes())
            if key in self.cache:
                out[k] = self.cache[key]
            else:
                todo.setdefault(length, []).append((k, s, cities, key))
        for length, items in todo.items():
            cities = np.stack([c for _, _, c, _ in items])
            start = time.perf_counter()
            heat = batch_softdist(coords[cities], self.tau)
            self.net_seconds += time.perf_counter() - start
            p_seg = heat[:, np.arange(length - 1), np.arange(1, length)]
            if self.learned is None:
                vals = 1.0 - p_seg.mean(axis=1)
            else:
                vals = self._score(batch_features(self.ctx, coords, cities, p_seg, [s for _, s, _, _ in items]))
            for (k, _, _, key), v in zip(items, vals):
                out[k] = self.cache[key] = v
        return out
