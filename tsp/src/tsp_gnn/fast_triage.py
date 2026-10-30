"""Cheap window scoring for the triage policies (phase 3, triage.ranked_refine).

The per-window version (hier.Heatmaps + triage.segment_disagreement / learned_triage.LearnedScore) builds one PyG
graph per window in Python: about 4 s per 10k instance. Two batched versions:

A. WindowBatchScorer ("fast"): the SAME windows, rescaled to [0, 1]² and k-NN graphs with k = round(0.5 m), but all
   the windows of a half-pass built at once with numpy (one (W, m, m) distance array, one argsort) and sent to the
   network as a single graph; segment probabilities, disagreement and the learned features vectorised. The scores
   are those of the per-window version up to floating point. A window whose cities (and start) did not change since
   it was last scored keeps its score (later passes only rescore what moved).
B. GlobalScorer ("global"): the network runs ONCE on the k-NN graph of the whole instance (k = min(0.5 n, 100), raw
   coordinates, as in the E1 candidates); every window takes its segment probabilities from that global p (an edge
   outside the global graph gets p = 0; the share is recorded). The network was trained on TSP-20, so on the whole
   instance it sees distances ~sqrt(n / 20) times smaller than in a rescaled window: B is a different signal and the
   learned model needs features computed this way.

Both expose prepare(coords) (once per instance) and values(coords, tour, wins) (one half-pass; wins = window_starts)
and keep net_seconds (network only, GPU synchronised).
"""

import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from tsp_gnn.learned_triage import FEATURES, instance_context

# ---------------------------------------------------------------------------------------------------- batched graphs


def rescale_batch(pts):
    """hier.rescale on each window of a (W, m, 2) array: moved to the origin, divided by its largest span."""
    p = pts - pts.min(axis=1, keepdims=True)
    span = p.reshape(len(p), -1).max(axis=1)
    span = np.where(span > 0, span, 1.0)
    return p / span[:, None, None]


def batch_knn_graph(pts):
    """One Data for W windows of m points each, every window as scale.knn_graph(rescale(window), round(0.5 m)) would
    build it: float32 positions, k nearest of each point (Euclidean, on the float32 values), both directions, no
    duplicates, edge_attr = length. Node w * m + i is point i of window w."""
    W, m, _ = pts.shape
    k = min(m - 1, round(0.5 * m))
    pos32 = rescale_batch(pts).astype(np.float32)
    pos = pos32.astype(np.float64)
    d = np.sqrt(((pos[:, :, None, :] - pos[:, None, :, :]) ** 2).sum(-1))
    d[:, np.arange(m), np.arange(m)] = np.inf
    nbr = np.argsort(d, axis=2, kind="stable")[:, :, :k]                 # (W, m, k)
    adj = np.zeros((W, m, m), dtype=bool)
    adj[np.arange(W)[:, None, None], np.arange(m)[None, :, None], nbr] = True
    adj |= adj.transpose(0, 2, 1)
    w, s, t = np.nonzero(adj)                                            # sorted by (window, src, dst)
    src, dst = w * m + s, w * m + t
    x = torch.from_numpy(pos32.reshape(-1, 2))
    edge_index = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
    edge_attr = (x[edge_index[0]] - x[edge_index[1]]).norm(dim=1, keepdim=True)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), (w, s, t)


@torch.no_grad()
def batch_heat(model, device, pts, max_windows=4096):
    """(W, m, m) symmetric probability matrices of the windows (as hier.Heatmaps) and the network time (s)."""
    W, m, _ = pts.shape
    heat = np.zeros((W, m, m))
    net = 0.0
    for lo in range(0, W, max_windows):
        g, (w, s, t) = batch_knn_graph(pts[lo:lo + max_windows])
        start = time.perf_counter()
        p = torch.sigmoid(model(g.to(device)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        net += time.perf_counter() - start
        np.add.at(heat, (lo + w, s, t), p.cpu().numpy().astype(np.float64) / 2)   # (p_ij + p_ji) / 2
    heat = heat + heat.transpose(0, 2, 1)
    return heat, net


# ---------------------------------------------------------------------------------------------------- features

def batch_features(ctx, coords, cities, p_seg, starts):
    """learned_triage.window_features for W windows of the same length at once.
    cities: (W, m) instance indices in tour order; p_seg: (W, m − 1) probabilities of the segment's edges."""
    pts = coords[cities]
    edges = np.sqrt(((pts[:, 1:] - pts[:, :-1]) ** 2).sum(-1))
    a, b = cities[:, :-1], cities[:, 1:]
    knn = ctx["knn"]
    in_knn = (knn[a] == b[..., None]).any(-1) | (knn[b] == a[..., None]).any(-1)
    p_mean = p_seg.mean(axis=1)
    return {"p_mean": p_mean, "p_min": p_seg.min(axis=1), "p_max": p_seg.max(axis=1), "disagreement": 1.0 - p_mean,
            "seg_len_nn": edges.sum(axis=1) / np.maximum(ctx["nn_dist"][cities].sum(axis=1), 1e-12),
            "max_edge_nn": edges.max(axis=1) / ctx["mean_nn"],
            "density": ctx["density"][cities].mean(axis=1),
            "position": np.asarray(starts, dtype=float) / ctx["n"],
            "out_knn5": (~in_knn).sum(axis=1).astype(float)}


def model_values(learned, feats):
    X = np.stack([feats[f] for f in learned["features"]], axis=1)
    m = learned["model"]
    return m.predict(X) if learned["kind"] == "hgb_reg" else m.predict_proba(X)[:, 1]


class _Scorer:
    """Shared part: window grouping by length, the learned model (or the disagreement), timing."""

    def __init__(self, model, device, learned=None):
        self.model, self.device, self.learned = model, device, learned
        if learned is not None:
            assert set(learned["features"]) <= set(FEATURES)
        self.net_seconds = 0.0
        self.ctx = None

    def _score(self, feats):
        return model_values(self.learned, feats) if self.learned is not None else feats["disagreement"]

    def _ctx(self, coords):
        # the learned features need the instance context; the disagreement does not
        return instance_context(coords) if self.learned is not None else None


class WindowBatchScorer(_Scorer):
    """Version A: per-window heatmaps, built and run in batch; scores cached by (start, cities)."""

    def prepare(self, coords):
        self.ctx = self._ctx(coords)
        self.cache = {}

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
            heat, net = batch_heat(self.model, self.device, coords[cities])
            self.net_seconds += net
            p_seg = heat[:, np.arange(length - 1), np.arange(1, length)]
            if self.learned is None:
                vals = 1.0 - p_seg.mean(axis=1)
            else:
                vals = self._score(batch_features(self.ctx, coords, cities, p_seg, [s for _, s, _, _ in items]))
            for (k, _, _, key), v in zip(items, vals):
                out[k] = self.cache[key] = v
        return out


# ---------------------------------------------------------------------------------------------------- version B

def global_knn_graph(coords, k):
    """scale.knn_graph(coords, k) built without Python loops: (Data, undirected keys lo * n + hi sorted)."""
    n = len(coords)
    k = min(k, n - 1)
    pos32 = coords.astype(np.float32)
    _, idx = cKDTree(pos32.astype(np.float64)).query(pos32.astype(np.float64), k=k + 1)
    self_col = idx == np.arange(n)[:, None]
    if not self_col[:, 0].all():                      # duplicate points: drop the city itself wherever it is
        rows = [r[r != i][:k] for i, r in enumerate(idx)]
        nbr = np.stack(rows)
    else:
        nbr = idx[:, 1:]
    src = np.repeat(np.arange(n), k)
    dst = nbr.reshape(-1)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keys = np.unique(lo.astype(np.int64) * n + hi)
    a, b = keys // n, keys % n
    s = np.concatenate([a, b])
    t = np.concatenate([b, a])
    order = np.lexsort((t, s))
    s, t = s[order], t[order]
    x = torch.from_numpy(pos32)
    edge_index = torch.from_numpy(np.stack([s, t]).astype(np.int64))
    edge_attr = (x[edge_index[0]] - x[edge_index[1]]).norm(dim=1, keepdim=True)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), keys


@torch.no_grad()
def edge_probs(model, data, chunk):
    """sigmoid(model(data)) with the edge head applied in chunks of `chunk` edges (same values as model(data); on a
    10k instance the head's (E, 3 · hidden) input alone is ~0.8 GiB)."""
    h, e = model.encode(data)
    src, dst = data.edge_index
    out = [model.head(torch.cat([h[src[lo:lo + chunk]], h[dst[lo:lo + chunk]], e[lo:lo + chunk]], dim=-1))
           for lo in range(0, src.numel(), chunk)]
    return torch.sigmoid(torch.cat(out).squeeze(-1))


class GlobalScorer(_Scorer):
    """Version B: one network pass on the whole instance; window probabilities read from the global p."""

    def __init__(self, model, device, learned=None, k_max=100, chunk=131072):
        super().__init__(model, device, learned)
        self.k_max, self.chunk = k_max, chunk
        self.missing = [0, 0]                         # tour edges looked up outside the global graph, all looked up

    @torch.no_grad()
    def prepare(self, coords):
        self.ctx = self._ctx(coords)
        n = len(coords)
        g, self.keys = global_knn_graph(coords, min(n - 1, round(0.5 * n), self.k_max))
        src, dst = g.edge_index.numpy().copy()                   # Data.to() moves the tensors in place
        start = time.perf_counter()
        p = edge_probs(self.model, g.to(self.device), self.chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.net_seconds += time.perf_counter() - start
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        pos = np.searchsorted(self.keys, lo.astype(np.int64) * n + hi)
        self.p = np.bincount(pos, weights=p.cpu().numpy().astype(np.float64), minlength=len(self.keys)) / 2
        self.n = n

    def edge_p(self, a, b):
        key = np.minimum(a, b).astype(np.int64) * self.n + np.maximum(a, b)
        pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
        found = self.keys[pos] == key
        self.missing[0] += int((~found).sum())
        self.missing[1] += int(found.size)
        return np.where(found, self.p[pos], 0.0)

    def values(self, coords, tour, wins):
        out = np.zeros(len(wins))
        groups = {}
        for k, (s, length) in enumerate(wins):
            groups.setdefault(length, []).append((k, s))
        for length, items in groups.items():
            starts = np.array([s for _, s in items])
            cities = tour[(starts[:, None] + np.arange(length)[None, :]) % len(tour)]
            p_seg = self.edge_p(cities[:, :-1], cities[:, 1:])
            if self.learned is None:
                vals = 1.0 - p_seg.mean(axis=1)
            else:
                vals = self._score(batch_features(self.ctx, coords, cities, p_seg, starts))
            out[[k for k, _ in items]] = vals
        return out
