"""Learned triage signal (phase 3, pending item): predict, per window (w = 20), whether LKH will improve the
current segment, from features that need neither the LKH solution nor the beam.

Per-window features (FEATURES; all scale-free, so the same model can be applied at n = 200 ... 10000):
    p_mean, p_min, p_max   network probability of the w − 1 edges of the current segment (0 outside its k-NN graph)
    disagreement           1 − p_mean (the raw signal of the triage experiment; kept for the importance table)
    seg_len_nn             segment length / sum of the nearest-neighbour distances of its cities (triage.simple_rule)
    max_edge_nn            longest edge of the segment / mean nearest-neighbour distance of the instance
    density                mean number of other cities within 2 × the mean nearest-neighbour distance, over the
                           segment's cities
    position               start of the window / n (where the window lies along the tour)
    out_knn5               edges of the segment whose ends are not among each other's 5 nearest neighbours
Extra, for ablations only (EXTRA): log10_n (training covers n = 200 and 1000 only, so trees cannot extrapolate it
to 10000) and logp (beam B=10 log-probability per edge, from the window log; the policy does not compute the beam).

Labels: improves = LKH path shorter than the current one by more than triage.IMPROVE_TOL; gain_rel = relative gain
of the window in % (current − LKH) / current × 100 (gain in pp of the gap would shrink as 1/n).
"""

import numpy as np
from scipy.spatial import cKDTree

from tsp_gnn.triage import IMPROVE_TOL, segment_disagreement
from tsp_gnn.window import path_length

FEATURES = ("p_mean", "p_min", "p_max", "disagreement", "seg_len_nn", "max_edge_nn", "density", "position",
            "out_knn5")
EXTRA = ("log10_n", "logp")
KNN = 5
DENSITY_RADIUS = 2.0   # in mean nearest-neighbour distances


def instance_context(coords):
    """What the features need from the whole instance, computed once: nearest-neighbour distances, their mean, the
    5 nearest neighbours of each city (as a set of sorted pairs) and the local density of each city."""
    n = len(coords)
    tree = cKDTree(coords)
    k = min(KNN, n - 1)
    dist, nbr = tree.query(coords, k=k + 1)                    # column 0 is the city itself
    nn_dist = dist[:, 1]
    mean_nn = float(nn_dist.mean())
    knn = np.zeros((n, k), dtype=np.int64)
    knn[:] = nbr[:, 1:]
    counts = np.array([len(c) - 1 for c in tree.query_ball_point(coords, DENSITY_RADIUS * mean_nn)], dtype=float)
    return {"n": n, "nn_dist": nn_dist, "mean_nn": mean_nn, "knn": knn, "density": counts}


def window_features(ctx, coords, cities, heat, start):
    """Feature dict of one window. cities: instance indices of the window in current tour order; heat: the
    network's (m, m) probability matrix for the window (hier.Heatmaps); start: position of cities[0] in the tour."""
    pts = coords[cities]
    m = len(cities)
    p = heat[np.arange(m - 1), np.arange(1, m)]
    edges = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    a, b = cities[:-1], cities[1:]
    in_knn = (ctx["knn"][a] == b[:, None]).any(axis=1) | (ctx["knn"][b] == a[:, None]).any(axis=1)
    return {"p_mean": float(p.mean()), "p_min": float(p.min()), "p_max": float(p.max()),
            "disagreement": segment_disagreement(heat),
            "seg_len_nn": path_length(pts) / max(float(ctx["nn_dist"][cities].sum()), 1e-12),
            "max_edge_nn": float(edges.max()) / ctx["mean_nn"],
            "density": float(ctx["density"][cities].mean()),
            "position": start / ctx["n"],
            "out_knn5": float((~in_knn).sum())}


def labels(cur_len, lkh_len):
    """(improves, gain_rel in %) from the window log's current and LKH path lengths."""
    cur_len, lkh_len = np.asarray(cur_len, float), np.asarray(lkh_len, float)
    return cur_len - lkh_len > IMPROVE_TOL, 100.0 * (cur_len - lkh_len) / cur_len


def split_instances(num, seed=0, frac=0.5):
    """Boolean mask over instances 0..num-1: True = training, False = validation. Depends only on (num, seed), so
    an instance is on the same side for every TAGCN seed and every window."""
    order = np.random.default_rng(seed).permutation(num)
    mask = np.zeros(num, dtype=bool)
    mask[order[:int(round(frac * num))]] = True
    return mask
