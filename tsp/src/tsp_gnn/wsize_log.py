"""Window logs for any window size w (phase 3, triage with larger windows).

Same windows as triage.window_log (first pass on the greedy edge + 2-opt tour, offsets 0 and w/2, each window
evaluated alone, nothing replaced), split in two parts:
    lkh_log        current and LKH path length and LKH time of every window: does not depend on the network, so it
                   runs once per (set, w) instead of once per TAGCN seed (no beam: the policies do not use it);
    seed_features  learned-triage features of the same windows from the network of one seed, computed in batch
                   (fast_triage.batch_heat / batch_features), with the labels of the LKH log.
"""

import time

import numpy as np

from tsp_gnn import fast_triage as F
from tsp_gnn import learned_triage as L
from tsp_gnn.window import lkh_path, path_length, window_starts


def window_cities(tour, w):
    """[(offset, start, cities)] of every window of both half-passes (offsets 0 and w // 2)."""
    n = len(tour)
    return [(offset, s, tour[(s + np.arange(length)) % n])
            for offset in (0, w // 2) for s, length in window_starts(n, w, offset)]


def lkh_log(coords, tour, w):
    """Per-window arrays: offset, start, size, cur_len, lkh_len, ms_lkh."""
    rows = {k: [] for k in ("offset", "start", "size", "cur_len", "lkh_len", "ms_lkh")}
    for offset, s, cities in window_cities(tour, w):
        pts = coords[cities]
        start = time.perf_counter()
        order = lkh_path(pts)
        ms = 1000 * (time.perf_counter() - start)
        for k, v in (("offset", offset), ("start", s), ("size", len(cities)), ("cur_len", path_length(pts)),
                     ("lkh_len", path_length(pts[order])), ("ms_lkh", ms)):
            rows[k].append(v)
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def seed_features(model, device, coords, tour, log):
    """Feature columns (learned_triage.FEATURES) of the windows of `log` (one instance), in the log's order."""
    n = len(tour)
    ctx = L.instance_context(coords)
    out = {f: np.zeros(len(log["start"])) for f in L.FEATURES}
    sizes = log["size"].astype(int)
    for m in np.unique(sizes):
        rows = np.flatnonzero(sizes == m)
        starts = log["start"][rows].astype(int)
        cities = tour[(starts[:, None] + np.arange(m)[None, :]) % n]
        heat, _ = F.batch_heat(model, device, coords[cities])
        p_seg = heat[:, np.arange(m - 1), np.arange(1, m)]
        for f, v in F.batch_features(ctx, coords, cities, p_seg, starts).items():
            out[f][rows] = v
    return out
