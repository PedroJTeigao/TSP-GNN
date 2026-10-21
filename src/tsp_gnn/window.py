"""Window refinement of an existing tour (phase 3, after the hierarchical construction).

Start from a tour (greedy edge + 2-opt, the baseline). A window is a segment of w consecutive cities of
the tour; its two end cities stay fixed and its interior is re-ordered by a path solver; the new
order replaces the segment only if the path is shorter. Unlike the hierarchy (one visit per group),
a window is any stretch of the current tour, so a region can be visited several times.

One pass = two half-passes with offsets 0 and w/2: each half-pass cuts the cycle into consecutive
windows of w cities (the last one shorter), which are disjoint, so all of them are solved at once
(the model runs in one batch); the second half-pass sees the tour after the first, and its windows
overlap the first ones by w/2. Passes repeat until one gives no improvement (at most max_passes).

Path solvers (all keep the ends of the segment):
    model    the network on the w cities rescaled to [0, 1]², greedy or beam path (hier.greedy_path / beam_path)
    lkh      LKH on the w cities plus a dummy city joined only to the two ends (an upper bound of a window)
    2opt     2-opt on the segment alone (control without model)
"""

import time

import elkai
import numpy as np

from tsp_gnn.data import distance_matrix
from tsp_gnn.hier import beam_path, greedy_path, rescale
from tsp_gnn.scale import greedy_sparse, neighbor_lists, two_opt_neighbors

LKH_SCALE = 10**5      # distances of the rescaled window × this, rounded (LKH needs integers)
LKH_BIG = 10**7        # cost of the dummy city to any non-end city (> any path through the window)


def initial_tour(coords, k):
    """Greedy edge + neighbour-list 2-opt (k neighbours): the baseline of the scale experiment."""
    empty = np.zeros(0, dtype=np.int64)
    tour = greedy_sparse(empty, empty, np.zeros(0), coords)
    tour, _ = two_opt_neighbors(coords, tour, neighbor_lists(coords, k))
    return tour


def path_length(pts):
    return float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1)).sum())


def window_starts(n, w, offset):
    """(start, length) of the consecutive windows of one half-pass; the last one takes what is left (>= 4)."""
    out, pos = [], 0
    while pos < n:
        length = min(w, n - pos)
        if length >= 4:
            out.append(((offset + pos) % n, length))
        pos += w
    return out


# ---------------------------------------------------------------- path solvers (ends = 0 and m-1)

def lkh_path(pts, runs=1):
    """Shortest Hamiltonian path from pts[0] to pts[-1] (LKH with a dummy city joined to the two ends)."""
    m = len(pts)
    if m <= 3:
        return np.arange(m)
    M = np.full((m + 1, m + 1), LKH_BIG, dtype=np.int64)
    M[:m, :m] = np.rint(distance_matrix(rescale(pts)) * LKH_SCALE).astype(np.int64)
    M[m, m] = 0
    M[m, 0] = M[0, m] = M[m, m - 1] = M[m - 1, m] = 0
    tour = np.array(elkai.solve_int_matrix(M.tolist(), runs=runs))
    tour = np.roll(tour, -int(np.where(tour == m)[0][0]))[1:]     # drop the dummy: a path between the ends
    if tour[0] != 0:
        tour = tour[::-1]
    assert tour[0] == 0 and tour[-1] == m - 1, "LKH did not keep the ends next to the dummy city"
    return tour


def two_opt_path(pts):
    """2-opt on an open path with fixed ends (the order of pts is the starting path)."""
    m = len(pts)
    p = np.arange(m)
    if m < 4:
        return p
    D = distance_matrix(pts)
    improved = True
    while improved:
        improved = False
        for i in range(m - 3):
            js = np.arange(i + 2, m - 1)                # reversing p[i+1..j] keeps p[0] and p[m-1]
            gain = D[p[i], p[js]] + D[p[i + 1], p[js + 1]] - D[p[i], p[i + 1]] - D[p[js], p[js + 1]]
            k = int(np.argmin(gain))
            if gain[k] < -1e-9:
                j = js[k]
                p[i + 1:j + 1] = p[i + 1:j + 1][::-1]
                improved = True
    return p


def make_solver(kind, heatmaps=None, beam_width=10):
    """solver(list of point arrays, each in current path order) -> list of new orders (local indices)."""
    if kind == "lkh":
        return lambda sets: [lkh_path(p) for p in sets]
    if kind == "2opt":
        return lambda sets: [two_opt_path(p) for p in sets]
    decode = (lambda h, p: greedy_path(h, p, 0, len(p) - 1)) if kind == "greedy" else \
        (lambda h, p: beam_path(h, p, 0, len(p) - 1, beam_width))

    def solve(sets):
        heats = heatmaps(sets)
        return [decode(h, p) for h, p in zip(heats, sets)]
    return solve


# ---------------------------------------------------------------- refinement

def refine(coords, tour, w, solver, max_passes=5):
    """Window refinement of `tour`. Returns (new tour, stats)."""
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    stats = {"windows": 0, "improved": 0, "passes": 0, "gain": 0.0}
    for _ in range(max_passes):
        stats["passes"] += 1
        pass_gain = 0.0
        for offset in (0, w // 2):
            wins = window_starts(n, w, offset)
            idx = [(s + np.arange(length)) % n for s, length in wins]
            sets = [coords[t[i]] for i in idx]
            orders = solver(sets)
            for i, pts, order in zip(idx, sets, orders):
                stats["windows"] += 1
                assert order[0] == 0 and order[-1] == len(pts) - 1 and len(set(order.tolist())) == len(pts)
                old, new = path_length(pts), path_length(pts[order])
                if new < old - 1e-9:
                    t[i] = t[i][order]
                    stats["improved"] += 1
                    pass_gain += old - new
        stats["gain"] += pass_gain
        if pass_gain <= 0:
            break
    return t, stats
