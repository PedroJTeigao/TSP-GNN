"""Vision probe (phase 5): can a vision model point at the parts of a tour where LKH will gain, from an image?

The instance (coordinates in [0, 1]²) is drawn with a 4 × 4 grid; cells are numbered 1..16 row by row from the TOP
left (cell 1 = x < 0.25, y >= 0.75; cell 16 = x >= 0.75, y < 0.25). A window (triage, w = 20) belongs to the cell of
the centroid of its cities. Per cell: share of its windows that LKH improves (label), their LKH gain, mean
disagreement. The cell policy sends to LKH, in up to 3 passes, every window whose centroid lies in the chosen cells.
"""

import time

import numpy as np

from tsp_gnn import fast_triage as F
from tsp_gnn.triage import IMPROVE_TOL
from tsp_gnn.window import lkh_path, path_length, window_starts

GRID = 4
CELLS = GRID * GRID


def cell_of(points):
    """Cell number (1..16) of each point (array (k, 2) or (2,))."""
    p = np.atleast_2d(points)
    col = np.clip((p[:, 0] * GRID).astype(int), 0, GRID - 1)
    row = GRID - 1 - np.clip((p[:, 1] * GRID).astype(int), 0, GRID - 1)      # row 0 = top
    return row * GRID + col + 1


def tour_edge_p(model, device, coords, tour, w=20):
    """Network probability of each tour edge (t[j], t[j+1]): mean over the windows of both half-passes that contain
    it (each window rescaled and scored as in the triage)."""
    n = len(tour)
    total, count = np.zeros(n), np.zeros(n)
    for offset in (0, w // 2):
        wins = window_starts(n, w, offset)
        for m in sorted({length for _, length in wins}):
            starts = np.array([s for s, length in wins if length == m])
            cities = tour[(starts[:, None] + np.arange(m)[None, :]) % n]
            heat, _ = F.batch_heat(model, device, coords[cities])
            p = heat[:, np.arange(m - 1), np.arange(1, m)]
            pos = (starts[:, None] + np.arange(m - 1)[None, :]) % n
            np.add.at(total, pos, p)
            np.add.at(count, pos, 1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def render(path, coords, tour, size=768, edge_p=None):
    """PNG of the tour with the numbered grid; edges coloured by p (green high, red low) if edge_p is given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    seg = np.stack([coords[tour], coords[np.roll(tour, -1)]], axis=1)
    if edge_p is None:
        lc = LineCollection(seg, colors="black", linewidths=0.8)
    else:
        lc = LineCollection(seg, cmap="RdYlGn", linewidths=1.2)
        lc.set_array(np.nan_to_num(edge_p, nan=0.5))
        lc.set_clim(0, 1)
    ax.add_collection(lc)
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c="#333333", zorder=3)
    for k in range(1, GRID):
        ax.axhline(k / GRID, color="#1f4fbf", lw=1.0, alpha=0.6)
        ax.axvline(k / GRID, color="#1f4fbf", lw=1.0, alpha=0.6)
    for cell in range(1, CELLS + 1):
        r, c = divmod(cell - 1, GRID)
        ax.text(c / GRID + 0.012, 1 - r / GRID - 0.012, str(cell), ha="left", va="top", fontsize=13,
                color="#1f4fbf", fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#1f4fbf")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def window_cells(coords, tour, starts, sizes):
    """Cell of each logged window (centroid of its cities)."""
    n = len(tour)
    return np.array([cell_of(coords[tour[(int(s) + np.arange(int(m))) % n]].mean(axis=0))[0]
                     for s, m in zip(starts, sizes)])


def cell_stats(cells, improves, gain, disagreement):
    """Per cell 1..16: windows, share that LKH improves, total LKH gain, mean disagreement (nan if no window)."""
    out = {k: np.full(CELLS, np.nan) for k in ("label", "gain", "disagreement")}
    out["windows"] = np.zeros(CELLS, dtype=int)
    for c in range(1, CELLS + 1):
        m = cells == c
        out["windows"][c - 1] = m.sum()
        if m.any():
            out["label"][c - 1] = improves[m].mean()
            out["gain"][c - 1] = gain[m].sum()
            out["disagreement"][c - 1] = disagreement[m].mean()
    return out


def parse_ranking(obj):
    """Ordered list of distinct cell numbers 1..16 from the model's JSON ({"cells": [...]}); None if unusable."""
    if not isinstance(obj, dict):
        return None
    cells = obj.get("cells")
    if not isinstance(cells, list):
        return None
    out = []
    for c in cells:
        try:
            v = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= CELLS and v not in out:
            out.append(v)
    return out or None


def ranking_scores(ranking, rng):
    """Score per cell (higher = earlier in the ranking); cells left out share the lowest score, ties broken at
    random (rng) so that AP does not depend on cell numbers."""
    score = np.zeros(CELLS) + rng.random(CELLS) * 1e-3
    for pos, c in enumerate(ranking or []):
        score[c - 1] += CELLS - pos
    return score


def top_cells(score, k=4):
    """The k cells (1..16) with the highest score."""
    return [int(c) + 1 for c in np.argsort(-score, kind="stable")[:k]]


def cell_refine(coords, tour, w, cells, max_passes=3):
    """Windows whose centroid lies in `cells` go to LKH (replaced if shorter), up to max_passes passes (stops when a
    pass gains nothing). Returns (tour, stats) with windows, to_lkh, improved, passes, ms_lkh."""
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    cells = set(int(c) for c in cells)
    stats = {"windows": 0, "to_lkh": 0, "improved": 0, "passes": 0, "ms_lkh": 0.0}
    for _ in range(max_passes):
        stats["passes"] += 1
        gain = 0.0
        for offset in (0, w // 2):
            for s, m in window_starts(n, w, offset):
                idx = (s + np.arange(m)) % n
                pts = coords[t[idx]]
                stats["windows"] += 1
                if cell_of(pts.mean(axis=0))[0] not in cells:
                    continue
                stats["to_lkh"] += 1
                start = time.perf_counter()
                order = lkh_path(pts)
                stats["ms_lkh"] += 1000 * (time.perf_counter() - start)
                old, new = path_length(pts), path_length(pts[order])
                if new < old - IMPROVE_TOL:
                    t[idx] = t[idx][order]
                    stats["improved"] += 1
                    gain += old - new
        if gain <= 0:
            break
    return t, stats
