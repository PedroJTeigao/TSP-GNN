"""Multilevel hierarchical TSP with a model trained on TSP-20 (phase 3).

    1. Hierarchy: level 0 = cities; each level is split into groups of ~20 (balanced k-means or
       recursive k-d bisection); the group centroids are the points of the next level, until <= 60 remain.
    2. Top: TSP over the top-level points with the model (k-NN graph, greedy + 2-opt) -> cyclic order.
    3. Descent: given the cyclic order of the groups of a level, each consecutive pair (a, b) is linked
       through one child of a (a's exit) and one child of b (b's entry), chosen by "attraction"
       (-distance, or the model's heatmap on a ∪ b); inside each group the children are visited along a
       Hamiltonian path from its entry to its exit, decoded from the model's heatmap on the children.
       This gives the cyclic order of the level below, down to the cities (the tour).
    4. The tour is then improved with the neighbour-list 2-opt of the scale experiment (scale.py).
Every set of points given to the model is first rescaled to [0, 1]² (the training scale).
"""

import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.loader import DataLoader

from tsp_gnn.decode import LOG_P_FLOOR, two_opt
from tsp_gnn.scale import edge_pairs, greedy_sparse, knn_graph

GROUP = 20        # target group size
TOP = 60          # stop when a level has at most this many points
PARTITIONS = ("kmeans", "kdtree")
ATTRACTS = ("distance", "heatmap")
HEATMAP_MAX_CITIES = 100   # heatmap attraction on the cities of a ∪ b only up to this many


# ---------------------------------------------------------------- hierarchy

def balanced_kmeans(points, size=GROUP, seed=0):
    """Labels of a k-means with k = round(m / size), repaired so that every group has between
    ceil(0.5 size) and floor(1.5 size) points.

    sklearn k-means first; then (1) groups that are too big keep their points closest to the centre
    and the others move to the nearest group with room; (2) groups that are too small take the
    nearest point from a group that can spare one.
    """
    from sklearn.cluster import KMeans

    m = len(points)
    k = max(1, round(m / size))
    if k == 1:
        return np.zeros(m, dtype=np.int64)
    lo, hi = int(np.ceil(0.5 * size)), int(np.floor(1.5 * size))
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(points)
    centers = km.cluster_centers_
    dist = cdist(points, centers)
    labels = dist.argmin(axis=1)
    sizes = np.bincount(labels, minlength=k)
    # (1) capacity: at most hi per group
    released = []
    for g in np.where(sizes > hi)[0]:
        members = np.where(labels == g)[0]
        far = members[np.argsort(dist[members, g])[hi:]]
        released.extend(far.tolist())
        sizes[g] = hi
    for p in sorted(released, key=lambda p: dist[p].min()):
        for g in np.argsort(dist[p]):
            if sizes[g] < hi:
                labels[p] = g
                sizes[g] += 1
                break
    # (2) at least lo per group
    while sizes.min() < lo:
        g = int(sizes.argmin())
        donors = sizes[labels] > lo
        cand = np.where(donors)[0]
        p = cand[np.argmin(dist[cand, g])]
        sizes[labels[p]] -= 1
        labels[p] = g
        sizes[g] += 1
    return labels


def kdtree_partition(points, size=GROUP):
    """Labels of a recursive bisection at the median of the longer axis, until groups have <= 1.5 size."""
    labels = np.zeros(len(points), dtype=np.int64)
    hi = int(np.floor(1.5 * size))
    stack, next_label = [np.arange(len(points))], 0
    while stack:
        idx = stack.pop()
        if len(idx) <= hi:
            labels[idx] = next_label
            next_label += 1
            continue
        p = points[idx]
        axis = int(np.argmax(p.max(axis=0) - p.min(axis=0)))
        order = idx[np.argsort(p[:, axis], kind="stable")]
        half = len(order) // 2
        stack.extend([order[half:], order[:half]])
    return labels


class Hierarchy:
    """points[l]: (m_l, 2) points of level l (level 0 = cities); parent[l][i]: group (node of level l+1)
    of node i of level l; children[l][g]: nodes of level l-1 in group g of level l (l >= 1)."""

    def __init__(self, coords, method="kmeans", size=GROUP, top=TOP, seed=0):
        self.points, self.parent = [np.asarray(coords, dtype=np.float64)], []
        while len(self.points[-1]) > top:
            pts = self.points[-1]
            labels = balanced_kmeans(pts, size, seed) if method == "kmeans" else kdtree_partition(pts, size)
            _, labels = np.unique(labels, return_inverse=True)       # labels 0..k-1, none empty
            k = labels.max() + 1
            cent = np.stack([np.bincount(labels, weights=pts[:, d], minlength=k) for d in (0, 1)], axis=1)
            cent /= np.bincount(labels, minlength=k)[:, None]
            self.parent.append(labels)
            self.points.append(cent)
        self.children = [None] + [[np.where(par == g)[0] for g in range(len(self.points[l + 1]))]
                                  for l, par in enumerate(self.parent)]
        # cities under each node of each level
        self.cities = [[np.array([i]) for i in range(len(coords))]]
        for l in range(1, len(self.points)):
            self.cities.append([np.concatenate([self.cities[l - 1][c] for c in ch]) for ch in self.children[l]])

    @property
    def top(self):
        return len(self.points) - 1

    def group_sizes(self, level):
        """Sizes of the groups of `level` (>= 1): how many nodes of level-1 each one holds."""
        return np.array([len(ch) for ch in self.children[level]])


# ---------------------------------------------------------------- model

def rescale(points):
    """Points moved and scaled (same factor on both axes) to fill [0, 1]²."""
    p = points - points.min(axis=0)
    span = p.max()
    return p / span if span > 0 else p


class Heatmaps:
    """Runs the model on many small point sets at once (batched) and keeps the time spent."""

    def __init__(self, model, device, batch_size=256):
        self.model, self.device, self.batch_size = model.eval(), device, batch_size
        self.seconds = 0.0

    @torch.no_grad()
    def __call__(self, point_sets):
        """(n, n) symmetric probability matrices for each set (0 outside its k-NN graph), k = round(0.5 m)."""
        out = [None] * len(point_sets)
        graphs, where = [], []
        for i, pts in enumerate(point_sets):
            m = len(pts)
            if m < 3:
                out[i] = np.zeros((m, m))
                continue
            graphs.append(knn_graph(rescale(pts), min(m - 1, round(0.5 * m))))
            where.append(i)
        start = time.perf_counter()
        probs = []
        for batch in DataLoader(graphs, batch_size=self.batch_size, shuffle=False):
            batch = batch.to(self.device)
            p = torch.sigmoid(self.model(batch)).cpu()
            per_graph = torch.bincount(batch.batch[batch.edge_index[0]], minlength=batch.num_graphs)
            probs.extend(p.split(per_graph.tolist()))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds += time.perf_counter() - start
        for i, g, p in zip(where, graphs, probs):
            m = len(point_sets[i])
            pi, pj, pr = edge_pairs(g.edge_index, p, m)
            heat = np.zeros((m, m))
            heat[pi, pj] = pr
            out[i] = heat + heat.T
        return out


# ---------------------------------------------------------------- paths with fixed ends

def _path_candidates(heat, pts):
    """All pairs i < j in the greedy order: graph pairs (heat > 0) by decreasing probability (ties:
    shortest), then the others by increasing distance. Returns (a, b, log-score)."""
    m = len(pts)
    i, j = np.triu_indices(m, k=1)
    d = np.sqrt(((pts[i] - pts[j]) ** 2).sum(axis=1))
    p = heat[i, j]
    g = p > 0
    og = np.lexsort((d[g], -p[g]))
    oo = np.argsort(d[~g], kind="stable")
    a = np.concatenate([i[g][og], i[~g][oo]])
    b = np.concatenate([j[g][og], j[~g][oo]])
    score = np.concatenate([np.log(np.maximum(p[g][og], 1e-12)), np.full((~g).sum(), LOG_P_FLOOR)])
    return a, b, score


def _walk(edges, m, s):
    """Visiting order of a Hamiltonian path given its m-1 edges, from end s."""
    adj = [[] for _ in range(m)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    path, prev, cur = [s], -1, s
    for _ in range(m - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        path.append(nxt)
        prev, cur = cur, nxt
    return np.array(path)


def greedy_path(heat, pts, s, t):
    """Hamiltonian path from s to t: the greedy decoder with the ends' degree capped at 1.

    An edge is accepted if both cities have spare degree (2, or 1 for s and t), they are in different
    fragments, and it does not join s's fragment to t's fragment before the last edge (that would
    close a path s..t that misses cities). The last edge closes what is left.
    """
    m = len(pts)
    if m == 1:
        return np.array([s])
    if m == 2:
        return np.array([s, t])
    cap = np.full(m, 2); cap[s] = cap[t] = 1
    deg = np.zeros(m, dtype=np.int64)
    parent = list(range(m))

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    edges = []
    a_all, b_all, _ = _path_candidates(heat, pts)
    for a, b in zip(a_all.tolist(), b_all.tolist()):
        if deg[a] >= cap[a] or deg[b] >= cap[b]:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if {ra, rb} == {find(s), find(t)} and len(edges) + 1 < m - 1:
            continue
        parent[ra] = rb
        deg[a] += 1; deg[b] += 1
        edges.append((a, b))
        if len(edges) == m - 1:
            break
    assert len(edges) == m - 1, "greedy_path could not complete the path"
    return _walk(edges, m, s)


def beam_path(heat, pts, s, t, beam_width=10, info=False):
    """Beam search version of greedy_path (same candidates and constraints, as decode.beam_search_decode).

    Each partial solution is extended by its first beam_width valid candidates, children ranked by the
    sum of log-probabilities, duplicate edge sets removed; the shortest final path is returned.
    With info=True also returns {"logp": sum of log-probabilities of the returned path,
    "scores": those of all final paths, highest first} (the model's confidence, phase 3 triage).
    """
    m = len(pts)
    if m <= 2:
        path = greedy_path(heat, pts, s, t)
        return (path, {"logp": 0.0, "scores": np.zeros(1)}) if info else path
    ca, cb, cand_score = _path_candidates(heat, pts)
    keys = np.random.default_rng(0).integers(0, 2**63, size=len(ca), dtype=np.uint64)
    cap = np.full(m, 2, dtype=np.int8); cap[s] = cap[t] = 1
    degree = np.zeros((1, m), dtype=np.int8)
    comp = np.arange(m)[None, :]
    score, key = np.zeros(1), np.zeros(1, dtype=np.uint64)
    edges = np.zeros((1, m - 1, 2), dtype=np.int64)
    for step in range(m - 1):
        cs, ct = comp[:, [s]], comp[:, [t]]
        ka, kb = comp[:, ca], comp[:, cb]
        joins_ends = ((ka == cs) & (kb == ct)) | ((ka == ct) & (kb == cs))
        valid = (degree[:, ca] < cap[ca]) & (degree[:, cb] < cap[cb]) & (ka != kb)
        if step < m - 2:
            valid &= ~joins_ends
        take = valid & (np.cumsum(valid, axis=1) <= beam_width)
        parent, c = np.nonzero(take)
        child_score = score[parent] + cand_score[c]
        child_key = key[parent] ^ keys[c]
        order = np.argsort(-child_score, kind="stable")
        _, first = np.unique(child_key[order], return_index=True)
        keep = order[np.sort(first)][:beam_width]
        parent, c = parent[keep], c[keep]
        rows = np.arange(len(keep))
        a, b = ca[c], cb[c]
        degree = degree[parent].copy()
        degree[rows, a] += 1; degree[rows, b] += 1
        comp = comp[parent]
        la, lb = comp[rows, a], comp[rows, b]
        comp = np.where(comp == lb[:, None], la[:, None], comp)
        score, key = child_score[keep], child_key[keep]
        edges = edges[parent].copy()
        edges[:, step, 0], edges[:, step, 1] = a, b
    paths = [_walk(e.tolist(), m, s) for e in edges]
    lengths = [np.sqrt(((pts[p[1:]] - pts[p[:-1]]) ** 2).sum(axis=1)).sum() for p in paths]
    best = int(np.argmin(lengths))
    if info:
        return paths[best], {"logp": float(score[best]), "scores": np.sort(score)[::-1]}
    return paths[best]


# ---------------------------------------------------------------- stitching

def best_links(attr, dist, c):
    """The c best (i, j) pairs of an attraction matrix (ties: shortest distance first)."""
    order = np.lexsort((dist.ravel(), -attr.ravel()))[:c]
    i, j = np.unravel_index(order, attr.shape)
    return list(zip(i.tolist(), j.tolist(), attr.ravel()[order].tolist()))


def choose_links(attrs, dists, sizes, c=1):
    """Pick one (exit child of a_k, entry child of a_{k+1}) per consecutive pair of the cycle.

    attrs[k], dists[k]: (|a_k|, |a_{k+1}|) matrices; sizes[k] = number of children of group a_k.
    Maximises the total attraction over the c best pairs of each link (dynamic programming around the
    cycle), with the constraint that a group with >= 2 children does not enter and exit through the
    same child. If no choice satisfies it, c is doubled (returned as the c actually used).
    """
    M = len(attrs)
    while True:
        cands = [best_links(attrs[k], dists[k], c) for k in range(M)]
        best = None
        for x0 in range(len(cands[0])):
            # value[x] = best total with link k at candidate x, given link 0 at x0
            value = {x0: cands[0][x0][2]}
            back = [{x0: None}]
            for k in range(1, M):
                g = k                      # group a_k: entered by link k-1, left by link k
                new, bk = {}, {}
                for x, (i, _, v) in enumerate(cands[k]):
                    for px, pv in value.items():
                        if sizes[g] >= 2 and cands[k - 1][px][1] == i:
                            continue
                        if x not in new or pv + v > new[x]:
                            new[x], bk[x] = pv + v, px
                value = new
                back.append(bk)
                if not value:
                    break
            if not value:
                continue
            for x, v in value.items():     # close the cycle: group a_0 entered by link M-1, left by link 0
                if sizes[0] >= 2 and cands[M - 1][x][1] == cands[0][x0][0]:
                    continue
                if best is None or v > best[0]:
                    best = (v, x0, x, back)
        if best is not None:
            _, x0, x, back = best
            picks = [x]
            for k in range(M - 1, 0, -1):
                picks.append(back[k][picks[-1]])
            picks = picks[::-1]
            return [cands[k][picks[k]][:2] for k in range(M)], c
        if c >= max(a.size for a in attrs):
            raise ValueError("no choice of links satisfies the entry != exit constraint")
        c *= 2


def attraction_matrices(H, level, order, attract, heatmaps):
    """Attraction (and distance) between the children of each consecutive pair of groups of `level`."""
    pts = H.points[level - 1]
    pairs = [(order[k], order[(k + 1) % len(order)]) for k in range(len(order))]
    dists = [cdist(pts[H.children[level][a]], pts[H.children[level][b]]) for a, b in pairs]
    if attract == "distance":
        return [-d for d in dists], dists
    sets, plans = [], []
    for a, b in pairs:
        ca, cb = H.children[level][a], H.children[level][b]
        n_cities = sum(len(H.cities[level][g]) for g in (a, b))
        if level - 1 > 0 and n_cities <= HEATMAP_MAX_CITIES:
            # the cities of a ∪ b; attraction(a_i, b_j) = sum of p over the edges between their cities
            owner = [np.full(len(H.cities[level - 1][x]), i) for i, x in enumerate(np.concatenate([ca, cb]))]
            cities = np.concatenate([H.cities[level - 1][x] for x in np.concatenate([ca, cb])])
            sets.append(H.points[0][cities])
            plans.append(("cities", len(ca), len(cb), np.concatenate(owner)))
        else:   # the children's own points (cities at the last level)
            sets.append(np.concatenate([pts[ca], pts[cb]]))
            plans.append(("points", len(ca), len(cb), None))
    heats = heatmaps(sets)
    attrs = []
    for heat, (kind, na, nb, owner) in zip(heats, plans):
        if kind == "points":
            attrs.append(heat[:na, na:na + nb])
        else:
            onehot = np.zeros((len(owner), na + nb))
            onehot[np.arange(len(owner)), owner] = 1
            agg = onehot.T @ heat @ onehot          # (na + nb)²: summed probabilities between groups
            attrs.append(agg[:na, na:])
    return attrs, dists


# ---------------------------------------------------------------- the whole method

def hierarchical_tour(H, heatmaps, attract="distance", cands=1, path_decoder="greedy", beam_width=10):
    """Tour of the cities (array starting anywhere) + statistics. Times in seconds per step."""
    times = {"top": 0.0, "stitch": 0.0, "paths": 0.0}
    stats = {"c_used_max": cands}
    model_before = heatmaps.seconds
    # 2) top: cyclic order of the top-level points
    start = time.perf_counter()
    top_pts = H.points[H.top]
    if len(top_pts) <= 3:
        order = np.arange(len(top_pts))
    else:
        heat = heatmaps([top_pts])[0]
        i, j = np.nonzero(np.triu(heat, 1))
        order = greedy_sparse(i, j, heat[i, j], top_pts)
        order = two_opt(top_pts, order)
    times["top"] += time.perf_counter() - start
    # 3) descent
    decode = (lambda h, p, s, t: greedy_path(h, p, s, t)) if path_decoder == "greedy" else \
        (lambda h, p, s, t: beam_path(h, p, s, t, beam_width))
    for level in range(H.top, 0, -1):
        start = time.perf_counter()
        attrs, dists = attraction_matrices(H, level, order, attract, heatmaps)
        sizes = [len(H.children[level][g]) for g in order]
        links, c_used = choose_links(attrs, dists, sizes, cands)
        stats["c_used_max"] = max(stats["c_used_max"], c_used)
        times["stitch"] += time.perf_counter() - start
        start = time.perf_counter()
        child_pts = H.points[level - 1]
        heats = heatmaps([child_pts[H.children[level][g]] for g in order])
        new_order = []
        M = len(order)
        for k, g in enumerate(order):
            ch = H.children[level][g]
            entry, exit_ = links[k - 1][1], links[k][0]
            path = decode(heats[k], child_pts[ch], entry, exit_)
            new_order.append(ch[path])
        order = np.concatenate(new_order)
        times["paths"] += time.perf_counter() - start
    times["model"] = heatmaps.seconds - model_before   # included in the steps above
    return order, times, stats


def boundary_stats(H, tour, ref_tour):
    """Tour edges between / inside the level-0 groups, and how many are not in the reference (LKH) tour."""
    group = H.parent[0] if H.parent else np.zeros(len(tour), dtype=np.int64)
    a, b = tour, np.roll(tour, -1)
    ref = {(min(x, y), max(x, y)) for x, y in zip(ref_tour.tolist(), np.roll(ref_tour, -1).tolist())}
    wrong = np.array([(min(x, y), max(x, y)) not in ref for x, y in zip(a.tolist(), b.tolist())])
    boundary = group[a] != group[b]
    return {"boundary_edges": int(boundary.sum()), "boundary_wrong": int((wrong & boundary).sum()),
            "interior_edges": int((~boundary).sum()), "interior_wrong": int((wrong & ~boundary).sum())}


def moved_between_groups(H, before, after):
    """Cities whose tour neighbours in other level-0 groups changed with the final 2-opt."""
    group = H.parent[0] if H.parent else np.zeros(len(before), dtype=np.int64)

    def outside(tour):
        nb = {}
        for x, y in zip(tour.tolist(), np.roll(tour, -1).tolist()):
            if group[x] != group[y]:
                nb.setdefault(x, set()).add(y); nb.setdefault(y, set()).add(x)
        return nb

    b, a = outside(before), outside(after)
    return sum(1 for c in set(b) | set(a) if b.get(c) != a.get(c))
