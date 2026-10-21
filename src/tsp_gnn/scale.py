"""Sparse versions of the graph, decoders and 2-opt, for large instances (phase 2, scale experiment).

The phase 1 code builds dense (n, n) matrices (distances, heatmap, all n(n-1)/2 candidate pairs),
which is fine up to TSP-100 but not for TSP-10000 (10⁸ entries). Here everything works on the
k-NN pairs only:
    knn_graph            k-NN graph with a KD-tree (O(n log n)), same edges as graph.knn_edges
    edge_pairs           the network's edge probabilities as undirected pairs (i < j), averaged
    greedy_sparse        same tour as decode.greedy_decode (exact, see the docstring)
    beam_search_sparse   same tour as decode.beam_search_decode (exact, see the docstring)
    two_opt_neighbors    2-opt restricted to the k nearest neighbours (neighbour lists)
The equivalences with the dense versions are checked in tests/test_scale.py.
"""

import heapq
import time
from collections import deque

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from tsp_gnn.decode import LOG_P_FLOOR, _edges_to_tour


def pair_dist(coords, a, b):
    """Distance between cities a[i] and b[i], computed exactly as data.distance_matrix does."""
    diff = coords[a] - coords[b]
    return np.sqrt((diff ** 2).sum(axis=-1))


def neighbor_lists(coords, k):
    """(n, k) indices of the k nearest neighbours of each city (KD-tree), nearest first."""
    n = len(coords)
    k = min(k, n - 1)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    idx = idx.reshape(n, k + 1)
    # Drop the city itself (normally column 0; with duplicate points it may be elsewhere).
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = idx[i][idx[i] != i]
        out[i] = row[:k]
    return out


def knn_graph(coords, k, return_lists=False):
    """Data(x, edge_index, edge_attr) of the k-NN graph, as graph.instance_to_data with edge_norm='none'.

    Same edges as graph.knn_edges (k nearest of each city, both directions, no duplicates), but
    found with a KD-tree instead of the dense (n, n) distance matrix. No labels (y) are needed here.
    return_lists=True also returns the (n, k) neighbour lists (used by two_opt_neighbors).
    """
    pos = torch.as_tensor(coords, dtype=torch.float32)
    n = pos.size(0)
    lists = neighbor_lists(pos.double().numpy(), k)
    nbrs = torch.as_tensor(lists)
    k = nbrs.size(1)
    src = torch.arange(n).repeat_interleave(k)
    dst = nbrs.reshape(-1)
    edge_index = torch.unique(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]), dim=1)
    s, d = edge_index
    edge_attr = (pos[s] - pos[d]).norm(dim=1, keepdim=True)
    data = Data(x=pos, edge_index=edge_index, edge_attr=edge_attr)
    return (data, lists) if return_lists else data


def edge_pairs(edge_index, probs, n):
    """Undirected pairs (i < j) of the graph and their probability (mean of i->j and j->i).

    Pairs are sorted by (i, j), the order of np.triu_indices used by the dense decoders.
    Returns (pi, pj, prob) as NumPy arrays.
    """
    src, dst = (edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index))
    p = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = lo.astype(np.int64) * n + hi
    uniq, inv = np.unique(key, return_inverse=True)
    total = np.bincount(inv, weights=p.astype(np.float64), minlength=len(uniq))
    # (p_ij + p_ji) / 2, the value of decode.edge_heatmap (a missing direction counts as 0 there too).
    return uniq // n, uniq % n, total / 2


def _graph_order(pi, pj, prob, coords):
    """Graph pairs in the greedy order: decreasing probability, ties broken by the shortest."""
    order = np.lexsort((pair_dist(coords, pi, pj), -prob))
    return pi[order], pj[order], prob[order]


class _Fragments:
    """Union-find over cities (same structure as in decode.greedy_decode)."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, c):
        parent = self.parent
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c


def _close_greedy(coords, degree, frags, adj, num_edges):
    """Finish a greedy run with the pairs outside the graph, by increasing distance.

    In decode.greedy_decode these pairs come after all graph pairs, in distance order. Degrees only
    grow and fragments only merge, so only pairs between cities that still have degree < 2 at this
    point can ever be accepted: this is a greedy edge run on those "free" cities. It is done with a
    KD-tree and a heap that holds, for each free city, its nearest not-yet-examined free neighbour:
    the heap minimum is then the shortest pair not yet examined, as in the dense version. A pair
    that is rejected (degree or same fragment) stays invalid forever, so it is never revisited.
    """
    n = len(coords)
    free = np.where(degree < 2)[0]
    if num_edges >= n - 1:
        return num_edges
    m = len(free)
    tree = cKDTree(coords[free])
    lists, ptr, queried = {}, {}, {}

    def refill(u):
        """(Re)query the neighbours of free city u (global index), doubling how many we ask for."""
        want = min(m, max(9, 2 * queried.get(u, 0)))
        _, loc = tree.query(coords[u], k=want)
        loc = np.atleast_1d(loc)
        cand = free[loc[loc < m]]
        cand = cand[cand != u]
        d = pair_dist(coords, np.full(len(cand), u), cand)
        order = np.lexsort((cand, d))               # by distance, then index (stable tie-break)
        lists[u] = (cand[order], d[order])
        queried[u] = want

    def push_next(u):
        """Push u's next unexamined neighbour on the heap (if any)."""
        while True:
            cand, d = lists[u]
            k = ptr[u]
            if k < len(cand):
                ptr[u] = k + 1
                v = int(cand[k])
                heapq.heappush(heap, (float(d[k]), min(u, v), max(u, v), u, v))
                return
            if queried[u] >= m:
                return                               # u has seen every free city
            done = len(cand)
            refill(u)
            ptr[u] = done                            # skip the ones already examined

    heap = []
    for u in free.tolist():
        refill(u)
        ptr[u] = 0
        push_next(u)
    while num_edges < n - 1 and heap:
        _, _, _, u, v = heapq.heappop(heap)
        if degree[u] >= 2:
            continue                                 # u is full: drop its entry for good
        if degree[v] < 2:
            ru, rv = frags.find(u), frags.find(v)
            if ru != rv:
                frags.parent[ru] = rv
                degree[u] += 1; degree[v] += 1
                adj[u].append(v); adj[v].append(u)
                num_edges += 1
        if degree[u] < 2:
            push_next(u)
    return num_edges


def greedy_sparse(pi, pj, prob, coords):
    """Greedy decoding from the graph pairs: the same tour as decode.greedy_decode.

    Graph pairs are accepted in order of decreasing probability (ties: shortest first) when both
    cities have degree < 2 and lie in different fragments; the remaining fragments are joined with
    the shortest pairs outside the graph (_close_greedy). With no graph pairs this is the classical
    greedy edge heuristic (baselines.greedy_distance_tour).
    """
    n = len(coords)
    degree = np.zeros(n, dtype=np.int64)
    adj = [[] for _ in range(n)]
    frags = _Fragments(n)
    num_edges = 0
    if len(pi):
        ga, gb, _ = _graph_order(pi, pj, prob, coords)
        deg = [0] * n                                 # Python list: faster in this loop
        for a, b in zip(ga.tolist(), gb.tolist()):
            if deg[a] >= 2 or deg[b] >= 2:
                continue
            ra, rb = frags.find(a), frags.find(b)
            if ra == rb:
                continue
            frags.parent[ra] = rb
            deg[a] += 1; deg[b] += 1
            adj[a].append(b); adj[b].append(a)
            num_edges += 1
            if num_edges == n - 1:
                break
        degree[:] = deg
    _close_greedy(coords, degree, frags, adj, num_edges)
    # Same closing and walk as greedy_decode (adjacency lists in acceptance order).
    a, b = np.where(degree == 1)[0]
    adj[a].append(b); adj[b].append(a)
    tour, prev, cur = [0], -1, 0
    for _ in range(n - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour)


def _pair_hash(key):
    """64-bit hash of a pair id (splitmix64): identifies an edge for duplicate detection."""
    z = key.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _fallback_children(deg, comp, coords, graph_keys, need):
    """First `need` valid pairs outside the graph for one partial solution, by increasing distance.

    Valid: both cities have degree < 2 and lie in different fragments. Ties keep the (i, j) order,
    as the stable distance sort of the dense beam search. Returns (a, b) arrays with a < b.
    """
    n = len(deg)
    free = np.where(deg < 2)[0]
    i, j = np.triu_indices(len(free), k=1)
    a, b = free[i], free[j]
    ok = comp[a] != comp[b]
    a, b = a[ok], b[ok]
    key = a.astype(np.int64) * n + b
    pos = np.searchsorted(graph_keys, key)
    in_graph = (pos < len(graph_keys)) & (graph_keys[np.minimum(pos, len(graph_keys) - 1)] == key)
    a, b = a[~in_graph], b[~in_graph]
    order = np.argsort(pair_dist(coords, a, b), kind="stable")[:need]
    return a[order], b[order]


def beam_search_sparse(pi, pj, prob, coords, beam_width):
    """Beam search on the graph pairs: the same tour as decode.beam_search_decode.

    Same algorithm (each partial solution is extended by its first beam_width valid candidates;
    children ranked by the sum of log-probabilities; duplicate edge sets removed; the shortest of
    the final tours is returned), with two changes that do not alter the result:
      * pairs outside the graph (score LOG_P_FLOOR, distance order) are only generated for a
        partial solution with fewer than beam_width valid graph pairs: in the dense version the
        others never reach them in their first beam_width valid candidates;
      * graph pairs that are invalid in every partial solution are dropped from the candidate
        list (they stay invalid in every descendant).
    Duplicate detection uses a hash of the pair id instead of a random key per candidate.
    """
    n = len(coords)
    ca, cb, p = _graph_order(pi, pj, prob, coords)
    cand_score = np.log(np.maximum(p, 1e-12))
    cand_hash = _pair_hash(ca.astype(np.int64) * n + cb)
    graph_keys = np.sort(ca.astype(np.int64) * n + cb)
    active = np.arange(len(ca))

    degree = np.zeros((1, n), dtype=np.int8)
    comp = np.arange(n)[None, :]
    score = np.zeros(1)
    key = np.zeros(1, dtype=np.uint64)
    edges = np.zeros((1, n - 1, 2), dtype=np.int64)

    for step in range(n - 1):
        A, Bc = ca[active], cb[active]
        valid = (degree[:, A] < 2) & (degree[:, Bc] < 2) & (comp[:, A] != comp[:, Bc])
        count = np.cumsum(valid, axis=1) if len(active) else np.zeros((len(score), 0), dtype=np.int64)
        take = valid & (count <= beam_width)
        parent, c = np.nonzero(take)
        par_l, a_l, b_l = [parent], [A[c]], [Bc[c]]
        sc_l, h_l = [score[parent] + cand_score[active[c]]], [key[parent] ^ cand_hash[active[c]]]
        nvalid = count[:, -1] if len(active) else np.zeros(len(score), dtype=np.int64)
        for s in np.where(nvalid < beam_width)[0]:
            fa, fb = _fallback_children(degree[s], comp[s], coords, graph_keys, beam_width - nvalid[s])
            par_l.append(np.full(len(fa), s)); a_l.append(fa); b_l.append(fb)
            sc_l.append(np.full(len(fa), score[s] + LOG_P_FLOOR))
            h_l.append(key[s] ^ _pair_hash(fa.astype(np.int64) * n + fb))
        parent, a, b = np.concatenate(par_l), np.concatenate(a_l), np.concatenate(b_l)
        child_score, child_key = np.concatenate(sc_l), np.concatenate(h_l)
        # Group by parent (graph children first, then fallback ones): the dense candidate order.
        grp = np.argsort(parent, kind="stable")
        parent, a, b, child_score, child_key = parent[grp], a[grp], b[grp], child_score[grp], child_key[grp]
        if len(active):
            active = active[valid.any(axis=0)]       # invalid everywhere -> invalid in every child

        order = np.argsort(-child_score, kind="stable")
        _, first = np.unique(child_key[order], return_index=True)
        keep = order[np.sort(first)][:beam_width]
        parent, a, b = parent[keep], a[keep], b[keep]

        rows = np.arange(len(keep))
        degree = degree[parent].copy()
        degree[rows, a] += 1; degree[rows, b] += 1
        comp = comp[parent]
        la, lb = comp[rows, a], comp[rows, b]
        comp = np.where(comp == lb[:, None], la[:, None], comp)
        score, key = child_score[keep], child_key[keep]
        edges = edges[parent].copy()
        edges[:, step, 0], edges[:, step, 1] = a, b

    tours = [_edges_to_tour(e.tolist(), n) for e in edges]
    lengths = [float(pair_dist(coords, t, np.roll(t, -1)).sum()) for t in tours]
    return tours[int(np.argmin(lengths))]


def two_opt_neighbors(coords, tour, nbrs, time_limit=None):
    """2-opt restricted to neighbour lists, with "don't look bits". Returns (tour, finished).

    A move removes two tour edges and reconnects the tour so that city a gets a new edge to one of
    its k nearest neighbours c (nbrs[a]): (a, succ a) + (c, succ c) -> (a, c) + (succ a, succ c),
    or the same with predecessors. Each city is examined once per sweep of the queue (O(k) work,
    so O(n·k) for a full sweep); after a move only its 4 endpoints are put back in the queue.
    The segment is reversed on the shorter side, so a move costs at most n/2 array writes.
    finished = False if time_limit (seconds) ran out first; the tour is still valid then.
    """
    t = np.array(tour, dtype=np.int64)
    n = len(t)
    if n < 5:
        return t, True
    pos = np.empty(n, dtype=np.int64)
    pos[t] = np.arange(n)
    queue = deque(t.tolist())
    queued = np.ones(n, dtype=bool)
    start = time.perf_counter()
    steps = 0

    def dist_to(a, b):
        diff = coords[b] - coords[a]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def reverse(i, j):
        """Reverse the tour between positions i and j (cyclic, i to j forward)."""
        length = (j - i) % n + 1
        if 2 * length > n:                          # reverse the complement instead: same cycle
            i, j, length = (j + 1) % n, (i - 1) % n, n - length
        idx = (i + np.arange(length)) % n
        t[idx] = t[idx[::-1]]
        pos[t[idx]] = idx

    while queue:
        steps += 1
        if time_limit is not None and steps % 256 == 0 and time.perf_counter() - start > time_limit:
            return np.roll(t, -int(pos[0])), False
        a = queue.popleft()
        queued[a] = False
        pa = pos[a]
        succ, pred = t[(pa + 1) % n], t[pa - 1]
        cs = nbrs[a]
        pc = pos[cs]
        sc, prc = t[(pc + 1) % n], t[pc - 1]
        d_ac = dist_to(a, cs)
        gain_s = dist_to(a, succ) + dist_to(cs, sc) - d_ac - pair_dist(coords, np.full(len(cs), succ), sc)
        gain_p = dist_to(a, pred) + dist_to(cs, prc) - d_ac - pair_dist(coords, np.full(len(cs), pred), prc)
        ks, kp = int(np.argmax(gain_s)), int(np.argmax(gain_p))
        if max(gain_s[ks], gain_p[kp]) <= 1e-9:     # same tolerance as decode.two_opt
            continue
        if gain_s[ks] >= gain_p[kp]:                 # (a, succ) + (c, sc) -> (a, c) + (succ, sc)
            c, d = int(cs[ks]), int(sc[ks])
            reverse(pa + 1, pos[c])
            ends = (a, succ, c, d)
        else:                                        # (pred, a) + (prc, c) -> (a, c) + (pred, prc)
            c, d = int(cs[kp]), int(prc[kp])
            reverse(pos[c], pa - 1)
            ends = (a, pred, c, d)
        for v in ends:
            if not queued[v]:
                queued[v] = True
                queue.append(v)
    return np.roll(t, -int(pos[0])), True
