"""Decoding: from edge probabilities (the GNN output) to a valid tour.

Pipeline: edge_heatmap -> greedy_decode -> two_opt (optional) -> verify_tour -> optimality_gap.
All in NumPy, on the CPU (one instance at a time).
"""

import numpy as np
import torch

from tsp_gnn.data import distance_matrix, is_valid_tour, tour_length


def _to_numpy(a):
    """Accept PyTorch tensors (also on the GPU) or arrays and return a NumPy array."""
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def edge_heatmap(edge_index, probs, n):
    """Symmetric (n, n) probability matrix + mask of the pairs that are in the graph.

    Each pair in the graph has two predictions (i->j and j->i); we use their average.
    """
    src, dst = _to_numpy(edge_index)
    heat = np.zeros((n, n))
    heat[src, dst] = _to_numpy(probs)
    in_graph = np.zeros((n, n), dtype=bool)
    in_graph[src, dst] = True
    return (heat + heat.T) / 2, in_graph | in_graph.T


def greedy_decode(heatmap, in_graph, coords):
    """Build a tour by accepting edges in order of preference.

    Candidate order: first the graph edges (decreasing probability, ties broken by the
    shortest); then all pairs outside the graph (increasing distance), to guarantee
    that a complete tour always comes out.
    An edge (i, j) is accepted if i and j have degree < 2 and are in different fragments.
    """
    n = len(coords)
    dist = distance_matrix(coords)
    i, j = np.triu_indices(n, k=1)               # all pairs i < j
    g = in_graph[i, j]

    gi, gj = i[g], j[g]                           # pairs in the graph
    order_g = np.lexsort((dist[gi, gj], -heatmap[gi, gj]))  # primary key: -prob
    oi, oj = i[~g], j[~g]                         # pairs outside the graph
    order_o = np.argsort(dist[oi, oj], kind="stable")
    candidates = np.concatenate([np.stack([gi[order_g], gj[order_g]], axis=1),
                                 np.stack([oi[order_o], oj[order_o]], axis=1)])

    # Union-find: parent[c] points to another city of the same fragment; the root identifies it.
    parent = list(range(n))

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]         # path compression (speeds it up)
            c = parent[c]
        return c

    degree = np.zeros(n, dtype=int)
    adj = [[] for _ in range(n)]                  # neighbours of each city in the tour
    num_edges = 0
    for a, b in candidates.tolist():
        if degree[a] >= 2 or degree[b] >= 2:
            continue                              # a city has only 2 neighbours in the tour
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue                              # would close a subtour
        parent[root_a] = root_b                   # merge the two fragments
        degree[a] += 1; degree[b] += 1
        adj[a].append(b); adj[b].append(a)
        num_edges += 1
        if num_edges == n - 1:
            break                                 # already a path through all cities

    # The n-th edge closes the cycle: it links the two ends (degree 1) of the path.
    a, b = np.where(degree == 1)[0]
    adj[a].append(b); adj[b].append(a)

    # Turn the edges into a visiting order, starting from city 0.
    tour, prev, cur = [0], -1, 0
    for _ in range(n - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour)


LOG_P_FLOOR = np.log(1e-12)  # log-probability of pairs outside the graph (and floor for p ≈ 0)


def _edges_to_tour(edges, n):
    """Visiting order from city 0, given the n-1 path edges in the order they were accepted.

    Same steps as the end of greedy_decode (closing edge, then the walk), so that the same
    edges in the same order give exactly the same array.
    """
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    a, b = [c for c in range(n) if len(adj[c]) == 1]   # the two ends of the path
    adj[a].append(b); adj[b].append(a)
    tour, prev, cur = [0], -1, 0
    for _ in range(n - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour)


def beam_search_decode(heatmap, in_graph, coords, beam_width):
    """Beam search over the same edge choices as greedy_decode; returns the shortest tour found.

    A partial solution is a set of accepted edges (a union of paths), as in greedy_decode.
    At each of the n-1 steps every partial solution is extended by one valid candidate edge
    (degrees < 2, no subtour); children are ranked by the sum of the log-probabilities of
    their edges and the best `beam_width` distinct edge sets are kept. Pairs outside the graph
    (the greedy fallback) get the low log-probability LOG_P_FLOOR and keep their distance order.
    At the end each of the (up to) beam_width paths is closed into a tour and the shortest one
    is returned (selection by length, as in Joshi et al. 2019).

    With beam_width = 1 this is exactly greedy_decode: same candidate order, same tie-breaks.
    Vectorised over the beam with NumPy, one instance at a time.
    """
    n = len(coords)
    dist = distance_matrix(coords)
    # Candidates in the greedy order (same code as greedy_decode).
    i, j = np.triu_indices(n, k=1)
    g = in_graph[i, j]
    gi, gj = i[g], j[g]
    order_g = np.lexsort((dist[gi, gj], -heatmap[gi, gj]))
    oi, oj = i[~g], j[~g]
    order_o = np.argsort(dist[oi, oj], kind="stable")
    cand = np.concatenate([np.stack([gi[order_g], gj[order_g]], axis=1),
                           np.stack([oi[order_o], oj[order_o]], axis=1)])
    ca, cb = cand[:, 0], cand[:, 1]
    # Score of each candidate. It never increases along the candidate list, so the first
    # valid candidates of a partial solution are also its best-scoring children.
    logp = np.log(np.maximum(heatmap[gi[order_g], gj[order_g]], 1e-12))
    cand_score = np.concatenate([logp, np.full(len(oi), LOG_P_FLOOR)])

    # Random 64-bit key per candidate: the XOR of the keys identifies an edge set whatever
    # the order in which its edges were added (used to remove duplicate partial solutions).
    keys = np.random.default_rng(0).integers(0, 2**63, size=len(cand), dtype=np.uint64)

    # State of each partial solution in the beam (one row each).
    degree = np.zeros((1, n), dtype=np.int8)
    comp = np.arange(n)[None, :]                   # fragment label of each city
    score = np.zeros(1)
    key = np.zeros(1, dtype=np.uint64)
    edges = np.zeros((1, n - 1, 2), dtype=np.int64)  # accepted edges, in order

    for step in range(n - 1):
        valid = (degree[:, ca] < 2) & (degree[:, cb] < 2) & (comp[:, ca] != comp[:, cb])
        # The first beam_width valid candidates of each partial solution are enough:
        # the global top beam_width children are always among them.
        take = valid & (np.cumsum(valid, axis=1) <= beam_width)
        parent, c = np.nonzero(take)               # grouped by parent, in candidate order
        child_score = score[parent] + cand_score[c]
        child_key = key[parent] ^ keys[c]
        # Best first; stable sort keeps (parent order, candidate order) on ties, like greedy.
        order = np.argsort(-child_score, kind="stable")
        _, first = np.unique(child_key[order], return_index=True)  # first copy of each edge set
        keep = order[np.sort(first)][:beam_width]
        parent, c = parent[keep], c[keep]

        rows = np.arange(len(keep))
        a, b = ca[c], cb[c]
        degree = degree[parent].copy()
        degree[rows, a] += 1; degree[rows, b] += 1
        comp = comp[parent]
        la, lb = comp[rows, a], comp[rows, b]
        comp = np.where(comp == lb[:, None], la[:, None], comp)   # merge b's fragment into a's
        score, key = child_score[keep], child_key[keep]
        edges = edges[parent].copy()
        edges[:, step] = cand[c]

    tours = [_edges_to_tour(e.tolist(), n) for e in edges]
    lengths = [tour_length(coords, t) for t in tours]
    return tours[int(np.argmin(lengths))]          # ties: the one with the higher score


def two_opt(coords, tour):
    """Improve the tour with 2-opt until no swap shortens it.

    Replacing edges (a,b) and (c,d) with (a,c) and (b,d) is the same as reversing the segment b..c.
    """
    dist = distance_matrix(coords)
    t = np.array(tour)
    n = len(t)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            # j runs over the edges not adjacent to (t[i], t[i+1]);
            # with i = 0, the last edge (t[n-1], t[0]) is adjacent and is left out.
            js = np.arange(i + 2, n if i > 0 else n - 1)
            a, b = t[i], t[i + 1]
            c, d = t[js], t[(js + 1) % n]
            gain = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]  # change in length
            k = np.argmin(gain)
            if gain[k] < -1e-9:                   # tolerance against rounding errors
                j = js[k]
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                improved = True
    return t                                      # t[0] never moves: it still starts at 0


def verify_tour(tour, n):
    """Raise a ValueError explaining the problem if the tour does not visit each city once."""
    if not is_valid_tour(tour, n):
        tour = np.asarray(tour)
        missing = sorted(set(range(n)) - set(tour.tolist()))
        repeated = sorted({c for c in tour.tolist() if (tour == c).sum() > 1})
        raise ValueError(f"invalid tour: length {len(tour)} (expected {n}), "
                         f"missing {missing}, repeated {repeated}")
    return tour


def optimality_gap(length, opt_len):
    """Optimality gap in %: how much longer the tour is than the LKH tour."""
    return 100.0 * (length / opt_len - 1.0)


def tour_edges(tour):
    """Set of the undirected edges of the tour, as (smaller, larger) pairs."""
    tour = np.asarray(tour)
    return {(min(a, b), max(a, b)) for a, b in zip(tour.tolist(), np.roll(tour, -1).tolist())}


def decode(data, probs, use_2opt=True):
    """Decode one graph (Data) given the probabilities of its edges.

    Returns a dictionary with the tour, its length and the gap to LKH (opt_len).
    """
    coords = _to_numpy(data.x).astype(np.float64)
    n = len(coords)
    heat, in_graph = edge_heatmap(data.edge_index, probs, n)
    tour = greedy_decode(heat, in_graph, coords)
    if use_2opt:
        tour = two_opt(coords, tour)
    verify_tour(tour, n)
    length = tour_length(coords, tour)
    return {"tour": tour, "length": length,
            "gap": optimality_gap(length, float(data.opt_len))}
