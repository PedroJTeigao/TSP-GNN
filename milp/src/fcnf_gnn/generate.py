"""Random FCNF instances (phase 1).

One instance with n nodes:
    positions   n points uniform in [0, 1]²
    arcs        for every node i and each of its k = 4 nearest neighbours j: arcs i -> j and j -> i
                (union, no duplicates), plus round(0.05 n) random arcs; if the graph is still not
                strongly connected, arcs in both directions between random nodes of different
                strongly connected components are added until it is
    b           round(0.2 n) supply nodes with integer supply U{10..50}; round(0.2 n) demand
                nodes share the total supply S in proportion to weights U(0.5, 1.5) (integers,
                each >= 1, sum exactly S); the other nodes have b = 0
    c           100 × Euclidean length of the arc
    u           integers U{ceil(0.2 S), S}; if the max flow from the supplies to the demands is
                below S, every capacity is multiplied by 1.25 (rounded up) until it is not
    F           ratio × c × u, with ratio drawn from {0.5, 1, 2} once per instance
The same seed gives the same instances.
"""

import networkx as nx
import numpy as np

K_NEAREST = 4
RANDOM_ARC_FRAC = 0.05
TERMINAL_FRAC = 0.2
RATIOS = (0.5, 1.0, 2.0)


def knn_arcs(pos, k=K_NEAREST):
    """Arcs to the k nearest neighbours, in both directions: (m, 2) array, sorted, no duplicates."""
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nearest = np.argsort(d, axis=1)[:, :k]
    i = np.repeat(np.arange(len(pos)), k)
    pairs = np.concatenate([np.stack([i, nearest.ravel()], 1), np.stack([nearest.ravel(), i], 1)])
    return np.unique(pairs, axis=0)


def strongly_connected(n, arcs):
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, arcs))
    return nx.is_strongly_connected(g), g


def add_arcs(n, arcs, rng, num_random):
    """Add num_random random arcs (not already present), then repair strong connectivity."""
    present = set(map(tuple, arcs.tolist()))
    new = []
    while len(new) < num_random:
        i, j = rng.choice(n, 2, replace=False)
        if (i, j) not in present:
            present.add((int(i), int(j)))
            new.append((int(i), int(j)))
    ok, g = strongly_connected(n, np.array(sorted(present)))
    while not ok:
        comps = list(nx.strongly_connected_components(g))
        a, b = rng.choice(len(comps), 2, replace=False)
        i, j = int(rng.choice(sorted(comps[a]))), int(rng.choice(sorted(comps[b])))
        for e in ((i, j), (j, i)):
            if e not in present:
                present.add(e)
                g.add_edge(*e)
        ok = nx.is_strongly_connected(g)
    return np.array(sorted(present), dtype=np.int64)


def supplies(n, rng):
    """b: integer supplies and demands that sum to 0 (see the module docstring)."""
    num_t = max(1, round(TERMINAL_FRAC * n))
    nodes = rng.permutation(n)
    sup, dem = nodes[:num_t], nodes[num_t:2 * num_t]
    b = np.zeros(n, dtype=np.int64)
    b[sup] = rng.integers(10, 51, num_t)
    total = int(b[sup].sum())
    w = rng.uniform(0.5, 1.5, num_t)
    share = np.maximum(1, np.floor(total * w / w.sum()).astype(np.int64))
    for t in rng.permutation(num_t):          # fix the rounding so the demands sum to exactly S
        diff = total - int(share.sum())
        if diff == 0:
            break
        share[t] = max(1, share[t] + diff)
    assert share.sum() == total
    b[dem] = -share
    return b


def max_flow(n, arcs, u, b):
    """Max flow from a super source (to every supply) to a super sink (from every demand)."""
    g = nx.DiGraph()
    for (i, j), cap in zip(arcs.tolist(), u.tolist()):
        g.add_edge(i, j, capacity=cap)
    for i in np.where(b > 0)[0]:
        g.add_edge("s", int(i), capacity=int(b[i]))
    for i in np.where(b < 0)[0]:
        g.add_edge(int(i), "t", capacity=int(-b[i]))
    return nx.maximum_flow_value(g, "s", "t")


def generate_instance(n, rng):
    pos = rng.random((n, 2))
    arcs = add_arcs(n, knn_arcs(pos), rng, round(RANDOM_ARC_FRAC * n))
    b = supplies(n, rng)
    total = int(b[b > 0].sum())
    u = rng.integers(int(np.ceil(0.2 * total)), total + 1, len(arcs)).astype(np.int64)
    rescaled = 0
    while max_flow(n, arcs, u, b) < total:
        u = np.ceil(1.25 * u).astype(np.int64)
        rescaled += 1
    length = np.sqrt(((pos[arcs[:, 0]] - pos[arcs[:, 1]]) ** 2).sum(-1))
    ratio = float(rng.choice(RATIOS))
    c = 100.0 * length
    return {"pos": pos, "b": b, "src": arcs[:, 0], "dst": arcs[:, 1], "c": c, "u": u.astype(float),
            "F": ratio * c * u, "ratio": ratio, "length": length, "cap_rescaled": rescaled}


def generate_instances(num, n, seed):
    rng = np.random.default_rng(seed)
    return [generate_instance(n, rng) for _ in range(num)]
