"""Which orders share a trip. Each rule returns clusters: lists of orders of at most cap orders.

    savings      Clarke & Wright: the cost of a trip goes from a station through its shelves back to a station, and two trips
                 merge in order of the steps saved, while the result fits; the saving is measured with 2-opt, which is cheap
                 enough for every pair, and the trip itself is sequenced by LKH-3 later;
    regions      k-means on the shelves of the queue, then the nearest orders of each region;
    exact_clusters, exact_pairs  the optimum for two orders per trip: a minimum-weight matching, its trips and their cost;
    cvrp_lkh     the same problem given to LKH-3 in CVRP mode, for up to some 20 orders.

For two orders the CVRP is exact as a model: with d(depot, i) = c(i) and d(i, j) = 2 c(i, j) - c(i) - c(j), the route depot, i, j,
depot costs 2 c(i, j) and the route depot, i, depot costs 2 c(i), so the optimum is twice the best total.
"""

import heapq
import itertools
import os
import re
import subprocess

import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2

from tsp_gnn import lkh3
from warehouse import sequencing as S


class TripCost:
    """Steps of the trip of a group of orders: walk from the first order's station, services, walk between stations."""

    def __init__(self, wh, metric, service):
        self.wh, self.metric, self.service = wh, metric, service
        self.unit = metric.edge(*next((c, n) for c in wh.free for n in wh.neighbours[c])) if hasattr(metric, "edge") else 1
        self.memo = {}

    def __call__(self, group):
        key = tuple(sorted(o.oid for o in group))
        out = self.memo.get(key)
        if out is None:
            shelves = sorted({s for o in group for s in o.shelves})
            cells = [self.wh.access[s] for s in shelves]
            out = min(self.way(cells, w) for w in S.delivery_orders(self.metric, group))
            out += self.service * self.unit * (len(shelves) + len(group))
            self.memo[key] = out
        return out

    def way(self, cells, way):
        start = way[0].station
        d = S.matrix(self.metric, start, cells, start)
        size = S.path_length(d, [0] + S.two_opt(d) + [len(d) - 1])
        return size + sum(self.metric.dist_to(y.station)[x.station] for x, y in zip(way, way[1:]))


def savings(orders, cost, cap):
    """Greedy merges by saving. Saved steps must be positive. A merged cluster is priced again against the others."""
    alive = {o.oid: [o] for o in orders}
    heap, tick = [], 0
    for a, b in itertools.combinations(orders, 2):
        s = cost([a]) + cost([b]) - cost([a, b])
        if s > 0:
            heap.append((-s, a.oid, b.oid, 1, 1))
    heapq.heapify(heap)
    while heap:
        _, i, j, n, m = heapq.heappop(heap)
        if i not in alive or j not in alive or len(alive[i]) != n or len(alive[j]) != m:
            continue
        alive[i] = alive[i] + alive.pop(j)
        if len(alive[i]) >= cap:
            continue
        for k, other in alive.items():
            if k != i and len(other) + len(alive[i]) <= cap:
                s = cost(alive[i]) + cost(other) - cost(alive[i] + other)
                if s > 0:
                    heapq.heappush(heap, (-s, i, k, len(alive[i]), len(other)))
    return sorted(alive.values(), key=lambda g: min(o.oid for o in g))


def regions(wh, orders, cap, k, seed):
    """k regions of the mean shelf of each order; inside a region, the oldest order with its nearest ones."""
    xy = np.array([np.mean([wh.shelves[s] for s in o.shelves], axis=0) for o in orders], dtype=np.float64)
    _, label = kmeans2(xy, k, minit="++", seed=seed)
    out = []
    for c in range(k):
        todo = [i for i in range(len(orders)) if label[i] == c]
        while todo:
            i = todo.pop(0)
            near = sorted(todo, key=lambda j: (np.abs(xy[i] - xy[j]).sum(), j))[:cap - 1]
            todo = [j for j in todo if j not in near]
            out.append([orders[j] for j in [i] + near])
    return sorted(out, key=lambda g: min(o.oid for o in g))


def exact_clusters(orders, cost):
    """The optimal pairs, and the orders left alone."""
    g = nx.Graph()
    g.add_nodes_from(o.oid for o in orders)
    for a, b in itertools.combinations(orders, 2):
        s = cost([a]) + cost([b]) - cost([a, b])
        if s > 0:
            g.add_edge(a.oid, b.oid, weight=s)
    book = {o.oid: o for o in orders}
    pairs = [sorted(p) for p in nx.max_weight_matching(g)]
    paired = {k for p in pairs for k in p}
    out = [[book[a], book[b]] for a, b in pairs] + [[o] for o in orders if o.oid not in paired]
    return sorted(out, key=lambda c: min(o.oid for o in c))


def exact_pairs(orders, cost):
    return total(exact_clusters(orders, cost), cost)


def total(clusters, cost):
    return sum(cost(c) for c in clusters)


def cvrp_lkh(orders, cost, workdir, seed=1, limit=20):
    """LKH-3 in CVRP mode, capacity 2: the best total found."""
    os.makedirs(workdir, exist_ok=True)
    n = len(orders)
    c = [cost([o]) for o in orders]
    rows = [[0] + c]
    for i, a in enumerate(orders):
        rows.append([c[i]] + [0 if i == j else max(0, 2 * cost([a, b]) - c[i] - c[j]) for j, b in enumerate(orders)])
    problem, par = os.path.join(workdir, "c.vrp"), os.path.join(workdir, "c.par")
    with open(problem, "w") as f:
        f.write(f"NAME : orders\nTYPE : CVRP\nDIMENSION : {n + 1}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
                f"CAPACITY : 2\nVEHICLES : {n}\nEDGE_WEIGHT_SECTION\n")
        f.write("".join(" ".join(map(str, r)) + "\n" for r in rows))
        f.write("DEMAND_SECTION\n1 0\n" + "".join(f"{i + 2} 1\n" for i in range(n)) + "DEPOT_SECTION\n1\n-1\nEOF\n")
    with open(par, "w") as f:
        f.write(f"PROBLEM_FILE = {problem}\nSPECIAL\nMTSP_MIN_SIZE = 0\nRUNS = 1\nMAX_TRIALS = 1000\nTIME_LIMIT = {limit}\nSEED = {seed}\nTRACE_LEVEL = 1\n")
    out = subprocess.run([lkh3.LKH_BIN, par], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, check=True).stdout
    return int(re.search(r"Cost\.min = (\d+)", out).group(1)) / 2
