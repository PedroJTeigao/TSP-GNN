"""Distances with traffic for layer 1: the steps a robot really loses, cell by cell, added to the empty grid.

The penalty of a cell is the mean number of steps lost (waits and detours) per pass through it, read from conflicts.analyse.
Moving between two cells costs 1 plus the mean penalty of both, so the distances stay symmetric, which LKH-3 needs here.
Distances are integers, in hundredths of a step; only ratios matter to layer 1.
"""

import heapq

import numpy as np

SCALE = 100


def lost_per_pass(results):
    """(H, W) penalty from one or more outputs of conflicts.analyse, pooled."""
    lost = sum(r["waits"] + r["detours"] for r in results).astype(np.float64)
    passes = sum(r["visits"] for r in results) - lost
    return np.divide(lost, passes, out=np.zeros_like(lost), where=passes > 0)


class Traffic:
    def __init__(self, wh, penalty):
        self.wh = wh
        self.penalty = np.asarray(penalty, dtype=np.float64).ravel().tolist()
        self._dist = {}

    def edge(self, a, b):
        return SCALE + round(SCALE * (self.penalty[a] + self.penalty[b]) / 2)

    def dist_to(self, goal):
        """Same as Warehouse.dist_to, by Dijkstra: a list indexed by cell, shelves at -1."""
        d = self._dist.get(goal)
        if d is None:
            d = [-1] * len(self.penalty)
            todo, nb = [(0, goal)], self.wh.neighbours
            while todo:
                x, v = heapq.heappop(todo)
                if d[v] >= 0:
                    continue
                d[v] = x
                for u in nb[v]:
                    if d[u] < 0:
                        heapq.heappush(todo, (x + self.edge(v, u), u))
            self._dist[goal] = d
        return d
