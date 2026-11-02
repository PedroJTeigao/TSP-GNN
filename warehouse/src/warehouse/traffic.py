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


class EdgeMetric:
    """Shortest paths with a cost per directed edge: 1 step plus extra[(u, v)]. dist_to(goal) gives, for each cell, the cost
    of going from it to goal, by Dijkstra, cached; integers in hundredths of a step, shelves at -1. symmetric=True takes the
    mean of both directions, for LKH-3 here, which reads a symmetric matrix."""

    def __init__(self, wh, extra, symmetric=False):
        self.wh = wh
        self.extra = {}
        for (u, v), x in extra.items():
            self.extra[(u, v)] = (x + extra.get((v, u), 0.0)) / 2 if symmetric else x
        if symmetric:
            for (u, v), x in list(self.extra.items()):
                self.extra[(v, u)] = x
        self._dist = {}

    def edge(self, a, b):
        return SCALE + round(SCALE * self.extra.get((a, b), 0.0))

    def dist_to(self, goal):
        d = self._dist.get(goal)
        if d is None:
            d = [-1] * (self.wh.height * self.wh.width)
            todo, nb = [(0, goal)], self.wh.neighbours
            while todo:
                x, v = heapq.heappop(todo)
                if d[v] >= 0:
                    continue
                d[v] = x
                for u in nb[v]:
                    if d[u] < 0:
                        heapq.heappush(todo, (x + self.edge(u, v), u))
            self._dist[goal] = d
        return d

    def steps_to(self, goal):
        """The same, in steps, for a heuristic."""
        return [x / SCALE for x in self.dist_to(goal)]


class EdgeStats:
    """Real time per edge and per cell, from what the robots did. A pass over the edge u -> v is a move that brings a robot
    closer to its target; the steps it lost since its last pass (waits and detours) are charged to that pass. Each edge and
    each cell keeps the plain totals and a mean with decay: m <- decay * m + (1 - decay) * lost."""

    def __init__(self, wh, decay=0.9):
        self.wh, self.decay = wh, decay
        self.edges, self.cells = {}, {}               # key -> [passes, lost, decayed mean of the lost steps]
        self.owed = {}

    def add(self, table, key, lost):
        e = table.get(key)
        if e is None:
            table[key] = [1, lost, float(lost)]
        else:
            e[0] += 1
            e[1] += lost
            e[2] = self.decay * e[2] + (1 - self.decay) * lost

    def observe(self, now, nxt, goals):
        """One step: cells before and after, and the target of each robot (negative: in service or at rest)."""
        for i, c in enumerate(goals):
            p, q = now[i], nxt[i]
            if c < 0 or p == c:
                self.owed[i] = 0
                continue
            d = self.wh.dist_to(c)
            if d[q] < d[p]:
                lost = self.owed.get(i, 0)
                self.add(self.edges, (p, q), lost)
                self.add(self.cells, p, lost)
                self.owed[i] = 0
            else:
                self.owed[i] = self.owed.get(i, 0) + 1

    def extra(self, decayed=False):
        """(u, v) -> lost steps per pass, the whole-episode mean or the decayed one: real time minus the planned 1 step."""
        return {k: (e[2] if decayed else e[1] / e[0]) for k, e in self.edges.items()}

    def export(self):
        """Arrays for one episode's .npz: edges (N, 2), passes, lost, decayed, delta (real - planned, in steps), and the
        same per cell, with cell ids."""
        keys = sorted(self.edges)
        cells = sorted(self.cells)
        e = np.array([self.edges[k] for k in keys], dtype=np.float64).reshape(-1, 3)
        c = np.array([self.cells[k] for k in cells], dtype=np.float64).reshape(-1, 3)
        return {"edges": np.array(keys, dtype=np.int32).reshape(-1, 2), "edge_passes": e[:, 0], "edge_lost": e[:, 1], "edge_decayed": e[:, 2],
                "edge_delta": e[:, 1] / np.maximum(e[:, 0], 1), "cells": np.array(cells, dtype=np.int32), "cell_passes": c[:, 0],
                "cell_lost": c[:, 1], "cell_decayed": c[:, 2]}


def edge_stats(wh, traj, targets, decay=0.9):
    s = EdgeStats(wh, decay)
    for t in range(len(targets)):
        s.observe(traj[t].tolist(), traj[t + 1].tolist(), targets[t].tolist())
    return s


def pooled_extra(exports):
    """(u, v) -> lost steps per pass, pooled over several exported episodes."""
    passes, lost = {}, {}
    for x in exports:
        for (u, v), n, m in zip(x["edges"].tolist(), x["edge_passes"].tolist(), x["edge_lost"].tolist()):
            passes[(u, v)] = passes.get((u, v), 0.0) + n
            lost[(u, v)] = lost.get((u, v), 0.0) + m
    return {k: lost[k] / passes[k] for k in passes}


def crisscross(wh, across, down):
    """Highways: corridors alternate direction, and going against one costs extra. across[k]: the extra steps on the k-th of
    len(across) bands of rows, down[k] on the bands of columns. Returns extra[(u, v)] for a directed EdgeMetric."""
    shelf = np.argwhere(wh.grid == 1)
    r0, r1, c0, c1 = shelf[:, 0].min(), shelf[:, 0].max(), shelf[:, 1].min(), shelf[:, 1].max()
    rows = [r for r in range(r0, r1 + 1) if (wh.grid[r, c0:c1 + 1] != 1).all()]
    cols = [c for c in range(c0, c1 + 1) if (wh.grid[r0:r1 + 1, c] != 1).all()]
    extra = {}
    for k, r in enumerate(rows):
        x = across[k * len(across) // len(rows)]
        for c in range(wh.width - 1):
            a, b = wh.cell(r, c), wh.cell(r, c + 1)
            if b in wh.neighbours[a]:
                extra[(b, a) if k % 2 == 0 else (a, b)] = x
    for k, c in enumerate(cols):
        x = down[k * len(down) // len(cols)]
        for r in range(wh.height - 1):
            a, b = wh.cell(r, c), wh.cell(r + 1, c)
            if b in wh.neighbours[a]:
                extra[(b, a) if k % 2 == 0 else (a, b)] = x
    return extra
