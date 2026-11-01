"""The warehouse: a 4-connected grid in the Kiva style, the catalogue and the orders.

Layout, from the top row down:
    parking rows (free cells where the robots start and rest);
    rows of shelf blocks (each block block_h x block_w shelf cells) separated by corridors of a given width;
    a free row along the bottom border, with the delivery stations evenly spaced on it.
Shelves are obstacles. Every shelf has one access cell, the free cell next to it where a robot stops to pick. Cells are
numbered row * width + col, which keeps the search code on plain integers.

Time is discrete: in one step a robot moves one cell or waits. One step counts as one second in the rates.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

STATION, PARKING, FREE, SHELF = 3, 2, 0, 1


@dataclass(frozen=True)
class Layout:
    name: str
    height: int
    width: int
    block_rows: int                # rows of blocks
    block_cols: int                # blocks in each row
    corridor: int = 1              # corridor width (1 or 2)
    stations: int = 4
    robots: int = 10
    block_h: int = 2
    block_w: int = 5
    parking_rows: int = 2


SIZES = {
    "small": Layout("small", 20, 30, 4, 4, corridor=1, stations=3, robots=10),
    "medium": Layout("medium", 40, 60, 9, 9, corridor=1, stations=6, robots=30),
    "large": Layout("large", 60, 100, 13, 14, corridor=1, stations=10, robots=80),
}
BATCH = {"small": 50, "medium": 150, "large": 400}     # K of the offline lots


class Warehouse:
    def __init__(self, layout):
        self.layout = lay = layout
        self.height, self.width = lay.height, lay.width
        self.grid = np.zeros((lay.height, lay.width), dtype=np.int8)
        self.grid[:lay.parking_rows] = PARKING
        pitch_r, pitch_c = lay.block_h + lay.corridor, lay.block_w + lay.corridor
        need_r = lay.block_rows * pitch_r + lay.corridor
        need_c = lay.block_cols * pitch_c + lay.corridor
        room_r = lay.height - lay.parking_rows - 1        # the bottom row is kept free for the stations
        if need_r > room_r or need_c > lay.width:
            raise ValueError(f"{lay.name}: {need_r} x {need_c} of blocks do not fit in {room_r} x {lay.width}")
        top = lay.parking_rows + (room_r - need_r) // 2 + lay.corridor
        left = (lay.width - need_c) // 2 + lay.corridor
        self.shelves = []
        for r in range(lay.block_rows):
            for c in range(lay.block_cols):
                r0, c0 = top + r * pitch_r, left + c * pitch_c
                self.grid[r0:r0 + lay.block_h, c0:c0 + lay.block_w] = SHELF
                self.shelves += [(r0 + i, c0 + j) for i in range(lay.block_h) for j in range(lay.block_w)]
        cols = np.linspace(0, lay.width - 1, lay.stations + 2)[1:-1].round().astype(int)
        self.stations = [self.cell(lay.height - 1, int(c)) for c in cols]
        self.grid[lay.height - 1, cols] = STATION
        self.free = np.flatnonzero(self.grid.ravel() != SHELF).tolist()
        # neighbours[c]: the free cells next to free cell c; an empty list for a shelf
        self.neighbours = [[] for _ in range(self.height * self.width)]
        for c in self.free:
            r, q = divmod(c, self.width)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = r + dr, q + dc
                if 0 <= a < self.height and 0 <= b < self.width and self.grid[a, b] != SHELF:
                    self.neighbours[c].append(self.cell(a, b))
        self.access = {}                                  # shelf index -> its access cell
        for k, (r, c) in enumerate(self.shelves):
            above, below = self.cell(r - 1, c), self.cell(r + 1, c)
            # a block two rows high: each shelf touches the corridor above or below it
            self.access[k] = above if self.grid[r - 1, c] != SHELF else below
            assert self.grid.ravel()[self.access[k]] != SHELF, "a shelf with no access cell"
        self.parking = np.flatnonzero(self.grid.ravel() == PARKING).tolist()
        self._dist = {}

    def cell(self, r, c):
        return r * self.width + c

    def xy(self, c):
        return divmod(c, self.width)

    def dist_to(self, goal):
        """Shortest distances to a cell on the empty grid (BFS, cached): a list indexed by cell; shelves stay at -1."""
        d = self._dist.get(goal)
        if d is None:
            d = [-1] * (self.height * self.width)
            d[goal] = 0
            todo = deque([goal])
            nb = self.neighbours
            while todo:
                v = todo.popleft()
                for u in nb[v]:
                    if d[u] < 0:
                        d[u] = d[v] + 1
                        todo.append(u)
            self._dist[goal] = d
        return d


@dataclass
class Order:
    oid: int
    products: list                # product ids (3-12)
    station: int                  # cell of the delivery station
    arrival: int                  # step at which the order arrives
    shelves: list = field(default_factory=list)  # distinct shelf indices of the products


class Catalogue:
    """P products, each on one shelf drawn at random; a shelf may hold several products."""

    def __init__(self, wh, products, seed):
        self.products = products
        self.shelf_of = np.random.default_rng([seed, 1]).integers(0, len(wh.shelves), products)


def make_order(rng, oid, wh, cat, arrival):
    k = int(rng.integers(3, 13))
    prods = rng.choice(cat.products, k, replace=False).tolist()
    return Order(oid, prods, wh.stations[int(rng.integers(len(wh.stations)))], arrival,
                 sorted({int(cat.shelf_of[p]) for p in prods}))


def batch_orders(wh, cat, k, seed):
    """Offline mode: k orders, all arrived at step 0."""
    rng = np.random.default_rng([seed, 2])
    return [make_order(rng, i, wh, cat, 0) for i in range(k)]


def poisson_orders(wh, cat, rate, steps, seed):
    """Lifelong mode: Poisson arrivals with rate orders per step, over the given steps."""
    rng = np.random.default_rng([seed, 3])
    out, t = [], 0.0
    while True:
        t += rng.exponential(1 / rate)
        if t >= steps:
            return out
        out.append(make_order(rng, len(out), wh, cat, int(t)))


def start_cells(wh, robots, seed):
    """The robots start on distinct parking cells, drawn at random: these are also their rest cells."""
    return np.random.default_rng([seed, 4]).choice(wh.parking, robots, replace=False).tolist()
