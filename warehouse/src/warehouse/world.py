"""The named warehouse of phase C: what a request in words can refer to.

Every block of shelves holds one kind of product (normal, fragile, frozen, heavy), so a shelf never mixes kinds. An order is
mostly normal, with at most one special kind: no order mixes two special kinds. Each order has a customer with a name and a
priority. Corridors are numbered: 1-10 run east to west and hold the shelves' access cells, 11-20 run north to south and hold none.
A corridor is the stretch beside the shelves, so the crossings stay open and closing one never cuts the other way. Stations are
letters from west to east. Robots have 2 types: A carries everything, B cannot carry heavy products; that is a fact of the
warehouse, not a rule of a request, and the code enforces it in every arm.
"""

import dataclasses

import numpy as np

from warehouse import env as W

KINDS = ("normal", "fragile", "frozen", "heavy")
SPECIAL = KINDS[1:]
SHARE = {"normal": 0.46, "fragile": 0.18, "frozen": 0.18, "heavy": 0.18}
BLOCK = 10                                      # shelves per block: 2 x 5
PLAIN_ORDERS = 0.4                              # share of orders with no special product
URGENT = 0.06
CUSTOMERS = 30
NAMES = ["Dona Maria", "Dona Rosa", "Dona Alice", "Dona Teresa", "Dona Celeste", "Dona Amelia", "Senhor Antunes", "Senhor Barros",
         "Senhor Cunha", "Senhor Duarte", "Senhor Faria", "Senhor Gomes", "Doutor Lima", "Doutor Matos", "Doutor Neves", "Doutora Paiva",
         "Doutora Reis", "Doutora Sousa", "Farmacia Lopes", "Farmacia Pinto", "Farmacia Vaz", "Mercearia Ramos", "Mercearia Tavares",
         "Mercearia Xavier", "Padaria Alves", "Padaria Brito", "Padaria Costa", "Talho Dias", "Talho Esteves", "Talho Freitas",
         "Quiosque Garcia", "Quiosque Henriques", "Livraria Infante", "Livraria Jardim", "Florista Leal", "Florista Moura",
         "Cafe Nunes", "Cafe Oliveira", "Oficina Pires", "Oficina Queiros", "Hotel Rocha", "Hotel Santos", "Clinica Teixeira",
         "Clinica Valente", "Escola Vieira", "Escola Zuzarte"]


@dataclasses.dataclass
class World:
    wh: object
    cat: object
    orders: list
    shelf_kind: list                            # shelf index -> kind
    corridors: dict                             # number -> sorted cells
    stations: dict                              # letter -> cell
    customers: list
    robots: list                                # robot index -> "A" or "B"
    starts: list
    seed: int

    def shelves_in(self, number):
        cells = set(self.corridors[number])
        return [k for k, c in self.wh.access.items() if c in cells]

    def corridor_of(self, cell):
        return next((k for k, v in self.corridors.items() if cell in v), None)


def corridors(wh):
    """number -> the free cells of that corridor beside the shelves; the crossings belong to none."""
    rows = sorted({r for r, _ in wh.shelves})
    cols = sorted({c for _, c in wh.shelves})
    lay = wh.layout
    pitch_r, pitch_c = lay.block_h + lay.corridor, lay.block_w + lay.corridor
    across = [rows[0] - 1 + k * pitch_r for k in range(lay.block_rows + 1)]
    down = [cols[0] - 1 + k * pitch_c for k in range(lay.block_cols + 1)]
    out = {}
    for k, r in enumerate(across, 1):
        out[k] = sorted(wh.cell(r, c) for c in cols)
    for k, c in enumerate(down, len(across) + 1):
        out[k] = sorted(wh.cell(r, c) for r in rows)
    return out


def shelf_kinds(wh, seed):
    rng = np.random.default_rng([seed, 51])
    blocks = len(wh.shelves) // BLOCK
    out = rng.choice(KINDS, blocks, p=list(SHARE.values())).tolist()
    while set(out) != set(KINDS):               # every kind exists in every world
        out = rng.choice(KINDS, blocks, p=list(SHARE.values())).tolist()
    return [out[k // BLOCK] for k in range(len(wh.shelves))]


def make_orders(wh, cat, kinds, count, seed):
    """All arrive at step 0: k products, some of one special kind and the rest normal."""
    rng = np.random.default_rng([seed, 52])
    by_kind = {x: np.flatnonzero([kinds[s] == x for s in cat.shelf_of]).tolist() for x in KINDS}
    names = rng.choice(NAMES, CUSTOMERS, replace=False).tolist()
    out = []
    for oid in range(count):
        k = int(rng.integers(3, 13))
        special = None if rng.random() < PLAIN_ORDERS else SPECIAL[int(rng.integers(len(SPECIAL)))]
        m = min(int(rng.integers(1, 4)), k) if special else 0
        prods = (rng.choice(by_kind[special], m, replace=False).tolist() if special else []) + rng.choice(by_kind["normal"], k - m, replace=False).tolist()
        out.append(W.Order(oid, prods, wh.stations[int(rng.integers(len(wh.stations)))], 0, sorted({int(cat.shelf_of[p]) for p in prods}),
                           customer=names[int(rng.integers(CUSTOMERS))], priority="urgent" if rng.random() < URGENT else "normal"))
    return out, sorted({o.customer for o in out})


def make(seed, size="medium", robots=30, count=150):
    wh = W.Warehouse(dataclasses.replace(W.SIZES[size], robots=robots))
    cat = W.Catalogue(wh, 2 * len(wh.shelves), seed)
    kinds = shelf_kinds(wh, seed)
    orders, names = make_orders(wh, cat, kinds, count, seed)
    letters = {chr(ord("A") + k): c for k, c in enumerate(wh.stations)}
    return World(wh, cat, orders, kinds, corridors(wh), letters, names, ["A" if i % 2 == 0 else "B" for i in range(robots)], W.start_cells(wh, robots, seed), seed)


def legend(w):
    """The table given with every request."""
    lo = len(w.corridors) // 2
    return "\n".join([f"Orders: numbered 0 to {len(w.orders) - 1}.",
                      f"Corridors: 1 to {lo} run east-west and hold the shelves; {lo + 1} to {len(w.corridors)} run north-south and hold no shelves.",
                      "Stations: " + ", ".join(w.stations) + ".",
                      "Product kinds: " + ", ".join(KINDS) + ".",
                      "Robot types: A (carries everything), B (cannot carry heavy products).",
                      "Customers: " + ", ".join(w.customers) + "."])
