"""TSP instances with side rules written in English, their formal translation, checkers and an LKH-3 reference solver.

Phase 5, step 1. Conventions shared by every instance (they are part of the preamble the text rules refer to):
    - customer 0 is the depot; the route starts there at time 0, visits every customer once and returns;
    - travel at speed 1, so travel time = Euclidean distance; waiting is allowed (arriving early at a customer with a
      time window means waiting until it opens), and the time of a visit is the time service starts;
    - the zones are fixed rectangles of the unit square (ZONES).

Rule types (RULE_TYPES), formal form and how the reference solver gives them to LKH-3 (3.0.6):
    precedence      {"before": a, "after": b}: a is visited before b on the route from the depot. A single
                    precedence never costs anything (the reversed route meets it); two or more can.
                    -> TYPE = SOP: the depot is the first node, a copy of it the last one, and entry (b, a) of the
                       matrix is -1 (LKH reads it as "a before b").
    forbidden_edge  {"i", "j"}: the route never goes directly between i and j (either direction).
                    -> cost FORBIDDEN (far above any tour) on (i, j) and (j, i) of the explicit matrix.
    mandatory_edge  {"i", "j"}: i and j are consecutive on the route (either direction).
                    -> FIXED_EDGES_SECTION. In SOP / TSPTW (asymmetric in LKH) a fixed edge has a direction, so every
                       combination of directions is solved and the best feasible route is kept.
    time_window     {"customer": k, "t1", "t2"}: service at k starts in [t1, t2]. Generated windows close before
                    the pure-TSP route reaches k in either direction (a later window costs only free waiting).
                    -> TYPE = TSPTW, TIME_WINDOW_SECTION, depot as DEPOT_SECTION. Travel times are rounded UP to
                       integers, so a route that LKH finds feasible is also feasible with the true distances.
    soft_zone       {"zone", "rect", "weight"}: driving inside the zone costs (1 + weight) per unit length.
                    -> each matrix entry gets + weight x (length of the segment inside the rectangle). Soft: it can
                       never be "violated"; the checker reports the length driven inside the zone.
A time window and a precedence rule are never generated in the same instance (LKH-3 has no TSPTW with precedences).
The objective of the reference solver is the true route length plus the soft-zone penalty.
"""

import itertools
import os
import re
import subprocess
import time

import numpy as np

from tsp_gnn.lkh3 import LKH_BIN

RULE_TYPES = ("precedence", "forbidden_edge", "mandatory_edge", "time_window", "soft_zone")
ZONES = {"north": (0.0, 0.75, 1.0, 1.0), "south": (0.0, 0.0, 1.0, 0.25), "east": (0.75, 0.0, 1.0, 1.0),
         "west": (0.0, 0.0, 0.25, 1.0), "central": (0.375, 0.375, 0.625, 0.625)}
SOFT_WEIGHT = 1.0                 # "avoid if possible": driving inside the zone costs double
LKH_SCALE = 10_000                # distances are multiplied by this and rounded up for LKH
FORBIDDEN = 1_000_000             # scaled cost of a forbidden edge (a tour of 1000 cities costs about 2.4e5)
NO_LIMIT = 10 ** 9                # latest time of customers without a window
TOL = 1e-9

PREAMBLE = ("Route rules. Customer 0 is the depot: the route starts there at time 0, visits every customer exactly "
            "once and returns to the depot. The vehicle travels at speed 1 (travel time = distance) and may wait "
            "before a customer opens. Zones: north = y > 0.75, south = y < 0.25, east = x > 0.75, west = x < 0.25, "
            "central = the square 0.375 < x, y < 0.625.")

TEMPLATES = {
    "precedence": ["Visit customer {before} before customer {after}.",
                   "Customer {after} can only be served after customer {before} has been visited.",
                   "The stop at customer {before} must come earlier in the route than the stop at customer {after}.",
                   "Serve customer {before} first; customer {after} has to come later in the route."],
    "forbidden_edge": ["The road between customers {i} and {j} is closed.",
                       "Do not travel directly from customer {i} to customer {j}, or from {j} to {i}.",
                       "The direct link between customers {i} and {j} is unavailable today.",
                       "Never drive straight between customer {i} and customer {j}."],
    "mandatory_edge": ["Customers {i} and {j} must be served consecutively.",
                       "Visit customer {j} immediately before or immediately after customer {i}.",
                       "Customers {i} and {j} share a loading dock: go directly from one to the other.",
                       "The route must contain the direct leg between customers {i} and {j}."],
    "time_window": ["Customer {customer} can only be visited between t = {t1:.2f} and t = {t2:.2f}.",
                    "Customer {customer} is open from time {t1:.2f} to time {t2:.2f} only.",
                    "Service at customer {customer} must start no earlier than {t1:.2f} and no later than {t2:.2f}.",
                    "The time window of customer {customer} is [{t1:.2f}, {t2:.2f}]."],
    "soft_zone": ["Avoid the {zone} zone if possible.",
                  "Try to keep the driving inside the {zone} zone to a minimum.",
                  "If you can, stay out of the {zone} zone.",
                  "Driving through the {zone} zone is discouraged."],
}


# ---------------------------------------------------------------------------------------------------- geometry

def dist_matrix(coords):
    d = np.asarray(coords, dtype=np.float64)
    return np.sqrt(((d[:, None, :] - d[None, :, :]) ** 2).sum(-1))


def inside_length(p, q, rect):
    """Length of the segments p -> q (arrays (..., 2)) inside the rectangle (x0, y0, x1, y1): Liang-Barsky."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    x0, y0, x1, y1 = rect
    d = q - p
    t0, t1 = np.zeros(d.shape[:-1]), np.ones(d.shape[:-1])
    empty = np.zeros(d.shape[:-1], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pk, qk in ((-d[..., 0], p[..., 0] - x0), (d[..., 0], x1 - p[..., 0]),
                       (-d[..., 1], p[..., 1] - y0), (d[..., 1], y1 - p[..., 1])):
            r = qk / pk
            empty |= (pk == 0) & (qk < 0)
            t0 = np.where(pk < 0, np.maximum(t0, r), t0)
            t1 = np.where(pk > 0, np.minimum(t1, r), t1)
    frac = np.where(empty, 0.0, np.clip(t1 - t0, 0.0, None))
    return frac * np.linalg.norm(d, axis=-1)


def route_length(coords, tour):
    c = np.asarray(coords, dtype=np.float64)[np.asarray(tour)]
    return float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).sum())


def zone_length(coords, tour, rect):
    c = np.asarray(coords, dtype=np.float64)[np.asarray(tour)]
    return float(inside_length(c, np.roll(c, -1, axis=0), rect).sum())


def objective(coords, tour, rules):
    """True length + soft-zone penalties (what the reference solver minimises)."""
    return route_length(coords, tour) + sum(r["formal"]["weight"] * zone_length(coords, tour, r["formal"]["rect"])
                                            for r in rules if r["type"] == "soft_zone")


def service_times(coords, tour, rules):
    """Time service starts at every position of the route (waiting at customers with a window)."""
    c = np.asarray(coords, dtype=np.float64)
    opens = {r["formal"]["customer"]: r["formal"]["t1"] for r in rules if r["type"] == "time_window"}
    t, out = 0.0, [0.0]
    for a, b in zip(tour[:-1], tour[1:]):
        t += float(np.linalg.norm(c[a] - c[b]))
        t = max(t, opens.get(int(b), 0.0))
        out.append(t)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------- checkers

def check_rule(rule, coords, tour):
    """(satisfied, value) for one rule on a route that starts at the depot (tour[0] == 0); direction matters for
    precedence and time windows. value: the quantity checked (positions, arrival time, length inside the zone)."""
    tour = np.asarray(tour)
    n = len(tour)
    pos = np.empty(n, dtype=np.int64)
    pos[tour] = np.arange(n)
    f, kind = rule["formal"], rule["type"]
    if kind == "precedence":
        return bool(pos[f["before"]] < pos[f["after"]]), (int(pos[f["before"]]), int(pos[f["after"]]))
    if kind in ("forbidden_edge", "mandatory_edge"):
        gap = abs(int(pos[f["i"]]) - int(pos[f["j"]]))
        adjacent = gap in (1, n - 1)
        return adjacent if kind == "mandatory_edge" else not adjacent, gap
    if kind == "time_window":
        t = float(service_times(coords, tour, [rule])[pos[f["customer"]]])
        return bool(f["t1"] - TOL <= t <= f["t2"] + TOL), t
    if kind == "soft_zone":
        return True, zone_length(coords, tour, f["rect"])
    raise ValueError(kind)


def check_all(rules, coords, tour):
    """True if the route is valid (starts at the depot, visits every customer once) and meets every hard rule.
    Time windows are checked together (waiting at one window delays the next)."""
    tour = np.asarray(tour)
    n = len(coords)
    if len(tour) != n or tour[0] != 0 or len(set(tour.tolist())) != n:
        return False
    hard = [r for r in rules if r["type"] not in ("time_window", "soft_zone")]
    if not all(check_rule(r, coords, tour)[0] for r in hard):
        return False
    tws = [r for r in rules if r["type"] == "time_window"]
    if tws:
        t = service_times(coords, tour, tws)
        pos = np.empty(n, dtype=np.int64)
        pos[tour] = np.arange(n)
        if any(t[pos[r["formal"]["customer"]]] > r["formal"]["t2"] + TOL for r in tws):
            return False
    return True


# ---------------------------------------------------------------------------------------------------- text

def rule_text(rule, variant):
    f = dict(rule["formal"])
    return TEMPLATES[rule["type"]][variant % len(TEMPLATES[rule["type"]])].format(**f)


def make_rule(kind, variant, **formal):
    rule = {"type": kind, "variant": int(variant), "formal": formal}
    rule["text"] = rule_text(rule, variant)
    return rule


def instance_text(rules):
    return PREAMBLE + "\n" + "\n".join(f"- {r['text']}" for r in rules)


# ---------------------------------------------------------------------------------------------------- LKH-3

def problem_kind(rules):
    types = {r["type"] for r in rules}
    if "time_window" in types and "precedence" in types:
        raise ValueError("time windows and precedences cannot be combined")
    return "TSPTW" if "time_window" in types else "SOP" if "precedence" in types else "TSP"


def cost_matrix(coords, rules):
    """Scaled integer costs (rounded up) with soft-zone penalties and forbidden edges."""
    c = np.asarray(coords, dtype=np.float64)
    w = dist_matrix(c)
    for r in rules:
        if r["type"] == "soft_zone":
            f = r["formal"]
            w = w + f["weight"] * inside_length(c[:, None, :], c[None, :, :], f["rect"])
    W = np.ceil(w * LKH_SCALE - 1e-9).astype(np.int64)
    np.fill_diagonal(W, 0)
    for r in rules:
        if r["type"] == "forbidden_edge":
            i, j = r["formal"]["i"], r["formal"]["j"]
            W[i, j] = W[j, i] = FORBIDDEN
    return W


def _write_matrix(f, W):
    f.write("\n".join(" ".join(map(str, row)) for row in W.tolist()))
    f.write("\n")


def write_problem(path, coords, rules, directions=()):
    """LKH problem file. directions: one (i, j) per mandatory edge, the direction fixed in SOP / TSPTW."""
    kind = problem_kind(rules)
    n = len(coords)
    W = cost_matrix(coords, rules)
    mand = [(r["formal"]["i"], r["formal"]["j"]) for r in rules if r["type"] == "mandatory_edge"]
    if kind != "TSP":
        mand = list(directions)
    with open(path, "w") as f:
        if kind == "SOP":                                         # nodes 0..n-1 = customers, node n = depot copy
            S = np.full((n + 1, n + 1), FORBIDDEN, dtype=np.int64)
            S[:n, :n] = W
            S[1:n, n] = W[1:n, 0]                                 # customer -> end = back to the depot
            S[:n, 0] = FORBIDDEN                                  # nothing enters the start ...
            S[n, 0] = 0                                           # ... except the end (closes the path)
            np.fill_diagonal(S, 0)
            for r in rules:
                if r["type"] == "precedence":
                    S[r["formal"]["after"], r["formal"]["before"]] = -1
            f.write(f"NAME : rules\nTYPE : SOP\nDIMENSION : {n + 1}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                    f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n{n + 1}\n")
            _write_matrix(f, S)
        else:
            f.write(f"NAME : rules\nTYPE : {kind}\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                    "EDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            _write_matrix(f, W)
        if mand:
            f.write("FIXED_EDGES_SECTION\n" + "".join(f"{i + 1} {j + 1}\n" for i, j in mand) + "-1\n")
        if kind == "TSPTW":
            win = {r["formal"]["customer"]: r["formal"] for r in rules if r["type"] == "time_window"}
            f.write("TIME_WINDOW_SECTION\n")
            for v in range(n):
                lo, hi = ((int(np.floor(win[v]["t1"] * LKH_SCALE)), int(np.floor(win[v]["t2"] * LKH_SCALE)))
                          if v in win else (0, NO_LIMIT))
                f.write(f"{v + 1} {lo} {hi}\n")
            f.write("DEPOT_SECTION\n1\n-1\n")
        f.write("EOF\n")


def read_route(path, n):
    """Route of an LKH tour file as customers 0..n-1 starting at the depot (the SOP end copy, node n, dropped)."""
    with open(path) as f:
        tokens = f.read().split("TOUR_SECTION", 1)[1].split()
    nodes = []
    for t in tokens:
        if t == "-1":
            break
        nodes.append(int(t) - 1)
    nodes = [v for v in nodes if v < n]
    nodes = np.array(nodes, dtype=np.int64)
    return np.roll(nodes, -int(np.where(nodes == 0)[0][0]))


def orient(coords, tour, rules):
    """The direction of the route that meets the rules (LKH's file may list an asymmetric tour either way)."""
    rev = np.concatenate([tour[:1], tour[1:][::-1]])
    if check_all(rules, coords, tour):
        return tour
    return rev if check_all(rules, coords, rev) else tour


def _run_lkh(workdir, problem, params, timeout):
    par, out = os.path.join(workdir, "run.par"), os.path.join(workdir, "best.tour")
    old = os.stat(out).st_mtime_ns if os.path.exists(out) else None
    with open(par, "w") as f:
        f.write(f"PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {out}\n" +
                "".join(f"{k} = {v}\n" for k, v in params.items()))
    try:
        proc = subprocess.run([LKH_BIN, par], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=timeout)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        # LKH-3 checks TIME_LIMIT only between trials and can spin forever inside one (seen on an SOP with fixed
        # edges); the process is killed, and a tour it already wrote in this run is still used.
        log = "timeout"
    if not os.path.exists(out) or os.stat(out).st_mtime_ns == old:
        return None, log
    return out, log


def violations(rules, coords, tour):
    """Number of hard rules a route breaks (time windows checked together)."""
    hard = [r for r in rules if r["type"] not in ("time_window", "soft_zone")]
    bad = sum(not check_rule(r, coords, tour)[0] for r in hard)
    tws = [r for r in rules if r["type"] == "time_window"]
    if tws:
        t = service_times(coords, tour, tws)
        pos = _pos(np.asarray(tour))
        bad += sum(t[pos[r["formal"]["customer"]]] > r["formal"]["t2"] + TOL for r in tws)
    return int(bad)


def violated_rules(rules, coords, tour):
    """Indices of the hard rules a route breaks (time windows checked together, as in check_all)."""
    tour = np.asarray(tour)
    tws = [r for r in rules if r["type"] == "time_window"]
    t = service_times(coords, tour, tws) if tws else None
    pos = _pos(tour)
    out = []
    for k, r in enumerate(rules):
        if r["type"] == "time_window":
            if t[pos[r["formal"]["customer"]]] > r["formal"]["t2"] + TOL:
                out.append(k)
        elif r["type"] != "soft_zone" and not check_rule(r, coords, tour)[0]:
            out.append(k)
    return out


def write_initial(path, coords, rules, tour):
    """INITIAL_TOUR_FILE from a route (e.g. the pure TSP one), in the direction that breaks fewer rules; in SOP the
    end copy of the depot (node n + 1) closes it."""
    tour = np.asarray(tour)
    rev = np.concatenate([tour[:1], tour[1:][::-1]])
    t = rev if violations(rules, coords, rev) < violations(rules, coords, tour) else tour
    nodes = [int(v) + 1 for v in t] + ([len(coords) + 1] if problem_kind(rules) == "SOP" else [])
    with open(path, "w") as f:
        f.write(f"TYPE : TOUR\nDIMENSION : {len(nodes)}\nTOUR_SECTION\n" + "".join(f"{v}\n" for v in nodes) +
                "-1\nEOF\n")


def solve(coords, rules, workdir, params=None, timeout=None, initial_tour=None):
    """Reference solution: LKH-3 on the formal translation of the rules.

    initial_tour: a route (e.g. the pure-TSP one) given to LKH as INITIAL_TOUR_FILE, in the direction that breaks
    fewer rules (in SOP and TSPTW, LKH's own start is far worse at n = 1000).
    params: LKH parameters (default RUNS = 1, MAX_TRIALS = 1000, SEED = 1). Returns a dict with tour (starting at the
    depot, in the direction that meets the rules; None if no feasible route was found), feasible, length (true),
    objective (length + soft penalty), kind (TSP / SOP / TSPTW), lkh_runs (direction combinations tried),
    wall_seconds (all runs, files included).
    """
    os.makedirs(workdir, exist_ok=True)
    n = len(coords)
    kind = problem_kind(rules)
    p = {"RUNS": 1, "MAX_TRIALS": 1000, "SEED": 1, "TRACE_LEVEL": 1}
    p.update(params or {})
    mand = [(r["formal"]["i"], r["formal"]["j"]) for r in rules if r["type"] == "mandatory_edge"]
    combos = [()] if kind == "TSP" or not mand else \
        list(itertools.product(*[((i, j), (j, i)) for i, j in mand]))
    problem = os.path.join(workdir, "problem.txt")
    best, runs, logs = None, 0, []
    start = time.perf_counter()
    if initial_tour is not None:
        p["INITIAL_TOUR_FILE"] = os.path.join(workdir, "initial.tour")
        write_initial(p["INITIAL_TOUR_FILE"], coords, rules, initial_tour)
    for dirs in combos:
        write_problem(problem, coords, rules, dirs)
        out, log = _run_lkh(workdir, problem, p, timeout)
        runs += 1
        logs.append(log)
        if out is None:
            continue
        tour = orient(coords, read_route(out, n), rules)
        if len(tour) == n and check_all(rules, coords, tour):
            obj = objective(coords, tour, rules)
            if best is None or obj < best[1]:
                best = (tour, obj)
    wall = time.perf_counter() - start
    tour = best[0] if best else None
    return {"tour": tour, "feasible": best is not None, "kind": kind, "lkh_runs": runs, "wall_seconds": wall,
            "length": route_length(coords, tour) if best else float("nan"),
            "objective": best[1] if best else float("nan"), "log": logs[-1] if logs else ""}


# ---------------------------------------------------------------------------------------------------- generator

def _pos(tour):
    pos = np.empty(len(tour), dtype=np.int64)
    pos[tour] = np.arange(len(tour))
    return pos


def sample_rule(kind, rng, coords, pure_tour, pure_length, taken):
    """One rule of the given type, made to bind against the pure-TSP route where possible (see module docstring).
    taken: set of customers / pairs already used by other rules of the instance (updated)."""
    n = len(coords)
    pos = _pos(pure_tour)
    variant = int(rng.integers(len(TEMPLATES[kind])))
    customers = np.arange(1, n)
    for _ in range(1000):
        if kind == "precedence":                       # violated by the pure route in a random direction
            a, b = (int(x) for x in rng.choice(customers, 2, replace=False))
            if (pos[a] < pos[b]) == bool(rng.integers(2)):  # forward: a after b; backward: a before b in the list
                a, b = b, a
            if abs(int(pos[a]) - int(pos[b])) >= n // 4 and ("c", a) not in taken and ("c", b) not in taken:
                taken |= {("c", a), ("c", b)}
                return make_rule(kind, variant, before=a, after=b)
        elif kind == "forbidden_edge":                 # an edge of the pure route, away from the depot
            p_ = int(rng.integers(1, n - 1))
            i, j = int(pure_tour[p_]), int(pure_tour[p_ + 1])
            if frozenset((i, j)) not in taken:
                taken.add(frozenset((i, j)))
                return make_rule(kind, variant, i=i, j=j)
        elif kind == "mandatory_edge":                 # a near neighbour that the pure route does not join
            i = int(rng.choice(customers))
            d = np.linalg.norm(coords - coords[i], axis=1)
            near = [int(j) for j in np.argsort(d)[1:6] if j != 0]
            near = [j for j in near if abs(int(pos[i]) - int(pos[j])) not in (1, n - 1)]
            if near:
                j = int(rng.choice(near))
                if frozenset((i, j)) not in taken and ("m", i) not in taken and ("m", j) not in taken:
                    taken |= {frozenset((i, j)), ("m", i), ("m", j)}
                    return make_rule(kind, variant, i=i, j=j)
        elif kind == "time_window":                    # width 10 % of the pure length, closing before the pure
            k = int(rng.choice(customers))             # route arrives in either direction (waiting is free, so
            fwd = float(service_times(coords, pure_tour, [])[pos[k]])   # a later window would not bind)
            first = min(fwd, pure_length - fwd)
            w = 0.1 * pure_length
            if first <= 0.1 * pure_length + w:
                continue
            centre = float(rng.uniform(0.1 * pure_length + w / 2, first - w / 2))
            t1, t2 = round(centre - w / 2, 2), round(centre + w / 2, 2)
            if ("c", k) not in taken and t2 < first:
                taken.add(("c", k))
                return make_rule(kind, variant, customer=k, t1=t1, t2=t2)
        elif kind == "soft_zone":
            zone = str(rng.choice(sorted(ZONES)))
            if ("z", zone) not in taken:
                taken.add(("z", zone))
                return make_rule(kind, variant, zone=zone, rect=list(ZONES[zone]), weight=SOFT_WEIGHT)
    raise RuntimeError(f"could not sample a {kind} rule")


def sample_types(rng, k):
    """k rule types (repeats allowed), never a time window together with a precedence."""
    types = []
    for _ in range(k):
        allowed = [t for t in RULE_TYPES if not (t == "time_window" and "precedence" in types)
                   and not (t == "precedence" and "time_window" in types)]
        types.append(str(rng.choice(allowed)))
    return types
