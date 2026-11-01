"""The verifier: judges an episode from what the simulator wrote, and shares nothing with it.

It reads the grid as a plain array, the orders as data, the shelf of each product, the trajectory and the events. It does not
import the simulator, the planners or the Warehouse class. Checks:
    every cell of the trajectory is free, every move is to a neighbour or a wait;
    no two robots in one cell in a step (vertex conflict) or swapping cells (edge conflict);
    every product of every delivered order was picked by the robot of that order, standing next to its shelf for the whole
    service time, after the arrival of the order and before the delivery, and the delivery happened at the right station;
    a robot does not carry two orders at once;
    the extra rules (phase C): forbidden cells, an order of products inside an order, deadlines.
"""

import numpy as np

SHELF = 1


def collisions(traj):
    """(vertex conflicts, edge conflicts) of a (T+1, R) array of cells."""
    srt = np.sort(traj, axis=1)
    vertex = int((srt[:, 1:] == srt[:, :-1]).sum())
    edge = 0
    for t in range(1, len(traj)):
        a, b = traj[t - 1], traj[t]
        moved = np.flatnonzero(a != b)
        pairs = set(zip(a[moved].tolist(), b[moved].tolist()))
        edge += sum((v, u) in pairs for u, v in pairs)
    return vertex, edge // 2


def bad_moves(traj, grid):
    h, w = grid.shape
    r, c = traj // w, traj % w
    steps = np.abs(np.diff(r, axis=0)) + np.abs(np.diff(c, axis=0))
    return int((steps > 1).sum()), int((grid[r, c] == SHELF).sum())


def check_rule(rule, traj, orders, picks, done, w):
    kind = rule["type"]
    if kind == "forbidden_cells":                       # {"cells": [...], "from": t0, "to": t1}
        part = traj[rule.get("from", 0):rule.get("to", len(traj))]
        return int(np.isin(part, rule["cells"]).sum())
    if kind == "product_order":                         # {"first": product, "then": product}: in every order with both
        bad = 0
        for o in orders:
            if o["oid"] in done and {rule["first"], rule["then"]} <= set(o["products"]):
                bad += picks[(o["oid"], rule["first"])] >= picks[(o["oid"], rule["then"])]
        return int(bad)
    if kind == "deadline":                              # {"order": id, "t": step}
        return int(rule["order"] not in done or done[rule["order"]] > rule["t"])
    raise ValueError(f"unknown rule {kind!r}")


def verify(traj, grid, orders, shelf_of, shelves, events, service, step_seconds, rules=(), steps_per_hour=3600):
    """orders: dicts with oid, products, station, arrival. shelf_of[p]: shelf index of a product; shelves[k]: its (row, col).
    Returns {"ok", "problems": {name: count}, metrics}."""
    traj = np.asarray(traj)
    w = grid.shape[1]
    problems = {}
    problems["vertex_conflicts"], problems["edge_conflicts"] = collisions(traj)
    problems["long_moves"], problems["on_shelf"] = bad_moves(traj, grid)

    assigned = {e["order"]: e for e in events if e["kind"] == "assign"}
    by_order = {}
    for e in events:
        if e["kind"] in ("pick", "deliver"):
            by_order.setdefault(e["order"], []).append(e)

    done, picks = {}, {}
    wrong = {"pick": 0, "delivery": 0, "dwell": 0, "overlap": 0}

    def dwelt(e):
        """The robot really stayed at the cell, for exactly the service time."""
        seg = traj[e["start"]:e["end"] + 1, e["robot"]]
        return e["end"] - e["start"] == service and len(seg) == service + 1 and bool((seg == e["cell"]).all())

    book = {o["oid"]: o for o in orders}
    for oid, evs in by_order.items():
        o = book[oid]
        ass = assigned.get(oid)
        delivery = [e for e in evs if e["kind"] == "deliver"]
        if ass is None or len(delivery) != 1:
            wrong["delivery"] += 1
            continue
        d = delivery[0]
        wrong["dwell"] += sum(not dwelt(e) for e in evs)
        if d["robot"] != ass["robot"] or d["cell"] != o["station"]:
            wrong["delivery"] += 1
        seen = {}
        for e in evs:
            if e["kind"] != "pick":
                continue
            r, c = divmod(e["cell"], w)
            sr, sc = shelves[e["shelf"]]
            good = e["robot"] == ass["robot"] and abs(r - sr) + abs(c - sc) == 1 and e["start"] >= o["arrival"] and \
                e["start"] >= ass["t"] and e["end"] <= d["start"]
            wrong["pick"] += not good
            seen[e["shelf"]] = e["end"]
        for p in o["products"]:
            if shelf_of[p] not in seen:
                wrong["pick"] += 1
            else:
                picks[(oid, p)] = seen[shelf_of[p]]
        done[oid] = d["end"]
    spans = {}
    for oid in done:
        ass = assigned[oid]
        spans.setdefault(ass["robot"], []).append((ass["t"], done[oid]))
    for s in spans.values():
        s.sort()
        wrong["overlap"] += sum(b[0] < a[1] for a, b in zip(s, s[1:]))
    problems.update({f"wrong_{k}": v for k, v in wrong.items()})
    rule_breaks = sum(check_rule(r, traj, orders, picks, done, w) for r in rules)
    problems["rules_broken"] = rule_breaks

    ends = np.array([done[o["oid"]] for o in orders if o["oid"] in done], dtype=np.float64)
    delays = np.array([done[o["oid"]] - o["arrival"] for o in orders if o["oid"] in done], dtype=np.float64)
    T = len(traj) - 1
    secs = np.asarray(step_seconds, dtype=np.float64)
    q = np.percentile(secs, [50, 95]) if len(secs) else (0.0, 0.0)
    return {"ok": not any(problems.values()), "problems": problems, "orders": len(orders), "delivered": len(done),
            "steps": T, "makespan": float(ends.max()) if len(ends) else float("nan"), "sum_completion": float(ends.sum()),
            "mean_delay": float(delays.mean()) if len(delays) else float("nan"),
            "orders_per_hour": len(done) / max(T, 1) * steps_per_hour, "step_ms_p50": 1e3 * float(q[0]),
            "step_ms_p95": 1e3 * float(q[1]), "step_ms_max": 1e3 * float(secs.max()) if len(secs) else 0.0,
            "step_ms_mean": 1e3 * float(secs.mean()) if len(secs) else 0.0}
