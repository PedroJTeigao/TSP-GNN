"""One episode under the rules of a request: Clarke & Wright with 3 orders per trip under saturation, LKH-3, Hungarian, prioritized under
action dependencies, with rotation and the delays of regime R1 of phase B0c. The verifier judges it twice: against the rules the arm
gave the solvers, and against the true rules.
"""

import pickle

import numpy as np

from warehouse import batching as B
from warehouse import constraints as C
from warehouse import exec as E
from warehouse import sim
from warehouse import verify as V

CAP = 3
R1 = (0.10, 0.002)
MAX_STEPS = 6000


class Costs(B.TripCost):
    """The trip cost with the station of each order in the key, so a memo can be kept between arms."""

    def __call__(self, group):
        key = tuple(sorted((o.oid, o.station) for o in group))
        out = self.memo.get(key)
        if out is None:
            out = B.TripCost.__call__(self, sorted(group, key=lambda o: o.oid))
            self.memo[key] = out
        return out


def load_memo(path):
    try:
        return pickle.load(open(path, "rb"))
    except (OSError, EOFError):
        return {}


def data(orders):
    return [{"oid": o.oid, "products": o.products, "station": o.station, "arrival": o.arrival} for o in orders]


def run(world, prims, true_checks, seed, memo=None, cache=None):
    """Returns (report, memo). report: makespan, mean delay, seconds, violations per true check, violations of the arm's own rules."""
    rules = C.Rules(world, prims)
    wh, grid = world.wh, rules.grid
    orders = rules.orders()
    cost = Costs(wh, wh, sim.SERVICE)
    if memo is not None:
        cost.memo = memo
    groups = B.savings(orders, cost, CAP, rules.penalty)
    cache = {} if cache is None else cache
    ex = E.Execution(wh, wh.layout.robots, E.Flat(*R1), seed)
    plan = C.RulePlanner(rules, seed)
    disp = C.RuleHungarian(grid, groups, rules, world.robots, cache=cache)
    ep = sim.run_episode(grid, world.cat, orders, "lkh", plan, world.seed, dispatcher=disp, execution=ex, cache=cache, router=rules.route, max_steps=MAX_STEPS)
    args = (ep["traj"], wh.grid, data(orders), world.cat.shelf_of, wh.shelves, ep["events"], ep["service"], ep["step_seconds"])
    own = V.verify(*args, rules=C.to_checks(world, prims) + C.world_checks(world), capacity=CAP, rotation=True)
    truth = V.verify(*args, rules=true_checks, capacity=CAP, rotation=True)
    return {"ok": bool(own["ok"]), "problems": {k: v for k, v in own["problems"].items() if v}, "structural": not any(v for k, v in truth["problems"].items() if k != "rules_broken"),
            "broken": truth["problems"]["rules_broken"], "by_check": [i for i, x in enumerate(truth["by_rule"]) if x], "delivered": truth["delivered"], "orders": truth["orders"],
            "makespan": truth["makespan"], "mean_delay": truth["mean_delay"], "step_ms": truth["step_ms_mean"], "seconds": float(np.sum(ep["step_seconds"])),
            "steps": truth["steps"], "trips": len(groups), "fallbacks": ep["planner"]["fallbacks"], "calls": ep["planner"]["calls"], "seq_seconds": ep["seq_seconds"]}, cost.memo
