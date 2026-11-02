"""Requests with exact labels (phase C). The rule comes first, as DSL; its sentence is written from it.

    L1  2-4 rules of different types, each from a sentence template: groups, exceptions, windows, names.
    L2  the same rules, rewritten by gemma3:12b as running text, with background remarks and one rule left implicit. The rewrite is not
        filtered by any parser; its label errors are measured by a manual audit.

A draw is kept only if the validator accepts all its rules; scripts/phaseC_language.py also redraws if the oracle episode breaks a rule,
so every label can be met.
"""

import numpy as np

from warehouse import dsl as L
from warehouse import world as Y

SEED = 20261102
MAX_DEADLINE_ORDERS = 8
WORD = {"fragile": "fragile", "frozen": "frozen", "heavy": "heavy"}
DISTRACTORS = [
    "{name} called to thank the team for last week.",
    "{name} has changed the delivery contact this month.",
    "Corridor {c} was repainted last week.",
    "Station {s} was cleaned on Sunday.",
    "Last week a robot happened to pick {k} products before {k2} products, which was only a coincidence.",
    "{name} and {name2} share the same accountant.",
    "The {k} products sold very well last month.",
    "The night shift finished early yesterday.",
    "Corridor {c} had a burnt-out lamp yesterday.",
    "The software of the type {r} robots was updated in the spring.",
    "{name} paid the last invoice on time.",
    "Station {s} and station {s2} were busy last month.",
]


def pick(rng, items):
    return items[int(rng.integers(len(items)))]


def cap(s):
    return s[0].upper() + s[1:]


def of(name):
    return f"the orders of {name}"


def shelf_corridor(rng, w):
    return int(rng.integers(1, 11))


def cross_corridor(rng, w):
    return int(rng.integers(11, 21))


# Each shape returns (dsl rule, sentence) or None. The sentence must say exactly what the rule says.

def pick_last(rng, w):
    k = pick(rng, Y.SPECIAL)
    dsl = {"type": "pick_order", "first": {"all": True, "except": [{"kind": k}]}, "then": {"kind": k}}
    return dsl, pick(rng, [f"{cap(k)} products must be picked last on every trip.", f"On every trip, leave the {k} products for the end.",
                           f"Nothing else may be picked after the {k} products on a trip."])


def pick_first(rng, w):
    k = pick(rng, Y.SPECIAL)
    dsl = {"type": "pick_order", "first": {"kind": k}, "then": {"all": True, "except": [{"kind": k}]}}
    return dsl, pick(rng, [f"{cap(k)} products must be picked first on every trip.", f"On every trip, start with the {k} products before anything else."])


def pick_pair(rng, w):
    a, b = rng.choice(Y.SPECIAL, 2, replace=False).tolist()
    dsl = {"type": "pick_order", "first": {"kind": a}, "then": {"kind": b}}
    return dsl, pick(rng, [f"Pick {a} products before {b} products on every trip.", f"On any trip, {b} products can only be picked after all the {a} products.",
                           f"{cap(b)} products never come ahead of {a} products on a trip."])


def pick_except(rng, w):
    a, b = rng.choice(Y.SPECIAL, 2, replace=False).tolist()
    c = shelf_corridor(rng, w)
    if int(rng.integers(2)):
        dsl = {"type": "pick_order", "first": {"kind": a}, "then": {"kind": b, "except": [{"corridor": c}]}}
        return dsl, f"Pick {a} products before {b} products on every trip, except the {b} products in corridor {c}."
    dsl = {"type": "pick_order", "first": {"kind": a, "except": [{"corridor": c}]}, "then": {"kind": b}}
    return dsl, f"Apart from those in corridor {c}, all {a} products come before the {b} products on a trip."


def close_all(rng, w):
    c = cross_corridor(rng, w)
    return {"type": "close", "corridors": [c]}, pick(rng, [f"Corridor {c} is closed.", f"Keep the robots out of corridor {c} for the whole shift.", f"Nobody may use corridor {c} today."])


def close_two(rng, w):
    a, b = sorted(rng.choice(range(11, 21), 2, replace=False).tolist())
    return {"type": "close", "corridors": [a, b]}, pick(rng, [f"Corridors {a} and {b} are closed.", f"Both corridor {a} and corridor {b} are out of use all day."])


def close_window(rng, w):
    c = pick(rng, [shelf_corridor(rng, w), cross_corridor(rng, w)])
    t1 = int(rng.integers(5, 50)) * 10
    t2 = t1 + int(rng.integers(5, 21)) * 10
    return {"type": "close", "corridors": [c], "t1": t1, "t2": t2}, pick(rng, [f"Corridor {c} is closed from step {t1} to step {t2}.", f"No robot may be in corridor {c} between step {t1} and step {t2}."])


def close_until(rng, w):
    c = pick(rng, [shelf_corridor(rng, w), cross_corridor(rng, w)])
    t2 = int(rng.integers(10, 41)) * 10
    return {"type": "close", "corridors": [c], "t2": t2}, pick(rng, [f"Corridor {c} stays closed until step {t2}.", f"Corridor {c} only opens at step {t2}."])


def close_from(rng, w):
    c = cross_corridor(rng, w)
    t1 = int(rng.integers(10, 61)) * 10
    return {"type": "close", "corridors": [c], "t1": t1}, f"From step {t1} on, corridor {c} is closed."


def mix_kinds(rng, w):
    a, b = rng.choice(Y.SPECIAL, 2, replace=False).tolist()
    dsl = {"type": "no_mix", "a": {"kind": a}, "b": {"kind": b}}
    return dsl, pick(rng, [f"{cap(a)} orders never share a trip with {b} orders.", f"Do not put {a} orders and {b} orders on the same trip.", f"A {a} order and a {b} order must not travel together."])


def mix_customers(rng, w):
    a, b = rng.choice(w.customers, 2, replace=False).tolist()
    dsl = {"type": "no_mix", "a": {"customer": a}, "b": {"customer": b}}
    return dsl, pick(rng, [f"The orders of {a} must not travel with the orders of {b}.", f"Never mix orders of {a} and {b} on one trip."])


def mix_kind_customer(rng, w):
    k, a = pick(rng, Y.SPECIAL), pick(rng, w.customers)
    dsl = {"type": "no_mix", "a": {"kind": k, "except": [{"customer": a}]}, "b": {"customer": a}}
    return dsl, f"{cap(of(a))} may not share a trip with {k} orders of other customers."


def mix_except(rng, w):
    a, b = rng.choice(Y.SPECIAL, 2, replace=False).tolist()
    n = pick(rng, w.customers)
    dsl = {"type": "no_mix", "a": {"kind": a, "except": [{"customer": n}]}, "b": {"kind": b}}
    return dsl, f"{cap(a)} orders never go with {b} orders, except the {a} orders of {n}."


def robot_kind(rng, w):
    k, r = pick(rng, ["frozen", "fragile"]), pick(rng, ["A", "B"])
    dsl = {"type": "robot_only", "who": {"kind": k}, "robot": r}
    return dsl, pick(rng, [f"{cap(k)} orders may only be carried by type {r} robots.", f"Only type {r} robots may carry {k} orders."])


def robot_customer(rng, w):
    n, r = pick(rng, w.customers), pick(rng, ["A", "B"])
    dsl = {"type": "robot_only", "who": {"customer": n}, "robot": r}
    return dsl, pick(rng, [f"{cap(of(n))} must go on a type {r} robot.", f"The orders of {n} may be carried by type {r} robots and by no others."])


def robot_station(rng, w):
    s, r = pick(rng, list(w.stations)), pick(rng, ["A", "B"])
    dsl = {"type": "robot_only", "who": {"station": s}, "robot": r}
    return dsl, f"Orders for station {s} are only for type {r} robots."


def robot_except(rng, w):
    n, r = pick(rng, w.customers), pick(rng, ["A", "B"])
    dsl = {"type": "robot_only", "who": {"priority": "urgent", "except": [{"customer": n}]}, "robot": r}
    return dsl, f"Urgent orders ride only on type {r} robots, except the orders of {n}."


def when(rng):
    return int(rng.integers(30, 81)) * 10


def dead_customer(rng, w):
    n, t = pick(rng, w.customers), when(rng)
    dsl = {"type": "deadline", "who": {"customer": n}, "t": t}
    return dsl, pick(rng, [f"{cap(of(n))} must be delivered by step {t}.", f"{n} needs everything delivered no later than step {t}."])


def dead_urgent(rng, w):
    t = when(rng)
    dsl = {"type": "deadline", "who": {"priority": "urgent"}, "t": t}
    return dsl, pick(rng, [f"All urgent orders must be delivered by step {t}.", f"No urgent order may arrive after step {t}."])


def dead_order(rng, w):
    o, t = int(rng.integers(len(w.orders))), when(rng)
    dsl = {"type": "deadline", "who": {"order": o}, "t": t}
    return dsl, pick(rng, [f"Order {o} must be delivered by step {t}.", f"Order {o} has to reach its station no later than step {t}."])


def dead_except(rng, w):
    n, t = pick(rng, w.customers), when(rng)
    dsl = {"type": "deadline", "who": {"priority": "urgent", "except": [{"customer": n}]}, "t": t}
    return dsl, f"Every urgent order except those of {n} must be delivered by step {t}."


def st_customer(rng, w):
    n, s = pick(rng, w.customers), pick(rng, list(w.stations))
    dsl = {"type": "station", "who": {"customer": n}, "station": s}
    return dsl, pick(rng, [f"All orders of {n} must be delivered to station {s}.", f"{cap(of(n))} end at station {s}."])


def st_kind(rng, w):
    k, s = pick(rng, Y.SPECIAL), pick(rng, list(w.stations))
    dsl = {"type": "station", "who": {"kind": k}, "station": s}
    return dsl, pick(rng, [f"{cap(k)} orders must be delivered to station {s}.", f"Station {s} takes every {k} order."])


def st_except(rng, w):
    k, s, n = pick(rng, Y.SPECIAL), pick(rng, list(w.stations)), pick(rng, w.customers)
    dsl = {"type": "station", "who": {"kind": k, "except": [{"customer": n}]}, "station": s}
    return dsl, f"Every {k} order except those of {n} must go to station {s}."


SHAPES = {"pick_order": [pick_last, pick_first, pick_pair, pick_except],
          "close": [close_all, close_two, close_window, close_until, close_from],
          "no_mix": [mix_kinds, mix_customers, mix_kind_customer, mix_except],
          "robot_only": [robot_kind, robot_customer, robot_station, robot_except],
          "deadline": [dead_customer, dead_urgent, dead_order, dead_except],
          "station": [st_customer, st_kind, st_except]}


def sample_rules(rng, ctx, k):
    """k rules of k different types that the validator accepts together."""
    out = []
    types = rng.permutation(list(SHAPES)).tolist()[:k]
    for kind in types:
        for _ in range(400):
            shape = pick(rng, SHAPES[kind])
            dsl, sentence = shape(rng, ctx.w)
            try:
                prims = L.expand_rule(dsl, ctx)
            except L.DslError:
                continue
            if sum(p["type"] == "deadline" for p in prims) > MAX_DEADLINE_ORDERS:
                continue
            cand = out + [{"dsl": dsl, "prims": prims, "sentence": sentence, "shape": shape.__name__}]
            if L.check(cand, ctx)[1]:
                continue
            out = cand
            break
        else:
            raise RuntimeError(f"could not sample a {kind} rule")
    return out


def distractors(rng, w, k):
    out = []
    for t in rng.choice(DISTRACTORS, k, replace=False).tolist():
        a, b = rng.choice(w.customers, 2, replace=False).tolist()
        x, y = rng.choice(Y.SPECIAL, 2, replace=False).tolist()
        s, u = rng.choice(list(w.stations), 2, replace=False).tolist()
        out.append(t.format(name=a, name2=b, c=int(rng.integers(1, 21)), s=s, s2=u, k=x, k2=y, r=pick(rng, ["A", "B"])))
    return out


L2_PROMPT = """You are a shift supervisor writing today's note for the people who run a robot warehouse.
Rewrite the rules below as ONE flowing paragraph of plain English, the way a person would write it.

Rules to include (every one of them, with exactly the same meaning):
{rules}

Background remarks to weave in among the rules (they are not rules; keep their meaning):
{distractors}

Requirements:
- Plain running text: no lists, no headings, no line breaks.
- Do not change, add or drop any customer, corridor, station, kind of product, robot type, order number or step, and do not add new conditions.
- Do not repeat a sentence as given: change the wording and the order of the words.
- When it reads naturally, refer back to something you have just mentioned ("the latter", "that corridor", "the same customer") instead of repeating its name, as long as it stays unambiguous.
- State exactly one of the rules above implicitly, never a new one: describe the situation behind it and let the reader infer the rule, without words such as must, may or only. For a closed corridor, describe the works going on in it; for a deadline, say when the customer's van leaves; for a station, say where the customer's staff will be waiting. The reader must still be able to recover the rule exactly, with all its numbers and names.
Write only the paragraph."""


def l2_prompt(sentences, background):
    return L2_PROMPT.format(rules="\n".join(f"- {s}" for s in sentences), distractors="\n".join(f"- {s}" for s in background))
