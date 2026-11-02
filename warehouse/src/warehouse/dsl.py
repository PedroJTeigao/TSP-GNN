"""The formal language of warehouse requests (phase C), on top of tsp_gnn.open_dsl: every arm writes rules, code expands them.

A rule is a JSON object:
    {"type": "pick_order", "first": SEL, "then": SEL}       on any trip, products of first are picked before products of then
    {"type": "close", "corridors": [5, 14], "t1": x, "t2": y}   no robot in those corridors in [t1, t2); both optional
    {"type": "no_mix", "a": SEL, "b": SEL}                  no trip holds an order of a and an order of b
    {"type": "robot_only", "who": SEL, "robot": "A"}        those orders only on robots of that type
    {"type": "deadline", "who": SEL, "t": 300}              those orders delivered by step t
    {"type": "station", "who": SEL, "station": "B"}         those orders delivered at that station
A selector SEL picks orders, or shelves in pick_order:
    {"kind": "frozen"}            orders with a product of that kind; in pick_order, the shelves of that kind
    {"corridor": 5}               orders with a product in that corridor; the shelves of that corridor
    {"station": "B"}              orders delivered there
    {"customer": "Dona Maria"}    the orders of a customer
    {"order": 17}                 one order
    {"priority": "urgent"}        orders of that priority
    {"all": true}                 everything
    several keys at once          all of them must hold
    {"any_of": [SEL, ...]}        union; a bare list means the same
    any of them + "except": [SEL, ...]   removed last
In pick_order a selector of orders stands for the shelves of those orders. An answer is {"rules": [...]}; items that do not fit are
dropped and counted (validate).

Scoring (score) works on primitives, so a rule may be written as one group or as many: an order pair, a shelf pair, a corridor with its
window, or an order with its value. A label rule is exact if all its primitives are predicted and no wrong primitive of the same type
touches it; a request is exact if the primitive sets are equal.

check (the validator) comes before any solver and rejects a whole rule, recording why: a cycle of pick orders, a closure that isolates
a needed shelf, an incompatibility inside one order, a robot type or station that contradicts another rule, a deadline out of reach.
"""

import json

from tsp_gnn import open_dsl as OD
from warehouse import world as Y

FOREVER = 10 ** 9
LONG = 5000                                     # a window that reaches this far is a closure for the whole episode
SERVICE = 3
DslError = OD.DslError
KIND_WORDS = {"fragile": "fragile", "frozen": "frozen", "heavy": "heavy", "normal": "normal", "cold": "frozen", "breakable": "fragile"}


class Context:
    def __init__(self, world):
        self.w = world
        self.book = {o.oid: o for o in world.orders}
        self.by_name = {c.lower(): c for c in world.customers}
        self.kinds = {o.oid: {world.shelf_kind[s] for s in o.shelves} for o in world.orders}
        self.in_corridor = {k: set(world.shelves_in(k)) for k in world.corridors}
        self.station_of = {c: s for s, c in world.stations.items()}
        self.used = sorted({s for o in world.orders for s in o.shelves})


def _kind(x):
    k = KIND_WORDS.get(str(x).lower().strip().replace(" goods", "").replace(" products", ""))
    if k is None:
        raise DslError(f"unknown kind {x!r}")
    return k


def corridor_number(x, ctx):
    s = str(x).lower().replace("corridor", "").strip()
    if not s.isdigit() or int(s) not in ctx.w.corridors:
        raise DslError(f"unknown corridor {x!r}")
    return int(s)


def _station(x, ctx):
    s = str(x).lower().replace("station", "").strip().upper()
    if s not in ctx.w.stations:
        raise DslError(f"unknown station {x!r}")
    return s


def _customer(x, ctx):
    key = str(x).lower().strip()
    if key not in ctx.by_name:
        raise DslError(f"unknown customer {x!r}")
    return ctx.by_name[key]


def _order(x, ctx):
    c = OD.F._int(x)
    if c not in ctx.book:
        raise DslError(f"order {c} out of range")
    return c


def _bare(sel, ctx):
    """A plain string or number: an order number, a corridor, a station, a kind or a customer."""
    if isinstance(sel, bool) or sel is None:
        raise DslError(f"bad selector {sel!r}")
    if isinstance(sel, (int, float)):
        return {"order": sel}
    if isinstance(sel, list):
        return {"any_of": sel}
    if isinstance(sel, str):
        s = sel.lower().strip()
        if s.isdigit():
            return {"order": int(s)}
        if s.startswith("corridor"):
            return {"corridor": s}
        if s.startswith("station"):
            return {"station": s}
        if s in ("urgent", "normal priority"):
            return {"priority": "urgent" if s == "urgent" else "normal"}
        if s in KIND_WORDS:
            return {"kind": s}
        return {"customer": sel}
    if isinstance(sel, dict):
        return sel
    raise DslError(f"bad selector {sel!r}")


KEYS = {"kind", "corridor", "station", "customer", "order", "priority", "all", "any_of", "except"}


def _picks(sel, ctx, base, one):
    """The shared walk over a selector: base is everything, one(key, value) the members of one test."""
    sel = _bare(sel, ctx)
    if sel.keys() - KEYS or not sel.keys() - {"except"}:
        raise DslError(f"bad selector keys {sorted(sel)}")
    if "any_of" in sel:
        if not isinstance(sel["any_of"], list) or not sel["any_of"]:
            raise DslError("empty any_of")
        got = set()
        for s in sel["any_of"]:
            got |= _picks(s, ctx, base, one)
    else:
        got = set(base)
    for key in sel.keys() - {"any_of", "except", "all"}:
        got &= one(key, sel[key])
    if "except" in sel:
        ex = sel["except"] if isinstance(sel["except"], list) else [sel["except"]]
        for s in ex:
            got -= _picks(s, ctx, base, one)
    if not got:
        raise DslError("empty selector")
    return got


def select_orders(sel, ctx):
    def one(key, x):
        if key == "kind":
            k = _kind(x)
            return {o for o, v in ctx.kinds.items() if k in v}
        if key == "corridor":
            shelves = ctx.in_corridor[corridor_number(x, ctx)]
            return {o.oid for o in ctx.w.orders if shelves & set(o.shelves)}
        if key == "station":
            c = ctx.w.stations[_station(x, ctx)]
            return {o.oid for o in ctx.w.orders if o.station == c}
        if key == "customer":
            c = _customer(x, ctx)
            return {o.oid for o in ctx.w.orders if o.customer == c}
        if key == "priority":
            p = str(x).lower().strip()
            if p not in ("normal", "urgent"):
                raise DslError(f"unknown priority {x!r}")
            return {o.oid for o in ctx.w.orders if o.priority == p}
        return {_order(x, ctx)}
    return _picks(sel, ctx, ctx.book, one)


def select_shelves(sel, ctx):
    def one(key, x):
        if key == "kind":
            k = _kind(x)
            return {s for s in ctx.used if ctx.w.shelf_kind[s] == k}
        if key == "corridor":
            return ctx.in_corridor[corridor_number(x, ctx)] & set(ctx.used)
        return {s for o in select_orders({key: x}, ctx) for s in ctx.book[o].shelves}
    return _picks(sel, ctx, ctx.used, one)


def _times(rule):
    t1 = rule.get("t1")
    t2 = rule.get("t2")
    t1, t2 = 0 if t1 is None else int(float(t1)), FOREVER if t2 is None else int(float(t2))
    if t1 < 0 or t1 >= t2:
        raise DslError("bad times")
    return t1, (FOREVER if t2 >= LONG else t2)


def _field(rule, *names):
    for k in names:
        if k in rule:
            return rule[k]
    raise DslError(f"missing field {names[0]}")


def expand_rule(rule, ctx):
    """The primitives of one rule; DslError if it is malformed or names nothing."""
    if not isinstance(rule, dict):
        raise DslError("rule is not an object")
    kind = rule.get("type")
    if kind == "pick_order":
        first, then = select_shelves(_field(rule, "first", "before"), ctx), select_shelves(_field(rule, "then", "after"), ctx)
        if not first - then and not then - first:
            raise DslError("the two sides are the same")
        return [{"type": kind, "first": sorted(first), "then": sorted(then)}]
    if kind == "close":
        got = _field(rule, "corridors", "corridor")
        got = got if isinstance(got, list) else [got]
        if not got:
            raise DslError("no corridor")
        t1, t2 = _times(rule)
        return [{"type": kind, "corridor": k, "t1": t1, "t2": t2} for k in sorted({corridor_number(x, ctx) for x in got})]
    if kind == "no_mix":
        a, b = select_orders(_field(rule, "a"), ctx), select_orders(_field(rule, "b"), ctx)
        pairs = {(min(x, y), max(x, y)) for x in a for y in b if x != y}
        return [{"type": kind, "pair": list(p)} for p in sorted(pairs)] + [{"type": "same_order", "order": x} for x in sorted(a & b)]
    if kind == "robot_only":
        robot = str(_field(rule, "robot")).upper().replace("TYPE", "").strip()
        if robot not in ("A", "B"):
            raise DslError(f"unknown robot type {robot!r}")
        return [{"type": kind, "order": o, "robot": robot} for o in sorted(select_orders(_field(rule, "who"), ctx))]
    if kind == "deadline":
        t = int(float(_field(rule, "t")))
        if t <= 0:
            raise DslError("bad time")
        return [{"type": kind, "order": o, "t": t} for o in sorted(select_orders(_field(rule, "who"), ctx))]
    if kind == "station":
        s = _station(_field(rule, "station"), ctx)
        return [{"type": kind, "order": o, "station": s} for o in sorted(select_orders(_field(rule, "who"), ctx))]
    raise DslError(f"unknown type {kind!r}")


def validate(obj, ctx):
    """(accepted, errors): accepted has {"dsl", "prims"} per rule that expands, errors one message per item that does not."""
    if not isinstance(obj, dict) or not isinstance(obj.get("rules"), list):
        return [], ["no rules list"]
    out, errors = [], []
    for item in obj["rules"]:
        try:
            out.append({"dsl": item, "prims": expand_rule(item, ctx)})
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
            errors.append(f"{type(e).__name__}: {e}")
    return out, errors


# ---------------------------------------------------------------------------------------------------- scoring

def keys_of(p):
    t = p["type"]
    if t == "pick_order":
        return {(t, a, b) for a in p["first"] for b in p["then"] if a != b}
    if t == "close":
        return {(t, p["corridor"], p["t1"], p["t2"])}
    if t == "no_mix":
        return {(t, *p["pair"])}
    if t == "same_order":
        return {(t, p["order"])}
    if t == "robot_only":
        return {(t, p["order"], p["robot"])}
    if t == "deadline":
        return {(t, p["order"], p["t"])}
    return {(t, p["order"], p["station"])}


def keyset(prims):
    return set().union(*map(keys_of, prims)) if prims else set()


def touched(key):
    """What a primitive is about."""
    t = key[0]
    if t in ("pick_order", "no_mix"):
        return {key[1], key[2]}
    return {key[1]}


def score(label, pred_prims):
    """label: list of {"prims"}. {"correct": per-rule bools, "exact": the request, "wrong": how many wrong primitives}."""
    L = keyset([p for r in label for p in r["prims"]])
    P = keyset(pred_prims)
    wrong = P - L
    by_type = {}
    for w in wrong:
        by_type.setdefault(w[0], set()).update(touched(w))
    correct = []
    for r in label:
        keys = keyset(r["prims"])
        about = set().union(*map(touched, keys))
        kind = next(iter(keys))[0]
        correct.append(bool(keys <= P and not by_type.get(kind, set()) & about))
    return {"correct": correct, "exact": P == L, "wrong": len(wrong)}


def flip(key):
    return ("pick_order", key[2], key[1]) if key[0] == "pick_order" else key


def classify_error(rule, pred_prims, ctx):
    """Why a label rule was missed: inverted, exception lost, wrong value, wrong selector or missing."""
    keys, P = keyset(rule["prims"]), keyset(pred_prims)
    kind = next(iter(keys))[0]
    flipped = {flip(k) for k in keys}
    if kind == "pick_order" and len(flipped & P) > len(keys & P):
        return "inverted"
    if kind == "robot_only":
        other = {(k[0], k[1], "B" if k[2] == "A" else "A") for k in keys}
        if len(other & P) > len(keys & P):
            return "inverted"
    if "except" in json.dumps(rule["dsl"]):
        try:
            full = keyset(expand_rule(OD.without_except(rule["dsl"]), ctx))
        except (KeyError, TypeError, ValueError):
            full = set()
        if (full - keys) & P:
            return "exception lost"
    same = {p for p in P if p[0] == kind}
    if kind in ("close", "deadline", "station") and not keys & P:
        if {k[1] for k in keys} <= {p[1] for p in same}:
            return "wrong value"
    about = set().union(*map(touched, keys))
    if keys & P or any(touched(p) & about for p in same):
        return "wrong selector"
    return "missing"


def invented(accepted, label):
    """The accepted rules that overlap no label primitive of their type: rules made up. Returns their DSL."""
    L = keyset([p for r in label for p in r["prims"]])
    out = []
    for r in accepted:
        keys = keyset(r["prims"])
        kinds = {k[0] for k in keys}
        about = set().union(*[touched(k) for k in L if k[0] in kinds]) if any(k[0] in kinds for k in L) else set()
        if not keys & L and not any(touched(k) & about for k in keys):
            out.append(r["dsl"])
    return out


# ---------------------------------------------------------------------------------------------------- validator

def reachable(ctx, closed):
    """The free cells a robot can reach from the parking rows with those cells shut."""
    wh = ctx.w.wh
    seen = {c for c in wh.parking if c not in closed}
    todo = list(seen)
    while todo:
        v = todo.pop()
        for u in wh.neighbours[v]:
            if u not in seen and u not in closed:
                seen.add(u)
                todo.append(u)
    return seen


def fastest(ctx, oid, station):
    """A lower bound on when an order can be delivered: the farthest shelf on the way of the closest robot, plus the services."""
    wh, o = ctx.w.wh, ctx.book[oid]
    to_station = wh.dist_to(station)
    walk = 0
    for s in o.shelves:
        cell = wh.access[s]
        walk = max(walk, min(wh.dist_to(cell)[p] for p in ctx.w.starts) + to_station[cell])
    return walk + SERVICE * (len(o.shelves) + 1)


def rule_stage(rule):
    """Deadlines come last, once the stations and closures that bound them are known."""
    return any(p["type"] == "deadline" for p in rule["prims"])


def check(rules, ctx):
    """The validator. rules: list of {"prims"} in answer order. Returns (accepted primitives, [{"rule": k, "reason": ...}])."""
    heavy = {o for o, v in ctx.kinds.items() if "heavy" in v}
    pairs, closed, windows, robots, stations, deadlines = set(), set(), [], {}, {}, {}
    accepted, rejected = [], []
    for k, r in sorted(enumerate(rules), key=lambda x: (rule_stage(x[1]), x[0])):
        ps = r["prims"]
        reason = None
        new_pairs = {(a, b) for p in ps if p["type"] == "pick_order" for a in p["first"] for b in p["then"] if a != b}
        shut = {c for p in ps if p["type"] == "close" and p["t2"] == FOREVER and p["t1"] == 0 for c in ctx.w.corridors[p["corridor"]]}
        new_rob = {p["order"]: p["robot"] for p in ps if p["type"] == "robot_only"}
        new_st = {p["order"]: p["station"] for p in ps if p["type"] == "station"}
        new_dl = {p["order"]: p["t"] for p in ps if p["type"] == "deadline"}
        new_win = [(ctx.w.corridors[p["corridor"]], p["t1"], p["t2"]) for p in ps if p["type"] == "close" and not (p["t2"] == FOREVER and p["t1"] == 0)]
        if new_pairs and OD._has_cycle(pairs | new_pairs):
            reason = "pick order cycle"
        elif any(p["type"] == "same_order" for p in ps):
            reason = "incompatible with itself: one order is on both sides"
        elif shut or any(t2 == FOREVER for _, _, t2 in new_win):
            all_shut = closed | shut | {c for cells, _, t2 in new_win if t2 == FOREVER for c in cells}
            seen = reachable(ctx, all_shut)
            if any(ctx.w.wh.access[s] not in seen for s in ctx.used):
                reason = "closure isolates a shelf"
        if reason is None and any(o in heavy and x == "B" for o, x in new_rob.items()):
            reason = "heavy order on a robot that cannot carry it"
        if reason is None and any(robots.get(o, x) != x for o, x in new_rob.items()):
            reason = "two robot types for one order"
        if reason is None and any(stations.get(o, x) != x for o, x in new_st.items()):
            reason = "two stations for one order"
        if reason is None and new_dl:
            for o, t in new_dl.items():
                cell = ctx.w.stations[stations[o]] if o in stations else ctx.book[o].station
                if fastest(ctx, o, cell) > t:
                    reason = "deadline out of reach"
                elif any(set(ctx.w.corridors[c]) & {ctx.w.wh.access[s] for s in ctx.book[o].shelves} and a <= t and b >= t for c, a, b in windows):
                    reason = "deadline inside a closure of its corridor"
        if reason:
            rejected.append({"rule": k, "reason": reason})
            continue
        pairs |= new_pairs
        closed |= shut | {c for cells, _, t2 in new_win if t2 == FOREVER for c in cells}
        robots.update(new_rob)
        stations.update(new_st)
        deadlines.update(new_dl)
        windows += [(p["corridor"], p["t1"], p["t2"]) for p in ps if p["type"] == "close"]
        accepted += ps
    return accepted, sorted(rejected, key=lambda x: x["rule"])


# ---------------------------------------------------------------------------------------------------- prompts

SPEC = """You translate the operating rules of a robot warehouse, written in English, into a formal JSON list.
Robots collect orders from shelves and deliver them to stations. One trip of a robot carries up to 3 orders. Time is counted in steps.
The legend tells you what exists in today's warehouse.

Rule types:
- pick_order: on any trip, the products of "first" are picked before the products of "then".
- close: no robot may be in those corridors from step t1 to step t2. Leave out t1 and t2 if the closure lasts all day; leave out t1 if it starts now, t2 if it never ends.
- no_mix: no trip may hold an order of "a" together with an order of "b".
- robot_only: the orders of "who" may only be carried by robots of that type.
- deadline: the orders of "who" must be delivered by step t.
- station: the orders of "who" must be delivered at that station.

Selectors say which orders (or, in pick_order, which products) a rule is about:
- a kind of product: {"kind": "frozen"}  (an order "of a kind" is an order that has a product of that kind)
- a corridor: {"corridor": 5}
- a station: {"station": "B"}
- a customer: {"customer": "Dona Maria"}
- one order: {"order": 17}
- a priority: {"priority": "urgent"}
- everything: {"all": true}
- two conditions at once: {"kind": "frozen", "station": "B"}
- a union: {"any_of": [{"kind": "frozen"}, {"customer": "Dona Maria"}]}
- add "except" to any selector to remove some: {"kind": "frozen", "except": [{"customer": "Dona Maria"}]}
Do not expand a group yourself: write the selector.

Instructions:
- Output every rule stated or clearly implied by the text, once. Some sentences contain no rule (background information): ignore those.
- "Last" means every other product comes first: {"first": {"all": true, "except": [{"kind": "frozen"}]}, "then": {"kind": "frozen"}}.
- Times are numbers of steps, as written in the text.
Answer only with JSON of the form:
{"rules": [
  {"type": "pick_order", "first": <selector>, "then": <selector>},
  {"type": "close", "corridors": [<number>, ...], "t1": <number>, "t2": <number>},
  {"type": "no_mix", "a": <selector>, "b": <selector>},
  {"type": "robot_only", "who": <selector>, "robot": "A"},
  {"type": "deadline", "who": <selector>, "t": <number>},
  {"type": "station", "who": <selector>, "station": "B"} ]}"""

# 3 worked examples in the style of L1, hand-written, with customers that are not in the worlds of the test set
EXAMPLES = [
    ("- Heavy products must be picked before fragile products on every trip.\n- Corridor 7 is closed from step 150 to step 300.",
     {"rules": [{"type": "pick_order", "first": {"kind": "heavy"}, "then": {"kind": "fragile"}},
                {"type": "close", "corridors": [7], "t1": 150, "t2": 300}]}),
    ("- Frozen orders never share a trip with fragile orders, except the orders of Pastelaria Sol.\n- All orders of Pastelaria Sol must be delivered to station C.",
     {"rules": [{"type": "no_mix", "a": {"kind": "frozen", "except": [{"customer": "Pastelaria Sol"}]}, "b": {"kind": "fragile", "except": [{"customer": "Pastelaria Sol"}]}},
                {"type": "station", "who": {"customer": "Pastelaria Sol"}, "station": "C"}]}),
    ("- The orders of Mestre Rui must be delivered by step 400.\n- Only type A robots may carry urgent orders.\n- Corridors 12 and 16 are closed.",
     {"rules": [{"type": "deadline", "who": {"customer": "Mestre Rui"}, "t": 400},
                {"type": "robot_only", "who": {"priority": "urgent"}, "robot": "A"},
                {"type": "close", "corridors": [12, 16]}]}),
]


def system_prompt():
    ex = "\n\n".join(f"Example {k + 1}.\nText:\n{t}\nAnswer:\n{json.dumps(a)}" for k, (t, a) in enumerate(EXAMPLES))
    return SPEC + "\n\nWorked examples:\n\n" + ex


def user_prompt(world, text):
    return f"Legend:\n{Y.legend(world)}\nText:\n{text}\nAnswer:"
