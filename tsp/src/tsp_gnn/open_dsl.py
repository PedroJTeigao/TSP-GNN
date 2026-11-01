"""A small formal language for open-language route rules (phase 5, step 3).

Every method (LLMs and parsers) writes DSL rules; this code expands them into the primitive rules of tsp_gnn.rules, so
"what the text means" and "what the solver gets" are separated. A rule is a JSON object:
    {"type": "precedence",     "before": SEL, "after": SEL}     every place of before ahead of every place of after
    {"type": "forbidden_edge", "a": SEL, "b": SEL}              no direct leg between any place of a and any place of b
    {"type": "mandatory_edge", "a": SEL, "b": SEL}              every place of a next to every place of b
    {"type": "time_window",    "who": SEL, "t1": x, "t2": y}    service at every place of who starts in [x, y]
    {"type": "soft_zone",      "zone": "north"}                 one of the 5 fixed zones (rules.ZONES)
A selector SEL picks places of the legend (the table of named places given with the instance):
    17 or {"id": 17}                  a customer by number (it does not have to be in the legend)
    "Rua Azul 3" or {"name": ...}     a place by name
    {"type_is": "hospital"}           every place of a type
    {"area_is": "Alfama"}             every place of an area
    {"type_is": ..., "area_is": ...}  both at once
    {"nearest": {"k": 3, "of": "Rua Azul 3"}}   the k legend places closest to a place (never itself); with type_is /
                                      area_is, the k closest among those
    {"any_of": [SEL, SEL, ...]}       union
    any of them + "except": [SEL, ...]   those places are removed last, so "the 3 closest except Y" can leave 2
An answer is {"rules": [...]}. Items that do not fit are dropped and counted (validate), as in tsp_gnn.formalize.

Scoring (score) works on primitives, so a method may write a rule as one group or as many single constraints:
    P = all primitives predicted, L = all primitives of the label;
    a label rule r is exact if all its primitives are in P and no wrong primitive (in P, not in L) of the same type
    touches a customer of r; an instance is exact if P == L.

validate_for_solver is the code check in front of LKH-3: rules that cannot be solved are rejected and recorded, and
the route is computed without them.
"""

import math

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import rules as R

KINDS = ("hospital", "school", "warehouse", "shop", "house", "pharmacy", "office")
PLURALS = {"pharmacies": "pharmacy", "warehouses": "warehouse", "hospitals": "hospital", "schools": "school",
           "shops": "shop", "houses": "house", "offices": "office"}
MAX_DIRECTIONS = 3            # mandatory edges in SOP / TSPTW: LKH-3 runs once per direction combination (2^edges)


class DslError(ValueError):
    pass


class Context:
    """What a selector can refer to: the legend (list of {id, name, kind, area}), coordinates and n."""

    def __init__(self, legend, coords, n):
        self.legend = legend
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = n
        self.by_name = {p["name"].lower().strip(): p["id"] for p in legend}
        self.areas = {p["area"].lower() for p in legend if p.get("area")}

    def place(self, name):
        key = str(name).lower().strip()
        if key.startswith("the ") and key not in self.by_name:
            key = key[4:]
        if key not in self.by_name:
            raise DslError(f"unknown place {name!r}")
        return self.by_name[key]


def _customer(x, ctx):
    c = F._int(x)                                # raises ValueError if it is not an integer
    if not 1 <= c < ctx.n:
        raise DslError(f"customer {c} out of range")
    return c


def _single(sel, ctx):
    """The one place a {"of": ...} field names: a name, a number or a selector with one place."""
    got = expand_selector(sel, ctx)
    if len(got) != 1:
        raise DslError(f"expected one place, got {len(got)}")
    return got[0]


def _kind(x):
    k = str(x).lower().strip()
    k = PLURALS.get(k, k)
    if k not in KINDS:
        raise DslError(f"unknown type {x!r}")
    return k


def expand_selector(sel, ctx):
    """Sorted list of the customers a selector picks; DslError / ValueError if it is malformed or empty."""
    if isinstance(sel, bool) or sel is None:
        raise DslError(f"bad selector {sel!r}")
    if isinstance(sel, (int, float)):
        return [_customer(sel, ctx)]
    if isinstance(sel, str):
        s = sel.strip()
        return [_customer(s, ctx)] if s.isdigit() else [ctx.place(s)]
    if isinstance(sel, list):                    # a bare list is read as a union
        sel = {"any_of": sel}
    if not isinstance(sel, dict):
        raise DslError(f"bad selector {sel!r}")
    known = {"id", "name", "type_is", "area_is", "nearest", "any_of", "except"}
    if not sel.keys() & (known - {"except"}) or sel.keys() - known:
        raise DslError(f"bad selector keys {sorted(sel)}")
    if "id" in sel:
        got = {_customer(sel["id"], ctx)}
    elif "name" in sel:
        got = {ctx.place(sel["name"])}
    elif "any_of" in sel:
        if not isinstance(sel["any_of"], list) or not sel["any_of"]:
            raise DslError("empty any_of")
        got = set()
        for s in sel["any_of"]:
            got |= set(expand_selector(s, ctx))
    else:
        pool = ctx.legend
        if "type_is" in sel:
            kind = _kind(sel["type_is"])
            pool = [p for p in pool if p["kind"] == kind]
        if "area_is" in sel:
            area = str(sel["area_is"]).lower().strip()
            if area not in ctx.areas:
                raise DslError(f"unknown area {sel['area_is']!r}")
            pool = [p for p in pool if p["area"].lower() == area]
        if "nearest" in sel:
            near = sel["nearest"]
            if not isinstance(near, dict) or "k" not in near or "of" not in near:
                raise DslError("nearest needs k and of")
            k = F._int(near["k"])
            of = _single(near["of"], ctx)
            d = np.linalg.norm(ctx.coords[[p["id"] for p in pool]] - ctx.coords[of], axis=1) if pool else []
            order = sorted((float(x), p["id"]) for x, p in zip(d, pool) if p["id"] != of)
            if k < 1:
                raise DslError("k < 1")
            got = {c for _, c in order[:k]}
        else:
            got = {p["id"] for p in pool}
    if "except" in sel:
        ex = sel["except"] if isinstance(sel["except"], list) else [sel["except"]]
        for s in ex:
            got -= set(expand_selector(s, ctx))
    if not got:
        raise DslError("empty selector")
    return sorted(got)


def _side(rule, *names):
    for k in names:
        if k in rule:
            return rule[k]
    raise DslError(f"missing field {names[0]}")


def expand_rule(rule, ctx):
    """The primitive rules (rules.py form: {"type", "formal"}) of one DSL rule, without repeats. Pairs of a place with
    itself are dropped; a rule that expands to nothing is an error."""
    if not isinstance(rule, dict):
        raise DslError("rule is not an object")
    kind = rule.get("type")
    prims = []
    if kind == "precedence":
        for a in expand_selector(_side(rule, "before"), ctx):
            for b in expand_selector(_side(rule, "after"), ctx):
                if a != b:
                    prims.append({"type": kind, "formal": {"before": a, "after": b}})
    elif kind in ("forbidden_edge", "mandatory_edge"):
        for a in expand_selector(_side(rule, "a", "i"), ctx):
            for b in expand_selector(_side(rule, "b", "j"), ctx):
                if a != b:
                    i, j = min(a, b), max(a, b)
                    prims.append({"type": kind, "formal": {"i": i, "j": j}})
    elif kind == "time_window":
        t1, t2 = float(rule["t1"]), float(rule["t2"])
        if not (math.isfinite(t1) and math.isfinite(t2)) or t1 > t2:
            raise DslError("bad times")
        for c in expand_selector(_side(rule, "who", "customer"), ctx):
            prims.append({"type": kind, "formal": {"customer": c, "t1": t1, "t2": t2}})
    elif kind == "soft_zone":
        zone = str(rule["zone"]).lower().strip()
        if zone not in R.ZONES:
            raise DslError(f"unknown zone {zone!r}")
        prims.append({"type": kind, "formal": {"zone": zone, "rect": list(R.ZONES[zone]), "weight": R.SOFT_WEIGHT}})
    else:
        raise DslError(f"unknown type {kind!r}")
    seen, out = set(), []
    for p in prims:
        if F.rule_key(p) not in seen:
            seen.add(F.rule_key(p))
            out.append(p)
    if not out:
        raise DslError("rule expands to nothing")
    return out


def validate(obj, ctx):
    """(accepted, errors): accepted is a list of {"dsl", "prims"}, one per rule that expands; errors has one message
    per item that does not. An answer that is not {"rules": [...]} gives ([], ["no rules list"])."""
    if not isinstance(obj, dict) or not isinstance(obj.get("rules"), list):
        return [], ["no rules list"]
    out, errors = [], []
    for item in obj["rules"]:
        try:
            out.append({"dsl": item, "prims": expand_rule(item, ctx)})
        except (KeyError, TypeError, ValueError, IndexError) as e:
            errors.append(f"{type(e).__name__}: {e}")
    return out, errors


# ---------------------------------------------------------------------------------------------------- scoring

def keyset(prims):
    return {F.rule_key(p) for p in prims}


def _key_customers(key):
    kind = key[0]
    if kind == "precedence":
        return {key[1], key[2]}
    if kind in ("forbidden_edge", "mandatory_edge"):
        return set(key[1])
    if kind == "time_window":
        return {key[1]}
    return {key[1]}                               # soft zone: the zone name


def score(label, pred_prims):
    """label: list of {"prims": [...]} (one per label rule); pred_prims: flat list of predicted primitives.
    Returns {"correct": per-rule bools, "exact": instance exact, "wrong": predicted primitives not in the label}."""
    L = set().union(*[keyset(r["prims"]) for r in label]) if label else set()
    P = keyset(pred_prims)
    wrong = P - L
    correct = []
    for r in label:
        keys = keyset(r["prims"])
        touched = set().union(*[_key_customers(k) for k in keys])
        kinds = {k[0] for k in keys}
        ok = keys <= P and not any(w[0] in kinds and _key_customers(w) & touched for w in wrong)
        correct.append(bool(ok))
    return {"correct": correct, "exact": P == L, "wrong": sorted(map(repr, wrong))}


def _flip(key):
    return ("precedence", key[2], key[1]) if key[0] == "precedence" else key


def without_except(dsl):
    """The rule with every "except" removed (recursive), to detect a lost exception."""
    if isinstance(dsl, dict):
        return {k: without_except(v) for k, v in dsl.items() if k != "except"}
    if isinstance(dsl, list):
        return [without_except(x) for x in dsl]
    return dsl


def classify_error(rule, pred_prims, ctx):
    """Why a label rule was missed: missing, inverted direction, lost exception, wrong type, wrong selector, or
    wrong times."""
    keys = keyset(rule["prims"])
    P = keyset(pred_prims)
    kind = next(iter(keys))[0]
    if kind == "precedence" and {_flip(k) for k in keys} & P and len({_flip(k) for k in keys} & P) >= len(keys & P):
        return "inverted"
    if "except" in str(rule["dsl"]):
        try:
            full = keyset(expand_rule(without_except(rule["dsl"]), ctx))
        except (KeyError, TypeError, ValueError):
            full = set()
        if full - keys and (full - keys) & P:
            return "exception lost"
    same_type = {p for p in P if p[0] == kind}
    touched = set().union(*[_key_customers(k) for k in keys])
    if kind == "time_window" and any(p[1] in touched for p in same_type) and not keys & P:
        if {p[1] for p in same_type} >= touched:
            return "wrong times"
    if keys & P or any(_key_customers(p) & touched for p in same_type):
        return "wrong selector"
    if any(_key_customers(p) & touched for p in P):
        return "wrong type"
    return "missing"


# ---------------------------------------------------------------------------------------------------- code check

def _has_cycle(edges):
    """True if the directed graph {(a, b)} has a cycle (Kahn)."""
    succ, indeg = {}, {}
    for a, b in edges:
        succ.setdefault(a, set()).add(b)
        indeg.setdefault(a, 0)
        indeg.setdefault(b, 0)
    for a in succ:
        for b in succ[a]:
            indeg[b] += 1
    todo = [v for v, d in indeg.items() if d == 0]
    seen = 0
    while todo:
        v = todo.pop()
        seen += 1
        for b in succ.get(v, ()):
            indeg[b] -= 1
            if indeg[b] == 0:
                todo.append(b)
    return seen != len(indeg)


def _mandatory_problem(edges, n):
    """Why mandatory edges cannot all hold (a customer with more than 2, or a closed sub-tour); None if they can."""
    deg, parent = {}, list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    closed = 0
    for i, j in edges:
        deg[i] = deg.get(i, 0) + 1
        deg[j] = deg.get(j, 0) + 1
        a, b = find(i), find(j)
        closed += a == b
        parent[a] = b
    if any(d > 2 for d in deg.values()):
        return "a customer with more than 2 mandatory edges"
    return "mandatory edges close a sub-tour" if closed else None


def validate_for_solver(rules, coords):
    """Code check before LKH-3. rules: list of {"prims": [...]} in answer order. A whole rule is rejected if one of its
    primitives cannot be solved together with the rules accepted so far. Mandatory edges are checked last, so that the
    number of direction combinations is known. Returns (accepted primitives, [{"rule": k, "reason": ...}])."""
    c = np.asarray(coords, dtype=np.float64)
    n = len(c)
    prec, forb, mand, windows, zones = set(), set(), set(), {}, []
    rejected = []

    def reject(k, reason):
        rejected.append({"rule": k, "reason": reason})

    def order(pass_mand):
        return [(k, r) for k, r in enumerate(rules)
                if any(p["type"] == "mandatory_edge" for p in r["prims"]) == pass_mand]

    for k, r in order(False) + order(True):
        kinds = {p["type"] for p in r["prims"]}
        new_prec = {(p["formal"]["before"], p["formal"]["after"]) for p in r["prims"] if p["type"] == "precedence"}
        new_forb = {(p["formal"]["i"], p["formal"]["j"]) for p in r["prims"] if p["type"] == "forbidden_edge"}
        new_mand = {(p["formal"]["i"], p["formal"]["j"]) for p in r["prims"] if p["type"] == "mandatory_edge"}
        new_win = {p["formal"]["customer"]: (p["formal"]["t1"], p["formal"]["t2"])
                   for p in r["prims"] if p["type"] == "time_window"}
        if new_prec and _has_cycle(prec | new_prec):
            reject(k, "precedence cycle")
        elif new_prec and windows:
            reject(k, "precedence with a time window (LKH-3 cannot take both)")
        elif new_win and prec:
            reject(k, "time window with a precedence (LKH-3 cannot take both)")
        elif any(new_win.get(v) is not None and windows.get(v, new_win[v]) != new_win[v] for v in new_win):
            reject(k, "two different windows for one customer")
        elif any(t2 < float(np.linalg.norm(c[v] - c[0])) for v, (_, t2) in new_win.items()):
            reject(k, "impossible window (closes before the customer can be reached)")
        elif new_forb & mand or new_mand & (forb | new_forb):
            reject(k, "an edge both mandatory and forbidden")
        elif new_mand and _mandatory_problem(sorted(mand | new_mand), n):
            reject(k, _mandatory_problem(sorted(mand | new_mand), n))
        elif new_mand and (prec or windows) and len(mand | new_mand) > MAX_DIRECTIONS:
            reject(k, f"more than {MAX_DIRECTIONS} mandatory edges in SOP / TSPTW")
        else:
            prec |= new_prec
            forb |= new_forb
            mand |= new_mand
            windows.update(new_win)
            zones += [p for p in r["prims"] if p["type"] == "soft_zone"]
        assert kinds                               # expand_rule never gives an empty rule
    prims = [{"type": "precedence", "formal": {"before": a, "after": b}} for a, b in sorted(prec)]
    prims += [{"type": "forbidden_edge", "formal": {"i": i, "j": j}} for i, j in sorted(forb)]
    prims += [{"type": "mandatory_edge", "formal": {"i": i, "j": j}} for i, j in sorted(mand)]
    prims += [{"type": "time_window", "formal": {"customer": v, "t1": t[0], "t2": t[1]}}
              for v, t in sorted(windows.items())]
    seen, soft = set(), []
    for p in zones:
        if p["formal"]["zone"] not in seen:
            seen.add(p["formal"]["zone"])
            soft.append(p)
    return prims + soft, sorted(rejected, key=lambda x: x["rule"])


# ---------------------------------------------------------------------------------------------------- prompts

SPEC = """You translate the route rules of a vehicle routing instance, written in English, into a formal JSON list.
Customer 0 is the depot. The legend lists named places: number | name | type | area | x y.

Rule types:
- precedence: every place of "before" is visited at some point earlier in the route than every place of "after".
- forbidden_edge: the route may not go directly between any place of "a" and any place of "b" (either direction).
- mandatory_edge: every place of "a" is visited consecutively with every place of "b" (either order).
- time_window: service at every place of "who" starts between times t1 and t2.
- soft_zone: driving inside the zone ("north", "south", "east", "west" or "central") should be avoided if possible.

Selectors say which places a rule is about:
- a customer number: 17
- a place name: "Rua Azul 3"
- every place of a type: {"type_is": "hospital"}  (types: hospital, school, warehouse, shop, house, pharmacy, office)
- every place of an area: {"area_is": "Alfama"}
- both: {"type_is": "school", "area_is": "Alfama"}
- the k legend places closest to a place, not counting itself: {"nearest": {"k": 3, "of": "Rua Azul 3"}}
- the k closest of one type: {"nearest": {"k": 1, "of": "Rua Azul 3"}, "type_is": "school"}
- a union: {"any_of": ["Rua Azul 3", {"type_is": "school"}]}
- add "except" to any selector to remove places: {"type_is": "school", "except": ["Rua Azul 3"]}
Groups refer to legend places only. Do not expand a group yourself: write the selector.

Instructions:
- Output every rule stated in the text, once. Some sentences contain no rule (background information): ignore those.
- The fixed description of the depot, speed and zones at the start is not a rule.
- Times are numbers, as written in the text.
Answer only with JSON of the form:
{"rules": [
  {"type": "precedence", "before": <selector>, "after": <selector>},
  {"type": "forbidden_edge", "a": <selector>, "b": <selector>},
  {"type": "mandatory_edge", "a": <selector>, "b": <selector>},
  {"type": "time_window", "who": <selector>, "t1": <number>, "t2": <number>},
  {"type": "soft_zone", "zone": "<zone>"} ]}"""

# 3 worked examples in the style of L0 (hand-written for n = 60, not in the test set; same as the first 3 of step 2)
EXAMPLES = [
    (F.FEW_SHOT[0][0], {"rules": [{"type": "precedence", "before": 12, "after": 7},
                                  {"type": "soft_zone", "zone": "east"}]}),
    (F.FEW_SHOT[1][0], {"rules": [{"type": "forbidden_edge", "a": 18, "b": 52},
                                  {"type": "mandatory_edge", "a": 5, "b": 40}]}),
    (F.FEW_SHOT[2][0], {"rules": [{"type": "time_window", "who": 23, "t1": 1.25, "t2": 2.4},
                                  {"type": "time_window", "who": 51, "t1": 0.8, "t2": 1.95}]}),
]

NO_LEGEND = "(no separate legend: any named places are listed in the text)"


def system_prompt():
    import json
    ex = "\n\n".join(f"Example {k + 1}.\nLegend:\n{NO_LEGEND}\nText:\n{t}\nAnswer:\n{json.dumps(a)}"
                     for k, (t, a) in enumerate(EXAMPLES))
    return SPEC + "\n\nWorked examples:\n\n" + ex


def legend_block(legend):
    return "\n".join(f"{p['id']} | {p['name']} | {p['kind']} | {p['area']} | {p['x']:.2f} {p['y']:.2f}" for p in legend)


def user_prompt(legend_text, text, n):
    return f"Instance with n = {n} (customers 1..{n - 1}).\nLegend:\n{legend_text}\nText:\n{text}\nAnswer:"
