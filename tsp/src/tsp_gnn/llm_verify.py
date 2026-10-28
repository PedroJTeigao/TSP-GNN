"""LLM as a verifier of routes (phase 5, step 2, part D; the supervisor's idea).

The 7B gets the rule text (text_v2b) and a route written as a list of visits with service-start times, and answers in
JSON which rules are violated. Variant "with report": it also gets the code checker's report and only has to
confirm / explain it. The code checker (tsp_gnn.rules) is right by construction and is the reference.

Routes: the oracle route of each instance (no violation) and the same route with ONE injected error (inject_error),
checked in code to break exactly one rule:
    precedence      swap the two customers of a precedence (b now visited before a);
    forbidden_edge  move j right after i (the forbidden leg is driven);
    mandatory_edge  move j to a random position away from i (the pair is split);
    time_window     move the customer later in the route, until service starts after t2 (window missed).
Soft zones cannot be violated, so they get no injected error.

The LLM names violated rules by quoting the customers involved; answers are mapped back to label rules by
match_claims (same type and same customers, or the zone).
"""

import json

import numpy as np

from tsp_gnn import rules as R

INJECTABLE = ("precedence", "forbidden_edge", "mandatory_edge", "time_window")


def _move(tour, c, new_pos):
    t = [v for v in tour.tolist() if v != c]
    t.insert(new_pos, c)
    return np.array(t)


def inject_error(rng, coords, tour, rules, kind, tries=200):
    """A copy of tour that breaks exactly one hard rule, of type kind; (route, index of that rule) or None."""
    tour = np.asarray(tour)
    n = len(tour)
    cands = [k for k, r in enumerate(rules) if r["type"] == kind]
    for _ in range(tries):
        k = int(rng.choice(cands))
        f = rules[k]["formal"]
        pos = {int(v): p for p, v in enumerate(tour)}
        if kind == "precedence":
            t = tour.copy()
            a, b = pos[f["before"]], pos[f["after"]]
            t[a], t[b] = t[b], t[a]
        elif kind == "forbidden_edge":
            i, j = (f["i"], f["j"]) if rng.random() < 0.5 else (f["j"], f["i"])
            t = [v for v in tour.tolist() if v != j]
            t.insert(t.index(i) + 1, j)
            t = np.array(t)
        elif kind == "mandatory_edge":
            j = f["j"] if rng.random() < 0.5 else f["i"]
            t = _move(tour, j, int(rng.integers(1, n)))
        else:                                                # move the customer later until its window is missed
            c = f["customer"]
            p0 = pos[c]
            if p0 >= n - 1:
                continue
            t = _move(tour, c, int(rng.integers(p0 + 1, n)))
        if t[0] == 0 and R.violated_rules(rules, coords, t) == [k]:
            return t, k
    return None


def route_text(coords, tour, rules, names=None):
    """The route as the LLM sees it: one visit per line with the time service starts (waiting included)."""
    times = R.service_times(coords, tour, rules)
    names = names or {}
    lines = []
    for p, (v, t) in enumerate(zip(tour, times)):
        v = int(v)
        label = "depot (customer 0)" if v == 0 else f"customer {v}" + (f" ({names[v]})" if v in names else "")
        lines.append(f"{p}. {label}, t = {t:.2f}")
    back = float(times[-1] + np.linalg.norm(np.asarray(coords[tour[-1]]) - np.asarray(coords[0])))
    lines.append(f"{len(tour)}. back at the depot, t = {back:.2f}")
    return "\n".join(lines)


def code_report(rules, coords, tour):
    """The code checker's report, in words (for the 'with report' variant)."""
    bad = set(R.violated_rules(rules, coords, tour))
    out = []
    for k, r in enumerate(rules):
        if r["type"] == "soft_zone":
            continue
        f = r["formal"]
        what = {"precedence": f"precedence {f.get('before')} before {f.get('after')}",
                "forbidden_edge": f"forbidden edge {f.get('i')}-{f.get('j')}",
                "mandatory_edge": f"mandatory edge {f.get('i')}-{f.get('j')}",
                "time_window": f"time window of customer {f.get('customer')} [{f.get('t1', 0):.2f}, "
                               f"{f.get('t2', 0):.2f}]"}[r["type"]]
        out.append(f"- {what}: {'VIOLATED' if k in bad else 'satisfied'}")
    return "Code checker report:\n" + "\n".join(out)


SYSTEM = """You check whether a delivery route obeys the route rules of an instance.
Rule types:
- precedence: one customer must be visited at some point earlier in the route than another.
- forbidden_edge: the route may not go directly between two customers (either direction).
- mandatory_edge: two customers must be visited consecutively (either order).
- time_window: service at a customer must start between t1 and t2 (the times listed in the route are service starts).
- soft_zone: a preference, never a violation; ignore it.
Some sentences in the text are background information and contain no rule. Places may be given names: use the
names dictionary to find the customer numbers.
Answer only with JSON:
{"violations": [ {"type": "<rule type>", "customers": [<customer numbers of the violated rule>],
                  "reason": "<short reason>"} ]}
Use an empty list if the route obeys every rule."""

SYSTEM_REPORT = SYSTEM + """
You also receive the report of an exact code checker. Confirm or correct it, and explain each violation briefly."""


def user_prompt(text, route, report=None):
    s = f"Rules text:\n{text}\n\nRoute ({route.count(chr(10)) + 1} lines):\n{route}\n"
    if report:
        s += f"\n{report}\n"
    return s + "\nWhich rules does the route violate?"


def _customers(rule):
    f = rule["formal"]
    return {v for k, v in f.items() if k in ("before", "after", "i", "j", "customer")}


def match_claims(obj, rules):
    """(set of label-rule indices the LLM says are violated, number of claims that match no hard rule)."""
    claims = obj.get("violations") if isinstance(obj, dict) else None
    if not isinstance(claims, list):
        return None, 0
    found, unmatched = set(), 0
    for c in claims:
        if not isinstance(c, dict):
            unmatched += 1
            continue
        try:
            cs = {int(x) for x in c.get("customers", [])}
        except (TypeError, ValueError):
            cs = set()
        kind = c.get("type")
        hit = [k for k, r in enumerate(rules) if r["type"] != "soft_zone" and r["type"] == kind and
               _customers(r) == cs]
        if not hit:                                          # right customers, wrong type name
            hit = [k for k, r in enumerate(rules) if r["type"] != "soft_zone" and cs and _customers(r) == cs]
        if hit:
            found.add(hit[0])
        else:
            unmatched += 1
    return found, unmatched


def parse(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
