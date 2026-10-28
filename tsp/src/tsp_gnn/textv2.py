"""Second wording of the text rules (phase 5, step 2, part A): text_v2, a linguistic test set that does NOT use the
generator's 4 formulations per type.

Built from each step-1 instance (its formal rules are the label) by:
    (i)   paraphrases written by the local qwen2.5:7b from the formal translation (temperature 0.8, 2 per rule),
          kept only if they pass check_paraphrase: no generator formulation inside (formalize.matches_template), not
          too close to any of them (difflib ratio <= MAX_SIMILARITY), one sentence, exactly the rule's numbers,
          the place name instead of the number for named customers, the zone name and no other zone;
    (ii)  names: a dictionary for 10 % of the customers ("customer 17 is the hospital"); the customers of about
          half the rules are in it and those rules refer to them by name only;
    (iii) 1-2 irrelevant sentences per instance (DISTRACTORS, context without a rule);
    (iv)  in 20 % of the instances with 2+ rules, two rules are written as one sentence (also by the LLM).
In 20 % of the instances without a precedence, an extra time window that the reference route already meets is
added (binding = False) to measure false positives of the verifiers; it is part of the label and of text_v1.
The same 7B later translates text_v2 back (arms b1 / b2), which may favour it: see the report.
"""

import difflib
import json
import math
import re

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import rules as R

MAX_SIMILARITY = 0.8
NAME_FRACTION = 0.1

OWNERS = ["Maria", "Ahmed", "Chen", "Olga", "Kofi", "Lucia", "Ravi", "Hana", "Tomas", "Amara", "Ivan", "Yuki",
          "Pedro", "Fatima", "Jonas", "Leila", "Marco", "Nadia", "Oscar", "Priya", "Sven", "Talia", "Emeka", "Rosa",
          "Diego"]
SHOPS = ["bakery", "pharmacy", "flower shop", "hardware store", "bookshop", "butcher", "tailor", "garage",
         "dental clinic", "print shop", "barber", "deli", "pet shop", "laundry", "music store", "toy shop", "café",
         "grocery", "repair shop", "wine bar"]
PLACES = ["the hospital", "the city library", "the town hall", "the fire station", "the primary school",
          "the train station", "the post office", "the museum", "the football stadium", "the community centre",
          "the swimming pool", "the old church", "the courthouse", "the university campus", "the police station",
          "the bus depot", "the shopping mall", "the cinema", "the youth hostel", "the farmers' market",
          "the city zoo", "the concert hall", "the nursing home", "the sports club", "the marina",
          "the botanical garden", "the art gallery", "the water plant", "the tennis club", "the science park"]

DISTRACTORS = [
    "Customer {c} prefers paper invoices.",
    "The driver of today's shift is new to the area.",
    "Customer {c} asked to be called on arrival.",
    "Please keep the delivery notes signed.",
    "Customer {c} has a large order this week.",
    "The van was serviced yesterday.",
    "Customer {c} recently changed its phone number.",
    "Remember to scan every parcel at the depot.",
    "Customer {c} is a long-standing client.",
    "Fuel prices went up this month.",
    "Customer {c} pays at the end of the month.",
    "The radio in the van is broken.",
    "Customer {c} would like a copy of the receipt.",
    "Traffic reports say the weather will be sunny.",
    "Customer {c} ordered {k} boxes.",
    "Customers {c} and {d} belong to the same company.",
    "There are {k} parcels in the back of the van.",
    "Customer {c} gave the driver a good review last time.",
    "The depot closes its office at the usual hour.",
    "Customer {c} has a dog, so ring the bell.",
    "Customer {c} and customer {d} were both visited last Tuesday.",
    "The loading ramp at the depot was repainted.",
    "Customer {c} asked about next month's prices.",
    "Please return empty crates to the depot.",
    "Customer {c} is closed on Sundays, but today is a weekday.",
    "{name} has a new manager.",
    "{name} ordered extra packaging material.",
    "{name} usually signs for parcels at the front desk.",
    "Our records show {name} has been a client for {k} years.",
    "{name} and customer {c} are on the same street.",
]

TYPE_MEANING = {
    "precedence": '"before" must be visited at some point earlier in the route than "after" (not necessarily '
                  'immediately before).',
    "forbidden_edge": 'the vehicle may not drive directly between "i" and "j", in either direction.',
    "mandatory_edge": '"i" and "j" must be visited one right after the other, in either order.',
    "time_window": 'service at "customer" must start no earlier than t1 and no later than t2 (time units of the '
                   'route; arriving early means waiting).',
    "soft_zone": 'driving inside the named zone is penalised but allowed: the route should avoid it if it can.',
}

SYSTEM_PARAPHRASE = (
    "You rewrite delivery-route constraints as natural English, as a dispatcher would say them to a driver. "
    "You receive constraints in a formal notation:\n"
    + "\n".join(f"- {k}: {v}" for k, v in TYPE_MEANING.items())
    + "\nRules for your sentences:\n"
    "- Keep the exact meaning, including which customer comes first. forbidden_edge and mandatory_edge are "
    "symmetric: say explicitly that they hold in both directions / in either order (do not impose an order). "
    "precedence does not mean 'immediately before'. soft_zone is a preference, not a ban.\n"
    "- Keep every number exactly as given (times with two decimals). Write customers as 'customer N' or a "
    "similar word (stop, client, address), unless a name is given: then use only the name, never its number.\n"
    "- Do not mention constraint type names (precedence, edge, window ...) and do not add new information.\n"
    "- Each sentence is a single sentence.\n"
    "- Vary the wording; do NOT use these phrasings or close variants of them:\n{avoid}\n"
    'Answer with JSON: {{"sentences": ["...", "..."]}}')


# ---------------------------------------------------------------------------------------------------- names

def name_pool(rng):
    """All distinct place names in a random order (30 places + 500 'owner's shop' names)."""
    names = PLACES + [f"{o}'s {s}" for o in OWNERS for s in SHOPS]
    return [names[i] for i in rng.permutation(len(names))]


def rule_customers(rule):
    f = rule["formal"]
    return {"precedence": [f.get("before"), f.get("after")], "forbidden_edge": [f.get("i"), f.get("j")],
            "mandatory_edge": [f.get("i"), f.get("j")], "time_window": [f.get("customer")],
            "soft_zone": []}[rule["type"]]


def assign_names(rng, n, rules, name_rules):
    """Dictionary {customer: name} for round(NAME_FRACTION * n) customers, holding every customer of the rules in
    name_rules (indices into rules); the rest are random customers."""
    want = max(1, round(NAME_FRACTION * n))
    chosen = []
    for r in name_rules:
        chosen += [c for c in rule_customers(rules[r]) if c not in chosen]
    others = [int(c) for c in rng.permutation(np.arange(1, n)) if int(c) not in chosen]
    chosen += others[:max(0, want - len(chosen))]
    pool = name_pool(rng)
    return {int(c): pool[k] for k, c in enumerate(chosen)}


def names_text(names, rng):
    order = rng.permutation(sorted(names))
    return "Named places: " + "; ".join(f"customer {int(c)} is {names[int(c)]}" for c in order) + "."


# ---------------------------------------------------------------------------------------------------- extra window

def extra_window(rng, coords, tour, rules, pure_length):
    """A time window for a customer not used by any rule, containing the time service starts there on the reference
    route (so the reference still meets every rule): width 10 % of the pure length, the arrival at a random point
    inside it. Marked binding = False."""
    n = len(coords)
    used = {c for r in rules for c in rule_customers(r)}
    pos = np.empty(n, dtype=np.int64)
    pos[tour] = np.arange(n)
    times = R.service_times(coords, tour, rules)
    w = 0.1 * pure_length
    for _ in range(1000):
        k = int(rng.integers(1, n))
        if k in used:
            continue
        arr = float(times[pos[k]])
        t1 = max(0.0, math.floor((arr - rng.uniform(0.2, 0.8) * w) * 100) / 100)
        t2 = math.ceil(max(t1 + w, arr) * 100) / 100
        rule = R.make_rule("time_window", int(rng.integers(len(R.TEMPLATES["time_window"]))),
                           customer=k, t1=t1, t2=t2)
        rule["binding"] = False
        if R.check_all(rules + [rule], coords, tour):
            return rule
    raise RuntimeError("no extra window found")


# ---------------------------------------------------------------------------------------------------- paraphrases

def _ref(c, names):
    return names[c] if c in names else f"customer {c}"


def formal_line(rule, names):
    """The formal translation shown to the paraphrasing LLM (names replace numbers for named customers)."""
    f, kind = rule["formal"], rule["type"]
    if kind == "precedence":
        args = {"before": _ref(f["before"], names), "after": _ref(f["after"], names)}
    elif kind in ("forbidden_edge", "mandatory_edge"):
        args = {"i": _ref(f["i"], names), "j": _ref(f["j"], names)}
    elif kind == "time_window":
        args = {"customer": _ref(f["customer"], names), "t1": f"{f['t1']:.2f}", "t2": f"{f['t2']:.2f}"}
    else:
        args = {"zone": f["zone"]}
    return f"{kind}({', '.join(f'{k}={v}' for k, v in args.items())})"


def avoid_list(kinds):
    return "\n".join(f"  * {t}" for k in dict.fromkeys(kinds) for t in R.TEMPLATES[k])


def paraphrase_prompt(rules, names):
    """(system, user) asking for 2 sentences that express all the given rules (1 rule, or 2 for a merged sentence)."""
    system = SYSTEM_PARAPHRASE.format(avoid=avoid_list([r["type"] for r in rules]))
    if len(rules) == 1:
        user = f"Constraint: {formal_line(rules[0], names)}\nWrite 2 different sentences, each expressing it."
    else:
        user = ("Constraints:\n" + "\n".join(f"- {formal_line(r, names)}" for r in rules)
                + "\nWrite 2 different sentences, each expressing BOTH constraints in a single sentence.")
    return system, user


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def expected_numbers(rules, names):
    out = []
    for r in rules:
        f = r["formal"]
        out += [float(c) for c in rule_customers(r) if c not in names]
        if r["type"] == "time_window":
            out += [round(f["t1"], 2), round(f["t2"], 2)]
    return sorted(out)


def check_paraphrase(text, rules, names):
    """None if text is an acceptable wording of the rules, else the reason it is rejected."""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "empty"
    t = text.strip()
    if F.matches_template(t):
        return "generator formulation"
    if re.search(r"[.!?;]\s+\S", t.rstrip(".!? ")):
        return "more than one sentence"
    names_used = {c: names[c] for r in rules for c in rule_customers(r) if c in names}
    for r in rules:
        for v in range(len(R.TEMPLATES[r["type"]])):
            rendered = R.rule_text(r, v)
            for c, nm in names_used.items():
                rendered = re.sub(rf"customer {c}\b", nm, rendered)
            if difflib.SequenceMatcher(None, t.lower(), rendered.lower()).ratio() > MAX_SIMILARITY:
                return "too close to a generator formulation"
    got = {round(float(x), 2) for x in NUMBER.findall(t)}           # a number may be repeated ("from 3 to 7 or
    if got != set(expected_numbers(rules, names)):                  # from 7 to 3")
        return "numbers differ"
    low = t.lower()
    for nm in names_used.values():
        if nm.lower() not in low:
            return "name missing"
    zones = [r["formal"]["zone"] for r in rules if r["type"] == "soft_zone"]
    for z in R.ZONES:                                   # word start: "northern" counts, "at least" is not "east"
        if bool(re.search(rf"\b{z}", low)) != (z in zones):
            return "zone missing" if z in zones else "other zone mentioned"
    return None


def parse_sentences(content):
    obj = None
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    s = obj.get("sentences") if isinstance(obj, dict) else None
    return [x for x in s if isinstance(x, str)] if isinstance(s, list) else []


# ---------------------------------------------------------------------------------------------------- assembly

def distractor(rng, n, names, used):
    """One irrelevant sentence (customers / names drawn at random, never the depot)."""
    for _ in range(100):
        t = DISTRACTORS[int(rng.integers(len(DISTRACTORS)))]
        if "{name}" in t and not names:
            continue
        if t in used:
            continue
        used.add(t)
        c, d = (int(x) for x in rng.choice(np.arange(1, n), 2, replace=False))
        nm = names[int(rng.choice(sorted(names)))] if names else ""
        s = t.format(c=c, d=d, k=int(rng.integers(2, 15)), name=nm)
        return s[0].upper() + s[1:]
    raise RuntimeError("no distractor")


def compose_v2(names_line, items, rng):
    """text_v2: preamble, the names dictionary, then the rule sentences and distractors in a random order."""
    order = rng.permutation(len(items))
    return R.PREAMBLE + "\n" + names_line + "\n" + "\n".join(f"- {items[i]}" for i in order)
