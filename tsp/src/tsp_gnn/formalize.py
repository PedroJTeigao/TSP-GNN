"""Text rules -> formal rules (phase 5, step 2).

Arm b0 is parse_regex: one regular expression per generator formulation (rules.TEMPLATES, 4 per type), searched
anywhere in the text. It is what one would write without an LLM, knowing the generator's phrasings. The same parser
is the filter that keeps paraphrases away from the generator's phrasings (tsp_gnn.textv2).

A formal rule is {"type": ..., "formal": {...}} with the fields of tsp_gnn.rules; rule_key gives the canonical
form used to compare a prediction with the label (edges are unordered, times rounded to 2 decimals).
"""

import json
import math
import re

from tsp_gnn import rules as R

_FIELD = {"before": r"\d+", "after": r"\d+", "i": r"\d+", "j": r"\d+", "customer": r"\d+",
          "t1": r"\d+(?:\.\d+)?", "t2": r"\d+(?:\.\d+)?", "zone": "|".join(R.ZONES)}


def _template_regex(template):
    """'Visit customer {before} before customer {after}.' -> regex with named groups (back-references when a
    field appears twice, as in 'from {i} to {j}, or from {j} to {i}')."""
    out, seen, last = [], set(), 0
    for m in re.finditer(r"\{(\w+)(?::[^}]*)?\}", template):
        out.append(re.escape(template[last:m.start()]))
        name = m.group(1)
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>{_FIELD[name]})")
        seen.add(name)
        last = m.end()
    out.append(re.escape(template[last:].rstrip(".")))      # the final full stop is optional
    return re.compile("".join(out), re.IGNORECASE)


PATTERNS = [(kind, v, _template_regex(t)) for kind, ts in R.TEMPLATES.items() for v, t in enumerate(ts)]


def _formal(kind, g):
    if kind == "precedence":
        return {"before": int(g["before"]), "after": int(g["after"])}
    if kind in ("forbidden_edge", "mandatory_edge"):
        return {"i": int(g["i"]), "j": int(g["j"])}
    if kind == "time_window":
        return {"customer": int(g["customer"]), "t1": float(g["t1"]), "t2": float(g["t2"])}
    zone = g["zone"].lower()
    return {"zone": zone, "rect": list(R.ZONES[zone]), "weight": R.SOFT_WEIGHT}


def parse_regex(text):
    """Arm b0: every generator formulation found in text, in order of appearance (no duplicates)."""
    found = []
    for kind, _, pat in PATTERNS:
        for m in pat.finditer(text):
            found.append((m.start(), {"type": kind, "formal": _formal(kind, m.groupdict())}))
    found.sort(key=lambda x: x[0])
    out, keys = [], set()
    for _, r in found:
        if rule_key(r) not in keys:
            keys.add(rule_key(r))
            out.append(r)
    return out


def matches_template(text):
    """True if any generator formulation occurs in text."""
    return any(pat.search(text) for _, _, pat in PATTERNS)


def rule_key(rule):
    """Canonical, hashable form of a formal rule."""
    f, kind = rule["formal"], rule["type"]
    if kind == "precedence":
        return kind, int(f["before"]), int(f["after"])
    if kind in ("forbidden_edge", "mandatory_edge"):
        return kind, frozenset((int(f["i"]), int(f["j"])))
    if kind == "time_window":
        return kind, int(f["customer"]), round(float(f["t1"]), 2), round(float(f["t2"]), 2)
    if kind == "soft_zone":
        return kind, str(f["zone"]).lower()
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------------- JSON schema

FIELDS = {"precedence": ("before", "after"), "forbidden_edge": ("i", "j"), "mandatory_edge": ("i", "j"),
          "time_window": ("customer", "t1", "t2"), "soft_zone": ("zone",)}

SCHEMA_TEXT = """{"rules": [ one object per rule, e.g.
  {"type": "precedence", "before": <customer number>, "after": <customer number>},
  {"type": "forbidden_edge", "i": <customer number>, "j": <customer number>},
  {"type": "mandatory_edge", "i": <customer number>, "j": <customer number>},
  {"type": "time_window", "customer": <customer number>, "t1": <number>, "t2": <number>},
  {"type": "soft_zone", "zone": "north" | "south" | "east" | "west" | "central"} ]}"""


def _int(x):
    if isinstance(x, bool):
        raise ValueError
    if isinstance(x, (int, float)) and float(x) == int(x):
        return int(x)
    if isinstance(x, str) and re.fullmatch(r"\s*\d+\s*", x):
        return int(x)
    raise ValueError


def validate(obj, n):
    """(rules, errors): the formal rules of a model answer that fit the schema, and one message per item that does
    not. Customers must be 1..n-1, distinct within a rule, t1 <= t2, the zone one of ZONES. An answer that is not
    {"rules": [...]} gives ([], ["no rules list"])."""
    if not isinstance(obj, dict) or not isinstance(obj.get("rules"), list):
        return [], ["no rules list"]
    out, errors = [], []
    for item in obj["rules"]:
        try:
            kind = item["type"]
            if kind not in FIELDS:
                raise ValueError(f"unknown type {kind!r}")
            if kind == "soft_zone":
                zone = str(item["zone"]).lower().strip()
                if zone not in R.ZONES:
                    raise ValueError(f"unknown zone {zone!r}")
                formal = {"zone": zone, "rect": list(R.ZONES[zone]), "weight": R.SOFT_WEIGHT}
            elif kind == "time_window":
                formal = {"customer": _int(item["customer"]), "t1": float(item["t1"]), "t2": float(item["t2"])}
                if not (math.isfinite(formal["t1"]) and math.isfinite(formal["t2"])):
                    raise ValueError("t1 or t2 not finite")    # e.g. "t2": "inf", which float() accepts
                if not formal["t1"] <= formal["t2"]:
                    raise ValueError("t1 > t2")
            else:
                formal = {k: _int(item[k]) for k in FIELDS[kind]}
            cs = [v for k, v in formal.items() if k in ("before", "after", "i", "j", "customer")]
            if any(not 1 <= c < n for c in cs) or len(set(cs)) != len(cs):
                raise ValueError(f"bad customers {cs}")
            rule = {"type": kind, "formal": formal}
            if rule_key(rule) not in {rule_key(r) for r in out}:
                out.append(rule)
        except (KeyError, TypeError, ValueError) as e:
            errors.append(f"{type(e).__name__}: {e}")
    return out, errors


def load_preds(path, n):
    """Records of formalize_eval.py, with the LLM answers validated again from the raw answer by the current
    validate (the schema gained the finite-time check after part B ran); the regex and keyword arms are kept."""
    from tsp_gnn.llm import extract_json
    out = []
    for line in open(path):
        e = json.loads(line)
        if e["arm"] in ("b1", "b2") and e.get("raw") is not None:
            obj = extract_json(e["raw"])
            e["pred"], e["errors"] = validate(obj, n)
            if obj is None:
                e["errors"] = ["invalid JSON"]
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------------------- scoring

def _customers(rule):
    f = rule["formal"]
    return frozenset(v for k, v in f.items() if k in ("before", "after", "i", "j", "customer"))


def score(label, pred):
    """Compare predicted rules with the label. Returns per-rule correctness (aligned with label), the confusion
    pairs (true type, predicted type / 'missing') for every label rule, and the spurious predicted types.
    A label rule is matched to a prediction with the same key (correct), else the same customers or zone (wrong
    parameters or type), else to nothing (missing); predictions left over are spurious."""
    left = list(pred)
    hits = []
    for r in label:                                   # exact matches first, so a near match cannot steal them
        hit = next((p for p in left if rule_key(p) == rule_key(r)), None)
        if hit is not None:
            left.remove(hit)
        hits.append(hit)
    correct = [h is not None for h in hits]
    for k, r in enumerate(label):
        if hits[k] is not None:
            continue
        if r["type"] == "soft_zone":
            hit = next((p for p in left if p["type"] == "soft_zone" and p["formal"]["zone"] == r["formal"]["zone"]),
                       None)
        else:
            hit = next((p for p in left if _customers(p) == _customers(r)), None) or \
                next((p for p in left if _customers(p) & _customers(r)), None)
        if hit is not None:
            left.remove(hit)
        hits[k] = hit
    pairs = [(r["type"], h["type"] if h else "missing") for r, h in zip(label, hits)]
    return {"correct": correct, "pairs": pairs, "spurious": [p["type"] for p in left],
            "exact": all(correct) and not left}


# ---------------------------------------------------------------------------------------------------- LLM arms

CATALOG = """Rule types (customer 0 is the depot; customers are numbered 1..n-1):
- precedence: customer "before" must be visited at some point earlier in the route than customer "after".
- forbidden_edge: the route may not go directly between customers i and j (either direction).
- mandatory_edge: customers i and j must be visited consecutively (either order).
- time_window: service at "customer" must start between times t1 and t2.
- soft_zone: driving inside the zone ("north", "south", "east", "west" or "central") should be avoided if possible."""

SYSTEM_B1 = ("You translate the route rules of a vehicle routing instance, written in English, into a formal JSON "
             "list.\n" + CATALOG + "\n"
             "Instructions:\n"
             "- Output every rule stated in the text, once. A sentence may contain several rules; a sentence may "
             "also contain no rule at all (background information): ignore those.\n"
             "- The fixed description of the depot, speed and zones at the start is not a rule.\n"
             "- If customers are given names (\"customer 17 is the hospital\"), write the customer NUMBER.\n"
             "- Times are numbers, as written in the text.\n"
             "Answer only with JSON of the form:\n" + SCHEMA_TEXT)

# 5 worked examples for arm b2 (hand-written for n = 60; not in the test set)
_EX_PRE = R.PREAMBLE
FEW_SHOT = [
    (_EX_PRE + "\nNamed places: customer 12 is the town hall; customer 31 is Lena's bakery; customer 7 is the "
     "harbour office.\n- The harbour office should get its parcel only once the town hall has been served.\n"
     "- Lena's bakery has a new delivery entrance.\n- Please stay away from the east zone where you can.",
     {"rules": [{"type": "precedence", "before": 12, "after": 7}, {"type": "soft_zone", "zone": "east"}]}),
    (_EX_PRE + "\nNamed places: customer 44 is the swimming hall; customer 5 is Omar's kiosk.\n"
     "- Stops 18 and 52 cannot be done back-to-back, in either order.\n"
     "- Make Omar's kiosk and customer 40 adjacent stops on the route, whichever comes first.\n"
     "- Customer 9 orders every Friday.",
     {"rules": [{"type": "forbidden_edge", "i": 18, "j": 52}, {"type": "mandatory_edge", "i": 5, "j": 40}]}),
    (_EX_PRE + "\nNamed places: customer 23 is the dental practice; customer 3 is the library.\n"
     "- The dental practice accepts the van only from 1.25 until 2.40, and stop 51 must be reached between 0.80 "
     "and 1.95.\n- The library usually pays by card.",
     {"rules": [{"type": "time_window", "customer": 23, "t1": 1.25, "t2": 2.4},
                {"type": "time_window", "customer": 51, "t1": 0.8, "t2": 1.95}]}),
    (_EX_PRE + "\nNamed places: customer 36 is the old mill; customer 14 is Ines's florist.\n"
     "- Drop off at client 27 prior to going to the old mill.\n"
     "- Do not drive straight from Ines's florist to customer 50 or from 50 to the florist.\n"
     "- Before you reach customer 8, make sure customer 33 has been served.\n"
     "- There are 6 boxes for customer 14 today.",
     {"rules": [{"type": "precedence", "before": 27, "after": 36}, {"type": "forbidden_edge", "i": 14, "j": 50},
                {"type": "precedence", "before": 33, "after": 8}]}),
    (_EX_PRE + "\nNamed places: customer 2 is the fire station; customer 58 is Paulo's garage.\n"
     "- The van has a full tank.\n- If it can be helped, limit the distance driven in the central zone, and go "
     "from the fire station directly to Paulo's garage or the other way round.",
     {"rules": [{"type": "soft_zone", "zone": "central"}, {"type": "mandatory_edge", "i": 2, "j": 58}]}),
]


def system_prompt(arm):
    if arm == "b1":
        return SYSTEM_B1
    if arm == "b2":
        ex = "\n\n".join(f"Example {k + 1}.\nText:\n{t}\nAnswer:\n{json.dumps(a)}" for k, (t, a) in enumerate(FEW_SHOT))
        return SYSTEM_B1 + "\n\nWorked examples:\n\n" + ex
    raise ValueError(arm)


def user_prompt(text, n):
    return f"Instance with n = {n} (customers 1..{n - 1}).\nText:\n{text}\nAnswer:"
