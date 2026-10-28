"""Text rules -> formal rules (phase 5, step 2).

Arm b0 is parse_regex: one regular expression per generator formulation (rules.TEMPLATES, 4 per type), searched
anywhere in the text. It is what one would write without an LLM, knowing the generator's phrasings. The same parser
is the filter that keeps paraphrases away from the generator's phrasings (tsp_gnn.textv2).

A formal rule is {"type": ..., "formal": {...}} with the fields of tsp_gnn.rules; rule_key gives the canonical
form used to compare a prediction with the label (edges are unordered, times rounded to 2 decimals).
"""

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
