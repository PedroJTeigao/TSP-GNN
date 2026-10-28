"""Arm b0+: a keyword parser built from the meaning filters of part A (tsp_gnn.textv2), no LLM.

text_v2b was built so that the regex arm b0 fails on it (no generator formulation is allowed in a paraphrase), so
b0's 0 % on text_v2b is by design. b0+ is the fair non-LLM baseline: it knows the vocabulary of the filters, not the
sentences. Steps, on every bullet line of the text (the preamble is skipped):
    1. names: the "Named places: customer 17 is the hospital; ..." line gives the name -> number dictionary;
    2. the line is split into clauses at ", and" (two rules in one sentence are usually joined like that);
    3. in each clause, customers are found by name (longest names first) or by number (integers that are not part
       of a decimal); decimals are times;
    4. soft zones: every zone word in a clause that says "zone(s)";
    5. time windows: each pair of decimals goes to the nearest customer mentioned before it (t1 = smaller);
    6. the customers left are paired in order of first mention; for each pair, the words around it give the type:
       an adjacency word (textv2.ADJACENT, plus "road") -> forbidden edge if there is a negation word, else
       mandatory edge; otherwise an order word (textv2.BEFORE_WORDS / AFTER_WORDS) -> precedence, with the
       direction given by textv2.precedence_direction (the position of the customers around the order word);
       otherwise no rule (e.g. "Customers 3 and 9 belong to the same company").
"""

import re

from tsp_gnn import rules as R
from tsp_gnn import textv2 as T

ADJ = re.compile(T.ADJACENT.pattern + r"|\broad\b", re.I)
NEG = re.compile(r"\b(?:not|never|cannot|avoid|refrain|no|closed|unavailable|forbidden|prohibited)\b|n't", re.I)
ZONE_WORD = re.compile(r"\b(" + "|".join(R.ZONES) + r")\b", re.I)
DECIMAL = re.compile(r"\d+\.\d+")
INTEGER = re.compile(r"(?<![\d.])\d+(?![\d]|\.\d)")
NAMES_LINE = re.compile(r"customer (\d+) is ([^;]+?)\s*(?:;|\.?$)")


def read_names(text):
    for line in text.splitlines():
        if line.startswith("Named places:"):
            return {int(c): nm.strip() for c, nm in NAMES_LINE.findall(line[len("Named places:"):].strip())}
    return {}


def mentions(clause, names):
    """[(start, end, customer)] in order: names first (longest first, no overlaps), then integers outside them."""
    taken, out = [], []
    for c, nm in sorted(names.items(), key=lambda x: -len(x[1])):
        for m in re.finditer(re.escape(nm), clause, re.I):
            if not any(m.start() < e and s < m.end() for s, e in taken):
                taken.append((m.start(), m.end()))
                out.append((m.start(), m.end(), c))
    for m in INTEGER.finditer(clause):
        if not any(m.start() < e and s < m.end() for s, e in taken):
            out.append((m.start(), m.end(), int(m.group())))
    return sorted(out)


def parse_clause(clause, names, n):
    rules = []
    if re.search(r"\bzones?\b", clause, re.I):
        for z in dict.fromkeys(m.group(1).lower() for m in ZONE_WORD.finditer(clause)):
            rules.append({"type": "soft_zone", "formal": {"zone": z, "rect": list(R.ZONES[z]),
                                                          "weight": R.SOFT_WEIGHT}})
    ms = [m for m in mentions(clause, names) if 1 <= m[2] < n]
    used = set()
    decs = list(DECIMAL.finditer(clause))
    for a, b in zip(decs[0::2], decs[1::2]):
        before = [m for m in ms if m[1] <= a.start()] or ms
        if not before:
            continue
        c = min(before, key=lambda m: abs(a.start() - m[1]))[2]
        t1, t2 = sorted((float(a.group()), float(b.group())))
        rules.append({"type": "time_window", "formal": {"customer": c, "t1": t1, "t2": t2}})
        used.add(c)
    order = []
    for s, e, c in ms:
        if c not in used and c not in [x[2] for x in order]:
            order.append((s, e, c))
    pairs = [order[k:k + 2] for k in range(0, len(order) - 1, 2)]
    for k, (x, y) in enumerate(pairs):
        lo = pairs[k - 1][1][1] if k else 0
        hi = pairs[k + 1][0][0] if k + 1 < len(pairs) else len(clause)
        span = clause[lo:hi]
        a, b = x[2], y[2]
        if ADJ.search(span):
            kind = "forbidden_edge" if NEG.search(span) else "mandatory_edge"
            rules.append({"type": kind, "formal": {"i": a, "j": b}})
        elif T.BEFORE_WORDS.search(span) or T.AFTER_WORDS.search(span):
            if T.precedence_direction(span, a, b, names) is None:
                rules.append({"type": "precedence", "formal": {"before": a, "after": b}})
            elif T.precedence_direction(span, b, a, names) is None:
                rules.append({"type": "precedence", "formal": {"before": b, "after": a}})
            else:                                          # undecided: text order
                rules.append({"type": "precedence", "formal": {"before": a, "after": b}})
    return rules


def parse_keywords(text, n):
    """Arm b0+: formal rules read from the bullet lines of an instance text."""
    from tsp_gnn.formalize import rule_key
    names = read_names(text)
    out, keys = [], set()
    for line in text.splitlines():
        if not line.startswith("- "):
            continue
        for clause in re.split(r",\s+and\s+", line[2:]):
            for r in parse_clause(clause, names, n):
                if rule_key(r) not in keys:
                    keys.add(rule_key(r))
                    out.append(r)
    return out
