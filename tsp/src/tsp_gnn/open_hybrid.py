"""Hybrid formalisation of phase 5, step 3 (offline, from the logged answers): when to trust the p1 parser.

The trigger uses only what p1 has at run time, the text and its own output, never the label:
    a sentence is a constraint sentence if p1 would read it (not a remark about the past) and it has an order, adjacency,
    time or zone word next to a rule word;
    the stock phrase "it is not acceptable to skip ..." that p1 deletes also counts, if it has one of those words: it can
    hold a rule that p1 never reads;
    the text goes to the LLM if a constraint sentence gives no valid rule, or if p1 has fewer valid rules than there are
    constraint sentences.
h1 takes the answer of q14 for the texts sent and the one of p1 for the others. h2 joins both answers, p1 first, and lets
the code check reject what cannot hold together, so on a contradiction the rule of p1 stays.
"""

import re

from tsp_gnn import keyword_parser as K
from tsp_gnn import open_dsl as D
from tsp_gnn import open_parser as P

MUST = re.compile(r"\b(must|may not|never|cannot|has to|have to|zones?)\b", re.I)


def constraint_words(clause):
    if P.PAST.search(clause) and not MUST.search(clause):
        return False
    times = len(P.TIME.findall(clause)) >= 2
    zone = bool(K.ZONE_WORD.search(clause) and re.search(r"\bzones?\b", clause, re.I))
    return times or zone or any(rx.search(clause) for rx in (P.ADJ, P.BEFORE, P.AFTER))


def is_constraint(parser, sentence):
    if not P.RULE_WORD.search(sentence):
        return False
    has_ref = lambda c: bool(parser.mentions(c)) or bool(K.ZONE_WORD.search(c))
    return any(constraint_words(c) for c in P.split_clauses(sentence, has_ref))


def sentence_rules_l0(sentence, names, n):
    """The keyword parser of p1 on L0 reads whole texts: one bullet at a time, with the line of names."""
    return P.parse_l0("\n".join([names, f"- {sentence}"]), n)["rules"]


def coverage(inst, n):
    """{"constraint": constraint sentences, "uncovered": those with no valid rule, "rules": valid rules of p1,
    "send": True if the text goes to the LLM}."""
    text, level = inst["text"], inst["level"]
    parser = P.Parser(inst["legend"], n)
    names = next((x for x in text.splitlines() if x.startswith("Named places:")), "")
    needs = sum(constraint_words(m.group()) for m in P.FILLER.finditer(text))
    uncovered = 0
    for s in P.split_sentences(text):
        found = sentence_rules_l0(s, names, n) if level == "L0" else parser.sentence_rules(s)
        if is_constraint(parser, s):
            needs += 1
            uncovered += not D.validate({"rules": found}, inst["ctx"])[0]
    obj = P.parse_l0(text, n) if level == "L0" else P.parse(text, inst["legend"], n)
    rules = len(D.validate(obj, inst["ctx"])[0])
    return {"constraint": needs, "uncovered": uncovered, "rules": rules, "send": uncovered > 0 or rules < needs}


def union(p1_rules, q14_rules, coords):
    """Arm h2: (primitives, rejected) of both answers after the code check, which keeps the rule seen first."""
    return D.validate_for_solver(list(p1_rules) + list(q14_rules), coords)
