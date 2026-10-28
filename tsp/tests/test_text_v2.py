"""Tests for the regex parser (arm b0, tsp_gnn.formalize) and the text_v2 builder (tsp_gnn.textv2). No LLM needed.
Run with: pytest tests/"""

import numpy as np
import pytest

from tsp_gnn import formalize as F
from tsp_gnn import rules as R
from tsp_gnn import textv2 as T

EXAMPLES = {"precedence": dict(before=12, after=7), "forbidden_edge": dict(i=12, j=7),
            "mandatory_edge": dict(i=12, j=7), "time_window": dict(customer=12, t1=1.5, t2=2.25),
            "soft_zone": dict(zone="central", rect=list(R.ZONES["central"]), weight=1.0)}


def test_parser_reads_every_formulation():
    for kind, formal in EXAMPLES.items():
        for v in range(len(R.TEMPLATES[kind])):
            rule = R.make_rule(kind, v, **formal)
            got = F.parse_regex(R.instance_text([rule]))           # inside the preamble + bullet format
            assert [F.rule_key(g) for g in got] == [F.rule_key(rule)], rule["text"]


def test_parser_several_rules_and_lower_case():
    rules = [R.make_rule("precedence", 0, before=3, after=9), R.make_rule("time_window", 3, customer=5, t1=0.5,
                                                                            t2=1.0)]
    text = "Note: " + rules[0]["text"].lower() + " Also, " + rules[1]["text"]
    assert {F.rule_key(g) for g in F.parse_regex(text)} == {F.rule_key(r) for r in rules}
    assert F.parse_regex("Customer 3 prefers paper invoices.") == []


def test_rule_key():
    a = {"type": "forbidden_edge", "formal": {"i": 3, "j": 9}}
    b = {"type": "forbidden_edge", "formal": {"i": 9, "j": 3}}
    assert F.rule_key(a) == F.rule_key(b)                                  # edges are unordered
    p = {"type": "precedence", "formal": {"before": 3, "after": 9}}
    q = {"type": "precedence", "formal": {"before": 9, "after": 3}}
    assert F.rule_key(p) != F.rule_key(q)                                  # precedence is ordered
    w1 = {"type": "time_window", "formal": {"customer": 4, "t1": 1.5, "t2": 2.25}}
    w2 = {"type": "time_window", "formal": {"customer": 4, "t1": 1.500001, "t2": 2.25}}
    assert F.rule_key(w1) == F.rule_key(w2)


def test_filter_rejects_generator_texts():
    for kind, formal in EXAMPLES.items():
        for v in range(len(R.TEMPLATES[kind])):
            rule = R.make_rule(kind, v, **formal)
            assert T.check_paraphrase(rule["text"], [rule], {}) is not None


def test_filter_cases():
    prec = R.make_rule("precedence", 0, before=12, after=7)
    assert T.check_paraphrase("Make sure customer 12 is on your route sometime earlier than customer 7.",
                              [prec], {}) is None
    assert T.check_paraphrase("Make sure customer 12 is on your route sometime earlier than customer 8.",
                              [prec], {}) == "numbers differ"
    assert T.check_paraphrase("Customer 12 first. Then customer 7 later on.", [prec], {}) == \
        "more than one sentence"
    names = {12: "the hospital"}
    assert T.check_paraphrase("Make sure the hospital is on your route sometime earlier than customer 7.",
                              [prec], names) is None
    assert T.check_paraphrase("Make sure customer 12 is on your route sometime earlier than customer 7.",
                              [prec], names) == "numbers differ"                 # number instead of the name
    zone = R.make_rule("soft_zone", 0, **EXAMPLES["soft_zone"])
    assert T.check_paraphrase("Please limit how much you drive through the central area.", [zone], {}) is None
    assert T.check_paraphrase("Please limit driving in the central and north areas.", [zone], {}) == \
        "other zone mentioned"
    assert T.check_paraphrase("Please drive as little as possible, at least in the central area.",
                              [zone], {}) is None                                # "least" is not "east"
    tw = R.make_rule("time_window", 0, customer=12, t1=1.5, t2=2.25)
    assert T.check_paraphrase("Customer 12 only accepts deliveries starting between 1.50 and 2.25.",
                              [tw], {}) is None


def test_names_and_distractors():
    rng = np.random.default_rng(0)
    rules = [R.make_rule("precedence", 0, before=12, after=7), R.make_rule("time_window", 0, customer=30,
                                                                             t1=1.0, t2=2.0)]
    names = T.assign_names(rng, 200, rules, [0])
    assert len(names) == 20 and 12 in names and 7 in names and 0 not in names
    assert len(set(names.values())) == len(names)
    assert all(not any(ch.isdigit() for ch in v) for v in names.values())
    used = set()
    for _ in range(2):
        s = T.distractor(rng, 200, names, used)
        assert F.parse_regex(s) == []
    assert len(used) == 2


def test_extra_window_is_not_binding():
    rng = np.random.default_rng(1)
    n = 50
    coords = rng.random((n, 2))
    tour = np.concatenate([[0], rng.permutation(np.arange(1, n))])
    rules = [R.make_rule("forbidden_edge", 0, i=int(tour[5]), j=int(tour[9]))]
    length = R.route_length(coords, tour)
    for _ in range(20):
        w = T.extra_window(rng, coords, tour, rules, length)
        assert w["binding"] is False and w["formal"]["customer"] not in (tour[5], tour[9])
        assert R.check_all(rules + [w], coords, tour)                    # the reference still meets it
        assert w["formal"]["t2"] - w["formal"]["t1"] >= 0.1 * length - 1e-9


def test_paraphrase_prompt_lists_generator_phrasings():
    rule = R.make_rule("mandatory_edge", 0, i=12, j=7)
    system, user = T.paraphrase_prompt([rule], {7: "the hospital"})
    assert "mandatory_edge(i=customer 12, j=the hospital)" in user
    for t in R.TEMPLATES["mandatory_edge"]:
        assert t in system
    assert R.TEMPLATES["precedence"][0] not in system
    assert T.parse_sentences('{"sentences": ["a", 3, "b"]}') == ["a", "b"]
    assert T.parse_sentences("not json") == []


def test_meaning_filters_text_v2b():
    prec = R.make_rule("precedence", 0, before=50, after=154)
    ok = ["Visit customer 50 at some point before customer 154.",
          "Customer 154 should only be visited after you've stopped at customer 50.",
          "Before customer 154, make sure to stop at customer 50.",
          "Visit customer 50 first, then customer 154.",
          "Don't visit customer 154 until customer 50 has been served."]
    bad = ["Visit customer 154 at some point before customer 50.",                       # inverted
           "Customer 50 should only be visited after customer 154.",                       # inverted
           "Customer 154 cannot be the first stop if customer 50 has not been visited yet.",
           "Customers 50 and 154 are both on the route today."]                            # no order word
    for t in ok:
        assert T.check_meaning(t, [prec], {}) is None, t
    for t in bad:
        assert T.check_meaning(t, [prec], {}) is not None, t
    # names, and times that contain the customer's number as decimals (1.50) are not mentions of customer 50
    assert T.check_meaning("Visit the hospital before customer 154, between 1.50 and 2.50 if you can.",
                           [prec], {50: "the hospital"}) is None
    mand = R.make_rule("mandatory_edge", 0, i=3, j=9)
    assert T.check_meaning("You should visit customer 3 and then customer 9 in either order.", [mand], {}) \
        == "mandatory_edge: no adjacency word"
    assert T.check_meaning("Visit customer 3 right before or right after customer 9.", [mand], {}) is None
    forb = R.make_rule("forbidden_edge", 0, i=3, j=9)
    assert T.check_meaning("You should not drive from customer 3 to customer 9, or from customer 9 to customer 3.",
                           [forb], {}) == "forbidden_edge: no adjacency word"
    for t in ("Customers 3 and 9 must not be visited back-to-back.", "Never go from 3 to 9 in a single step.",
              "Do not travel directly between customer 3 and customer 9."):
        assert T.check_meaning(t, [forb], {}) is None, t
    assert T.check_paraphrase("You should visit customer 3 and then customer 9 in either order.", [mand], {}) is None
    assert T.check_paraphrase("You should visit customer 3 and then customer 9 in either order.", [mand], {},
                              strict=True) is not None
