"""Tests of the hybrid trigger and the union arm of phase 5, step 3."""

import numpy as np
import pytest

from tsp_gnn import open_dsl as D
from tsp_gnn import open_hybrid as H


@pytest.fixture
def inst():
    coords = np.zeros((10, 2))
    coords[:, 0] = np.arange(10) / 10
    legend = [{"id": c, "name": name, "kind": "shop", "area": "Alfama", "x": 0.0, "y": 0.0}
              for c, name in ((1, "Rua Azul 3"), (2, "Rua Verde 1"), (3, "Rua Nova 2"), (4, "Rua Alta 9"))]

    def make(text):
        return {"text": "Route rules. Fixed preamble.\n" + text, "level": "L2", "legend": legend,
                "ctx": D.Context(legend, coords, 10)}
    return make


def test_covered_text_stays_with_the_parser(inst):
    cov = H.coverage(inst("You must visit Rua Azul 3 before Rua Verde 1."), 10)
    assert cov["constraint"] == 1 and cov["rules"] == 1 and not cov["send"]


def test_uncovered_constraint_sentence_is_sent(inst):
    cov = H.coverage(inst("You must visit Rua Azul 3 before Rua Verde 1. Never go straight to Rua Nova 2."), 10)
    assert cov["uncovered"] == 1 and cov["send"]


def test_past_remark_is_not_a_constraint(inst):
    assert not H.coverage(inst("Rua Azul 3 was visited before Rua Verde 1 last week."), 10)["send"]


def test_deleted_stock_phrase_counts(inst):
    text = "You must visit Rua Azul 3 before Rua Verde 1. It is not acceptable to skip Rua Nova 2 until Rua Alta 9 is served."
    cov = H.coverage(inst(text), 10)
    assert cov["rules"] == 1 and cov["constraint"] == 2 and cov["send"]


def test_union_keeps_the_rule_of_p1_on_a_contradiction():
    coords = np.random.default_rng(0).random((6, 2))
    forb = {"prims": [{"type": "forbidden_edge", "formal": {"i": 1, "j": 2}}]}
    mand = {"prims": [{"type": "mandatory_edge", "formal": {"i": 1, "j": 2}}]}
    prims, rejected = H.union([forb], [mand], coords)
    assert [p["type"] for p in prims] == ["forbidden_edge"] and rejected == [{"rule": 1, "reason": "an edge both mandatory and forbidden"}]
