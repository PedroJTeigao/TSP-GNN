"""Tests for src/tsp_gnn/triage.py (window triage: nothing / model / LKH).  Run with: pytest tests/"""

import numpy as np
import pytest
import torch

from tsp_gnn.data import generate_instances, is_valid_tour, tour_length
from tsp_gnn.hier import Heatmaps
from tsp_gnn.triage import (beam_with_confidence, choose_thresholds, segment_disagreement, targets,
                            triage_refine, window_log)
from tsp_gnn.window import initial_tour


class ShortEdges(torch.nn.Module):
    """Stand-in for the trained model: shorter edges get higher logits."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        return -20 * data.edge_attr.squeeze(-1) + 2 + self.dummy


@pytest.fixture(scope="module")
def setup():
    coords = generate_instances(1, 300, 5)[0]
    return coords, initial_tour(coords, 50), Heatmaps(ShortEdges(), torch.device("cpu"))


def test_segment_disagreement():
    heat = np.zeros((4, 4))
    heat[0, 1] = heat[1, 0] = 1.0
    heat[1, 2] = heat[2, 1] = 0.5
    assert np.isclose(segment_disagreement(heat), 1 - 1.5 / 3)      # edge (2, 3) outside the graph: p = 0


def test_beam_confidence(setup):
    coords, tour, heatmaps = setup
    pts = coords[tour[:20]]
    order, logp, margin = beam_with_confidence(heatmaps([pts])[0], pts)
    assert order[0] == 0 and order[-1] == 19 and sorted(order.tolist()) == list(range(20))
    assert logp <= 0 and margin >= 0


def test_window_log(setup):
    coords, tour, heatmaps = setup
    log = window_log(coords, tour, 20, heatmaps)
    assert len(log["start"]) == 30 and set(log["offset"].tolist()) == {0, 10}
    assert np.all(log["lkh_len"] <= log["cur_len"] + 1e-9)          # the current order is one of LKH's options
    assert np.all(log["lkh_len"] <= log["model_len"] + 1e-6)
    assert np.all((log["disagreement"] >= 0) & (log["disagreement"] <= 1))
    lkh_improves, catches = targets(log)
    assert lkh_improves.any()
    th = choose_thresholds(log, (10, 50))
    assert np.isclose((log["disagreement"] >= th["dis"]["50"]).mean(), 0.5, atol=0.05)
    assert th["conf_signal"] in ("logp", "margin")


@pytest.mark.parametrize("policy", [{"kind": "model"}, {"kind": "lkh"}, {"kind": "dis", "dis": 0.3},
                                    {"kind": "mixed", "dis": 0.3, "conf": -0.5, "conf_signal": "logp"}])
def test_triage_refine_valid_and_never_worse(setup, policy):
    coords, tour, heatmaps = setup
    new, stats = triage_refine(coords, tour, 20, policy, heatmaps)
    assert is_valid_tour(new, 300)
    before, after = tour_length(coords, tour), tour_length(coords, new)
    assert after <= before + 1e-12 and np.isclose(before - after, stats["gain"])
    assert stats["to_model"] + stats["to_lkh"] <= stats["windows"] and 1 <= stats["passes"] <= 3


def test_triage_limits(setup):
    """Extreme thresholds reduce P2 / P3 to "nothing", "LKH everywhere" and "model everywhere"."""
    coords, tour, heatmaps = setup
    run = lambda p: triage_refine(coords, tour, 20, p, heatmaps)
    t_none, s_none = run({"kind": "dis", "dis": np.inf})
    assert np.array_equal(t_none, tour) and s_none["to_lkh"] == 0
    assert np.array_equal(run({"kind": "dis", "dis": -np.inf})[0], run({"kind": "lkh"})[0])
    t_mix, s_mix = run({"kind": "mixed", "dis": np.inf, "conf": -np.inf, "conf_signal": "margin"})
    assert np.array_equal(t_mix, run({"kind": "model"})[0]) and s_mix["to_model"] == s_mix["windows"]
