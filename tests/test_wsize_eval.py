"""Tests for scripts/wsize_eval.py (window size vs. solver).  Run with: pytest tests/"""

import importlib.util
import os

import numpy as np

from tsp_gnn.data import generate_instances
from tsp_gnn.window import initial_tour, make_solver, refine

_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
_spec = importlib.util.spec_from_file_location("wsize_eval", os.path.join(_dir, "wsize_eval.py"))
wsize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsize)


def test_batch_size_keeps_edges_bounded():
    assert wsize.batch_size(20) == 256 and wsize.batch_size(300) == 1
    for w in (20, 50, 100, 200, 300):
        assert wsize.batch_size(w) * w * round(0.5 * w) <= max(wsize.EDGES_PER_BATCH, w * round(0.5 * w))


def test_refine_passes_equals_refine():
    coords = generate_instances(1, 300, 7)[0]
    t0 = initial_tour(coords, 50)
    solver = make_solver("2opt")
    lkh = make_solver("lkh")
    for s in (solver, lkh):
        a, sa = refine(coords, t0, 30, s, max_passes=2)
        b, sb, first_ms = wsize.refine_passes(coords, t0, 30, s, 2)
        assert np.array_equal(a, b) and sa["passes"] == sb["passes"] and np.isclose(sa["gain"], sb["gain"])
        assert first_ms > 0


def test_refine_passes_time_limit_stops_after_one_pass():
    coords = generate_instances(1, 300, 8)[0]
    t0 = initial_tour(coords, 50)
    _, stats, _ = wsize.refine_passes(coords, t0, 30, make_solver("lkh"), 2, limit_ms=0.0)
    assert stats["passes"] == 1
