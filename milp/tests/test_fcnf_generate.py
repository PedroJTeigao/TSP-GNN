"""Tests for fcnf_gnn.generate and fcnf_gnn.data.  Run with: pytest tests/"""

import numpy as np

from fcnf_gnn.data import get_instance, load_dataset, num_instances, save_dataset
from fcnf_gnn.generate import RATIOS, generate_instances, max_flow, strongly_connected
from fcnf_gnn.solver import is_feasible, solve_fcnf


def test_balance_connectivity_and_feasibility():
    for n in (20, 50):
        for inst in generate_instances(5, n, seed=7):
            b = inst["b"]
            assert b.sum() == 0
            assert (b > 0).sum() == round(0.2 * n) and (b < 0).sum() == round(0.2 * n)
            arcs = np.stack([inst["src"], inst["dst"]], 1)
            assert len(np.unique(arcs, axis=0)) == len(arcs) and (arcs[:, 0] != arcs[:, 1]).all()
            assert strongly_connected(n, arcs)[0]
            assert max_flow(n, arcs, inst["u"], b) >= b[b > 0].sum()      # capacity is enough
            assert inst["ratio"] in RATIOS
            assert np.allclose(inst["F"], inst["ratio"] * inst["c"] * inst["u"])
            assert np.allclose(inst["c"], 100 * inst["length"])


def test_arc_count_and_seed():
    a = generate_instances(3, 20, seed=1)
    b = generate_instances(3, 20, seed=1)
    assert all(np.array_equal(x["src"], y["src"]) and np.array_equal(x["b"], y["b"]) for x, y in zip(a, b))
    assert all(80 <= len(inst["src"]) <= 130 for inst in a)                # "~100 arcs" for n = 20


def test_save_load_roundtrip(tmp_path):
    insts = generate_instances(3, 20, seed=2)
    for inst in insts:
        r = solve_fcnf(inst)
        inst.update(x=r["x"], y=r["y"], cost=r["cost"], status=0, mip_gap=0.0, bound=r["bound"],
                    nodes=r["nodes"], seconds=r["seconds"])
    path = str(tmp_path / "d.npz")
    save_dataset(path, insts, n=20)
    d = load_dataset(path)
    assert num_instances(d) == 3
    for i, inst in enumerate(insts):
        back = get_instance(d, i)
        assert np.array_equal(back["src"], inst["src"]) and np.array_equal(back["y"], inst["y"])
        assert is_feasible(back, back["x"], back["y"]) and abs(back["cost"] - inst["cost"]) < 1e-9
