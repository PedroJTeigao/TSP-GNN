"""Tests for fcnf_gnn.decode: pruning with infeasibility repair, connectivity repair, the other decoders.
Run with: pytest tests/"""

import numpy as np

from fcnf_gnn.decode import connect_terminals, delegate, lp_round, prune, slope_scaling, trust_region, warm
from fcnf_gnn.generate import generate_instances
from fcnf_gnn.solver import is_feasible, solve_fcnf


def bottleneck():
    """Supply 0 (b=+10) -> demand 2 (b=-10). Arc 0: 0->2 with capacity 4 (the heatmap likes it);
    arcs 1, 2: 0->1->2 with capacity 10 (low p). Keeping arc 0 only is connected but infeasible."""
    inst = dict(b=np.array([10.0, 0, -10]), src=np.array([0, 0, 1]), dst=np.array([2, 1, 2]),
                c=np.array([1.0, 1, 1]), F=np.array([5.0, 5, 5]), u=np.array([4.0, 10, 10]))
    return inst, np.array([0.9, 0.2, 0.1])


def test_prune_repairs_infeasibility():
    inst, p = bottleneck()
    r = prune(inst, p, q=33)                       # keeps ceil(0.99) = 1 arc (arc 0): BFS ok, MILP infeasible
    assert r["bfs_added"] == 0 and r["repairs"] == 2 and not r["first_feasible"]
    assert r["kept_frac"] == 1.0
    # optimum with all arcs: 10 units over 0->1->2 (cost 20 + 10) = 30 < 4 direct + 6 path (4+5 + 12+10 = 31)
    assert abs(r["cost"] - solve_fcnf(inst)["cost"]) < 1e-9 and is_feasible(inst, r["x"], r["y"])


def test_connectivity_repair_adds_likeliest_path():
    inst, _ = bottleneck()
    p = np.array([0.01, 0.9, 0.8])
    keep, added = connect_terminals(inst, p, np.array([False, True, False]))   # 0->1 kept, 1->2 missing
    assert added == 1 and keep.tolist() == [False, True, True]                  # completes 0->1->2, not 0->2


def test_decoders_feasible_on_random_instances():
    for inst in generate_instances(4, 20, seed=11):
        opt = solve_fcnf(inst)["cost"]
        p = np.random.default_rng(0).random(len(inst["src"]))
        heur = slope_scaling(inst)
        results = [prune(inst, p, 20), prune(inst, p, 50), warm(inst, p), heur, delegate(inst, p, heuristic=heur),
                   lp_round(inst)]
        for r in results:
            assert r["x"] is not None and is_feasible(inst, r["x"], r["y"])
            assert r["cost"] >= opt - 1e-6 * opt
        assert abs(results[2]["cost"] - opt) < 1e-6 * opt                       # warm start = full MILP
        assert results[4]["cost"] <= heur["cost"] + 1e-9                         # delegation never worse


def test_trust_region_budget():
    inst, p = bottleneck()                         # q = 33: arc 0 free, arcs 1, 2 outside the prediction
    assert trust_region(inst, p, q=33, delta=1)["x"] is None             # 0->1->2 needs 2 outside arcs
    r = trust_region(inst, p, q=33, delta=2)
    assert r["outside_used"] == 2 and abs(r["cost"] - solve_fcnf(inst)["cost"]) < 1e-9
    assert is_feasible(inst, r["x"], r["y"])
    for inst in generate_instances(3, 20, seed=12):                     # delta = m: same as the full MILP
        p = np.random.default_rng(1).random(len(inst["src"]))
        r = trust_region(inst, p, q=20, delta=len(p))
        assert abs(r["cost"] - solve_fcnf(inst)["cost"]) < 1e-6 * r["cost"]
