"""Tests for fcnf_gnn.solver on instances with an optimum known by hand.  Run with: pytest tests/"""

import numpy as np

from fcnf_gnn.solver import is_feasible, solution_cost, solve_fcnf


def triangle():
    """s=0 (b=+2) -> t=2 (b=-2): direct arc 0 (c=1, F=10) or path 1, 2 through node 1 (c=1, F=1 each).
    Direct: 2·1 + 10 = 12. Path: 2·1 + 1 + 2·1 + 1 = 6 (optimal)."""
    return dict(b=np.array([2.0, 0, -2]), src=np.array([0, 0, 1]), dst=np.array([2, 1, 2]),
                c=np.array([1.0, 1, 1]), F=np.array([10.0, 1, 1]), u=np.array([5.0, 5, 5]))


def test_hand_optimum():
    r = solve_fcnf(triangle())
    assert r["status"] == "optimal" and r["mip_gap"] == 0.0
    assert abs(r["cost"] - 6.0) < 1e-9
    assert r["y"].tolist() == [0, 1, 1] and np.allclose(r["x"], [0, 2, 2])
    assert is_feasible(triangle(), r["x"], r["y"])
    assert abs(solution_cost(triangle(), r["x"], r["y"]) - r["cost"]) < 1e-9


def test_fixed_zero_and_infeasible():
    r = solve_fcnf(triangle(), fixed_zero=[1])           # path closed: must use the direct arc
    assert r["status"] == "optimal" and abs(r["cost"] - 12.0) < 1e-9 and r["y"].tolist() == [1, 0, 0]
    r = solve_fcnf(triangle(), fixed_zero=[0, 1])        # no path left
    assert r["status"] == "infeasible" and r["y"] is None


def test_capacity_splits_flow():
    """Capacity 1 on the cheap path: splitting costs (1+1+1+1) + (1+10) = 15 vs 12
    for the direct arc alone (capacity 5), so the optimum is the direct arc only."""
    inst = triangle()
    inst["u"] = np.array([5.0, 1, 1])
    r = solve_fcnf(inst)
    assert abs(r["cost"] - 12.0) < 1e-9 and r["y"].tolist() == [1, 0, 0]
    inst["F"] = np.array([3.0, 1, 1])                    # now direct-only = 2 + 3 = 5 < split 1+1+1+1+1+3 = 8
    assert abs(solve_fcnf(inst)["cost"] - 5.0) < 1e-9


def test_lp_relaxation_and_warm_start():
    r = solve_fcnf(triangle(), relax=True)               # y = x/u = 0.4 on the path: 4 + 0.4 + 0.4
    assert abs(r["cost"] - 4.8) < 1e-9
    r = solve_fcnf(triangle(), warm_start=[1, 0, 0])     # a worse start does not change the optimum
    assert abs(r["cost"] - 6.0) < 1e-9
