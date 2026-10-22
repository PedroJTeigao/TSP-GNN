"""Exact FCNF solver with HiGHS (formulation in docs/fcnf_formulation.md).

An instance is a dict with
    b       (n,)   net supply of each node (> 0 supply, < 0 demand, sum 0)
    src     (m,)   tail node of each arc
    dst     (m,)   head node of each arc
    c       (m,)   variable cost per unit of flow
    F       (m,)   fixed cost (paid if the arc is open)
    u       (m,)   capacity
Columns of the MILP: x_0..x_(m-1), then y_0..y_(m-1).
"""

import time

import highspy
import numpy as np

OPEN_TOL = 1e-6      # y > 0.5 is "open"; flows below this are treated as zero


def _build(inst, fixed_zero=None, relax=False, fixed_one=None):
    b, src, dst = np.asarray(inst["b"], float), np.asarray(inst["src"]), np.asarray(inst["dst"])
    c, F, u = (np.asarray(inst[k], float) for k in ("c", "F", "u"))
    n, m = len(b), len(src)
    x_upper = u.copy()
    y_upper = np.ones(m)
    if fixed_zero is not None:
        x_upper[np.asarray(list(fixed_zero), dtype=int)] = 0.0
        y_upper[np.asarray(list(fixed_zero), dtype=int)] = 0.0
    y_lower = np.zeros(m)
    if fixed_one is not None:
        y_lower[np.asarray(list(fixed_one), dtype=int)] = 1.0

    lp = highspy.HighsLp()
    lp.num_col_ = 2 * m
    lp.num_row_ = n + m
    lp.col_cost_ = np.concatenate([c, F])
    lp.col_lower_ = np.concatenate([np.zeros(m), y_lower])
    lp.col_upper_ = np.concatenate([x_upper, y_upper])
    # rows 0..n-1: out - in = b_i (equality); rows n..n+m-1: x_a - u_a y_a <= 0
    lp.row_lower_ = np.concatenate([b, np.full(m, -highspy.kHighsInf)])
    lp.row_upper_ = np.concatenate([b, np.zeros(m)])
    # column-wise matrix: x_a has +1 in row src, -1 in row dst, +1 in row n+a; y_a has -u_a in row n+a
    arcs = np.arange(m)
    x_index = np.stack([src, dst, n + arcs], axis=1).ravel()
    x_value = np.tile([1.0, -1.0, 1.0], m)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.concatenate([np.arange(0, 3 * m + 1, 3), 3 * m + np.arange(1, m + 1)])
    lp.a_matrix_.index_ = np.concatenate([x_index, n + arcs]).astype(np.int32)
    lp.a_matrix_.value_ = np.concatenate([x_value, -u])
    if not relax:
        lp.integrality_ = [highspy.HighsVarType.kContinuous] * m + [highspy.HighsVarType.kInteger] * m
    return lp


def solve_fcnf(inst, time_limit=None, fixed_zero=None, warm_start=None, relax=False, threads=1, fixed_one=None):
    """Solve one FCNF instance (MILP, or its LP relaxation with relax=True).

    time_limit   seconds (None = until optimality)
    fixed_zero   iterable of arc indices forced closed (y = 0, x = 0), e.g. the pruned arcs
    fixed_one    iterable of arc indices forced open (y = 1; F is paid even without flow)
    warm_start   (m,) 0/1 vector of y used as a starting point; HiGHS completes x itself
                 (it is only used if it can be completed to a feasible solution)
    Returns a dict: status ("optimal" / "time_limit" / "infeasible" / other HiGHS status),
    cost, x, y (None if no feasible solution was found), mip_gap (relative, 0 when optimal),
    bound (best lower bound), nodes, seconds.
    """
    m = len(inst["src"])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(_build(inst, fixed_zero, relax, fixed_one))
    if warm_start is not None and not relax:
        y0 = np.asarray(warm_start, dtype=float)
        h.setSolution(m, np.arange(m, 2 * m, dtype=np.int32), y0)
    start = time.perf_counter()
    h.run()
    seconds = time.perf_counter() - start

    model_status = h.getModelStatus()
    info = h.getInfo()
    status = {highspy.HighsModelStatus.kOptimal: "optimal",
              highspy.HighsModelStatus.kTimeLimit: "time_limit",
              highspy.HighsModelStatus.kInfeasible: "infeasible"}.get(model_status,
                                                                     h.modelStatusToString(model_status))
    has_solution = info.primal_solution_status == 2       # 2 = feasible point available
    out = {"status": status, "seconds": seconds, "cost": None, "x": None, "y": None,
           "mip_gap": None, "bound": None, "nodes": None}
    if not relax:
        out["nodes"] = int(info.mip_node_count)
        out["bound"] = float(info.mip_dual_bound)
    if has_solution:
        col = np.asarray(h.getSolution().col_value)
        x, y = col[:m], col[m:]
        out["x"] = np.where(x > OPEN_TOL, x, 0.0)
        out["y"] = y if relax else (y > 0.5).astype(np.int8)
        out["cost"] = float(info.objective_function_value)
        if not relax:
            out["mip_gap"] = 0.0 if status == "optimal" else float(info.mip_gap)
    return out


def solution_cost(inst, x, y):
    """Objective value of (x, y) recomputed from the data."""
    return float(np.dot(inst["c"], x) + np.dot(inst["F"], y))


def is_feasible(inst, x, y, tol=1e-6):
    """True if (x, y) satisfies conservation, 0 <= x <= u y and y binary."""
    b, src, dst, u = (np.asarray(inst[k]) for k in ("b", "src", "dst", "u"))
    x, y = np.asarray(x, float), np.asarray(y, float)
    net = np.zeros(len(b))
    np.add.at(net, src, x)
    np.add.at(net, dst, -x)
    scale = max(1.0, np.abs(b).max())
    return bool(np.all(np.abs(net - b) <= tol * scale) and np.all(x >= -tol * scale)
                and np.all(x <= u * y + tol * scale) and np.all((y == 0) | (y == 1)))
