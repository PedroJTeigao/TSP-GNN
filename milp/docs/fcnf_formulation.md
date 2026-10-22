# Fixed-charge network flow (FCNF): formulation

## Data

- Directed graph G = (V, A), |V| = n nodes, |A| = m arcs.
- Each node i ∈ V has a net supply b_i: b_i > 0 is a supply (source), b_i < 0 a demand (sink),
  b_i = 0 a transshipment node. The problem is balanced: Σ_i b_i = 0.
- Each arc (i, j) ∈ A has
  - a variable cost c_ij ≥ 0 per unit of flow,
  - a fixed cost F_ij ≥ 0, paid once if the arc carries any flow ("the arc is open"),
  - a capacity u_ij > 0.

## Variables

- x_ij ≥ 0: flow on arc (i, j) (continuous).
- y_ij ∈ {0, 1}: 1 if arc (i, j) is open.

## Mixed-integer linear program

```
minimize    Σ_(i,j)∈A  c_ij x_ij + F_ij y_ij

subject to  Σ_(i,j)∈A x_ij − Σ_(j,i)∈A x_ji = b_i      for every node i ∈ V       (flow conservation)
            0 ≤ x_ij ≤ u_ij y_ij                       for every arc (i, j) ∈ A   (flow only on open arcs)
            y_ij ∈ {0, 1}                              for every arc (i, j) ∈ A
```

The first sum is the flow leaving i and the second the flow entering it, so supplies send out b_i
and demands receive −b_i. The coupling constraint x_ij ≤ u_ij y_ij makes y_ij = 1 whenever
x_ij > 0; since F_ij ≥ 0, an optimal solution never opens an arc without flow, so "open in the
optimum" and "carries flow in the optimum" are the same thing (up to the solver's tolerances).

## Why it is hard, and what the network learns

Without the fixed costs this is a min-cost flow problem (an LP with integral solutions, solved in
polynomial time). The fixed costs make the cost of an arc a concave step function of its flow,
which rewards sending flow along few shared arcs; the problem is NP-hard. The ratio F / (c u)
(fixed cost relative to the cost of filling the arc) controls how much the fixed costs matter.

The LP relaxation (y_ij ∈ [0, 1]) sets y_ij = x_ij / u_ij, charging only the fraction of F_ij
used: it is weak when capacities are loose, which is why branch and bound needs many nodes.

The labels for the GNN are y*_ij of an optimal solution (1 = arc open). The network predicts a
probability p_ij that each arc is open (the "heatmap"), and the heatmap guides the solver:
pruning arcs with low p (fixing y = 0), a warm start, or freeing only the arcs where the heatmap
and a cheap heuristic disagree.

## Implementation

- Solver: HiGHS (`highspy`), in `src/fcnf_gnn/solver.py` (`solve_fcnf`). Columns are
  x_0..x_(m−1) then y_0..y_(m−1); rows are the n conservation equalities then the m coupling rows
  x_a − u_a y_a ≤ 0. Arcs in `fixed_zero` get y_a ∈ [0, 0] and x_a ∈ [0, 0].
- Instances: `src/fcnf_gnn/generate.py` (see its docstring for the distributions).
