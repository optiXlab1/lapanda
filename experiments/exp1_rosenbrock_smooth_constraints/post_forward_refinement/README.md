# Post-forward penalty refinement

This diagnostic compares two ways to use a larger penalty after the standard
forward ALM solve has converged:

1. `direct`: keep the converged forward solution and use a scaled penalty only
   in the backward operators;
2. `aligned`: scale the penalty, re-solve the final ALM subproblem from the
   converged primal-dual state, and differentiate the refined subproblem.

The first strategy is inexpensive but its backward system is not the
sensitivity system of the subproblem that produced the retained forward
solution.  The second strategy preserves forward-backward alignment and
separately records the additional subproblem-solve time.

The base and direct variants are evaluated in separate deterministic solver
calls so that both gradients can be recorded.  The script verifies that these
calls return the same forward solution; the reported cost of direct refinement
is its backward time because it requires no additional forward refinement.

For the aligned strategy, the script runs a forward-only refinement call to
measure its complete wall-clock cost, including call overhead and per-call
initialization, and a repeated call with differentiation enabled to record the
backward time.  The script verifies that both refined calls return the same
solution.

Run the paper configuration from the repository root:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\post_forward_refinement\run_comparison.py
```

The script writes the configuration and comparison table to `results/` in
this directory.
