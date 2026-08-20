# Exp.1 Rosenbrock With Smooth General Constraints

This experiment is the paper Rosenbrock benchmark for general nonlinear
constraints.  It is not an OCP experiment.

## Problem

The benchmark solves

```text
min_x  sum_i theta_0 (x_{i+1} - x_i^2)^2
       + theta_1 (1 - x_i)^2
       + 0.5 theta_2 ||x||^2
s.t.   -2 <= x_i <= 2
       x_i^2 + x_{i+1}^2 - r_i(theta)^2 <= 0
```

The outer loss used for sensitivity evaluation is

```text
L(x*) = 0.5 ||x* - target||^2.
```

## File Roles

```text
problem.py
```

Defines the Rosenbrock objective, box bounds, nonlinear radius-chain
constraints, nominal parameter, initial points, and target vector.

```text
run_scaling_study.py
```

Main paper timing experiment.  It runs lapanda and CasADi, records forward
time, backward time, constraint violation, and lapanda errors relative to
the CasADi baseline.

```text
casadi_sensitivity.py
```

CasADi baseline implementation.  The forward pass uses IPOPT.  The backward
pass uses CasADi solver differentiation through `sqpmethod` with `qpoases`,
initialized around the IPOPT forward solution.

```text
run_memory_lapanda.py
run_memory_casadi.py
```

Separate memory measurements for lapanda and CasADi.  They use sampled RSS
peak deltas for build and solve stages.

```text
plot_constraint_activity.py
```

Plots the representative normalized constraint value
`sqrt(x_i^2+x_{i+1}^2)/r_i(theta)` for `N=200`.  Values below `1` are feasible;
values near `1` are active or nearly active.

## Run Commands

Timing and gradient accuracy:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\run_scaling_study.py
```

Memory:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\run_memory_lapanda.py
python experiments\exp1_rosenbrock_smooth_constraints\run_memory_casadi.py
```

Constraint figure:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\plot_constraint_activity.py
```

Run `run_scaling_study.py` first, because it writes
`results/exp1_constraint_sample.npz`, which is the input for this plot.

The defaults match the paper setting: warm-up size `N=20`, formal records
starting from `N=50`, shared tolerance `1e-3`, and `10` timing trials.

## Paper Result Files

Keep these files in `results/`:

```text
config_scaling.json
config_memory_lapanda.json
config_memory_casadi.json
exp1_scaling_raw.csv
exp1_scaling_summary.csv
exp1_constraint_sample.npz
exp1_constraint_activity.png
exp1_constraint_activity.pdf
exp1_constraint_activity.svg
exp1_lapanda_memory_raw.csv
exp1_lapanda_memory_summary.csv
exp1_casadi_memory_raw.csv
exp1_casadi_memory_summary.csv
```
