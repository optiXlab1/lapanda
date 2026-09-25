# Exp. 1: constrained Rosenbrock benchmark

Run all commands from the repository root.

## Paper map

| Paper item | Run | Result used by the paper |
| --- | --- | --- |
| Table 2: matched-accuracy time and memory | `run_matched_accuracy_timing.py`, `run_matched_accuracy_casadi.py`, `run_matched_accuracy_memory.py`, then `run_matched_accuracy_summary.py` | `matched_accuracy/summary.csv` |
| Fig. 2: constraint activity at `n=200` | `run_constraint_activity.py` | `results/exp1_constraint_activity.pdf` |
| Table 5: solver settings | no separate run; the settings are the arguments below | configuration recorded with each matched-accuracy run |
| Table 6: penalty sweep at `n=200` | `run_penalty_gradient_sweep.py` | `results/exp1_penalty_gradient_sweep.csv` |
| Table 7: forward-converged penalty | `run_matched_accuracy_timing.py --methods lapanda_base ...` | `matched_accuracy/balanced/summary.csv` |
| Table 8: direct and aligned refinement | `post_forward_refinement/run_comparison.py` | `post_forward_refinement/results/post_forward_refinement.csv` |
| Table 9: matched gradient errors | same runs as Table 2 | `matched_accuracy/summary.csv` |

The generated PDF for Fig. 2 is copied to the paper as `figures/fig2.pdf`.
The tables are typeset in the paper from the listed CSV files.

## Tables 2 and 9

The paper uses `n = 100, 200, 500, 1000`, ten instances per size and five
timing repetitions. The forward tolerance is `1e-3`; both backward solvers use
`1e-2`. lapanda uses CG with automatic MINRES fallback. Explicit KKT uses
MINRES because the original KKT system is indefinite.

First run lapanda. This also writes the common high-accuracy KKT reference at
the aligned point:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_timing.py --sizes 100 200 500 1000 --trials 10 --repetitions 5 --penalty-scales 10 --refinement-tolerances 1e-3 --backward-tol 1e-2 --methods lapanda --linear-solver cg --reference-active-tol 1e-7 --reference-at-refined-point --save-refined-reference-csv experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\reference.csv --output-dir experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\lapanda
```

Then run Explicit KKT and CasADi against that reference:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_timing.py --sizes 100 200 500 1000 --trials 10 --repetitions 5 --methods explicit_kkt --backward-tol 1e-2 --linear-solver minres --reference-csv experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\reference.csv --output-dir experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\explicit

python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_casadi.py --reference-csv experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\reference.csv --output-dir experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\casadi
```

Measure memory in fresh processes and combine the results:

```powershell
python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_memory.py --trials 10 --output-dir experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\memory
python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_summary.py
```

## Figures and appendix tables

```powershell
# Fig. 2
python experiments\exp1_rosenbrock_smooth_constraints\run_constraint_activity.py

# Table 6
python experiments\exp1_rosenbrock_smooth_constraints\run_penalty_gradient_sweep.py

# Table 7
python experiments\exp1_rosenbrock_smooth_constraints\run_matched_accuracy_timing.py --sizes 100 200 500 1000 --trials 10 --repetitions 5 --methods lapanda_base --backward-tol 1e-2 --linear-solver cg --reference-at-base-point --save-base-reference-csv experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\base_reference.csv --output-dir experiments\exp1_rosenbrock_smooth_constraints\matched_accuracy\balanced

# Table 8
python experiments\exp1_rosenbrock_smooth_constraints\post_forward_refinement\run_comparison.py
```

`problem.py` defines the benchmark. `kkt_utils.py` contains the common KKT
reference routines, and `casadi_sensitivity.py` contains the CasADi baseline.
