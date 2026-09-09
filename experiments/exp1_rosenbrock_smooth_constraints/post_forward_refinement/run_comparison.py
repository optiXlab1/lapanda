"""Compare two post-forward penalty-refinement strategies.

The direct strategy keeps the converged forward solution and only scales the
constraint penalty used by the backward operators.  The aligned strategy
scales the penalty, re-solves the final ALM subproblem from the converged
forward state, and differentiates the refined subproblem.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.common import git_commit_hash, machine_info, relative_error, write_json
from experiments.exp1_rosenbrock_smooth_constraints.casadi_sensitivity import (
    build_casadi_sensitivity,
    build_ipopt_solver,
    evaluate_sensitivity,
    solve_ipopt,
)
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    constraint_violation_inf,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inner_options(args: argparse.Namespace) -> SolverOptions:
    options = SolverOptions()
    options.max_iterations = args.inner_max_iter
    options.tolerance = args.tol
    options.buffer_size = 10
    options.max_stable_iter = 120
    options.verbose = 0
    return options


def forward_alm_options(args: argparse.Namespace) -> AlmOptions:
    options = AlmOptions()
    options.max_iterations = args.max_outer
    options.tolerance = args.tol
    options.initial_penalty = args.initial_penalty
    options.penalty_update_factor = args.penalty_update_factor
    options.max_penalty = args.max_penalty
    options.sufficient_decrease_factor = 0.25
    options.warm_start_inner = False
    options.verbose = 0
    return options


def refined_subproblem_options(args: argparse.Namespace) -> AlmOptions:
    options = AlmOptions()
    options.max_iterations = 1
    options.tolerance = args.tol
    options.initial_penalty = args.initial_penalty * args.penalty_scale
    options.penalty_update_factor = 1.0
    options.max_penalty = args.max_penalty
    options.sufficient_decrease_factor = 0.25
    options.warm_start_inner = True
    options.verbose = 0
    return options


def backward_options(
    args: argparse.Namespace,
    penalty_scale: float,
    enable: bool = True,
) -> BackwardOptions:
    options = BackwardOptions()
    options.enable = enable
    options.tolerance = args.tol
    options.max_iterations = args.backward_max_iter
    options.constraint_penalty_scale = penalty_scale
    options.constraint_penalty_max = args.max_penalty
    return options


def solve_forward(solver, instance, x0, theta, target, args, *, backward_scale: float | None):
    enable_backward = backward_scale is not None
    return solver.solve_lapanda(
        x0,
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=forward_alm_options(args),
        backward_options=backward_options(
            args,
            penalty_scale=1.0 if backward_scale is None else backward_scale,
            enable=enable_backward,
        ),
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    )


def solve_aligned_refinement(solver, instance, base, theta, target, args, *, enable_backward: bool):
    refined_penalties = args.penalty_scale * np.asarray(base["penalties"], dtype=float)
    return solver.solve_lapanda(
        np.asarray(base["solution"], dtype=float),
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=refined_subproblem_options(args),
        backward_options=backward_options(args, penalty_scale=1.0, enable=enable_backward),
        multiplier0=np.asarray(base["multipliers"], dtype=float),
        penalty0=refined_penalties,
        adjoint0=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 500, 1000])
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--penalty-scale", type=float, default=10.0)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--inner-max-iter", type=int, default=4000)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--initial-penalty", type=float, default=2.0)
    parser.add_argument("--penalty-update-factor", type=float, default=10.0)
    parser.add_argument("--max-penalty", type=float, default=1e8)
    parser.add_argument("--backward-max-iter", type=int, default=800)
    parser.add_argument("--ipopt-max-iter", type=int, default=3000)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "results"
    output_path = output_dir / "post_forward_refinement.csv"
    config_path = output_dir / "config.json"
    theta = theta_nominal()
    rows: list[dict] = []

    write_json(
        config_path,
        {
            "experiment": "exp1_post_forward_penalty_refinement",
            "sizes": args.sizes,
            "trial_id": args.trial_id,
            "constraint_stride": args.constraint_stride,
            "penalty_scale": args.penalty_scale,
            "tolerance": args.tol,
            "inner_max_iterations": args.inner_max_iter,
            "alm_max_iterations": args.max_outer,
            "initial_penalty": args.initial_penalty,
            "penalty_update_factor": args.penalty_update_factor,
            "max_penalty": args.max_penalty,
            "backward_max_iterations": args.backward_max_iter,
            "ipopt_max_iterations": args.ipopt_max_iter,
            "backend": args.backend,
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
        },
    )

    for n in args.sizes:
        instance = build_problem(n, constraint_stride=args.constraint_stride)
        target = sinusoidal_target(n)
        x0 = initial_point(n, 100 + args.trial_id)

        ipopt_solver = build_ipopt_solver(
            instance,
            args.tol,
            args.ipopt_max_iter,
            f"exp1_refine_ipopt_n{n}",
        )
        ipopt_solution = np.asarray(
            solve_ipopt(instance, ipopt_solver, x0, theta)["x"],
            dtype=float,
        ).reshape(-1)
        sensitivity = build_casadi_sensitivity(
            n,
            args.constraint_stride,
            args.tol,
            args.ipopt_max_iter,
            f"exp1_refine_casadi_n{n}",
        )
        _, _, casadi_grad_raw = evaluate_sensitivity(sensitivity, theta, target, ipopt_solution)
        casadi_grad = np.asarray(casadi_grad_raw, dtype=float).reshape(-1)

        solver = build_solver(
            instance.problem,
            backend=args.backend,
            name=f"exp1_post_forward_refinement_n{n}",
            cache_dir=(
                Path(tempfile.gettempdir())
                / "lapanda_experiments"
                / "exp1_post_forward_refinement"
            ),
            force=args.force,
        )

        base = solve_forward(
            solver,
            instance,
            x0,
            theta,
            target,
            args,
            backward_scale=1.0,
        )
        direct = solve_forward(
            solver,
            instance,
            x0,
            theta,
            target,
            args,
            backward_scale=args.penalty_scale,
        )
        aligned_forward_t0 = time.perf_counter()
        aligned_forward = solve_aligned_refinement(
            solver,
            instance,
            base,
            theta,
            target,
            args,
            enable_backward=False,
        )
        aligned_refinement_solve_time_sec = time.perf_counter() - aligned_forward_t0
        aligned = solve_aligned_refinement(
            solver,
            instance,
            base,
            theta,
            target,
            args,
            enable_backward=True,
        )

        base_penalties = np.asarray(base["penalties"], dtype=float)
        direct_solution = np.asarray(direct["solution"], dtype=float)
        aligned_solution = np.asarray(aligned_forward["solution"], dtype=float)
        aligned_backward_solution = np.asarray(aligned["solution"], dtype=float)
        direct_grad = np.asarray(direct["grad_theta"], dtype=float)
        base_grad = np.asarray(base["grad_theta"], dtype=float)
        aligned_grad = np.asarray(aligned["grad_theta"], dtype=float)
        base_solution = np.asarray(base["solution"], dtype=float)
        direct_base_mismatch = relative_error(direct_solution, base_solution)
        if direct_base_mismatch > 1e-10:
            raise RuntimeError(
                f"direct and reference forward solves differ for n={n}: "
                f"relative mismatch={direct_base_mismatch:.3e}"
            )
        aligned_repeat_mismatch = relative_error(aligned_solution, aligned_backward_solution)
        if aligned_repeat_mismatch > 1e-10:
            raise RuntimeError(
                f"aligned forward solves differ for n={n}: "
                f"relative mismatch={aligned_repeat_mismatch:.3e}"
            )

        row = {
            "n": n,
            "m": instance.m,
            "trial_id": args.trial_id,
            "base_penalty_max": float(np.max(base_penalties)),
            "refined_penalty_max": float(np.max(args.penalty_scale * base_penalties)),
            "base_gradient_relative_error_vs_casadi": relative_error(base_grad, casadi_grad),
            "base_constraint_violation_inf": constraint_violation_inf(
                instance, base_solution, theta
            ),
            "base_backward_time_sec": float(base["backward_time_sec"]),
            "direct_gradient_relative_error_vs_casadi": relative_error(direct_grad, casadi_grad),
            "direct_constraint_violation_inf": constraint_violation_inf(
                instance, direct_solution, theta
            ),
            "direct_backward_time_sec": float(direct["backward_time_sec"]),
            "direct_backward_iterations": int(direct["backward_iterations"]),
            "direct_backward_residual": float(direct["backward_residual"]),
            "aligned_gradient_relative_error_vs_casadi": relative_error(aligned_grad, casadi_grad),
            "aligned_constraint_violation_inf": constraint_violation_inf(
                instance, aligned_solution, theta
            ),
            "aligned_refinement_forward_time_sec": float(aligned_forward["forward_time_sec"]),
            "aligned_refinement_solve_wall_time_sec": float(aligned_refinement_solve_time_sec),
            "aligned_backward_time_sec": float(aligned["backward_time_sec"]),
            "aligned_total_refinement_time_sec": float(
                aligned_refinement_solve_time_sec + aligned["backward_time_sec"]
            ),
            "aligned_forward_iterations": int(
                np.sum(np.asarray(aligned_forward["inner_iterations"], dtype=int))
            ),
            "aligned_backward_iterations": int(aligned["backward_iterations"]),
            "aligned_forward_residual": float(aligned_forward["final_residual"]),
            "aligned_backward_residual": float(aligned["backward_residual"]),
            "direct_vs_base_solution_relative_error": direct_base_mismatch,
            "aligned_vs_base_solution_relative_change": relative_error(
                aligned_solution, base_solution
            ),
            "aligned_repeated_solution_relative_error": aligned_repeat_mismatch,
        }
        rows.append(row)
        print(
            f"n={n}: rho={row['base_penalty_max']:.3e} -> {row['refined_penalty_max']:.3e}; "
            f"base err={row['base_gradient_relative_error_vs_casadi']:.3e}; "
            f"direct err={row['direct_gradient_relative_error_vs_casadi']:.3e}, "
            f"bwd={1e3 * row['direct_backward_time_sec']:.2f} ms; "
            f"aligned err={row['aligned_gradient_relative_error_vs_casadi']:.3e}, "
            f"refine-solve={1e3 * row['aligned_refinement_solve_wall_time_sec']:.2f} ms, "
            f"bwd={1e3 * row['aligned_backward_time_sec']:.2f} ms, "
            f"total={1e3 * row['aligned_total_refinement_time_sec']:.2f} ms"
        )

    write_csv(output_path, rows)
    print(f"wrote {output_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
