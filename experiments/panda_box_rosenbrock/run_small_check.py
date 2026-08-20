"""Small reproducibility check for Exp. 1.

Run from the repository root:

    python experiments/panda_box_rosenbrock/run_small_check.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from experiments.common import (
    append_csv,
    ensure_python_path,
    finite_difference_gradient,
    fresh_output_path,
    git_commit_hash,
    machine_info,
    relative_error,
    sampled_call,
    write_json,
)
from experiments.panda_box_rosenbrock.problem import (
    build_problem,
    initial_point,
    projected_residual_inf,
    sinusoidal_target,
    theta_nominal,
)

ensure_python_path()

from lapanda import BackwardOptions, SolverOptions, build_solver


def solver_options() -> SolverOptions:
    options = SolverOptions()
    options.max_iterations = 10000
    options.tolerance = 1e-3
    options.buffer_size = 10
    options.max_stable_iter = 200
    return options


def backward_options() -> BackwardOptions:
    options = BackwardOptions()
    options.enable = True
    options.tolerance = 1e-3
    options.max_iterations = 500
    return options


def solve_once(solver, x0, theta, target, compute_backward: bool):
    return solver.solve_panda(
        x0,
        theta,
        variable=target,
        solver_options=solver_options(),
        backward_options=backward_options() if compute_backward else BackwardOptions(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 500, 1000, 2000, 5000])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--fd-size-limit", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    result_path = fresh_output_path(out_dir / "exp1_small_check.csv")
    print(f"writing incremental results to {result_path}")

    write_json(
        out_dir / "config.json",
        {
            "experiment": "panda_box_rosenbrock_small_check",
            "sizes": args.sizes,
            "trials": args.trials,
            "backend": args.backend,
            "finite_difference_epsilon": args.epsilon,
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
        },
    )

    for n in args.sizes:
        instance = build_problem(n)
        build_stats = sampled_call(
            lambda: build_solver(
                instance.problem,
                backend=args.backend,
                name=f"exp1_rosenbrock_n{n}",
                cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp1",
                force=False,
            )
        )
        solver = build_stats["value"]
        theta = theta_nominal()
        target = sinusoidal_target(n)
        for trial_id in range(args.trials):
            seed = trial_id
            x0 = initial_point(n, seed)
            solve_stats = sampled_call(
                lambda: solve_once(solver, x0, theta, target, compute_backward=True)
            )
            result = solve_stats["value"]
            solution = np.asarray(result["solution"])
            objective = float(instance.objective_fun(solution, theta))
            loss = float(instance.outer_loss_fun(solution, target))
            gradient = np.asarray(result["grad_theta"])
            gradient_reference = np.full_like(gradient, np.nan)
            gradient_error = float("nan")

            if n <= args.fd_size_limit:
                def loss_at_theta(theta_value):
                    fd_result = solve_once(solver, x0, theta_value, target, compute_backward=False)
                    return float(instance.outer_loss_fun(np.asarray(fd_result["solution"]), target))

                gradient_reference = finite_difference_gradient(loss_at_theta, theta, args.epsilon)
                gradient_error = relative_error(gradient, gradient_reference)

            row = {
                "n": n,
                "solver_name": f"PANDA-{args.backend}",
                "trial_id": trial_id,
                "build_time_sec": build_stats["elapsed_sec"],
                "build_rss_before_mb": build_stats["rss_before_mb"],
                "build_rss_peak_mb": build_stats["rss_peak_mb"],
                "build_rss_after_mb": build_stats["rss_after_mb"],
                "build_rss_peak_delta_mb": build_stats["rss_peak_delta_mb"],
                "build_rss_samples": build_stats["rss_samples"],
                "total_wall_time_sec": solve_stats["elapsed_sec"],
                "forward_time_sec": float(result.get("forward_time_sec", np.nan)),
                "backward_time_sec": float(result.get("backward_time_sec", np.nan)),
                "solve_rss_before_mb": solve_stats["rss_before_mb"],
                "solve_rss_peak_mb": solve_stats["rss_peak_mb"],
                "solve_rss_after_mb": solve_stats["rss_after_mb"],
                "solve_rss_peak_delta_mb": solve_stats["rss_peak_delta_mb"],
                "solve_rss_samples": solve_stats["rss_samples"],
                "objective_value": objective,
                "outer_loss_value": loss,
                "projected_residual_inf": projected_residual_inf(instance, solution, theta, result["gamma"]),
                "num_iterations": int(result["iterations"]),
                "forward_final_residual": float(result["final_residual"]),
                "success": bool(result["final_residual"] < solver_options().tolerance * 10.0),
                "gradient_theta": gradient.tolist(),
                "gradient_reference": gradient_reference.tolist(),
                "gradient_relative_error": gradient_error,
                "adjoint_residual_inf": float(result.get("backward_residual", np.nan)),
                "krylov_iterations": int(result.get("backward_iterations", -1)),
            }
            append_csv(result_path, [row])
            print(
                f"Exp1 n={n} trial={trial_id}: "
                f"iters={row['num_iterations']}, residual={result['final_residual']:.3e}, "
                f"grad_err={gradient_error:.3e}"
            )

    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
