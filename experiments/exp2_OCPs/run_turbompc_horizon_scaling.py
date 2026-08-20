"""Run the TurboMPC-GPU side of the constrained linear MPC scaling test.

Run this script inside the TurboMPC Python environment (normally under WSL).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_HORIZONS = (2, 5, 10, 20, 40, 80, 160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbompc-root", type=Path, default=Path.home() / "turbompc")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def block(tree) -> None:
    import jax

    jax.tree_util.tree_map(lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value, tree)


def problem_data(nx: int, nu: int, seed: int = 0):
    np.random.seed(seed)
    q = np.eye(nx)
    r = np.eye(nu)
    a_matrix = np.eye(nx) + 0.1 * np.random.randn(nx, nx)
    eigenvalues, eigenvectors = np.linalg.eig(a_matrix)
    for index, value in enumerate(eigenvalues):
        if abs(value) >= 1.0 - 1e-2:
            eigenvalues[index] = value / (abs(value) + 1e-2)
    a_matrix = (eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)).real
    b_matrix = np.random.randn(nx, nu)
    offset = 0.01 * np.random.randn(nx)
    initial_state = 5.0 * np.random.randn(nx)
    return q, r, a_matrix, b_matrix, offset, initial_state


def main() -> None:
    args = parse_args()
    linear_dir = args.turbompc_root / "benchmarking" / "linear-system"
    sys.path[:0] = [str(args.turbompc_root), str(linear_dir)]

    from jax import config

    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    from benchmark_problem_setup import build_turbompc_linear_problem
    from turbompc.problems.optimal_control_problem import OptimalControlProblem
    from turbompc.solvers import TurboMPCSolver
    from turbompc.utils.load_params import load_solver_params

    nx, nu = 8, 4
    q, r, a_matrix, b_matrix, offset, initial_state = problem_data(nx, nu)
    theta = jnp.ones(nx + nu, dtype=jnp.float64)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        dynamics, params = build_turbompc_linear_problem(horizon=horizon, umax=10.0, n_state=nx, n_ctrl=nu)
        params.update(
            {
                "initial_state": jnp.asarray(initial_state),
                "weights_penalization_reference_state_trajectory": jnp.diag(jnp.asarray(q)),
                "weights_penalization_control_squared": jnp.diag(jnp.asarray(r)),
                "dynamics_state_dot_params": {
                    "A": jnp.asarray(a_matrix) - jnp.eye(nx),
                    "B": jnp.asarray(b_matrix),
                    "b": jnp.asarray(offset),
                },
            }
        )
        problem = OptimalControlProblem(dynamics, params=params)
        solver_params = load_solver_params("turbompc.yaml")
        solver_params["num_sqp_iteration_max"] = 1
        solver_params["tol_convergence"] = 1e-3
        solver_params["warm_start_backward"] = True
        solver_params["linesearch"] = False
        solver_params["admm"]["max_iter"] = 100
        solver_params["admm"]["check_termination_every"] = 1
        solver_params["admm"]["eps_abs"] = 1e-3
        solver_params["admm"]["eps_rel"] = 1e-3
        solver = TurboMPCSolver(
            problem,
            solver_params,
            forward_backend="admm_fused_cudss",
            backward_backend="direct_cudss_ffi",
            use_full_hessian=True,
        )

        def weights(current_theta):
            return {
                "weights_penalization_reference_state_trajectory": current_theta[:nx],
                "weights_penalization_control_squared": jnp.tile(current_theta[nx:][None, :], (horizon + 1, 1)),
            }

        def objective(current_theta, guess):
            current_weights = weights(current_theta)
            solution = solver.solve(guess, problem_params=params, weights=current_weights)
            loss = 0.5 * (jnp.sum(solution.states**2) + jnp.sum(solution.controls**2))
            return loss, (solution.states, solution.controls, solution.convergence_error, solution.num_iter)

        forward = jax.jit(objective)
        value_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
        guess = solver.initial_guess(params)
        block(forward(theta, guess))
        block(value_grad(theta, guess))

        rows = []
        for repetition in range(args.repetitions):
            start = time.perf_counter()
            forward_result = forward(theta, guess)
            block(forward_result)
            forward_ms = 1000.0 * (time.perf_counter() - start)

            start = time.perf_counter()
            value_grad_result = value_grad(theta, guess)
            block(value_grad_result)
            total_ms = 1000.0 * (time.perf_counter() - start)
            backward_ms = max(total_ms - forward_ms, 0.0)
            (_, aux), _ = value_grad_result
            states, controls, convergence_error, num_iter = aux
            guess = guess._replace(states=states, controls=controls)
            row = {
                "repetition": repetition,
                "horizon": horizon,
                "forward_time_ms": forward_ms,
                "backward_time_ms": backward_ms,
                "value_grad_time_ms": total_ms,
                "constraint_violation_inf": float(convergence_error),
                "sqp_iterations": int(num_iter),
            }
            rows.append(row)
            print(
                f"TurboMPC-GPU N={horizon:4d} rep={repetition:02d} "
                f"fwd={forward_ms:9.3f}ms bwd={backward_ms:9.3f}ms "
                f"res={float(convergence_error):.3e}"
            )

        path = args.outdir / f"linear_h{horizon}_turbompc_gpu_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
