"""CasADi/IPOPT baseline for Exp. 1.

Run from the repository root:

    python experiments/panda_box_rosenbrock/run_casadi_ipopt.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import casadi as ca
import numpy as np

from experiments.common import (
    append_csv,
    fresh_output_path,
    git_commit_hash,
    machine_info,
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


def build_ipopt_solver(instance, tol: float, max_iter: int):
    nlp = {
        "x": instance.problem.u,
        "p": instance.problem.theta,
        "f": instance.problem.cost,
    }
    return ca.nlpsol(
        "exp1_ipopt",
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": tol,
            "ipopt.constr_viol_tol": tol,
            "ipopt.max_iter": max_iter,
        },
    )


def build_casadi_sensitivity_function(n: int, tol: float, max_iter: int):
    """Build a CasADi NLP-sensitivity function using Opti + sqpmethod.

    The returned Function takes (theta, target, initial_x) and returns
    (solution, outer_loss, d outer_loss / d theta).  This follows the
    parametric NLP sensitivity pattern from the CasADi blog.
    """

    opti = ca.Opti()
    x = opti.variable(n)
    theta = opti.parameter(3)
    target = opti.parameter(n)

    cost = ca.MX(0)
    for i in range(n - 1):
        cost += theta[0] * (x[i + 1] - x[i] ** 2) ** 2 + theta[1] * (1.0 - x[i]) ** 2
    cost += 0.5 * theta[2] * ca.dot(x, x)
    outer_loss = 0.5 * ca.sumsqr(x - target)

    opti.minimize(cost)
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.solver(
        "sqpmethod",
        {
            "qpsol": "qrqp",
            "qpsol_options": {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
            },
            "print_header": False,
            "print_iteration": False,
            "print_time": False,
            "max_iter": max_iter,
            "tol_pr": tol,
            "tol_du": tol,
        },
    )

    solve_fun = opti.to_function("exp1_casadi_solve", [theta, x], [x])
    th = ca.MX.sym("th", 3)
    tar = ca.MX.sym("tar", n)
    x0 = ca.MX.sym("x0", n)
    x_star = solve_fun(th, x0)
    loss_value = 0.5 * ca.sumsqr(x_star - tar)
    grad_theta = ca.jacobian(loss_value, th)
    return ca.Function(
        "exp1_casadi_sensitivity",
        [th, tar, x0],
        [x_star, loss_value, grad_theta],
    )


def solve_ipopt(solver, instance, x0, theta):
    return solver(
        x0=x0,
        p=theta,
        lbx=instance.lower,
        ubx=instance.upper,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 500, 1000, 2000])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--sensitivity-size-limit", type=int, default=200)
    parser.add_argument("--sensitivity-max-iter", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    result_path = fresh_output_path(out_dir / "exp1_casadi_ipopt.csv")
    print(f"writing incremental results to {result_path}")

    write_json(
        out_dir / "config_ipopt.json",
        {
            "experiment": "exp1_casadi_ipopt_box_rosenbrock",
            "sizes": args.sizes,
            "trials": args.trials,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "sensitivity_size_limit": args.sensitivity_size_limit,
            "sensitivity_max_iter": args.sensitivity_max_iter,
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
        },
    )

    theta = theta_nominal()
    for n in args.sizes:
        instance = build_problem(n)
        target = sinusoidal_target(n)
        build_stats = sampled_call(lambda: build_ipopt_solver(instance, args.tol, args.max_iter))
        solver = build_stats["value"]
        sensitivity_fun = None
        sensitivity_build_stats = None
        if n <= args.sensitivity_size_limit:
            try:
                sensitivity_build_stats = sampled_call(
                    lambda: build_casadi_sensitivity_function(
                        n,
                        args.tol,
                        args.sensitivity_max_iter,
                    )
                )
                sensitivity_fun = sensitivity_build_stats["value"]
            except Exception as exc:
                print(f"CasADi sensitivity build failed for n={n}: {exc}")
        for trial_id in range(args.trials):
            seed = trial_id
            x0 = initial_point(n, seed)
            solve_stats = sampled_call(lambda: solve_ipopt(solver, instance, x0, theta))
            result = solve_stats["value"]
            solution = np.asarray(result["x"]).reshape(-1)
            objective = float(result["f"])
            stats = solver.stats()
            iterations = stats.get("iter_count", -1)
            success = bool(stats.get("success", False))
            sensitivity_success = False
            sensitivity_time_sec = float("nan")
            sensitivity_rss_peak_delta_mb = float("nan")
            sensitivity_gradient_theta = [float("nan")] * 3
            sensitivity_loss_value = float("nan")
            sensitivity_error_message = ""

            if sensitivity_fun is not None:
                try:
                    sens_stats = sampled_call(lambda: sensitivity_fun(theta, target, solution))
                    sens_solution, sens_loss, sens_grad = sens_stats["value"]
                    sensitivity_time_sec = sens_stats["elapsed_sec"]
                    sensitivity_rss_peak_delta_mb = sens_stats["rss_peak_delta_mb"]
                    sensitivity_gradient_theta = np.asarray(sens_grad).reshape(-1).tolist()
                    sensitivity_loss_value = float(sens_loss)
                    sensitivity_success = True
                except Exception as exc:
                    sensitivity_error_message = str(exc)

            row = {
                "n": n,
                "solver_name": "CasADi-IPOPT",
                "trial_id": trial_id,
                "build_time_sec": build_stats["elapsed_sec"],
                "build_rss_before_mb": build_stats["rss_before_mb"],
                "build_rss_peak_mb": build_stats["rss_peak_mb"],
                "build_rss_after_mb": build_stats["rss_after_mb"],
                "build_rss_peak_delta_mb": build_stats["rss_peak_delta_mb"],
                "build_rss_samples": build_stats["rss_samples"],
                "sensitivity_build_time_sec": (
                    float("nan") if sensitivity_build_stats is None else sensitivity_build_stats["elapsed_sec"]
                ),
                "sensitivity_build_rss_peak_delta_mb": (
                    float("nan")
                    if sensitivity_build_stats is None
                    else sensitivity_build_stats["rss_peak_delta_mb"]
                ),
                "solve_wall_time_sec": solve_stats["elapsed_sec"],
                "solve_rss_before_mb": solve_stats["rss_before_mb"],
                "solve_rss_peak_mb": solve_stats["rss_peak_mb"],
                "solve_rss_after_mb": solve_stats["rss_after_mb"],
                "solve_rss_peak_delta_mb": solve_stats["rss_peak_delta_mb"],
                "solve_rss_samples": solve_stats["rss_samples"],
                "objective_value": objective,
                "outer_loss_value": float(0.5 * np.sum((solution - target) ** 2)),
                "projected_residual_inf_gamma1": projected_residual_inf(instance, solution, theta, 1.0),
                "num_iterations": int(iterations),
                "success": success,
                "return_status": stats.get("return_status", ""),
                "sensitivity_success": sensitivity_success,
                "sensitivity_time_sec": sensitivity_time_sec,
                "sensitivity_rss_peak_delta_mb": sensitivity_rss_peak_delta_mb,
                "sensitivity_loss_value": sensitivity_loss_value,
                "sensitivity_gradient_theta": sensitivity_gradient_theta,
                "sensitivity_error_message": sensitivity_error_message,
            }
            append_csv(result_path, [row])
            print(
                f"IPOPT n={n} trial={trial_id}: "
                f"iters={iterations}, success={success}, "
                f"time={solve_stats['elapsed_sec']:.3e}s"
            )

    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
