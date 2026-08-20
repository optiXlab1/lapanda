"""Exp.1 scaling study for lapanda and CasADi solver differentiation.

This script runs independent single-shot solves without warm-starting across
trials.  It records raw trial rows and a compact summary table.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.common import append_csv, fresh_output_path, git_commit_hash, machine_info, relative_error, sampled_call, write_json
from experiments.exp1_rosenbrock_smooth_constraints.casadi_sensitivity import (
    build_casadi_sensitivity,
    build_ipopt_solver,
    evaluate_sensitivity,
    solve_ipopt,
)
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    active_constraint_count,
    box_violation_inf,
    build_problem,
    constraint_violation_inf,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver


def inner_options(args):
    options = SolverOptions()
    options.max_iterations = args.inner_max_iter
    options.tolerance = args.inner_tol
    options.buffer_size = 10
    options.max_stable_iter = 120
    return options


def alm_options(args):
    options = AlmOptions()
    options.max_iterations = args.max_outer
    options.tolerance = args.alm_tol
    options.initial_penalty = args.initial_penalty
    options.penalty_update_factor = args.penalty_update_factor
    options.max_penalty = args.max_penalty
    options.sufficient_decrease_factor = 0.25
    options.warm_start_inner = False
    return options


def backward_options(args):
    options = BackwardOptions()
    options.enable = True
    options.tolerance = args.backward_tol
    options.max_iterations = args.backward_max_iter
    options.constraint_penalty_scale = args.backward_penalty_scale
    options.constraint_penalty_max = args.backward_penalty_max
    return options


def solve_alm(solver, instance, x0, theta, target, args):
    return solver.solve_lapanda(
        x0,
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=alm_options(args),
        backward_options=backward_options(args),
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    )


def finite_or_nan(value):
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def summarize_rows(raw_rows, summary_path):
    numeric_fields = [
        "build_time_sec",
        "build_rss_peak_delta_mb",
        "forward_time_sec",
        "backward_time_sec",
        "forward_rss_peak_delta_mb",
        "backward_rss_peak_delta_mb",
        "memory_peak_delta_mb",
        "total_time_sec",
        "objective_value",
        "outer_loss_value",
        "constraint_violation_inf",
        "box_violation_inf",
        "gradient_relative_error_vs_casadi",
        "solution_error_vs_casadi",
        "active_constraint_ratio",
        "num_forward_iterations",
        "num_backward_iterations",
        "alm_outer_iterations",
        "alm_inner_iterations_total",
        "alm_final_residual",
        "alm_final_penalty",
    ]
    groups = {}
    for row in raw_rows:
        if row["status"] != "ok":
            continue
        key = (row["method"], row["n"], row["m"])
        groups.setdefault(key, []).append(row)
    for (method, n, m), rows in groups.items():
        summary = {"method": method, "n": n, "m": m, "trials": len(rows)}
        for field in numeric_fields:
            values = np.asarray([float(r[field]) for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        append_csv(summary_path, [summary])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 200, 500, 1000])
    parser.add_argument("--warmup-size", type=int, default=20, help="Run this size first as warm-up and exclude it from CSV summaries.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-3, help="Shared forward/backward tolerance for fair timing.")
    parser.add_argument("--inner-max-iter", type=int, default=4000)
    parser.add_argument("--inner-tol", type=float, default=None)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--alm-tol", type=float, default=None)
    parser.add_argument("--initial-penalty", type=float, default=2.0)
    parser.add_argument("--penalty-update-factor", type=float, default=10.0)
    parser.add_argument("--max-penalty", type=float, default=1e8)
    parser.add_argument("--backward-tol", type=float, default=None)
    parser.add_argument("--backward-max-iter", type=int, default=800)
    parser.add_argument("--backward-penalty-scale", type=float, default=1.0)
    parser.add_argument("--backward-penalty-max", type=float, default=0.0)
    parser.add_argument("--ipopt-tol", type=float, default=None)
    parser.add_argument("--ipopt-max-iter", type=int, default=3000)
    parser.add_argument("--active-tol", type=float, default=None)
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("--sample-trial", type=int, default=0)
    args = parser.parse_args()
    args.inner_tol = args.tol if args.inner_tol is None else args.inner_tol
    args.alm_tol = args.tol if args.alm_tol is None else args.alm_tol
    args.backward_tol = args.tol if args.backward_tol is None else args.backward_tol
    args.ipopt_tol = args.tol if args.ipopt_tol is None else args.ipopt_tol
    args.active_tol = args.tol if args.active_tol is None else args.active_tol

    out_dir = Path(__file__).resolve().parent / "results"
    raw_path = fresh_output_path(out_dir / "exp1_scaling_raw.csv")
    summary_path = fresh_output_path(out_dir / "exp1_scaling_summary.csv")
    sample_path = out_dir / "exp1_constraint_sample.npz"
    raw_rows = []
    theta = theta_nominal()

    write_json(
        out_dir / "config_scaling.json",
        {
        "experiment": "exp1_radius_chain_scaling_casadi_sensitivity",
            "sizes": args.sizes,
            "warmup_size": args.warmup_size,
            "formal_recording_starts_at": min([size for size in args.sizes if size != args.warmup_size], default=None),
            "trials": args.trials,
            "constraint_stride": args.constraint_stride,
            "backend": args.backend,
            "tol": args.tol,
            "inner_tol": args.inner_tol,
            "alm_tol": args.alm_tol,
            "backward_tol": args.backward_tol,
            "ipopt_tol": args.ipopt_tol,
            "active_tol": args.active_tol,
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
            "casadi_baseline": "IPOPT forward plus CasADi sqpmethod/qpoases solver differentiation backward",
            "note_safepdp": (
                "SafePDP is not run in Exp.1 because this radius-chain problem is a "
                "general non-OCP NLP. SafePDP should be used in the OCP imitation experiments."
            ),
        },
    )

    for n in args.sizes:
        is_warmup = n == args.warmup_size
        instance = build_problem(n, constraint_stride=args.constraint_stride)
        target = sinusoidal_target(n)
        print(f"building solvers for n={n}, m={instance.m}" + (" [warm-up only]" if is_warmup else ""))
        alm_build = sampled_call(
            lambda: build_solver(
                instance.problem,
                backend=args.backend,
                name=f"exp1_scaling_n{n}",
                cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp1_scaling",
                force=args.force,
            )
        )
        alm_solver = alm_build["value"]
        ipopt_build = sampled_call(lambda: build_ipopt_solver(instance, args.ipopt_tol, args.ipopt_max_iter, f"exp1_ipopt_n{n}"))
        ipopt_solver = ipopt_build["value"]
        casadi_sensitivity_build = sampled_call(
            lambda: build_casadi_sensitivity(n, args.constraint_stride, args.tol, args.ipopt_max_iter, f"exp1_casadi_n{n}")
        )
        casadi_sensitivity = casadi_sensitivity_build["value"]

        for trial_id in range(args.trials):
            x0 = initial_point(n, 100 + trial_id)
            casadi_forward = sampled_call(lambda: solve_ipopt(instance, ipopt_solver, x0, theta))
            ipopt_result = casadi_forward["value"]
            ipopt_stats = ipopt_solver.stats()
            ipopt_solution = np.asarray(ipopt_result["x"], dtype=float).reshape(-1)
            casadi_backward = sampled_call(
                lambda: evaluate_sensitivity(casadi_sensitivity, theta, target, ipopt_solution)
            )
            casadi_solution_raw, casadi_loss_raw, casadi_grad_raw = casadi_backward["value"]
            casadi_solution = np.asarray(casadi_solution_raw, dtype=float).reshape(-1)
            casadi_grad = np.asarray(casadi_grad_raw, dtype=float).reshape(-1)

            casadi_row = {
                "method": "CasADi-IPOPT-SQP",
                "status": "ok",
                "n": n,
                "m": instance.m,
                "trial_id": trial_id,
                "build_time_sec": ipopt_build["elapsed_sec"] + casadi_sensitivity_build["elapsed_sec"],
                "build_rss_peak_delta_mb": max(ipopt_build["rss_peak_delta_mb"], casadi_sensitivity_build["rss_peak_delta_mb"]),
                "forward_time_sec": casadi_forward["elapsed_sec"],
                "backward_time_sec": casadi_backward["elapsed_sec"],
                "forward_rss_peak_delta_mb": casadi_forward["rss_peak_delta_mb"],
                "backward_rss_peak_delta_mb": casadi_backward["rss_peak_delta_mb"],
                "memory_peak_delta_mb": max(casadi_forward["rss_peak_delta_mb"], casadi_backward["rss_peak_delta_mb"]),
                "solve_rss_peak_delta_mb": casadi_forward["rss_peak_delta_mb"],
                "total_time_sec": casadi_forward["elapsed_sec"] + casadi_backward["elapsed_sec"],
                "objective_value": float(ipopt_result["f"]),
                "outer_loss_value": float(casadi_loss_raw),
                "constraint_violation_inf": constraint_violation_inf(instance, casadi_solution, theta),
                "box_violation_inf": box_violation_inf(instance, casadi_solution),
                "gradient_theta": casadi_grad.tolist(),
                "gradient_relative_error_vs_casadi": 0.0,
                "solution_error_vs_casadi": relative_error(casadi_solution, ipopt_solution),
                "active_constraint_count": active_constraint_count(instance, casadi_solution, theta, args.active_tol),
                "active_constraint_ratio": active_constraint_count(instance, casadi_solution, theta, args.active_tol) / max(instance.m, 1),
                "active_box_count": int(np.sum(casadi_solution <= instance.lower + args.active_tol) + np.sum(casadi_solution >= instance.upper - args.active_tol)),
                "num_forward_iterations": int(ipopt_stats.get("iter_count", -1)),
                "num_backward_iterations": -1,
                "alm_outer_iterations": -1,
                "alm_inner_iterations_total": -1,
                "alm_final_residual": float("nan"),
                "alm_final_penalty": float("nan"),
                "casadi_backward_mode": "sqpmethod_qpoases_solver_differentiation",
                "return_status": ipopt_stats.get("return_status", ""),
            }
            if not is_warmup:
                append_csv(raw_path, [casadi_row])
                raw_rows.append(casadi_row)

            alm_solve = sampled_call(lambda: solve_alm(alm_solver, instance, x0, theta, target, args))
            alm_result = alm_solve["value"]
            alm_solution = np.asarray(alm_result["solution"], dtype=float)
            alm_grad = np.asarray(alm_result["grad_theta"], dtype=float)
            inner_iterations = np.asarray(alm_result["inner_iterations"], dtype=int)
            alm_outer_iterations = int(alm_result.get("iterations", inner_iterations.size))
            alm_inner_iterations_total = int(np.sum(inner_iterations))
            alm_row = {
                "method": "lapanda",
                "status": "ok",
                "n": n,
                "m": instance.m,
                "trial_id": trial_id,
                "build_time_sec": alm_build["elapsed_sec"],
                "build_rss_peak_delta_mb": alm_build["rss_peak_delta_mb"],
                "forward_time_sec": finite_or_nan(alm_result.get("forward_time_sec", np.nan)),
                "backward_time_sec": finite_or_nan(alm_result.get("backward_time_sec", np.nan)),
                "forward_rss_peak_delta_mb": alm_solve["rss_peak_delta_mb"],
                "backward_rss_peak_delta_mb": float("nan"),
                "memory_peak_delta_mb": alm_solve["rss_peak_delta_mb"],
                "solve_rss_peak_delta_mb": alm_solve["rss_peak_delta_mb"],
                "total_time_sec": alm_solve["elapsed_sec"],
                "objective_value": float(instance.objective_fun(alm_solution, theta)),
                "outer_loss_value": float(instance.outer_loss_fun(alm_solution, target)),
                "constraint_violation_inf": constraint_violation_inf(instance, alm_solution, theta),
                "box_violation_inf": box_violation_inf(instance, alm_solution),
                "gradient_theta": alm_grad.tolist(),
                "gradient_relative_error_vs_casadi": relative_error(alm_grad, casadi_grad),
                "solution_error_vs_casadi": relative_error(alm_solution, casadi_solution),
                "active_constraint_count": active_constraint_count(instance, alm_solution, theta, args.active_tol),
                "active_constraint_ratio": active_constraint_count(instance, alm_solution, theta, args.active_tol) / max(instance.m, 1),
                "active_box_count": int(np.sum(alm_solution <= instance.lower + args.active_tol) + np.sum(alm_solution >= instance.upper - args.active_tol)),
                "num_forward_iterations": alm_inner_iterations_total,
                "num_backward_iterations": int(alm_result.get("backward_iterations", -1)),
                "alm_outer_iterations": alm_outer_iterations,
                "alm_inner_iterations_total": alm_inner_iterations_total,
                "alm_final_residual": finite_or_nan(alm_result.get("final_residual", np.nan)),
                "alm_final_penalty": finite_or_nan(alm_result.get("penalty", np.nan)),
                "casadi_backward_mode": "",
                "return_status": "ok",
            }
            if not is_warmup:
                append_csv(raw_path, [alm_row])
                raw_rows.append(alm_row)

            if n == args.sample_n and trial_id == args.sample_trial:
                np.savez(
                    sample_path,
                    n=n,
                    trial_id=trial_id,
                    theta=theta,
                    target=target,
                    x0=x0,
                    alm_solution=alm_solution,
                    casadi_solution=casadi_solution,
                    alm_constraints=np.asarray(instance.constraint_fun(alm_solution, theta), dtype=float).reshape(-1),
                    casadi_constraints=np.asarray(instance.constraint_fun(casadi_solution, theta), dtype=float).reshape(-1),
                )

            print(
                f"n={n} trial={trial_id}: "
                f"{'warm-up; ' if is_warmup else ''}"
                f"ALM fwd={alm_row['forward_time_sec']:.4f}s bwd={alm_row['backward_time_sec']:.4f}s "
                f"outer={alm_row['alm_outer_iterations']} inner={alm_row['alm_inner_iterations_total']} "
                f"res={alm_row['alm_final_residual']:.2e} "
                f"viol={alm_row['constraint_violation_inf']:.2e} grad_err_casadi={alm_row['gradient_relative_error_vs_casadi']:.2e}; "
                f"CasADi fwd={casadi_row['forward_time_sec']:.4f}s bwd={casadi_row['backward_time_sec']:.4f}s "
                f"viol={casadi_row['constraint_violation_inf']:.2e}"
            )

    summarize_rows(raw_rows, summary_path)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    if sample_path.exists():
        print(f"wrote {sample_path}")


if __name__ == "__main__":
    main()
