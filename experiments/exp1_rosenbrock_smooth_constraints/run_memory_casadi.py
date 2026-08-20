"""Exp.1 CasADi baseline memory study.

This mirrors the Exp.1 CasADi memory style: IPOPT forward-solver build,
solver-differentiation backward-function build, forward solve, and backward
sensitivity call are recorded as separate sampled RSS peak stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.common import append_csv, fresh_output_path, git_commit_hash, machine_info, sampled_call, write_json
from experiments.exp1_rosenbrock_smooth_constraints.casadi_sensitivity import (
    build_casadi_sensitivity,
    build_ipopt_solver,
    evaluate_sensitivity,
    solve_ipopt,
)
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)


def summarize(raw_rows, summary_path):
    groups = {}
    for row in raw_rows:
        key = (row["method"], row["n"], row["m"])
        groups.setdefault(key, []).append(row)
    fields = [
        "problem_build_time_sec",
        "problem_build_rss_peak_delta_mb",
        "forward_solver_build_time_sec",
        "forward_solver_build_rss_peak_delta_mb",
        "backward_function_build_time_sec",
        "backward_function_build_rss_peak_delta_mb",
        "forward_solve_time_sec",
        "forward_solve_rss_peak_delta_mb",
        "backward_solve_time_sec",
        "backward_solve_rss_peak_delta_mb",
    ]
    for (method, n, m), rows in groups.items():
        summary = {"method": method, "n": n, "m": m, "trials": len(rows)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            summary[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        append_csv(summary_path, [summary])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 200, 500, 1000])
    parser.add_argument("--warmup-size", type=int, default=20, help="Run this size first as warm-up and exclude it from CSV summaries.")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--ipopt-max-iter", type=int, default=3000)
    parser.add_argument("--sensitivity-max-iter", type=int, default=3000)
    parser.add_argument("--qpsol", choices=["qpoases", "qrqp"], default="qpoases")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    raw_path = fresh_output_path(out_dir / "exp1_casadi_memory_raw.csv")
    summary_path = fresh_output_path(out_dir / "exp1_casadi_memory_summary.csv")
    write_json(
        out_dir / "config_memory_casadi.json",
        {
            "experiment": "exp1_casadi_memory_peak_sampled",
            "sizes": args.sizes,
            "warmup_size": args.warmup_size,
            "formal_recording_starts_at": min([size for size in args.sizes if size != args.warmup_size], default=None),
            "trials": args.trials,
            "tol": args.tol,
            "memory_metric": "rss_peak_delta_mb sampled during each stage",
            "baseline": f"IPOPT forward solve plus sqpmethod/{args.qpsol} solver-differentiation backward",
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
        },
    )

    theta = theta_nominal()
    raw_rows = []
    for n in args.sizes:
        is_warmup = n == args.warmup_size
        problem_build = sampled_call(lambda: build_problem(n, constraint_stride=args.constraint_stride))
        instance = problem_build["value"]
        target = sinusoidal_target(n)
        print(f"CasADi memory n={n}, m={instance.m}" + (" [warm-up only]" if is_warmup else ""))

        forward_build = sampled_call(
            lambda: build_ipopt_solver(instance, args.tol, args.ipopt_max_iter, f"exp1_memory_ipopt_n{n}")
        )
        forward_solver = forward_build["value"]
        backward_build = sampled_call(
            lambda: build_casadi_sensitivity(
                n,
                args.constraint_stride,
                args.tol,
                args.sensitivity_max_iter,
                f"exp1_memory_casadi_n{n}",
                qpsol=args.qpsol,
            )
        )
        sensitivity_fun = backward_build["value"]

        for trial_id in range(args.trials):
            x0 = initial_point(n, 3000 + trial_id)
            forward_solve = sampled_call(lambda: solve_ipopt(instance, forward_solver, x0, theta))
            ipopt_result = forward_solve["value"]
            backward_initial = np.asarray(ipopt_result["x"], dtype=float).reshape(-1)
            backward_solve = sampled_call(lambda: evaluate_sensitivity(sensitivity_fun, theta, target, backward_initial))
            row = {
                "method": f"CasADi-IPOPT-SQP-{args.qpsol}",
                "n": n,
                "m": instance.m,
                "trial_id": trial_id,
                "problem_build_time_sec": problem_build["elapsed_sec"],
                "problem_build_rss_before_mb": problem_build["rss_before_mb"],
                "problem_build_rss_peak_mb": problem_build["rss_peak_mb"],
                "problem_build_rss_after_mb": problem_build["rss_after_mb"],
                "problem_build_rss_peak_delta_mb": problem_build["rss_peak_delta_mb"],
                "problem_build_rss_samples": problem_build["rss_samples"],
                "forward_solver_build_time_sec": forward_build["elapsed_sec"],
                "forward_solver_build_rss_before_mb": forward_build["rss_before_mb"],
                "forward_solver_build_rss_peak_mb": forward_build["rss_peak_mb"],
                "forward_solver_build_rss_after_mb": forward_build["rss_after_mb"],
                "forward_solver_build_rss_peak_delta_mb": forward_build["rss_peak_delta_mb"],
                "forward_solver_build_rss_samples": forward_build["rss_samples"],
                "backward_function_build_time_sec": backward_build["elapsed_sec"],
                "backward_function_build_rss_before_mb": backward_build["rss_before_mb"],
                "backward_function_build_rss_peak_mb": backward_build["rss_peak_mb"],
                "backward_function_build_rss_after_mb": backward_build["rss_after_mb"],
                "backward_function_build_rss_peak_delta_mb": backward_build["rss_peak_delta_mb"],
                "backward_function_build_rss_samples": backward_build["rss_samples"],
                "forward_solve_time_sec": forward_solve["elapsed_sec"],
                "forward_solve_rss_before_mb": forward_solve["rss_before_mb"],
                "forward_solve_rss_peak_mb": forward_solve["rss_peak_mb"],
                "forward_solve_rss_after_mb": forward_solve["rss_after_mb"],
                "forward_solve_rss_peak_delta_mb": forward_solve["rss_peak_delta_mb"],
                "forward_solve_rss_samples": forward_solve["rss_samples"],
                "backward_solve_time_sec": backward_solve["elapsed_sec"],
                "backward_solve_rss_before_mb": backward_solve["rss_before_mb"],
                "backward_solve_rss_peak_mb": backward_solve["rss_peak_mb"],
                "backward_solve_rss_after_mb": backward_solve["rss_after_mb"],
                "backward_solve_rss_peak_delta_mb": backward_solve["rss_peak_delta_mb"],
                "backward_solve_rss_samples": backward_solve["rss_samples"],
            }
            if not is_warmup:
                append_csv(raw_path, [row])
                raw_rows.append(row)
            print(
                f"  trial={trial_id}: {'warm-up; ' if is_warmup else ''}"
                f"problem_peak={row['problem_build_rss_peak_delta_mb']:.2f} MB, "
                f"ipopt_build_peak={row['forward_solver_build_rss_peak_delta_mb']:.2f} MB, "
                f"bwd_build_peak={row['backward_function_build_rss_peak_delta_mb']:.2f} MB, "
                f"fwd_peak={row['forward_solve_rss_peak_delta_mb']:.2f} MB, "
                f"bwd_peak={row['backward_solve_rss_peak_delta_mb']:.2f} MB"
            )

    summarize(raw_rows, summary_path)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
