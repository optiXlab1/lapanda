"""Exp.1 lapanda memory study.

This follows the Exp.1 memory-recording style: RSS is sampled while each stage
runs and `rss_peak_delta_mb` is reported.  This captures temporary codegen and
compile memory that is invisible in a simple after-before RSS delta.
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

from experiments.common import append_csv, fresh_output_path, git_commit_hash, machine_info, sampled_call, write_json
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from experiments.exp1_rosenbrock_smooth_constraints.run_scaling_study import (
    alm_options,
    backward_options,
    inner_options,
)
from lapanda import build_solver


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


def summarize(raw_rows, summary_path):
    groups = {}
    for row in raw_rows:
        key = (row["method"], row["n"], row["m"])
        groups.setdefault(key, []).append(row)
    fields = [
        "problem_build_time_sec",
        "problem_build_rss_peak_delta_mb",
        "solver_build_time_sec",
        "solver_build_rss_peak_delta_mb",
        "solve_time_sec",
        "solve_rss_peak_delta_mb",
        "forward_time_sec",
        "backward_time_sec",
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
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--inner-max-iter", type=int, default=4000)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--initial-penalty", type=float, default=2.0)
    parser.add_argument("--penalty-update-factor", type=float, default=10.0)
    parser.add_argument("--max-penalty", type=float, default=1e8)
    parser.add_argument("--backward-max-iter", type=int, default=800)
    parser.add_argument("--backward-penalty-scale", type=float, default=1.0)
    parser.add_argument("--backward-penalty-max", type=float, default=0.0)
    args = parser.parse_args()
    args.inner_tol = args.tol
    args.alm_tol = args.tol
    args.backward_tol = args.tol
    args.active_tol = args.tol

    out_dir = Path(__file__).resolve().parent / "results"
    raw_path = fresh_output_path(out_dir / "exp1_lapanda_memory_raw.csv")
    summary_path = fresh_output_path(out_dir / "exp1_lapanda_memory_summary.csv")
    write_json(
        out_dir / "config_memory_lapanda.json",
        {
            "experiment": "exp1_lapanda_memory_peak_sampled",
            "sizes": args.sizes,
            "warmup_size": args.warmup_size,
            "formal_recording_starts_at": min([size for size in args.sizes if size != args.warmup_size], default=None),
            "trials": args.trials,
            "tol": args.tol,
            "backend": args.backend,
            "force": args.force,
            "memory_metric": "rss_peak_delta_mb sampled during each stage",
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
        print(f"lapanda memory n={n}, m={instance.m}" + (" [warm-up only]" if is_warmup else ""))

        solver_build = sampled_call(
            lambda: build_solver(
                instance.problem,
                backend=args.backend,
                name=f"exp1_memory_alm_n{n}",
                cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp1_memory",
                force=args.force,
            )
        )
        solver = solver_build["value"]

        for trial_id in range(args.trials):
            x0 = initial_point(n, 3000 + trial_id)
            solve = sampled_call(lambda: solve_alm(solver, instance, x0, theta, target, args))
            result = solve["value"]
            row = {
                "method": "lapanda",
                "n": n,
                "m": instance.m,
                "trial_id": trial_id,
                "problem_build_time_sec": problem_build["elapsed_sec"],
                "problem_build_rss_before_mb": problem_build["rss_before_mb"],
                "problem_build_rss_peak_mb": problem_build["rss_peak_mb"],
                "problem_build_rss_after_mb": problem_build["rss_after_mb"],
                "problem_build_rss_peak_delta_mb": problem_build["rss_peak_delta_mb"],
                "problem_build_rss_samples": problem_build["rss_samples"],
                "solver_build_time_sec": solver_build["elapsed_sec"],
                "solver_build_rss_before_mb": solver_build["rss_before_mb"],
                "solver_build_rss_peak_mb": solver_build["rss_peak_mb"],
                "solver_build_rss_after_mb": solver_build["rss_after_mb"],
                "solver_build_rss_peak_delta_mb": solver_build["rss_peak_delta_mb"],
                "solver_build_rss_samples": solver_build["rss_samples"],
                "solve_time_sec": solve["elapsed_sec"],
                "solve_rss_before_mb": solve["rss_before_mb"],
                "solve_rss_peak_mb": solve["rss_peak_mb"],
                "solve_rss_after_mb": solve["rss_after_mb"],
                "solve_rss_peak_delta_mb": solve["rss_peak_delta_mb"],
                "solve_rss_samples": solve["rss_samples"],
                "forward_time_sec": float(result.get("forward_time_sec", np.nan)),
                "backward_time_sec": float(result.get("backward_time_sec", np.nan)),
            }
            if not is_warmup:
                append_csv(raw_path, [row])
                raw_rows.append(row)
            print(
                f"  trial={trial_id}: {'warm-up; ' if is_warmup else ''}"
                f"problem_peak={row['problem_build_rss_peak_delta_mb']:.2f} MB, "
                f"build_peak={row['solver_build_rss_peak_delta_mb']:.2f} MB, "
                f"solve_peak={row['solve_rss_peak_delta_mb']:.2f} MB"
            )

    summarize(raw_rows, summary_path)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
