"""Measure build- and solve-stage RSS for the 2%-accuracy comparison.

Each method/size/trial is measured in a fresh subprocess.  Construction and
the first complete solve are sampled separately so their peak RSS increments
can be reported using the same protocol for all methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.common import sampled_call
from experiments.exp1_rosenbrock_smooth_constraints.kkt_utils import (
    alm_options,
    backward_options,
    build_reference_ipopt_solver,
    build_kkt_derivative_function,
    explicit_kkt_minres_backward,
    inner_options,
)
from experiments.exp1_rosenbrock_smooth_constraints.casadi_sensitivity import (
    build_casadi_sensitivity,
    evaluate_sensitivity,
    suppress_native_output,
)
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from lapanda import AlmOptions, build_solver


LAPANDA_SCALE = {50: 10.0, 100: 10.0, 200: 10.0, 500: 10.0, 1000: 10.0}
EXPLICIT_TOLERANCE = {50: 1e-2, 100: 1e-2, 200: 1e-2, 500: 1e-2, 1000: 1e-2}


def experiment_args() -> SimpleNamespace:
    return SimpleNamespace(
        inner_tol=1e-3,
        alm_tol=1e-3,
        backward_tol=1e-2,
        inner_max_iter=4000,
        max_stable_iter=120,
        warm_start_inner=False,
        max_outer=20,
        initial_penalty=2.0,
        penalty_update_factor=10.0,
        max_penalty=1e8,
        backward_penalty_max=1e14,
        backward_max_iter=200,
        gmres_restart=40,
        lapanda_linear_solver="cg",
    )


def refined_alm_options(args) -> AlmOptions:
    options = AlmOptions()
    options.max_iterations = 1
    options.tolerance = args.alm_tol
    options.initial_penalty = args.initial_penalty
    options.penalty_update_factor = 1.0
    options.max_penalty = args.max_penalty
    options.sufficient_decrease_factor = 0.25
    options.warm_start_inner = True
    options.verbose = 0
    return options


def call_lapanda(solver, instance, x0, theta, target, args, penalty_scale):
    base_backward = backward_options(args, 1.0)
    base_backward.enable = False
    base = solver.solve_lapanda(
        x0,
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=alm_options(args),
        backward_options=base_backward,
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    )
    refined_penalties = penalty_scale * np.asarray(base["penalties"], dtype=float)
    refined = solver.solve_lapanda(
        np.asarray(base["solution"], dtype=float),
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=refined_alm_options(args),
        backward_options=backward_options(args, 1.0),
        multiplier0=np.asarray(base["multipliers"], dtype=float),
        penalty0=refined_penalties,
        adjoint0=None,
    )
    return {"base": base, "refined": refined}


def call_explicit(solver, derivative_fun, instance, x0, theta, target, args, tolerance):
    options = backward_options(args, 1.0)
    options.enable = False
    forward = solver.solve_lapanda(
        x0,
        theta,
        target,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(args),
        alm_options=alm_options(args),
        backward_options=options,
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    )
    solution = np.asarray(forward["solution"], dtype=float).reshape(-1)
    multipliers = np.maximum(
        np.asarray(forward["multipliers"], dtype=float).reshape(-1), 0.0
    )
    backward = explicit_kkt_minres_backward(
        derivative_fun,
        instance,
        solution,
        theta,
        target,
        multipliers,
        args.alm_tol,
        1e-7,
        tolerance,
        args.backward_max_iter,
    )
    return {"forward": forward, "backward": backward}


def run_worker(method: str, n: int, trial: int) -> dict:
    args = experiment_args()
    theta = theta_nominal()
    target = sinusoidal_target(n)
    x0 = initial_point(n, 100 + trial)

    def construct():
        instance = build_problem(n, constraint_stride=1)
        if method == "casadi":
            sensitivity = build_casadi_sensitivity(
                n, 1, 1e-3, 5000, f"matched_memory_casadi_n{n}"
            )
            forward_solver = build_reference_ipopt_solver(
                instance, 1e-3, 5000, f"matched_memory_ipopt_n{n}"
            )
            return instance, forward_solver, sensitivity
        solver = build_solver(
            instance.problem,
            backend="compiled",
            name=f"matched_memory_n{n}",
            cache_dir=Path(tempfile.gettempdir())
            / "lapanda_experiments"
            / "matched_2pct_memory",
            force=False,
        )
        derivative_fun = None
        if method == "explicit_kkt":
            derivative_fun = build_kkt_derivative_function(
                instance, f"matched_memory_kkt_n{n}"
            )
        return instance, solver, derivative_fun

    constructed = sampled_call(construct, sample_interval_sec=0.0001)
    instance, solver_or_forward, derivative_or_sensitivity = constructed["value"]

    if method == "lapanda":
        measured = sampled_call(
            lambda: call_lapanda(
                solver_or_forward,
                instance,
                x0,
                theta,
                target,
                args,
                LAPANDA_SCALE[n],
            ),
            sample_interval_sec=0.0001,
        )
        result = measured["value"]
        forward_time = float(
            result["base"]["forward_time_sec"]
            + result["refined"]["forward_time_sec"]
        )
        backward_time = float(result["refined"]["backward_time_sec"])
        iterations = int(result["refined"]["backward_iterations"])
        residual = float(result["refined"]["backward_residual"])
        native_backward_workspace_bytes = int(
            result["refined"]["backward_peak_workspace_bytes"]
        )
        solver_used = str(result["refined"].get("backward_solver_used", ""))
        fallback_used = bool(result["refined"].get("backward_fallback_used", False))
        explicit_kkt_storage_bytes = 0
        setting = LAPANDA_SCALE[n]
    elif method == "explicit_kkt":
        measured = sampled_call(
            lambda: call_explicit(
                solver_or_forward,
                derivative_or_sensitivity,
                instance,
                x0,
                theta,
                target,
                args,
                EXPLICIT_TOLERANCE[n],
            ),
            sample_interval_sec=0.0001,
        )
        result = measured["value"]
        forward_time = float(result["forward"]["forward_time_sec"])
        backward_time = float(result["backward"]["total_time_sec"])
        iterations = int(result["backward"]["iterations"])
        residual = float(result["backward"]["residual"])
        native_backward_workspace_bytes = int(
            result["backward"]["minres_peak_workspace_bytes"]
        )
        explicit_kkt_storage_bytes = int(result["backward"]["kkt_storage_bytes"])
        solver_used = "minres"
        fallback_used = False
        setting = EXPLICIT_TOLERANCE[n]
    else:
        def solve_casadi():
            with suppress_native_output(True):
                start = time.perf_counter()
                forward = solver_or_forward(
                    x0=x0,
                    p=theta,
                    lbx=instance.lower,
                    ubx=instance.upper,
                    lbg=instance.constraint_lower,
                    ubg=instance.constraint_upper,
                )
                forward_time = time.perf_counter() - start
            solution = np.asarray(forward["x"], dtype=float).reshape(-1)
            start = time.perf_counter()
            evaluate_sensitivity(
                derivative_or_sensitivity, theta, target, solution
            )
            backward_time = time.perf_counter() - start
            return forward_time, backward_time

        measured = sampled_call(solve_casadi, sample_interval_sec=0.0001)
        forward_time, backward_time = measured["value"]
        iterations = -1
        residual = float("nan")
        native_backward_workspace_bytes = 0
        explicit_kkt_storage_bytes = 0
        solver_used = ""
        fallback_used = False
        setting = 1e-3

    return {
        "method": (
            "lapanda"
            if method == "lapanda"
            else "Explicit KKT (dense)"
            if method == "explicit_kkt"
            else "CasADi"
        ),
        "n": n,
        "trial_id": trial,
        "setting": setting,
        "inner_tolerance": args.inner_tol,
        "alm_tolerance": args.alm_tol,
        "backward_tolerance": args.backward_tol,
        "linear_solver": "CG" if method == "lapanda" else "MINRES",
        "backward_solver_used": solver_used,
        "backward_fallback_used": fallback_used,
        "backward_max_iterations": args.backward_max_iter,
        "build_rss_before_mb": constructed["rss_before_mb"],
        "build_rss_peak_mb": constructed["rss_peak_mb"],
        "build_rss_after_mb": constructed["rss_after_mb"],
        "build_rss_peak_delta_mb": constructed["rss_peak_delta_mb"],
        "rss_before_mb": measured["rss_before_mb"],
        "rss_peak_mb": measured["rss_peak_mb"],
        "rss_after_mb": measured["rss_after_mb"],
        "solve_rss_peak_delta_mb": measured["rss_peak_delta_mb"],
        "rss_samples": measured["rss_samples"],
        "solve_wall_time_sec": measured["elapsed_sec"],
        "forward_time_sec": forward_time,
        "backward_time_sec": backward_time,
        "backward_iterations": iterations,
        "backward_residual": residual,
        "native_backward_workspace_mb": native_backward_workspace_bytes
        / (1024.0**2),
        "explicit_kkt_storage_mb": explicit_kkt_storage_bytes / (1024.0**2),
        "backward_core_peak_mb": (
            float("nan")
            if method == "casadi"
            else (
                native_backward_workspace_bytes + explicit_kkt_storage_bytes
            )
            / (1024.0**2)
        ),
        "total_memory_mb": constructed["rss_peak_delta_mb"]
        + measured["rss_peak_delta_mb"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for method in ("lapanda", "Explicit KKT (dense)", "CasADi"):
        for n in sorted({int(row["n"]) for row in rows}):
            group = [row for row in rows if row["method"] == method and row["n"] == n]
            if not group:
                continue
            memory = np.asarray(
                [row["solve_rss_peak_delta_mb"] for row in group], dtype=float
            )
            output.append(
                {
                    "method": method,
                    "n": n,
                    "trials": len(group),
                    "solve_rss_peak_delta_mb_mean": float(np.mean(memory)),
                    "solve_rss_peak_delta_mb_std": float(np.std(memory, ddof=1))
                    if len(memory) > 1
                    else 0.0,
                    "solve_rss_peak_delta_mb_max": float(np.max(memory)),
                    "build_rss_peak_delta_mb_mean": float(
                        np.mean([row["build_rss_peak_delta_mb"] for row in group])
                    ),
                    "total_memory_mb_mean": float(
                        np.mean([row["total_memory_mb"] for row in group])
                    ),
                    "backward_core_peak_mb_mean": float(
                        np.mean([row["backward_core_peak_mb"] for row in group])
                    ),
                    "forward_time_sec_mean": float(
                        np.mean([row["forward_time_sec"] for row in group])
                    ),
                    "backward_time_sec_mean": float(
                        np.mean([row["backward_time_sec"] for row in group])
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500, 1000])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lapanda", "explicit_kkt", "casadi"],
        default=["lapanda", "explicit_kkt", "casadi"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "matched_accuracy"
        / "memory",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=["lapanda", "explicit_kkt", "casadi"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--worker-output", type=Path)
    parsed = parser.parse_args()

    if parsed.worker:
        row = run_worker(parsed.method, parsed.n, parsed.trial)
        parsed.worker_output.parent.mkdir(parents=True, exist_ok=True)
        parsed.worker_output.write_text(json.dumps(row), encoding="utf-8")
        return

    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lapanda_memory_workers_") as worker_dir:
        worker_root = Path(worker_dir)
        for n in parsed.sizes:
            for trial in range(parsed.trials):
                for method in parsed.methods:
                    worker_output = worker_root / f"worker_{method}_n{n}_trial{trial}.json"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--method",
                        method,
                        "--n",
                        str(n),
                        "--trial",
                        str(trial),
                        "--worker-output",
                        str(worker_output),
                    ]
                    print(f"measuring {method} n={n} trial={trial}", flush=True)
                    subprocess.run(command, check=True)
                    row = json.loads(worker_output.read_text(encoding="utf-8"))
                    rows.append(row)
                    print(
                        f"  solve RSS peak delta={row['solve_rss_peak_delta_mb']:.3f} MB, "
                        f"samples={row['rss_samples']}",
                        flush=True,
                    )

    write_csv(parsed.output_dir / "solve_memory_raw.csv", rows)
    summaries = summarize(rows)
    write_csv(parsed.output_dir / "solve_memory_summary.csv", summaries)
    (parsed.output_dir / "config.json").write_text(
        json.dumps(
            {
                "measurement": "separately sampled RSS peak increments during construction and the first complete solve",
                "construction_included": True,
                "methods_run_in_fresh_processes": True,
                "inner_tolerance": 1e-3,
                "alm_tolerance": 1e-3,
                "lapanda_inner_warm_start": False,
                "backward_tolerance": 1e-2,
                "lapanda_linear_solver": "CG with MINRES fallback",
                "explicit_kkt_linear_solver": "MINRES",
                "backward_max_iterations": 200,
                "sizes": parsed.sizes,
                "trials": parsed.trials,
                "methods": parsed.methods,
                "lapanda_refinement": "aligned final-subproblem refinement",
                "lapanda_penalty_scale_by_size": LAPANDA_SCALE,
                "explicit_kkt_tolerance_by_size": EXPLICIT_TOLERANCE,
                "casadi_tolerance": 1e-3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
