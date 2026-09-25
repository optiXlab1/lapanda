"""Run repeated CasADi timings for the matched-accuracy Table 2 benchmark."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import git_commit_hash, machine_info, relative_error, write_json
from experiments.exp1_rosenbrock_smooth_constraints.casadi_sensitivity import (
    build_casadi_sensitivity,
    evaluate_sensitivity,
    suppress_native_output,
)
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    constraint_violation_inf,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)


def build_ipopt_solver(instance, tolerance: float, max_iterations: int, name: str):
    return ca.nlpsol(
        name,
        "ipopt",
        {
            "x": instance.problem.u,
            "p": instance.problem.theta,
            "f": instance.problem.cost,
            "g": instance.problem.constraints,
        },
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": tolerance,
            "ipopt.acceptable_tol": tolerance,
            "ipopt.constr_viol_tol": tolerance,
            "ipopt.dual_inf_tol": tolerance,
            "ipopt.compl_inf_tol": tolerance,
            "ipopt.max_iter": max_iterations,
        },
    )


def read_references(path: Path) -> dict[tuple[int, int], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["n"]), int(row["trial_id"])): np.asarray(
                ast.literal_eval(row["gradient"]), dtype=float
            )
            for row in csv.DictReader(stream)
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for n in sorted({int(row["n"]) for row in rows}):
        group = [row for row in rows if int(row["n"]) == n]
        summary = {"n": n, "trials": len({int(row["trial_id"]) for row in group})}
        summary["repetitions"] = len({int(row["repetition"]) for row in group})
        for field in (
            "gradient_relative_error",
            "forward_time_sec",
            "backward_time_sec",
            "constraint_violation_inf",
        ):
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(np.std(values, ddof=1))
            summary[f"{field}_max"] = float(np.max(values))
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500, 1000])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "matched_accuracy"
        / "reference.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "matched_accuracy"
        / "casadi",
    )
    args = parser.parse_args()

    references = read_references(args.reference_csv)
    theta = theta_nominal()
    rows: list[dict] = []

    for n in args.sizes:
        instance = build_problem(n, constraint_stride=args.constraint_stride)
        target = sinusoidal_target(n)
        suffix = str(args.tolerance).replace(".", "p").replace("-", "m")
        forward_solver = build_ipopt_solver(
            instance, args.tolerance, args.max_iterations, f"table2_ipopt_n{n}_{suffix}"
        )
        sensitivity_fun = build_casadi_sensitivity(
            n,
            args.constraint_stride,
            args.tolerance,
            args.max_iterations,
            f"table2_casadi_n{n}_{suffix}",
        )

        # Match the other methods: one untimed warm-up at each problem size.
        warm_x0 = initial_point(n, 9999)
        with suppress_native_output(True):
            warm_forward = forward_solver(
                x0=warm_x0,
                p=theta,
                lbx=instance.lower,
                ubx=instance.upper,
                lbg=instance.constraint_lower,
                ubg=instance.constraint_upper,
            )
        evaluate_sensitivity(
            sensitivity_fun,
            theta,
            target,
            np.asarray(warm_forward["x"], dtype=float).reshape(-1),
        )

        for repetition in range(args.repetitions):
            for trial_id in range(args.trial_start, args.trial_start + args.trials):
                x0 = initial_point(n, 100 + trial_id)
                start = time.perf_counter()
                with suppress_native_output(True):
                    forward = forward_solver(
                        x0=x0,
                        p=theta,
                        lbx=instance.lower,
                        ubx=instance.upper,
                        lbg=instance.constraint_lower,
                        ubg=instance.constraint_upper,
                    )
                forward_time = time.perf_counter() - start
                x_ipopt = np.asarray(forward["x"], dtype=float).reshape(-1)

                start = time.perf_counter()
                solution_raw, _, gradient_raw = evaluate_sensitivity(
                    sensitivity_fun, theta, target, x_ipopt
                )
                backward_time = time.perf_counter() - start
                solution = np.asarray(solution_raw, dtype=float).reshape(-1)
                gradient = np.asarray(gradient_raw, dtype=float).reshape(-1)
                rows.append(
                    {
                        "n": n,
                        "repetition": repetition,
                        "trial_id": trial_id,
                        "gradient_relative_error": relative_error(
                            gradient, references[(n, trial_id)]
                        ),
                        "forward_time_sec": forward_time,
                        "backward_time_sec": backward_time,
                        "constraint_violation_inf": constraint_violation_inf(
                            instance, solution, theta
                        ),
                    }
                )
        print(f"completed n={n}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "summary.csv", summarize(rows))
    reference_path = args.reference_csv.resolve()
    try:
        reference_record = str(reference_path.relative_to(REPO_ROOT))
    except ValueError:
        reference_record = str(reference_path)
    write_json(
        args.output_dir / "config.json",
        {
            "experiment": "exp1_matched_accuracy_casadi_repeated",
            "sizes": args.sizes,
            "trials": args.trials,
            "repetitions": args.repetitions,
            "trial_start": args.trial_start,
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
            "warm_up": "one untimed forward and backward call per problem size",
            "reference_csv": reference_record,
            "git_commit": git_commit_hash(),
            "machine": machine_info(),
        },
    )


if __name__ == "__main__":
    main()
