"""Evaluate scenario-level robustness and backward sensitivity accuracy.

The script reuses the 32 fixed demonstration scenarios and the final learned
parameter vector from each open-loop experiment.  It reports per-scenario
loss, constraint violation, runtime, relative gradient error, and gradient
cosine similarity.  The reference gradient is obtained from a high-accuracy
IPOPT solution followed by the active-set KKT sensitivity solve already used
by the experiment utilities.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp2_OCPs.config import ALM_DEFAULTS, BACKWARD_DEFAULTS, TRAIN_SOLVER_DEFAULTS
from experiments.exp2_OCPs.utils import (
    build_derivatives,
    build_instance,
    build_ipopt_solver,
    build_lapanda_solver,
    default_teacher_archive_path,
    kkt_sensitivity,
    load_snapshot_archive,
    make_variable_from_state,
    solve_ipopt,
    solve_lapanda,
    violation_inf,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "summary_results"
MODEL_PREFIX = {
    "cartpole": "exp2_cartpole",
    "quadrotor": "exp2_quadrotor",
    "robot_arm": "exp2_robot_arm",
}


def final_training_path(model: str) -> Path:
    prefix = MODEL_PREFIX[model]
    suffix = "feasible_v2" if model == "robot_arm" else "fixed32"
    return ROOT / model / "results" / f"{prefix}_open_loop_train_lapanda_{suffix}.csv"


def teacher_archive_path(model: str, horizon: int, teacher_seed: int) -> Path:
    path = default_teacher_archive_path(model, horizon, 32, teacher_seed)
    if model == "robot_arm":
        path = path.with_name(path.name.replace(".mpc_snapshots.npz", "_feasible_v2.mpc_snapshots.npz"))
    return path


def read_final_theta(model: str) -> np.ndarray:
    path = final_training_path(model)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no training rows found in {path}")
    return np.asarray(ast.literal_eval(rows[-1]["theta"]), dtype=float)


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom == 0.0:
        return 1.0 if np.linalg.norm(lhs - rhs) == 0.0 else 0.0
    return float(np.dot(lhs, rhs) / denom)


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    return float(np.linalg.norm(value - reference) / max(denom, np.finfo(float).eps))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(model: str, rows: list[dict]) -> dict:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    gradient = np.mean(
        np.asarray([json.loads(row["gradient"]) for row in rows], dtype=float), axis=0
    )
    reference_gradient = np.mean(
        np.asarray([json.loads(row["reference_gradient"]) for row in rows], dtype=float), axis=0
    )
    return {
        "task": model,
        "scenarios": len(rows),
        "success_rate": float(np.mean(values("success"))),
        "loss_mean": float(np.mean(values("loss"))),
        "loss_max": float(np.max(values("loss"))),
        "violation_mean": float(np.mean(values("constraint_violation_inf"))),
        "violation_max": float(np.max(values("constraint_violation_inf"))),
        "gradient_relative_error": relative_error(gradient, reference_gradient),
        "gradient_cosine_similarity": cosine_similarity(gradient, reference_gradient),
        "forward_time_ms_mean": float(1e3 * np.mean(values("forward_time_sec"))),
        "forward_time_ms_max": float(1e3 * np.max(values("forward_time_sec"))),
        "backward_time_ms_mean": float(1e3 * np.mean(values("backward_time_sec"))),
        "backward_time_ms_max": float(1e3 * np.max(values("backward_time_sec"))),
    }


def evaluate_model(model: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    instance = build_instance(model, args.horizon)
    theta = read_final_theta(model)
    snapshots = load_snapshot_archive(
        teacher_archive_path(model, args.horizon, args.teacher_seed)
    )[: args.scenarios]
    if len(snapshots) != args.scenarios:
        raise RuntimeError(f"{model}: expected {args.scenarios} scenarios, found {len(snapshots)}")

    defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm = ALM_DEFAULTS[model]
    backward = BACKWARD_DEFAULTS[model]
    lapanda_solver = build_lapanda_solver(
        instance,
        model,
        name=f"exp2_{model}_scenario_statistics",
        backend=defaults.backend,
        force=args.force,
    )
    ipopt_solver = build_ipopt_solver(instance, args.reference_tolerance, args.ipopt_max_iter)
    derivatives = build_derivatives(instance)

    rows: list[dict] = []
    for index, snapshot in enumerate(snapshots):
        demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
        demo_states = np.asarray(snapshot["states"], dtype=float)
        state = demo_states[0].reshape(instance.nx)
        variable = make_variable_from_state(instance, state, demo_u, demo_states)

        result = solve_lapanda(
            lapanda_solver,
            instance,
            demo_u,
            theta,
            variable,
            inner_max_iter=defaults.inner_max_iter,
            inner_tol=defaults.inner_tol,
            max_outer=defaults.max_outer,
            alm_tol=defaults.alm_tol,
            compute_backward=True,
            alm_max_penalty=alm.max_penalty,
            alm_initial_penalty=alm.initial_penalty,
            alm_penalty_update_factor=alm.penalty_update_factor,
            backward_max_iterations=backward.max_iterations,
            backward_linear_solver=backward.linear_solver,
            backward_constraint_penalty_scale=backward.constraint_penalty_scale,
            backward_constraint_penalty_max=backward.constraint_penalty_max,
        )
        solution = np.asarray(result["solution"], dtype=float).reshape(-1)
        gradient = np.asarray(result["grad_theta"], dtype=float).reshape(-1)

        reference_result = solve_ipopt(instance, ipopt_solver, solution, theta, variable)
        reference = kkt_sensitivity(
            instance,
            derivatives,
            reference_result,
            theta,
            variable,
            args.active_tolerance,
        )
        reference_gradient = np.asarray(reference["grad_theta"], dtype=float).reshape(-1)
        loss = float(instance.outer_loss_fun(solution, theta, variable))
        row = {
            "task": model,
            "scenario": index,
            "success": int(np.all(np.isfinite(solution)) and np.all(np.isfinite(gradient))),
            "loss": loss,
            "constraint_violation_inf": violation_inf(instance, solution, theta, variable),
            "gradient_relative_error": relative_error(gradient, reference_gradient),
            "gradient_cosine_similarity": cosine_similarity(gradient, reference_gradient),
            "gradient": json.dumps(gradient.tolist()),
            "reference_gradient": json.dumps(reference_gradient.tolist()),
            "forward_time_sec": float(result["forward_time_sec"]),
            "backward_time_sec": float(result["backward_time_sec"]),
            "reference_kkt_condition": float(reference["kkt_cond"]),
        }
        rows.append(row)
        print(
            f"{model} {index + 1:02d}/{len(snapshots)} "
            f"loss={loss:.3e} err={row['gradient_relative_error']:.3e} "
            f"cos={row['gradient_cosine_similarity']:.6f}"
        )

    return rows, aggregate(model, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PREFIX), default=list(MODEL_PREFIX))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--reference-tolerance", type=float, default=1e-10)
    parser.add_argument("--active-tolerance", type=float, default=1e-6)
    parser.add_argument("--ipopt-max-iter", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model in args.models:
        rows, summary = evaluate_model(model, args)
        raw_rows.extend(rows)
        summary_rows.append(summary)

    write_csv(RESULTS / "ocp_scenario_statistics_raw.csv", raw_rows)
    write_csv(RESULTS / "ocp_scenario_statistics_summary.csv", summary_rows)
    print(f"wrote {RESULTS / 'ocp_scenario_statistics_summary.csv'}")


if __name__ == "__main__":
    main()
