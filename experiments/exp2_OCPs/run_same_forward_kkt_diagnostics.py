"""Compare lapanda and original-KKT sensitivities at identical OCP forward points."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from experiments.exp2_OCPs.config import ALM_DEFAULTS, BACKWARD_DEFAULTS, TRAIN_SOLVER_DEFAULTS
from experiments.exp2_OCPs.run_gradient_checkpoint_diagnostics import (
    closed_loop_teacher_path,
    closed_loop_training_path,
    read_training_rows,
)
from experiments.exp2_OCPs.run_scenario_statistics import (
    cosine_similarity,
    final_training_path,
    relative_error,
    teacher_archive_path,
    write_csv,
)
from experiments.exp2_OCPs.utils import (
    build_derivatives,
    build_instance,
    build_lapanda_solver,
    first_next_state,
    load_snapshot_archive,
    make_variable_from_state,
    shift_control,
    solve_lapanda,
    violation_inf,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "summary_results"
MODELS = ("cartpole", "quadrotor", "robot_arm")


def independent_rows(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=int), 0
    _, r, pivots = sla.qr(matrix.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.zeros(0, dtype=int), 0
    rank = int(np.count_nonzero(diagonal > tolerance * diagonal[0]))
    return np.asarray(pivots[:rank], dtype=int), rank


def same_forward_kkt_reference(
    instance,
    derivatives,
    result,
    theta: np.ndarray,
    variable: np.ndarray,
    *,
    active_tolerance: float,
    rank_tolerance: float,
) -> dict:
    u = np.asarray(result["solution"], dtype=float).reshape(-1)
    multipliers = np.asarray(result["multipliers"], dtype=float).reshape(-1)
    penalties = np.asarray(result["penalties"], dtype=float).reshape(-1)
    constraints = np.asarray(instance.constraint_fun(u, theta, variable), dtype=float).reshape(-1)
    lower_c = np.asarray(instance.constraint_lower, dtype=float).reshape(-1)
    upper_c = np.asarray(instance.constraint_upper, dtype=float).reshape(-1)

    shifted = constraints + multipliers / penalties
    projected = np.minimum(np.maximum(shifted, lower_c), upper_c)
    shifted_multipliers = penalties * (shifted - projected)
    equality = np.abs(upper_c - lower_c) <= active_tolerance
    active_c = (
        equality
        | (shifted <= lower_c + active_tolerance)
        | (shifted >= upper_c - active_tolerance)
    )

    lower_u = np.asarray(instance.problem.box_lower, dtype=float).reshape(-1)
    upper_u = np.asarray(instance.problem.box_upper, dtype=float).reshape(-1)
    active_lower = u <= lower_u + active_tolerance
    active_upper = u >= upper_u - active_tolerance

    jac_u = np.asarray(derivatives["jac_c_u"](u, theta, variable), dtype=float)
    jac_theta = np.asarray(derivatives["jac_c_theta"](u, theta, variable), dtype=float)
    zero_box_multiplier = np.zeros_like(u)
    hessian = np.asarray(
        derivatives["hess"](
            u,
            theta,
            variable,
            shifted_multipliers,
            zero_box_multiplier,
            zero_box_multiplier,
        ),
        dtype=float,
    )
    lagrangian_u_theta = np.asarray(
        derivatives["hutheta"](
            u,
            theta,
            variable,
            shifted_multipliers,
            zero_box_multiplier,
            zero_box_multiplier,
        ),
        dtype=float,
    )
    _, loss_u_raw, loss_theta_raw = derivatives["loss_grad"](u, theta, variable)
    loss_u = np.asarray(loss_u_raw, dtype=float).reshape(-1)
    loss_theta = np.asarray(loss_theta_raw, dtype=float).reshape(-1)

    rows_u = [jac_u[active_c, :]]
    rows_theta = [jac_theta[active_c, :]]
    for indices, sign in (
        (np.flatnonzero(active_lower), -1.0),
        (np.flatnonzero(active_upper), 1.0),
    ):
        if indices.size:
            block = np.zeros((indices.size, u.size), dtype=float)
            block[np.arange(indices.size), indices] = sign
            rows_u.append(block)
            rows_theta.append(np.zeros((indices.size, theta.size), dtype=float))

    active_jacobian_full = np.vstack(rows_u)
    active_theta_full = np.vstack(rows_theta)
    selected, rank = independent_rows(active_jacobian_full, rank_tolerance)
    active_jacobian = active_jacobian_full[selected, :]
    active_theta = active_theta_full[selected, :]
    active_count = active_jacobian.shape[0]
    kkt = np.block(
        [
            [hessian, active_jacobian.T],
            [active_jacobian, np.zeros((active_count, active_count), dtype=float)],
        ]
    )
    rhs = np.concatenate([loss_u, np.zeros(active_count)])
    adjoint = np.linalg.solve(kkt.T, rhs)
    for _ in range(2):
        adjoint += np.linalg.solve(kkt.T, rhs - kkt.T @ adjoint)
    residual = float(
        np.linalg.norm(kkt.T @ adjoint - rhs) / max(np.linalg.norm(rhs), 1e-16)
    )
    gradient = (
        loss_theta
        - lagrangian_u_theta.T @ adjoint[: u.size]
        - active_theta.T @ adjoint[u.size :]
    )
    return {
        "gradient": np.asarray(gradient, dtype=float).reshape(-1),
        "linear_residual": residual,
        "active_rows": int(active_jacobian_full.shape[0]),
        "independent_active_rows": int(rank),
        "dropped_active_rows": int(active_jacobian_full.shape[0] - rank),
        "condition": float(np.linalg.cond(kkt)),
    }


def solve_sample(
    model,
    instance,
    solver,
    derivatives,
    u0,
    theta,
    variable,
    args,
    *,
    multiplier0=None,
):
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm = ALM_DEFAULTS[model]
    backward = BACKWARD_DEFAULTS[model]
    result = solve_lapanda(
        solver,
        instance,
        u0,
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
        multiplier0=multiplier0,
    )
    reference = same_forward_kkt_reference(
        instance,
        derivatives,
        result,
        theta,
        variable,
        active_tolerance=args.active_tolerance,
        rank_tolerance=args.rank_tolerance,
    )
    if reference["linear_residual"] > args.reference_residual_tolerance:
        raise RuntimeError(
            f"same-forward KKT residual {reference['linear_residual']:.3e} exceeds "
            f"{args.reference_residual_tolerance:.3e}"
        )
    return result, reference


def summarize(protocol: str, model: str, epoch: int, rows: list[dict]) -> dict:
    gradients = np.asarray([json.loads(row["gradient"]) for row in rows], dtype=float)
    references = np.asarray([json.loads(row["reference_gradient"]) for row in rows], dtype=float)
    gradient = np.mean(gradients, axis=0)
    reference = np.mean(references, axis=0)
    sample_errors = np.asarray(
        [relative_error(value, target) for value, target in zip(gradients, references)],
        dtype=float,
    )
    sample_cosines = np.asarray(
        [cosine_similarity(value, target) for value, target in zip(gradients, references)],
        dtype=float,
    )
    return {
        "protocol": protocol,
        "task": model,
        "epoch": epoch,
        "samples": len(rows),
        "aggregate_relative_error": relative_error(gradient, reference),
        "aggregate_cosine_similarity": cosine_similarity(gradient, reference),
        "sample_relative_error_mean": float(np.mean(sample_errors)),
        "sample_relative_error_median": float(np.median(sample_errors)),
        "sample_relative_error_max": float(np.max(sample_errors)),
        "sample_cosine_similarity_mean": float(np.mean(sample_cosines)),
        "sample_cosine_similarity_min": float(np.min(sample_cosines)),
        "reference_residual_max": float(max(row["reference_residual"] for row in rows)),
        "kkt_condition_median": float(np.median([row["kkt_condition"] for row in rows])),
        "kkt_condition_max": float(max(row["kkt_condition"] for row in rows)),
        "dropped_active_rows_total": int(sum(row["dropped_active_rows"] for row in rows)),
        "violation_max": float(max(row["violation"] for row in rows)),
    }


def evaluate_open_loop(model: str, checkpoints: list[int], args):
    instance = build_instance(model, args.horizon)
    training = read_training_rows(final_training_path(model))
    snapshots = load_snapshot_archive(
        teacher_archive_path(model, args.horizon, args.teacher_seed)
    )[: args.open_scenarios]
    solver = build_lapanda_solver(
        instance,
        model,
        f"exp2_{model}_same_forward_open",
        TRAIN_SOLVER_DEFAULTS[model].backend,
        args.force,
    )
    derivatives = build_derivatives(instance)
    raw = []
    summaries = []
    for epoch in checkpoints:
        theta = np.asarray(ast.literal_eval(training[epoch]["theta"]), dtype=float)
        epoch_rows = []
        for sample, snapshot in enumerate(snapshots):
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            variable = make_variable_from_state(instance, demo_states[0], demo_u, demo_states)
            result, reference = solve_sample(
                model, instance, solver, derivatives, demo_u, theta, variable, args
            )
            solution = np.asarray(result["solution"], dtype=float).reshape(-1)
            row = {
                "protocol": "open_loop",
                "task": model,
                "epoch": epoch,
                "sample": sample,
                "gradient": json.dumps(np.asarray(result["grad_theta"], dtype=float).tolist()),
                "reference_gradient": json.dumps(reference["gradient"].tolist()),
                "reference_residual": reference["linear_residual"],
                "kkt_condition": reference["condition"],
                "dropped_active_rows": reference["dropped_active_rows"],
                "violation": violation_inf(instance, solution, theta, variable),
            }
            raw.append(row)
            epoch_rows.append(row)
        item = summarize("open_loop", model, epoch, epoch_rows)
        summaries.append(item)
        print(
            f"open_loop {model} epoch={epoch}: "
            f"err={100 * item['aggregate_relative_error']:.3f}% "
            f"cos={item['aggregate_cosine_similarity']:.8f}"
        )
    return raw, summaries


def evaluate_closed_loop(model: str, checkpoints: list[int], args):
    instance = build_instance(model, args.horizon)
    training = read_training_rows(closed_loop_training_path(model))
    snapshots = load_snapshot_archive(closed_loop_teacher_path(model))[: args.closed_steps]
    solver = build_lapanda_solver(
        instance,
        model,
        f"exp2_{model}_same_forward_closed",
        TRAIN_SOLVER_DEFAULTS[model].backend,
        args.force,
    )
    derivatives = build_derivatives(instance)
    initial_state = np.asarray(snapshots[0]["states"], dtype=float)[0].reshape(instance.nx)
    raw = []
    summaries = []
    for epoch in checkpoints:
        theta = np.asarray(ast.literal_eval(training[epoch]["theta"]), dtype=float)
        state = initial_state.copy()
        u_warm = np.asarray(snapshots[0]["controls"], dtype=float).reshape(-1).copy()
        multiplier = None
        epoch_rows = []
        for step, snapshot in enumerate(snapshots):
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            variable = make_variable_from_state(instance, state, demo_u, demo_states)
            result, reference = solve_sample(
                model,
                instance,
                solver,
                derivatives,
                u_warm,
                theta,
                variable,
                args,
                multiplier0=multiplier,
            )
            solution = np.asarray(result["solution"], dtype=float).reshape(-1)
            row = {
                "protocol": "closed_loop",
                "task": model,
                "epoch": epoch,
                "sample": step,
                "gradient": json.dumps(np.asarray(result["grad_theta"], dtype=float).tolist()),
                "reference_gradient": json.dumps(reference["gradient"].tolist()),
                "reference_residual": reference["linear_residual"],
                "kkt_condition": reference["condition"],
                "dropped_active_rows": reference["dropped_active_rows"],
                "violation": violation_inf(instance, solution, theta, variable),
            }
            raw.append(row)
            epoch_rows.append(row)
            state = first_next_state(instance, solution, theta, variable)
            u_warm = shift_control(solution, instance.nu)
            multiplier = np.asarray(result["multipliers"], dtype=float).copy()
        item = summarize("closed_loop", model, epoch, epoch_rows)
        summaries.append(item)
        print(
            f"closed_loop {model} epoch={epoch}: "
            f"err={100 * item['aggregate_relative_error']:.3f}% "
            f"cos={item['aggregate_cosine_similarity']:.8f}"
        )
    return raw, summaries


def parse_epochs(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("open_loop", "closed_loop"),
        default=["open_loop", "closed_loop"],
    )
    parser.add_argument("--open-epochs", default="0,200,400,800")
    parser.add_argument("--closed-epochs", default="0,125,250,500")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--open-scenarios", type=int, default=32)
    parser.add_argument("--closed-steps", type=int, default=50)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--active-tolerance", type=float, default=1e-8)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    parser.add_argument("--reference-residual-tolerance", type=float, default=1e-9)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = []
    summaries = []
    for model in args.models:
        if "open_loop" in args.protocols:
            model_raw, model_summary = evaluate_open_loop(
                model, parse_epochs(args.open_epochs), args
            )
            raw.extend(model_raw)
            summaries.extend(model_summary)
        if "closed_loop" in args.protocols:
            model_raw, model_summary = evaluate_closed_loop(
                model, parse_epochs(args.closed_epochs), args
            )
            raw.extend(model_raw)
            summaries.extend(model_summary)

    write_csv(RESULTS / "ocp_same_forward_kkt_raw.csv", raw)
    write_csv(RESULTS / "ocp_same_forward_kkt_summary.csv", summaries)
    config = {
        "experiment": "exp2_same_forward_explicit_kkt_accuracy",
        "models": args.models,
        "protocols": args.protocols,
        "open_epochs": parse_epochs(args.open_epochs),
        "closed_epochs": parse_epochs(args.closed_epochs),
        "open_scenarios": args.open_scenarios,
        "closed_steps": args.closed_steps,
        "active_tolerance": args.active_tolerance,
        "rank_tolerance": args.rank_tolerance,
        "reference_residual_tolerance": args.reference_residual_tolerance,
        "reference": (
            "original-NLP active-set KKT adjoint at the unchanged lapanda "
            "forward point, using the forward-converged shifted multiplier"
        ),
    }
    (RESULTS / "ocp_same_forward_kkt_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
