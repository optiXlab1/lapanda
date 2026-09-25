"""Track lapanda sensitivity accuracy over open- and closed-loop training.

Open-loop metrics compare the batch gradient over all 32 fixed scenarios.
Closed-loop metrics compare the gradient averaged over the complete 50-step
rollout generated at each selected training checkpoint.  Reference gradients
come from a high-accuracy IPOPT solution and the original active-set KKT
sensitivity system.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp2_OCPs.config import ALM_DEFAULTS, BACKWARD_DEFAULTS, TRAIN_SOLVER_DEFAULTS
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
    build_ipopt_solver,
    build_lapanda_solver,
    first_next_state,
    kkt_sensitivity,
    load_snapshot_archive,
    make_variable_from_state,
    shift_control,
    solve_ipopt,
    solve_lapanda,
    violation_inf,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "summary_results"
MODELS = ("cartpole", "quadrotor", "robot_arm")


def read_training_rows(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(float(row["epoch"])): row for row in rows}


def closed_loop_training_path(model: str) -> Path:
    prefix = f"exp2_{model}"
    return ROOT / model / "results" / f"{prefix}_closed_loop_train_lapanda_closed_loop.csv"


def closed_loop_teacher_path(model: str) -> Path:
    training_path = closed_loop_training_path(model)
    config_path = training_path.with_name(f"config_{training_path.stem}.json")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return Path(config["teacher_source"])


def evaluate_sample(
    model,
    instance,
    lapanda_solver,
    ipopt_solver,
    derivatives,
    u0,
    theta,
    variable,
    *,
    multiplier0=None,
    penalty0=None,
    active_tolerance: float,
):
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm = ALM_DEFAULTS[model]
    backward = BACKWARD_DEFAULTS[model]
    result = solve_lapanda(
        lapanda_solver,
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
        penalty0=penalty0,
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
        active_tolerance,
    )
    return result, solution, gradient, np.asarray(reference["grad_theta"], dtype=float).reshape(-1)


def summarize(protocol: str, model: str, epoch: int, rows: list[dict]) -> dict:
    gradients = np.asarray([json.loads(row["gradient"]) for row in rows], dtype=float)
    references = np.asarray([json.loads(row["reference_gradient"]) for row in rows], dtype=float)
    gradient = np.mean(gradients, axis=0)
    reference = np.mean(references, axis=0)
    losses = np.asarray([row["loss"] for row in rows], dtype=float)
    violations = np.asarray([row["violation"] for row in rows], dtype=float)
    return {
        "protocol": protocol,
        "task": model,
        "epoch": epoch,
        "samples": len(rows),
        "loss_mean": float(np.mean(losses)),
        "loss_max": float(np.max(losses)),
        "violation_mean": float(np.mean(violations)),
        "violation_max": float(np.max(violations)),
        "gradient_relative_error": relative_error(gradient, reference),
        "gradient_cosine_similarity": cosine_similarity(gradient, reference),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "reference_gradient_norm": float(np.linalg.norm(reference)),
    }


def evaluate_open_loop(model: str, checkpoints: list[int], args) -> tuple[list[dict], list[dict]]:
    instance = build_instance(model, args.horizon)
    training = read_training_rows(final_training_path(model))
    snapshots = load_snapshot_archive(teacher_archive_path(model, args.horizon, args.teacher_seed))
    snapshots = snapshots[: args.open_scenarios]
    lapanda_solver = build_lapanda_solver(
        instance, model, f"exp2_{model}_open_gradient_checkpoints", TRAIN_SOLVER_DEFAULTS[model].backend, args.force
    )
    ipopt_solver = build_ipopt_solver(instance, args.reference_tolerance, args.ipopt_max_iter)
    derivatives = build_derivatives(instance)
    raw = []
    summary = []
    for epoch in checkpoints:
        theta = np.asarray(ast.literal_eval(training[epoch]["theta"]), dtype=float)
        epoch_rows = []
        for sample, snapshot in enumerate(snapshots):
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            variable = make_variable_from_state(instance, demo_states[0], demo_u, demo_states)
            _, solution, gradient, reference = evaluate_sample(
                model,
                instance,
                lapanda_solver,
                ipopt_solver,
                derivatives,
                demo_u,
                theta,
                variable,
                active_tolerance=args.active_tolerance,
            )
            row = {
                "protocol": "open_loop",
                "task": model,
                "epoch": epoch,
                "sample": sample,
                "loss": float(instance.outer_loss_fun(solution, theta, variable)),
                "violation": violation_inf(instance, solution, theta, variable),
                "gradient": json.dumps(gradient.tolist()),
                "reference_gradient": json.dumps(reference.tolist()),
            }
            raw.append(row)
            epoch_rows.append(row)
        item = summarize("open_loop", model, epoch, epoch_rows)
        summary.append(item)
        print(
            f"open_loop {model} epoch={epoch}: "
            f"err={100 * item['gradient_relative_error']:.3f}% "
            f"cos={item['gradient_cosine_similarity']:.8f}"
        )
    return raw, summary


def evaluate_closed_loop(model: str, checkpoints: list[int], args) -> tuple[list[dict], list[dict]]:
    instance = build_instance(model, args.horizon)
    training = read_training_rows(closed_loop_training_path(model))
    snapshots = load_snapshot_archive(closed_loop_teacher_path(model))[: args.closed_steps]
    lapanda_solver = build_lapanda_solver(
        instance, model, f"exp2_{model}_closed_gradient_checkpoints", TRAIN_SOLVER_DEFAULTS[model].backend, args.force
    )
    ipopt_solver = build_ipopt_solver(instance, args.reference_tolerance, args.ipopt_max_iter)
    derivatives = build_derivatives(instance)
    initial_state = np.asarray(snapshots[0]["states"], dtype=float)[0].reshape(instance.nx)
    raw = []
    summary = []
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
            result, solution, gradient, reference = evaluate_sample(
                model,
                instance,
                lapanda_solver,
                ipopt_solver,
                derivatives,
                u_warm,
                theta,
                variable,
                multiplier0=multiplier,
                active_tolerance=args.active_tolerance,
            )
            row = {
                "protocol": "closed_loop",
                "task": model,
                "epoch": epoch,
                "sample": step,
                "loss": float(instance.outer_loss_fun(solution, theta, variable)),
                "violation": violation_inf(instance, solution, theta, variable),
                "gradient": json.dumps(gradient.tolist()),
                "reference_gradient": json.dumps(reference.tolist()),
            }
            raw.append(row)
            epoch_rows.append(row)
            state = first_next_state(instance, solution, theta, variable)
            u_warm = shift_control(solution, instance.nu)
            multiplier = np.asarray(result["multipliers"], dtype=float).copy()
        item = summarize("closed_loop", model, epoch, epoch_rows)
        summary.append(item)
        print(
            f"closed_loop {model} epoch={epoch}: "
            f"err={100 * item['gradient_relative_error']:.3f}% "
            f"cos={item['gradient_cosine_similarity']:.8f}"
        )
    return raw, summary


def parse_epochs(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--protocols", nargs="+", choices=("open_loop", "closed_loop"), default=["open_loop", "closed_loop"])
    parser.add_argument("--open-epochs", default="0,200,400,800")
    parser.add_argument("--closed-epochs", default="0,125,250,500")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--open-scenarios", type=int, default=32)
    parser.add_argument("--closed-steps", type=int, default=50)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--reference-tolerance", type=float, default=1e-10)
    parser.add_argument("--active-tolerance", type=float, default=1e-6)
    parser.add_argument("--ipopt-max-iter", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = []
    summary = []
    for model in args.models:
        if "open_loop" in args.protocols:
            model_raw, model_summary = evaluate_open_loop(model, parse_epochs(args.open_epochs), args)
            raw.extend(model_raw)
            summary.extend(model_summary)
        if "closed_loop" in args.protocols:
            model_raw, model_summary = evaluate_closed_loop(model, parse_epochs(args.closed_epochs), args)
            raw.extend(model_raw)
            summary.extend(model_summary)

    write_csv(RESULTS / "ocp_gradient_checkpoint_raw.csv", raw)
    write_csv(RESULTS / "ocp_gradient_checkpoint_summary.csv", summary)
    print(f"wrote {RESULTS / 'ocp_gradient_checkpoint_summary.csv'}")


if __name__ == "__main__":
    main()
