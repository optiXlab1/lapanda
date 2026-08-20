"""Closed-loop rollout imitation training.

Each epoch starts from the teacher rollout's initial state.  At each MPC step,
the learner solves an OCP with the current learner state and imitates the
teacher's planned control sequence at the matching step.  The applied first
control advances the learner state, so the training loss is measured on a
closed-loop rollout rather than independent open-loop samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import append_csv, fresh_output_path, profiled_call, write_json
from experiments.exp2_OCPs.utils import (
    build_lapanda_solver,
    build_instance,
    solve_lapanda,
    violation_inf,
)
from experiments.exp2_OCPs.config import (
    ALM_DEFAULTS,
    BACKWARD_DEFAULTS,
    CLOSED_LOOP_LEARNING_RATE,
    MODEL_PREFIX,
    TRAIN_SOLVER_DEFAULTS,
    legacy_mpc_snapshot_path,
    model_prefix,
    results_dir,
)
from experiments.exp2_OCPs.utils import (
    apply_theta_overrides,
    clip_theta,
    first_next_state,
    load_snapshot_archive,
    make_variable_from_state,
    seeded_theta_initial,
    shift_control,
)


def add_closed_loop_arguments(parser: argparse.ArgumentParser, model: str) -> None:
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm_defaults = ALM_DEFAULTS[model]
    backward_defaults = BACKWARD_DEFAULTS[model]
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=CLOSED_LOOP_LEARNING_RATE[model])
    parser.add_argument("--backend", choices=["compiled", "callback"], default=defaults.backend)
    parser.add_argument("--inner-max-iter", type=int, default=defaults.inner_max_iter)
    parser.add_argument("--inner-tol", type=float, default=defaults.inner_tol)
    parser.add_argument("--max-outer", type=int, default=defaults.max_outer)
    parser.add_argument("--alm-tol", type=float, default=defaults.alm_tol)
    parser.add_argument("--alm-initial-penalty", type=float, default=alm_defaults.initial_penalty)
    parser.add_argument("--alm-penalty-update-factor", type=float, default=alm_defaults.penalty_update_factor)
    parser.add_argument("--alm-max-penalty", type=float, default=alm_defaults.max_penalty)
    parser.add_argument("--backward-max-iterations", type=int, default=backward_defaults.max_iterations)
    parser.add_argument(
        "--backward-constraint-penalty-scale",
        type=float,
        default=backward_defaults.constraint_penalty_scale,
    )
    parser.add_argument(
        "--backward-constraint-penalty-max",
        type=float,
        default=backward_defaults.constraint_penalty_max,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--theta-init", choices=["default", "uniform01"], default="default")
    parser.add_argument(
        "--learn-theta-indices",
        type=str,
        default="",
        help="Comma-separated theta indices to update; empty updates all parameters.",
    )
    parser.add_argument("--output-tag", type=str, default="closed_loop")
    parser.add_argument("--teacher-source", type=str, default="")
    parser.add_argument("--theta-true-override", type=str, default="")
    parser.add_argument("--theta-initial-override", type=str, default="")
    parser.add_argument(
        "--warm-start-penalty",
        action="store_true",
        help="Also inherit ALM penalties between closed-loop MPC steps.",
    )
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory, then stop.")


def output_path(model: str, tag: str | None) -> Path:
    path = results_dir(model) / f"{model_prefix(model)}_closed_loop_train_lapanda.csv"
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    return fresh_output_path(path)


def _teacher_path(model: str, args) -> Path:
    if args.teacher_source:
        return Path(args.teacher_source)
    return legacy_mpc_snapshot_path(model)


def run_closed_loop_train(model: str, args) -> Path:
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    problem_build = profiled_call(
        lambda: apply_theta_overrides(
            build_instance(model, args.horizon),
            args.theta_true_override,
            args.theta_initial_override,
        ),
        measure_memory=measure_memory,
        sample_tree=True,
    )
    instance = problem_build["value"]
    solver_build = profiled_call(
        lambda: build_lapanda_solver(
            instance,
            model,
            name=f"{MODEL_PREFIX[model]}_closed_loop_train_h{args.horizon}",
            backend=args.backend,
            force=args.force,
        ),
        measure_memory=measure_memory,
        sample_tree=True,
    )
    solver = solver_build["value"]
    build_time_sec = problem_build["elapsed_sec"] + solver_build["elapsed_sec"]
    build_mem_mb = problem_build["rss_tree_peak_delta_mb"] + solver_build["rss_tree_peak_delta_mb"]
    if measure_memory:
        print(
            f"{model} closed-loop build "
            f"problem_mem={problem_build['rss_tree_peak_delta_mb']:.3f}MB "
            f"solver_mem={solver_build['rss_tree_peak_delta_mb']:.3f}MB "
            f"build_mem={build_mem_mb:.3f}MB "
            f"time={build_time_sec:.4f}s"
        )
    else:
        print(f"{model} closed-loop build time={build_time_sec:.4f}s")
    teacher_path = _teacher_path(model, args)
    teacher_snapshots = load_snapshot_archive(teacher_path)
    if not teacher_snapshots:
        raise FileNotFoundError(f"missing teacher MPC snapshots: {teacher_path}")
    teacher_snapshots = teacher_snapshots[: args.steps]
    if memory_only:
        teacher_snapshots = teacher_snapshots[:1]
    effective_epochs = 0 if memory_only else args.epochs

    out_path = output_path(model, args.output_tag)
    config_payload = {
            "model": model,
            "instance": instance.name,
            "horizon": args.horizon,
            "steps": len(teacher_snapshots),
            "teacher_source": str(teacher_path),
            "theta_true": instance.theta_true.tolist(),
            "theta_initial": instance.theta_initial.tolist(),
            "problem_build_time_sec": problem_build["elapsed_sec"],
            "solver_build_time_sec": solver_build["elapsed_sec"],
            "build_time_sec": build_time_sec,
            "measure_memory": measure_memory,
            "memory_only": memory_only,
            "learn_theta_indices": args.learn_theta_indices,
            "alm_initial_penalty": args.alm_initial_penalty,
            "alm_penalty_update_factor": args.alm_penalty_update_factor,
            "alm_max_penalty": args.alm_max_penalty,
            "backward_tolerance": args.alm_tol,
            "backward_max_iterations": args.backward_max_iterations,
            "backward_constraint_penalty_scale": args.backward_constraint_penalty_scale,
            "backward_constraint_penalty_max": args.backward_constraint_penalty_max,
            "solver": vars(args),
    }
    if measure_memory:
        config_payload.update(
            {
                "problem_build_rss_peak_delta_mb": problem_build["rss_peak_delta_mb"],
                "problem_build_rss_tree_peak_delta_mb": problem_build["rss_tree_peak_delta_mb"],
                "solver_build_rss_peak_delta_mb": solver_build["rss_peak_delta_mb"],
                "solver_build_rss_tree_peak_delta_mb": solver_build["rss_tree_peak_delta_mb"],
                "build_rss_delta_mb": problem_build["rss_peak_delta_mb"] + solver_build["rss_peak_delta_mb"],
                "build_rss_tree_delta_mb": build_mem_mb,
            }
        )
    write_json(out_path.parent / f"config_{out_path.stem}.json", config_payload)

    theta = seeded_theta_initial(instance, args.theta_init, args.seed)
    learn_mask = np.ones_like(theta)
    if args.learn_theta_indices:
        learn_mask[:] = 0.0
        indices = [int(item.strip()) for item in args.learn_theta_indices.split(",") if item.strip()]
        for index in indices:
            if index < 0 or index >= theta.size:
                raise ValueError(f"--learn-theta-indices index {index} out of range for theta size {theta.size}")
            learn_mask[index] = 1.0
    first_teacher_states = np.asarray(teacher_snapshots[0]["states"], dtype=float)
    initial_state = first_teacher_states[0].reshape(instance.nx)

    for epoch in range(effective_epochs + 1):
        state = initial_state.copy()
        u_warm = np.asarray(teacher_snapshots[0]["controls"], dtype=float).reshape(-1).copy()
        multiplier = None
        penalty = None
        trajectory = [state.copy()]
        sample_grads = []
        sample_losses = []
        sample_violations = []
        forward_times = []
        backward_times = []
        outer_iters = []
        inner_iters = []
        backward_iters = []
        residuals = []
        rss = []

        for step, snapshot in enumerate(teacher_snapshots):
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            variable = make_variable_from_state(instance, state, demo_u, demo_states)
            try:
                stats = profiled_call(
                    lambda: solve_lapanda(
                        solver,
                        instance,
                        u_warm,
                        theta,
                        variable,
                        inner_max_iter=args.inner_max_iter,
                        inner_tol=args.inner_tol,
                        max_outer=args.max_outer,
                        alm_tol=args.alm_tol,
                        compute_backward=True,
                        alm_max_penalty=args.alm_max_penalty,
                        alm_initial_penalty=args.alm_initial_penalty,
                        alm_penalty_update_factor=args.alm_penalty_update_factor,
                        backward_max_iterations=args.backward_max_iterations,
                        backward_constraint_penalty_scale=args.backward_constraint_penalty_scale,
                        backward_constraint_penalty_max=args.backward_constraint_penalty_max,
                        multiplier0=multiplier,
                        penalty0=penalty if args.warm_start_penalty else None,
                    ),
                    measure_memory=measure_memory,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{model} closed-loop backward/solve failed at "
                    f"epoch={epoch}, step={step}, "
                    f"theta={theta.tolist()}, state={state.tolist()}, "
                    f"warm_start_penalty={bool(args.warm_start_penalty)}"
                ) from exc
            result = stats["value"]
            solution = np.asarray(result["solution"], dtype=float).reshape(-1)
            grad = np.asarray(result["grad_theta"], dtype=float).reshape(-1)
            sample_grads.append(grad)
            sample_losses.append(float(instance.outer_loss_fun(solution, theta, variable)))
            sample_violations.append(violation_inf(instance, solution, theta, variable))
            forward_times.append(float(result.get("forward_time_sec", np.nan)))
            backward_times.append(float(result.get("backward_time_sec", np.nan)))
            outer_iters.append(float(result["iterations"]))
            inner_iters.append(float(np.sum(np.asarray(result["inner_iterations"], dtype=int))))
            backward_iters.append(float(result.get("backward_iterations", -1)))
            residuals.append(float(result["final_residual"]))
            if measure_memory:
                rss.append(float(stats["rss_peak_delta_mb"]))

            state = first_next_state(instance, solution, theta, variable)
            trajectory.append(state.copy())
            u_warm = shift_control(solution, instance.nu)
            multiplier = np.asarray(result["multipliers"], dtype=float).copy()
            penalty = np.asarray(result["penalties"], dtype=float).copy()

        grad = np.mean(np.asarray(sample_grads, dtype=float), axis=0)
        loss = float(np.mean(sample_losses))
        row = {
            "epoch": epoch,
            "seed": args.seed,
            "loss": loss,
            "loss_mean": loss,
            "loss_std": float(np.std(sample_losses, ddof=1)) if len(sample_losses) > 1 else 0.0,
            "theta": theta.tolist(),
            "grad_theta": grad.tolist(),
            "grad_norm": float(np.linalg.norm(grad)),
            "wall_time_sec": float(np.nansum(forward_times) + np.nansum(backward_times)),
            "forward_time_sec": float(np.nanmean(forward_times)),
            "backward_time_sec": float(np.nanmean(backward_times)),
            "warm_start": True,
            "penalty_warm_start": bool(args.warm_start_penalty),
            "constraint_violation_inf": float(np.mean(sample_violations)),
            "outer_iterations": float(np.mean(outer_iters)),
            "inner_iterations_total": float(np.mean(inner_iters)),
            "backward_iterations": float(np.mean(backward_iters)),
            "alm_residual_inf": float(np.mean(residuals)),
            "closed_loop_steps": len(teacher_snapshots),
            "final_state": trajectory[-1].tolist(),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = float(np.mean(rss))
        append_csv(out_path, [row])
        msg = (
            f"{model} closed-loop epoch={epoch:03d} loss={loss:.6e} "
            f"fwd={row['forward_time_sec']:.4f}s bwd={row['backward_time_sec']:.4f}s "
        )
        if measure_memory:
            msg += f"mem={row['rss_peak_delta_mb']:.3f}MB "
        msg += f"grad_norm={row['grad_norm']:.3e}"
        print(msg)
        if epoch == effective_epochs:
            break
        update_grad = grad * learn_mask
        theta = clip_theta(instance, theta - args.learning_rate * update_grad / (np.linalg.norm(update_grad) + 1e-9))

    print(f"wrote {out_path}")
    return out_path


def main(model: str) -> None:
    parser = argparse.ArgumentParser()
    add_closed_loop_arguments(parser, model)
    args = parser.parse_args()
    run_closed_loop_train(model, args)


