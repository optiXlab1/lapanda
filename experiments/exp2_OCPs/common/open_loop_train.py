"""Open-loop fixed-expert-archive imitation training."""

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
    MODEL_PREFIX,
    OPEN_LOOP_LEARNING_RATE,
    TRAIN_SOLVER_DEFAULTS,
    model_prefix,
    results_dir,
)
from experiments.exp2_OCPs.utils import (
    clip_theta,
    default_teacher_archive_path,
    load_snapshot_archive,
    make_variable_from_state,
    seeded_theta_initial,
)


def add_open_loop_arguments(parser: argparse.ArgumentParser, model: str) -> None:
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm_defaults = ALM_DEFAULTS[model]
    backward_defaults = BACKWARD_DEFAULTS[model]
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=OPEN_LOOP_LEARNING_RATE[model])
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
        "--backward-linear-solver",
        choices=["auto", "cg", "minres", "gmres"],
        default=backward_defaults.linear_solver,
    )
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
    parser.add_argument("--theta-init", choices=["default", "uniform01"], default="uniform01")
    parser.add_argument("--output-tag", type=str, default="fixed32")
    parser.add_argument("--teacher-source", type=str, default="")
    parser.add_argument("--teacher-samples", type=int, default=32)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory, then stop.")


def output_path(model: str, tag: str | None) -> Path:
    path = results_dir(model) / f"{model_prefix(model)}_open_loop_train_lapanda.csv"
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    return fresh_output_path(path)


def _teacher_path(model: str, args) -> Path:
    if args.teacher_source:
        return Path(args.teacher_source)
    return default_teacher_archive_path(model, args.horizon, args.teacher_samples, args.teacher_seed)


def run_open_loop_train(model: str, args) -> Path:
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    problem_build = profiled_call(lambda: build_instance(model, args.horizon), measure_memory=measure_memory, sample_tree=True)
    instance = problem_build["value"]
    solver_build = profiled_call(
        lambda: build_lapanda_solver(
            instance,
            model,
            name=f"{MODEL_PREFIX[model]}_open_loop_train_h{args.horizon}",
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
            f"{model} open-loop build "
            f"problem_mem={problem_build['rss_tree_peak_delta_mb']:.3f}MB "
            f"solver_mem={solver_build['rss_tree_peak_delta_mb']:.3f}MB "
            f"build_mem={build_mem_mb:.3f}MB "
            f"time={build_time_sec:.4f}s"
        )
    else:
        print(f"{model} open-loop build time={build_time_sec:.4f}s")
    teacher_path = _teacher_path(model, args)
    teacher_snapshots = load_snapshot_archive(teacher_path)
    if not teacher_snapshots:
        raise FileNotFoundError(f"missing teacher rollout snapshots: {teacher_path}")
    if memory_only:
        teacher_snapshots = teacher_snapshots[:1]
    effective_epochs = 0 if memory_only else args.epochs

    out_path = output_path(model, args.output_tag)
    config_payload = {
            "model": model,
            "instance": instance.name,
            "horizon": args.horizon,
            "teacher_source": str(teacher_path),
            "teacher_samples": len(teacher_snapshots),
            "theta_true": instance.theta_true.tolist(),
            "theta_initial": instance.theta_initial.tolist(),
            "problem_build_time_sec": problem_build["elapsed_sec"],
            "solver_build_time_sec": solver_build["elapsed_sec"],
            "build_time_sec": build_time_sec,
            "measure_memory": measure_memory,
            "memory_only": memory_only,
            "alm_initial_penalty": args.alm_initial_penalty,
            "alm_penalty_update_factor": args.alm_penalty_update_factor,
            "alm_max_penalty": args.alm_max_penalty,
            "backward_tolerance": args.alm_tol,
            "backward_max_iterations": args.backward_max_iterations,
            "backward_linear_solver": args.backward_linear_solver,
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
    u_warm_by_sample = [
        np.asarray(snapshot["controls"], dtype=float).reshape(-1).copy()
        for snapshot in teacher_snapshots
    ]
    multiplier_warm_by_sample = [None for _ in teacher_snapshots]

    for epoch in range(effective_epochs + 1):
        sample_grads = []
        sample_losses = []
        sample_violations = []
        sample_forward_times = []
        sample_backward_times = []
        sample_rss = []
        sample_outer_iters = []
        sample_inner_iters = []
        sample_backward_iters = []
        sample_residuals = []
        sample_backward_solvers = []
        sample_backward_fallbacks = []

        for sample_index, snapshot in enumerate(teacher_snapshots):
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            state = demo_states[0].reshape(instance.nx)
            variable = make_variable_from_state(instance, state, demo_u, demo_states)
            stats = profiled_call(
                lambda: solve_lapanda(
                    solver,
                    instance,
                    u_warm_by_sample[sample_index],
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
                    multiplier0=multiplier_warm_by_sample[sample_index],
                    penalty0=None,
                ),
                measure_memory=measure_memory,
            )
            result = stats["value"]
            solution = np.asarray(result["solution"], dtype=float).reshape(-1)
            u_warm_by_sample[sample_index] = solution.copy()
            multiplier_warm_by_sample[sample_index] = np.asarray(result["multipliers"], dtype=float).copy()
            grad = np.asarray(result["grad_theta"], dtype=float).reshape(-1)
            sample_grads.append(grad)
            sample_losses.append(float(instance.outer_loss_fun(solution, theta, variable)))
            sample_violations.append(violation_inf(instance, solution, theta, variable))
            sample_forward_times.append(float(result.get("forward_time_sec", np.nan)))
            sample_backward_times.append(float(result.get("backward_time_sec", np.nan)))
            if measure_memory:
                sample_rss.append(float(stats["rss_peak_delta_mb"]))
            sample_outer_iters.append(float(result["iterations"]))
            sample_inner_iters.append(float(np.sum(np.asarray(result["inner_iterations"], dtype=int))))
            sample_backward_iters.append(float(result.get("backward_iterations", -1)))
            sample_residuals.append(float(result["final_residual"]))
            sample_backward_solvers.append(str(result.get("backward_solver_used", "unknown")))
            sample_backward_fallbacks.append(bool(result.get("backward_fallback_used", False)))

        grad = np.mean(np.asarray(sample_grads, dtype=float), axis=0)
        loss = float(np.mean(sample_losses))
        loss_std = float(np.std(sample_losses, ddof=1)) if len(sample_losses) > 1 else 0.0
        row = {
            "epoch": epoch,
            "seed": args.seed,
            "loss": loss,
            "loss_mean": loss,
            "loss_std": loss_std,
            "loss_sem": loss_std / np.sqrt(len(sample_losses)) if sample_losses else 0.0,
            "theta": theta.tolist(),
            "grad_theta": grad.tolist(),
            "grad_norm": float(np.linalg.norm(grad)),
            "wall_time_sec": float(np.nansum(sample_forward_times) + np.nansum(sample_backward_times)),
            "forward_time_sec": float(np.nanmean(sample_forward_times)),
            "backward_time_sec": float(np.nanmean(sample_backward_times)),
            "warm_start": True,
            "penalty_warm_start": False,
            "constraint_violation_inf": float(np.mean(sample_violations)),
            "outer_iterations": float(np.mean(sample_outer_iters)),
            "inner_iterations_total": float(np.mean(sample_inner_iters)),
            "backward_iterations": float(np.mean(sample_backward_iters)),
            "backward_solver_used": "/".join(sorted(set(sample_backward_solvers))),
            "backward_fallback_count": int(sum(sample_backward_fallbacks)),
            "alm_residual_inf": float(np.mean(sample_residuals)),
            "teacher_rollout_samples": len(teacher_snapshots),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = float(np.mean(sample_rss))
        append_csv(out_path, [row])
        msg = (
            f"{model} open-loop epoch={epoch:03d} loss={loss:.6e} "
            f"fwd={row['forward_time_sec']:.4f}s bwd={row['backward_time_sec']:.4f}s "
        )
        if measure_memory:
            msg += f"mem={row['rss_peak_delta_mb']:.3f}MB "
        msg += f"grad_norm={row['grad_norm']:.3e}"
        print(msg)
        if epoch == effective_epochs:
            break
        theta = clip_theta(instance, theta - args.learning_rate * grad / (np.linalg.norm(grad) + 1e-9))

    print(f"wrote {out_path}")
    return out_path


def main(model: str) -> None:
    parser = argparse.ArgumentParser()
    add_open_loop_arguments(parser, model)
    args = parser.parse_args()
    run_open_loop_train(model, args)
