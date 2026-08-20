"""Closed-loop MPC rollout / teacher generation for OCP examples."""

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
    MPC_SOLVER_DEFAULTS,
    model_prefix,
    results_dir,
)
from experiments.exp2_OCPs.utils import apply_theta_overrides, first_next_state, make_variable_from_state, shift_control


def add_mpc_arguments(parser: argparse.ArgumentParser, model: str) -> None:
    defaults = MPC_SOLVER_DEFAULTS[model]
    alm_defaults = ALM_DEFAULTS[model]
    backward_defaults = BACKWARD_DEFAULTS[model]
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
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
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--theta-true-override", type=str, default="")
    parser.add_argument("--theta-initial-override", type=str, default="")
    parser.add_argument(
        "--compute-backward",
        action="store_true",
        help="Also run the lapanda backward pass during rollout.",
    )
    parser.add_argument(
        "--no-penalty-warm-start",
        action="store_true",
        help="Warm-start controls and multipliers but reset ALM penalties at each MPC step.",
    )
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory, then stop.")


def output_path(model: str, tag: str | None) -> Path:
    path = results_dir(model) / f"{model_prefix(model)}_mpc_lapanda.csv"
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    return fresh_output_path(path)


def run_mpc_rollout(model: str, args) -> Path:
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    instance = build_instance(model, args.horizon)
    apply_theta_overrides(instance, args.theta_true_override, args.theta_initial_override)
    solver = build_lapanda_solver(
        instance,
        model,
        name=f"{MODEL_PREFIX[model]}_mpc_h{args.horizon}",
        backend=args.backend,
        force=args.force,
    )

    out_path = output_path(model, args.output_tag)
    write_json(
        out_path.parent / f"config_{out_path.stem}.json",
        {
            "model": model,
            "instance": instance.name,
            "horizon": args.horizon,
            "steps": args.steps,
            "theta_true": instance.theta_true.tolist(),
            "alm_initial_penalty": args.alm_initial_penalty,
            "alm_penalty_update_factor": args.alm_penalty_update_factor,
            "alm_max_penalty": args.alm_max_penalty,
            "backward_tolerance": args.alm_tol,
            "backward_max_iterations": args.backward_max_iterations,
            "backward_constraint_penalty_scale": args.backward_constraint_penalty_scale,
            "backward_constraint_penalty_max": args.backward_constraint_penalty_max,
            "measure_memory": measure_memory,
            "memory_only": memory_only,
            "solver": vars(args),
        },
    )

    state = instance.x_initial.copy()
    u_warm = np.zeros(instance.horizon * instance.nu)
    multiplier = None
    penalty = None
    trajectory = [state.copy()]
    snapshots = []

    effective_steps = 1 if memory_only else args.steps
    for step in range(effective_steps):
        variable = make_variable_from_state(instance, state, np.zeros(instance.horizon * instance.nu))
        try:
            stats = profiled_call(
                lambda: solve_lapanda(
                    solver,
                    instance,
                    u_warm,
                    instance.theta_true,
                    variable,
                    inner_max_iter=args.inner_max_iter,
                    inner_tol=args.inner_tol,
                    max_outer=args.max_outer,
                    alm_tol=args.alm_tol,
                    compute_backward=args.compute_backward,
                    alm_max_penalty=args.alm_max_penalty,
                    alm_initial_penalty=args.alm_initial_penalty,
                    alm_penalty_update_factor=args.alm_penalty_update_factor,
                    backward_max_iterations=args.backward_max_iterations,
                    backward_constraint_penalty_scale=args.backward_constraint_penalty_scale,
                    backward_constraint_penalty_max=args.backward_constraint_penalty_max,
                    multiplier0=multiplier,
                    penalty0=None if args.no_penalty_warm_start else penalty,
                ),
                measure_memory=measure_memory,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{model} ALM-MPC failed at step={step}, "
                f"compute_backward={bool(args.compute_backward)}, "
                f"state={state.tolist()}"
            ) from exc
        result = stats["value"]
        solution = np.asarray(result["solution"], dtype=float).reshape(-1)
        planned_states = np.asarray(instance.rollout_fun(solution, instance.theta_true, variable), dtype=float)
        violation = violation_inf(instance, solution, instance.theta_true, variable)
        row = {
            "step": step,
            "state": state.tolist(),
            "first_control": solution[: instance.nu].tolist(),
            "objective": float(instance.cost_fun(solution, instance.theta_true, variable)),
            "constraint_violation_inf": violation,
            "wall_time_sec": stats["elapsed_sec"],
            "forward_time_sec": float(result.get("forward_time_sec", np.nan)),
            "backward_time_sec": float(result.get("backward_time_sec", np.nan)),
            "compute_backward": bool(args.compute_backward),
            "outer_iterations": int(result["iterations"]),
            "inner_iterations_total": int(np.sum(np.asarray(result["inner_iterations"], dtype=int))),
            "backward_iterations": int(result.get("backward_iterations", -1)),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = stats["rss_peak_delta_mb"]
        append_csv(out_path, [row])
        snapshots.append(
            {
                "step": int(step),
                "states": planned_states.copy(),
                "controls": solution.copy(),
                "state": state.copy(),
                "constraint_violation_inf": float(violation),
            }
        )
        state = first_next_state(instance, solution, instance.theta_true, variable)
        trajectory.append(state.copy())
        u_warm = shift_control(solution, instance.nu)
        multiplier = np.asarray(result["multipliers"], dtype=float).copy()
        penalty = np.asarray(result["penalties"], dtype=float).copy()
        print(
            f"{model} ALM-MPC step={step:02d} "
            f"fwd={row['forward_time_sec']:.4f}s bwd={row['backward_time_sec']:.4f}s "
            f"outer={row['outer_iterations']} viol={violation:.2e}"
        )

    np.save(out_path.with_suffix(".trajectory.npy"), np.asarray(trajectory))
    np.savez(out_path.with_suffix(".mpc_snapshots.npz"), snapshots=np.asarray(snapshots, dtype=object), allow_pickle=True)
    print(f"wrote {out_path}")
    return out_path


def main(model: str) -> None:
    parser = argparse.ArgumentParser()
    add_mpc_arguments(parser, model)
    args = parser.parse_args()
    run_mpc_rollout(model, args)
