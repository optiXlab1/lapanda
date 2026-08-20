"""Benchmark warm-started MPC rollouts across OCP horizons."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.exp2_OCPs.config import ALM_DEFAULTS, BACKWARD_DEFAULTS, TRAIN_SOLVER_DEFAULTS
from experiments.exp2_OCPs.utils import (
    build_lapanda_solver,
    first_next_state,
    make_variable_from_state,
    shift_control,
    solve_lapanda,
    violation_inf,
)
from experiments.exp2_OCPs.problem_bank import build_quadrotor_problem


DEFAULT_HORIZONS = tuple(range(10, 201, 10))
DEFAULT_DATA_DIR = (
    ROOT
    / "experiments"
    / "exp2_OCPs"
    / "summary_results"
    / "nonlinear_horizon_scaling_mixed_warm_start"
    / "raw"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prediction-time", type=float, default=2.5)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--inner-max-iter", type=int, default=None)
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument("--backward-max-iterations", type=int, default=None)
    parser.add_argument("--output-tag", default="lapanda_mixed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def feedback_rollout(horizon: int, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    state = np.array([-1.0, 0.65, 0.0, 0.0, 0.15, 0.0])
    states = [state.copy()]
    controls = []
    for _ in range(horizon):
        px, altitude, vx, vz, angle, omega = state
        desired_ax = np.clip(0.8 * (0.75 - px) - 2.5 * vx, -2.0, 2.0)
        desired_angle = np.clip(-np.arcsin(np.clip(desired_ax / 9.81, -0.25, 0.25)), -0.28, 0.28)
        torque = np.clip((12.0 * (desired_angle - angle) - 5.0 * omega) / 4.0, -3.0, 3.0)
        desired_az = np.clip(1.5 * (1.10 - altitude) - 1.8 * vz, -3.0, 3.0)
        thrust = np.clip((9.81 + desired_az) / max(np.cos(angle), 0.8), 0.0, 2.2 * 9.81)
        controls.append([thrust, torque])

        ax = -thrust * np.sin(angle)
        az = thrust * np.cos(angle) - 9.81
        omega_next = omega + dt * 4.0 * torque
        angle_next = angle + dt * omega_next
        vx_next = vx + dt * ax
        vz_next = vz + dt * az
        state = np.array(
            [
                px + dt * vx_next,
                altitude + dt * vz_next,
                vx_next,
                vz_next,
                angle_next,
                omega_next,
            ]
        )
        states.append(state.copy())
    return np.asarray(states), np.asarray(controls)


def run_horizon(
    horizon: int,
    steps: int,
    prediction_time: float,
    data_dir: Path,
    inner_max_iter: int | None,
    max_outer: int | None,
    backward_max_iterations: int | None,
    output_tag: str,
    force: bool,
) -> Path:
    model = "quadrotor"
    solver_defaults = TRAIN_SOLVER_DEFAULTS[model]
    alm_defaults = ALM_DEFAULTS[model]
    backward_defaults = BACKWARD_DEFAULTS[model]
    dt = prediction_time / horizon
    instance = build_quadrotor_problem(horizon=horizon, dt=dt)
    solver = build_lapanda_solver(
        instance,
        model,
        name=f"exp2_quadrotor_nonlinear_scaling_h{horizon}",
        backend=solver_defaults.backend,
        force=force,
    )
    demo_states, demo_controls = feedback_rollout(horizon, dt)
    theta = np.asarray(instance.theta_initial, dtype=float)
    state = demo_states[0].copy()
    solution_warm = demo_controls.reshape(-1).copy()
    multiplier_warm = None
    penalty_warm = None
    rows = []

    initial_variable = make_variable_from_state(instance, state, np.zeros_like(solution_warm))
    initial_violation = violation_inf(instance, solution_warm, theta, initial_variable)
    if initial_violation > 1e-10:
        raise RuntimeError(f"feedback warm start is infeasible at N={horizon}: {initial_violation}")

    for step in range(steps):
        variable = make_variable_from_state(instance, state, np.zeros_like(solution_warm))
        result = solve_lapanda(
            solver,
            instance,
            solution_warm,
            theta,
            variable,
            inner_max_iter=inner_max_iter or solver_defaults.inner_max_iter,
            inner_tol=solver_defaults.inner_tol,
            max_outer=max_outer or solver_defaults.max_outer,
            alm_tol=solver_defaults.alm_tol,
            compute_backward=True,
            alm_max_penalty=alm_defaults.max_penalty,
            alm_initial_penalty=alm_defaults.initial_penalty,
            alm_penalty_update_factor=alm_defaults.penalty_update_factor,
            backward_max_iterations=(
                backward_max_iterations or backward_defaults.max_iterations
            ),
            backward_constraint_penalty_scale=backward_defaults.constraint_penalty_scale,
            backward_constraint_penalty_max=backward_defaults.constraint_penalty_max,
            multiplier0=multiplier_warm,
            penalty0=penalty_warm,
        )
        solution_warm = np.asarray(result["solution"], dtype=float).reshape(-1)
        multiplier_warm = np.asarray(result["multipliers"], dtype=float).reshape(-1)
        penalty_warm = np.asarray(result["penalties"], dtype=float).reshape(-1)
        forward_ms = 1000.0 * float(result["forward_time_sec"])
        backward_ms = 1000.0 * float(result["backward_time_sec"])
        row = {
            "step": step,
            "initial_state": state.tolist(),
            "horizon": horizon,
            "dt": dt,
            "forward_time_ms": forward_ms,
            "backward_time_ms": backward_ms,
            "value_grad_time_ms": forward_ms + backward_ms,
            "constraint_violation_inf": violation_inf(instance, solution_warm, theta, variable),
            "outer_iterations": int(result["iterations"]),
            "inner_iterations": int(np.sum(np.asarray(result["inner_iterations"], dtype=int))),
            "backward_iterations": int(result.get("backward_iterations", -1)),
            "final_residual": float(result["final_residual"]),
        }
        rows.append(row)
        print(
            f"lapanda MPC N={horizon:3d} step={step:02d} "
            f"fwd={forward_ms:8.3f}ms bwd={backward_ms:7.3f}ms "
            f"viol={row['constraint_violation_inf']:.3e}"
        )

        state = first_next_state(instance, solution_warm, theta, variable)
        solution_warm = shift_control(solution_warm, instance.nu)

    path = data_dir / f"quadrotor_h{horizon}_{output_tag}_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
    np.savez(
        data_dir / f"quadrotor_feedback_h{horizon}.npz",
        snapshots=np.asarray(
            [{"initial_state": demo_states[0], "states": demo_states, "controls": demo_controls}],
            dtype=object,
        ),
    )
    return path


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        run_horizon(
            horizon,
            args.steps,
            args.prediction_time,
            args.data_dir,
            args.inner_max_iter,
            args.max_outer,
            args.backward_max_iterations,
            args.output_tag,
            args.force,
        )


if __name__ == "__main__":
    main()
