"""Benchmark lapanda on the constrained linear MPC scaling problem."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


DEFAULT_DATA_DIR = (
    ROOT
    / "experiments"
    / "exp2_OCPs"
    / "summary_results"
    / "horizon_scaling"
    / "raw"
)
DEFAULT_HORIZONS = (2, 5, 10, 20, 40, 80, 160)
NX = 8
NU = 4
UMAX = 10.0
TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def problem_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    a_matrix = np.eye(NX) + 0.1 * np.random.randn(NX, NX)
    eigenvalues, eigenvectors = np.linalg.eig(a_matrix)
    for index, value in enumerate(eigenvalues):
        if abs(value) >= 1.0 - 1e-2:
            eigenvalues[index] = value / (abs(value) + 1e-2)
    a_matrix = (eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)).real
    b_matrix = np.random.randn(NX, NU)
    offset = 0.01 * np.random.randn(NX)
    initial_state = 5.0 * np.random.randn(NX)
    return a_matrix, b_matrix, offset, initial_state


def build_problem(horizon: int) -> tuple[CasadiProblem, np.ndarray, np.ndarray, np.ndarray]:
    a_matrix, b_matrix, offset, initial_state = problem_data()
    states = ca.SX.sym("x", NX, horizon)
    controls = ca.SX.sym("u", NU, horizon)
    theta = ca.SX.sym("theta", NX + NU)
    variable = ca.SX.sym("x0", NX)
    decision = ca.vertcat(ca.vec(states), ca.vec(controls))

    objective = 0
    outer_loss = 0
    constraints = []
    previous_state = variable
    for stage in range(horizon):
        state = states[:, stage]
        control = controls[:, stage]
        constraints.append(state - (a_matrix @ previous_state + b_matrix @ control + offset))
        if stage < horizon - 1:
            objective += ca.dot(theta[:NX] * state, state)
        objective += ca.dot(theta[NX:] * control, control)
        outer_loss += 0.5 * (ca.sumsqr(state) + ca.sumsqr(control))
        previous_state = state

    lower = np.concatenate(
        [np.full(NX * horizon, -1e20), np.full(NU * horizon, -UMAX)]
    )
    upper = np.concatenate(
        [np.full(NX * horizon, 1e20), np.full(NU * horizon, UMAX)]
    )
    problem = CasadiProblem(
        u=decision,
        theta=theta,
        variable=variable,
        cost=objective,
        box_lower=lower,
        box_upper=upper,
        constraints=ca.vertcat(*constraints),
        outer_loss=outer_loss,
    )

    state_guess = []
    current = initial_state.copy()
    for _ in range(horizon):
        current = a_matrix @ current + offset
        state_guess.append(current.copy())
    initial_guess = np.concatenate([np.asarray(state_guess).reshape(-1), np.zeros(NU * horizon)])
    return problem, initial_state, initial_guess, np.zeros(NX * horizon)


def solver_options() -> tuple[SolverOptions, AlmOptions, BackwardOptions]:
    inner = SolverOptions()
    inner.max_iterations = 1500
    inner.tolerance = TOLERANCE
    inner.buffer_size = 10
    inner.max_stable_iter = 80

    alm = AlmOptions()
    alm.max_iterations = 20
    alm.tolerance = TOLERANCE
    alm.initial_penalty = 10.0
    alm.penalty_update_factor = 5.0
    alm.max_penalty = 0.0
    alm.sufficient_decrease_factor = 0.25
    alm.warm_start_inner = True

    backward = BackwardOptions()
    backward.enable = True
    backward.tolerance = TOLERANCE
    backward.max_iterations = 800
    backward.constraint_penalty_scale = 10.0
    backward.constraint_penalty_max = 1e6
    return inner, alm, backward


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_horizon(horizon: int, repetitions: int, data_dir: Path, force: bool) -> Path:
    problem, initial_state, solution_warm, constraint_value = build_problem(horizon)
    solver = build_solver(
        problem,
        backend="compiled",
        name=f"exp2_linear_horizon_scaling_h{horizon}",
        cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp2_horizon_scaling",
        force=force,
    )
    inner, alm, backward = solver_options()
    theta = np.ones(NX + NU)
    multiplier_warm = None
    rows: list[dict[str, float]] = []

    for repetition in range(repetitions):
        result = solver.solve_lapanda(
            solution_warm,
            theta,
            initial_state,
            constraint_value,
            constraint_value,
            inner_solver_options=inner,
            alm_options=alm,
            backward_options=backward,
            multiplier0=multiplier_warm,
        )
        solution_warm = np.asarray(result["solution"], dtype=float).reshape(-1)
        multiplier_warm = np.asarray(result["multipliers"], dtype=float).reshape(-1)
        forward_ms = 1000.0 * float(result["forward_time_sec"])
        backward_ms = 1000.0 * float(result["backward_time_sec"])
        row = {
            "repetition": repetition,
            "horizon": horizon,
            "forward_time_ms": forward_ms,
            "backward_time_ms": backward_ms,
            "value_grad_time_ms": forward_ms + backward_ms,
            "constraint_violation_inf": float(result["final_residual"]),
            "outer_iterations": int(result["iterations"]),
            "inner_iterations": int(np.sum(np.asarray(result["inner_iterations"], dtype=int))),
            "backward_iterations": int(result.get("backward_iterations", -1)),
            "final_residual": float(result["final_residual"]),
        }
        rows.append(row)
        print(
            f"lapanda N={horizon:4d} rep={repetition:02d} "
            f"fwd={forward_ms:9.3f}ms bwd={backward_ms:9.3f}ms "
            f"res={row['final_residual']:.3e}"
        )

    path = data_dir / f"linear_h{horizon}_lapanda_metrics.csv"
    write_rows(path, rows)
    print(f"wrote {path}")
    return path


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for horizon in args.horizons:
        run_horizon(horizon, args.repetitions, args.data_dir, args.force)


if __name__ == "__main__":
    main()
