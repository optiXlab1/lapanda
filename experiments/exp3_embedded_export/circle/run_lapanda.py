"""Run Python lapanda on the circle/rectangle obstacle tests.

The problem data uses the bicycle obstacle setup:
- circle: bicycle model, circular obstacle at the origin
- rectangle: bicycle model, product-of-fmax rectangle obstacle
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
import psutil

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import config as exp3_config
try:
    from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver
    from lapanda._lapanda import solve_lapanda_compiled, solve_panda_compiled
except ImportError:
    # The acados WSL environment imports only the shared symbolic problem
    # definitions and does not need the platform-specific lapanda extension.
    AlmOptions = BackwardOptions = SolverOptions = build_solver = None
    solve_lapanda_compiled = solve_panda_compiled = None

    @dataclass
    class CasadiProblem:
        u: ca.SX
        theta: ca.SX
        variable: ca.SX
        cost: ca.SX
        box_lower: object = None
        box_upper: object = None
        constraints: object = None
        outer_loss: object = None


WHEELBASE = 0.45
STEER_LIMIT = np.deg2rad(25.0)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


@dataclass
class ObstacleCase:
    name: str
    problem: CasadiProblem
    horizon: int
    dt: float
    theta: np.ndarray
    variable: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    x0: np.ndarray
    meta: dict


def angle_error(a, b):
    return ca.atan2(ca.sin(a - b), ca.cos(a - b))


def bicycle_step(px, py, heading, speed, steer, dt, wheelbase=WHEELBASE):
    return (
        px + dt * speed * ca.cos(heading),
        py + dt * speed * ca.sin(heading),
        heading + dt * speed * ca.tan(steer) / wheelbase,
    )


def bicycle_initial_guess(start, target, horizon, dt, y_amp, speed_limit=1.0, steer_limit=STEER_LIMIT):
    waypoint_x = np.linspace(start[0], target[0], horizon + 1)
    waypoint_y = y_amp * np.sin(np.linspace(0.0, np.pi, horizon + 1))
    previous = np.asarray(start[:2], dtype=float)
    previous_heading = float(start[2])
    warm = np.zeros(2 * horizon, dtype=float)
    for k in range(horizon):
        point = np.array([waypoint_x[k + 1], waypoint_y[k + 1]], dtype=float)
        diff = point - previous
        distance = float(np.linalg.norm(diff))
        desired_heading = math.atan2(diff[1], diff[0]) if distance > 1e-12 else previous_heading
        speed = min(max(distance / dt, -speed_limit), speed_limit)
        heading_rate = math.atan2(math.sin(desired_heading - previous_heading), math.cos(desired_heading - previous_heading)) / dt
        steer = 0.0 if abs(speed) < 1e-8 else math.atan(WHEELBASE * heading_rate / speed)
        warm[2 * k : 2 * k + 2] = [speed, min(max(steer, -steer_limit), steer_limit)]
        previous = point
        previous_heading = desired_heading
    return warm


def build_circle_case(horizon: int = exp3_config.CIRCLE_HORIZON, speed_limit: float = exp3_config.CIRCLE_SPEED_LIMIT) -> ObstacleCase:
    dt = 0.12
    n = 2 * horizon
    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 7)
    variable = ca.SX.sym("variable", 6)
    q_pos, q_heading, r_speed, r_steer, terminal_weight, safe_radius, obstacle_y = [theta[i] for i in range(7)]
    px, py, heading = variable[0], variable[1], variable[2]
    target_x, target_y, target_heading = variable[3], variable[4], variable[5]
    cost = ca.SX(0)
    constraints = []
    for k in range(horizon):
        speed = u[2 * k]
        steer = u[2 * k + 1]
        px, py, heading = bicycle_step(px, py, heading, speed, steer, dt)
        dx = px - target_x
        dy = py - target_y
        dheading = angle_error(heading, target_heading)
        cost += q_pos * (dx**2 + dy**2)
        cost += q_heading * dheading**2
        cost += r_speed * speed**2 + r_steer * steer**2
        constraints.append(safe_radius**2 - (px**2 + (py - obstacle_y) ** 2))
    terminal_heading_error = angle_error(heading, target_heading)
    cost += terminal_weight * (q_pos * ((px - target_x) ** 2 + (py - target_y) ** 2) + q_heading * terminal_heading_error**2)
    start = np.array(exp3_config.START_STATE, dtype=float)
    target = np.array(exp3_config.TARGET_STATE, dtype=float)
    theta_value = np.array(exp3_config.CIRCLE_THETA, dtype=float)
    variable_value = np.concatenate([start, target])
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=np.tile(np.array([-speed_limit, -STEER_LIMIT]), horizon),
        box_upper=np.tile(np.array([speed_limit, STEER_LIMIT]), horizon),
        constraints=ca.vertcat(*constraints),
        outer_loss=0.5 * ca.sumsqr(u),
    )
    return ObstacleCase(
        name="circle",
        problem=problem,
        horizon=horizon,
        dt=dt,
        theta=theta_value,
        variable=variable_value,
        lower=-1e20 * np.ones(horizon),
        upper=np.zeros(horizon),
        x0=np.zeros(n, dtype=float),
        meta={"safe_radius": 0.3, "obstacle": [0.0, 0.2]},
    )


def build_rectangle_case() -> ObstacleCase:
    horizon = exp3_config.RECTANGLE_HORIZON
    dt = exp3_config.RECTANGLE_DT
    n = 2 * horizon
    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 9)
    variable = ca.SX.sym("variable", 6)
    q_pos, q_heading, r_speed, r_steer, terminal_weight = [theta[i] for i in range(5)]
    margin_left, margin_right, margin_bottom, margin_top = [theta[i] for i in range(5, 9)]
    px, py, heading = variable[0], variable[1], variable[2]
    target_x, target_y, target_heading = variable[3], variable[4], variable[5]
    xmin = -0.35 - margin_left
    xmax = 0.35 + margin_right
    ymin = -0.22 - margin_bottom
    ymax = 0.22 + margin_top
    cost = ca.SX(0)
    constraints = []
    for k in range(horizon):
        speed = u[2 * k]
        steer = u[2 * k + 1]
        px, py, heading = bicycle_step(px, py, heading, speed, steer, dt)
        dx = px - target_x
        dy = py - target_y
        dheading = angle_error(heading, target_heading)
        cost += q_pos * (dx**2 + dy**2)
        cost += q_heading * dheading**2
        cost += r_speed * speed**2 + r_steer * steer**2
        constraints.append(
            0.5
            * ca.fmax(px - xmin, 0) ** 2
            * ca.fmax(xmax - px, 0) ** 2
            * ca.fmax(py - ymin, 0) ** 2
            * ca.fmax(ymax - py, 0) ** 2
        )
    terminal_heading_error = angle_error(heading, target_heading)
    cost += terminal_weight * (q_pos * ((px - target_x) ** 2 + (py - target_y) ** 2) + q_heading * terminal_heading_error**2)
    start = np.array(exp3_config.START_STATE, dtype=float)
    target = np.array(exp3_config.TARGET_STATE, dtype=float)
    theta_value = np.array([*exp3_config.RECTANGLE_THETA_PREFIX, *exp3_config.RECTANGLE_EXPORT_MARGINS], dtype=float)
    variable_value = np.concatenate([start, target])
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=np.tile(np.array([-exp3_config.RECTANGLE_SPEED_LIMIT, -STEER_LIMIT]), horizon),
        box_upper=np.tile(np.array([exp3_config.RECTANGLE_SPEED_LIMIT, STEER_LIMIT]), horizon),
        constraints=ca.vertcat(*constraints),
        outer_loss=0.5 * ca.sumsqr(u),
    )
    return ObstacleCase(
        name="rectangle",
        problem=problem,
        horizon=horizon,
        dt=dt,
        theta=theta_value,
        variable=variable_value,
        lower=-1e20 * np.ones(horizon),
        upper=np.zeros(horizon),
        x0=bicycle_initial_guess(start, target, horizon, dt, y_amp=0.8),
        meta={"rectangle": [-0.45, 0.45, -0.32, 0.32]},
    )


def rollout(u, variable, dt):
    px, py, heading = [float(v) for v in variable[:3]]
    traj = np.zeros((len(u) // 2, 3), dtype=float)
    for k in range(len(u) // 2):
        speed = float(u[2 * k])
        steer = float(u[2 * k + 1])
        px = px + dt * speed * math.cos(heading)
        py = py + dt * speed * math.sin(heading)
        heading = heading + dt * speed * math.tan(steer) / WHEELBASE
        traj[k] = [px, py, heading]
    return traj


def metrics(case: ObstacleCase, solution):
    traj = rollout(solution, case.variable, case.dt)
    pos = traj[:, :2]
    target = case.variable[3:5]
    out = {
        "final_distance": float(np.linalg.norm(pos[-1] - target)),
        "max_speed_abs": float(np.max(np.abs(solution[0::2]))),
        "max_steer_abs": float(np.max(np.abs(solution[1::2]))),
    }
    if case.name == "circle":
        radius = float(case.meta["safe_radius"])
        center = np.asarray(case.meta.get("obstacle", [0.0, 0.0]), dtype=float)
        dist = np.linalg.norm(pos - center, axis=1)
        out["min_distance"] = float(np.min(dist))
        out["geometric_violation"] = float(np.max(radius - dist))
        out["constraint_max"] = float(np.max(radius**2 - np.sum((pos - center) ** 2, axis=1)))
    else:
        xmin, xmax, ymin, ymax = case.meta["rectangle"]
        side = np.column_stack([pos[:, 0] - xmin, xmax - pos[:, 0], pos[:, 1] - ymin, ymax - pos[:, 1]])
        inside_margin = np.min(side, axis=1)
        product = 0.5 * np.prod(np.maximum(side, 0.0) ** 2, axis=1)
        out["inside_margin_max"] = float(np.max(inside_margin))
        out["geometric_violation"] = float(max(0.0, out["inside_margin_max"]))
        out["constraint_max"] = float(np.max(product))
    return out


def solve_case(case: ObstacleCase, args):
    cache = REPO_ROOT / ".lapanda_cache" / "obstacle_cases"
    rss_start = rss_mb()
    build_t0 = time.perf_counter()
    solver = build_solver(
        case.problem,
        backend="compiled",
        name=f"matlab_{case.name}_bicycle",
        cache_dir=cache,
        force=args.force_build,
        build=True,
    )
    build_time = time.perf_counter() - build_t0
    rss_after_build = rss_mb()
    metadata = solver.generated_oracle.metadata
    solver_options = SolverOptions()
    solver_options.max_iterations = args.inner_max_iter
    solver_options.tolerance = args.inner_tol
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = args.inner_max_stable_iter
    alm_options = AlmOptions()
    alm_options.max_iterations = args.alm_max_iter
    alm_options.tolerance = args.alm_tol
    alm_options.initial_penalty = args.alm_initial_penalty
    alm_options.penalty_update_factor = args.alm_penalty_update_factor
    alm_options.max_penalty = args.alm_max_penalty
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True
    def call_unconstrained_solver(enable_backward: bool):
        backward_options = BackwardOptions()
        backward_options.enable = enable_backward
        backward_options.tolerance = args.backward_tol
        backward_options.max_iterations = args.backward_max_iter
        backward_options.linear_solver = args.linear_solver
        return solve_panda_compiled(
            str(solver.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["box_lower"],
            metadata["box_upper"],
            case.x0,
            case.theta,
            case.variable,
            solver_options,
            backward_options,
        )

    def call_solver(enable_backward: bool):
        if case.problem.constraints is None:
            return call_unconstrained_solver(enable_backward)
        backward_options = BackwardOptions()
        backward_options.enable = enable_backward
        backward_options.tolerance = args.backward_tol
        backward_options.max_iterations = args.backward_max_iter
        backward_options.linear_solver = args.linear_solver
        backward_options.constraint_penalty_scale = args.backward_constraint_penalty_scale
        backward_options.constraint_penalty_max = args.backward_constraint_penalty_max
        return solve_lapanda_compiled(
            str(solver.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["ncon"],
            metadata["box_lower"],
            metadata["box_upper"],
            case.x0,
            case.theta,
            case.variable,
            case.lower,
            case.upper,
            solver_options,
            alm_options,
            backward_options,
            None,
            None,
        )

    result = call_solver(args.compute_backward)
    rss_after_solve = rss_mb()

    forward_time = float(result["forward_time_sec"])
    backward_time = float(result["backward_time_sec"]) if args.compute_backward else float("nan")
    total_time = forward_time + (0.0 if math.isnan(backward_time) else backward_time)

    solution = np.asarray(result["solution"], dtype=float).reshape(-1)
    row = {
        "problem": case.name,
        "build_time_sec": build_time,
        "forward_time_sec": forward_time,
        "backward_time_sec": backward_time,
        "total_with_backward_time_sec": total_time,
        "solve_time_sec": total_time,
        "rss_start_mb": rss_start,
        "rss_after_build_mb": rss_after_build,
        "rss_after_forward_mb": rss_after_solve,
        "rss_after_backward_mb": rss_after_solve,
        "build_rss_delta_mb": rss_after_build - rss_start,
        "forward_rss_delta_mb": float("nan"),
        "backward_rss_delta_mb": float("nan"),
        "solve_rss_delta_mb": rss_after_solve - rss_after_build,
        "total_rss_delta_mb": rss_after_solve - rss_start,
        "outer_iterations": int(result.get("iterations", -1)),
        "final_residual": float(result.get("final_residual", np.nan)),
        "penalty": float(result.get("penalty", np.nan)),
        "inner_iterations": json.dumps(np.asarray(result.get("inner_iterations", []), dtype=int).tolist()),
        "inner_iterations_sum": int(np.sum(np.asarray(result.get("inner_iterations", []), dtype=int))),
        "backward_iterations": int(result.get("backward_iterations", -1)),
        "backward_residual": float(result.get("backward_residual", np.nan)),
        "backward_solver_used": result.get("backward_solver_used", ""),
        "backward_fallback_used": bool(result.get("backward_fallback_used", False)),
        **metrics(case, solution),
    }
    return row, solution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["circle", "rectangle", "all"], default="circle")
    parser.add_argument("--drop-constraints", action="store_true")
    parser.add_argument("--outdir", default=str(exp3_config.CIRCLE_RESULT_DIR))
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--init-solution", default="")
    parser.add_argument("--terminal-weight", type=float, default=None)
    parser.add_argument("--circle-radius", type=float, default=exp3_config.CIRCLE_RADIUS)
    parser.add_argument("--circle-center-y", type=float, default=exp3_config.CIRCLE_CENTER_Y)
    parser.add_argument("--horizon", type=int, default=exp3_config.CIRCLE_HORIZON)
    parser.add_argument("--speed-limit", type=float, default=exp3_config.CIRCLE_SPEED_LIMIT)
    parser.add_argument("--inner-max-iter", type=int, default=exp3_config.INNER_MAX_ITER)
    parser.add_argument("--inner-max-stable-iter", type=int, default=exp3_config.INNER_MAX_STABLE_ITER)
    parser.add_argument("--inner-tol", type=float, default=exp3_config.CIRCLE_INNER_TOL)
    parser.add_argument("--alm-max-iter", type=int, default=exp3_config.ALM_MAX_ITER)
    parser.add_argument("--alm-tol", type=float, default=exp3_config.CIRCLE_ALM_TOL)
    parser.add_argument("--alm-initial-penalty", type=float, default=exp3_config.ALM_INITIAL_PENALTY)
    parser.add_argument("--alm-penalty-update-factor", type=float, default=exp3_config.ALM_PENALTY_UPDATE_FACTOR)
    parser.add_argument("--alm-max-penalty", type=float, default=exp3_config.ALM_MAX_PENALTY)
    parser.add_argument("--compute-backward", action="store_true")
    parser.add_argument("--backward-max-iter", type=int, default=exp3_config.BACKWARD_MAX_ITER)
    parser.add_argument("--backward-tol", type=float, default=exp3_config.BACKWARD_TOL)
    parser.add_argument("--linear-solver", choices=["cg", "minres", "auto"], default="cg")
    parser.add_argument("--backward-constraint-penalty-scale", type=float, default=1.0)
    parser.add_argument("--backward-constraint-penalty-max", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.problem in ("circle", "all"):
        cases.append(build_circle_case(args.horizon, args.speed_limit))
    if args.problem in ("rectangle", "all"):
        cases.append(build_rectangle_case())
    if args.drop_constraints:
        for case in cases:
            case.problem.constraints = None
            case.lower = np.zeros(0)
            case.upper = np.zeros(0)
    rows = []
    for case in cases:
        if args.terminal_weight is not None:
            case.theta[4] = args.terminal_weight
        if args.circle_radius is not None and case.name == "circle":
            case.theta[5] = args.circle_radius
            case.meta["safe_radius"] = args.circle_radius
        if case.name == "circle":
            case.theta[6] = args.circle_center_y
            case.meta["obstacle"] = [0.0, args.circle_center_y]
        if args.init_solution:
            case.x0 = np.loadtxt(args.init_solution, delimiter=",").reshape(-1)
        row, solution = solve_case(case, args)
        rows.append(row)
        np.savetxt(outdir / f"{case.name}_solution.csv", solution, delimiter=",")
        print(
            "{problem}: outer={outer_iterations} residual={final_residual:.3e} "
            "constraint_max={constraint_max:.3e} geom_viol={geometric_violation:.3e} "
            "inner_sum={inner_iterations_sum} solve={solve_time_sec:.4f}s".format(**row)
        )
    csv_path = outdir / "obstacle_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / "obstacle_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "args": vars(args)}, f, indent=2)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
