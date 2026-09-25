"""Shared problem definition for the paper's smoothed-rectangle benchmark.

The Figure 11 lapanda and acados rollout scripts import the vehicle model,
smooth obstacle constraint, initialization, and solver builders from here.
The command-line diagnostic entry point is retained for optional debugging.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import casadi as ca
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_OUTDIR = EXP_DIR / "results" / "rectangle" / "mpcc_diagnostics"

HORIZON = 20
DT = 0.12
PREDICTION_TIME = HORIZON * DT
WHEELBASE = 0.45
STEER_LIMIT = math.radians(25.0)
SPEED_LIMIT = 1.0
OBSTACLE_SCALE = 1.0
START = np.array([-1.2, 0.0, 0.0], dtype=float)
TARGET = np.array([1.2, 0.0, 0.0], dtype=float)
THETA_PREFIX = np.array([5.0, 0.2, 1e-2, 1e-2, 20.0], dtype=float)
INITIAL_MARGINS = np.array([0.15, 0.15, 0.16, 0.22], dtype=float)
TEACHER_MARGINS = np.array([0.12, 0.12, 0.01, 0.22], dtype=float)
BASE_RECTANGLE = np.array([-0.35, 0.35, -0.22, 0.22], dtype=float)
DEFAULT_TAUS = (0.20, 0.10, 0.05, 0.02, 0.01)


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def bicycle_step(x, u):
    px, py, heading = x[0], x[1], x[2]
    speed, steer = u[0], u[1]
    return ca.vertcat(
        px + DT * speed * ca.cos(heading),
        py + DT * speed * ca.sin(heading),
        heading + DT * speed * ca.tan(steer) / WHEELBASE,
    )


def angle_error(value, reference):
    return ca.atan2(ca.sin(value - reference), ca.cos(value - reference))


def rectangle_bounds(margins: np.ndarray) -> tuple[float, float, float, float]:
    xmin = BASE_RECTANGLE[0] - margins[0]
    xmax = BASE_RECTANGLE[1] + margins[1]
    ymin = BASE_RECTANGLE[2] - margins[2]
    ymax = BASE_RECTANGLE[3] + margins[3]
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    half_width = 0.5 * OBSTACLE_SCALE * (xmax - xmin)
    half_height = 0.5 * OBSTACLE_SCALE * (ymax - ymin)
    return center_x - half_width, center_x + half_width, center_y - half_height, center_y + half_height


def initial_controls(y_amplitude: float = 0.8) -> np.ndarray:
    waypoint_x = np.linspace(START[0], TARGET[0], HORIZON + 1)
    waypoint_y = y_amplitude * np.sin(np.linspace(0.0, np.pi, HORIZON + 1))
    previous = START[:2].copy()
    previous_heading = float(START[2])
    controls = np.zeros(2 * HORIZON, dtype=float)
    for k in range(HORIZON):
        point = np.array([waypoint_x[k + 1], waypoint_y[k + 1]])
        delta = point - previous
        distance = float(np.linalg.norm(delta))
        desired_heading = math.atan2(delta[1], delta[0]) if distance > 1e-12 else previous_heading
        speed = np.clip(distance / DT, -SPEED_LIMIT, SPEED_LIMIT)
        heading_rate = math.atan2(
            math.sin(desired_heading - previous_heading),
            math.cos(desired_heading - previous_heading),
        ) / DT
        steer = 0.0 if abs(speed) < 1e-8 else math.atan(WHEELBASE * heading_rate / speed)
        controls[2 * k : 2 * k + 2] = [speed, np.clip(steer, -STEER_LIMIT, STEER_LIMIT)]
        previous = point
        previous_heading = desired_heading
    return controls


def handcrafted_feasible_controls() -> np.ndarray:
    """Solver-independent feasible arc used for fair timing diagnostics."""
    steering = np.array([0.6] * 5 + [-0.6] * 3 + [0.0] * 4 + [-0.6] * 3 + [0.6] * 5)
    controls = np.empty(2 * HORIZON, dtype=float)
    controls[0::2] = 1.0
    controls[1::2] = steering
    return controls


def load_teacher() -> np.ndarray:
    path = (
        EXP_DIR
        / "results"
        / "rectangle"
        / "lapanda"
        / "teacher_solution.csv"
    )
    values = np.loadtxt(path, delimiter=",").reshape(-1, 2)
    if values.shape[0] == HORIZON:
        return values.reshape(-1)
    source_time = (np.arange(values.shape[0]) + 0.5) / values.shape[0]
    target_time = (np.arange(HORIZON) + 0.5) / HORIZON
    return np.column_stack(
        [np.interp(target_time, source_time, values[:, j]) for j in range(2)]
    ).reshape(-1)


def signed_outside_distances(px, py, margins):
    xmin, xmax, ymin, ymax = rectangle_bounds(margins)
    return ca.vertcat(xmin - px, px - xmax, ymin - py, py - ymax)


def obstacle_expression(px, py, margins, mode: str, tau: float):
    distances = signed_outside_distances(px, py, margins)
    if mode == "product2":
        inside = ca.fmax(-distances, 0)
        return 0.5 * inside[0] ** 2 * inside[1] ** 2 * inside[2] ** 2 * inside[3] ** 2
    if mode == "product1":
        inside = ca.fmax(-distances, 0)
        return inside[0] * inside[1] * inside[2] * inside[3]
    if mode == "smooth":
        # Conservative smooth maximum: smooth_max <= max(distances).
        return tau * ca.log(ca.sum1(ca.exp(distances / tau))) - tau * math.log(4.0)
    if mode == "smoothsqrt":
        # Balanced pairwise under-approximation of max(a, b).
        def smooth_max_below(a, b):
            return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + tau**2) - tau)

        left = smooth_max_below(distances[0], distances[1])
        right = smooth_max_below(distances[2], distances[3])
        return smooth_max_below(left, right)
    if mode == "scholtes":
        # Smooth, dimensionless positive parts with a Scholtes-type product
        # relaxation. Here tau is the relaxation level epsilon.
        delta = 1e-3
        xmin, xmax, ymin, ymax = rectangle_bounds(margins)
        width = xmax - xmin
        height = ymax - ymin
        inside = ca.vertcat(
            (px - xmin) / width,
            (xmax - px) / width,
            (py - ymin) / height,
            (ymax - py) / height,
        )
        positive = 0.5 * (inside + ca.sqrt(inside**2 + delta**2))
        return tau - positive[0] * positive[1] * positive[2] * positive[3]
    if mode == "superellipse":
        order = int(round(tau))
        if order < 2 or order % 2:
            raise ValueError("superellipse order must be an even integer >= 2")
        xmin, xmax, ymin, ymax = rectangle_bounds(margins)
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        half_width = 0.5 * (xmax - xmin)
        half_height = 0.5 * (ymax - ymin)
        # The sublevel set below 2 contains the rectangle and converges to it
        # as the even order increases.
        return (
            ((px - center_x) / half_width) ** order
            + ((py - center_y) / half_height) ** order
            - 2.0
        )
    raise ValueError(mode)


def reduced_problem(mode: str, tau: float, margins_value: np.ndarray, teacher: np.ndarray):
    controls = ca.SX.sym("u", 2 * HORIZON)
    theta = ca.SX.sym("theta", 9)
    variable = ca.SX.sym("variable", 6 + 2 * HORIZON)
    margins = theta[5:9]
    state = variable[:3]
    target = variable[3:6]
    teacher_symbol = variable[6:]
    q_pos, q_heading, r_speed, r_steer, terminal_weight = [theta[i] for i in range(5)]
    cost = ca.SX(0)
    constraints = []
    states = []
    for k in range(HORIZON):
        control = controls[2 * k : 2 * k + 2]
        state = bicycle_step(state, control)
        states.append(state)
        dx = state[0] - target[0]
        dy = state[1] - target[1]
        heading_error = angle_error(state[2], target[2])
        cost += q_pos * (dx**2 + dy**2)
        cost += q_heading * heading_error**2
        cost += r_speed * control[0] ** 2 + r_steer * control[1] ** 2
        constraints.append(obstacle_expression(state[0], state[1], margins, mode, tau))
    terminal_error = angle_error(state[2], target[2])
    cost += terminal_weight * (
        q_pos * ((state[0] - target[0]) ** 2 + (state[1] - target[1]) ** 2)
        + q_heading * terminal_error**2
    )
    constraint_vector = ca.vertcat(*constraints)
    theta_value = np.r_[THETA_PREFIX, margins_value]
    variable_value = np.r_[START, TARGET, teacher]
    lower = np.zeros(HORIZON)
    upper = np.zeros(HORIZON) if mode.startswith("product") else np.full(HORIZON, 1e20)
    diagnostic = ca.Function(
        "rectangle_diagnostic",
        [controls, theta, variable],
        [
            cost,
            constraint_vector,
            ca.jacobian(constraint_vector, controls),
            ca.hcat(states).T,
        ],
    )
    return {
        "controls": controls,
        "theta": theta,
        "variable": variable,
        "cost": cost,
        "constraints": constraint_vector,
        "outer_loss": 0.5 * ca.sumsqr(controls - teacher_symbol),
        "theta_value": theta_value,
        "variable_value": variable_value,
        "lower": lower,
        "upper": upper,
        "diagnostic": diagnostic,
    }


def geometric_metrics(states: np.ndarray, margins: np.ndarray) -> dict[str, float]:
    xmin, xmax, ymin, ymax = rectangle_bounds(margins)
    side_distances = np.column_stack(
        [states[:, 0] - xmin, xmax - states[:, 0], states[:, 1] - ymin, ymax - states[:, 1]]
    )
    inside_depth = np.min(side_distances, axis=1)
    return {
        "geometric_violation": float(max(0.0, np.max(inside_depth))),
        "minimum_clearance_signed": float(-np.max(inside_depth)),
        "passed_obstacle": bool(np.max(states[:, 0]) > xmax),
        "final_distance": float(np.linalg.norm(states[-1, :2] - TARGET[:2])),
    }


def numerical_metrics(problem, solution: np.ndarray) -> dict[str, float | int | list[float]]:
    cost, constraints, jacobian, states = problem["diagnostic"](
        solution,
        problem["theta_value"],
        problem["variable_value"],
    )
    constraints = np.asarray(constraints, dtype=float).reshape(-1)
    jacobian = np.asarray(jacobian, dtype=float)
    states = np.asarray(states, dtype=float)
    row_norms = np.linalg.norm(jacobian, axis=1)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    result = {
        "objective": float(cost),
        "constraint_min": float(np.min(constraints)),
        "constraint_max": float(np.max(constraints)),
        "jacobian_rank_1e8": int(np.linalg.matrix_rank(jacobian, tol=1e-8)),
        "jacobian_row_norm_min": float(np.min(row_norms)),
        "jacobian_row_norm_max": float(np.max(row_norms)),
        "jacobian_sigma_min": float(np.min(singular_values)),
        "jacobian_sigma_max": float(np.max(singular_values)),
        "states": states.tolist(),
        "constraint_values": constraints.tolist(),
    }
    result.update(geometric_metrics(states, problem["theta_value"][5:9]))
    return result


def run_lapanda(
    mode: str,
    tau: float,
    outdir: Path,
    guess_source: str = "default",
    repetitions: int = 1,
    initial_penalty: float = 1e4,
    initial_guess_file: Path | None = None,
    inner_tolerance: float = 1e-4,
) -> dict:
    import sys

    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver
    from lapanda._lapanda import solve_lapanda_compiled

    teacher = load_teacher()
    problem = reduced_problem(mode, tau, INITIAL_MARGINS, teacher)
    casadi_problem = CasadiProblem(
        u=problem["controls"],
        theta=problem["theta"],
        variable=problem["variable"],
        cost=problem["cost"],
        box_lower=np.tile([-SPEED_LIMIT, -STEER_LIMIT], HORIZON),
        box_upper=np.tile([SPEED_LIMIT, STEER_LIMIT], HORIZON),
        constraints=problem["constraints"],
        outer_loss=problem["outer_loss"],
    )
    if mode.startswith("product"):
        tag = mode
    elif mode == "superellipse":
        tag = f"se{int(round(tau))}"
    else:
        short_mode = {"smooth": "lse", "smoothsqrt": "sqrt", "scholtes": "sch"}.get(mode, mode)
        tag = f"{short_mode}{tau:g}".replace(".", "p").replace("-", "m")
    configuration_key = (
        f"n{HORIZON}_t{PREDICTION_TIME:g}_v{SPEED_LIMIT:g}_"
        f"l{WHEELBASE:g}_d{STEER_LIMIT:g}_o{OBSTACLE_SCALE:g}"
    )
    configuration_tag = hashlib.sha1(configuration_key.encode("ascii")).hexdigest()[:8]
    solver = build_solver(
        casadi_problem,
        backend="compiled",
        name=f"rectdiag_{tag}_{configuration_tag}".replace(".", "p"),
        cache_dir=REPO_ROOT / ".lapanda_cache" / "mpcc" / tag / configuration_tag,
        build=True,
    )
    solver_options = SolverOptions()
    solver_options.max_iterations = 20000
    solver_options.tolerance = inner_tolerance
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 80
    alm_options = AlmOptions()
    alm_options.max_iterations = 100
    alm_options.tolerance = 1e-4
    alm_options.initial_penalty = initial_penalty
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True
    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-4
    backward_options.max_iterations = 1000
    backward_options.linear_solver = "cg"
    metadata = solver.generated_oracle.metadata
    forward_times = []
    backward_times = []
    wall_times = []
    result = None
    if initial_guess_file is not None:
        initial_guess = np.asarray(
            json.loads(initial_guess_file.read_text(encoding="utf-8"))["solution"], dtype=float
        )
        guess_source = "file"
    elif guess_source == "acados":
        tau_tag = "none" if mode.startswith("product") else f"{tau:g}".replace(".", "p")
        guess_path = outdir / f"acados_{mode}_tau_{tau_tag}_zero_exact.json"
        initial_guess = np.asarray(
            json.loads(guess_path.read_text(encoding="utf-8"))["solution"], dtype=float
        )
    elif guess_source == "handcrafted":
        initial_guess = handcrafted_feasible_controls()
    elif guess_source == "zero":
        initial_guess = np.zeros(2 * HORIZON)
    else:
        initial_guess = initial_controls()
    if guess_source == "continuation":
        continuation_options = AlmOptions()
        continuation_options.max_iterations = 100
        continuation_options.tolerance = 1e-4
        continuation_options.initial_penalty = 1.0
        continuation_options.penalty_update_factor = 10.0
        continuation_options.sufficient_decrease_factor = 0.25
        continuation_options.warm_start_inner = True
        continuation_backward = BackwardOptions()
        continuation_backward.enable = False
        continuation_backward.linear_solver = "cg"
        continuation_result = solve_lapanda_compiled(
            str(solver.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["ncon"],
            metadata["box_lower"],
            metadata["box_upper"],
            initial_guess,
            problem["theta_value"],
            problem["variable_value"],
            problem["lower"],
            problem["upper"],
            solver_options,
            continuation_options,
            continuation_backward,
            None,
            None,
        )
        initial_guess = np.asarray(continuation_result["solution"], dtype=float)
    for repeat in range(repetitions + 1):
        t0 = time.perf_counter()
        result = solve_lapanda_compiled(
            str(solver.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["ncon"],
            metadata["box_lower"],
            metadata["box_upper"],
            initial_guess,
            problem["theta_value"],
            problem["variable_value"],
            problem["lower"],
            problem["upper"],
            solver_options,
            alm_options,
            backward_options,
            None,
            None,
        )
        elapsed = time.perf_counter() - t0
        if repeat > 0:
            wall_times.append(elapsed)
            forward_times.append(float(result.get("forward_time_sec", np.nan)))
            backward_times.append(float(result.get("backward_time_sec", np.nan)))
    assert result is not None
    solution = np.asarray(result["solution"], dtype=float)
    record = {
        "solver": "lapanda",
        "mode": mode,
        "tau": tau,
        "guess_source": guess_source,
        "initial_penalty": initial_penalty,
        "horizon": HORIZON,
        "prediction_time": PREDICTION_TIME,
        "speed_limit": SPEED_LIMIT,
        "steer_limit": STEER_LIMIT,
        "wheelbase": WHEELBASE,
        "obstacle_scale": OBSTACLE_SCALE,
        "terminal_weight": float(THETA_PREFIX[4]),
        "forward_tolerance": inner_tolerance,
        "alm_tolerance": 1e-4,
        "backward_tolerance": 1e-4,
        "success": bool(np.all(np.isfinite(solution))),
        "repetitions": repetitions,
        "forward_time_sec": float(np.mean(forward_times)),
        "forward_time_std_sec": float(np.std(forward_times, ddof=1)) if repetitions > 1 else 0.0,
        "backward_time_sec": float(np.mean(backward_times)),
        "backward_time_std_sec": float(np.std(backward_times, ddof=1)) if repetitions > 1 else 0.0,
        "wall_time_sec": float(np.mean(wall_times)),
        "outer_iterations": int(result.get("outer_iterations", result.get("iterations", -1))),
        "solver_status": result.get("status"),
        "inner_iterations": np.asarray(result.get("inner_iterations", []), dtype=int).tolist(),
        "inner_iterations_sum": int(np.sum(np.asarray(result.get("inner_iterations", []), dtype=int))),
        "backward_iterations": int(result.get("backward_iterations", -1)),
        "backward_residual": float(result.get("backward_residual", np.nan)),
        "backward_solver_used": result.get("backward_solver_used", ""),
        "backward_fallback_used": bool(result.get("backward_fallback_used", False)),
        "final_residual": float(result.get("final_residual", np.nan)),
        "penalty": float(result.get("penalty", np.nan)),
        "imitation_loss": 0.5 * float(np.sum((solution - teacher) ** 2)),
        "solution": solution.tolist(),
    }
    record.update(numerical_metrics(problem, solution))
    record["success"] = bool(
        record["success"]
        and record["final_residual"] <= 1e-4
        and record["geometric_violation"] <= 1e-4
        and record["passed_obstacle"]
    )
    return record


def build_acados_ocp(
    mode: str,
    tau: float,
    sensitivity: bool,
    workdir: Path,
    forward_hessian: str = "gn",
    termination: str = "kkt",
):
    from acados_template import AcadosModel, AcadosOcp

    model = AcadosModel()
    tag = (
        mode
        if mode.startswith("product")
        else f"{mode}_{tau:g}".replace(".", "p").replace("-", "m")
    )
    model.name = f"rectangle_diag_{tag}_{'sens' if sensitivity else 'fwd'}".replace(".", "p")
    x = ca.SX.sym("x", 3)
    u = ca.SX.sym("u", 2)
    p = ca.SX.sym("p_global", 9)
    q_pos, q_heading, r_speed, r_steer, terminal_weight = [p[i] for i in range(5)]
    margins = p[5:9]
    x_next = bicycle_step(x, u)
    dx = x_next[0] - TARGET[0]
    dy = x_next[1] - TARGET[1]
    heading_error = angle_error(x_next[2], TARGET[2])
    terminal_dx = x[0] - TARGET[0]
    terminal_dy = x[1] - TARGET[1]
    terminal_heading_error = angle_error(x[2], TARGET[2])
    model.x = x
    model.u = u
    model.p_global = p
    model.disc_dyn_expr = x_next
    use_external = sensitivity or forward_hessian == "exact"
    if use_external:
        model.cost_expr_ext_cost = (
            q_pos * (dx**2 + dy**2)
            + q_heading * heading_error**2
            + r_speed * u[0] ** 2
            + r_steer * u[1] ** 2
        )
        model.cost_expr_ext_cost_e = terminal_weight * (
            q_pos * (terminal_dx**2 + terminal_dy**2)
            + q_heading * terminal_heading_error**2
        )
    else:
        model.cost_y_expr = ca.vertcat(
            ca.sqrt(q_pos) * dx,
            ca.sqrt(q_pos) * dy,
            ca.sqrt(q_heading) * heading_error,
            ca.sqrt(r_speed) * u[0],
            ca.sqrt(r_steer) * u[1],
        )
        model.cost_y_expr_e = ca.vertcat(
            ca.sqrt(terminal_weight * q_pos) * terminal_dx,
            ca.sqrt(terminal_weight * q_pos) * terminal_dy,
            ca.sqrt(terminal_weight * q_heading) * terminal_heading_error,
        )
    model.con_h_expr = ca.vertcat(obstacle_expression(x_next[0], x_next[1], margins, mode, tau))

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = HORIZON
    ocp.solver_options.tf = HORIZON * DT
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 10000
    ocp.solver_options.nlp_solver_tol_stat = 1e2 if termination == "feasibility" else 1e-5
    ocp.solver_options.nlp_solver_tol_eq = 1e-5
    ocp.solver_options.nlp_solver_tol_ineq = 1e-5
    ocp.solver_options.nlp_solver_tol_comp = 1e2 if termination == "feasibility" else 1e-5
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = HORIZON
    ocp.solver_options.qp_solver_iter_max = 500
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_ric_alg = 0
    ocp.solver_options.hessian_approx = "EXACT" if use_external else "GAUSS_NEWTON"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "NO_REGULARIZE" if sensitivity else "MIRROR"
    if sensitivity:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True
    if use_external:
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
    else:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = 2.0 * np.eye(5)
        ocp.cost.W_e = 2.0 * np.eye(3)
        ocp.cost.yref = np.zeros(5)
        ocp.cost.yref_e = np.zeros(3)
    ocp.constraints.x0 = START
    ocp.constraints.lbu = np.array([-SPEED_LIMIT, -STEER_LIMIT])
    ocp.constraints.ubu = np.array([SPEED_LIMIT, STEER_LIMIT])
    ocp.constraints.idxbu = np.array([0, 1])
    if mode.startswith("product"):
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([0.0])
    else:
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([1e20])
    ocp.p_global_values = np.r_[THETA_PREFIX, INITIAL_MARGINS]
    ocp.code_export_directory = str(workdir / f"codegen_{model.name}")
    return ocp


def set_acados_guess(solver, controls: np.ndarray):
    state = START.copy()
    solver.set(0, "x", state)
    for k in range(HORIZON):
        control = controls[2 * k : 2 * k + 2]
        solver.set(k, "u", control)
        speed, steer = control
        state = np.array(
            [
                state[0] + DT * speed * np.cos(state[2]),
                state[1] + DT * speed * np.sin(state[2]),
                state[2] + DT * speed * np.tan(steer) / WHEELBASE,
            ]
        )
        solver.set(k + 1, "x", state)


def acados_solution(solver) -> np.ndarray:
    return np.concatenate([np.asarray(solver.get(k, "u"), dtype=float) for k in range(HORIZON)])


def acados_stats(solver) -> dict:
    output = {}
    for key in ("sqp_iter", "time_tot", "time_qp", "time_lin", "residuals", "statistics"):
        try:
            value = np.asarray(solver.get_stats(key))
            output[key] = value.tolist()
        except Exception as exc:
            output[f"{key}_error"] = repr(exc)
    return output


def lapanda_guess(outdir: Path, mode: str, tau: float) -> np.ndarray:
    tau_tag = "none" if mode.startswith("product") else f"{tau:g}".replace(".", "p")
    path = outdir / f"lapanda_{mode}_tau_{tau_tag}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["solution"], dtype=float)


def run_acados(
    mode: str,
    tau: float,
    outdir: Path,
    guess_source: str,
    forward_hessian: str,
    repetitions: int,
    termination: str,
) -> dict:
    from acados_template import AcadosOcpSolver

    teacher = load_teacher()
    problem = reduced_problem(mode, tau, INITIAL_MARGINS, teacher)
    workdir = Path(tempfile.mkdtemp(prefix=f"rectangle_{mode}_", dir=str(outdir)))
    try:
        forward_ocp = build_acados_ocp(mode, tau, False, workdir, forward_hessian, termination)
        sensitivity_ocp = build_acados_ocp(mode, tau, True, workdir, forward_hessian, termination)
        forward = AcadosOcpSolver(
            forward_ocp,
            json_file=str(workdir / "forward.json"),
            verbose=False,
        )
        sensitivity = AcadosOcpSolver(
            sensitivity_ocp,
            json_file=str(workdir / "sensitivity.json"),
            verbose=False,
        )
        if guess_source == "default":
            guess = initial_controls()
        elif guess_source == "handcrafted":
            guess = handcrafted_feasible_controls()
        elif guess_source == "zero":
            guess = np.zeros(2 * HORIZON)
        else:
            guess = lapanda_guess(outdir, mode, tau)
        p_global = np.r_[THETA_PREFIX, INITIAL_MARGINS]
        forward.set_p_global_and_precompute_dependencies(p_global)
        sensitivity.set_p_global_and_precompute_dependencies(p_global)
        forward_times = []
        backward_times = []
        statuses = []
        solution = guess.copy()
        grad = np.full(9, np.nan)
        backward_status = "not_run"
        for repeat in range(repetitions + 1):
            forward.reset()
            set_acados_guess(forward, guess)
            forward.set_p_global_and_precompute_dependencies(p_global)
            t0 = time.perf_counter()
            status = int(forward.solve())
            elapsed_forward = time.perf_counter() - t0
            solution = acados_solution(forward)
            residual = solution - teacher
            seeds_u = [(k, residual[2 * k : 2 * k + 2].reshape(2, 1)) for k in range(HORIZON)]
            t1 = time.perf_counter()
            try:
                if hasattr(forward, "get_flat_iterate"):
                    iterate = forward.get_flat_iterate()
                else:
                    iterate = forward.store_iterate_to_flat_obj()
                sensitivity.load_iterate_from_flat_obj(iterate)
                sensitivity.setup_qp_matrices_and_factorize()
                grad = np.asarray(
                    sensitivity.eval_adjoint_solution_sensitivity(
                        seed_x=None,
                        seed_u=seeds_u,
                        with_respect_to="p_global",
                        sanity_checks=False,
                    ),
                    dtype=float,
                ).reshape(-1)[:9]
                backward_status = "ok"
            except Exception as exc:
                backward_status = repr(exc)
            elapsed_backward = time.perf_counter() - t1
            if repeat > 0:
                forward_times.append(elapsed_forward)
                backward_times.append(elapsed_backward)
                statuses.append(status)
        status = statuses[-1]
        residual = solution - teacher
        record = {
            "solver": "acados",
            "mode": mode,
            "tau": tau,
            "guess_source": guess_source,
            "forward_hessian": forward_hessian,
            "termination": termination,
            "horizon": HORIZON,
            "prediction_time": PREDICTION_TIME,
            "speed_limit": SPEED_LIMIT,
            "steer_limit": STEER_LIMIT,
            "wheelbase": WHEELBASE,
            "obstacle_scale": OBSTACLE_SCALE,
            "terminal_weight": float(THETA_PREFIX[4]),
            "forward_tolerance": 1e-4,
            "nlp_max_iterations": 10000,
            "qp_max_iterations": 500,
            "success": bool(all(item == 0 for item in statuses) and np.all(np.isfinite(solution))),
            "status": status,
            "statuses": statuses,
            "backward_status": backward_status,
            "repetitions": repetitions,
            "forward_time_sec": float(np.mean(forward_times)),
            "forward_time_std_sec": float(np.std(forward_times, ddof=1)) if repetitions > 1 else 0.0,
            "backward_time_sec": float(np.mean(backward_times)),
            "backward_time_std_sec": float(np.std(backward_times, ddof=1)) if repetitions > 1 else 0.0,
            "imitation_loss": 0.5 * float(np.sum(residual**2)),
            "gradient_norm": float(np.linalg.norm(grad)),
            "solution": solution.tolist(),
            "acados_stats": acados_stats(forward),
        }
        record.update(numerical_metrics(problem, solution))
        record["success"] = bool(
            record["success"]
            and record["geometric_violation"] <= 1e-4
            and record["passed_obstacle"]
        )
        return record
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def write_record(outdir: Path, record: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    tau_tag = "none" if record["mode"].startswith("product") else f"{record['tau']:g}".replace(".", "p")
    if record["solver"] == "acados":
        guess_tag = f"_{record['guess_source']}_{record.get('forward_hessian', 'gn')}"
    else:
        guess_tag = "" if record.get("guess_source", "default") == "default" else f"_{record['guess_source']}"
    path = outdir / f"{record['solver']}_{record['mode']}_tau_{tau_tag}{guess_tag}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    concise = {key: value for key, value in record.items() if key not in {"states", "solution", "constraint_values", "acados_stats"}}
    print(json.dumps(concise, indent=2))


def summarize(outdir: Path):
    rows = []
    paths = sorted(outdir.glob("lapanda_*.json")) + sorted(outdir.glob("acados_*.json"))
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "solver" not in data:
            continue
        stats = data.get("acados_stats", {})
        residuals = stats.get("residuals", [float("nan")] * 4)
        rows.append(
            {
                "file": path.name,
                "solver": data["solver"],
                "mode": data["mode"],
                "tau": data["tau"],
                "guess_source": data.get("guess_source", "default"),
                "forward_hessian": data.get("forward_hessian", "n/a"),
                "success": data["success"],
                "status": data.get("status", 0 if data["success"] else -1),
                "sqp_iterations": stats.get("sqp_iter", -1),
                "forward_time_ms": 1e3 * data.get("forward_time_sec", float("nan")),
                "backward_time_ms": 1e3 * data.get("backward_time_sec", float("nan")),
                "imitation_loss": data.get("imitation_loss", float("nan")),
                "geometric_violation": data.get("geometric_violation", float("nan")),
                "constraint_min": data.get("constraint_min", float("nan")),
                "jacobian_rank_1e8": data.get("jacobian_rank_1e8", -1),
                "jacobian_row_norm_min": data.get("jacobian_row_norm_min", float("nan")),
                "jacobian_sigma_min": data.get("jacobian_sigma_min", float("nan")),
                "stationarity_residual": residuals[0],
                "equality_residual": residuals[1],
                "inequality_residual": residuals[2],
                "complementarity_residual": residuals[3],
                "exception": data.get("exception", ""),
            }
        )
    (outdir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    canonical = []
    for row in rows:
        filename = row["file"]
        keep_lapanda = row["solver"] == "lapanda"
        keep_acados_default = filename.endswith("_default_gn.json") or filename.endswith("_default.json")
        keep_acados_exact = filename.endswith("_lapanda_exact.json")
        if keep_lapanda or keep_acados_default or keep_acados_exact:
            canonical.append(row)
    with (outdir / "canonical_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(canonical[0]))
        writer.writeheader()
        writer.writerows(canonical)
    print(json.dumps(rows, indent=2))


def main():
    global HORIZON, DT, PREDICTION_TIME, WHEELBASE, STEER_LIMIT, SPEED_LIMIT, OBSTACLE_SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["lapanda", "acados", "summarize"], required=True)
    parser.add_argument("--modes", default="product2,product1,smooth")
    parser.add_argument("--taus", type=parse_float_list, default=DEFAULT_TAUS)
    parser.add_argument("--acados-guesses", default="default")
    parser.add_argument("--acados-hessians", default="gn")
    parser.add_argument("--acados-termination", choices=["kkt", "feasibility"], default="kkt")
    parser.add_argument(
        "--lapanda-guess",
        choices=["default", "zero", "handcrafted", "acados", "continuation"],
        default="default",
    )
    parser.add_argument("--lapanda-initial-penalty", type=float, default=1e4)
    parser.add_argument("--lapanda-initial-guess-file", type=Path, default=None)
    parser.add_argument("--lapanda-inner-tol", type=float, default=1e-4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--prediction-time", type=float, default=PREDICTION_TIME)
    parser.add_argument("--wheelbase", type=float, default=WHEELBASE)
    parser.add_argument("--steer-limit", type=float, default=STEER_LIMIT)
    parser.add_argument("--speed-limit", type=float, default=SPEED_LIMIT)
    parser.add_argument("--obstacle-scale", type=float, default=OBSTACLE_SCALE)
    parser.add_argument("--terminal-weight", type=float, default=float(THETA_PREFIX[4]))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    HORIZON = args.horizon
    PREDICTION_TIME = args.prediction_time
    DT = PREDICTION_TIME / HORIZON
    WHEELBASE = args.wheelbase
    STEER_LIMIT = args.steer_limit
    SPEED_LIMIT = args.speed_limit
    OBSTACLE_SCALE = args.obstacle_scale
    THETA_PREFIX[4] = args.terminal_weight
    if args.solver == "summarize":
        summarize(args.outdir)
        return
    for mode in (item.strip() for item in args.modes.split(",") if item.strip()):
        taus = args.taus if mode in {"smooth", "smoothsqrt", "scholtes", "superellipse"} else (0.0,)
        for tau in taus:
            guesses = (None,) if args.solver == "lapanda" else tuple(
                item.strip() for item in args.acados_guesses.split(",") if item.strip()
            )
            for guess_source in guesses:
                hessians = (None,) if args.solver == "lapanda" else tuple(
                    item.strip() for item in args.acados_hessians.split(",") if item.strip()
                )
                for forward_hessian in hessians:
                    try:
                        record = (
                            run_lapanda(
                                mode,
                                tau,
                                args.outdir,
                                args.lapanda_guess,
                                args.repetitions,
                                args.lapanda_initial_penalty,
                                args.lapanda_initial_guess_file,
                                args.lapanda_inner_tol,
                            )
                            if args.solver == "lapanda"
                            else run_acados(
                                mode,
                                tau,
                                args.outdir,
                                guess_source,
                                forward_hessian,
                                args.repetitions,
                                args.acados_termination,
                            )
                        )
                    except Exception as exc:
                        record = {
                            "solver": args.solver,
                            "mode": mode,
                            "tau": tau,
                            "guess_source": guess_source,
                            "forward_hessian": forward_hessian,
                            "success": False,
                            "exception": repr(exc),
                        }
                    write_record(args.outdir, record)


if __name__ == "__main__":
    main()
