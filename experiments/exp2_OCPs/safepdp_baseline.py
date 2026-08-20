"""Unified SafePDP baselines for Exp.2 OCPs."""

from __future__ import annotations

import argparse
import contextlib
import ast
import os
import sys
from pathlib import Path

import casadi as ca
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
SAFEPDP_ROOT = REPO_ROOT / "external" / "Safe-PDP"
if str(SAFEPDP_ROOT) not in sys.path:
    sys.path.insert(0, str(SAFEPDP_ROOT))

from SafePDP import SafePDP  # noqa: E402

from experiments.common import append_csv, fresh_output_path, profiled_call, write_json  # noqa: E402
from experiments.exp2_OCPs.config import SAFEPDP_DEFAULTS  # noqa: E402
from experiments.exp2_OCPs.problem_bank import STATE_IMITATION_WEIGHT, build_problem  # noqa: E402
from experiments.exp2_OCPs.utils import (  # noqa: E402
    apply_theta_overrides,
    clip_theta,
    default_teacher_archive_path,
    load_snapshot_archive,
    make_variable_from_state,
    seeded_theta_initial,
    shift_control,
)


CARTPOLE_DT = 0.05
CARTPOLE_U_LOWER = -2.0
CARTPOLE_U_UPPER = 6.0
CARTPOLE_CONTROL_SCALE = max(abs(CARTPOLE_U_LOWER), abs(CARTPOLE_U_UPPER))
CARTPOLE_ENERGY_BUDGET = 70.0
QUADROTOR_FLOOR_BASE = 0.43
QUADROTOR_CEILING_HEIGHT = 1.20
QUADROTOR_ANGLE_BASE = 0.38
ARM_OBSTACLE_CENTER = np.array([1.30, 0.30], dtype=float)
ARM_OBSTACLE_RADIUS_BASE = 0.29
ARM_LINK_LENGTHS = (1.0, 0.8)


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def arm_fk(q0, q1):
    l1, l2 = ARM_LINK_LENGTHS
    return ca.vertcat(l1 * ca.cos(q0) + l2 * ca.cos(q0 + q1), l1 * ca.sin(q0) + l2 * ca.sin(q0 + q1))


def cartpole_step(x, u, dt: float):
    cart_mass = 1.0
    pole_mass = 0.1
    pole_half_length = 0.5
    gravity = 9.81
    total_mass = cart_mass + pole_mass

    vel = x[1]
    ang = x[2]
    ang_vel = x[3]
    force = u[0]

    sin_ang = ca.sin(ang)
    cos_ang = ca.cos(ang)
    temp = (force + pole_mass * pole_half_length * ang_vel**2 * sin_ang) / total_mass
    ang_acc = (gravity * sin_ang - cos_ang * temp) / (
        pole_half_length * (4.0 / 3.0 - pole_mass * cos_ang**2 / total_mass)
    )
    pos_acc = temp - pole_mass * pole_half_length * ang_acc * cos_ang / total_mass
    next_vel = vel + dt * pos_acc
    next_ang_vel = ang_vel + dt * ang_acc
    return ca.vertcat(
        x[0] + dt * next_vel,
        next_vel,
        ang + dt * next_ang_vel,
        next_ang_vel,
    )


def build_cartpole_coc(instance):
    """Build SafePDP COCsys for cartpole with an augmented energy state."""

    state = ca.SX.sym("x", instance.nx + 1)
    control = ca.SX.sym("u", instance.nu)
    theta = ca.SX.sym("theta", instance.theta_true.size)
    x_cart = state[: instance.nx]
    energy = state[instance.nx]

    q_pos, q_ang, q_vel, q_ang_vel, r_force = [theta[i] for i in range(5)]
    x_next = cartpole_step(x_cart, control, instance.dt)
    dyn = ca.vertcat(x_next, energy + control[0] ** 2)

    target = ca.DM(instance.x_target)
    err = x_cart - target
    path_cost = (
        q_pos * err[0] ** 2
        + q_vel * err[1] ** 2
        + q_ang * err[2] ** 2
        + q_ang_vel * err[3] ** 2
        + r_force * control[0] ** 2
    )
    terminal_err = x_cart - target
    final_cost = 10.0 * (
        q_pos * terminal_err[0] ** 2
        + q_vel * terminal_err[1] ** 2
        + q_ang * terminal_err[2] ** 2
        + q_ang_vel * terminal_err[3] ** 2
    )
    final_inequ = energy - CARTPOLE_ENERGY_BUDGET

    coc = SafePDP.COCsys(project_name=f"{instance.name}_safepdp")
    coc.setAuxvarVariable(theta)
    coc.setStateVariable(state)
    coc.setControlVariable(control, control_lb=[CARTPOLE_U_LOWER], control_ub=[CARTPOLE_U_UPPER])
    coc.setDyn(dyn)
    coc.setPathCost(path_cost)
    coc.setFinalCost(final_cost)
    coc.setPathInequCstr(None)
    coc.setPathEquCstr(None)
    coc.setFinalInequCstr(final_inequ)
    coc.setFinalEquCstr(None)
    coc.diffCPMP()
    return coc


def build_quadrotor_coc(instance):
    """Build SafePDP COCsys for the planar quadrotor OCP.

    Path inequalities are written in SafePDP convention g(x, u) <= 0:
    floor_height - z_next <= 0, z_next - ceiling_height <= 0,
    x_next - target_x <= 0,
    angle_next - angle_max <= 0, and -angle_next - angle_max <= 0.
    """

    x = ca.SX.sym("x", instance.nx)
    u = ca.SX.sym("u", instance.nu)
    theta = ca.SX.sym("theta", instance.theta_true.size)
    target = ca.DM(instance.x_target)

    q_pos, q_vel, q_angle, q_omega, r_thrust, r_torque, _unused_margin = [theta[i] for i in range(7)]
    gravity = 9.81
    inertia_inv = 4.0
    floor_height = QUADROTOR_FLOOR_BASE
    ceiling_height = QUADROTOR_CEILING_HEIGHT
    angle_max = 0.34

    err = x - target
    hover_thrust = gravity
    path_cost = (
        q_pos * (err[0] ** 2 + err[1] ** 2)
        + q_vel * (err[2] ** 2 + err[3] ** 2)
        + q_angle * err[4] ** 2
        + q_omega * err[5] ** 2
        + r_thrust * (u[0] - hover_thrust) ** 2
        + r_torque * u[1] ** 2
    )
    acc_x = -u[0] * ca.sin(x[4])
    acc_z = u[0] * ca.cos(x[4]) - gravity
    omega_next = x[5] + instance.dt * inertia_inv * u[1]
    angle_next = x[4] + instance.dt * omega_next
    vx_next = x[2] + instance.dt * acc_x
    vz_next = x[3] + instance.dt * acc_z
    dyn = ca.vertcat(
        x[0] + instance.dt * vx_next,
        x[1] + instance.dt * vz_next,
        vx_next,
        vz_next,
        angle_next,
        omega_next,
    )
    path_inequ = ca.vertcat(
        floor_height - dyn[1],
        dyn[1] - ceiling_height,
        dyn[0] - target[0],
        dyn[4] - angle_max,
        -dyn[4] - angle_max,
    )
    terminal_err = x - target
    final_cost = 15.0 * (
        q_pos * (terminal_err[0] ** 2 + terminal_err[1] ** 2)
        + q_vel * (terminal_err[2] ** 2 + terminal_err[3] ** 2)
        + q_angle * terminal_err[4] ** 2
        + q_omega * terminal_err[5] ** 2
    )
    return build_coc(instance, x, u, theta, dyn, path_cost, final_cost, path_inequ)


def build_arm_coc(instance):
    """Build SafePDP COCsys for the two-link robot-arm OCP.

    Path inequalities are written in SafePDP convention g(x, u) <= 0:
    obstacle_radius^2 - distance_to_obstacle^2 <= 0 and joint box
    upper/lower inequalities for both joints.
    """

    x = ca.SX.sym("x", instance.nx)
    u = ca.SX.sym("u", instance.nu)
    theta = ca.SX.sym("theta", instance.theta_true.size)
    target = ca.DM(instance.x_target)

    q_joint, q_ee, r_vel1, r_vel2, obstacle_margin = [theta[i] for i in range(5)]
    obstacle_center = ARM_OBSTACLE_CENTER
    obstacle_radius = ARM_OBSTACLE_RADIUS_BASE + obstacle_margin
    q_lower = ca.vertcat(-1.10, -0.65)
    q_upper = ca.vertcat(0.75, 1.28)
    target_ee = arm_fk(target[0], target[1])
    ee = arm_fk(x[0], x[1])
    joint_err = x - target
    ee_err = ee - target_ee
    path_cost = q_joint * ca.dot(joint_err, joint_err) + q_ee * ca.dot(ee_err, ee_err)
    path_cost += r_vel1 * u[0] ** 2 + r_vel2 * u[1] ** 2
    dyn = x + instance.dt * ca.vertcat(u[0], u[1])
    ee_next = arm_fk(dyn[0], dyn[1])
    dist2 = (ee_next[0] - obstacle_center[0]) ** 2 + (ee_next[1] - obstacle_center[1]) ** 2
    path_inequ = ca.vertcat(
        obstacle_radius**2 - dist2,
        dyn[0] - q_upper[0],
        q_lower[0] - dyn[0],
        dyn[1] - q_upper[1],
        q_lower[1] - dyn[1],
    )
    terminal_ee = arm_fk(x[0], x[1])
    terminal_joint_err = x - target
    terminal_ee_err = terminal_ee - target_ee
    final_cost = 20.0 * (
        q_joint * ca.dot(terminal_joint_err, terminal_joint_err) + q_ee * ca.dot(terminal_ee_err, terminal_ee_err)
    )
    return build_coc(instance, x, u, theta, dyn, path_cost, final_cost, path_inequ)


def build_coc(instance, state, control, theta, dyn, path_cost, final_cost, path_inequ):
    lower = np.asarray(instance.problem.box_lower, dtype=float)[: instance.nu].tolist()
    upper = np.asarray(instance.problem.box_upper, dtype=float)[: instance.nu].tolist()
    coc = SafePDP.COCsys(project_name=f"{instance.name}_safepdp")
    coc.setAuxvarVariable(theta)
    coc.setStateVariable(state)
    coc.setControlVariable(control, control_lb=lower, control_ub=upper)
    coc.setDyn(dyn)
    coc.setPathCost(path_cost)
    coc.setFinalCost(final_cost)
    coc.setPathInequCstr(path_inequ)
    coc.setPathEquCstr(None)
    coc.setFinalInequCstr(None)
    coc.setFinalEquCstr(None)
    coc.diffCPMP()
    return coc


def build_safepdp_coc(instance):
    if instance.name == "cartpole":
        return build_cartpole_coc(instance)
    if instance.name == "planar_quadrotor":
        return build_quadrotor_coc(instance)
    if instance.name == "two_link_arm":
        return build_arm_coc(instance)
    raise ValueError(instance.name)


def solve_coc(coc, horizon, init_state, theta, control_traj0=None):
    with suppress_native_output(True):
        return coc.ocSolver(
            horizon=horizon,
            init_state=init_state,
            auxvar_value=np.asarray(theta, dtype=float),
            control_traj0=control_traj0,
        )


def solve_barrier_oc(coc, horizon, init_state, theta, control_traj0=None):
    with suppress_native_output(True):
        return coc.solveBarrierOC(
            horizon=horizon,
            init_state=init_state,
            auxvar_value=np.asarray(theta, dtype=float),
            control_traj0=control_traj0,
        )


def solve_forward(args, coc, horizon, init_state, theta, control_traj0=None):
    if args.sensitivity_mode == "barrier":
        return solve_barrier_oc(coc, horizon, init_state, theta, control_traj0)
    return solve_coc(coc, horizon, init_state, theta, control_traj0)


def _state_matrix_for_loss(instance, states):
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    return states[:, : instance.nx]


def imitation_loss_and_grad(instance, demo_u, demo_states, traj, aux_sol):
    demo_u = np.asarray(demo_u, dtype=float).reshape(-1)
    control_matrix = np.asarray(traj["control_traj_opt"], dtype=float)
    if control_matrix.ndim == 1:
        control_matrix = control_matrix.reshape(-1, 1)
    control = control_matrix.reshape(-1)
    dldu = control - demo_u
    loss = 0.5 * float(np.dot(dldu, dldu))
    grad = np.zeros(np.asarray(traj["auxvar_value"], dtype=float).size)
    n_control = control_matrix.shape[1]
    for t, dudtheta in enumerate(aux_sol["control_traj_opt"]):
        grad += (
            np.asarray(dldu[t * n_control : (t + 1) * n_control]).reshape(1, -1)
            @ np.asarray(dudtheta, dtype=float)
        ).reshape(-1)
    if demo_states is not None:
        state_matrix = _state_matrix_for_loss(instance, traj["state_traj_opt"])
        demo_state_matrix = _state_matrix_for_loss(instance, demo_states)
        state_error = state_matrix[1 : control_matrix.shape[0] + 1] - demo_state_matrix[1 : control_matrix.shape[0] + 1]
        loss += 0.5 * STATE_IMITATION_WEIGHT * float(np.sum(state_error * state_error))
        for t, dxdtheta in enumerate(aux_sol["state_traj_opt"][1 : control_matrix.shape[0] + 1]):
            grad += STATE_IMITATION_WEIGHT * (
                state_error[t].reshape(1, -1) @ np.asarray(dxdtheta, dtype=float)[: instance.nx, :]
            ).reshape(-1)
    return loss, grad.reshape(-1)


def imitation_loss_only(instance, demo_u, demo_states, traj):
    demo_u = np.asarray(demo_u, dtype=float).reshape(-1)
    control_matrix = np.asarray(traj["control_traj_opt"], dtype=float)
    if control_matrix.ndim == 1:
        control_matrix = control_matrix.reshape(-1, 1)
    dldu = control_matrix.reshape(-1) - demo_u
    loss = 0.5 * float(np.dot(dldu, dldu))
    if demo_states is not None:
        state_matrix = _state_matrix_for_loss(instance, traj["state_traj_opt"])
        demo_state_matrix = _state_matrix_for_loss(instance, demo_states)
        state_error = state_matrix[1 : control_matrix.shape[0] + 1] - demo_state_matrix[1 : control_matrix.shape[0] + 1]
        loss += 0.5 * STATE_IMITATION_WEIGHT * float(np.sum(state_error * state_error))
    return loss


def align_traj_states_to_problem(instance, traj, theta, state, demo_u):
    """Use the shared OCP rollout for physical closed-loop state updates."""

    if instance.name == "cartpole":
        return traj
    solution = np.asarray(traj["control_traj_opt"], dtype=float).reshape(-1)
    variable = make_variable_from_state(instance, state, demo_u)
    aligned = dict(traj)
    aligned["state_traj_opt"] = np.asarray(instance.rollout_fun(solution, theta, variable), dtype=float)
    return aligned


def safepdp_backward(instance, coc, clqr, traj, demo_u, demo_states, threshold):
    auxsys = coc.getAuxSys(opt_sol=traj, threshold=threshold)
    clqr.auxsys2Eqctlqr(auxsys=auxsys)
    aux_sol = clqr.eqctlqrSolver(threshold=threshold)
    return imitation_loss_and_grad(instance, demo_u, demo_states, traj, aux_sol)


def safepdp_barrier_backward(instance, coc, traj, demo_u, demo_states, gamma):
    aux_sol = coc.auxSysBarrierOC(opt_sol=traj)
    return imitation_loss_and_grad(instance, demo_u, demo_states, traj, aux_sol)


def fallback_backward(args, instance, coc, clqr, traj, demo_u, demo_states):
    if "v_path" in traj:
        return (*safepdp_backward(instance, coc, clqr, traj, demo_u, demo_states, args.tol), True)
    grad = np.zeros(np.asarray(traj["auxvar_value"], dtype=float).size)
    return imitation_loss_only(instance, demo_u, demo_states, traj), grad, True


def safe_backward(args, instance, coc, clqr, traj, demo_u, demo_states=None):
    if args.sensitivity_mode != "barrier":
        loss, grad = safepdp_backward(instance, coc, clqr, traj, demo_u, demo_states, args.tol)
        return loss, grad, False
    try:
        loss, grad = safepdp_barrier_backward(instance, coc, traj, demo_u, demo_states, args.barrier_gamma)
    except Exception:
        return fallback_backward(args, instance, coc, clqr, traj, demo_u, demo_states)
    if np.isfinite(loss) and np.all(np.isfinite(grad)):
        return loss, grad, False
    return fallback_backward(args, instance, coc, clqr, traj, demo_u, demo_states)


def rollout_constraint_violation(instance, traj, theta):
    control = np.asarray(traj["control_traj_opt"], dtype=float).reshape(-1)
    states = np.asarray(traj["state_traj_opt"], dtype=float)
    if instance.name == "cartpole":
        return float(max(states[-1, instance.nx] - CARTPOLE_ENERGY_BUDGET, 0.0))
    demo = np.zeros(instance.horizon * instance.nu)
    variable = make_variable_from_state(instance, states[0, : instance.nx], demo)
    c = np.asarray(instance.constraint_fun(control, theta, variable), dtype=float).reshape(-1)
    return float(np.max(np.maximum(c, 0.0)))


def cartpole_constraint_margin_trace(states):
    return CARTPOLE_ENERGY_BUDGET - np.asarray(states, dtype=float)[:, 4], "energy budget margin"


def quadrotor_constraint_margin_trace(instance, states, theta):
    """Return min positive safety margin for quadrotor path constraints.

    The plotted margin is min(z-floor, ceiling-z, progress-to-target,
    angle_max-angle, angle_max+angle).  Positive means feasible; zero means
    the active boundary; negative means violation.
    """

    target = np.asarray(instance.x_target, dtype=float)
    floor_height = QUADROTOR_FLOOR_BASE
    ceiling_height = QUADROTOR_CEILING_HEIGHT
    angle_max = 0.34
    values = []
    for x in states[1:]:
        values.append(
            np.asarray(
                [
                    x[1] - floor_height,
                    ceiling_height - x[1],
                    target[0] - x[0],
                    angle_max - x[4],
                    angle_max + x[4],
                ],
                dtype=float,
            )
        )
    return np.min(np.vstack(values), axis=1), "min height/progress/angle margin"


def arm_constraint_margin_trace(states, theta):
    """Return min positive margin for obstacle and joint-corridor constraints."""

    obstacle_radius = ARM_OBSTACLE_RADIUS_BASE + theta[4]
    q_lower = np.asarray([-1.10, -0.65], dtype=float)
    q_upper = np.asarray([0.75, 1.28], dtype=float)
    values = []
    for x in states[1:]:
        q0, q1 = x
        ee = np.asarray(
            [
                np.cos(q0) + ARM_LINK_LENGTHS[1] * np.cos(q0 + q1),
                np.sin(q0) + ARM_LINK_LENGTHS[1] * np.sin(q0 + q1),
            ],
            dtype=float,
        )
        values.append(
            np.asarray(
                [
                    np.linalg.norm(ee - ARM_OBSTACLE_CENTER) - obstacle_radius,
                    q_upper[0] - q0,
                    q0 - q_lower[0],
                    q_upper[1] - q1,
                    q1 - q_lower[1],
                ],
                dtype=float,
            )
        )
    return np.min(np.vstack(values), axis=1), "min obstacle/joint-corridor margin"


def constraint_margin_trace(instance, traj, theta):
    states = np.asarray(traj["state_traj_opt"], dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if instance.name == "cartpole":
        return cartpole_constraint_margin_trace(states)
    if instance.name == "planar_quadrotor":
        return quadrotor_constraint_margin_trace(instance, states, theta)
    if instance.name == "two_link_arm":
        return arm_constraint_margin_trace(states, theta)
    raise ValueError(instance.name)


def snapshot_from_traj(epoch, instance, traj, theta, sample_index=None):
    states = np.asarray(traj["state_traj_opt"], dtype=float)
    controls = np.asarray(traj["control_traj_opt"], dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    margin, label = constraint_margin_trace(instance, traj, theta)
    physical_states = states[:, : instance.nx]
    if instance.name == "cartpole":
        control_scale = CARTPOLE_CONTROL_SCALE
    else:
        upper = np.asarray(instance.problem.box_upper, dtype=float)[: instance.nu]
        lower = np.asarray(instance.problem.box_lower, dtype=float)[: instance.nu]
        control_scale = float(max(np.max(np.abs(upper)), np.max(np.abs(lower)), 1.0))
    return {
        "epoch": int(epoch),
        "sample_index": None if sample_index is None else int(sample_index),
        "theta": theta.copy(),
        "states": physical_states.copy(),
        "controls": controls.copy(),
        "constraint_margin": margin.copy(),
        "control_scale": control_scale,
        "margin_label": label,
    }


def output_paths(model):
    if model == "cartpole":
        prefix = "exp2_cartpole"
    elif model == "quadrotor":
        prefix = "exp2_quadrotor"
    elif model == "robot_arm":
        prefix = "exp2_robot_arm"
    else:
        raise ValueError(model)
    exp_dir = REPO_ROOT / "experiments" / "exp2_OCPs" / model
    return exp_dir / "baselines" / "results", prefix


def build_stats_config(build_stats):
    config = {
        "problem_build_time_sec": build_stats.get("problem_build_time_sec", np.nan),
        "solver_build_time_sec": build_stats.get("solver_build_time_sec", np.nan),
        "build_time_sec": build_stats.get("elapsed_sec", np.nan),
        "measure_memory": bool(build_stats.get("measure_memory", False)),
    }
    optional_keys = (
        ("problem_build_rss_peak_delta_mb", "problem_build_rss_peak_delta_mb"),
        ("problem_build_rss_tree_peak_delta_mb", "problem_build_rss_tree_peak_delta_mb"),
        ("solver_build_rss_peak_delta_mb", "solver_build_rss_peak_delta_mb"),
        ("solver_build_rss_tree_peak_delta_mb", "solver_build_rss_tree_peak_delta_mb"),
        ("build_rss_delta_mb", "rss_peak_delta_mb"),
        ("build_rss_tree_delta_mb", "rss_tree_peak_delta_mb"),
    )
    for out_key, source_key in optional_keys:
        if source_key in build_stats:
            config[out_key] = build_stats[source_key]
    return config


def run_imitation(args, instance, coc, out_dir, prefix, build_stats=None):
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    clqr = SafePDP.EQCLQR()
    teacher_path = (
        Path(args.teacher_source)
        if args.teacher_source
        else default_teacher_archive_path(args.model, args.horizon)
    )
    teacher_snapshots = load_snapshot_archive(teacher_path)
    if not teacher_snapshots:
        raise FileNotFoundError(f"missing teacher rollout snapshots: {teacher_path}")
    if memory_only:
        teacher_snapshots = teacher_snapshots[:1]
    effective_epochs = 0 if memory_only else args.epochs
    horizon = int(np.asarray(teacher_snapshots[0]["controls"], dtype=float).reshape(-1).size // instance.nu)

    result_name = f"{prefix}_imitation_safepdp.csv"
    if args.output_tag:
        result_name = f"{prefix}_imitation_safepdp_{args.output_tag}.csv"
    path = fresh_output_path(out_dir / result_name)
    config_payload = {
        "model": instance.name,
        "horizon": horizon,
        "epochs": args.epochs,
        "tol": args.tol,
        "learning_rate": args.learning_rate,
        "sensitivity_mode": args.sensitivity_mode,
        "barrier_gamma": args.barrier_gamma if args.sensitivity_mode == "barrier" else None,
        "theta_true": instance.theta_true.tolist(),
        "theta_initial": seeded_theta_initial(instance, args.theta_init, args.seed).tolist(),
        "theta_init_mode": args.theta_init,
        "seed": args.seed,
        "teacher_rollout_samples": len(teacher_snapshots),
        "teacher_rollout_source": str(teacher_path),
        "measure_memory": measure_memory,
        "memory_only": memory_only,
    }
    if build_stats is not None:
        config_payload.update(build_stats_config(build_stats))
    write_json(
        out_dir / f"config_{path.stem}.json",
        config_payload,
    )
    if args.start_theta:
        theta = np.asarray(ast.literal_eval(args.start_theta), dtype=float).reshape(-1)
    else:
        theta = seeded_theta_initial(instance, args.theta_init, args.seed)
    rows = []
    snapshots = []
    u_warm_by_sample = [
        np.asarray(snapshot["controls"], dtype=float).reshape(-1).copy()
        for snapshot in teacher_snapshots
    ]
    for epoch in range(effective_epochs + 1):
        sample_losses = []
        sample_grads = []
        sample_forward_times = []
        sample_backward_times = []
        sample_violations = []
        sample_rss = []
        sample_fallbacks = []
        for sample_index, snapshot in enumerate(teacher_snapshots):
            init_state = np.asarray(snapshot["states"], dtype=float)[0]
            if instance.name == "cartpole":
                init_state = np.concatenate([init_state[: instance.nx], np.zeros(1)])
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            forward = profiled_call(
                lambda: solve_forward(args, coc, horizon, init_state, theta, u_warm_by_sample[sample_index]),
                measure_memory=measure_memory,
            )
            traj = forward["value"]
            u_warm_by_sample[sample_index] = np.asarray(traj["control_traj_opt"], dtype=float).reshape(-1).copy()
            backward = profiled_call(
                lambda: safe_backward(args, instance, coc, clqr, traj, demo_u, demo_states),
                measure_memory=measure_memory,
            )
            loss, grad, fallback = backward["value"]
            sample_losses.append(float(loss))
            sample_grads.append(np.asarray(grad, dtype=float))
            sample_forward_times.append(float(forward["elapsed_sec"]))
            sample_backward_times.append(float(backward["elapsed_sec"]))
            sample_violations.append(float(rollout_constraint_violation(instance, traj, theta)))
            if measure_memory:
                sample_rss.append(float(max(forward["rss_peak_delta_mb"], backward["rss_peak_delta_mb"])))
            sample_fallbacks.append(int(fallback))
            snapshots.append(snapshot_from_traj(epoch, instance, traj, theta, sample_index))
        mean_grad = np.mean(np.asarray(sample_grads, dtype=float), axis=0)
        loss_mean = float(np.mean(sample_losses))
        loss_std = float(np.std(sample_losses, ddof=1)) if len(sample_losses) > 1 else 0.0
        row = {
            "epoch": epoch,
            "seed": args.seed if args.seed is not None else "",
            "loss": loss_mean,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "loss_sem": loss_std / np.sqrt(len(sample_losses)) if sample_losses else 0.0,
            "theta": theta.tolist(),
            "grad_theta": mean_grad.tolist(),
            "grad_norm": float(np.linalg.norm(mean_grad)),
            "forward_time_sec": float(np.mean(sample_forward_times)),
            "backward_time_sec": float(np.mean(sample_backward_times)),
            "warm_start": True,
            "constraint_violation_inf": float(np.mean(sample_violations)),
            "barrier_fallbacks": int(np.sum(sample_fallbacks)),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = float(np.mean(sample_rss))
        rows.append(row)
        append_csv(path, [row])
        msg = (
            f"{args.model} SafePDP imitation epoch={epoch:02d} loss={row['loss']:.6e} "
            f"forward={row['forward_time_sec']:.6f}s backward={row['backward_time_sec']:.6f}s "
        )
        if measure_memory:
            msg += f"mem={row['rss_peak_delta_mb']:.3f}MB "
        msg += f"viol={row['constraint_violation_inf']:.2e}"
        print(msg)
        if epoch == effective_epochs:
            break
        theta = clip_theta(instance, theta - args.learning_rate * mean_grad / (np.linalg.norm(mean_grad) + 1e-9))
    snapshot_path = fresh_output_path(path.with_suffix(".snapshots.npz"))
    np.savez(
        snapshot_path,
        snapshots=np.asarray(snapshots, dtype=object),
        allow_pickle=True,
    )
    print(f"wrote {snapshot_path}")
    print(f"{args.model} SafePDP imitation mean_forward={np.mean([r['forward_time_sec'] for r in rows]):.6f}s")
    print(f"{args.model} SafePDP imitation mean_backward={np.mean([r['backward_time_sec'] for r in rows]):.6f}s")
    return path


def run_closed_loop_imitation(args, instance, coc, out_dir, prefix, build_stats=None):
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    clqr = SafePDP.EQCLQR()
    teacher_path = (
        Path(args.teacher_source)
        if args.teacher_source
        else default_teacher_archive_path(args.model, args.horizon)
    )
    teacher_snapshots = load_snapshot_archive(teacher_path)
    if not teacher_snapshots:
        raise FileNotFoundError(f"missing teacher rollout snapshots: {teacher_path}")
    teacher_snapshots = teacher_snapshots[: args.steps]
    if memory_only:
        teacher_snapshots = teacher_snapshots[:1]
    effective_epochs = 0 if memory_only else args.epochs
    horizon = int(np.asarray(teacher_snapshots[0]["controls"], dtype=float).reshape(-1).size // instance.nu)
    if args.start_theta:
        theta = np.asarray(ast.literal_eval(args.start_theta), dtype=float).reshape(-1)
    else:
        theta = seeded_theta_initial(instance, args.theta_init, args.seed)

    result_name = f"{prefix}_closed_loop_train_safepdp.csv"
    if args.output_tag:
        result_name = f"{prefix}_closed_loop_train_safepdp_{args.output_tag}.csv"
    path = fresh_output_path(out_dir / result_name)
    config_payload = {
            "model": instance.name,
            "horizon": horizon,
            "epochs": args.epochs,
            "steps": len(teacher_snapshots),
            "tol": args.tol,
            "learning_rate": args.learning_rate,
            "sensitivity_mode": args.sensitivity_mode,
            "barrier_gamma": args.barrier_gamma if args.sensitivity_mode == "barrier" else None,
            "theta_true": instance.theta_true.tolist(),
            "theta_initial": theta.tolist(),
            "theta_init_mode": args.theta_init,
            "epoch_offset": args.epoch_offset,
            "seed": args.seed,
            "teacher_rollout_source": str(teacher_path),
            "measure_memory": measure_memory,
            "memory_only": memory_only,
    }
    if build_stats is not None:
        config_payload.update(build_stats_config(build_stats))
    write_json(out_dir / f"config_{Path(result_name).stem}.json", config_payload)

    first_teacher_states = np.asarray(teacher_snapshots[0]["states"], dtype=float)
    initial_state = first_teacher_states[0, : instance.nx].reshape(instance.nx)

    for epoch in range(effective_epochs + 1):
        state = initial_state.copy()
        u_warm = np.asarray(teacher_snapshots[0]["controls"], dtype=float).reshape(-1).copy()
        trajectory = [state.copy()]
        sample_losses = []
        sample_grads = []
        sample_forward_times = []
        sample_backward_times = []
        sample_violations = []
        sample_rss = []
        sample_fallbacks = []

        for step, snapshot in enumerate(teacher_snapshots):
            init_state = state
            if instance.name == "cartpole":
                init_state = np.concatenate([state[: instance.nx], np.zeros(1)])
            demo_u = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
            demo_states = np.asarray(snapshot["states"], dtype=float)
            forward = profiled_call(
                lambda: solve_forward(args, coc, horizon, init_state, theta, u_warm),
                measure_memory=measure_memory,
            )
            traj = forward["value"]
            solution = np.asarray(traj["control_traj_opt"], dtype=float).reshape(-1).copy()
            backward = profiled_call(
                lambda: safe_backward(args, instance, coc, clqr, traj, demo_u, demo_states),
                measure_memory=measure_memory,
            )
            loss, grad, fallback = backward["value"]
            aligned_traj = align_traj_states_to_problem(instance, traj, theta, state, demo_u)
            states = np.asarray(aligned_traj["state_traj_opt"], dtype=float)

            sample_losses.append(float(loss))
            sample_grads.append(np.asarray(grad, dtype=float))
            sample_forward_times.append(float(forward["elapsed_sec"]))
            sample_backward_times.append(float(backward["elapsed_sec"]))
            sample_violations.append(float(rollout_constraint_violation(instance, aligned_traj, theta)))
            if measure_memory:
                sample_rss.append(float(max(forward["rss_peak_delta_mb"], backward["rss_peak_delta_mb"])))
            sample_fallbacks.append(int(fallback))

            state = states[1, : instance.nx].copy()
            trajectory.append(state.copy())
            u_warm = shift_control(solution, instance.nu)

        mean_grad = np.mean(np.asarray(sample_grads, dtype=float), axis=0)
        loss_mean = float(np.mean(sample_losses))
        loss_std = float(np.std(sample_losses, ddof=1)) if len(sample_losses) > 1 else 0.0
        row = {
            "epoch": int(args.epoch_offset + epoch),
            "seed": args.seed if args.seed is not None else "",
            "loss": loss_mean,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "loss_sem": loss_std / np.sqrt(len(sample_losses)) if sample_losses else 0.0,
            "theta": theta.tolist(),
            "grad_theta": mean_grad.tolist(),
            "grad_norm": float(np.linalg.norm(mean_grad)),
            "forward_time_sec": float(np.mean(sample_forward_times)),
            "backward_time_sec": float(np.mean(sample_backward_times)),
            "warm_start": True,
            "constraint_violation_inf": float(np.mean(sample_violations)),
            "barrier_fallbacks": int(np.sum(sample_fallbacks)),
            "closed_loop_steps": len(teacher_snapshots),
            "final_state": trajectory[-1].tolist(),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = float(np.mean(sample_rss))
        append_csv(path, [row])
        epoch_label = int(args.epoch_offset + epoch)
        msg = (
            f"{args.model} SafePDP closed-loop epoch={epoch_label:03d} loss={row['loss']:.6e} "
            f"forward={row['forward_time_sec']:.6f}s backward={row['backward_time_sec']:.6f}s "
        )
        if measure_memory:
            msg += f"mem={row['rss_peak_delta_mb']:.3f}MB "
        msg += f"viol={row['constraint_violation_inf']:.2e}"
        print(msg)
        if epoch == effective_epochs:
            break
        theta = clip_theta(instance, theta - args.learning_rate * mean_grad / (np.linalg.norm(mean_grad) + 1e-9))
    print(f"{args.model} SafePDP closed-loop mean_forward={row['forward_time_sec']:.6f}s")
    print(f"{args.model} SafePDP closed-loop mean_backward={row['backward_time_sec']:.6f}s")
    return path


def run_mpc(args, instance, coc, out_dir, prefix, build_stats=None):
    memory_only = bool(getattr(args, "memory_only", False))
    measure_memory = memory_only
    clqr = SafePDP.EQCLQR()
    result_name = f"{prefix}_mpc_safepdp.csv"
    if args.output_tag:
        result_name = f"{prefix}_mpc_safepdp_{args.output_tag}.csv"
    path = fresh_output_path(out_dir / result_name)
    config_payload = {
            "model": instance.name,
            "horizon": instance.horizon,
            "steps": args.steps,
            "tol": args.tol,
            "sensitivity_mode": args.sensitivity_mode,
            "barrier_gamma": args.barrier_gamma if args.sensitivity_mode == "barrier" else None,
            "theta_true": instance.theta_true.tolist(),
            "measure_memory": measure_memory,
            "memory_only": memory_only,
    }
    if build_stats is not None:
        config_payload.update(build_stats_config(build_stats))
    write_json(out_dir / "config_mpc_safepdp.json", config_payload)
    state = instance.x_initial.copy()
    trajectory = [state.copy()]
    rows = []
    snapshots = []
    solution_warm = None
    effective_steps = 1 if memory_only else args.steps
    for step in range(effective_steps):
        solver_state = state
        if instance.name == "cartpole":
            solver_state = np.concatenate([state[: instance.nx], np.zeros(1)])
        forward = profiled_call(
            lambda: solve_forward(args, coc, instance.horizon, solver_state, instance.theta_true, solution_warm),
            measure_memory=measure_memory,
        )
        traj = forward["value"]
        solution = np.asarray(traj["control_traj_opt"], dtype=float).reshape(-1)
        backward = profiled_call(
            lambda: safe_backward(args, instance, coc, clqr, traj, solution),
            measure_memory=measure_memory,
        )
        _, _, fallback = backward["value"]
        states = np.asarray(traj["state_traj_opt"], dtype=float)
        row = {
            "step": step,
            "state": state.tolist(),
            "first_control": float(solution[0]) if instance.nu == 1 else solution[: instance.nu].tolist(),
            "objective": float(np.asarray(traj["cost"], dtype=float).reshape(-1)[0]),
            "constraint_violation_inf": rollout_constraint_violation(instance, traj, instance.theta_true),
            "forward_time_sec": forward["elapsed_sec"],
            "backward_time_sec": backward["elapsed_sec"],
            "barrier_fallback": int(fallback),
        }
        if measure_memory:
            row["rss_peak_delta_mb"] = max(forward["rss_peak_delta_mb"], backward["rss_peak_delta_mb"])
        snapshots.append(snapshot_from_traj(step, instance, traj, instance.theta_true))
        rows.append(row)
        append_csv(path, [row])
        solution_warm = shift_control(solution, instance.nu)
        state = states[1, : instance.nx].copy()
        trajectory.append(state.copy())
        msg = (
            f"{args.model} SafePDP MPC step={step:02d} forward={row['forward_time_sec']:.6f}s "
            f"backward={row['backward_time_sec']:.6f}s "
        )
        if measure_memory:
            msg += f"mem={row['rss_peak_delta_mb']:.3f}MB "
        msg += f"viol={row['constraint_violation_inf']:.2e}"
        print(msg)
    trajectory_path = fresh_output_path(path.with_suffix(".trajectory.npy"))
    np.save(trajectory_path, np.asarray(trajectory))
    snapshot_path = fresh_output_path(path.with_suffix(".mpc_snapshots.npz"))
    np.savez(
        snapshot_path,
        snapshots=np.asarray(snapshots, dtype=object),
        allow_pickle=True,
    )
    print(f"wrote {trajectory_path}")
    print(f"wrote {snapshot_path}")
    print(f"{args.model} SafePDP MPC mean_forward={np.mean([r['forward_time_sec'] for r in rows]):.6f}s")
    print(f"{args.model} SafePDP MPC mean_backward={np.mean([r['backward_time_sec'] for r in rows]):.6f}s")
    return path


def main(argv=None, model_override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cartpole", "quadrotor", "robot_arm"], required=model_override is None)
    parser.add_argument("--mode", choices=["imitation", "closed_loop", "mpc", "both"], default="both")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--theta-init", choices=["default", "uniform01"], default="default")
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--teacher-source", type=str, default="")
    parser.add_argument("--theta-true-override", type=str, default="")
    parser.add_argument("--theta-initial-override", type=str, default="")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--sensitivity-mode", choices=["coc", "barrier"], default=None)
    parser.add_argument("--barrier-gamma", type=float, default=None)
    parser.add_argument("--start-theta", type=str, default="")
    parser.add_argument("--epoch-offset", type=int, default=0)
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory, then stop.")
    args = parser.parse_args(argv)
    if model_override is not None:
        args.model = model_override
    defaults = SAFEPDP_DEFAULTS[args.model]
    if args.sensitivity_mode is None:
        args.sensitivity_mode = defaults.sensitivity_mode
    if args.barrier_gamma is None:
        args.barrier_gamma = defaults.barrier_gamma
    if args.tol is None:
        args.tol = defaults.tol or 1e-3
    measure_memory = bool(args.memory_only)

    out_dir, prefix = output_paths(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem_model = {"cartpole": "cartpole", "quadrotor": "quadrotor", "robot_arm": "arm"}[args.model]

    def build_problem_context():
        instance = build_problem(problem_model, horizon=args.horizon)
        apply_theta_overrides(instance, args.theta_true_override, args.theta_initial_override)
        return instance

    problem_build = profiled_call(build_problem_context, measure_memory=measure_memory, sample_tree=True)
    instance = problem_build["value"]

    def build_solver_context():
        coc = build_safepdp_coc(instance)
        if args.sensitivity_mode == "barrier":
            coc.convert2BarrierOC(gamma=args.barrier_gamma)
        return coc

    solver_build = profiled_call(build_solver_context, measure_memory=measure_memory, sample_tree=True)
    coc = solver_build["value"]
    build_stats = {
        "problem_build_time_sec": problem_build["elapsed_sec"],
        "solver_build_time_sec": solver_build["elapsed_sec"],
        "elapsed_sec": problem_build["elapsed_sec"] + solver_build["elapsed_sec"],
        "measure_memory": measure_memory,
    }
    if measure_memory:
        build_stats.update(
            {
                "problem_build_rss_peak_delta_mb": problem_build["rss_peak_delta_mb"],
                "problem_build_rss_tree_peak_delta_mb": problem_build["rss_tree_peak_delta_mb"],
                "solver_build_rss_peak_delta_mb": solver_build["rss_peak_delta_mb"],
                "solver_build_rss_tree_peak_delta_mb": solver_build["rss_tree_peak_delta_mb"],
                "rss_peak_delta_mb": problem_build["rss_peak_delta_mb"] + solver_build["rss_peak_delta_mb"],
                "rss_tree_peak_delta_mb": problem_build["rss_tree_peak_delta_mb"] + solver_build["rss_tree_peak_delta_mb"],
            }
        )
        print(
            f"{args.model} SafePDP build "
            f"problem_mem={build_stats['problem_build_rss_tree_peak_delta_mb']:.3f}MB "
            f"solver_mem={build_stats['solver_build_rss_tree_peak_delta_mb']:.3f}MB "
            f"build_mem={build_stats['rss_tree_peak_delta_mb']:.3f}MB "
            f"time={build_stats['elapsed_sec']:.4f}s"
        )
    else:
        print(f"{args.model} SafePDP build time={build_stats['elapsed_sec']:.4f}s")
    if args.mode in {"imitation", "both"}:
        print(f"wrote {run_imitation(args, instance, coc, out_dir, prefix, build_stats)}")
    if args.mode == "closed_loop":
        print(f"wrote {run_closed_loop_imitation(args, instance, coc, out_dir, prefix, build_stats)}")
    if args.mode in {"mpc", "both"}:
        print(f"wrote {run_mpc(args, instance, coc, out_dir, prefix, build_stats)}")


if __name__ == "__main__":
    main()
