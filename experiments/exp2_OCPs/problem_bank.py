"""Shared OCP problem definitions for Exp.2."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from lapanda import CasadiProblem

STATE_IMITATION_WEIGHT = 1.0

@dataclass
class OcpInstance:
    name: str
    problem: CasadiProblem
    horizon: int
    nu: int
    nx: int
    dt: float
    theta_true: np.ndarray
    theta_initial: np.ndarray
    x_initial: np.ndarray
    x_target: np.ndarray
    variable_size: int
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    cost_fun: ca.Function
    outer_loss_fun: ca.Function
    constraint_fun: ca.Function
    rollout_fun: ca.Function


def cartpole_theta_true() -> np.ndarray:
    return np.array([1.0, 18.0, 0.05, 0.25, 0.015], dtype=float)


def cartpole_theta_initial() -> np.ndarray:
    return np.array([0.4, 4.0, 0.02, 0.08, 0.08], dtype=float)


def cartpole_initial_state() -> np.ndarray:
    return np.array([0.0, 0.0, 0.35, 0.0], dtype=float)


def cartpole_target_state() -> np.ndarray:
    return np.zeros(4, dtype=float)


def make_cartpole_variable(x0, target, demo_u) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(x0, dtype=float).reshape(4),
            np.asarray(target, dtype=float).reshape(4),
            np.asarray(demo_u, dtype=float).reshape(-1),
        ]
    )


def _cartpole_step(x, u, dt: float):
    cart_mass = 1.0
    pole_mass = 0.1
    pole_half_length = 0.5
    gravity = 9.81
    total_mass = cart_mass + pole_mass

    pos = x[0]
    vel = x[1]
    angle = x[2]
    angle_vel = x[3]
    force = u[0]

    sin_angle = ca.sin(angle)
    cos_angle = ca.cos(angle)
    temp = (force + pole_mass * pole_half_length * angle_vel**2 * sin_angle) / total_mass
    angle_acc = (gravity * sin_angle - cos_angle * temp) / (
        pole_half_length * (4.0 / 3.0 - pole_mass * cos_angle**2 / total_mass)
    )
    pos_acc = temp - pole_mass * pole_half_length * angle_acc * cos_angle / total_mass
    next_vel = vel + dt * pos_acc
    next_angle_vel = angle_vel + dt * angle_acc
    return ca.vertcat(pos + dt * next_vel, next_vel, angle + dt * next_angle_vel, next_angle_vel)


def build_cartpole_problem(
    horizon: int = 20,
    dt: float = 0.05,
    u_lower: float = -2.0,
    u_upper: float = 6.0,
    energy_budget: float = 70.0,
) -> OcpInstance:
    nx = 4
    nu = 1
    u = ca.SX.sym("u", horizon)
    theta = ca.SX.sym("theta", 5)
    demo_state_size = horizon * nx
    variable = ca.SX.sym("variable", 2 * nx + horizon + demo_state_size)
    x = variable[:nx]
    target = variable[nx : 2 * nx]
    demo_u = variable[2 * nx : 2 * nx + horizon]
    demo_x = variable[2 * nx + horizon :]

    q_pos, q_ang, q_vel, q_ang_vel, r_force = [theta[i] for i in range(5)]
    states = [x]
    cost = ca.SX(0)
    for k in range(horizon):
        uk = ca.vertcat(u[k])
        err = x - target
        cost += (
            q_pos * err[0] ** 2
            + q_vel * err[1] ** 2
            + q_ang * err[2] ** 2
            + q_ang_vel * err[3] ** 2
            + r_force * u[k] ** 2
        )
        x = _cartpole_step(x, uk, dt)
        states.append(x)

    err = x - target
    cost += 10.0 * (
        q_pos * err[0] ** 2
        + q_vel * err[1] ** 2
        + q_ang * err[2] ** 2
        + q_ang_vel * err[3] ** 2
    )
    rollout_tail = ca.vertcat(*states[1:])
    outer_loss = 0.5 * ca.sumsqr(u - demo_u) + 0.5 * STATE_IMITATION_WEIGHT * ca.sumsqr(rollout_tail - demo_x)
    constraints = ca.vertcat(ca.sumsqr(u) - energy_budget)
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=u_lower * np.ones(horizon),
        box_upper=u_upper * np.ones(horizon),
        constraints=constraints,
        outer_loss=outer_loss,
    )
    return OcpInstance(
        name="cartpole",
        problem=problem,
        horizon=horizon,
        nu=nu,
        nx=nx,
        dt=dt,
        theta_true=cartpole_theta_true(),
        theta_initial=cartpole_theta_initial(),
        x_initial=cartpole_initial_state(),
        x_target=cartpole_target_state(),
        variable_size=2 * nx + horizon + demo_state_size,
        constraint_lower=-1e20 * np.ones(1),
        constraint_upper=np.zeros(1),
        cost_fun=ca.Function("cartpole_cost", [u, theta, variable], [cost]),
        outer_loss_fun=ca.Function("cartpole_outer_loss", [u, theta, variable], [outer_loss]),
        constraint_fun=ca.Function("cartpole_constraints", [u, theta, variable], [constraints]),
        rollout_fun=ca.Function("cartpole_rollout", [u, theta, variable], [ca.hcat(states).T]),
    )


def _box_from_pairs(pairs, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for _ in range(horizon):
        for lo, hi in pairs:
            lower.append(lo)
            upper.append(hi)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _arm_fk(q0, q1):
    return ca.vertcat(ca.cos(q0) + 0.8 * ca.cos(q0 + q1), ca.sin(q0) + 0.8 * ca.sin(q0 + q1))


def build_quadrotor_problem(horizon: int = 20, dt: float = 0.05) -> OcpInstance:
    nx = 6
    nu = 2
    u = ca.SX.sym("u", horizon * nu)
    theta = ca.SX.sym("theta", 7)
    demo_control_size = horizon * nu
    demo_state_size = horizon * nx
    variable = ca.SX.sym("variable", 2 * nx + demo_control_size + demo_state_size)
    x = variable[:nx]
    target = variable[nx : 2 * nx]
    demo_u = variable[2 * nx : 2 * nx + demo_control_size]
    demo_x = variable[2 * nx + demo_control_size :]

    q_pos, q_vel, q_angle, q_omega, r_thrust, r_torque, _unused_margin = [theta[i] for i in range(7)]
    gravity = 9.81
    inertia_inv = 4.0
    floor_height = 0.43
    ceiling_height = 1.20
    angle_max = 0.34
    states = [x]
    constraints = []
    cost = ca.SX(0)

    for k in range(horizon):
        uk = u[k * nu : (k + 1) * nu]
        err = x - target
        cost += (
            q_pos * (err[0] ** 2 + err[1] ** 2)
            + q_vel * (err[2] ** 2 + err[3] ** 2)
            + q_angle * err[4] ** 2
            + q_omega * err[5] ** 2
            + r_thrust * (uk[0] - gravity) ** 2
            + r_torque * uk[1] ** 2
        )
        acc_x = -uk[0] * ca.sin(x[4])
        acc_z = uk[0] * ca.cos(x[4]) - gravity
        omega_next = x[5] + dt * inertia_inv * uk[1]
        angle_next = x[4] + dt * omega_next
        vx_next = x[2] + dt * acc_x
        vz_next = x[3] + dt * acc_z
        x = ca.vertcat(x[0] + dt * vx_next, x[1] + dt * vz_next, vx_next, vz_next, angle_next, omega_next)
        states.append(x)
        constraints += [
            floor_height - x[1],
            x[1] - ceiling_height,
            x[0] - target[0],
            x[4] - angle_max,
            -x[4] - angle_max,
        ]

    err = x - target
    cost += 15.0 * (
        q_pos * (err[0] ** 2 + err[1] ** 2)
        + q_vel * (err[2] ** 2 + err[3] ** 2)
        + q_angle * err[4] ** 2
        + q_omega * err[5] ** 2
    )
    rollout_tail = ca.vertcat(*states[1:])
    outer_loss = 0.5 * ca.sumsqr(u - demo_u) + 0.5 * STATE_IMITATION_WEIGHT * ca.sumsqr(rollout_tail - demo_x)
    lower, upper = _box_from_pairs([(0.0, 2.2 * gravity), (-3.0, 3.0)], horizon)
    c = ca.vertcat(*constraints)
    problem = CasadiProblem(u=u, theta=theta, variable=variable, cost=cost, box_lower=lower, box_upper=upper, constraints=c, outer_loss=outer_loss)
    return OcpInstance(
        name="planar_quadrotor",
        problem=problem,
        horizon=horizon,
        nu=nu,
        nx=nx,
        dt=dt,
        theta_true=np.array([4.0, 0.8, 12.0, 0.5, 0.02, 0.08, 0.08], dtype=float),
        theta_initial=np.array([1.2, 0.25, 4.0, 0.15, 0.08, 0.25, 0.02], dtype=float),
        x_initial=np.array([-1.0, 0.65, 0.0, 0.0, 0.15, 0.0], dtype=float),
        x_target=np.array([1.0, 1.2, 0.0, 0.0, 0.0, 0.0], dtype=float),
        variable_size=2 * nx + demo_control_size + demo_state_size,
        constraint_lower=-1e20 * np.ones(5 * horizon),
        constraint_upper=np.zeros(5 * horizon),
        cost_fun=ca.Function("quad_cost", [u, theta, variable], [cost]),
        outer_loss_fun=ca.Function("quad_outer_loss", [u, theta, variable], [outer_loss]),
        constraint_fun=ca.Function("quad_constraints", [u, theta, variable], [c]),
        rollout_fun=ca.Function("quad_rollout", [u, theta, variable], [ca.hcat(states).T]),
    )


def build_robot_arm_problem(horizon: int = 20, dt: float = 0.05) -> OcpInstance:
    nx = 2
    nu = 2
    u = ca.SX.sym("u", horizon * nu)
    theta = ca.SX.sym("theta", 5)
    demo_control_size = horizon * nu
    demo_state_size = horizon * nx
    variable = ca.SX.sym("variable", 2 * nx + demo_control_size + demo_state_size)
    x = variable[:nx]
    target = variable[nx : 2 * nx]
    demo_u = variable[2 * nx : 2 * nx + demo_control_size]
    demo_x = variable[2 * nx + demo_control_size :]

    q_joint, q_ee, r_vel1, r_vel2, obstacle_margin = [theta[i] for i in range(5)]
    obstacle_center = np.array([1.30, 0.30], dtype=float)
    obstacle_radius = 0.29 + obstacle_margin
    q_lower = ca.vertcat(-1.10, -0.65)
    q_upper = ca.vertcat(0.75, 1.28)
    target_ee = _arm_fk(target[0], target[1])
    states = [x]
    constraints = []
    cost = ca.SX(0)

    for k in range(horizon):
        uk = u[k * nu : (k + 1) * nu]
        ee = _arm_fk(x[0], x[1])
        cost += q_joint * ca.dot(x - target, x - target) + q_ee * ca.dot(ee - target_ee, ee - target_ee)
        cost += r_vel1 * uk[0] ** 2 + r_vel2 * uk[1] ** 2
        x = x + dt * ca.vertcat(uk[0], uk[1])
        states.append(x)
        ee_next = _arm_fk(x[0], x[1])
        dist2 = (ee_next[0] - obstacle_center[0]) ** 2 + (ee_next[1] - obstacle_center[1]) ** 2
        constraints += [
            obstacle_radius**2 - dist2,
            x[0] - q_upper[0],
            q_lower[0] - x[0],
            x[1] - q_upper[1],
            q_lower[1] - x[1],
        ]

    ee = _arm_fk(x[0], x[1])
    cost += 20.0 * (q_joint * ca.dot(x - target, x - target) + q_ee * ca.dot(ee - target_ee, ee - target_ee))
    rollout_tail = ca.vertcat(*states[1:])
    outer_loss = 0.5 * ca.sumsqr(u - demo_u) + 0.5 * STATE_IMITATION_WEIGHT * ca.sumsqr(rollout_tail - demo_x)
    lower, upper = _box_from_pairs([(-3.0, 3.0), (-3.0, 3.0)], horizon)
    c = ca.vertcat(*constraints)
    problem = CasadiProblem(u=u, theta=theta, variable=variable, cost=cost, box_lower=lower, box_upper=upper, constraints=c, outer_loss=outer_loss)
    return OcpInstance(
        name="two_link_arm",
        problem=problem,
        horizon=horizon,
        nu=nu,
        nx=nx,
        dt=dt,
        theta_true=np.array([0.8, 8.0, 0.02, 0.02, 0.12], dtype=float),
        theta_initial=np.array([0.25, 2.0, 0.08, 0.08, 0.04], dtype=float),
        x_initial=np.array([-1.0, 1.25], dtype=float),
        x_target=np.array([0.75, -0.65], dtype=float),
        variable_size=2 * nx + demo_control_size + demo_state_size,
        constraint_lower=-1e20 * np.ones(5 * horizon),
        constraint_upper=np.zeros(5 * horizon),
        cost_fun=ca.Function("arm_cost", [u, theta, variable], [cost]),
        outer_loss_fun=ca.Function("arm_outer_loss", [u, theta, variable], [outer_loss]),
        constraint_fun=ca.Function("arm_constraints", [u, theta, variable], [c]),
        rollout_fun=ca.Function("arm_rollout", [u, theta, variable], [ca.hcat(states).T]),
    )


def build_problem(model: str, horizon: int = 20, dt: float = 0.05) -> OcpInstance:
    if model == "cartpole":
        return build_cartpole_problem(horizon=horizon, dt=dt)
    if model == "quadrotor":
        return build_quadrotor_problem(horizon=horizon, dt=dt)
    if model in {"arm", "robotarm", "robot_arm", "two_link_arm"}:
        return build_robot_arm_problem(horizon=horizon, dt=dt)
    raise ValueError(f"unknown model {model!r}; choose 'cartpole', 'quadrotor', or 'arm'")


def constraint_violation_inf(instance: OcpInstance, u, theta, variable) -> float:
    c = np.asarray(instance.constraint_fun(u, theta, variable), dtype=float).reshape(-1)
    return float(
        max(
            np.max(np.maximum(c - instance.constraint_upper, 0.0)),
            np.max(np.maximum(instance.constraint_lower - c, 0.0)),
        )
    )
