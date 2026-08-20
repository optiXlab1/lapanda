"""Smooth general-constraint experiment for lapanda."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from experiments.common import ensure_python_path

ensure_python_path()

from lapanda import CasadiProblem


@dataclass
class RadiusChainInstance:
    problem: CasadiProblem
    n: int
    m: int
    lower: np.ndarray
    upper: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    objective_fun: ca.Function
    constraint_fun: ca.Function
    outer_loss_fun: ca.Function


def theta_nominal() -> np.ndarray:
    return np.array([10.0, 1.0, 1e-3, 1.35])


def sinusoidal_target(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    return 0.3 * np.sin(2.0 * np.pi * i / max(n, 1))


def initial_point(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.3, 0.3, size=n)


def build_problem(n: int, constraint_stride: int = 1) -> RadiusChainInstance:
    x = ca.SX.sym("x", n)
    theta = ca.SX.sym("theta", 4)
    target = ca.SX.sym("target", n)

    cost = ca.SX(0)
    for i in range(n - 1):
        cost += theta[0] * (x[i + 1] - x[i] ** 2) ** 2 + theta[1] * (1.0 - x[i]) ** 2
    cost += 0.5 * theta[2] * ca.dot(x, x)

    constraints = []
    for i in range(0, n - 1, constraint_stride):
        phase = 2.0 * np.pi * (i + 1) / max(n, 1)
        radius = theta[3] + 0.1 * np.sin(phase)
        constraints.append(x[i] ** 2 + x[i + 1] ** 2 - radius**2)

    g = ca.vertcat(*constraints)
    outer_loss = 0.5 * ca.sumsqr(x - target)
    lower = -2.0 * np.ones(n)
    upper = 2.0 * np.ones(n)
    constraint_lower = -1e20 * np.ones(g.numel())
    constraint_upper = np.zeros(g.numel())

    problem = CasadiProblem(
        u=x,
        theta=theta,
        variable=target,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=g,
        outer_loss=outer_loss,
    )
    return RadiusChainInstance(
        problem=problem,
        n=n,
        m=g.numel(),
        lower=lower,
        upper=upper,
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        objective_fun=ca.Function("exp1_objective", [x, theta], [cost]),
        constraint_fun=ca.Function("exp1_constraints", [x, theta], [g]),
        outer_loss_fun=ca.Function("exp1_outer_loss", [x, target], [outer_loss]),
    )


def constraint_violation_inf(instance: RadiusChainInstance, x_value, theta_value) -> float:
    g = np.asarray(instance.constraint_fun(x_value, theta_value)).reshape(-1)
    return float(np.max(np.maximum(g, 0.0))) if g.size else 0.0


def active_constraint_count(instance: RadiusChainInstance, x_value, theta_value, tol: float = 1e-5) -> int:
    g = np.asarray(instance.constraint_fun(x_value, theta_value)).reshape(-1)
    return int(np.sum(np.abs(g) <= tol))


def box_violation_inf(instance: RadiusChainInstance, x_value) -> float:
    x_value = np.asarray(x_value)
    return float(np.max(np.maximum(instance.lower - x_value, np.maximum(x_value - instance.upper, 0.0))))
