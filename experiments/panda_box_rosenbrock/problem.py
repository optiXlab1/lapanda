"""Box-constrained Rosenbrock experiment for PANDA."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from experiments.common import ensure_python_path

ensure_python_path()

from lapanda import CasadiProblem


@dataclass
class BoxRosenbrockInstance:
    problem: CasadiProblem
    n: int
    lower: np.ndarray
    upper: np.ndarray
    objective_fun: ca.Function
    outer_loss_fun: ca.Function


def sinusoidal_target(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    return 0.5 * np.sin(2.0 * np.pi * i / max(n, 1))


def initial_point(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.5, 1.5, size=n)


def theta_nominal() -> np.ndarray:
    return np.array([100.0, 1.0, 1e-3])


def build_problem(n: int) -> BoxRosenbrockInstance:
    x = ca.SX.sym("x", n)
    theta = ca.SX.sym("theta", 3)
    target = ca.SX.sym("target", n)

    cost = ca.SX(0)
    for i in range(n - 1):
        cost += theta[0] * (x[i + 1] - x[i] ** 2) ** 2 + theta[1] * (1.0 - x[i]) ** 2
    cost += 0.5 * theta[2] * ca.dot(x, x)

    outer_loss = 0.5 * ca.sumsqr(x - target)
    lower = -2.0 * np.ones(n)
    upper = 2.0 * np.ones(n)
    problem = CasadiProblem(
        u=x,
        theta=theta,
        variable=target,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        outer_loss=outer_loss,
    )
    return BoxRosenbrockInstance(
        problem=problem,
        n=n,
        lower=lower,
        upper=upper,
        objective_fun=ca.Function("exp1_objective", [x, theta], [cost]),
        outer_loss_fun=ca.Function("exp1_outer_loss", [x, target], [outer_loss]),
    )


def projected_residual_inf(instance: BoxRosenbrockInstance, x_value, theta_value, gamma) -> float:
    grad_fun = ca.Function(
        "exp1_grad",
        [instance.problem.u, instance.problem.theta],
        [ca.gradient(instance.problem.cost, instance.problem.u)],
    )
    grad = np.asarray(grad_fun(x_value, theta_value)).reshape(-1)
    projected = np.clip(np.asarray(x_value) - gamma * grad, instance.lower, instance.upper)
    return float(np.max(np.abs((np.asarray(x_value) - projected) / gamma)))
