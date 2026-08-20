"""CasADi solver-differentiation baseline for Exp.1."""

from __future__ import annotations

import contextlib
import os
import sys

import casadi as ca
import numpy as np


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


def evaluate_sensitivity(sensitivity_fun, theta, target, x0):
    with suppress_native_output(True):
        return sensitivity_fun(theta, target, x0)


def build_ipopt_solver(instance, tol: float, max_iter: int, name: str):
    nlp = {
        "x": instance.problem.u,
        "p": instance.problem.theta,
        "f": instance.problem.cost,
        "g": instance.problem.constraints,
    }
    return ca.nlpsol(
        name,
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": tol,
            "ipopt.constr_viol_tol": tol,
            "ipopt.max_iter": max_iter,
        },
    )


def solve_ipopt(instance, solver, x0, theta):
    return solver(
        x0=x0,
        p=theta,
        lbx=instance.lower,
        ubx=instance.upper,
        lbg=instance.constraint_lower,
        ubg=instance.constraint_upper,
    )


def radius_chain_constraints(n: int, constraint_stride: int, x, theta):
    constraints = []
    for i in range(0, n - 1, constraint_stride):
        phase = 2.0 * np.pi * (i + 1) / max(n, 1)
        radius = theta[3] + 0.1 * np.sin(phase)
        constraints.append(x[i] ** 2 + x[i + 1] ** 2 - radius**2)
    return ca.vertcat(*constraints)


def rosenbrock_cost(n: int, x, theta):
    cost = ca.MX(0)
    for i in range(n - 1):
        cost += theta[0] * (x[i + 1] - x[i] ** 2) ** 2 + theta[1] * (1.0 - x[i]) ** 2
    cost += 0.5 * theta[2] * ca.dot(x, x)
    return cost


def _qpsol_options(qpsol: str):
    if qpsol == "qpoases":
        return {"printLevel": "none"}
    if qpsol == "qrqp":
        return {
            "print_header": False,
            "print_iter": False,
            "print_info": False,
        }
    raise ValueError(f"Unknown qpsol: {qpsol}")


def build_casadi_sensitivity(
    n: int,
    constraint_stride: int,
    tol: float,
    max_iter: int,
    name: str,
    qpsol: str = "qpoases",
):
    opti = ca.Opti()
    x = opti.variable(n)
    theta = opti.parameter(4)
    cost = rosenbrock_cost(n, x, theta)
    constraints = radius_chain_constraints(n, constraint_stride, x, theta)
    opti.minimize(cost)
    opti.subject_to(opti.bounded(-2.0, x, 2.0))
    opti.subject_to(constraints <= 0.0)
    with suppress_native_output(True):
        opti.solver(
            "sqpmethod",
            {
                "qpsol": qpsol,
                "qpsol_options": _qpsol_options(qpsol),
                "print_header": False,
                "print_iteration": False,
                "print_status": False,
                "print_time": False,
                "record_time": False,
                "max_iter": max_iter,
                "tol_pr": tol,
                "tol_du": tol,
            },
        )

        solve_fun = opti.to_function(f"{name}_solve", [theta, x], [x])
    th = ca.MX.sym("th", 4)
    target = ca.MX.sym("target", n)
    x0 = ca.MX.sym("x0", n)
    x_star = solve_fun(th, x0)
    loss = 0.5 * ca.sumsqr(x_star - target)
    grad_theta = ca.jacobian(loss, th)
    with suppress_native_output(True):
        return ca.Function(f"{name}_sensitivity", [th, target, x0], [x_star, loss, grad_theta])
