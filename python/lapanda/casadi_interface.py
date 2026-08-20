from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import casadi as ca
import numpy as np

try:
    from ._lapanda import Oracle
except ImportError:  # pragma: no cover
    Oracle = None


ArrayLike = Sequence[float]


@dataclass
class CasadiProblem:
    """Symbolic problem description used to build PANDA oracle callbacks."""

    u: ca.SX
    theta: ca.SX
    variable: ca.SX
    cost: ca.SX
    box_lower: Optional[ArrayLike] = None
    box_upper: Optional[ArrayLike] = None
    constraints: Optional[ca.SX] = None
    outer_loss: Optional[ca.SX] = None


def _numel(x) -> int:
    return int(np.prod(x.shape))


def _flat_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _call_function(fun, *args) -> Tuple:
    out = fun(*args)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def _make_box_prox(lower, upper):
    lower = None if lower is None else _flat_array(lower)
    upper = None if upper is None else _flat_array(upper)

    def prox(u, gamma, theta, variable):
        del gamma, theta, variable
        z = np.asarray(u, dtype=float).copy()
        if lower is not None:
            z = np.maximum(z, lower)
        if upper is not None:
            z = np.minimum(z, upper)
        return z, 0.0

    def jprox(u_star, gamma, theta, variable):
        del gamma, theta, variable
        active_tol = 1e-10
        J = np.ones_like(np.asarray(u_star, dtype=float).reshape(-1))
        if lower is not None:
            J[u_star <= lower + active_tol] = 0.0
        if upper is not None:
            J[u_star >= upper - active_tol] = 0.0
        return J

    return prox, jprox


def _make_identity_prox(n: int):
    def prox(u, gamma, theta, variable):
        del gamma, theta, variable
        return np.asarray(u, dtype=float).copy(), 0.0

    def jprox(u_star, gamma, theta, variable):
        del gamma, theta, variable
        return np.ones(n, dtype=float)

    return prox, jprox


def generate_oracle(problem: CasadiProblem):
    """Build a pybind-ready oracle from a CasADi symbolic problem.

    The generated oracle supports smooth costs plus either no nonsmooth term or
    a box indicator prox. If constraints are supplied, the same oracle can be
    used by ``solve_lapanda``.
    """

    if Oracle is None:
        raise ImportError("The _lapanda extension is not built yet.")

    u = ca.vec(problem.u)
    theta = ca.vec(problem.theta)
    variable = ca.vec(problem.variable)
    cost = problem.cost

    n = _numel(u)
    ntheta = _numel(theta)
    nvar = _numel(variable)

    grad = ca.gradient(cost, u)
    cost_grad_fun = ca.Function("panda_cost_grad", [u, theta, variable], [cost, grad])

    v = ca.SX.sym("v", n)
    hvp_expr = ca.jtimes(grad, u, v)
    hvp_fun = ca.Function("panda_hvp", [v, u, theta, variable], [hvp_expr])
    vjp_expr = ca.jacobian(grad, theta).T @ v
    vjp_fun = ca.Function("panda_vjp", [v, u, theta, variable], [vjp_expr])

    if problem.box_lower is None and problem.box_upper is None:
        prox, jprox = _make_identity_prox(n)
    else:
        prox, jprox = _make_box_prox(problem.box_lower, problem.box_upper)

    if problem.outer_loss is not None:
        loss_grad_expr = ca.gradient(problem.outer_loss, u)
        loss_grad_fun = ca.Function(
            "panda_loss_grad",
            [u, theta, variable],
            [problem.outer_loss, loss_grad_expr],
        )

        def loss_grad(u_star, theta_value, variable_value):
            L, grad_u = _call_function(loss_grad_fun, u_star, theta_value, variable_value)
            return float(L), _flat_array(grad_u)
    else:
        def loss_grad(u_star, theta_value, variable_value):
            del u_star, theta_value, variable_value
            return 0.0, np.zeros(n, dtype=float)

    oracle = Oracle()
    oracle.n = n
    oracle.ntheta = ntheta
    oracle.nvar = nvar

    def cost_gradient(u_value, theta_value, variable_value):
        f, g = _call_function(cost_grad_fun, u_value, theta_value, variable_value)
        return float(f), _flat_array(g)

    def hvp(v_value, u_value, theta_value, variable_value):
        return _flat_array(hvp_fun(v_value, u_value, theta_value, variable_value))

    def vjp(v_value, u_value, theta_value, variable_value):
        return _flat_array(vjp_fun(v_value, u_value, theta_value, variable_value))

    oracle.cost_gradient = cost_gradient
    oracle.prox = prox
    oracle.jprox = jprox
    oracle.hvp = hvp
    oracle.loss_grad = loss_grad
    oracle.vjp = vjp

    if problem.constraints is not None:
        c = ca.vec(problem.constraints)
        ncon = _numel(c)
        constraint_fun = ca.Function("panda_constraints", [u, theta, variable], [c])
        weight = ca.SX.sym("weight", ncon)
        jtprod_expr = ca.jacobian(c, u).T @ weight
        jtprod_fun = ca.Function(
            "panda_constraint_jtprod",
            [u, theta, variable, weight],
            [jtprod_expr],
        )
        jv_expr = ca.jtimes(c, u, v)
        jv_fun = ca.Function("panda_constraint_jv", [v, u, theta, variable], [jv_expr])
        weighted_hvp_expr = ca.jtimes(jtprod_expr, u, v)
        weighted_hvp_fun = ca.Function(
            "panda_constraint_weighted_hvp",
            [v, u, theta, variable, weight],
            [weighted_hvp_expr],
        )
        weighted_vjp_expr = ca.jacobian(jtprod_expr, theta).T @ v
        weighted_vjp_fun = ca.Function(
            "panda_constraint_weighted_vjp",
            [v, u, theta, variable, weight],
            [weighted_vjp_expr],
        )
        theta_jtprod_expr = ca.jacobian(c, theta).T @ weight
        theta_jtprod_fun = ca.Function(
            "panda_constraint_theta_jtprod",
            [u, theta, variable, weight],
            [theta_jtprod_expr],
        )

        def constraint(u_value, theta_value, variable_value):
            return _flat_array(constraint_fun(u_value, theta_value, variable_value))

        def constraint_jtprod(u_value, theta_value, variable_value, multiplier_value):
            return _flat_array(
                jtprod_fun(u_value, theta_value, variable_value, multiplier_value)
            )

        def constraint_projection_data(
            u_value,
            theta_value,
            variable_value,
            multiplier_value,
            penalty_value,
            constraint_lower_value,
            constraint_upper_value,
        ):
            c_value = _flat_array(constraint_fun(u_value, theta_value, variable_value))
            multiplier_value = _flat_array(multiplier_value)
            penalty_value = _flat_array(penalty_value)
            lower_value = _flat_array(constraint_lower_value)
            upper_value = _flat_array(constraint_upper_value)

            shifted = c_value + multiplier_value / penalty_value
            projected = np.minimum(np.maximum(shifted, lower_value), upper_value)
            equality_value = np.abs(upper_value - lower_value) <= 1e-8
            active_value = (
                (shifted <= lower_value + 1e-8)
                | (shifted >= upper_value - 1e-8)
                | equality_value
            ).astype(float)
            projected_multiplier = penalty_value * (shifted - projected)
            active_penalty = penalty_value * active_value
            return projected_multiplier, active_penalty

        def constraint_hvp(
            v_value,
            u_value,
            theta_value,
            variable_value,
            multiplier_value,
            penalty_value,
            constraint_lower_value,
            constraint_upper_value,
        ):
            projected_multiplier, active_penalty = constraint_projection_data(
                u_value,
                theta_value,
                variable_value,
                multiplier_value,
                penalty_value,
                constraint_lower_value,
                constraint_upper_value,
            )
            jv_value = _flat_array(jv_fun(v_value, u_value, theta_value, variable_value))
            curvature = _flat_array(
                weighted_hvp_fun(
                    v_value, u_value, theta_value, variable_value, projected_multiplier
                )
            )
            gauss_newton = _flat_array(
                jtprod_fun(
                    u_value, theta_value, variable_value, active_penalty * jv_value
                )
            )
            return curvature + gauss_newton

        def constraint_vjp(
            v_value,
            u_value,
            theta_value,
            variable_value,
            multiplier_value,
            penalty_value,
            constraint_lower_value,
            constraint_upper_value,
        ):
            projected_multiplier, active_penalty = constraint_projection_data(
                u_value,
                theta_value,
                variable_value,
                multiplier_value,
                penalty_value,
                constraint_lower_value,
                constraint_upper_value,
            )
            jv_value = _flat_array(jv_fun(v_value, u_value, theta_value, variable_value))
            curvature = _flat_array(
                weighted_vjp_fun(
                    v_value, u_value, theta_value, variable_value, projected_multiplier
                )
            )
            gauss_newton = _flat_array(
                theta_jtprod_fun(
                    u_value, theta_value, variable_value, active_penalty * jv_value
                )
            )
            return curvature + gauss_newton

        oracle.ncon = ncon
        oracle.constraint = constraint
        oracle.constraint_jtprod = constraint_jtprod
        oracle.constraint_hvp = constraint_hvp
        oracle.constraint_vjp = constraint_vjp

    return oracle
