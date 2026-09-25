import casadi as ca
import numpy as np
import pytest

from lapanda import (
    AlmOptions,
    BackwardOptions,
    CasadiProblem,
    SolverOptions,
    build_solver,
)


def _solver_options(tolerance=1e-9, max_iterations=80):
    options = SolverOptions()
    options.max_iterations = max_iterations
    options.tolerance = tolerance
    options.buffer_size = 5
    options.max_stable_iter = 0
    return options


def _backward_options(tolerance=1e-10, max_iterations=80):
    options = BackwardOptions()
    options.enable = True
    options.tolerance = tolerance
    options.max_iterations = max_iterations
    return options


def _alm_options(tolerance=1e-6, max_iterations=20):
    options = AlmOptions()
    options.max_iterations = max_iterations
    options.tolerance = tolerance
    options.initial_penalty = 10.0
    options.penalty_update_factor = 5.0
    options.sufficient_decrease_factor = 0.25
    return options


def test_panda_box_qp_forward_matches_projection():
    u = ca.SX.sym("u", 4)
    theta = ca.SX.sym("theta", 4)
    variable = ca.SX.sym("variable", 4)
    lower = np.array([-1.0, -0.5, -2.0, 0.0])
    upper = np.array([1.0, 0.5, 2.0, 1.5])
    theta_value = np.array([-2.0, 0.25, 3.0, 0.75])
    target = np.array([-0.2, -0.1, 1.25, 1.0])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=0.5 * ca.sumsqr(u - theta),
        box_lower=lower,
        box_upper=upper,
        outer_loss=0.5 * ca.sumsqr(u - variable),
    )

    result = build_solver(problem, backend="callback").solve_panda(
        x0=np.zeros(4),
        theta=theta_value,
        variable=target,
        solver_options=_solver_options(),
    )

    expected = np.clip(theta_value, lower, upper)
    np.testing.assert_allclose(result["solution"], expected, atol=2e-7)
    assert result["final_residual"] < 1e-6


def test_panda_box_qp_backward_matches_active_set_formula():
    u = ca.SX.sym("u", 4)
    theta = ca.SX.sym("theta", 4)
    variable = ca.SX.sym("variable", 4)
    lower = np.array([-1.0, -0.5, -2.0, 0.0])
    upper = np.array([1.0, 0.5, 2.0, 1.5])
    theta_value = np.array([-2.0, 0.25, 3.0, 0.75])
    target = np.array([-0.2, -0.1, 1.25, 1.0])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=0.5 * ca.sumsqr(u - theta),
        box_lower=lower,
        box_upper=upper,
        outer_loss=0.5 * ca.sumsqr(u - variable),
    )

    result = build_solver(problem, backend="callback").solve_panda(
        x0=np.clip(theta_value, lower, upper),
        theta=theta_value,
        variable=target,
        solver_options=_solver_options(tolerance=1e-10),
        backward_options=_backward_options(),
    )

    solution = np.clip(theta_value, lower, upper)
    active = (solution <= lower + 1e-10) | (solution >= upper - 1e-10)
    expected_grad = solution - target
    expected_grad[active] = 0.0

    np.testing.assert_allclose(result["solution"], solution, atol=2e-7)
    np.testing.assert_allclose(result["grad_theta"], expected_grad, atol=2e-7)
    assert result["backward_residual"] < 1e-8
    assert result["backward_solver_used"] == "cg"
    assert result["backward_fallback_used"] is False


@pytest.mark.parametrize(
    "theta_value",
    [
        np.array([2.0, -0.25]),
        np.array([-0.5, 3.0]),
        np.array([1.5, 1.5]),
    ],
)
def test_lapanda_equality_qp_forward_matches_projection(theta_value):
    u = ca.SX.sym("u", 2)
    theta = ca.SX.sym("theta", 2)
    variable = ca.SX.sym("variable", 1)

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=0.5 * ca.sumsqr(u - theta),
        constraints=ca.vertcat(u[0] + u[1]),
    )

    result = build_solver(problem, backend="callback").solve_lapanda(
        x0=np.zeros(2),
        theta=theta_value,
        variable=np.zeros(1),
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=_solver_options(tolerance=1e-8, max_iterations=120),
        alm_options=_alm_options(),
    )

    shift = (theta_value.sum() - 1.0) / 2.0
    expected = theta_value - shift

    np.testing.assert_allclose(result["solution"], expected, atol=2e-4)
    np.testing.assert_allclose(result["solution"].sum(), 1.0, atol=2e-6)
    assert result["final_residual"] < 1e-6


def test_lapanda_backward_matches_last_inner_subproblem_formula():
    u = ca.SX.sym("u", 2)
    theta = ca.SX.sym("theta", 2)
    variable = ca.SX.sym("variable", 2)
    target = np.array([0.25, -0.5])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=0.5 * ca.sumsqr(u - theta),
        constraints=ca.vertcat(u[0] + u[1]),
        outer_loss=0.5 * ca.sumsqr(u - variable),
    )

    result = build_solver(problem, backend="callback").solve_lapanda(
        x0=np.zeros(2),
        theta=np.array([2.0, -0.25]),
        variable=target,
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=_solver_options(tolerance=1e-8, max_iterations=120),
        alm_options=_alm_options(),
        backward_options=_backward_options(tolerance=1e-8, max_iterations=120),
    )

    a = np.ones((2, 1))
    hessian = np.eye(2) + result["penalty"] * (a @ a.T)
    expected = np.linalg.solve(hessian, result["solution"] - target)

    np.testing.assert_allclose(result["grad_theta"], expected, atol=2e-4)
    assert result["backward_residual"] < 1e-6
    assert result["backward_solver_used"] == "cg"
    assert result["backward_fallback_used"] is False
