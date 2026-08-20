import casadi as ca
import numpy as np
import pytest

from lapanda import (
    AlmOptions,
    CasadiProblem,
    SolverOptions,
    build_solver,
)


def _solver_options(tolerance=1e-9, max_iterations=120):
    options = SolverOptions()
    options.max_iterations = max_iterations
    options.tolerance = tolerance
    options.buffer_size = 5
    options.max_stable_iter = 0
    return options


def _alm_options(tolerance=1e-6, max_iterations=30):
    options = AlmOptions()
    options.max_iterations = max_iterations
    options.tolerance = tolerance
    options.initial_penalty = 10.0
    options.penalty_update_factor = 5.0
    options.sufficient_decrease_factor = 0.25
    return options


def _ipopt_solver(nlp):
    try:
        return ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.tol": 1e-10,
            },
        )
    except RuntimeError as exc:
        pytest.skip(f"CasADi IPOPT plugin is not available: {exc}")


def test_panda_box_qp_matches_ipopt():
    u = ca.SX.sym("u", 4)
    theta = ca.SX.sym("theta", 4)
    variable = ca.SX.sym("variable", 1)
    theta_value = np.array([-2.0, 0.25, 3.0, 0.75])
    lower = np.array([-1.0, -0.5, -2.0, 0.0])
    upper = np.array([1.0, 0.5, 2.0, 1.5])
    cost = 0.5 * ca.sumsqr(u - theta)

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
    )
    panda_result = build_solver(problem, backend="callback").solve_panda(
        x0=np.zeros(4),
        theta=theta_value,
        variable=np.zeros(1),
        solver_options=_solver_options(),
    )

    solver = _ipopt_solver({"x": u, "p": theta, "f": cost})
    ipopt_result = solver(
        x0=np.zeros(4),
        p=theta_value,
        lbx=lower,
        ubx=upper,
    )
    ipopt_solution = np.array(ipopt_result["x"]).reshape(-1)

    np.testing.assert_allclose(panda_result["solution"], ipopt_solution, atol=2e-6)
    assert abs(float(ipopt_result["f"]) - 0.5 * np.sum((panda_result["solution"] - theta_value) ** 2)) < 1e-7


def test_lapanda_equality_qp_matches_ipopt():
    u = ca.SX.sym("u", 2)
    theta = ca.SX.sym("theta", 2)
    variable = ca.SX.sym("variable", 1)
    theta_value = np.array([2.0, -0.25])
    cost = 0.5 * ca.sumsqr(u - theta)
    constraints = ca.vertcat(u[0] + u[1])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        constraints=constraints,
    )
    alm_result = build_solver(problem, backend="callback").solve_lapanda(
        x0=np.zeros(2),
        theta=theta_value,
        variable=np.zeros(1),
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=_solver_options(tolerance=1e-8),
        alm_options=_alm_options(),
    )

    solver = _ipopt_solver({"x": u, "p": theta, "f": cost, "g": constraints})
    ipopt_result = solver(
        x0=np.zeros(2),
        p=theta_value,
        lbg=np.array([1.0]),
        ubg=np.array([1.0]),
    )
    ipopt_solution = np.array(ipopt_result["x"]).reshape(-1)

    np.testing.assert_allclose(alm_result["solution"], ipopt_solution, atol=2e-4)
    np.testing.assert_allclose(alm_result["solution"].sum(), 1.0, atol=2e-6)
    assert abs(float(ipopt_result["f"]) - 0.5 * np.sum((alm_result["solution"] - theta_value) ** 2)) < 1e-5


def test_lapanda_box_and_equality_constraints_match_ipopt():
    u = ca.SX.sym("u", 3)
    theta = ca.SX.sym("theta", 3)
    variable = ca.SX.sym("variable", 1)
    theta_value = np.array([2.0, -1.0, 0.25])
    lower = np.array([0.0, -0.25, -0.5])
    upper = np.array([1.0, 0.75, 0.75])
    cost = 0.5 * ca.sumsqr(u - theta)
    constraints = ca.vertcat(u[0] + u[1] + u[2])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=constraints,
    )
    alm_result = build_solver(problem, backend="callback").solve_lapanda(
        x0=np.array([0.5, 0.0, 0.5]),
        theta=theta_value,
        variable=np.zeros(1),
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=_solver_options(tolerance=1e-8, max_iterations=150),
        alm_options=_alm_options(tolerance=1e-6, max_iterations=40),
    )

    solver = _ipopt_solver({"x": u, "p": theta, "f": cost, "g": constraints})
    ipopt_result = solver(
        x0=np.array([0.5, 0.0, 0.5]),
        p=theta_value,
        lbx=lower,
        ubx=upper,
        lbg=np.array([1.0]),
        ubg=np.array([1.0]),
    )
    ipopt_solution = np.array(ipopt_result["x"]).reshape(-1)

    np.testing.assert_allclose(alm_result["solution"], ipopt_solution, atol=5e-4)
    assert np.all(alm_result["solution"] >= lower - 1e-7)
    assert np.all(alm_result["solution"] <= upper + 1e-7)
    np.testing.assert_allclose(alm_result["solution"].sum(), 1.0, atol=2e-6)
    assert abs(float(ipopt_result["f"]) - 0.5 * np.sum((alm_result["solution"] - theta_value) ** 2)) < 1e-5


def test_lapanda_box_on_variables_and_box_on_constraints_match_ipopt():
    u = ca.SX.sym("u", 3)
    theta = ca.SX.sym("theta", 3)
    variable = ca.SX.sym("variable", 1)
    theta_value = np.array([1.8, -0.9, 0.35])
    lower = np.array([0.0, -0.4, -0.25])
    upper = np.array([1.2, 0.8, 0.9])
    constraint_lower = np.array([0.7, -0.15])
    constraint_upper = np.array([1.0, 0.25])

    cost = 0.5 * ca.sumsqr(u - theta)
    constraints = ca.vertcat(
        u[0] + u[1] + u[2],
        u[0] - 2.0 * u[2],
    )

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=constraints,
    )
    alm_result = build_solver(problem, backend="callback").solve_lapanda(
        x0=np.array([0.7, 0.0, 0.2]),
        theta=theta_value,
        variable=np.zeros(1),
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        inner_solver_options=_solver_options(tolerance=1e-8, max_iterations=180),
        alm_options=_alm_options(tolerance=1e-6, max_iterations=50),
    )

    solver = _ipopt_solver({"x": u, "p": theta, "f": cost, "g": constraints})
    ipopt_result = solver(
        x0=np.array([0.7, 0.0, 0.2]),
        p=theta_value,
        lbx=lower,
        ubx=upper,
        lbg=constraint_lower,
        ubg=constraint_upper,
    )
    ipopt_solution = np.array(ipopt_result["x"]).reshape(-1)
    alm_constraint = np.array(
        [
            alm_result["solution"][0] + alm_result["solution"][1] + alm_result["solution"][2],
            alm_result["solution"][0] - 2.0 * alm_result["solution"][2],
        ]
    )

    np.testing.assert_allclose(alm_result["solution"], ipopt_solution, atol=8e-4)
    assert np.all(alm_result["solution"] >= lower - 1e-7)
    assert np.all(alm_result["solution"] <= upper + 1e-7)
    assert np.all(alm_constraint >= constraint_lower - 2e-6)
    assert np.all(alm_constraint <= constraint_upper + 2e-6)
    assert abs(float(ipopt_result["f"]) - 0.5 * np.sum((alm_result["solution"] - theta_value) ** 2)) < 1e-5
