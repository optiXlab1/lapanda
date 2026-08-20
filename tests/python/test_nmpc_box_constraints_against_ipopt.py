import casadi as ca
import numpy as np
import pytest

from lapanda import (
    BackwardOptions,
    CasadiProblem,
    SolverOptions,
    build_solver,
)


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
                "ipopt.max_iter": 1000,
            },
        )
    except RuntimeError as exc:
        pytest.skip(f"CasADi IPOPT plugin is not available: {exc}")


def _build_nmpc_problem(horizon=20):
    nu = 2
    n = horizon * nu
    dt = 0.1
    wheelbase = 2.5

    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 5)
    variable = ca.SX.sym("variable", 1)

    x = ca.vertcat(0, 0, 0)
    cost = ca.SX(0)
    outer_loss = ca.SX(0)

    for k in range(horizon):
        uk = u[2 * k : 2 * k + 2]
        cost += (
            theta[0] * (x[0] - 10) ** 2
            + theta[1] * (x[1] - 10) ** 2
            + theta[2] * x[2] ** 2
            + theta[3] * uk[0] ** 2
            + theta[4] * uk[1] ** 2
        )
        outer_loss += 20 * (uk[0] - 1) ** 2 + 20 * (uk[1] - 1) ** 2
        x = ca.vertcat(
            x[0] + dt * uk[0] * ca.cos(x[2]),
            x[1] + dt * uk[0] * ca.sin(x[2]),
            x[2] + dt * uk[0] * ca.tan(uk[1]) / wheelbase,
        )

    lower = np.empty(n)
    upper = np.empty(n)
    lower[0::2] = 0.0
    upper[0::2] = 4.0
    lower[1::2] = -0.5
    upper[1::2] = 0.5

    return u, theta, variable, cost, outer_loss, lower, upper


def test_nmpc_panda_solution_matches_ipopt_with_box_constraints():
    u, theta, variable, cost, outer_loss, lower, upper = _build_nmpc_problem()
    theta_value = np.array([20.0, 20.0, 1.0, 0.1, 0.1])
    x0 = np.zeros(u.numel())

    solver_options = SolverOptions()
    solver_options.max_iterations = 3000
    solver_options.tolerance = 1e-7
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 5

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        outer_loss=outer_loss,
    )
    panda_result = build_solver(problem, backend="callback").solve_panda(
        x0=x0,
        theta=theta_value,
        variable=np.zeros(1),
        solver_options=solver_options,
    )

    ipopt = _ipopt_solver({"x": u, "p": theta, "f": cost})
    ipopt_result = ipopt(x0=x0, p=theta_value, lbx=lower, ubx=upper)
    ipopt_solution = np.array(ipopt_result["x"]).reshape(-1)

    panda_solution = np.asarray(panda_result["solution"])
    panda_cost = float(ca.Function("cost", [u, theta], [cost])(panda_solution, theta_value))

    np.testing.assert_allclose(panda_solution, ipopt_solution, atol=5e-4)
    assert abs(panda_cost - float(ipopt_result["f"])) < 1e-3
    assert panda_result["final_residual"] < 1e-6


def test_nmpc_panda_backward_returns_finite_sensitivity():
    u, theta, variable, cost, outer_loss, lower, upper = _build_nmpc_problem()
    theta_value = np.array([20.0, 20.0, 1.0, 0.1, 0.1])

    solver_options = SolverOptions()
    solver_options.max_iterations = 1200
    solver_options.tolerance = 1e-6
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 5

    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-8
    backward_options.max_iterations = 500

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        outer_loss=outer_loss,
    )
    result = build_solver(problem, backend="callback").solve_panda(
        x0=np.zeros(u.numel()),
        theta=theta_value,
        variable=np.zeros(1),
        solver_options=solver_options,
        backward_options=backward_options,
    )

    grad_theta = np.asarray(result["grad_theta"])
    print(grad_theta)
    assert grad_theta.shape == (5,)
    assert np.all(np.isfinite(grad_theta))
    assert result["backward_residual"] < 1e-7
