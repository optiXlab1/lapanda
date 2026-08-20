import tempfile
import time

import casadi as ca
import numpy as np
import pytest

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


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

    acceleration_budget = ca.sum1(u[0::2])
    constraints = ca.vertcat(acceleration_budget)

    return u, theta, variable, cost, outer_loss, constraints, lower, upper


def _ipopt_solver(nlp, tol=1e-4):
    try:
        return ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.tol": tol,
                "ipopt.constr_viol_tol": tol,
                "ipopt.acceptable_tol": tol,
                "ipopt.max_iter": 1000,
            },
        )
    except RuntimeError as exc:
        pytest.skip(f"CasADi IPOPT plugin is not available: {exc}")


def _solver_options(tol=1e-4):
    options = SolverOptions()
    options.max_iterations = 1000
    options.tolerance = tol
    options.buffer_size = 10
    options.max_stable_iter = 80
    return options


def _alm_options(tol=1e-4):
    options = AlmOptions()
    options.max_iterations = 50
    options.tolerance = tol
    options.initial_penalty = 1.0
    options.penalty_update_factor = 2.0
    options.sufficient_decrease_factor = 0.25
    return options


def _casadi_sqp_reference(horizon=20):
    opti = ca.Opti()
    nu = 2
    u = opti.variable(horizon * nu)
    theta = opti.parameter(5)
    dt = 0.1
    wheelbase = 2.5

    x = ca.vertcat(0, 0, 0)
    cost = ca.MX(0)
    outer_loss = ca.MX(0)
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

    opti.minimize(cost)
    opti.subject_to(opti.bounded(0, u[0::2], 4))
    opti.subject_to(opti.bounded(-0.5, u[1::2], 0.5))
    acceleration_budget = ca.sum1(u[0::2])
    opti.subject_to(opti.bounded(0, acceleration_budget, 70))
    opti.solver(
        "sqpmethod",
        {
            "qpsol": "qrqp",
            "print_header": False,
            "print_iteration": False,
            "print_time": False,
        },
    )

    solve_fun = opti.to_function(
        "casadi_sqp_solve",
        [theta, u],
        [u, cost, outer_loss, acceleration_budget],
    )
    loss_fun = opti.to_function("casadi_sqp_loss", [theta, u], [outer_loss])

    th = ca.MX.sym("th", 5)
    u0 = ca.MX.sym("u0", horizon * nu)
    loss_expr = loss_fun(th, u0)
    sens_fun = ca.Function(
        "casadi_sqp_sens",
        [th, u0],
        [loss_expr, ca.jacobian(loss_expr, th)],
    )
    return solve_fun, sens_fun


def test_compiled_panda_nmpc_box_matches_ipopt_forward():
    u, theta, variable, cost, outer_loss, constraints, lower, upper = _build_nmpc_problem()
    del constraints
    theta_value = np.array([20.0, 20.0, 1.0, 0.1, 0.1])
    x0 = np.zeros(u.numel())

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        outer_loss=outer_loss,
    )
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_compiled_panda_nmpc",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )
    solver_options = _solver_options()
    backward_options = BackwardOptions()

    start = time.perf_counter()
    panda_result = solver.solve_panda(
        x0,
        theta_value,
        np.zeros(1),
        solver_options=solver_options,
        backward_options=backward_options,
    )
    panda_time = time.perf_counter() - start

    ipopt = _ipopt_solver({"x": u, "p": theta, "f": cost})
    start = time.perf_counter()
    ipopt_result = ipopt(x0=x0, p=theta_value, lbx=lower, ubx=upper)
    ipopt_time = time.perf_counter() - start

    panda_solution = np.asarray(panda_result["solution"])
    ipopt_solution = np.asarray(ipopt_result["x"]).reshape(-1)
    cost_fun = ca.Function("cost_fun", [u, theta], [cost])
    panda_cost = float(cost_fun(panda_solution, theta_value))
    ipopt_cost = float(ipopt_result["f"])

    print(f"compiled PANDA time: {panda_time:.6f} s")
    print(f"IPOPT time:          {ipopt_time:.6f} s")
    print(f"compiled PANDA iterations: {panda_result['iterations']}")

    np.testing.assert_allclose(panda_solution, ipopt_solution, atol=3e-2)
    assert abs(panda_cost - ipopt_cost) < 2e-3


def test_compiled_alm_nmpc_budget_matches_ipopt_forward_and_backward():
    u, theta, variable, cost, outer_loss, constraints, lower, upper = _build_nmpc_problem()
    theta_value = np.array([20.0, 20.0, 1.0, 0.1, 0.1])
    x0 = np.zeros(u.numel())
    constraint_lower = np.array([0.0])
    constraint_upper = np.array([70.0])

    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=constraints,
        outer_loss=outer_loss,
    )
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_compiled_alm_nmpc",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )
    inner_options = _solver_options()
    alm_options = _alm_options()
    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-8
    backward_options.max_iterations = 800
    backward_options.constraint_penalty_scale = 1000.0
    backward_options.constraint_penalty_max = 10000.0

    start = time.perf_counter()
    alm_result = solver.solve_lapanda(
        x0,
        theta_value,
        np.zeros(1),
        constraint_lower,
        constraint_upper,
        inner_solver_options=inner_options,
        alm_options=alm_options,
        backward_options=backward_options,
    )
    alm_time = time.perf_counter() - start

    ipopt = _ipopt_solver({"x": u, "p": theta, "f": cost, "g": constraints})
    start = time.perf_counter()
    ipopt_result = ipopt(
        x0=x0,
        p=theta_value,
        lbx=lower,
        ubx=upper,
        lbg=constraint_lower,
        ubg=constraint_upper,
    )
    ipopt_time = time.perf_counter() - start

    alm_solution = np.asarray(alm_result["solution"])
    ipopt_solution = np.asarray(ipopt_result["x"]).reshape(-1)
    cost_fun = ca.Function("cost_fun", [u, theta], [cost])
    budget_fun = ca.Function("budget_fun", [u], [constraints])
    alm_cost = float(cost_fun(alm_solution, theta_value))
    ipopt_cost = float(ipopt_result["f"])
    alm_budget = np.asarray(budget_fun(alm_solution)).reshape(-1)
    ipopt_budget = np.asarray(budget_fun(ipopt_solution)).reshape(-1)
    grad_theta = np.asarray(alm_result["grad_theta"])

    casadi_solve, casadi_sens = _casadi_sqp_reference(horizon=20)
    start = time.perf_counter()
    casadi_solution, casadi_cost, casadi_loss, casadi_budget = casadi_solve(
        theta_value,
        x0,
    )
    casadi_forward_time = time.perf_counter() - start
    start = time.perf_counter()
    casadi_loss_from_sens, casadi_grad = casadi_sens(theta_value, x0)
    casadi_backward_time = time.perf_counter() - start
    casadi_solution = np.asarray(casadi_solution).reshape(-1)
    casadi_grad = np.asarray(casadi_grad).reshape(-1)

    print(f"compiled ALM+PANDA forward+backward time: {alm_time:.6f} s")
    print(f"IPOPT forward time:                      {ipopt_time:.6f} s")
    print(f"CasADi SQP forward time:                 {casadi_forward_time:.6f} s")
    print(f"CasADi SQP backward time:                {casadi_backward_time:.6f} s")
    print(
        "compiled ALM summary: "
        f"outer={alm_result['iterations']}, "
        f"residual={alm_result['final_residual']:.6e}, "
        f"penalty={alm_result['penalty']:.6e}, "
        f"backward={alm_result['backward_iterations']}"
    )
    print(f"compiled ALM grad_theta: {grad_theta}")
    print(f"CasADi SQP grad_theta:   {casadi_grad}")
    print(f"grad difference:         {grad_theta - casadi_grad}")
    print(f"grad difference norm:    {np.linalg.norm(grad_theta - casadi_grad):.6e}")
    print(
        "CasADi SQP summary: "
        f"cost={float(casadi_cost):.6f}, "
        f"loss={float(casadi_loss):.6f}, "
        f"loss_from_sens={float(casadi_loss_from_sens):.6f}, "
        f"budget={float(casadi_budget):.6f}"
    )

    np.testing.assert_allclose(alm_solution, ipopt_solution, atol=8e-4)
    np.testing.assert_allclose(alm_budget, ipopt_budget, atol=1e-4)
    np.testing.assert_allclose(alm_solution, casadi_solution, atol=8e-4)
    assert abs(alm_cost - ipopt_cost) < 5e-3
    assert np.all(np.isfinite(grad_theta))
    assert np.all(np.isfinite(casadi_grad))
