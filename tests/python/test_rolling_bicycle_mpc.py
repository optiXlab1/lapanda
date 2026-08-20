import tempfile
import time

import casadi as ca
import numpy as np

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


def _build_bicycle_mpc_problem(horizon=12, dt=0.2, wheelbase=2.5, with_state_constraints=False):
    nu = 2
    n = horizon * nu

    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 6)
    variable = ca.SX.sym("variable", 1)

    x = theta[0]
    y = theta[1]
    psi = theta[2]
    target_x = theta[3]
    target_y = theta[4]
    target_psi = theta[5]

    cost = ca.SX(0)
    constraints = []
    for k in range(horizon):
        uk = u[2 * k : 2 * k + 2]
        v = uk[0]
        delta = uk[1]
        dx = x - target_x
        dy = y - target_y
        dpsi = ca.atan2(ca.sin(psi - target_psi), ca.cos(psi - target_psi))
        stage_weight = 1.0 + 4.0 * (k + 1) / horizon

        cost += (
            stage_weight * (dx**2 + dy**2)
            + 0.15 * dpsi**2
            + 0.03 * v**2
            + 0.05 * delta**2
        )
        if k > 0:
            previous = u[2 * (k - 1) : 2 * (k - 1) + 2]
            cost += 0.05 * (v - previous[0]) ** 2 + 0.1 * (delta - previous[1]) ** 2

        x = x + dt * v * ca.cos(psi)
        y = y + dt * v * ca.sin(psi)
        psi = psi + dt * v * ca.tan(delta) / wheelbase
        if with_state_constraints:
            constraints.append(x)
            constraints.append(y)

    dx = x - target_x
    dy = y - target_y
    dpsi = ca.atan2(ca.sin(psi - target_psi), ca.cos(psi - target_psi))
    cost += 25.0 * (dx**2 + dy**2) + 2.0 * dpsi**2

    lower = np.empty(n)
    upper = np.empty(n)
    lower[0::2] = 0.0
    upper[0::2] = 3.0
    lower[1::2] = -0.55
    upper[1::2] = 0.55

    problem_constraints = ca.vertcat(*constraints) if constraints else None
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=problem_constraints,
    )
    return problem, n, dt, wheelbase


def _step_bicycle(state, control, dt, wheelbase):
    next_state = state.copy()
    v, delta = control
    next_state[0] += dt * v * np.cos(state[2])
    next_state[1] += dt * v * np.sin(state[2])
    next_state[2] += dt * v * np.tan(delta) / wheelbase
    next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
    return next_state


def test_compiled_rolling_bicycle_mpc_reaches_target_with_one_build():
    problem, n, dt, wheelbase = _build_bicycle_mpc_problem()
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_rolling_bicycle_mpc",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )
    assert solver.generated_oracle.library_path is not None
    assert solver.generated_oracle.library_path.exists()

    solver_options = SolverOptions()
    solver_options.max_iterations = 800
    solver_options.tolerance = 1e-4
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 50

    backward_options = BackwardOptions()
    state = np.array([0.0, 0.0, 0.0])
    target = np.array([4.0, 2.0, 0.0])
    variable = np.zeros(1)
    warm_start = np.zeros(n)
    trajectory = [state.copy()]
    controls = []

    solve_start = time.perf_counter()
    for _ in range(35):
        theta = np.concatenate([state, target])
        result = solver.solve_panda(
            warm_start,
            theta,
            variable,
            solver_options=solver_options,
            backward_options=backward_options,
        )
        solution = np.asarray(result["solution"])
        control = solution[:2]
        controls.append(control)

        state = _step_bicycle(state, control, dt, wheelbase)
        trajectory.append(state.copy())
        warm_start = np.concatenate([solution[2:], solution[-2:]])

        if np.linalg.norm(state[:2] - target[:2]) < 0.2:
            break

    rolling_solve_time = time.perf_counter() - solve_start
    trajectory = np.asarray(trajectory)
    controls = np.asarray(controls)
    final_distance = np.linalg.norm(state[:2] - target[:2])

    print(f"rolling bicycle MPC solves: {len(controls)}")
    print(f"rolling bicycle MPC solve-only time: {rolling_solve_time:.6f} s")
    print(f"final state: {state}")
    print(f"final target distance: {final_distance:.6f}")

    assert len(controls) > 1
    assert len(controls) < 35
    assert final_distance < 0.25
    assert trajectory[-1, 0] > 3.8
    assert trajectory[-1, 1] > 1.8
    assert np.all(controls[:, 0] >= -1e-9)
    assert np.all(controls[:, 0] <= 3.0 + 1e-9)
    assert np.all(controls[:, 1] >= -0.55 - 1e-9)
    assert np.all(controls[:, 1] <= 0.55 + 1e-9)


def test_compiled_rolling_bicycle_alm_mpc_respects_state_constraint():
    problem, n, dt, wheelbase = _build_bicycle_mpc_problem(with_state_constraints=True)
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_rolling_bicycle_alm_mpc",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )
    assert solver.generated_oracle.library_path is not None
    assert solver.generated_oracle.library_path.exists()

    inner_options = SolverOptions()
    inner_options.max_iterations = 1000
    inner_options.tolerance = 1e-4
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 50

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-4
    alm_options.initial_penalty = 10.0
    alm_options.penalty_update_factor = 2.0
    alm_options.sufficient_decrease_factor = 0.25

    backward_options = BackwardOptions()
    state = np.array([0.0, 0.0, 0.0])
    target = np.array([4.0, 2.0, 0.0])
    variable = np.zeros(1)
    warm_start = np.zeros(n)
    x_upper = 3.5
    y_upper = 1.9
    constraint_lower = -1.0e20 * np.ones(problem.constraints.numel())
    constraint_upper = np.tile([x_upper, y_upper], n // 2)
    trajectory = [state.copy()]
    controls = []
    final_violations = []
    multiplier_warm_start = np.zeros(problem.constraints.numel())

    solve_start = time.perf_counter()
    for _ in range(35):
        theta = np.concatenate([state, target])
        result = solver.solve_lapanda(
            warm_start,
            theta,
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options=inner_options,
            alm_options=alm_options,
            backward_options=backward_options,
            multiplier0=multiplier_warm_start,
        )
        solution = np.asarray(result["solution"])
        multipliers = np.asarray(result["multipliers"])
        control = solution[:2]
        controls.append(control)
        final_violations.append(float(result["final_residual"]))

        state = _step_bicycle(state, control, dt, wheelbase)
        trajectory.append(state.copy())
        warm_start = np.concatenate([solution[2:], solution[-2:]])
        multiplier_warm_start = np.concatenate([multipliers[2:], multipliers[-2:]])

        if (
            abs(state[0] - x_upper) < 5e-3
            and state[1] > 1.45
        ):
            break

    rolling_solve_time = time.perf_counter() - solve_start
    trajectory = np.asarray(trajectory)
    controls = np.asarray(controls)
    final_distance = np.linalg.norm(state[:2] - target[:2])
    x_boundary_error = abs(state[0] - x_upper)

    print(f"rolling bicycle ALM-MPC solves: {len(controls)}")
    print(f"rolling bicycle ALM-MPC solve-only time: {rolling_solve_time:.6f} s")
    print(f"final state: {state}")
    print(f"final target distance: {final_distance:.6f}")
    print(f"final x-boundary error: {x_boundary_error:.6f}")
    print(f"max closed-loop x: {np.max(trajectory[:, 0]):.6f}")
    print(f"max closed-loop y: {np.max(trajectory[:, 1]):.6f}")
    print(f"max ALM residual: {np.max(final_violations):.6e}")

    assert len(controls) > 1
    assert len(controls) < 35
    assert x_boundary_error < 5e-3
    assert trajectory[-1, 1] > 1.45
    assert np.max(trajectory[:, 0]) <= x_upper + 1e-3
    assert np.max(trajectory[:, 1]) <= y_upper + 1e-6
    assert np.max(final_violations) < 1e-3
    assert np.all(controls[:, 0] >= -1e-9)
    assert np.all(controls[:, 0] <= 3.0 + 1e-9)
    assert np.all(controls[:, 1] >= -0.55 - 1e-9)
    assert np.all(controls[:, 1] <= 0.55 + 1e-9)
