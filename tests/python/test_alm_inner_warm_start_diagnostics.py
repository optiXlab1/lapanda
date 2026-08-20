import tempfile
import time

import casadi as ca
import numpy as np
import pytest

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver
from test_rolling_bicycle_mpc import _build_bicycle_mpc_problem, _step_bicycle


def _outer_loss(u):
    reference = ca.DM.zeros(u.numel())
    reference[0::2] = 1.0
    return ca.sumsqr(ca.vec(u) - reference)


def _with_outer_loss(problem):
    return CasadiProblem(
        u=problem.u,
        theta=problem.theta,
        variable=problem.variable,
        cost=problem.cost,
        box_lower=problem.box_lower,
        box_upper=problem.box_upper,
        constraints=problem.constraints,
        outer_loss=_outer_loss(problem.u),
    )


def _backward_options():
    options = BackwardOptions()
    options.enable = True
    options.tolerance = 1e-6
    options.max_iterations = 800
    options.constraint_penalty_scale = 100.0
    options.constraint_penalty_max = 10000.0
    return options


def _check_backward_result(result, ntheta):
    grad_theta = np.asarray(result["grad_theta"])
    assert grad_theta.shape == (ntheta,)
    assert np.all(np.isfinite(grad_theta))
    assert np.isfinite(float(result["backward_residual"]))
    return grad_theta


def _run_lapanda_case():
    horizon = 20
    problem, n, dt, wheelbase = _build_bicycle_mpc_problem(
        horizon=horizon,
        with_state_constraints=True,
    )
    problem = _with_outer_loss(problem)
    solver = build_solver(
        problem,
        backend="compiled",
        name="diag_h20_lapanda",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    inner_options = SolverOptions()
    inner_options.max_iterations = 800
    inner_options.tolerance = 0.5
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-3
    alm_options.initial_penalty = 2.0
    alm_options.penalty_update_factor = 2.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True

    backward_options = BackwardOptions()
    enabled_backward_options = _backward_options()
    state = np.array([0.0, 0.0, 0.0])
    target = np.array([4.0, 2.0, 0.0])
    variable = np.zeros(1)
    warm_start = np.zeros(n)
    multiplier_warm_start = np.zeros(problem.constraints.numel())

    x_upper = 3.5
    y_upper = 1.9
    constraint_lower = -1.0e20 * np.ones(problem.constraints.numel())
    constraint_upper = np.tile([x_upper, y_upper], horizon)

    outer_iterations = []
    inner_iteration_sums = []
    step_times = []
    backward_call_times = []
    backward_overhead_times = []
    backward_iterations = []
    backward_residuals = []
    grad_norms = []
    max_residual = 0.0
    trajectory = [state.copy()]

    for _ in range(35):
        step_start = time.perf_counter()
        result = solver.solve_lapanda(
            warm_start,
            np.concatenate([state, target]),
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options=inner_options,
            alm_options=alm_options,
            backward_options=backward_options,
            multiplier0=multiplier_warm_start,
        )
        forward_time = time.perf_counter() - step_start
        step_times.append(forward_time)

        backward_start = time.perf_counter()
        backward_result = solver.solve_lapanda(
            warm_start,
            np.concatenate([state, target]),
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options=inner_options,
            alm_options=alm_options,
            backward_options=enabled_backward_options,
            multiplier0=multiplier_warm_start,
        )
        backward_call_time = time.perf_counter() - backward_start
        backward_call_times.append(backward_call_time)
        backward_overhead_times.append(max(0.0, backward_call_time - forward_time))
        grad_theta = _check_backward_result(backward_result, problem.theta.numel())
        print(grad_theta)
        backward_iterations.append(int(backward_result["backward_iterations"]))
        backward_residuals.append(float(backward_result["backward_residual"]))
        grad_norms.append(float(np.linalg.norm(grad_theta)))

        solution = np.asarray(result["solution"])
        multipliers = np.asarray(result["multipliers"])
        inner_iterations = np.asarray(result["inner_iterations"], dtype=int)
        outer_iterations.append(int(result["iterations"]))
        inner_iteration_sums.append(int(np.sum(inner_iterations)))
        max_residual = max(max_residual, float(result["final_residual"]))

        state = _step_bicycle(state, solution[:2], dt, wheelbase)
        trajectory.append(state.copy())
        warm_start = np.concatenate([solution[2:], solution[-2:]])
        multiplier_warm_start = np.concatenate([multipliers[2:], multipliers[-2:]])

        if abs(state[0] - x_upper) < 5e-3 and state[1] > 1.45:
            break

    steps = len(step_times)
    trajectory = np.asarray(trajectory)
    return {
        "steps": steps,
        "ms_per_step": 1000.0 * np.sum(step_times) / steps,
        "total_inner_iterations": int(np.sum(inner_iteration_sums)),
        "total_outer_iterations": int(np.sum(outer_iterations)),
        "max_residual": max_residual,
        "final_state": state,
        "max_x": float(np.max(trajectory[:, 0])),
        "max_y": float(np.max(trajectory[:, 1])),
        "backward_ms_per_step": 1000.0 * np.sum(backward_overhead_times) / steps,
        "forward_backward_ms_per_step": 1000.0 * np.sum(backward_call_times) / steps,
        "backward_iterations": backward_iterations,
        "backward_residuals": backward_residuals,
        "grad_norms": grad_norms,
        "outer_iterations": outer_iterations,
        "inner_iteration_sums": inner_iteration_sums,
        "step_times_ms": [1000.0 * value for value in step_times],
    }


def _build_penalty_panda_problem(horizon, penalty_weight):
    constrained_problem, n, dt, wheelbase = _build_bicycle_mpc_problem(
        horizon=horizon,
        with_state_constraints=True,
    )
    x_upper = 3.5
    y_upper = 1.9
    constraint_upper = np.tile([x_upper, y_upper], horizon)
    violation = ca.fmax(constrained_problem.constraints - constraint_upper, 0.0)
    problem = CasadiProblem(
        u=constrained_problem.u,
        theta=constrained_problem.theta,
        variable=constrained_problem.variable,
        cost=constrained_problem.cost + penalty_weight * ca.dot(violation, violation),
        box_lower=constrained_problem.box_lower,
        box_upper=constrained_problem.box_upper,
        outer_loss=_outer_loss(constrained_problem.u),
    )
    return problem, n, dt, wheelbase


def _run_penalty_panda_case():
    horizon = 20
    penalty_weight = 1.0e3
    problem, n, dt, wheelbase = _build_penalty_panda_problem(horizon, penalty_weight)
    solver = build_solver(
        problem,
        backend="compiled",
        name="diag_h20_penalty_panda",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    solver_options = SolverOptions()
    solver_options.max_iterations = 800
    solver_options.tolerance = 1e-3
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 100

    backward_options = BackwardOptions()
    enabled_backward_options = _backward_options()
    state = np.array([0.0, 0.0, 0.0])
    target = np.array([4.0, 2.0, 0.0])
    variable = np.zeros(1)
    warm_start = np.zeros(n)
    x_upper = 3.5
    y_upper = 1.9
    trajectory = [state.copy()]
    iterations = []
    step_times = []
    backward_call_times = []
    backward_overhead_times = []
    backward_iterations = []
    backward_residuals = []
    grad_norms = []

    for _ in range(35):
        step_start = time.perf_counter()
        result = solver.solve_panda(
            warm_start,
            np.concatenate([state, target]),
            variable,
            solver_options=solver_options,
            backward_options=backward_options,
        )
        forward_time = time.perf_counter() - step_start
        step_times.append(forward_time)

        backward_start = time.perf_counter()
        backward_result = solver.solve_panda(
            warm_start,
            np.concatenate([state, target]),
            variable,
            solver_options=solver_options,
            backward_options=enabled_backward_options,
        )
        backward_call_time = time.perf_counter() - backward_start
        backward_call_times.append(backward_call_time)
        backward_overhead_times.append(max(0.0, backward_call_time - forward_time))
        grad_theta = _check_backward_result(backward_result, problem.theta.numel())
        print(grad_theta)
        backward_iterations.append(int(backward_result["backward_iterations"]))
        backward_residuals.append(float(backward_result["backward_residual"]))
        grad_norms.append(float(np.linalg.norm(grad_theta)))

        solution = np.asarray(result["solution"])
        iterations.append(int(result["iterations"]))
        state = _step_bicycle(state, solution[:2], dt, wheelbase)
        trajectory.append(state.copy())
        warm_start = np.concatenate([solution[2:], solution[-2:]])

        if abs(state[0] - x_upper) < 5e-3 and state[1] > 1.45:
            break

    steps = len(step_times)
    trajectory = np.asarray(trajectory)
    return {
        "steps": steps,
        "ms_per_step": 1000.0 * np.sum(step_times) / steps,
        "total_iterations": int(np.sum(iterations)),
        "final_state": state,
        "max_x": float(np.max(trajectory[:, 0])),
        "max_y": float(np.max(trajectory[:, 1])),
        "backward_ms_per_step": 1000.0 * np.sum(backward_overhead_times) / steps,
        "forward_backward_ms_per_step": 1000.0 * np.sum(backward_call_times) / steps,
        "backward_iterations": backward_iterations,
        "backward_residuals": backward_residuals,
        "grad_norms": grad_norms,
        "iterations": iterations,
        "penalty_weight": penalty_weight,
        "step_times_ms": [1000.0 * value for value in step_times],
    }


def _run_ipopt_case():
    horizon = 20
    problem, n, dt, wheelbase = _build_bicycle_mpc_problem(
        horizon=horizon,
        with_state_constraints=True,
    )
    constraints = ca.vec(problem.constraints)
    ipopt = ca.nlpsol(
        "rolling_bicycle_ipopt",
        "ipopt",
        {"x": problem.u, "p": problem.theta, "f": problem.cost, "g": constraints},
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": 1e-3,
            "ipopt.constr_viol_tol": 1e-3,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.max_iter": 1000,
        },
    )

    state = np.array([0.0, 0.0, 0.0])
    target = np.array([4.0, 2.0, 0.0])
    warm_start = np.zeros(n)
    x_upper = 3.5
    y_upper = 1.9
    constraint_lower = -1.0e20 * np.ones(problem.constraints.numel())
    constraint_upper = np.tile([x_upper, y_upper], horizon)

    step_times = []
    iteration_counts = []
    trajectory = [state.copy()]
    start = time.perf_counter()
    for _ in range(35):
        step_start = time.perf_counter()
        try:
            result = ipopt(
                x0=warm_start,
                p=np.concatenate([state, target]),
                lbx=problem.box_lower,
                ubx=problem.box_upper,
                lbg=constraint_lower,
                ubg=constraint_upper,
            )
        except RuntimeError as exc:
            pytest.skip(f"IPOPT is not available in this CasADi install: {exc}")
        step_times.append(time.perf_counter() - step_start)

        solution = np.asarray(result["x"]).reshape(-1)
        iteration_counts.append(int(ipopt.stats().get("iter_count", -1)))
        state = _step_bicycle(state, solution[:2], dt, wheelbase)
        trajectory.append(state.copy())
        warm_start = np.concatenate([solution[2:], solution[-2:]])

        if abs(state[0] - x_upper) < 5e-3 and state[1] > 1.45:
            break

    elapsed = time.perf_counter() - start
    steps = len(step_times)
    trajectory = np.asarray(trajectory)
    return {
        "steps": steps,
        "ms_per_step": 1000.0 * elapsed / steps,
        "total_iterations": int(np.sum(iteration_counts)),
        "final_state": state,
        "max_x": float(np.max(trajectory[:, 0])),
        "max_y": float(np.max(trajectory[:, 1])),
        "iteration_counts": iteration_counts,
        "step_times_ms": [1000.0 * value for value in step_times],
    }


def test_constrained_bicycle_mpc_solver_diagnostics():
    alm_row = _run_lapanda_case()
    penalty_row = _run_penalty_panda_case()
    ipopt_row = _run_ipopt_case()

    print("\nConstrained bicycle MPC solver diagnostics")
    print(
        "horizon=20, n=40, ncon=40, inner_tol=5e-1, "
        "alm_tol=1e-3, panda_penalty_tol=1e-3, ipopt_tol=1e-3"
    )

    print("\nlapanda")
    print(
        "steps, ms_per_step, total_inner_iterations, "
        "total_outer_iterations, max_residual: "
        f"{alm_row['steps']}, "
        f"{alm_row['ms_per_step']:.3f}, "
        f"{alm_row['total_inner_iterations']}, "
        f"{alm_row['total_outer_iterations']}, "
        f"{alm_row['max_residual']:.3e}"
    )
    print(f"  outer per MPC step: {alm_row['outer_iterations']}")
    print(f"  inner sum per MPC step: {alm_row['inner_iteration_sums']}")
    print(
        "  backward estimated ms/step, forward+backward ms/step: "
        f"{alm_row['backward_ms_per_step']:.3f}, "
        f"{alm_row['forward_backward_ms_per_step']:.3f}"
    )
    print(f"  backward iterations per MPC step: {alm_row['backward_iterations']}")
    print(f"  max backward residual: {np.max(alm_row['backward_residuals']):.3e}")
    print(f"  grad theta norm per MPC step: {alm_row['grad_norms']}")
    print(f"  max x/y: {alm_row['max_x']:.6f}, {alm_row['max_y']:.6f}")
    print(f"  final state: {alm_row['final_state']}")

    print("\nPenalty-PANDA")
    print(
        "steps, ms_per_step, total_iterations, penalty_weight: "
        f"{penalty_row['steps']}, "
        f"{penalty_row['ms_per_step']:.3f}, "
        f"{penalty_row['total_iterations']}, "
        f"{penalty_row['penalty_weight']:.1e}"
    )
    print(f"  PANDA iterations per MPC step: {penalty_row['iterations']}")
    print(
        "  backward estimated ms/step, forward+backward ms/step: "
        f"{penalty_row['backward_ms_per_step']:.3f}, "
        f"{penalty_row['forward_backward_ms_per_step']:.3f}"
    )
    print(f"  backward iterations per MPC step: {penalty_row['backward_iterations']}")
    print(f"  max backward residual: {np.max(penalty_row['backward_residuals']):.3e}")
    print(f"  grad theta norm per MPC step: {penalty_row['grad_norms']}")
    print(f"  max x/y: {penalty_row['max_x']:.6f}, {penalty_row['max_y']:.6f}")
    print(f"  final state: {penalty_row['final_state']}")

    print("\nCasADi IPOPT")
    print(
        "steps, ms_per_step, total_iterations: "
        f"{ipopt_row['steps']}, "
        f"{ipopt_row['ms_per_step']:.3f}, "
        f"{ipopt_row['total_iterations']}"
    )
    print(f"  IPOPT iterations per MPC step: {ipopt_row['iteration_counts']}")
    print(f"  max x/y: {ipopt_row['max_x']:.6f}, {ipopt_row['max_y']:.6f}")
    print(f"  final state: {ipopt_row['final_state']}")
