import tempfile
import time

import casadi as ca
import numpy as np

from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver
from test_alm_circle_obstacle_avoidance import _build_circle_obstacle_problem


def _circle_initial_guess(start, target, horizon, dt):
    waypoint_x = np.linspace(start[0], target[0], horizon + 1)[1:]
    waypoint_y = 0.65 * np.sin(np.linspace(0, np.pi, horizon))
    previous = start.copy()
    warm_start = []
    for point in np.column_stack([waypoint_x, waypoint_y]):
        warm_start.extend((point - previous) / dt)
        previous = point
    return np.asarray(warm_start, dtype=float)


def _shift_controls(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.concatenate([values[2:], values[-2:]])


def _shift_constraints(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.concatenate([values[1:], values[-1:]])


def _circle_metrics(trajectory, obstacle, safe_radius, target):
    points = np.asarray(trajectory)[1:]
    distances = np.linalg.norm(points - obstacle, axis=1)
    final_distance = np.linalg.norm(np.asarray(trajectory)[-1] - target)
    return {
        "min_distance": float(np.min(distances)),
        "max_violation": float(np.max(safe_radius - distances)),
        "final_distance": float(final_distance),
    }


def _build_ipopt_solver(problem):
    u = ca.vec(problem.u)
    p = ca.vertcat(ca.vec(problem.theta), ca.vec(problem.variable))
    g = ca.vec(problem.constraints)
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-3,
        "ipopt.constr_viol_tol": 1e-3,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.max_iter": 2000,
        "ipopt.warm_start_init_point": "yes",
    }
    return ca.nlpsol(
        "rolling_circle_mpc_ipopt",
        "ipopt",
        {"x": u, "p": p, "f": problem.cost, "g": g},
        opts,
    )


def _run_rolling_ipopt(problem, n, dt, obstacle, safe_radius):
    horizon = n // 2
    start = np.array([-1.2, 0.0])
    target = np.array([1.2, 0.0])
    variable = np.zeros(1)

    build_start = time.perf_counter()
    solver = _build_ipopt_solver(problem)
    build_time = time.perf_counter() - build_start

    u0 = _circle_initial_guess(start, target, horizon, dt)
    lam_x0 = np.zeros(n)
    lam_g0 = np.zeros(horizon)
    lower = np.asarray(problem.box_lower).reshape(-1)
    upper = np.asarray(problem.box_upper).reshape(-1)
    constraint_lower = -1.0e20 * np.ones(horizon)
    constraint_upper = np.zeros(horizon)

    state = start.copy()
    trajectory = [state.copy()]
    controls = []
    iterations = []
    per_step_times = []

    solve_start = time.perf_counter()
    for _ in range(50):
        theta = np.concatenate([state, target])
        step_start = time.perf_counter()
        result = solver(
            x0=u0,
            p=np.concatenate([theta, variable]),
            lbx=lower,
            ubx=upper,
            lbg=constraint_lower,
            ubg=constraint_upper,
            lam_x0=lam_x0,
            lam_g0=lam_g0,
        )
        per_step_times.append(time.perf_counter() - step_start)

        solution = np.asarray(result["x"]).reshape(-1)
        control = solution[:2]
        controls.append(control)
        state = state + dt * control
        trajectory.append(state.copy())

        stats = solver.stats()
        iterations.append(int(stats.get("iter_count", -1)))
        u0 = _shift_controls(solution)
        lam_x0 = _shift_controls(result["lam_x"])
        lam_g0 = _shift_constraints(result["lam_g"])

        if np.linalg.norm(state - target) < 0.15:
            break

    solve_time = time.perf_counter() - solve_start
    metrics = _circle_metrics(trajectory, obstacle, safe_radius, target)
    return {
        "name": "IPOPT",
        "build_time": build_time,
        "solve_time": solve_time,
        "per_step_times": per_step_times,
        "steps": len(controls),
        "iterations": iterations,
        **metrics,
    }


def _run_rolling_lapanda(problem, n, dt, obstacle, safe_radius):
    horizon = n // 2
    start = np.array([-1.2, 0.0])
    target = np.array([1.2, 0.0])
    variable = np.zeros(1)

    build_start = time.perf_counter()
    solver = build_solver(
        problem,
        backend="compiled",
        name="rolling_circle_mpc_alm",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )
    build_time = time.perf_counter() - build_start

    inner_options = SolverOptions()
    inner_options.max_iterations = 800
    inner_options.tolerance = 1e-1
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-3
    alm_options.initial_penalty = 1e3
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True

    u0 = _circle_initial_guess(start, target, horizon, dt)
    multiplier0 = np.zeros(horizon)
    penalty0 = alm_options.initial_penalty * np.ones(horizon)
    constraint_lower = -1.0e20 * np.ones(horizon)
    constraint_upper = np.zeros(horizon)

    state = start.copy()
    trajectory = [state.copy()]
    controls = []
    outer_iterations = []
    inner_total_iterations = []
    residuals = []
    per_step_times = []

    solve_start = time.perf_counter()
    for _ in range(50):
        theta = np.concatenate([state, target])
        step_start = time.perf_counter()
        result = solver.solve_lapanda(
            u0,
            theta,
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options=inner_options,
            alm_options=alm_options,
            backward_options=BackwardOptions(),
            multiplier0=multiplier0
        )
        per_step_times.append(time.perf_counter() - step_start)

        solution = np.asarray(result["solution"]).reshape(-1)
        control = solution[:2]
        controls.append(control)
        state = state + dt * control
        trajectory.append(state.copy())

        outer_iterations.append(int(result["iterations"]))
        inner_total_iterations.append(int(np.sum(np.asarray(result["inner_iterations"], dtype=int))))
        residuals.append(float(result["final_residual"]))
        u0 = _shift_controls(solution)
        multiplier0 = _shift_constraints(result["multipliers"])
        penalty0 = _shift_constraints(result["penalties"])

        if np.linalg.norm(state - target) < 0.15:
            break

    solve_time = time.perf_counter() - solve_start
    metrics = _circle_metrics(trajectory, obstacle, safe_radius, target)
    return {
        "name": "lapanda",
        "build_time": build_time,
        "solve_time": solve_time,
        "per_step_times": per_step_times,
        "steps": len(controls),
        "outer_iterations": outer_iterations,
        "inner_total_iterations": inner_total_iterations,
        "residuals": residuals,
        **metrics,
    }


def _print_summary(row):
    per_step = np.asarray(row["per_step_times"], dtype=float)
    print(f"\n{row['name']} rolling circle MPC")
    print(f"build time              = {row['build_time']:.6f} s")
    print(f"solve-only total time   = {row['solve_time']:.6f} s")
    print(f"steps                   = {row['steps']}")
    print(f"mean per step           = {np.mean(per_step):.6f} s")
    print(f"median per step         = {np.median(per_step):.6f} s")
    print(f"min/max per step        = {np.min(per_step):.6f} / {np.max(per_step):.6f} s")
    print(f"final distance          = {row['final_distance']:.6e}")
    print(f"min distance to obstacle= {row['min_distance']:.6f}")
    print(f"max circle violation    = {row['max_violation']:.6e}")
    if row["name"] == "IPOPT":
        print(f"IPOPT iterations        = {row['iterations']}")
        print(f"IPOPT total iterations  = {sum(i for i in row['iterations'] if i >= 0)}")
    else:
        print(f"ALM outer iterations    = {row['outer_iterations']}")
        print(f"ALM residual max/final  = {max(row['residuals']):.6e} / {row['residuals'][-1]:.6e}")
        print(f"PANDA inner iterations  = {sum(row['inner_total_iterations'])}")


def test_rolling_circle_mpc_ipopt_vs_lapanda_timing():
    problem, n, dt, obstacle, safe_radius = _build_circle_obstacle_problem()

    ipopt = _run_rolling_ipopt(problem, n, dt, obstacle, safe_radius)
    alm = _run_rolling_lapanda(problem, n, dt, obstacle, safe_radius)

    _print_summary(ipopt)
    _print_summary(alm)

    assert ipopt["final_distance"] < 0.15
    assert alm["final_distance"] < 0.15
    assert ipopt["min_distance"] >= safe_radius - 2e-3
    assert alm["min_distance"] >= safe_radius - 2e-3
