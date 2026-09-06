import importlib.util
import tempfile
import time
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_quadratic_helpers():
    path = Path(__file__).with_name("test_alm_quadratic_band_obstacle_avoidance.py")
    spec = importlib.util.spec_from_file_location("quadratic_band_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shift_controls(solution):
    return np.concatenate([solution[2:], solution[-2:]])


def _shift_constraints(values):
    return np.concatenate([values[1:], values[-1:]])


def _initial_control_guess(horizon):
    return np.tile(np.array([0.6, np.deg2rad(30.0)]), horizon)


def _roll_one_step(helpers, state, target, control, dt):
    return helpers._rollout(np.concatenate([state, target]), control, dt)[-1]


def _write_rolling_plot(helpers, alm_traj, ipopt_traj, start, target, safety_margin, path):
    xs = np.linspace(-2.5, 1.8, 600)
    original_lower = xs**2
    original_upper = 1.0 + 0.5 * xs**2
    safe_lower = xs**2 - safety_margin
    safe_upper = 1.0 + 0.5 * xs**2 + safety_margin

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    ax.fill_between(
        xs,
        safe_lower,
        safe_upper,
        color="#cc4c4c",
        alpha=0.14,
        label="safety-expanded forbidden band",
    )
    ax.fill_between(
        xs,
        original_lower,
        original_upper,
        color="#cc4c4c",
        alpha=0.28,
        label="original quadratic band",
    )
    ax.plot(xs, safe_lower, color="#9b2f2f", linewidth=1.0, linestyle="--")
    ax.plot(xs, safe_upper, color="#9b2f2f", linewidth=1.0, linestyle="--")
    ax.plot(
        alm_traj[:, 0],
        alm_traj[:, 1],
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        markersize=2.8,
        label="rolling lapanda",
    )
    ax.plot(
        ipopt_traj[:, 0],
        ipopt_traj[:, 1],
        "s-",
        color="#ff7f0e",
        linewidth=1.5,
        markersize=2.2,
        label="rolling IPOPT",
    )
    ax.scatter([start[0]], [start[1]], marker="s", s=70, color="#2ca02c", label="start")
    ax.scatter([target[0]], [target[1]], marker="*", s=120, color="#d62728", label="target")
    ax.set_xlim(-2.4, 0.5)
    ax.set_ylim(0.0, 2.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Rolling MPC: quadratic-band obstacle")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run_rolling_lapanda(helpers, problem, n, dt, start, target, safety_margin):
    horizon = n // 2
    ncon = problem.constraints.numel()
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_rolling_quadratic_band_alm",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    inner_options = SolverOptions()
    inner_options.max_iterations = 2000
    inner_options.tolerance = 1e-3
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-4
    alm_options.initial_penalty = 1e4
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True

    variable = np.zeros(1)
    state = start.copy()
    control_warm_start = _initial_control_guess(horizon)
    multiplier_warm_start = np.zeros(ncon)
    penalty_warm_start = np.full(ncon, alm_options.initial_penalty)
    lower = np.zeros(ncon)
    upper = np.zeros(ncon)
    trajectory = [state.copy()]
    controls = []
    residuals = []
    outer_iterations = []
    inner_iteration_sums = []
    max_penalties = []

    solve_start = time.perf_counter()
    for _ in range(80):
        theta = np.concatenate([state, target])
        result = solver.solve_lapanda(
            control_warm_start,
            theta,
            variable,
            lower,
            upper,
            inner_solver_options=inner_options,
            alm_options=alm_options,
            backward_options=BackwardOptions(),
            multiplier0=multiplier_warm_start,
            penalty0=penalty_warm_start,
        )

        solution = np.asarray(result["solution"])
        multipliers = np.asarray(result["multipliers"])
        penalties = np.asarray(result["penalties"])
        control = solution[:2]
        controls.append(control)
        residuals.append(float(result["final_residual"]))
        outer_iterations.append(int(result["iterations"]))
        inner_iteration_sums.append(int(np.sum(np.asarray(result["inner_iterations"], dtype=int))))
        max_penalties.append(float(np.max(penalties)))

        state = _roll_one_step(helpers, state, target, control, dt)
        trajectory.append(state.copy())

        control_warm_start = _shift_controls(solution)
        multiplier_warm_start = _shift_constraints(multipliers)
        penalty_warm_start = _shift_constraints(penalties)

        if np.linalg.norm(state[:2] - target[:2]) < 0.15:
            break

    solve_time = time.perf_counter() - solve_start
    trajectory = np.asarray(trajectory)
    controls = np.asarray(controls)
    product = helpers._quadratic_band_product(trajectory[1:, :2], safety_margin)
    signed = helpers._quadratic_band_signed_violation(trajectory[1:, :2], safety_margin)
    return {
        "trajectory": trajectory,
        "controls": controls,
        "steps": len(controls),
        "solve_time": solve_time,
        "final_distance": float(np.linalg.norm(trajectory[-1, :2] - target[:2])),
        "max_product": float(np.max(product)) if product.size else 0.0,
        "max_signed_violation": float(np.max(signed)) if signed.size else 0.0,
        "max_residual": float(np.max(residuals)) if residuals else 0.0,
        "outer_iterations": outer_iterations,
        "inner_iteration_sums": inner_iteration_sums,
        "max_penalties": max_penalties,
    }


def _run_rolling_ipopt(helpers, problem, n, dt, start, target, safety_margin):
    horizon = n // 2
    ncon = problem.constraints.numel()
    u = ca.vec(problem.u)
    p = ca.vertcat(ca.vec(problem.theta), ca.vec(problem.variable))
    g = ca.vec(problem.constraints)
    solver = ca.nlpsol(
        "test_rolling_quadratic_band_ipopt",
        "ipopt",
        {"x": u, "p": p, "f": problem.cost, "g": g},
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 2000,
            "ipopt.warm_start_init_point": "yes",
        },
    )

    variable = np.zeros(1)
    state = start.copy()
    control_warm_start = _initial_control_guess(horizon)
    lam_x_warm_start = np.zeros(n)
    lam_g_warm_start = np.zeros(ncon)
    trajectory = [state.copy()]
    controls = []
    iteration_counts = []

    solve_start = time.perf_counter()
    for _ in range(80):
        solution = solver(
            x0=control_warm_start,
            p=np.concatenate([state, target, variable]),
            lbx=np.asarray(problem.box_lower, dtype=float),
            ubx=np.asarray(problem.box_upper, dtype=float),
            lbg=np.zeros(ncon),
            ubg=np.zeros(ncon),
            lam_x0=lam_x_warm_start,
            lam_g0=lam_g_warm_start,
        )
        control_sequence = np.asarray(solution["x"], dtype=float).reshape(-1)
        lam_x_warm_start = np.asarray(solution["lam_x"], dtype=float).reshape(-1)
        lam_g_warm_start = np.asarray(solution["lam_g"], dtype=float).reshape(-1)
        control = control_sequence[:2]
        controls.append(control)
        iteration_counts.append(int(solver.stats().get("iter_count", 0)))

        state = _roll_one_step(helpers, state, target, control, dt)
        trajectory.append(state.copy())

        control_warm_start = _shift_controls(control_sequence)
        lam_x_warm_start = _shift_controls(lam_x_warm_start)
        lam_g_warm_start = _shift_constraints(lam_g_warm_start)

        if np.linalg.norm(state[:2] - target[:2]) < 0.15:
            break

    solve_time = time.perf_counter() - solve_start
    trajectory = np.asarray(trajectory)
    controls = np.asarray(controls)
    product = helpers._quadratic_band_product(trajectory[1:, :2], safety_margin)
    signed = helpers._quadratic_band_signed_violation(trajectory[1:, :2], safety_margin)
    return {
        "trajectory": trajectory,
        "controls": controls,
        "steps": len(controls),
        "solve_time": solve_time,
        "final_distance": float(np.linalg.norm(trajectory[-1, :2] - target[:2])),
        "max_product": float(np.max(product)) if product.size else 0.0,
        "max_signed_violation": float(np.max(signed)) if signed.size else 0.0,
        "iteration_counts": iteration_counts,
    }


def test_rolling_quadratic_band_mpc_approaches_target_with_warm_starts():
    helpers = _load_quadratic_helpers()
    safety_margin = 0.15
    problem, n, dt = helpers._build_quadratic_band_problem(safety_margin=safety_margin)
    start = np.array([-0.6, 2.0, np.pi*0.8])
    target = np.array([-1.5, 1.0, -np.pi*0.2])

    alm = _run_rolling_lapanda(helpers, problem, n, dt, start, target, safety_margin)
    ipopt = _run_rolling_ipopt(helpers, problem, n, dt, start, target, safety_margin)

    plot_path = Path(__file__).with_name("rolling_quadratic_band_mpc_debug.png")
    _write_rolling_plot(
        helpers,
        alm["trajectory"],
        ipopt["trajectory"],
        start,
        target,
        safety_margin,
        plot_path,
    )

    print("\nRolling quadratic-band MPC")
    print(
        "lapanda: "
        f"steps={alm['steps']}, "
        f"solve_time={alm['solve_time']:.6f} s, "
        f"final_distance={alm['final_distance']:.6e}, "
        f"max_residual={alm['max_residual']:.6e}, "
        f"max_product={alm['max_product']:.6e}, "
        f"max_signed_violation={alm['max_signed_violation']:.6e}"
    )
    print(f"ALM outer iterations: {alm['outer_iterations']}")
    print(f"ALM inner iteration sums: {alm['inner_iteration_sums']}")
    print(f"ALM max penalties: {alm['max_penalties']}")
    print(
        "IPOPT: "
        f"steps={ipopt['steps']}, "
        f"solve_time={ipopt['solve_time']:.6f} s, "
        f"final_distance={ipopt['final_distance']:.6e}, "
        f"max_product={ipopt['max_product']:.6e}, "
        f"max_signed_violation={ipopt['max_signed_violation']:.6e}"
    )
    print(f"IPOPT iterations: {ipopt['iteration_counts']}")
    print(f"plot path: {plot_path}")

    assert alm["final_distance"] < 0.35
    assert ipopt["final_distance"] < 0.35
    assert alm["max_product"] < 1e-3
    assert ipopt["max_product"] < 1e-3
    assert np.all(alm["controls"][:, 0] >= -1e-9)
    assert np.all(alm["controls"][:, 0] <= 0.6 + 1e-9)
    assert np.all(alm["controls"][:, 1] >= -np.deg2rad(30.0) - 1e-9)
    assert np.all(alm["controls"][:, 1] <= np.deg2rad(30.0) + 1e-9)
    assert np.all(ipopt["controls"][:, 0] >= -1e-7)
    assert np.all(ipopt["controls"][:, 0] <= 0.6 + 1e-7)
    assert np.all(ipopt["controls"][:, 1] >= -np.deg2rad(30.0) - 1e-7)
    assert np.all(ipopt["controls"][:, 1] <= np.deg2rad(30.0) + 1e-7)
