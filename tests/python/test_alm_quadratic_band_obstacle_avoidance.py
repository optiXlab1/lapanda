import tempfile
import time
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _angle_error(a, b):
    return ca.atan2(ca.sin(a - b), ca.cos(a - b))


def _bicycle_step(px, py, heading, control, dt, wheelbase):
    speed = control[0]
    steer = control[1]
    return (
        px + dt * speed * ca.cos(heading),
        py + dt * speed * ca.sin(heading),
        heading + dt * speed * ca.tan(steer) / wheelbase,
    )


def _build_quadratic_band_problem(horizon=40, dt=0.1, safety_margin=0.1):
    nu = 2
    n = horizon * nu
    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 6)
    variable = ca.SX.sym("variable", 1)

    px = theta[0]
    py = theta[1]
    heading = theta[2]
    target_x = theta[3]
    target_y = theta[4]
    target_heading = theta[5]

    position_weight = 1
    heading_weight = 0.01
    speed_weight = 0.01
    steer_weight = 0.01
    terminal_weight = 40.0
    wheelbase = 0.45

    cost = ca.SX(0)
    constraints = []
    for k in range(horizon):
        uk = u[2 * k : 2 * k + 2]
        px, py, heading = _bicycle_step(px, py, heading, uk, dt, wheelbase)

        dx = px - target_x
        dy = py - target_y
        dheading = _angle_error(heading, target_heading)
        cost += (
            position_weight * (dx**2 + dy**2)
            + heading_weight * dheading**2
            + speed_weight * uk[0] ** 2
            + steer_weight * uk[1] ** 2
        )

        lower_curve = px**2 - safety_margin
        upper_curve = 1.0 + 0.5 * px**2 + safety_margin
        h_lower = py - lower_curve
        h_upper = upper_curve - py
        band_product = 0.5 * ca.fmax(h_lower, 0.0) ** 2 * ca.fmax(h_upper, 0.0) ** 2
        constraints.append(band_product)

    terminal_heading_error = _angle_error(heading, target_heading)
    cost += terminal_weight * (
        position_weight * ((px - target_x) ** 2 + (py - target_y) ** 2)
        + heading_weight * terminal_heading_error**2
    )

    lower = np.tile(np.array([0, -np.deg2rad(30.0)]), horizon)
    upper = np.tile(np.array([0.6, np.deg2rad(30.0)]), horizon)
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=ca.vertcat(*constraints),
    )
    return problem, n, dt


def _rollout(theta, solution, dt, wheelbase=0.45):
    px, py, heading = theta[:3]
    states = []
    for control in np.asarray(solution).reshape(-1, 2):
        speed, steer = control
        px = px + dt * speed * np.cos(heading)
        py = py + dt * speed * np.sin(heading)
        heading = heading + dt * speed * np.tan(steer) / wheelbase
        heading = np.arctan2(np.sin(heading), np.cos(heading))
        states.append([px, py, heading])
    return np.asarray(states)


def _quadratic_band_product(points, safety_margin):
    x = points[:, 0]
    y = points[:, 1]
    lower_curve = x**2 - safety_margin
    upper_curve = 1.0 + 0.5 * x**2 + safety_margin
    h_lower = y - lower_curve
    h_upper = upper_curve - y
    return 0.5 * np.maximum(h_lower, 0.0) ** 2 * np.maximum(h_upper, 0.0) ** 2


def _quadratic_band_signed_violation(points, safety_margin):
    x = points[:, 0]
    y = points[:, 1]
    lower_curve = x**2 - safety_margin
    upper_curve = 1.0 + 0.5 * x**2 + safety_margin
    return np.minimum(y - lower_curve, upper_curve - y)


def _bicycle_warm_start(start, target, horizon, dt, wheelbase=0.45):
    waypoint_x = np.linspace(start[0], target[0], horizon + 1)[1:]
    waypoint_y = -0.45 * np.sin(np.linspace(0, np.pi, horizon))
    waypoints = np.column_stack([waypoint_x, waypoint_y])

    previous = start[:2].copy()
    previous_heading = start[2]
    warm_start = []
    for point in waypoints:
        diff = point - previous
        distance = np.linalg.norm(diff)
        desired_heading = np.arctan2(diff[1], diff[0]) if distance > 1e-12 else previous_heading
        speed = np.clip(distance / dt, -1.0, 1.0)
        heading_rate = np.arctan2(
            np.sin(desired_heading - previous_heading),
            np.cos(desired_heading - previous_heading),
        ) / dt
        steer = 0.0 if abs(speed) < 1e-8 else np.arctan(wheelbase * heading_rate / speed)
        warm_start.extend([speed, np.clip(steer, -np.deg2rad(30.0), np.deg2rad(30.0))])
        previous = point
        previous_heading = desired_heading
    return np.asarray(warm_start)


def _write_quadratic_band_plot(trajectory, start, target, safety_margin, output_path):
    xs = np.linspace(-1.6, 1.6, 500)
    original_lower = xs**2
    original_upper = 1.0 + 0.5 * xs**2
    safe_lower = xs**2 - safety_margin
    safe_upper = 1.0 + 0.5 * xs**2 + safety_margin

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=160)
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
    ax.plot(xs, safe_lower, color="#9b2f2f", linewidth=1.2, linestyle="--")
    ax.plot(xs, safe_upper, color="#9b2f2f", linewidth=1.2, linestyle="--")

    plotted = np.vstack([start[:2], trajectory[:, :2]])
    ax.plot(
        plotted[:, 0],
        plotted[:, 1],
        "o-",
        color="#1f77b4",
        linewidth=2.0,
        markersize=3.5,
        label="lapanda trajectory",
    )
    ax.scatter([start[0]], [start[1]], marker="s", s=70, color="#2ca02c", label="start")
    ax.scatter([target[0]], [target[1]], marker="*", s=120, color="#ff7f0e", label="target")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.8, 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bicycle model quadratic-band obstacle")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def test_alm_quadratic_band_obstacle_avoidance_with_bicycle_model():
    safety_margin = 0.15
    problem, n, dt = _build_quadratic_band_problem(safety_margin=safety_margin)
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_alm_quadratic_band_obstacle",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    start = np.array([0.8, 0.0, np.pi])
    target = np.array([-1.5, 1.0, -np.pi*0.2])
    theta = np.concatenate([start, target])
    variable = np.zeros(1)
    horizon = n // 2
    initial_guess = np.zeros(n)

    inner_options = SolverOptions()
    inner_options.max_iterations = 2000
    inner_options.tolerance = 1e-1
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-3
    alm_options.initial_penalty = 1e4
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True

    start_time = time.perf_counter()
    result = solver.solve_lapanda(
        initial_guess,
        theta,
        variable,
        np.zeros(problem.constraints.numel()),
        np.zeros(problem.constraints.numel()),
        inner_solver_options=inner_options,
        alm_options=alm_options,
        backward_options=BackwardOptions(),
    )
    solve_time = time.perf_counter() - start_time

    solution = np.asarray(result["solution"])
    trajectory = _rollout(theta, solution, dt)
    product_values = _quadratic_band_product(trajectory[:, :2], safety_margin)
    signed_violation = _quadratic_band_signed_violation(trajectory[:, :2], safety_margin)
    final_distance = np.linalg.norm(trajectory[-1, :2] - target[:2])
    inner_iterations = np.asarray(result["inner_iterations"], dtype=int)
    plot_path = Path(__file__).with_name("quadratic_band_obstacle_avoidance_debug.png")
    _write_quadratic_band_plot(trajectory, start, target, safety_margin, plot_path)

    print("\nALM bicycle quadratic-band obstacle avoidance")
    print(f"solve time: {solve_time:.6f} s")
    print(
        "ALM summary: "
        f"outer={result['iterations']}, "
        f"residual={result['final_residual']:.6e}, "
        f"penalty={result['penalty']:.6e}, "
        f"max_penalty={alm_options.max_penalty:.6e}, "
        f"inner_iterations={inner_iterations}"
    )
    print(f"inner max-hit count: {np.sum(inner_iterations >= inner_options.max_iterations)}")
    print(f"max quadratic-band product: {np.max(product_values):.6e}")
    print(f"max quadratic-band signed violation: {np.max(signed_violation):.6e}")
    print(f"final state: {trajectory[-1]}")
    print(f"final distance: {final_distance:.6e}")
    print(f"plot path: {plot_path}")

    assert result["final_residual"] < 1e-3
    assert np.max(product_values) < 1e-3
