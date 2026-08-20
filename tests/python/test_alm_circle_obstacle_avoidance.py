import tempfile
import time
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_circle_obstacle_problem(horizon=30, dt=0.12):
    nu = 2
    n = horizon * nu
    u = ca.SX.sym("u", n)
    theta = ca.SX.sym("theta", 4)
    variable = ca.SX.sym("variable", 1)

    px = theta[0]
    py = theta[1]
    target_x = theta[2]
    target_y = theta[3]
    obstacle = np.array([0.0, 0.0])
    safe_radius = 0.55

    position_weight = 10.0
    input_weight = 1e-2
    terminal_weight = 30.0

    cost = ca.SX(0)
    constraints = []
    for k in range(horizon):
        uk = u[2 * k : 2 * k + 2]
        px = px + dt * uk[0]
        py = py + dt * uk[1]

        dx = px - target_x
        dy = py - target_y
        cost += position_weight * (dx**2 + dy**2) + input_weight * ca.dot(uk, uk)

        obstacle_value = safe_radius**2 - ((px - obstacle[0]) ** 2 + (py - obstacle[1]) ** 2)
        constraints.append(obstacle_value)

    cost += terminal_weight * position_weight * ((px - target_x) ** 2 + (py - target_y) ** 2)

    lower = -0.5 * np.ones(n)
    upper = 0.5 * np.ones(n)
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=ca.vertcat(*constraints),
    )
    return problem, n, dt, obstacle, safe_radius


def _rollout(theta, solution, dt):
    px, py = theta[:2]
    points = []
    for control in np.asarray(solution).reshape(-1, 2):
        px = px + dt * control[0]
        py = py + dt * control[1]
        points.append([px, py])
    return np.asarray(points)


def _write_circle_plot(trajectory, start, target, obstacle, safe_radius, output_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=160)
    circle = plt.Circle(obstacle, safe_radius, color="#cc4c4c", alpha=0.25, label="safe obstacle")
    boundary = plt.Circle(obstacle, safe_radius, color="#9b2f2f", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.add_patch(boundary)

    plotted = np.vstack([start, trajectory])
    ax.plot(plotted[:, 0], plotted[:, 1], "o-", color="#1f77b4", linewidth=2.0, markersize=3.5, label="lapanda trajectory")
    ax.scatter([start[0]], [start[1]], marker="s", s=70, color="#2ca02c", label="start")
    ax.scatter([target[0]], [target[1]], marker="*", s=120, color="#ff7f0e", label="target")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Circle obstacle avoidance with ALM hard constraints")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def test_alm_circle_obstacle_avoidance_hard_constraint():
    problem, n, dt, obstacle, safe_radius = _build_circle_obstacle_problem()
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_alm_circle_obstacle",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    start = np.array([-1.2, 0.0])
    target = np.array([1.2, 0.0])
    theta = np.concatenate([start, target])
    variable = np.zeros(1)

    horizon = n // 2
    waypoint_x = np.linspace(start[0], target[0], horizon + 1)[1:]
    waypoint_y = 0.65 * np.sin(np.linspace(0, np.pi, horizon))
    waypoints = np.column_stack([waypoint_x, waypoint_y])
    previous = start.copy()
    warm_start = []
    for point in waypoints:
        warm_start.extend((point - previous) / dt)
        previous = point
    warm_start = np.asarray(warm_start)

    inner_options = SolverOptions()
    inner_options.max_iterations = 800
    inner_options.tolerance = 1e-2
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-3
    alm_options.initial_penalty = 1e3
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True

    start_time = time.perf_counter()
    result = solver.solve_lapanda(
        warm_start,
        theta,
        variable,
        -1.0e20 * np.ones(problem.constraints.numel()),
        np.zeros(problem.constraints.numel()),
        inner_solver_options=inner_options,
        alm_options=alm_options,
        backward_options=BackwardOptions(),
    )
    solve_time = time.perf_counter() - start_time

    solution = np.asarray(result["solution"])
    trajectory = _rollout(theta, solution, dt)
    distances = np.linalg.norm(trajectory - obstacle, axis=1)
    final_distance = np.linalg.norm(trajectory[-1] - target)
    inner_iterations = np.asarray(result["inner_iterations"], dtype=int)
    plot_path = Path(__file__).with_name("circle_obstacle_avoidance_debug.png")
    _write_circle_plot(trajectory, start, target, obstacle, safe_radius, plot_path)

    print("\nALM circle obstacle avoidance")
    print(f"solve time: {solve_time:.6f} s")
    print(
        "ALM summary: "
        f"outer={result['iterations']}, "
        f"residual={result['final_residual']:.6e}, "
        f"penalty={result['penalty']:.6e}, "
        f"inner_iterations={inner_iterations}"
    )
    print(f"inner max-hit count: {np.sum(inner_iterations >= inner_options.max_iterations)}")
    print(f"min distance to obstacle: {np.min(distances):.6f}")
    print(f"final point: {trajectory[-1]}")
    print(f"final distance: {final_distance:.6e}")
    print(f"plot path: {plot_path}")

    assert result["final_residual"] < 1e-3
    assert np.min(distances) >= safe_radius - 2e-3
    assert final_distance < 0.08
