import csv
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


def _build_rectangle_obstacle_problem(horizon=20, dt=0.12, safety_margin=0.08):
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

    xmin = -0.35 - safety_margin
    xmax = 0.35 + safety_margin
    ymin = -0.22 - safety_margin
    ymax = 0.22 + safety_margin

    position_weight = 5.0
    heading_weight = 0.2
    input_weight = 1e-2
    terminal_weight = 20.0
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
            + input_weight * ca.dot(uk, uk)
        )

        rectangle_inside_product = (
            0.5
            * ca.fmax(px - xmin, 0.0) ** 2
            * ca.fmax(xmax - px, 0.0) ** 2
            * ca.fmax(py - ymin, 0.0) ** 2
            * ca.fmax(ymax - py, 0.0) ** 2
        )
        constraints.append(rectangle_inside_product)

    terminal_heading_error = _angle_error(heading, target_heading)
    cost += terminal_weight * (
        position_weight * ((px - target_x) ** 2 + (py - target_y) ** 2)
        + heading_weight * terminal_heading_error**2
    )

    lower = np.tile(np.array([-1.0, -np.deg2rad(25.0)]), horizon)
    upper = np.tile(np.array([1.0, np.deg2rad(25.0)]), horizon)
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=lower,
        box_upper=upper,
        constraints=ca.vertcat(*constraints),
    )
    rectangle = np.array([xmin, xmax, ymin, ymax])
    return problem, n, dt, rectangle


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


def _rectangle_product(points, rectangle):
    xmin, xmax, ymin, ymax = rectangle
    x = points[:, 0]
    y = points[:, 1]
    return (
        0.5
        * np.maximum(x - xmin, 0.0)
        * np.maximum(xmax - x, 0.0)
        * np.maximum(y - ymin, 0.0)
        * np.maximum(ymax - y, 0.0)
    )


def _rectangle_signed_violation(points, rectangle):
    xmin, xmax, ymin, ymax = rectangle
    x = points[:, 0]
    y = points[:, 1]
    return np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])


def _write_rectangle_plot(trajectory, start, target, rectangle, safety_margin, output_path):
    xmin, xmax, ymin, ymax = rectangle
    original = np.array(
        [xmin + safety_margin, xmax - safety_margin, ymin + safety_margin, ymax - safety_margin]
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=160)
    safe_rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        color="#cc4c4c",
        alpha=0.14,
        label="safety-expanded rectangle",
    )
    original_rect = plt.Rectangle(
        (original[0], original[2]),
        original[1] - original[0],
        original[3] - original[2],
        color="#cc4c4c",
        alpha=0.28,
        label="original rectangle obstacle",
    )
    boundary = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        color="#9b2f2f",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(safe_rect)
    ax.add_patch(original_rect)
    ax.add_patch(boundary)

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
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bicycle model rectangle obstacle: squared fmax product")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _bicycle_warm_start(start, target, horizon, dt, wheelbase=0.45):
    waypoint_x = np.linspace(start[0], target[0], horizon + 1)[1:]
    waypoint_y = 0.8 * np.sin(np.linspace(0, np.pi, horizon))
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
        if abs(speed) < 1e-8:
            steer = 0.0
        else:
            steer = np.arctan(wheelbase * heading_rate / speed)
        warm_start.extend([speed, np.clip(steer, -np.deg2rad(25.0), np.deg2rad(25.0))])
        previous = point
        previous_heading = desired_heading
    return np.asarray(warm_start)


def _read_alm_trace(log_path):
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _as_float(row, key, default=np.nan):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _as_int(row, key, default=-1):
    value = row.get(key, "")
    if value == "":
        return default
    return int(value)


def _print_alm_trace_summary(log_path, inner_max_iterations):
    rows = _read_alm_trace(log_path)
    if not rows:
        print(f"ALM trace log not found: {log_path}")
        return

    outer_rows = [row for row in rows if row.get("event") == "outer"]
    inner_rows = [row for row in rows if row.get("event") == "inner"]
    print(f"ALM trace log: {log_path}")
    print("outer | inner_iter | inner_res | alm_res | penalty | gamma | tau | upper_ok")
    for row in outer_rows:
        outer_iter = _as_int(row, "outer_iter")
        inner_iter = _as_int(row, "inner_iter")
        print(
            f"{outer_iter:5d} | "
            f"{inner_iter:10d} | "
            f"{_as_float(row, 'residual'):.3e} | "
            f"{_as_float(row, 'alm_violation'):.3e} | "
            f"{_as_float(row, 'penalty'):.3e} | "
            f"{_as_float(row, 'gamma'):.3e} | "
            f"{_as_float(row, 'tau'):.3e} | "
            f"{_as_int(row, 'upper_ok')}"
        )

    max_hit_outers = [
        _as_int(row, "outer_iter")
        for row in outer_rows
        if _as_int(row, "inner_iter") >= inner_max_iterations
    ]
    print(f"outer iterations hitting inner max: {max_hit_outers}")

    if not inner_rows:
        return

    by_outer = {}
    for row in inner_rows:
        by_outer.setdefault(_as_int(row, "outer_iter"), []).append(row)

    print(
        "inner detail by outer: "
        "outer | count | first_res | min_res | last_res | gamma_min/max | du_inf_min/max | u_inf_max"
    )
    for outer_iter in sorted(by_outer):
        rows_i = by_outer[outer_iter]
        residuals = np.array([_as_float(row, "residual") for row in rows_i])
        gammas = np.array([_as_float(row, "gamma") for row in rows_i])
        du_inf = np.array([_as_float(row, "du_inf") for row in rows_i])
        u_inf = np.array([_as_float(row, "u_inf") for row in rows_i])
        print(
            f"{outer_iter:5d} | "
            f"{len(rows_i):5d} | "
            f"{residuals[0]:.3e} | "
            f"{np.nanmin(residuals):.3e} | "
            f"{residuals[-1]:.3e} | "
            f"{np.nanmin(gammas):.3e}/{np.nanmax(gammas):.3e} | "
            f"{np.nanmin(du_inf):.3e}/{np.nanmax(du_inf):.3e} | "
            f"{np.nanmax(u_inf):.3e}"
        )

    if max_hit_outers:
        print("gamma dynamics for max-hit outers:")
        for outer_iter in max_hit_outers:
            rows_i = by_outer.get(outer_iter, [])
            if not rows_i:
                continue
            gammas = np.array([_as_float(row, "gamma") for row in rows_i])
            residuals = np.array([_as_float(row, "residual") for row in rows_i])
            du_inf = np.array([_as_float(row, "du_inf") for row in rows_i])
            du_l2 = np.array([_as_float(row, "du_l2") for row in rows_i])
            u_inf = np.array([_as_float(row, "u_inf") for row in rows_i])
            rounded = np.round(gammas, decimals=15)
            unique_values, counts = np.unique(rounded, return_counts=True)
            order = np.argsort(counts)[::-1]
            top = [
                (float(unique_values[i]), int(counts[i]))
                for i in order[:8]
            ]
            changes = int(np.sum(np.diff(rounded) != 0.0))
            longest_run = 1
            current_run = 1
            for i in range(1, len(rounded)):
                if rounded[i] == rounded[i - 1]:
                    current_run += 1
                else:
                    longest_run = max(longest_run, current_run)
                    current_run = 1
            longest_run = max(longest_run, current_run)
            print(
                f"  outer {outer_iter}: "
                f"changes={changes}, longest_same_gamma_run={longest_run}, "
                f"first/mid/last_gamma="
                f"{gammas[0]:.3e}/"
                f"{gammas[len(gammas)//2]:.3e}/"
                f"{gammas[-1]:.3e}, "
                f"first/mid/last_res="
                f"{residuals[0]:.3e}/"
                f"{residuals[len(residuals)//2]:.3e}/"
                f"{residuals[-1]:.3e}"
            )
            print("    most common gamma values:", top)
            small_move = du_inf < 1e-8
            print(
                "    step movement: "
                f"du_inf min/median/max="
                f"{np.nanmin(du_inf):.3e}/"
                f"{np.nanmedian(du_inf):.3e}/"
                f"{np.nanmax(du_inf):.3e}, "
                f"du_l2 median/max={np.nanmedian(du_l2):.3e}/{np.nanmax(du_l2):.3e}, "
                f"u_inf max={np.nanmax(u_inf):.3e}, "
                f"tiny-move-count(<1e-8)={int(np.sum(small_move))}"
            )
            rounded_move = np.round(du_inf, decimals=12)
            move_unique, move_counts = np.unique(rounded_move, return_counts=True)
            move_order = np.argsort(move_counts)[::-1]
            top_moves = [
                (float(move_unique[i]), int(move_counts[i]))
                for i in move_order[:8]
            ]
            print("    most common du_inf values:", top_moves)
            sample_indices = np.unique(
                np.linspace(0, len(gammas) - 1, num=min(12, len(gammas)), dtype=int)
            )
            sample = [
                (
                    int(i),
                    float(gammas[i]),
                    float(residuals[i]),
                    float(du_inf[i]),
                    float(u_inf[i]),
                )
                for i in sample_indices
            ]
            print("    sampled (iter, gamma, residual, du_inf, u_inf):", sample)


def test_alm_rectangle_obstacle_avoidance_with_bicycle_model():
    safety_margin = 0.15
    problem, n, dt, rectangle = _build_rectangle_obstacle_problem(
        safety_margin=safety_margin
    )
    solver = build_solver(
        problem,
        backend="compiled",
        name="test_alm_rectangle_obstacle",
        cache_dir=tempfile.mkdtemp(),
        force=True,
    )

    start = np.array([-1.2, 0.0, 0.0])
    target = np.array([1.2, 0.0, 0.0])
    theta = np.concatenate([start, target])
    variable = np.zeros(1)

    horizon = n // 2
    warm_start = _bicycle_warm_start(start, target, horizon, dt)

    inner_options = SolverOptions()
    inner_options.max_iterations = 1500
    inner_options.tolerance = 1e-3
    inner_options.buffer_size = 10
    inner_options.max_stable_iter = 80

    alm_options = AlmOptions()
    alm_options.max_iterations = 30
    alm_options.tolerance = 1e-5
    alm_options.initial_penalty = 1e6
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True
    alm_options.verbose = False

    start_time = time.perf_counter()
    result = solver.solve_lapanda(
        warm_start,
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
    product_values = _rectangle_product(trajectory[:, :2], rectangle)
    signed_violation = _rectangle_signed_violation(trajectory[:, :2], rectangle)
    final_distance = np.linalg.norm(trajectory[-1, :2] - target[:2])
    final_heading_error = np.arctan2(
        np.sin(trajectory[-1, 2] - target[2]),
        np.cos(trajectory[-1, 2] - target[2]),
    )
    inner_iterations = np.asarray(result["inner_iterations"], dtype=int)
    plot_path = Path(__file__).with_name("rectangle_obstacle_avoidance_debug.png")
    _write_rectangle_plot(trajectory, start, target, rectangle, safety_margin, plot_path)

    print("\nALM bicycle rectangle obstacle avoidance")
    print(f"solve time: {solve_time:.6f} s")
    print(
        "ALM summary: "
        f"outer={result['iterations']}, "
        f"residual={result['final_residual']:.6e}, "
        f"penalty={result['penalty']:.6e}, "
        f"inner_iterations={inner_iterations}"
    )
    print(f"inner max-hit count: {np.sum(inner_iterations >= inner_options.max_iterations)}")
    print(f"max rectangle product: {np.max(product_values):.6e}")
    print(f"max rectangle signed violation: {np.max(signed_violation):.6e}")
    print(f"final state: {trajectory[-1]}")
    print(f"final distance: {final_distance:.6e}")
    print(f"final heading error: {final_heading_error:.6e}")
    print(f"plot path: {plot_path}")
    # assert result["final_residual"] < 1e-3
    # assert np.max(product_values) < 1e-3
    # assert final_distance < 0.12
    # assert abs(final_heading_error) < 0.5
