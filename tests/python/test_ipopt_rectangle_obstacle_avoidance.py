import time
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

from test_alm_rectangle_obstacle_avoidance import (
    _bicycle_warm_start,
    _build_rectangle_obstacle_problem,
    _rectangle_product,
    _rectangle_signed_violation,
    _rollout,
    _write_rectangle_plot,
)


matplotlib.use("Agg")


def _ipopt_options():
    return {
        "print_time": False,
        "ipopt.print_level": 5,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-6,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.max_iter": 2000,
    }


def test_ipopt_rectangle_obstacle_feasibility():
    safety_margin = 0.14
    problem, n, dt, rectangle = _build_rectangle_obstacle_problem(
        safety_margin=safety_margin
    )

    start = np.array([-1.2, 0.0, 0.0])
    target = np.array([1.2, 0.0, 0.0])
    theta = np.concatenate([start, target])
    variable = np.zeros(1)
    horizon = n // 2
    warm_start = _bicycle_warm_start(start, target, horizon, dt)

    u = ca.vec(problem.u)
    p = ca.vertcat(ca.vec(problem.theta), ca.vec(problem.variable))
    g = ca.vec(problem.constraints)
    solver = ca.nlpsol(
        "ipopt_rectangle_obstacle",
        "ipopt",
        {"x": u, "p": p, "f": problem.cost, "g": g},
        _ipopt_options(),
    )

    solve_start = time.perf_counter()
    result = solver(
        x0=warm_start,
        p=np.concatenate([theta, variable]),
        lbx=np.asarray(problem.box_lower).reshape(-1),
        ubx=np.asarray(problem.box_upper).reshape(-1),
        lbg=np.zeros(problem.constraints.numel()),
        ubg=np.zeros(problem.constraints.numel()),
    )
    solve_time = time.perf_counter() - solve_start

    solution = np.asarray(result["x"]).reshape(-1)
    trajectory = _rollout(theta, solution, dt)
    product_values = _rectangle_product(trajectory[:, :2], rectangle)
    signed_violation = _rectangle_signed_violation(trajectory[:, :2], rectangle)
    final_distance = np.linalg.norm(trajectory[-1, :2] - target[:2])
    final_heading_error = np.arctan2(
        np.sin(trajectory[-1, 2] - target[2]),
        np.cos(trajectory[-1, 2] - target[2]),
    )
    stats = solver.stats()
    plot_path = Path(__file__).with_name("ipopt_rectangle_obstacle_debug.png")
    _write_rectangle_plot(trajectory, start, target, rectangle, safety_margin, plot_path)

    print("\nIPOPT bicycle rectangle obstacle feasibility")
    print(f"success: {stats.get('success')}")
    print(f"return_status: {stats.get('return_status')}")
    print(f"iterations: {stats.get('iter_count')}")
    print(f"solve time: {solve_time:.6f} s")
    print(f"objective: {float(result['f']):.6e}")
    print(f"max rectangle product: {np.max(product_values):.6e}")
    print(f"max rectangle signed violation: {np.max(signed_violation):.6e}")
    print(f"final state: {trajectory[-1]}")
    print(f"final distance: {final_distance:.6e}")
    print(f"final heading error: {final_heading_error:.6e}")
    print(f"plot path: {plot_path}")

    assert np.all(np.isfinite(solution))
