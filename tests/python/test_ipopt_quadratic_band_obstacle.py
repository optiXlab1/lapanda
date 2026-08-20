import importlib.util
import time
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_quadratic_helpers():
    path = Path(__file__).with_name("test_alm_quadratic_band_obstacle_avoidance.py")
    spec = importlib.util.spec_from_file_location("quadratic_band_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_ipopt_plot(trajectory, start, target, safety_margin, output_path):
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
        label="IPOPT trajectory",
    )
    ax.scatter([start[0]], [start[1]], marker="s", s=70, color="#2ca02c", label="start")
    ax.scatter([target[0]], [target[1]], marker="*", s=120, color="#ff7f0e", label="target")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.8, 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("IPOPT bicycle quadratic-band obstacle")
    ax.legend(loc="lower center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def test_ipopt_quadratic_band_obstacle_nonconvex_behavior():
    helpers = _load_quadratic_helpers()

    safety_margin = 0.15
    problem, n, dt = helpers._build_quadratic_band_problem(safety_margin=safety_margin)
    start = np.array([-0.3, 2.0, np.pi*0.8])
    target = np.array([-1.5, 1.0, -np.pi*0.2])
    theta = np.concatenate([start, target])
    variable = np.zeros(1)
    initial_guess = np.zeros(n)

    u = ca.vec(problem.u)
    p = ca.vertcat(ca.vec(problem.theta), ca.vec(problem.variable))
    g = ca.vec(problem.constraints)
    nlp = {"x": u, "p": p, "f": problem.cost, "g": g}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.tol": 1e-4,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.max_iter": 2000,
    }

    build_start = time.perf_counter()
    solver = ca.nlpsol("quadratic_band_ipopt", "ipopt", nlp, opts)
    build_time = time.perf_counter() - build_start

    solve_start = time.perf_counter()
    sol = solver(
        x0=initial_guess,
        p=np.concatenate([theta, variable]),
        lbx=np.asarray(problem.box_lower, dtype=float),
        ubx=np.asarray(problem.box_upper, dtype=float),
        lbg=np.zeros(g.numel()),
        ubg=np.zeros(g.numel()),
    )
    solve_time = time.perf_counter() - solve_start

    solution = np.asarray(sol["x"], dtype=float).reshape(-1)
    trajectory = helpers._rollout(theta, solution, dt)
    product_values = helpers._quadratic_band_product(trajectory[:, :2], safety_margin)
    signed_violation = helpers._quadratic_band_signed_violation(
        trajectory[:, :2], safety_margin
    )
    final_distance = np.linalg.norm(trajectory[-1, :2] - target[:2])
    max_product_index = int(np.argmax(product_values))
    max_signed_index = int(np.argmax(signed_violation))
    stats = solver.stats()

    plot_path = Path(__file__).with_name("ipopt_quadratic_band_obstacle_debug.png")
    _write_ipopt_plot(trajectory, start, target, safety_margin, plot_path)

    print("\nIPOPT bicycle quadratic-band obstacle")
    print(f"build time: {build_time:.6f} s")
    print(f"solve time: {solve_time:.6f} s")
    print(f"status: {stats.get('return_status')}")
    print(f"iterations: {stats.get('iter_count')}")
    print(f"objective: {float(sol['f']):.6e}")
    print(f"max quadratic-band product: {np.max(product_values):.6e}")
    print(f"max quadratic-band signed violation: {np.max(signed_violation):.6e}")
    print(
        "max product point: "
        f"k={max_product_index}, state={trajectory[max_product_index]}, "
        f"control={solution.reshape(-1, 2)[max_product_index]}"
    )
    print(
        "max signed-violation point: "
        f"k={max_signed_index}, state={trajectory[max_signed_index]}, "
        f"control={solution.reshape(-1, 2)[max_signed_index]}"
    )
    print(f"final state: {trajectory[-1]}")
    print(f"final distance: {final_distance:.6e}")
    print(f"plot path: {plot_path}")

    assert solution.shape == (n,)
