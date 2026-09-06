"""acados baseline for the hard-constrained circular obstacle problem."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import psutil

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import config as exp3_config
from run_lapanda import STEER_LIMIT, WHEELBASE, build_circle_case, metrics, rollout


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def build_ocp(
    case,
    outdir: Path,
    name: str,
    sensitivity: bool,
    forward_mode: str,
    max_iter: int,
    regularize: str,
    qp_solver: str,
    h_lower: float,
    qp_iter_max: int,
    globalization: str,
    slack_weight: float,
    nlp_tol_stat: float,
    nlp_tol_eq: float,
    nlp_tol_ineq: float,
    nlp_tol_comp: float,
):
    from acados_template import AcadosModel, AcadosOcp

    nx, nu = 3, 2
    nparam = case.theta.size + case.variable.size
    model = AcadosModel()
    model.name = name
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p_global", nparam)
    px, py, heading = x[0], x[1], x[2]
    speed, steer = u[0], u[1]
    q_pos, q_heading, r_speed, r_steer, terminal_weight, safe_radius, obstacle_y = [p[i] for i in range(7)]
    target_x, target_y, target_heading = p[10], p[11], p[12]
    px_next = px + case.dt * speed * ca.cos(heading)
    py_next = py + case.dt * speed * ca.sin(heading)
    heading_next = heading + case.dt * speed * ca.tan(steer) / WHEELBASE

    model.x = x
    model.u = u
    model.p_global = p
    model.disc_dyn_expr = ca.vertcat(px_next, py_next, heading_next)

    dx = px - target_x
    dy = py - target_y
    dheading = ca.atan2(ca.sin(heading - target_heading), ca.cos(heading - target_heading))
    use_external_cost = sensitivity or forward_mode == "exact"
    if use_external_cost:
        model.cost_expr_ext_cost = q_pos * (dx**2 + dy**2) + q_heading * dheading**2 + r_speed * speed**2 + r_steer * steer**2
        model.cost_expr_ext_cost_e = terminal_weight * (q_pos * (dx**2 + dy**2) + q_heading * dheading**2)
    else:
        model.cost_y_expr = ca.vertcat(
            ca.sqrt(q_pos) * dx,
            ca.sqrt(q_pos) * dy,
            ca.sqrt(q_heading) * dheading,
            ca.sqrt(r_speed) * speed,
            ca.sqrt(r_steer) * steer,
        )
        model.cost_y_expr_e = ca.vertcat(
            ca.sqrt(terminal_weight * q_pos) * dx,
            ca.sqrt(terminal_weight * q_pos) * dy,
            ca.sqrt(terminal_weight * q_heading) * dheading,
        )
    model.con_h_expr = ca.vertcat(safe_radius**2 - (px_next**2 + (py_next - obstacle_y) ** 2))

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = case.horizon
    ocp.solver_options.tf = case.horizon * case.dt
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.solver_options.nlp_solver_tol_stat = nlp_tol_stat
    ocp.solver_options.nlp_solver_tol_eq = nlp_tol_eq
    ocp.solver_options.nlp_solver_tol_ineq = nlp_tol_ineq
    ocp.solver_options.nlp_solver_tol_comp = nlp_tol_comp
    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.qp_solver_cond_N = case.horizon
    ocp.solver_options.qp_solver_iter_max = qp_iter_max
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_ric_alg = 0
    ocp.solver_options.hessian_approx = "EXACT" if use_external_cost else "GAUSS_NEWTON"
    ocp.solver_options.globalization = globalization
    ocp.solver_options.regularize_method = "NO_REGULARIZE" if sensitivity else regularize
    ocp.solver_options.print_level = 0
    if sensitivity:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True
    if use_external_cost:
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
    else:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = 2.0 * np.eye(5)
        ocp.cost.W_e = 2.0 * np.eye(3)
        ocp.cost.yref = np.zeros(5)
        ocp.cost.yref_e = np.zeros(3)
    ocp.constraints.x0 = case.variable[:3]
    ocp.constraints.lbu = np.array([float(case.problem.box_lower[0]), -STEER_LIMIT])
    ocp.constraints.ubu = np.array([float(case.problem.box_upper[0]), STEER_LIMIT])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lh = np.array([h_lower])
    ocp.constraints.uh = np.array([0.0])
    if slack_weight > 0.0:
        ocp.constraints.idxsh = np.array([0])
        ocp.cost.Zl = np.array([slack_weight])
        ocp.cost.Zu = np.array([slack_weight])
        ocp.cost.zl = np.array([slack_weight])
        ocp.cost.zu = np.array([slack_weight])
    ocp.p_global_values = np.r_[case.theta, case.variable]
    ocp.code_export_directory = str(outdir / f"acados_codegen_{name}")
    return ocp


def build_two_solvers(
    case,
    outdir: Path,
    forward_mode: str,
    max_iter: int,
    regularize: str,
    qp_solver: str,
    h_lower: float,
    qp_iter_max: int,
    globalization: str,
    slack_weight: float,
    nlp_tol_stat: float,
    nlp_tol_eq: float,
    nlp_tol_ineq: float,
    nlp_tol_comp: float,
):
    from acados_template import AcadosOcpSolver

    forward_ocp = build_ocp(case, outdir, "circle_forward", False, forward_mode, max_iter, regularize, qp_solver, h_lower, qp_iter_max, globalization, slack_weight, nlp_tol_stat, nlp_tol_eq, nlp_tol_ineq, nlp_tol_comp)
    forward_solver = AcadosOcpSolver(forward_ocp, json_file=str(outdir / "acados_forward_ocp.json"), verbose=False)
    sens_ocp = build_ocp(case, outdir, "circle_sensitivity", True, forward_mode, max_iter, regularize, qp_solver, h_lower, qp_iter_max, globalization, slack_weight, nlp_tol_stat, nlp_tol_eq, nlp_tol_ineq, nlp_tol_comp)
    sensitivity_solver = AcadosOcpSolver(sens_ocp, json_file=str(outdir / "acados_sensitivity_ocp.json"), verbose=False)
    return forward_solver, sensitivity_solver


def get_iterate_obj(solver):
    if hasattr(solver, "get_flat_iterate"):
        return solver.get_flat_iterate()
    return solver.store_iterate_to_flat_obj()


def main():
    forward_mode = os.environ.get("ACADOS_CIRCLE_FORWARD_MODE", "exact")
    max_iter = int(os.environ.get("ACADOS_CIRCLE_MAX_ITER", "1000"))
    qp_iter_max = int(os.environ.get("ACADOS_CIRCLE_QP_ITER_MAX", "200"))
    regularize = os.environ.get("ACADOS_CIRCLE_REGULARIZE", "MIRROR")
    qp_solver = os.environ.get("ACADOS_CIRCLE_QP_SOLVER", "PARTIAL_CONDENSING_HPIPM")
    globalization = os.environ.get("ACADOS_CIRCLE_GLOBALIZATION", "MERIT_BACKTRACKING")
    slack_weight = float(os.environ.get("ACADOS_CIRCLE_SLACK_WEIGHT", "0"))
    nlp_tol = float(
        os.environ.get("ACADOS_CIRCLE_TOL", str(exp3_config.CIRCLE_ACADOS_TOL))
    )
    nlp_tol_stat = float(os.environ.get("ACADOS_CIRCLE_TOL_STAT", str(nlp_tol)))
    nlp_tol_eq = float(os.environ.get("ACADOS_CIRCLE_TOL_EQ", str(nlp_tol)))
    nlp_tol_ineq = float(os.environ.get("ACADOS_CIRCLE_TOL_INEQ", str(nlp_tol)))
    nlp_tol_comp = float(os.environ.get("ACADOS_CIRCLE_TOL_COMP", str(nlp_tol)))
    h_lower = float(os.environ.get("ACADOS_CIRCLE_H_LOWER", "-1e3"))
    init_solution_path = os.environ.get("ACADOS_CIRCLE_INIT_SOLUTION", "")
    terminal_weight = os.environ.get("ACADOS_CIRCLE_TERMINAL_WEIGHT", "")
    circle_radius = os.environ.get("ACADOS_CIRCLE_RADIUS", str(exp3_config.CIRCLE_RADIUS))
    circle_center_y = os.environ.get("ACADOS_CIRCLE_CENTER_Y", str(exp3_config.CIRCLE_CENTER_Y))
    horizon = int(os.environ.get("ACADOS_CIRCLE_HORIZON", str(exp3_config.CIRCLE_HORIZON)))
    speed_limit = float(os.environ.get("ACADOS_CIRCLE_SPEED_LIMIT", str(exp3_config.CIRCLE_SPEED_LIMIT)))
    outdir_env = os.environ.get("ACADOS_CIRCLE_OUTDIR", "")
    outdir = Path(outdir_env) if outdir_env else exp3_config.CIRCLE_ACADOS_RESULT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    for code_dir in [outdir / "acados_codegen_circle_forward", outdir / "acados_codegen_circle_sensitivity"]:
        if code_dir.exists():
            shutil.rmtree(code_dir)

    case = build_circle_case(horizon, speed_limit)
    if terminal_weight:
        case.theta[4] = float(terminal_weight)
    if circle_radius:
        case.theta[5] = float(circle_radius)
        case.meta["safe_radius"] = float(circle_radius)
    if circle_center_y:
        case.theta[6] = float(circle_center_y)
        case.meta["obstacle"] = [0.0, float(circle_center_y)]
    if init_solution_path:
        case.x0 = np.loadtxt(init_solution_path, delimiter=",").reshape(-1)
    rss_start = rss_mb()
    t0 = time.perf_counter()
    forward_solver, sensitivity_solver = build_two_solvers(case, outdir, forward_mode, max_iter, regularize, qp_solver, h_lower, qp_iter_max, globalization, slack_weight, nlp_tol_stat, nlp_tol_eq, nlp_tol_ineq, nlp_tol_comp)
    build_time = time.perf_counter() - t0
    rss_after_build = rss_mb()

    p_global = np.r_[case.theta, case.variable]
    states = np.vstack([case.variable[:3], rollout(case.x0, case.variable, case.dt)])
    for solver in [forward_solver, sensitivity_solver]:
        for k in range(case.horizon + 1):
            solver.set(k, "x", states[k])
        for k in range(case.horizon):
            solver.set(k, "u", case.x0[2 * k : 2 * k + 2])
        solver.set_p_global_and_precompute_dependencies(p_global)

    t0 = time.perf_counter()
    status = forward_solver.solve()
    forward_time = time.perf_counter() - t0
    rss_after_forward = rss_mb()

    solution = np.zeros(2 * case.horizon)
    for k in range(case.horizon):
        solution[2 * k : 2 * k + 2] = forward_solver.get(k, "u")

    try:
        stat = np.asarray(forward_solver.get_stats("statistics"), dtype=float)
        last_col = stat[:, -1] if stat.ndim == 2 and stat.shape[1] > 0 else np.full(8, np.nan)
        sqp_iterations = int(last_col[0]) if np.isfinite(last_col[0]) else None
        last_qp_status = int(last_col[5]) if last_col.size > 5 and np.isfinite(last_col[5]) else None
        qp_iter_sum = int(np.nansum(stat[6, :])) if stat.ndim == 2 and stat.shape[0] > 6 else None
    except Exception:
        sqp_iterations = None
        last_qp_status = None
        qp_iter_sum = None

    seeds_u = [(k, solution[2 * k : 2 * k + 2].reshape(2, 1)) for k in range(case.horizon)]
    t0 = time.perf_counter()
    try:
        sensitivity_solver.load_iterate_from_flat_obj(get_iterate_obj(forward_solver))
        sensitivity_solver.setup_qp_matrices_and_factorize()
        sensitivity = sensitivity_solver.eval_adjoint_solution_sensitivity(
            seed_x=None,
            seed_u=seeds_u,
            with_respect_to="p_global",
            sanity_checks=True,
        )
        backward_status = "ok"
        grad = np.asarray(sensitivity, dtype=float).reshape(-1)
    except Exception as exc:
        backward_status = repr(exc)
        grad = np.full(p_global.shape, np.nan)
    backward_time = time.perf_counter() - t0
    rss_after_backward = rss_mb()

    row = {
        "method": "acados",
        "problem": "circle",
        "forward_mode": forward_mode,
        "regularize": regularize,
        "qp_solver": qp_solver,
        "globalization": globalization,
        "slack_weight": slack_weight,
        "nlp_tol": max(nlp_tol_stat, nlp_tol_eq, nlp_tol_ineq, nlp_tol_comp),
        "nlp_tol_stat": nlp_tol_stat,
        "nlp_tol_eq": nlp_tol_eq,
        "nlp_tol_ineq": nlp_tol_ineq,
        "nlp_tol_comp": nlp_tol_comp,
        "status": int(status),
        "max_iter": max_iter,
        "qp_iter_max": qp_iter_max,
        "h_lower": h_lower,
        "init_solution": init_solution_path,
        "terminal_weight": float(case.theta[4]),
        "circle_radius": float(case.theta[5]),
        "circle_center_y": float(case.theta[6]),
        "horizon": int(case.horizon),
        "speed_limit": float(speed_limit),
        "sqp_iterations": sqp_iterations,
        "last_qp_status": last_qp_status,
        "qp_iter_sum": qp_iter_sum,
        "backward_status": backward_status,
        "build_time_sec": build_time,
        "forward_time_sec": forward_time,
        "backward_time_sec": backward_time,
        "rss_start_mb": rss_start,
        "rss_after_build_mb": rss_after_build,
        "rss_after_forward_mb": rss_after_forward,
        "rss_after_backward_mb": rss_after_backward,
        "build_rss_delta_mb": rss_after_build - rss_start,
        "forward_rss_delta_mb": rss_after_forward - rss_after_build,
        "backward_rss_delta_mb": rss_after_backward - rss_after_forward,
        "total_rss_delta_mb": rss_after_backward - rss_start,
        "grad_norm": float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else float("nan"),
        **metrics(case, solution),
    }
    np.savetxt(outdir / "acados_circle_solution.csv", solution, delimiter=",")
    with (outdir / "acados_circle_result.json").open("w", encoding="utf-8") as f:
        json.dump({**row, "solution": solution.tolist(), "grad_p_global": grad.tolist()}, f, indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
