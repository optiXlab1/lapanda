"""Shared helpers for OCP-style lapanda experiments."""

from __future__ import annotations

import tempfile
from pathlib import Path

import casadi as ca
import numpy as np

from experiments.common import append_csv, fresh_output_path, sampled_call, write_json

from experiments.exp2_OCPs.config import AlmDefaults, BackwardDefaults, MODEL_TO_PROBLEM
from experiments.exp2_OCPs.problem_bank import build_problem as build_ocp_problem
from experiments.exp2_OCPs.problem_bank import constraint_violation_inf
from lapanda import build_solver

OCP_ROOT = Path(__file__).resolve().parent

DEFAULT_ALM_CONFIG = AlmDefaults()
DEFAULT_BACKWARD_CONFIG = BackwardDefaults()
DEFAULT_ALM_INITIAL_PENALTY = DEFAULT_ALM_CONFIG.initial_penalty
DEFAULT_ALM_PENALTY_UPDATE_FACTOR = DEFAULT_ALM_CONFIG.penalty_update_factor


def make_variable_from_state(instance, state, demo_u, demo_states=None) -> np.ndarray:
    if demo_states is None:
        demo_x = np.zeros(instance.horizon * instance.nx)
    else:
        demo_x = np.asarray(demo_states, dtype=float)
        if demo_x.ndim == 2:
            demo_x = demo_x[1 : instance.horizon + 1].reshape(-1)
        else:
            demo_x = demo_x.reshape(-1)
        if demo_x.size != instance.horizon * instance.nx:
            raise ValueError(
                f"demo_states expected {instance.horizon * instance.nx} values, got {demo_x.size}"
            )
    return np.concatenate(
        [
            np.asarray(state, dtype=float).reshape(instance.nx),
            np.asarray(instance.x_target, dtype=float).reshape(instance.nx),
            np.asarray(demo_u, dtype=float).reshape(instance.horizon * instance.nu),
            demo_x,
        ]
    )


def shift_control(u, nu):
    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size <= nu:
        return np.zeros_like(u)
    return np.concatenate([u[nu:], u[-nu:]])


def first_next_state(instance, u, theta, variable):
    states = np.asarray(instance.rollout_fun(u, theta, variable), dtype=float)
    return states[1].copy()


def inner_options(SolverOptions, max_iter: int, tol: float):
    options = SolverOptions()
    options.max_iterations = max_iter
    options.tolerance = tol
    options.buffer_size = 10
    options.max_stable_iter = 120
    return options


def alm_options(
    AlmOptions,
    max_outer: int,
    tol: float,
    max_penalty: float | None = None,
    *,
    config: AlmDefaults = DEFAULT_ALM_CONFIG,
    initial_penalty: float | None = None,
    penalty_update_factor: float | None = None,
):
    options = AlmOptions()
    options.max_iterations = max_outer
    options.tolerance = tol
    options.initial_penalty = config.initial_penalty if initial_penalty is None else initial_penalty
    options.penalty_update_factor = (
        config.penalty_update_factor if penalty_update_factor is None else penalty_update_factor
    )
    options.max_penalty = config.max_penalty if max_penalty is None else max_penalty
    options.sufficient_decrease_factor = config.sufficient_decrease_factor
    options.warm_start_inner = config.warm_start_inner
    return options


def backward_options(
    BackwardOptions,
    enable: bool,
    tol: float,
    *,
    config: BackwardDefaults = DEFAULT_BACKWARD_CONFIG,
    max_iterations: int | None = None,
    constraint_penalty_scale: float | None = None,
    constraint_penalty_max: float | None = None,
):
    options = BackwardOptions()
    options.enable = enable
    options.tolerance = tol
    options.max_iterations = config.max_iterations if max_iterations is None else max_iterations
    options.constraint_penalty_scale = (
        config.constraint_penalty_scale if constraint_penalty_scale is None else constraint_penalty_scale
    )
    options.constraint_penalty_max = (
        config.constraint_penalty_max if constraint_penalty_max is None else constraint_penalty_max
    )
    return options


def build_alm_solver(build_solver, instance, name: str, backend: str, force: bool, cache_tag: str):
    return build_solver(
        instance.problem,
        backend=backend,
        name=name,
        cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / cache_tag,
        force=force,
    )


def clip_theta(instance, theta):
    theta = np.asarray(theta, dtype=float).copy()
    lower = np.full(theta.size, 1e-5)
    upper = np.full(theta.size, 100.0)
    if instance.name == "planar_quadrotor":
        upper[-1] = 0.35
    elif instance.name == "two_link_arm":
        upper[-2] = 0.4
        upper[-1] = 0.8
    return np.minimum(np.maximum(theta, lower), upper)


def seeded_theta_initial(instance, mode: str, seed: int | None):
    """Return a reproducible learner initialization.

    ``default`` preserves the experiment's hand-picked initial theta.  ``uniform01``
    follows the imitation-learning convention used in the DiffMPC/Trajax style
    benchmarks: each learnable cost parameter is initialized from U([0, 1]).
    The value is clipped by the model-specific theta bounds before training.
    """

    if mode == "default":
        return instance.theta_initial.copy()
    if mode != "uniform01":
        raise ValueError(f"unknown theta initialization mode: {mode}")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 1.0, size=instance.theta_initial.size)
    if instance.name == "two_link_arm":
        # The final arm parameter is a geometric obstacle margin.  Sampling it
        # from U([0, 1]) can make the obstacle nearly cover the reachable
        # workspace, which creates a degenerate imitation problem before the
        # optimizer has learned anything useful.
        theta[-1] = rng.uniform(0.0, 0.15)
    return clip_theta(instance, theta)


def parse_theta_override(raw: str, expected_size: int, name: str) -> np.ndarray | None:
    if not raw:
        return None
    values = np.asarray([float(item.strip()) for item in raw.split(",") if item.strip()], dtype=float)
    if values.size != expected_size:
        raise ValueError(f"{name} expected {expected_size} comma-separated values, got {values.size}")
    return values


def apply_theta_overrides(instance, theta_true_raw: str = "", theta_initial_raw: str = ""):
    theta_true = parse_theta_override(theta_true_raw, instance.theta_true.size, "--theta-true-override")
    theta_initial = parse_theta_override(theta_initial_raw, instance.theta_initial.size, "--theta-initial-override")
    if theta_true is not None:
        instance.theta_true = clip_theta(instance, theta_true)
    if theta_initial is not None:
        instance.theta_initial = clip_theta(instance, theta_initial)
    return instance


def write_run_config(out_dir: Path, name: str, args, instance, extra=None):
    alm_initial_penalty = getattr(args, "alm_initial_penalty", DEFAULT_ALM_INITIAL_PENALTY)
    alm_penalty_update_factor = getattr(args, "alm_penalty_update_factor", DEFAULT_ALM_PENALTY_UPDATE_FACTOR)
    alm_max_penalty = getattr(args, "alm_max_penalty", DEFAULT_ALM_CONFIG.max_penalty)
    backward_tolerance = getattr(args, "alm_tol", 1e-3)
    backward_max_iterations = getattr(args, "backward_max_iterations", DEFAULT_BACKWARD_CONFIG.max_iterations)
    backward_constraint_penalty_scale = getattr(
        args,
        "backward_constraint_penalty_scale",
        DEFAULT_BACKWARD_CONFIG.constraint_penalty_scale,
    )
    backward_constraint_penalty_max = getattr(
        args,
        "backward_constraint_penalty_max",
        DEFAULT_BACKWARD_CONFIG.constraint_penalty_max,
    )
    data = {
        "model": instance.name,
        "horizon": instance.horizon,
        "dt": instance.dt,
        "theta_true": instance.theta_true.tolist(),
        "theta_initial": instance.theta_initial.tolist(),
        "alm_initial_penalty": alm_initial_penalty,
        "alm_penalty_update_factor": alm_penalty_update_factor,
        "alm_max_penalty": alm_max_penalty,
        "backward_tolerance": backward_tolerance,
        "backward_max_iterations": backward_max_iterations,
        "backward_constraint_penalty_scale": backward_constraint_penalty_scale,
        "backward_constraint_penalty_max": backward_constraint_penalty_max,
    }
    data.update(vars(args))
    if extra:
        data.update(extra)
    write_json(out_dir / f"config_{name}.json", data)


def result_csv(script_file, filename: str):
    return fresh_output_path(Path(script_file).resolve().parent / "results" / filename)


def tagged_result_csv(script_file, filename: str, output_tag: str | None):
    if not output_tag:
        return result_csv(script_file, filename)
    path = Path(script_file).resolve().parent / "results" / filename
    tagged = path.with_name(f"{path.stem}_{output_tag}{path.suffix}")
    return fresh_output_path(tagged)


def load_snapshot_archive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    return [dict(item) for item in data["snapshots"].tolist()]


def default_teacher_archive_path(model: str, horizon: int, sample_count: int = 32, teacher_seed: int = 0) -> Path:
    """Return the fixed expert-dataset path used by multi-seed imitation runs."""

    if model == "cartpole":
        exp_dir = OCP_ROOT / "cartpole" / "results"
        prefix = "exp2_cartpole"
    elif model == "quadrotor":
        exp_dir = OCP_ROOT / "quadrotor" / "results"
        prefix = "exp2_quadrotor"
    elif model == "robot_arm":
        exp_dir = OCP_ROOT / "robot_arm" / "results"
        prefix = "exp2_robot_arm"
    else:
        raise ValueError(f"unknown model {model!r}")
    return exp_dir / f"{prefix}_expert{sample_count}_h{horizon}_seed{teacher_seed}.mpc_snapshots.npz"


def append_row(path: Path, row: dict):
    append_csv(path, [row])


def solve_alm_once(
    solver,
    instance,
    u0,
    theta,
    variable,
    args,
    SolverOptions,
    AlmOptions,
    BackwardOptions,
    compute_backward,
    multiplier0=None,
    penalty0=None,
    adjoint0=None,
):
    return solver.solve_lapanda(
        u0,
        theta,
        variable,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(SolverOptions, args.inner_max_iter, args.inner_tol),
        alm_options=alm_options(
            AlmOptions,
            args.max_outer,
            args.alm_tol,
            getattr(args, "alm_max_penalty", DEFAULT_ALM_CONFIG.max_penalty),
            initial_penalty=getattr(args, "alm_initial_penalty", DEFAULT_ALM_INITIAL_PENALTY),
            penalty_update_factor=getattr(args, "alm_penalty_update_factor", DEFAULT_ALM_PENALTY_UPDATE_FACTOR),
        ),
        backward_options=backward_options(
            BackwardOptions,
            compute_backward,
            args.alm_tol,
            max_iterations=getattr(args, "backward_max_iterations", DEFAULT_BACKWARD_CONFIG.max_iterations),
            constraint_penalty_scale=getattr(
                args,
                "backward_constraint_penalty_scale",
                DEFAULT_BACKWARD_CONFIG.constraint_penalty_scale,
            ),
            constraint_penalty_max=getattr(
                args,
                "backward_constraint_penalty_max",
                DEFAULT_BACKWARD_CONFIG.constraint_penalty_max,
            ),
        ),
        multiplier0=multiplier0,
        penalty0=penalty0,
        adjoint0=adjoint0,
    )


def build_ipopt_solver(instance, ipopt_tol: float, ipopt_max_iter: int):
    nlp = {
        "x": instance.problem.u,
        "p": ca.vertcat(instance.problem.theta, instance.problem.variable),
        "f": instance.problem.cost,
        "g": instance.problem.constraints,
    }
    return ca.nlpsol(
        f"{instance.name}_ipopt",
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": ipopt_tol,
            "ipopt.acceptable_tol": ipopt_tol,
            "ipopt.constr_viol_tol": ipopt_tol,
            "ipopt.max_iter": ipopt_max_iter,
        },
    )


def solve_ipopt(instance, solver, u0, theta, variable):
    return solver(
        x0=u0,
        p=np.concatenate([theta, variable]),
        lbx=instance.problem.box_lower,
        ubx=instance.problem.box_upper,
        lbg=instance.constraint_lower,
        ubg=instance.constraint_upper,
    )


def build_derivatives(instance):
    u = ca.vec(instance.problem.u)
    theta = ca.vec(instance.problem.theta)
    variable = ca.vec(instance.problem.variable)
    c = ca.vec(instance.problem.constraints)
    f = instance.problem.cost
    loss = instance.problem.outer_loss
    n = int(u.numel())
    m = int(c.numel())
    lam_g = ca.SX.sym("lam_g", m)
    lam_x_l = ca.SX.sym("lam_x_l", n)
    lam_x_u = ca.SX.sym("lam_x_u", n)
    lower = np.asarray(instance.problem.box_lower, dtype=float)
    upper = np.asarray(instance.problem.box_upper, dtype=float)
    lag = f + ca.dot(lam_g, c) + ca.dot(lam_x_u, u - upper) + ca.dot(lam_x_l, lower - u)
    grad_lag_u = ca.gradient(lag, u)
    return {
        "hess": ca.Function("ocp_lag_uu", [u, theta, variable, lam_g, lam_x_l, lam_x_u], [ca.jacobian(grad_lag_u, u)]),
        "hutheta": ca.Function("ocp_lag_utheta", [u, theta, variable, lam_g, lam_x_l, lam_x_u], [ca.jacobian(grad_lag_u, theta)]),
        "jac_c_u": ca.Function("ocp_jac_c_u", [u, theta, variable], [ca.jacobian(c, u)]),
        "jac_c_theta": ca.Function("ocp_jac_c_theta", [u, theta, variable], [ca.jacobian(c, theta)]),
        "loss_grad": ca.Function("ocp_loss_grad", [u, theta, variable], [loss, ca.gradient(loss, u), ca.gradient(loss, theta)]),
    }


def kkt_sensitivity(instance, derivs, result, theta, variable, active_tol):
    u_star = np.asarray(result["x"], dtype=float).reshape(-1)
    lam_g = np.asarray(result["lam_g"], dtype=float).reshape(-1)
    lam_x = np.asarray(result["lam_x"], dtype=float).reshape(-1)
    lower = np.asarray(instance.problem.box_lower, dtype=float)
    upper = np.asarray(instance.problem.box_upper, dtype=float)
    c_val = np.asarray(instance.constraint_fun(u_star, theta, variable), dtype=float).reshape(-1)
    con_active = c_val >= -active_tol
    lower_active = u_star <= lower + active_tol
    upper_active = u_star >= upper - active_tol
    jac_c_u = np.asarray(derivs["jac_c_u"](u_star, theta, variable), dtype=float)
    jac_c_theta = np.asarray(derivs["jac_c_theta"](u_star, theta, variable), dtype=float)
    n = u_star.size
    ntheta = theta.size
    active_rows = []
    active_theta_rows = []
    for i, active in enumerate(con_active):
        if active:
            active_rows.append(jac_c_u[i, :])
            active_theta_rows.append(jac_c_theta[i, :])
    for i, active in enumerate(lower_active):
        if active:
            row = np.zeros(n)
            row[i] = -1.0
            active_rows.append(row)
            active_theta_rows.append(np.zeros(ntheta))
    for i, active in enumerate(upper_active):
        if active:
            row = np.zeros(n)
            row[i] = 1.0
            active_rows.append(row)
            active_theta_rows.append(np.zeros(ntheta))
    active_c = np.vstack(active_rows) if active_rows else np.zeros((0, n))
    active_ctheta = np.vstack(active_theta_rows) if active_theta_rows else np.zeros((0, ntheta))
    lam_x_l = np.maximum(-lam_x, 0.0)
    lam_x_u = np.maximum(lam_x, 0.0)
    hess = np.asarray(derivs["hess"](u_star, theta, variable, lam_g, lam_x_l, lam_x_u), dtype=float)
    hutheta = np.asarray(derivs["hutheta"](u_star, theta, variable, lam_g, lam_x_l, lam_x_u), dtype=float)
    loss_val, loss_u_raw, loss_theta_raw = derivs["loss_grad"](u_star, theta, variable)
    loss_u = np.asarray(loss_u_raw, dtype=float).reshape(-1)
    loss_theta = np.asarray(loss_theta_raw, dtype=float).reshape(-1)
    m_active = active_c.shape[0]
    kkt = np.block([[hess, active_c.T], [active_c, np.zeros((m_active, m_active))]])
    adj = np.linalg.solve(kkt.T, np.concatenate([loss_u, np.zeros(m_active)]))
    grad = loss_theta - hutheta.T @ adj[:n] - active_ctheta.T @ adj[n:]
    return {
        "solution": u_star,
        "loss": float(loss_val),
        "grad_theta": grad,
        "active_constraints": int(np.count_nonzero(con_active)),
        "active_box": int(np.count_nonzero(lower_active) + np.count_nonzero(upper_active)),
        "kkt_cond": float(np.linalg.cond(kkt)),
    }


def timed_sample(fn):
    return sampled_call(fn)


def build_instance(model: str, horizon: int):
    return build_ocp_problem(MODEL_TO_PROBLEM[model], horizon=horizon)


def build_lapanda_solver(instance, model: str, name: str, backend: str, force: bool):
    return build_solver(
        instance.problem,
        backend=backend,
        name=name,
        cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp2_OCPs" / model,
        force=force,
    )


def solve_lapanda(
    solver,
    instance,
    u0,
    theta,
    variable,
    *,
    inner_max_iter: int,
    inner_tol: float,
    max_outer: int,
    alm_tol: float,
    compute_backward: bool,
    alm_max_penalty: float = 0.0,
    alm_initial_penalty: float | None = None,
    alm_penalty_update_factor: float | None = None,
    backward_max_iterations: int | None = None,
    backward_constraint_penalty_scale: float | None = None,
    backward_constraint_penalty_max: float | None = None,
    multiplier0=None,
    penalty0=None,
):
    from lapanda import AlmOptions, BackwardOptions, SolverOptions

    return solver.solve_lapanda(
        u0,
        theta,
        variable,
        instance.constraint_lower,
        instance.constraint_upper,
        inner_solver_options=inner_options(SolverOptions, inner_max_iter, inner_tol),
        alm_options=alm_options(
            AlmOptions,
            max_outer,
            alm_tol,
            alm_max_penalty,
            initial_penalty=alm_initial_penalty,
            penalty_update_factor=alm_penalty_update_factor,
        ),
        backward_options=backward_options(
            BackwardOptions,
            compute_backward,
            alm_tol,
            max_iterations=backward_max_iterations,
            constraint_penalty_scale=backward_constraint_penalty_scale,
            constraint_penalty_max=backward_constraint_penalty_max,
        ),
        multiplier0=multiplier0,
        penalty0=penalty0,
    )


def violation_inf(instance, solution, theta, variable) -> float:
    return constraint_violation_inf(instance, solution, theta, variable)


def as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)
