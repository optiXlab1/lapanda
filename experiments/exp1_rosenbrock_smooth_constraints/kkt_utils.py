"""KKT construction, reference-solve, and Explicit-KKT utilities for Exp. 1."""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for path in (REPO_ROOT, PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_local_lapanda() -> None:
    package_dir = PYTHON_DIR / "lapanda"
    extension_paths = sorted(package_dir.glob("_lapanda*.pyd"))
    if not extension_paths:
        extension_paths = sorted(package_dir.glob("_lapanda*.so"))
    if not extension_paths:
        raise ImportError(f"cannot find the local lapanda extension in {package_dir}")
    extension_spec = importlib.util.spec_from_file_location(
        "lapanda._lapanda", extension_paths[0]
    )
    if extension_spec is None or extension_spec.loader is None:
        raise ImportError(f"cannot load the lapanda extension from {extension_paths[0]}")
    extension = importlib.util.module_from_spec(extension_spec)
    sys.modules["lapanda._lapanda"] = extension
    extension_spec.loader.exec_module(extension)

    spec = importlib.util.spec_from_file_location(
        "lapanda",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lapanda from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lapanda"] = module
    spec.loader.exec_module(module)


load_local_lapanda()

from lapanda import solve_dense_minres


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=float).reshape(-1)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(lhs, rhs) / denominator)


def dm_to_csr(matrix: ca.DM) -> sp.csr_matrix:
    rows, cols = matrix.sparsity().get_triplet()
    values = np.asarray(matrix.nonzeros(), dtype=float).reshape(-1)
    return sp.coo_matrix((values, (rows, cols)), shape=matrix.shape).tocsr()


def build_reference_ipopt_solver(instance, tol: float, max_iter: int, name: str):
    nlp = {
        "x": instance.problem.u,
        "p": instance.problem.theta,
        "f": instance.problem.cost,
        "g": instance.problem.constraints,
    }
    return ca.nlpsol(
        name,
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": tol,
            "ipopt.constr_viol_tol": tol,
            "ipopt.dual_inf_tol": tol,
            "ipopt.compl_inf_tol": tol,
            "ipopt.max_iter": max_iter,
        },
    )


def build_kkt_derivative_function(instance, name: str) -> ca.Function:
    x = instance.problem.u
    theta = instance.problem.theta
    constraints = instance.problem.constraints
    multipliers = ca.SX.sym(f"{name}_lambda", instance.m)
    lagrangian = instance.problem.cost + ca.dot(multipliers, constraints)
    stationarity = ca.gradient(lagrangian, x)
    return ca.Function(
        name,
        [x, theta, multipliers],
        [
            ca.jacobian(stationarity, x),
            ca.jacobian(constraints, x),
            ca.jacobian(stationarity, theta),
            ca.jacobian(constraints, theta),
            stationarity,
        ],
    )


def same_forward_kkt_reference(
    instance,
    derivative_fun,
    x: np.ndarray,
    theta: np.ndarray,
    target: np.ndarray,
    multipliers: np.ndarray,
    penalties: np.ndarray,
    active_tolerance: float,
) -> dict:
    """Solve the zero-regularization KKT adjoint at the final ALM point."""

    constraints = np.asarray(instance.constraint_fun(x, theta), dtype=float).reshape(-1)
    lower_c = np.asarray(instance.constraint_lower, dtype=float).reshape(-1)
    upper_c = np.asarray(instance.constraint_upper, dtype=float).reshape(-1)
    shifted = constraints + multipliers / penalties
    projected = np.minimum(np.maximum(shifted, lower_c), upper_c)
    shifted_multipliers = penalties * (shifted - projected)
    equality = np.abs(upper_c - lower_c) <= active_tolerance
    active_constraints = (
        equality
        | (shifted <= lower_c + active_tolerance)
        | (shifted >= upper_c - active_tolerance)
    )

    hessian_dm, jacobian_dm, lut_dm, gt_dm, stationarity_dm = derivative_fun(
        x, theta, shifted_multipliers
    )
    hessian = dm_to_csr(hessian_dm)
    jacobian = dm_to_csr(jacobian_dm)
    lagrangian_u_theta = np.asarray(lut_dm, dtype=float)
    constraint_theta = np.asarray(gt_dm, dtype=float)
    stationarity = np.asarray(stationarity_dm, dtype=float).reshape(-1)

    active_rows = np.flatnonzero(active_constraints)
    blocks = [jacobian[active_rows, :]]
    theta_blocks = [constraint_theta[active_rows, :]]
    lower_active = np.flatnonzero(x <= instance.lower + active_tolerance)
    upper_active = np.flatnonzero(x >= instance.upper - active_tolerance)
    if lower_active.size:
        blocks.append(
            sp.csr_matrix(
                (-np.ones(lower_active.size), (np.arange(lower_active.size), lower_active)),
                shape=(lower_active.size, x.size),
            )
        )
        theta_blocks.append(np.zeros((lower_active.size, theta.size)))
    if upper_active.size:
        blocks.append(
            sp.csr_matrix(
                (np.ones(upper_active.size), (np.arange(upper_active.size), upper_active)),
                shape=(upper_active.size, x.size),
            )
        )
        theta_blocks.append(np.zeros((upper_active.size, theta.size)))

    active_jacobian = sp.vstack(blocks, format="csr")
    active_constraint_theta = np.vstack(theta_blocks)
    active_count = active_jacobian.shape[0]
    kkt = sp.bmat(
        [
            [hessian, active_jacobian.T],
            [active_jacobian, sp.csr_matrix((active_count, active_count))],
        ],
        format="csc",
    )
    rhs = np.concatenate([x - target, np.zeros(active_count)])
    operator = kkt.T.tocsc()
    factor = spla.splu(operator)
    adjoint = factor.solve(rhs)
    for _ in range(2):
        adjoint += factor.solve(rhs - operator @ adjoint)
    residual = float(
        np.linalg.norm(operator @ adjoint - rhs) / max(np.linalg.norm(rhs), 1e-16)
    )
    gradient = -(
        adjoint[: x.size] @ lagrangian_u_theta
        + adjoint[x.size :] @ active_constraint_theta
    )
    return {
        "gradient": np.asarray(gradient, dtype=float).reshape(-1),
        "linear_residual": residual,
        "active_nonlinear": int(active_rows.size),
        "active_lower_box": int(lower_active.size),
        "active_upper_box": int(upper_active.size),
        "shifted_multiplier_gap": float(np.linalg.norm(shifted_multipliers - multipliers)),
        "stationarity_without_box_inf": float(np.linalg.norm(stationarity, ord=np.inf)),
    }

def active_constraint_matrix_dense(
    jacobian: np.ndarray,
    constraints: np.ndarray,
    multipliers: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_tol: float,
    multiplier_tol: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    nonlinear_active = np.flatnonzero(
        (constraints >= -active_tol) | (multipliers > multiplier_tol)
    )
    blocks = [jacobian[nonlinear_active, :]]
    lower_active = np.flatnonzero(x <= lower + active_tol)
    upper_active = np.flatnonzero(x >= upper - active_tol)
    if lower_active.size:
        lower_rows = np.zeros((lower_active.size, x.size), dtype=float)
        lower_rows[np.arange(lower_active.size), lower_active] = -1.0
        blocks.append(lower_rows)
    if upper_active.size:
        upper_rows = np.zeros((upper_active.size, x.size), dtype=float)
        upper_rows[np.arange(upper_active.size), upper_active] = 1.0
        blocks.append(upper_rows)
    return (
        np.vstack(blocks),
        nonlinear_active,
        int(lower_active.size),
        int(upper_active.size),
    )


def explicit_kkt_minres_backward(
    derivative_fun: ca.Function,
    instance,
    x: np.ndarray,
    theta: np.ndarray,
    target: np.ndarray,
    multipliers: np.ndarray,
    active_tol: float,
    multiplier_tol: float,
    minres_tol: float,
    max_iterations: int,
) -> dict:
    total_start = time.perf_counter()
    evaluation_start = total_start
    hessian_dm, jacobian_dm, lut_dm, gt_dm, _ = derivative_fun(
        x, theta, multipliers
    )
    evaluation_time = time.perf_counter() - evaluation_start

    assembly_start = time.perf_counter()
    hessian = np.asarray(hessian_dm, dtype=float)
    jacobian = np.asarray(jacobian_dm, dtype=float)
    lagrangian_u_theta = np.asarray(lut_dm, dtype=float)
    constraint_theta = np.asarray(gt_dm, dtype=float)
    constraints = np.asarray(instance.constraint_fun(x, theta), dtype=float).reshape(-1)
    active_jacobian, active_rows, lower_count, upper_count = active_constraint_matrix_dense(
        jacobian,
        constraints,
        multipliers,
        x,
        instance.lower,
        instance.upper,
        active_tol,
        multiplier_tol,
    )
    active_count = active_jacobian.shape[0]
    kkt = np.block(
        [
            [hessian, active_jacobian.T],
            [active_jacobian, np.zeros((active_count, active_count), dtype=float)],
        ]
    )
    rhs = np.ascontiguousarray(
        np.concatenate([x - target, np.zeros(active_count)]), dtype=np.float64
    )
    operator = np.ascontiguousarray(kkt.T, dtype=np.float64)
    kkt_storage_bytes = int(operator.nbytes)
    assembly_time = time.perf_counter() - assembly_start

    solve_start = time.perf_counter()
    result = solve_dense_minres(
        operator,
        rhs,
        minres_tol,
        max_iterations,
    )
    solve_time = time.perf_counter() - solve_start
    adjoint = np.asarray(result["solution"], dtype=float).reshape(-1)
    active_constraint_theta = constraint_theta[active_rows, :]
    if lower_count or upper_count:
        active_constraint_theta = np.vstack(
            [
                active_constraint_theta,
                np.zeros((lower_count + upper_count, theta.size)),
            ]
        )
    gradient = -(
        adjoint[: x.size] @ lagrangian_u_theta
        + adjoint[x.size :] @ active_constraint_theta
    )
    residual = float(
        np.linalg.norm(operator @ adjoint - rhs) / max(np.linalg.norm(rhs), 1e-16)
    )
    total_time = time.perf_counter() - total_start
    storage_bytes = kkt_storage_bytes + int(result["peak_workspace_bytes"])
    return {
        "gradient": np.asarray(gradient, dtype=float).reshape(-1),
        "evaluation_time_sec": evaluation_time,
        "assembly_time_sec": assembly_time,
        "solve_time_sec": solve_time,
        "total_time_sec": total_time,
        "residual": residual,
        "iterations": int(result["iterations"]),
        "status": int(result["status"]),
        "minres_peak_workspace_bytes": int(result["peak_workspace_bytes"]),
        "kkt_storage_bytes": kkt_storage_bytes,
        "matrix_storage": "dense",
        "linear_storage_mb": storage_bytes / (1024.0**2),
        "active_nonlinear_constraints": int(active_rows.size),
        "active_lower_box_constraints": lower_count,
        "active_upper_box_constraints": upper_count,
    }
