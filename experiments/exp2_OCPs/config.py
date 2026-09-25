"""Configuration shared by the unified OCP experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent

MODEL_TO_PROBLEM = {
    "cartpole": "cartpole",
    "quadrotor": "quadrotor",
    "robot_arm": "arm",
}

MODEL_PREFIX = {
    "cartpole": "exp2_cartpole",
    "quadrotor": "exp2_quadrotor",
    "robot_arm": "exp2_robot_arm",
}

MODEL_DISPLAY = {
    "cartpole": "Cartpole",
    "quadrotor": "Quadrotor",
    "robot_arm": "Robot arm",
}

OPEN_LOOP_LEARNING_RATE = {
    "cartpole": 2e-3,
    "quadrotor": 2e-3,
    "robot_arm": 2e-3,
}

CLOSED_LOOP_LEARNING_RATE = {
    "cartpole": 1e-3,
    "quadrotor": 1e-3,
    "robot_arm": 1e-3,
}


@dataclass(frozen=True)
class SolverDefaults:
    inner_max_iter: int
    inner_tol: float = 1e-3
    max_outer: int = 20
    alm_tol: float = 1e-3
    backend: str = "compiled"


@dataclass(frozen=True)
class AlmDefaults:
    initial_penalty: float = 1e3
    penalty_update_factor: float = 5.0
    max_penalty: float = 0.0
    sufficient_decrease_factor: float = 0.25
    warm_start_inner: bool = True


@dataclass(frozen=True)
class BackwardDefaults:
    max_iterations: int = 800
    linear_solver: str = "cg"
    constraint_penalty_scale: float = 1.0
    constraint_penalty_max: float = 1e6


@dataclass(frozen=True)
class SafePdpDefaults:
    sensitivity_mode: str = "coc"
    barrier_gamma: float = 1e-2
    tol: float | None = None


TRAIN_SOLVER_DEFAULTS = {
    "cartpole": SolverDefaults(inner_max_iter=1500),
    "quadrotor": SolverDefaults(inner_max_iter=1500),
    "robot_arm": SolverDefaults(inner_max_iter=1500),
}

MPC_SOLVER_DEFAULTS = {
    "cartpole": SolverDefaults(inner_max_iter=1500, max_outer=20),
    "quadrotor": SolverDefaults(inner_max_iter=1500, max_outer=20),
    "robot_arm": SolverDefaults(inner_max_iter=1500, max_outer=20),
}


ALM_DEFAULTS = {
    "cartpole": AlmDefaults(initial_penalty=10),
    "quadrotor": AlmDefaults(initial_penalty=10),
    "robot_arm": AlmDefaults(initial_penalty=1e3),
}

BACKWARD_DEFAULTS = {
    "cartpole": BackwardDefaults(),
    "quadrotor": BackwardDefaults(),
    "robot_arm": BackwardDefaults(),
}

SAFEPDP_DEFAULTS = {
    "cartpole": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
    "quadrotor": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
    "robot_arm": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
}


def model_dir(model: str) -> Path:
    return EXPERIMENT_ROOT / model


def results_dir(model: str) -> Path:
    return model_dir(model) / "results"


def model_prefix(model: str) -> str:
    return MODEL_PREFIX[model]


def legacy_mpc_snapshot_path(model: str) -> Path:
    prefix = MODEL_PREFIX[model]
    return results_dir(model) / f"{prefix}_mpc_lapanda.mpc_snapshots.npz"
