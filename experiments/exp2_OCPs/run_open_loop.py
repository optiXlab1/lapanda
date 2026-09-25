"""Run the open-loop fixed-archive OCP imitation experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver

from experiments.exp2_OCPs.common.open_loop_train import run_open_loop_train
from experiments.exp2_OCPs.config import (
    ALM_DEFAULTS,
    BACKWARD_DEFAULTS,
    MODEL_PREFIX,
    OPEN_LOOP_LEARNING_RATE,
    SAFEPDP_DEFAULTS,
    TRAIN_SOLVER_DEFAULTS,
    results_dir,
)
from experiments.exp2_OCPs.problem_bank import build_problem, constraint_violation_inf
from experiments.exp2_OCPs.safepdp_baseline import main as run_safepdp
from experiments.exp2_OCPs.utils import build_alm_solver, make_variable_from_state, solve_alm_once

MODELS = ("cartpole", "quadrotor", "robot_arm")
TEACHER_SAMPLING_VERSION = "feasible_v2"


def teacher_archive_path(model: str, horizon: int, samples: int, seed: int) -> Path:
    suffix = f"_{TEACHER_SAMPLING_VERSION}" if model == "robot_arm" else ""
    return results_dir(model) / f"{MODEL_PREFIX[model]}_expert{samples}_h{horizon}_seed{seed}{suffix}.mpc_snapshots.npz"


def sample_initial_states(instance, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if instance.name == "cartpole":
        states = np.zeros((count, instance.nx))
        states[:, 0] = rng.uniform(-0.5, 0.5, size=count)
        states[:, 1] = rng.uniform(-0.5, 0.5, size=count)
        states[:, 2] = rng.uniform(-np.pi, np.pi, size=count)
        states[:, 3] = rng.uniform(-1.0, 1.0, size=count)
        return states
    if instance.name == "planar_quadrotor":
        states = np.tile(np.asarray(instance.x_initial, dtype=float), (count, 1))
        states[:, 0] += rng.uniform(-0.25, 0.25, size=count)
        states[:, 1] += rng.uniform(-0.10, 0.10, size=count)
        states[:, 2] += rng.uniform(-0.10, 0.10, size=count)
        states[:, 3] += rng.uniform(-0.10, 0.10, size=count)
        states[:, 4] += rng.uniform(-0.08, 0.08, size=count)
        states[:, 5] += rng.uniform(-0.10, 0.10, size=count)
        return states
    if instance.name == "two_link_arm":
        states = np.empty((count, instance.nx), dtype=float)
        states[:, 0] = rng.uniform(-1.10, -0.80, size=count)
        states[:, 1] = rng.uniform(1.05, 1.28, size=count)
        return states
    raise ValueError(instance.name)


def model_initial_penalty(args, model: str) -> float:
    model_value = getattr(args, f"{model}_alm_initial_penalty")
    if model_value is not None:
        return model_value
    if args.alm_initial_penalty is not None:
        return args.alm_initial_penalty
    return ALM_DEFAULTS[model].initial_penalty


def common_alm_settings(args, model: str):
    alm = ALM_DEFAULTS[model]
    backward = BACKWARD_DEFAULTS[model]
    return {
        "alm_initial_penalty": model_initial_penalty(args, model),
        "alm_penalty_update_factor": args.alm_penalty_update_factor or alm.penalty_update_factor,
        "alm_max_penalty": alm.max_penalty if args.alm_max_penalty is None else args.alm_max_penalty,
        "backward_max_iterations": args.backward_max_iterations or backward.max_iterations,
        "backward_linear_solver": args.backward_linear_solver or backward.linear_solver,
        "backward_constraint_penalty_scale": (
            backward.constraint_penalty_scale
            if args.backward_constraint_penalty_scale is None
            else args.backward_constraint_penalty_scale
        ),
        "backward_constraint_penalty_max": (
            backward.constraint_penalty_max
            if args.backward_constraint_penalty_max is None
            else args.backward_constraint_penalty_max
        ),
    }


def generate_teacher_archive(model: str, args, settings: dict, out_path: Path) -> None:
    if args.dry_run:
        print(f"Generate teacher archive {model}: {out_path}")
        return

    instance = build_problem({"cartpole": "cartpole", "quadrotor": "quadrotor", "robot_arm": "arm"}[model], args.horizon)
    solver = build_alm_solver(
        build_solver,
        instance,
        name=f"{model}_expert_h{args.horizon}",
        backend="compiled",
        force=args.force,
        cache_tag=f"{model}_expert",
    )
    solve_args = SimpleNamespace(
        inner_max_iter=args.teacher_inner_max_iter,
        inner_tol=args.inner_tol,
        max_outer=args.teacher_max_outer,
        alm_tol=args.alm_tol,
        **settings,
    )
    snapshots = []
    sampled_states = sample_initial_states(instance, args.teacher_samples, args.teacher_seed)
    if model == "robot_arm":
        assert np.all((-1.10 <= sampled_states[:, 0]) & (sampled_states[:, 0] <= 0.75))
        assert np.all((-0.65 <= sampled_states[:, 1]) & (sampled_states[:, 1] <= 1.28))
    for sample_index, state in enumerate(sampled_states):
        demo_zero = np.zeros(instance.horizon * instance.nu)
        variable = make_variable_from_state(instance, state, demo_zero)
        result = solve_alm_once(
            solver,
            instance,
            demo_zero,
            instance.theta_true,
            variable,
            solve_args,
            SolverOptions,
            AlmOptions,
            BackwardOptions,
            compute_backward=False,
        )
        solution = np.asarray(result["solution"], dtype=float).reshape(-1)
        states = np.asarray(instance.rollout_fun(solution, instance.theta_true, variable), dtype=float)
        snapshots.append(
            {
                "sample_index": int(sample_index),
                "initial_state": np.asarray(state, dtype=float).copy(),
                "states": states.copy(),
                "controls": solution.copy(),
                "constraint_violation_inf": float(constraint_violation_inf(instance, solution, instance.theta_true, variable)),
                "outer_iterations": int(result["iterations"]),
                "inner_iterations": np.asarray(result["inner_iterations"], dtype=int).copy(),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        snapshots=np.asarray(snapshots, dtype=object),
        model=model,
        horizon=args.horizon,
        samples=args.teacher_samples,
        teacher_seed=args.teacher_seed,
        allow_pickle=True,
    )
    print(f"wrote {out_path}")


def run_alm_training(model: str, args, settings: dict, teacher_path: Path) -> None:
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    train_args = SimpleNamespace(
        horizon=args.horizon,
        epochs=args.epochs,
        learning_rate=args.learning_rate or OPEN_LOOP_LEARNING_RATE[model],
        backend=defaults.backend,
        inner_max_iter=defaults.inner_max_iter,
        inner_tol=args.inner_tol,
        max_outer=defaults.max_outer,
        alm_tol=args.alm_tol,
        force=args.force,
        seed=args.theta_seed,
        theta_init=args.theta_init,
        output_tag=args.output_tag,
        teacher_source=str(teacher_path),
        teacher_samples=args.teacher_samples,
        teacher_seed=args.teacher_seed,
        memory_only=args.memory_only,
        **settings,
    )
    if args.dry_run:
        print(f"ALM open-loop train {model}: teacher={teacher_path}")
    else:
        run_open_loop_train(model, train_args)


def run_safepdp_training(model: str, args, teacher_path: Path, *, sensitivity_mode_override=None, output_tag=None) -> None:
    safepdp = SAFEPDP_DEFAULTS[model]
    sensitivity_mode = sensitivity_mode_override or args.safepdp_sensitivity_mode or safepdp.sensitivity_mode
    gamma = args.safepdp_barrier_gamma if args.safepdp_barrier_gamma is not None else safepdp.barrier_gamma
    argv = [
        "--mode",
        "imitation",
        "--epochs",
        str(args.epochs),
        "--horizon",
        str(args.horizon),
        "--learning-rate",
        str(args.learning_rate or OPEN_LOOP_LEARNING_RATE[model]),
        "--seed",
        str(args.theta_seed),
        "--theta-init",
        args.theta_init,
        "--output-tag",
        output_tag or args.output_tag,
        "--teacher-source",
        str(teacher_path),
        "--tol",
        str(args.safepdp_tol if args.safepdp_tol is not None else (safepdp.tol or args.inner_tol)),
        "--sensitivity-mode",
        sensitivity_mode,
        "--barrier-gamma",
        str(gamma),
    ]
    if args.memory_only:
        argv.append("--memory-only")
    if args.dry_run:
        print(f"SafePDP open-loop train {model}: {' '.join(argv)}")
    else:
        run_safepdp(argv, model_override=model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lapanda", "safepdp", "safepdp_coc"],
        default=["lapanda", "safepdp_coc"],
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--theta-seed", type=int, default=1)
    parser.add_argument("--theta-init", choices=["default", "uniform01"], default="uniform01")
    parser.add_argument("--teacher-samples", type=int, default=32)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--teacher-inner-max-iter", type=int, default=1500)
    parser.add_argument("--teacher-max-outer", type=int, default=20)
    parser.add_argument("--inner-tol", type=float, default=1e-3)
    parser.add_argument("--alm-tol", type=float, default=1e-3)
    parser.add_argument("--alm-initial-penalty", type=float, default=None)
    parser.add_argument("--alm-penalty-update-factor", type=float, default=None)
    parser.add_argument("--alm-max-penalty", type=float, default=None)
    parser.add_argument("--cartpole-alm-initial-penalty", dest="cartpole_alm_initial_penalty", type=float, default=None)
    parser.add_argument("--quadrotor-alm-initial-penalty", dest="quadrotor_alm_initial_penalty", type=float, default=None)
    parser.add_argument("--robot-arm-alm-initial-penalty", dest="robot_arm_alm_initial_penalty", type=float, default=None)
    parser.add_argument("--backward-max-iterations", type=int, default=None)
    parser.add_argument(
        "--backward-linear-solver",
        choices=("auto", "cg", "minres", "gmres"),
        default=None,
    )
    parser.add_argument("--backward-constraint-penalty-scale", type=float, default=None)
    parser.add_argument("--backward-constraint-penalty-max", type=float, default=None)
    parser.add_argument("--safepdp-tol", type=float, default=None)
    parser.add_argument("--safepdp-sensitivity-mode", choices=["coc", "barrier"], default=None)
    parser.add_argument("--safepdp-barrier-gamma", type=float, default=None)
    parser.add_argument("--output-tag", type=str, default="fixed32")
    parser.add_argument("--skip-teacher-generation", action="store_true")
    parser.add_argument("--force-teacher-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for model in args.models:
        settings = common_alm_settings(args, model)
        teacher_path = teacher_archive_path(model, args.horizon, args.teacher_samples, args.teacher_seed)
        if not args.skip_teacher_generation and (args.force_teacher_generation or not teacher_path.exists()):
            generate_teacher_archive(model, args, settings, teacher_path)
        if "lapanda" in args.methods:
            run_alm_training(model, args, settings, teacher_path)
        if "safepdp" in args.methods:
            run_safepdp_training(model, args, teacher_path)
        if "safepdp_coc" in args.methods:
            run_safepdp_training(
                model,
                args,
                teacher_path,
                sensitivity_mode_override="coc",
                output_tag=f"{args.output_tag}_coc",
            )


if __name__ == "__main__":
    main()
