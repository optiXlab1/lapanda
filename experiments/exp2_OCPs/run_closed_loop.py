"""Run the closed-loop OCP imitation experiment.

This is the main entry point for the closed-loop OCP results. It can generate
the lapanda teacher rollout, train lapanda, and run the SafePDP baseline
for cartpole, quadrotor, and robot arm with one command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.exp2_OCPs.common.closed_loop_train import run_closed_loop_train
from experiments.exp2_OCPs.common.mpc_rollout import run_mpc_rollout
from experiments.exp2_OCPs.config import (
    ALM_DEFAULTS,
    BACKWARD_DEFAULTS,
    CLOSED_LOOP_LEARNING_RATE,
    MODEL_PREFIX,
    MPC_SOLVER_DEFAULTS,
    SAFEPDP_DEFAULTS,
    TRAIN_SOLVER_DEFAULTS,
)
from experiments.exp2_OCPs.safepdp_baseline import main as run_safepdp

MODELS = ("cartpole", "quadrotor", "robot_arm")


def teacher_mpc_path(model: str, tag: str | None) -> Path:
    stem = f"{MODEL_PREFIX[model]}_mpc_lapanda"
    if tag:
        stem = f"{stem}_{tag}"
    return REPO_ROOT / "experiments" / "exp2_OCPs" / model / "results" / f"{stem}.mpc_snapshots.npz"


def safepdp_mpc_path(model: str, tag: str | None) -> Path:
    stem = f"{MODEL_PREFIX[model]}_mpc_safepdp"
    if tag:
        stem = f"{stem}_{tag}"
    return REPO_ROOT / "experiments" / "exp2_OCPs" / model / "baselines" / "results" / f"{stem}.mpc_snapshots.npz"


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


def run_alm_teacher(model: str, args, settings: dict, teacher_path: Path) -> None:
    defaults = MPC_SOLVER_DEFAULTS[model]
    mpc_args = SimpleNamespace(
        horizon=args.horizon,
        steps=args.steps,
        backend=defaults.backend,
        inner_max_iter=defaults.inner_max_iter,
        inner_tol=args.inner_tol,
        max_outer=defaults.max_outer,
        alm_tol=args.alm_tol,
        force=args.force,
        output_tag=args.teacher_tag,
        theta_true_override=args.theta_true_override,
        theta_initial_override=args.theta_initial_override,
        compute_backward=True,
        no_penalty_warm_start=False,
        memory_only=args.memory_only,
        **settings,
    )
    if args.dry_run:
        print(f"ALM teacher {model}: {teacher_path}")
    else:
        run_mpc_rollout(model, mpc_args)


def run_alm_training(model: str, args, settings: dict, teacher_path: Path) -> None:
    defaults = TRAIN_SOLVER_DEFAULTS[model]
    train_args = SimpleNamespace(
        horizon=args.horizon,
        epochs=args.epochs,
        steps=args.steps,
        learning_rate=args.learning_rate or CLOSED_LOOP_LEARNING_RATE[model],
        backend=defaults.backend,
        inner_max_iter=defaults.inner_max_iter,
        inner_tol=args.inner_tol,
        max_outer=defaults.max_outer,
        alm_tol=args.alm_tol,
        force=args.force,
        seed=args.theta_seed,
        theta_init=args.theta_init,
        learn_theta_indices=args.learn_theta_indices,
        output_tag=args.output_tag,
        teacher_source=str(teacher_path),
        theta_true_override=args.theta_true_override,
        theta_initial_override=args.theta_initial_override,
        warm_start_penalty=args.warm_start_penalty,
        memory_only=args.memory_only,
        **settings,
    )
    if args.dry_run:
        print(f"ALM train {model}: teacher={teacher_path}")
    else:
        run_closed_loop_train(model, train_args)


def run_safepdp_closed_loop(model: str, args, teacher_path: Path, *, sensitivity_mode_override=None, output_tag=None) -> None:
    safepdp = SAFEPDP_DEFAULTS[model]
    sensitivity_mode = sensitivity_mode_override or args.safepdp_sensitivity_mode or safepdp.sensitivity_mode
    gamma = args.safepdp_barrier_gamma if args.safepdp_barrier_gamma is not None else safepdp.barrier_gamma
    tol = args.safepdp_tol if args.safepdp_tol is not None else (safepdp.tol or args.inner_tol)
    learning_rate = args.learning_rate or CLOSED_LOOP_LEARNING_RATE[model]

    timing_path = safepdp_mpc_path(model, args.teacher_tag)
    if not args.skip_safepdp_mpc and (args.force_teacher_generation or not timing_path.exists()):
        mpc_argv = [
            "--mode",
            "mpc",
            "--horizon",
            str(args.horizon),
            "--steps",
            str(args.steps),
            "--tol",
            str(tol),
            "--sensitivity-mode",
            sensitivity_mode,
            "--barrier-gamma",
            str(gamma),
            "--output-tag",
            args.teacher_tag,
        ]
        if args.memory_only:
            mpc_argv.append("--memory-only")
        if args.dry_run:
            print(f"SafePDP MPC {model}: {' '.join(mpc_argv)}")
        else:
            run_safepdp(mpc_argv, model_override=model)

    train_argv = [
        "--mode",
        "closed_loop",
        "--epochs",
        str(args.epochs),
        "--horizon",
        str(args.horizon),
        "--steps",
        str(args.steps),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(args.theta_seed),
        "--theta-init",
        args.theta_init,
        "--output-tag",
        output_tag or args.output_tag,
        "--teacher-source",
        str(teacher_path),
        "--tol",
        str(tol),
        "--sensitivity-mode",
        sensitivity_mode,
        "--barrier-gamma",
        str(gamma),
    ]
    if args.memory_only:
        train_argv.append("--memory-only")
    if args.dry_run:
        print(f"SafePDP train {model}: {' '.join(train_argv)}")
    else:
        run_safepdp(train_argv, model_override=model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lapanda", "safepdp", "safepdp_coc"],
        default=["lapanda", "safepdp_coc"],
        help="Paper default runs lapanda and the reported SafePDP-COC baseline.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--theta-seed", type=int, default=1)
    parser.add_argument("--theta-init", choices=["default", "uniform01"], default="default")
    parser.add_argument("--theta-true-override", type=str, default="")
    parser.add_argument("--theta-initial-override", type=str, default="")
    parser.add_argument(
        "--learn-theta-indices",
        type=str,
        default="",
        help="Comma-separated theta indices to update for lapanda training; empty updates all parameters.",
    )
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
    parser.add_argument("--teacher-tag", type=str, default="teacher")
    parser.add_argument("--output-tag", type=str, default="closed_loop")
    parser.add_argument("--skip-teacher-generation", action="store_true")
    parser.add_argument("--force-teacher-generation", action="store_true")
    parser.add_argument("--skip-safepdp-mpc", action="store_true")
    parser.add_argument("--warm-start-penalty", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-only", action="store_true", help="Measure build and first solve memory only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for model in args.models:
        settings = common_alm_settings(args, model)
        teacher_path = teacher_mpc_path(model, args.teacher_tag)

        if "lapanda" in args.methods:
            if not args.skip_teacher_generation and (args.force_teacher_generation or not teacher_path.exists()):
                run_alm_teacher(model, args, settings, teacher_path)
            run_alm_training(model, args, settings, teacher_path)

        if "safepdp" in args.methods:
            run_safepdp_closed_loop(model, args, teacher_path)
        if "safepdp_coc" in args.methods:
            run_safepdp_closed_loop(
                model,
                args,
                teacher_path,
                sensitivity_mode_override="coc",
                output_tag=f"{args.output_tag}_coc",
            )


if __name__ == "__main__":
    main()
