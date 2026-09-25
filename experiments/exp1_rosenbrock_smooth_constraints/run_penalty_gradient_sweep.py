"""Diagnostic sweep: gradient error versus ALM penalty.

This script fixes one smooth constrained Rosenbrock instance and evaluates
lapanda with different constant penalty values. At every returned lapanda
solution, the original-NLP KKT adjoint is solved to high accuracy and used as
the sensitivity reference.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from experiments.common import relative_error
from experiments.common import write_json
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    constraint_violation_inf,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from experiments.exp1_rosenbrock_smooth_constraints.kkt_utils import (
    build_kkt_derivative_function,
    cosine_similarity,
    same_forward_kkt_reference,
)
from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver


BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#00FF00"
GRID = "#E6E8F0"
INK = "#1F2430"
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"

TICK_FONTSIZE = 8.0
LABEL_FONTSIZE = 8.0
LEGEND_FONTSIZE = 7.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def solver_options(args: argparse.Namespace) -> SolverOptions:
    options = SolverOptions()
    options.max_iterations = args.inner_max_iter
    options.tolerance = args.tol
    options.buffer_size = 10
    options.max_stable_iter = 120
    options.verbose = 0
    return options


def alm_options(args: argparse.Namespace, rho: float) -> AlmOptions:
    options = AlmOptions()
    options.max_iterations = args.max_outer
    options.tolerance = args.tol
    options.initial_penalty = rho
    options.penalty_update_factor = 1.0
    options.max_penalty = rho
    options.sufficient_decrease_factor = 0.25
    options.warm_start_inner = False
    options.verbose = 0
    return options


def backward_options(args: argparse.Namespace) -> BackwardOptions:
    options = BackwardOptions()
    options.enable = True
    options.tolerance = args.backward_tol
    options.max_iterations = args.backward_max_iter
    options.linear_solver = "cg"
    options.constraint_penalty_scale = 1.0
    options.constraint_penalty_max = 0.0
    return options


def style_axis(ax) -> None:
    ax.grid(True, color=GRID, linestyle="-", linewidth=0.42, alpha=0.70)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=2.4, width=0.68, pad=1.3)
    ax.xaxis.labelpad = 1.8
    ax.yaxis.labelpad = 1.8
    for spine in ax.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#20232A")


def plot_rows(rows: list[dict], out_path: Path) -> None:
    rho = np.asarray([float(row["penalty"]) for row in rows], dtype=float)
    grad_err = np.asarray([float(row["gradient_relative_error"]) for row in rows], dtype=float)
    violation = np.asarray([float(row["constraint_violation_inf"]) for row in rows], dtype=float)
    backward_time_ms = 1e3 * np.asarray([float(row["backward_time_sec"]) for row in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.05), dpi=300)
    ax.plot(
        rho,
        grad_err,
        color=BLUE,
        marker="o",
        markersize=3.2,
        linewidth=1.25,
        label="gradient error",
        zorder=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Penalty")
    ax.set_ylabel("Relative gradient error")
    style_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(
        rho,
        backward_time_ms,
        color=RED,
        marker="s",
        markersize=3.0,
        linewidth=1.10,
        linestyle="--",
        label="backward time",
        zorder=7,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Backward time (ms)")
    ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=2.4, width=0.68, pad=1.3)
    ax2.yaxis.label.set_size(LABEL_FONTSIZE)
    ax2.yaxis.labelpad = 1.8
    for spine in ax2.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#20232A")

    handles = ax.get_lines() + ax2.get_lines()
    legend = ax.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        borderpad=0.24,
        labelspacing=0.16,
        handlelength=1.50,
        handletextpad=0.42,
    )
    legend.get_frame().set_linewidth(0.50)

    fig.subplots_adjust(left=0.165, right=0.855, bottom=0.195, top=0.965)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--penalties", nargs="+", type=float, default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3])
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--backward-tol", type=float, default=1e-2)
    parser.add_argument("--inner-max-iter", type=int, default=4000)
    parser.add_argument("--max-outer", type=int, default=1)
    parser.add_argument("--backward-max-iter", type=int, default=800)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--reference-active-tol", type=float, default=1e-7)
    parser.add_argument("--reference-residual-tol", type=float, default=1e-10)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    csv_path = out_dir / "exp1_penalty_gradient_sweep.csv"
    fig_path = out_dir / "exp1_penalty_gradient_sweep.png"

    instance = build_problem(args.n, constraint_stride=args.constraint_stride)
    theta = theta_nominal()
    target = sinusoidal_target(args.n)
    x0 = initial_point(args.n, 100 + args.trial_id)

    derivative_fun = build_kkt_derivative_function(
        instance, f"exp1_penalty_reference_derivatives_n{args.n}"
    )

    solver = build_solver(
        instance.problem,
        backend=args.backend,
        name=f"exp1_penalty_sweep_n{args.n}",
        cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp1_penalty_sweep",
        force=args.force,
    )

    rows: list[dict] = []
    for rho in args.penalties:
        def solve_once():
            return solver.solve_lapanda(
                x0,
                theta,
                target,
                instance.constraint_lower,
                instance.constraint_upper,
                inner_solver_options=solver_options(args),
                alm_options=alm_options(args, rho),
                backward_options=backward_options(args),
                multiplier0=None,
                penalty0=None,
                adjoint0=None,
            )

        solve_once()  # Untimed warm-up for dispatch and workspace initialization.
        timed_results = [solve_once() for _ in range(args.timing_repetitions)]
        result = timed_results[-1]
        solution = np.asarray(result["solution"], dtype=float)
        grad = np.asarray(result["grad_theta"], dtype=float)
        reference = same_forward_kkt_reference(
            instance,
            derivative_fun,
            solution,
            theta,
            target,
            np.asarray(result["multipliers"], dtype=float),
            np.asarray(result["penalties"], dtype=float),
            args.reference_active_tol,
        )
        if reference["linear_residual"] > args.reference_residual_tol:
            raise RuntimeError(
                f"KKT reference residual is too large for rho={rho}: "
                f"{reference['linear_residual']:.3e}"
            )
        reference_grad = np.asarray(reference["gradient"], dtype=float)
        row = {
            "n": args.n,
            "m": instance.m,
            "trial_id": args.trial_id,
            "penalty": float(rho),
            "final_penalty": float(result.get("penalty", np.nan)),
            "max_outer": args.max_outer,
            "tol": args.tol,
            "outer_loss_value": float(instance.outer_loss_fun(solution, target)),
            "reference_linear_residual": reference["linear_residual"],
            "constraint_violation_inf": constraint_violation_inf(instance, solution, theta),
            "gradient_relative_error": relative_error(grad, reference_grad),
            "gradient_cosine_similarity": cosine_similarity(grad, reference_grad),
            "timing_repetitions": args.timing_repetitions,
            "forward_time_sec": float(
                np.mean([item.get("forward_time_sec", np.nan) for item in timed_results])
            ),
            "backward_time_sec": float(
                np.mean([item.get("backward_time_sec", np.nan) for item in timed_results])
            ),
            "backward_time_sec_std": float(
                np.std(
                    [item.get("backward_time_sec", np.nan) for item in timed_results],
                    ddof=1 if args.timing_repetitions > 1 else 0,
                )
            ),
            "forward_iterations": int(np.sum(np.asarray(result.get("inner_iterations", []), dtype=int))),
            "backward_iterations": int(result.get("backward_iterations", -1)),
            "backward_solver_used": result.get("backward_solver_used", "unknown"),
            "backward_fallback_used": bool(result.get("backward_fallback_used", False)),
            "final_residual": float(result.get("final_residual", np.nan)),
        }
        rows.append(row)
        print(
            f"rho={rho:.1e} grad_err={row['gradient_relative_error']:.3e} "
            f"viol={row['constraint_violation_inf']:.3e} "
            f"bwd={1e3 * row['backward_time_sec']:.3f}ms "
            f"loss={row['outer_loss_value']:.3e}"
        )

    write_csv(csv_path, rows)
    write_json(
        out_dir / "config_penalty_gradient_sweep.json",
        {
            **vars(args),
            "reference": "high-accuracy original-NLP KKT sensitivity solve at each lapanda solution",
            "linear_solver": "CG with automatic MINRES fallback",
            "timing": "mean internal solver time after one untimed warm-up per penalty",
        },
    )
    plot_rows(rows, fig_path)
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
