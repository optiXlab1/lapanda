"""Plot teacher, initial student, and trained student plans."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
circle_dir = EXP_DIR / "circle"
if str(circle_dir) not in sys.path:
    sys.path.insert(0, str(circle_dir))

import config as exp3_config

_circle_spec = importlib.util.spec_from_file_location(
    "circle_lapanda_impl", circle_dir / "run_lapanda.py"
)
if _circle_spec is None or _circle_spec.loader is None:
    raise RuntimeError("could not load circle/run_lapanda.py")
_circle_impl = importlib.util.module_from_spec(_circle_spec)
sys.modules[_circle_spec.name] = _circle_impl
_circle_spec.loader.exec_module(_circle_impl)
rollout = _circle_impl.rollout

_rect_spec = importlib.util.spec_from_file_location(
    "rectangle_lapanda_impl", Path(__file__).resolve().parent / "run_lapanda.py"
)
if _rect_spec is None or _rect_spec.loader is None:
    raise RuntimeError("could not load rectangle/run_lapanda.py")
_rect_impl = importlib.util.module_from_spec(_rect_spec)
sys.modules[_rect_spec.name] = _rect_impl
_rect_spec.loader.exec_module(_rect_impl)

build_alm_solver = _rect_impl.build_alm_solver
build_rectangle_imitation_case = _rect_impl.build_rectangle_imitation_case
solve_alm_once = _rect_impl.solve_alm_once
STEER_LIMIT = _rect_impl.STEER_LIMIT


BASE_RECT = np.asarray(exp3_config.RECTANGLE_BASE, dtype=float)
PLOT_RECT_Y_SHIFT = 0.0
TRUE_RECT_Y_SHIFT = 0.10
BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#00FF00"
ORANGE = "#B37A2C"
GRAY = "#666666"
GRID = "#E6E8F0"
INK = "#1F2430"
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"
LEGEND_FRAME_ALPHA = 0.96
C_EXPORT_TEACHER_MARGINS = np.asarray(exp3_config.RECTANGLE_TEACHER_MARGINS, dtype=float)
ICLR_TEXT_WIDTH_IN = 6.75

TICK_FONTSIZE = 8.0
LABEL_FONTSIZE = 8.0
LEGEND_FONTSIZE = 5.2
PANEL_LABEL_FONTSIZE = 9.8


def rect_bounds(margins):
    left, right, bottom, top = np.asarray(margins, dtype=float)
    return BASE_RECT + np.asarray([-left, right, -bottom + PLOT_RECT_Y_SHIFT, top + PLOT_RECT_Y_SHIFT], dtype=float)


def plot_base_rect():
    return BASE_RECT + np.asarray([0.0, 0.0, TRUE_RECT_Y_SHIFT, TRUE_RECT_Y_SHIFT], dtype=float)


def draw_rect(ax, bounds, label, color, linestyle="-", linewidth=1.6, zorder=2):
    xmin, xmax, ymin, ymax = bounds
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=zorder)


def draw_hatched_rect(ax, bounds, label="rectangle"):
    from matplotlib.patches import Rectangle

    xmin, xmax, ymin, ymax = bounds
    patch = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        facecolor="none",
        edgecolor="black",
        linewidth=1.35,
        linestyle="-",
        hatch="////",
        label=label,
        zorder=8,
    )
    ax.add_patch(patch)
    return patch


def read_training(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if str(row.get("epoch", "")).isdigit()]
    first = rows[0]
    last = rows[-1]
    if "margins" in first:
        initial_margins = np.asarray(json.loads(first["margins"]), dtype=float)
        final_margins = np.asarray(json.loads(last["margins"]), dtype=float)
        teacher_margins = np.asarray(json.loads(first["teacher_margins"]), dtype=float)
    else:
        initial_margins = np.asarray(
            [first["margin_left"], first["margin_right"], first["margin_bottom"], first["margin_top"]],
            dtype=float,
        )
        final_margins = np.asarray(
            [last["margin_left"], last["margin_right"], last["margin_bottom"], last["margin_top"]],
            dtype=float,
        )
        teacher_margins = C_EXPORT_TEACHER_MARGINS.copy()
    return {
        "teacher_margins": teacher_margins,
        "initial_margins": initial_margins,
        "final_margins": final_margins,
        "initial_loss": float(first["loss"]),
        "final_loss": float(last["loss"]),
    }


def plot_training(training_csv: Path, outdir: Path):
    with training_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if str(row.get("epoch", "")).isdigit()]
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    if "margins" in rows[0]:
        margin_rows = [json.loads(r["margins"]) for r in rows]
        teacher = json.loads(rows[0]["teacher_margins"])
    else:
        margin_rows = [
            [float(r["margin_left"]), float(r["margin_right"]), float(r["margin_bottom"]), float(r["margin_top"])]
            for r in rows
        ]
        teacher = C_EXPORT_TEACHER_MARGINS.tolist()
    margin_cols = list(zip(*margin_rows))
    margin_names = ["left", "right", "bottom", "top"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axs[0].plot(epochs, losses, marker="o", markersize=3)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("lapanda margin imitation")

    for idx, col in enumerate(margin_names):
        axs[1].plot(epochs, margin_cols[idx], marker="o", markersize=2, label=col)
        axs[1].axhline(teacher[idx], color=f"C{idx}", linestyle=":", alpha=0.6)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("margin")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=8)
    axs[1].set_title("learned margins")

    path = outdir / "loss_margins.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def solve_plan(margins, teacher_u, args):
    case = build_rectangle_imitation_case(margins, teacher_u, steer_limit=args.steer_limit)
    x0 = np.zeros_like(case.x0) if args.zero_initial_guess else case.x0.copy()
    result = solve_alm_once(args.solver, case, x0, False, args)
    u = np.asarray(result["solution"], dtype=float)
    traj = rollout(u, case.variable[:6], case.dt)
    return case, u, traj


def connect_start(traj, case):
    traj = np.asarray(traj, dtype=float)
    if traj.size == 0:
        return traj
    start = np.asarray(case.variable[: traj.shape[1]], dtype=float)
    if np.linalg.norm(traj[0, :2] - start[:2]) < 1e-10:
        return traj
    return np.vstack([start, traj])


def style_axis(ax):
    ax.grid(True, color=GRID, linestyle="-", linewidth=0.42, alpha=0.70)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=2.4, width=0.68, pad=1.3)
    ax.xaxis.labelpad = 1.8
    ax.yaxis.labelpad = 1.8
    for spine in ax.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#20232A")


def style_legend(ax, loc="upper right"):
    legend = ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=LEGEND_FRAME_ALPHA,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        borderpad=0.24,
        labelspacing=0.16,
        handlelength=1.30,
        handletextpad=0.38,
        loc=loc,
    )
    legend.get_frame().set_linewidth(0.50)
    return legend


def plot_publication_1x3(training_csv: Path, outdir: Path, info: dict, args) -> Path:
    with training_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if str(row.get("epoch", "")).isdigit()]
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=int)
    losses = np.asarray([float(row["loss"]) for row in rows], dtype=float)
    if "margins" in rows[0]:
        margins = np.asarray([json.loads(row["margins"]) for row in rows], dtype=float)
        teacher_margins = np.asarray(json.loads(rows[0]["teacher_margins"]), dtype=float)
    else:
        margins = np.asarray(
            [
                [row["margin_left"], row["margin_right"], row["margin_bottom"], row["margin_top"]]
                for row in rows
            ],
            dtype=float,
        )
        teacher_margins = C_EXPORT_TEACHER_MARGINS.copy()
    margin_names = ["left", "right", "bottom"]
    margin_colors = [BLUE, RED, GREEN]

    teacher_case = build_rectangle_imitation_case(info["teacher_margins"], steer_limit=args.steer_limit)
    teacher_x0 = np.zeros_like(teacher_case.x0) if args.zero_initial_guess else teacher_case.x0.copy()
    teacher_result = solve_alm_once(args.solver, teacher_case, teacher_x0, False, args)
    teacher_u = np.asarray(teacher_result["solution"], dtype=float)
    teacher_traj = connect_start(rollout(teacher_u, teacher_case.variable[:6], teacher_case.dt), teacher_case)
    initial_case, initial_u, initial_traj = solve_plan(info["initial_margins"], teacher_u, args)
    final_case, final_u, final_traj = solve_plan(info["final_margins"], teacher_u, args)
    initial_traj = connect_start(initial_traj, initial_case)
    final_traj = connect_start(final_traj, final_case)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(ICLR_TEXT_WIDTH_IN, 1.70), dpi=300)

    ax = axes[0]
    ax.plot(epochs, losses, color=BLUE, linewidth=1.15)
    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Imitation loss", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(float(epochs.min()), float(epochs.max()))
    ax.margins(x=0.0)
    style_axis(ax)

    ax = axes[1]
    teacher_style = "-"
    initial_style = "-."
    learned_style = "--"
    draw_hatched_rect(ax, plot_base_rect(), "_nolegend_")
    draw_rect(ax, rect_bounds(info["teacher_margins"]), "_nolegend_", GREEN, teacher_style, linewidth=0.86, zorder=6)
    draw_rect(ax, rect_bounds(info["initial_margins"]), "_nolegend_", RED, initial_style, linewidth=0.82, zorder=8)
    draw_rect(ax, rect_bounds(info["final_margins"]), "_nolegend_", BLUE, learned_style, linewidth=0.98, zorder=10)
    ax.plot(
        teacher_traj[:, 0],
        teacher_traj[:, 1],
        color=GREEN,
        linestyle=teacher_style,
        linewidth=0.98,
        marker="^",
        markersize=1.35,
        markevery=3,
        label="target",
        zorder=8,
    )
    ax.plot(
        initial_traj[:, 0],
        initial_traj[:, 1],
        color=RED,
        linestyle=initial_style,
        linewidth=0.92,
        marker="s",
        markersize=1.28,
        markevery=3,
        label="initial",
        zorder=10,
    )
    ax.plot(
        final_traj[:, 0],
        final_traj[:, 1],
        color=BLUE,
        linestyle=learned_style,
        linewidth=1.06,
        marker="o",
        markersize=1.28,
        markevery=3,
        label="learned",
        zorder=12,
    )
    ax.scatter(
        [teacher_case.variable[0], teacher_case.variable[3]],
        [teacher_case.variable[1], teacher_case.variable[4]],
        c=[INK, GREEN],
        s=9,
        zorder=5,
    )
    ax.set_aspect("auto")
    ax.set_xlabel("$p_x$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("$p_y$", fontsize=LABEL_FONTSIZE)
    style_legend(ax, loc="upper right")
    style_axis(ax)
    ymax = max(np.max(teacher_traj[:, 1]), np.max(initial_traj[:, 1]), np.max(final_traj[:, 1]), 0.55)
    ax.set_ylim(-0.50, ymax + 0.30)

    ax = axes[2]
    teacher_margin_offsets = np.asarray([-0.002, 0.002, 0.0, 0.0], dtype=float)
    for index, name in enumerate(margin_names):
        color = margin_colors[index]
        ax.plot(
            epochs,
            margins[:, index],
            color=color,
            linewidth=0.96,
            alpha=0.94,
            label=name,
            zorder=3,
        )
        ax.axhline(
            teacher_margins[index] + teacher_margin_offsets[index],
            color=color,
            linestyle=(0, (3.5, 2.0)),
            linewidth=0.86,
            alpha=0.58,
            zorder=12,
        )
    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Margin", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(float(epochs.min()), float(epochs.max()))
    ax.set_ylim(0.0, 0.225)
    ax.margins(x=0.0)
    style_legend(ax, loc="upper right")
    style_axis(ax)

    stages = np.arange(teacher_u.size // 2)
    ax = axes[3]
    ax.plot(
        stages,
        teacher_u[1::2],
        color=GREEN,
        linestyle=teacher_style,
        linewidth=0.98,
        label="target",
        zorder=8,
    )
    ax.plot(
        stages,
        initial_u[1::2],
        color=RED,
        linestyle=initial_style,
        linewidth=0.92,
        label="initial",
        zorder=10,
    )
    ax.plot(
        stages,
        final_u[1::2],
        color=BLUE,
        linestyle=learned_style,
        linewidth=1.04,
        label="learned",
        zorder=12,
    )
    ax.set_xlabel("Stage", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Steer", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(float(stages.min()), float(stages.max()))
    ax.set_xticks([5, 10, 15])
    steer_values = np.concatenate([teacher_u[1::2], initial_u[1::2], final_u[1::2]])
    ax.set_ylim(float(np.min(steer_values)) - 0.08, float(np.max(steer_values)) + 0.50)
    ax.margins(x=0.0)
    style_legend(ax, loc="upper right")
    style_axis(ax)

    fig.subplots_adjust(left=0.066, right=0.996, top=0.900, bottom=0.350, wspace=0.42)
    panel_labels = ["(a) Loss", "(b) Trajectory", "(c) Margins", "(d) Steer"]
    for label, ax in zip(panel_labels, axes):
        ax.text(
            0.5,
            -0.43,
            label,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontfamily="Times New Roman",
            color=INK,
            transform=ax.transAxes,
            clip_on=False,
        )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"rectangle_imitation_1x4.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)
    return outdir / "rectangle_imitation_1x4.png"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "training_csv",
        nargs="?",
        default=str(
            exp3_config.RECTANGLE_C_EXPORT_RESULT_DIR / "rectangle_margin_imitation_c.csv"
        ),
    )
    p.add_argument("--outdir", default=None)
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--inner-max-iter", type=int, default=exp3_config.INNER_MAX_ITER)
    p.add_argument("--inner-max-stable-iter", type=int, default=exp3_config.INNER_MAX_STABLE_ITER)
    p.add_argument("--inner-tol", type=float, default=exp3_config.RECTANGLE_INNER_TOL)
    p.add_argument("--alm-max-iter", type=int, default=200)
    p.add_argument("--alm-tol", type=float, default=1e-6)
    p.add_argument("--alm-initial-penalty", type=float, default=exp3_config.ALM_INITIAL_PENALTY)
    p.add_argument("--alm-penalty-update-factor", type=float, default=exp3_config.ALM_PENALTY_UPDATE_FACTOR)
    p.add_argument("--backward-max-iter", type=int, default=exp3_config.BACKWARD_MAX_ITER)
    p.add_argument("--backward-tol", type=float, default=exp3_config.BACKWARD_TOL)
    p.add_argument("--backward-constraint-penalty-scale", type=float, default=1.0)
    p.add_argument("--backward-constraint-penalty-max", type=float, default=0.0)
    p.add_argument("--steer-limit", type=float, default=exp3_config.RECTANGLE_PLOT_STEER_LIMIT)
    p.add_argument("--zero-initial-guess", action="store_true")
    args = p.parse_args()

    training_csv = Path(args.training_csv)
    outdir = Path(args.outdir) if args.outdir else training_csv.parent / "figure"
    outdir.mkdir(parents=True, exist_ok=True)
    info = read_training(training_csv)
    training_plot = plot_training(training_csv, outdir)
    args.solver = build_alm_solver(args)[0]
    publication_plot = plot_publication_1x3(training_csv, outdir, info, args)

    teacher_case = build_rectangle_imitation_case(info["teacher_margins"], steer_limit=args.steer_limit)
    teacher_x0 = np.zeros_like(teacher_case.x0) if args.zero_initial_guess else teacher_case.x0.copy()
    teacher_result = solve_alm_once(args.solver, teacher_case, teacher_x0, False, args)
    teacher_u = np.asarray(teacher_result["solution"], dtype=float)
    teacher_traj = rollout(teacher_u, teacher_case.variable[:6], teacher_case.dt)

    initial_case, initial_u, initial_traj = solve_plan(info["initial_margins"], teacher_u, args)
    final_case, final_u, final_traj = solve_plan(info["final_margins"], teacher_u, args)
    final_loss_recomputed = 0.5 * float(np.sum((final_u - teacher_u) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    draw_rect(ax, rect_bounds(info["teacher_margins"]), "teacher obstacle", "tab:blue", "-")
    draw_rect(ax, rect_bounds(info["initial_margins"]), "initial obstacle", "tab:orange", "--")
    draw_rect(ax, rect_bounds(info["final_margins"]), "trained obstacle", "tab:green", ":")
    ax.plot(teacher_traj[:, 0], teacher_traj[:, 1], "o-", color="tab:blue", markersize=3, label="teacher")
    ax.plot(initial_traj[:, 0], initial_traj[:, 1], "s--", color="tab:orange", markersize=3, label="student init")
    ax.plot(final_traj[:, 0], final_traj[:, 1], "^-.", color="tab:green", markersize=3, label="student final")
    ax.scatter([teacher_case.variable[0], teacher_case.variable[3]], [teacher_case.variable[1], teacher_case.variable[4]], c=["black", "green"], zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"loss {info['initial_loss']:.2e} -> {final_loss_recomputed:.2e}")

    ax = axes[1]
    k = np.arange(teacher_case.horizon)
    ax.plot(k, teacher_u[1::2], color="tab:blue", label="teacher steer")
    ax.plot(k, initial_u[1::2], "--", color="tab:orange", label="init steer")
    ax.plot(k, final_u[1::2], "-.", color="tab:green", label="final steer")
    ax.set_xlabel("stage")
    ax.set_ylabel("steer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Steering profile")

    fig_path = outdir / "teacher_initial_final.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "training_csv": str(training_csv),
                "initial_loss": info["initial_loss"],
                "final_loss": info["final_loss"],
                "final_loss_recomputed": final_loss_recomputed,
                "teacher_margins": info["teacher_margins"].tolist(),
                "initial_margins": info["initial_margins"].tolist(),
                "final_margins": info["final_margins"].tolist(),
                "loss_margins_plot": str(training_plot),
                "publication_plot": str(publication_plot),
                "trajectory_plot": str(fig_path),
            },
            f,
            indent=2,
        )
    print(training_plot)
    print(publication_plot)
    print(fig_path)
    print(summary_path)


if __name__ == "__main__":
    main()
