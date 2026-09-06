"""Plot cold-start circle obstacle solutions."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as exp3_config

_circle_spec = importlib.util.spec_from_file_location(
    "circle_lapanda_impl", Path(__file__).resolve().parent / "run_lapanda.py"
)
if _circle_spec is None or _circle_spec.loader is None:
    raise RuntimeError("could not load circle/run_lapanda.py")
_circle_impl = importlib.util.module_from_spec(_circle_spec)
sys.modules[_circle_spec.name] = _circle_impl
_circle_spec.loader.exec_module(_circle_impl)

build_circle_case = _circle_impl.build_circle_case
metrics = _circle_impl.metrics
rollout = _circle_impl.rollout


BLUE = "#1F77B4"
RED = "#D62728"
ORANGE = "#B37A2C"
GREEN = "#00A83B"
SAFE_FILL = "#00FF00"
VIOLATION_FILL = RED
GRID = "#D7D9E0"
INK = "#1F2430"
PANEL_FACE = "#F3F3F3"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#D3D6DE"
LEGEND_FRAME_ALPHA = 0.96
ICLR_TEXT_WIDTH_IN = 5.5
PANEL_BOX_ASPECT = 1.2 / 2.6
TICK_FONTSIZE = 7.0
LABEL_FONTSIZE = 7.0
LEGEND_FONTSIZE = 7.0
PANEL_LABEL_FONTSIZE = 8.0


def load_u(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").reshape(-1)


def obstacle_clearance(case, u: np.ndarray) -> np.ndarray:
    traj = connect_start(rollout(u, case.variable, case.dt), case)
    center = np.asarray(case.meta["obstacle"], dtype=float)
    radius = float(case.theta[5])
    return np.linalg.norm(traj[:, :2] - center.reshape(1, 2), axis=1) - radius


def connect_start(traj: np.ndarray, case) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.size == 0:
        return traj
    start = np.asarray(case.variable[: traj.shape[1]], dtype=float)
    if np.linalg.norm(traj[0, :2] - start[:2]) < 1e-10:
        return traj
    return np.vstack([start, traj])


def connect_goal_for_plot(traj: np.ndarray, case) -> np.ndarray:
    """Connect the displayed trajectory to the target without changing solver data."""
    traj = np.asarray(traj, dtype=float)
    if traj.size == 0:
        return traj
    goal = np.asarray(case.variable[3:6], dtype=float)
    if np.linalg.norm(traj[-1, :2] - goal[:2]) < 1e-10:
        return traj
    return np.vstack([traj, goal])


def style_axis(ax):
    ax.set_facecolor(PANEL_FACE)
    ax.grid(True, color="#C6C6C6", linestyle="-", linewidth=0.45, alpha=0.42)
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        top=False,
        right=False,
        bottom=True,
        left=True,
        direction="out",
        length=1.8,
        width=0.60,
        color="#20232A",
        pad=1.3,
    )
    ax.xaxis.labelpad = 1.8
    ax.yaxis.labelpad = 1.8
    for spine in ax.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#20232A")


def save_paper_style_figure(outdir: Path, case, series: list[tuple[str, np.ndarray, str, str]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ICLR_TEXT_WIDTH_IN, 2.5),
        dpi=300,
        gridspec_kw={"width_ratios": [0.65, 0.35]},
    )
    ax = axes[0]
    center = np.asarray(case.meta["obstacle"], dtype=float)
    radius = float(case.theta[5])
    obstacle = plt.Circle(
        tuple(center),
        radius,
        facecolor="#FFFFFF",
        edgecolor="none",
        alpha=1.0,
        label="_nolegend_",
        zorder=4,
    )
    ax.add_patch(obstacle)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color=GREEN,
        lw=1.20,
        label="_nolegend_",
        zorder=5,
    )
    ax.scatter(
        [center[0]],
        [center[1]],
        marker="+",
        color=GREEN,
        s=25,
        linewidths=1.20,
        label="_nolegend_",
        zorder=6,
    )
    for label, u, color, style in series:
        traj = connect_start(rollout(u, case.variable, case.dt), case)
        traj = connect_goal_for_plot(traj, case)
        zorder = 15 if style != "-" else 12
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linestyle=style,
            lw=1.50 if label == "lapanda" else 1.00,
            label=label,
            zorder=zorder,
        )
    ax.scatter([case.variable[0]], [case.variable[1]], color="black", marker="^", s=18, label="_nolegend_", zorder=16)
    ax.scatter([case.variable[3]], [case.variable[4]], color="black", marker="*", s=25, label="_nolegend_", zorder=17)
    ax.set_xlabel("x position", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("y position", fontsize=LABEL_FONTSIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-0.20, 0.60)
    ax.set_box_aspect(0.80 / 2.80)
    style_axis(ax)
    ax.xaxis.set_major_locator(FixedLocator([-1.40, -0.70, 0.00, 0.70, 1.40]))
    ax.yaxis.set_major_locator(FixedLocator([-0.20, 0.00, 0.20, 0.40, 0.60]))
    legend = ax.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=LEGEND_FRAME_ALPHA,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        borderpad=0.30,
        labelspacing=0.22,
        handlelength=1.70,
        handletextpad=0.48,
    )
    legend.get_frame().set_linewidth(0.50)
    legend.get_frame().set_facecolor(LEGEND_FACE)
    legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    ax.text(0.5, -0.36, "(a) Trajectory", transform=ax.transAxes, ha="center", va="top", fontsize=PANEL_LABEL_FONTSIZE, fontfamily="Times New Roman", color=INK)

    ax = axes[1]
    clearance_values = [obstacle_clearance(case, u) for _, u, _, _ in series]
    max_len = max((values.size for values in clearance_values), default=case.horizon)
    k = np.arange(max_len)
    ymax = 1.25
    ymin = -0.25
    ax.axhspan(0.0, ymax, color=SAFE_FILL, alpha=0.035, label="Safe region", zorder=0)
    ax.axhspan(ymin, 0.0, color=VIOLATION_FILL, alpha=0.050, label="Violation region", zorder=0)
    ax.axhline(0.0, color="#7A7A7A", linestyle=(0, (3.0, 2.0)), lw=0.65, label="_nolegend_", zorder=2)
    for label, u, color, style in series:
        clearance = obstacle_clearance(case, u)
        zorder = 15 if style != "-" else 12
        ax.plot(
            np.arange(clearance.size),
            clearance,
            color=color,
            linestyle=style,
            lw=1.50 if label == "lapanda" else 1.00,
            label=label,
            zorder=zorder,
        )
    ax.set_xlabel("Horizon step", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Clearance", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(0, 12)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect((0.65 / 0.35) * (0.80 / 2.80))
    style_axis(ax)
    ax.xaxis.set_major_locator(FixedLocator([0, 3, 6, 9, 12]))
    ax.yaxis.set_major_locator(FixedLocator([-0.25, 0.25, 0.75, 1.25]))
    legend = ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        framealpha=LEGEND_FRAME_ALPHA,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        borderpad=0.30,
        labelspacing=0.22,
        columnspacing=0.85,
        handlelength=1.45,
        handletextpad=0.48,
    )
    legend.get_frame().set_linewidth(0.50)
    legend.get_frame().set_facecolor(LEGEND_FACE)
    legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    ax.text(0.5, -0.36, "(b) Obstacle clearance", transform=ax.transAxes, ha="center", va="top", fontsize=PANEL_LABEL_FONTSIZE, fontfamily="Times New Roman", color=INK)

    fig.subplots_adjust(left=0.070, right=0.994, top=0.955, bottom=0.350, wspace=0.18)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"circle_obstacle_solution.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)


def main():
    exp = EXP_DIR
    outdir = exp / "results" / "circle" / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    radius = float(sys.argv[1]) if len(sys.argv) > 1 else exp3_config.CIRCLE_RADIUS
    center_y = float(sys.argv[2]) if len(sys.argv) > 2 else exp3_config.CIRCLE_CENTER_Y
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else exp3_config.CIRCLE_HORIZON
    speed_limit = float(sys.argv[4]) if len(sys.argv) > 4 else exp3_config.CIRCLE_SPEED_LIMIT
    alm_dir = sys.argv[5] if len(sys.argv) > 5 else str(exp3_config.CIRCLE_RESULT_DIR.relative_to(exp))
    acados_dir = sys.argv[6] if len(sys.argv) > 6 else str(exp3_config.CIRCLE_ACADOS_RESULT_DIR.relative_to(exp))
    label_suffix = f"r={radius:g}"

    case = build_circle_case(horizon, speed_limit)
    case.theta[4] = 10.0
    case.theta[5] = radius
    case.theta[6] = center_y
    case.meta["safe_radius"] = radius
    case.meta["obstacle"] = [0.0, center_y]
    initial_u = case.x0
    alm_u = load_u(exp / alm_dir / "circle_solution.csv")

    series = [
        ("initial", initial_u, "#777777", "--"),
        ("lapanda", alm_u, RED, "-"),
    ]
    acados_solution = exp / acados_dir / "acados_circle_solution.csv"
    if acados_solution.exists():
        acados_u = load_u(acados_solution)
        acados_label = "acados"
        result_json = exp / acados_dir / "acados_circle_result.json"
        if result_json.exists():
            with result_json.open("r", encoding="utf-8") as f:
                json.load(f)
        series.append((acados_label, acados_u, BLUE, "--"))

    save_paper_style_figure(outdir, case, series[1:])

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    center = np.asarray(case.meta["obstacle"], dtype=float)
    circle = plt.Circle(tuple(center), case.theta[5], color="black", alpha=0.14)
    ax.add_patch(circle)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(center[0] + case.theta[5] * np.cos(theta), center[1] + case.theta[5] * np.sin(theta), color="black", lw=1.2)
    for label, u, color, style in series:
        traj = rollout(u, case.variable, case.dt)
        ax.plot(traj[:, 0], traj[:, 1], style, color=color, lw=2.0, label=label)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=18)
    ax.scatter([case.variable[0]], [case.variable[1]], c="black", marker="o", s=28, label="start")
    ax.scatter([case.variable[3]], [case.variable[4]], c="black", marker="x", s=42, label="target")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.set_title(f"Circle obstacle cold-start trajectories ({label_suffix})")
    fig.tight_layout()
    fig.savefig(outdir / f"circle_coldstart_trajectories_r{str(radius).replace('.', 'p')}.png", dpi=180)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)
    k = np.arange(case.horizon)
    for label, u, color, style in series:
        axes[0].plot(k, u[0::2], style, color=color, lw=1.8, label=label)
        axes[1].plot(k, u[1::2], style, color=color, lw=1.8)
    axes[0].set_ylabel("speed")
    axes[1].set_ylabel("steer")
    axes[1].set_xlabel("stage")
    axes[1].axhline(exp3_config.CIRCLE_PLOT_STEER_LIMIT, color="black", linestyle=":", linewidth=1.0, alpha=0.55)
    axes[1].axhline(-exp3_config.CIRCLE_PLOT_STEER_LIMIT, color="black", linestyle=":", linewidth=1.0, alpha=0.55)
    axes[1].set_ylim(-1.08 * exp3_config.CIRCLE_PLOT_STEER_LIMIT, 1.08 * exp3_config.CIRCLE_PLOT_STEER_LIMIT)
    for ax in axes:
        ax.grid(True, alpha=0.35)
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / f"circle_coldstart_controls_r{str(radius).replace('.', 'p')}.png", dpi=180)

    summary = {}
    for label, u, _, _ in series:
        summary[label] = metrics(case, u)
    with (outdir / "circle_coldstart_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(outdir / f"circle_coldstart_trajectories_r{str(radius).replace('.', 'p')}.png")
    print(outdir / f"circle_coldstart_controls_r{str(radius).replace('.', 'p')}.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
