"""Create the appendix figure for the smoothed rectangle MPCC diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
RESULT_DIR = EXP_DIR / "results" / "rectangle" / "smoothed_mpcc_full_rollout"
LAPANDA_PATH = RESULT_DIR / "lapanda_smooth_rollout_30.json"
ACADOS_PATH = RESULT_DIR / "acados_smooth_rollout_to_target.json"

_spec = importlib.util.spec_from_file_location("smoothed_trajectory_plot", THIS_DIR / "plot_smoothed_mpcc_trajectories.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load plot_smoothed_mpcc_trajectories.py")
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)

BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#00A83B"
GRAY = "#707070"
GRID = "#D8DCE3"
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"
INK = "#1F2430"
ICLR_TEXT_WIDTH_IN = 6.75
PANEL_BOX_ASPECT = 0.46
TICK_FONTSIZE = 8.0
LABEL_FONTSIZE = 8.0
LEGEND_FONTSIZE = 6.2
PANEL_LABEL_FONTSIZE = 11.0


def load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_times(record: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = record["rows"]
    forward_key = "forward_ms"
    backward_key = "backward_ms"
    return (
        np.asarray([row[forward_key] for row in rows], dtype=float),
        np.asarray([row[backward_key] for row in rows], dtype=float),
    )


def smooth_vertices() -> np.ndarray:
    gx = np.linspace(-0.62, 0.62, 600)
    gy = np.linspace(-0.38, 0.60, 500)
    xx, yy = np.meshgrid(gx, gy)
    values = geometry.smooth_constraint(xx, yy)
    temporary_figure, temporary_axis = plt.subplots()
    contour = temporary_axis.contour(xx, yy, values, levels=[0.0])
    vertices = np.asarray(contour.get_paths()[0].vertices, dtype=float)
    plt.close(temporary_figure)
    return vertices


def style_axis(axis) -> None:
    axis.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        width=0.68,
        length=2.4,
        pad=1.3,
    )
    axis.xaxis.labelpad = 1.8
    axis.yaxis.labelpad = 1.8
    axis.grid(True, color="#E6E8F0", linewidth=0.42, alpha=0.70)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#20232A")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    lapanda = load_record(LAPANDA_PATH)
    acados = load_record(ACADOS_PATH)
    lapanda_states = np.asarray(lapanda["states"], dtype=float)
    acados_states = np.asarray(acados["states"], dtype=float)
    lapanda_forward, lapanda_backward = extract_times(lapanda)
    acados_forward, acados_backward = extract_times(acados)

    lapanda_total = lapanda_forward + lapanda_backward
    acados_total = acados_forward + acados_backward

    figure, (trajectory_axis, time_axis) = plt.subplots(
        1,
        2,
        figsize=(ICLR_TEXT_WIDTH_IN, 2.18),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    vertices = smooth_vertices()
    trajectory_axis.fill(
        vertices[:, 0],
        vertices[:, 1],
        facecolor="#FBE8E8",
        edgecolor=GREEN,
        linewidth=1.35,
        alpha=0.64,
        zorder=0,
    )
    xmin, xmax, ymin, ymax = geometry.safety_bounds()
    trajectory_axis.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=GRAY,
            linewidth=0.95,
            linestyle=(0, (3.2, 2.2)),
            zorder=1,
        )
    )
    bxmin, bxmax, bymin, bymax = geometry.BASE_RECTANGLE
    bymin += 0.10
    bymax += 0.10
    trajectory_axis.add_patch(
        Rectangle(
            (bxmin, bymin),
            bxmax - bxmin,
            bymax - bymin,
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            hatch="////",
            zorder=2,
        )
    )

    trajectory_axis.plot(
        lapanda_states[:, 0],
        lapanda_states[:, 1],
        color=BLUE,
        linewidth=1.7,
        marker="o",
        markersize=2.9,
        markevery=3,
        zorder=7,
    )
    trajectory_axis.plot(
        acados_states[:, 0],
        acados_states[:, 1],
        color=RED,
        linewidth=1.35,
        linestyle=(0, (4.5, 2.0)),
        marker="s",
        markersize=2.9,
        markevery=(1, 3),
        zorder=8,
    )
    trajectory_axis.scatter([-1.2], [0.0], marker="^", s=23, color=GREEN, zorder=9)
    trajectory_axis.scatter([1.2], [0.0], marker="*", s=34, color=GREEN, zorder=9)
    trajectory_axis.set_xlim(-1.30, 1.30)
    trajectory_axis.set_ylim(-0.42, 0.95)
    trajectory_axis.set_box_aspect(PANEL_BOX_ASPECT)
    trajectory_axis.set_xlabel(r"$p_x$", fontsize=LABEL_FONTSIZE)
    trajectory_axis.set_ylabel(r"$p_y$", fontsize=LABEL_FONTSIZE)
    style_axis(trajectory_axis)
    trajectory_axis.legend(
        handles=[
            Patch(facecolor="#FBE8E8", edgecolor=GREEN, label="Smoothed obstacle"),
            Line2D([0], [0], color=BLUE, linewidth=1.7, marker="o", markersize=3.2, label="lapanda"),
            Line2D([0], [0], color=RED, linewidth=1.35, linestyle=(0, (4.5, 2.0)), marker="s", markersize=3.2, label="acados"),
        ],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=0.98,
        borderpad=0.26,
        handlelength=1.55,
        handletextpad=0.42,
        labelspacing=0.18,
    )

    lapanda_steps = np.arange(1, len(lapanda_total) + 1)
    acados_steps = np.arange(1, len(acados_total) + 1)
    time_axis.plot(
        lapanda_steps,
        lapanda_total,
        color=BLUE,
        linewidth=1.35,
        marker="o",
        markersize=3.0,
        label="lapanda",
        zorder=4,
    )
    time_axis.plot(
        acados_steps,
        acados_total,
        color=RED,
        linewidth=1.2,
        linestyle="--",
        marker="s",
        markersize=2.8,
        label="acados",
        zorder=3,
    )
    time_axis.set_yscale("log")
    max_step = max(len(lapanda_total), len(acados_total))
    time_axis.set_xlim(0.6, max_step + 0.4)
    positive_times = np.r_[lapanda_total[lapanda_total > 0], acados_total[acados_total > 0]]
    time_axis.set_ylim(max(0.35, float(np.min(positive_times)) * 0.72), float(np.max(positive_times)) * 7.0)
    time_axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
    time_axis.set_xlabel("MPC step", fontsize=LABEL_FONTSIZE)
    time_axis.set_ylabel("Total time (ms)", fontsize=LABEL_FONTSIZE)
    time_axis.set_box_aspect(PANEL_BOX_ASPECT)
    style_axis(time_axis)
    time_axis.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=0.98,
        borderpad=0.26,
        handlelength=1.55,
        handletextpad=0.42,
        labelspacing=0.18,
    )

    for axis, label in zip(
        (trajectory_axis, time_axis),
        ("(a) Closed-loop trajectory", "(b) Per-step computation time"),
    ):
        axis.text(
            0.5,
            -0.30,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontfamily="Times New Roman",
            color=INK,
        )
    figure.subplots_adjust(left=0.070, right=0.994, top=0.955, bottom=0.315, wspace=0.24)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "smoothed_mpcc_rollout_and_step_time"
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
