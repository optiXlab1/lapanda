"""Create the appendix figure for the smoothed rectangle MPCC diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
RESULT_DIR = EXP_DIR / "results" / "rectangle" / "comparison"
LAPANDA_PATH = RESULT_DIR / "lapanda_smooth_rollout_30.json"
ACADOS_PATH = RESULT_DIR / "acados_smooth_rollout_to_target.json"

BASE_RECTANGLE = np.array([-0.35, 0.35, -0.22, 0.22], dtype=float)
MARGINS = np.array([0.12, 0.12, 0.01, 0.22], dtype=float)
TAU = 0.05

BLUE = "#1F77B4"
RED = "#D62728"
GREEN = "#2CA02C"
LIGHT_GREEN = "#7FBF7B"
SECONDARY_DASH = (0, (3.2, 4.8))
GRAY = "#707070"
GRID = "#D8DCE3"
PANEL_FACE = "#F3F3F3"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#D3D6DE"
LEGEND_FRAME_ALPHA = 0.96
INK = "#1F2430"
ICLR_TEXT_WIDTH_IN = 5.5
PANEL_BOX_ASPECT = 0.46
TICK_FONTSIZE = 7.0
LABEL_FONTSIZE = 7.0
LEGEND_FONTSIZE = 7.0
PANEL_LABEL_FONTSIZE = 8.0


def safety_bounds() -> tuple[float, float, float, float]:
    return (
        BASE_RECTANGLE[0] - MARGINS[0],
        BASE_RECTANGLE[1] + MARGINS[1],
        BASE_RECTANGLE[2] - MARGINS[2],
        BASE_RECTANGLE[3] + MARGINS[3],
    )


def smooth_constraint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = safety_bounds()
    distances = np.stack((xmin - x, x - xmax, ymin - y, y - ymax), axis=0)
    scaled = distances / TAU
    offset = np.max(scaled, axis=0, keepdims=True)
    logsumexp = np.squeeze(offset, axis=0) + np.log(
        np.sum(np.exp(scaled - offset), axis=0)
    )
    return TAU * logsumexp - TAU * np.log(4.0)


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
    values = smooth_constraint(xx, yy)
    temporary_figure, temporary_axis = plt.subplots()
    contour = temporary_axis.contour(xx, yy, values, levels=[0.0])
    vertices = np.asarray(contour.get_paths()[0].vertices, dtype=float)
    plt.close(temporary_figure)
    return vertices


def style_axis(axis) -> None:
    axis.set_facecolor(PANEL_FACE)
    axis.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        top=False,
        right=False,
        bottom=True,
        left=True,
        direction="out",
        width=0.60,
        length=1.8,
        color="#20232A",
        pad=1.3,
    )
    axis.xaxis.labelpad = 1.8
    axis.yaxis.labelpad = 1.8
    axis.grid(True, color="#C6C6C6", linewidth=0.45, alpha=0.42)
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
        figsize=(ICLR_TEXT_WIDTH_IN, 2.5),
        gridspec_kw={"width_ratios": [0.65, 0.35]},
    )

    vertices = smooth_vertices()
    trajectory_axis.fill(
        vertices[:, 0],
        vertices[:, 1],
        facecolor="none",
        edgecolor=LIGHT_GREEN,
        linewidth=0.85,
        alpha=1.0,
        zorder=1,
    )
    xmin, xmax, ymin, ymax = safety_bounds()
    trajectory_axis.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=GREEN,
            linewidth=0.85,
            linestyle=(0, (3.2, 2.2)),
            zorder=2,
        )
    )
    bxmin, bxmax, bymin, bymax = BASE_RECTANGLE
    bymin += 0.10
    bymax += 0.10
    trajectory_axis.add_patch(
        Rectangle(
            (bxmin, bymin),
            bxmax - bxmin,
            bymax - bymin,
            facecolor="#FFFFFF",
            edgecolor=GREEN,
            linewidth=1.80,
            zorder=3,
        )
    )
    trajectory_axis.scatter(
        [0.5 * (bxmin + bxmax)],
        [0.5 * (bymin + bymax)],
        marker="+",
        color=GREEN,
        s=15,
        linewidths=1.80,
        zorder=4,
    )

    trajectory_axis.plot(
        lapanda_states[:, 0],
        lapanda_states[:, 1],
        color=RED,
        linewidth=0.85,
        zorder=7,
    )
    trajectory_axis.plot(
        acados_states[:, 0],
        acados_states[:, 1],
        color=BLUE,
        linewidth=0.85,
        linestyle=SECONDARY_DASH,
        dash_capstyle="round",
        zorder=8,
    )
    trajectory_axis.scatter([-1.2], [0.0], marker="^", s=23, color="black", zorder=9)
    trajectory_axis.scatter([1.2], [0.0], marker="*", s=34, color="black", zorder=9)
    trajectory_axis.set_xlim(-1.40, 1.40)
    trajectory_axis.set_xticks([-1.40, -0.70, 0.00, 0.70, 1.40])
    trajectory_axis.set_ylim(-0.40, 0.60)
    trajectory_axis.set_yticks([-0.40, -0.20, 0.00, 0.20, 0.40, 0.60])
    trajectory_axis.set_box_aspect(1.00 / 2.80)
    trajectory_axis.set_xlabel(r"$p_x$", fontsize=LABEL_FONTSIZE)
    trajectory_axis.set_ylabel(r"$p_y$", fontsize=LABEL_FONTSIZE)
    style_axis(trajectory_axis)
    trajectory_legend = trajectory_axis.legend(
        handles=[
            Line2D([0], [0], color=RED, linewidth=0.85, label="lapanda"),
            Line2D([0], [0], color=BLUE, linewidth=0.85, linestyle=SECONDARY_DASH, dash_capstyle="round", label="acados"),
        ],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=LEGEND_FRAME_ALPHA,
        borderpad=0.30,
        handlelength=1.70,
        handletextpad=0.48,
        labelspacing=0.22,
    )
    trajectory_legend.get_frame().set_linewidth(0.50)

    lapanda_steps = np.arange(1, len(lapanda_total) + 1)
    acados_steps = np.arange(1, len(acados_total) + 1)
    time_axis.plot(
        lapanda_steps,
        lapanda_total,
        color=RED,
        linewidth=1.00,
        marker="o",
        markersize=2.0,
        label="lapanda",
        zorder=4,
    )
    time_axis.plot(
        acados_steps,
        acados_total,
        color=BLUE,
        linewidth=1.00,
        linestyle=SECONDARY_DASH,
        dash_capstyle="round",
        marker="s",
        markersize=2.0,
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
    time_axis.set_box_aspect((0.65 / 0.35) * (1.00 / 2.80))
    style_axis(time_axis)
    time_legend = time_axis.legend(
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=LEGEND_FRAME_ALPHA,
        borderpad=0.30,
        handlelength=1.70,
        handletextpad=0.48,
        labelspacing=0.22,
    )
    time_legend.get_frame().set_linewidth(0.50)

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
    figure.subplots_adjust(left=0.070, right=0.994, top=0.955, bottom=0.315, wspace=0.18)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "smoothed_mpcc_rollout_and_step_time"
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
