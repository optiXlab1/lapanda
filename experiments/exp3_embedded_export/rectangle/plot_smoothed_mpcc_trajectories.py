"""Plot aligned lapanda/acados rollouts for the smoothed rectangle constraint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
RESULT_DIR = (
    EXP_DIR
    / "results"
    / "rectangle"
    / "nonzero_initialization_mpc_rollout10"
)

ACADOS_PATH = RESULT_DIR / "acados_rollout10.json"
LAPANDA_PATH = RESULT_DIR / "lapanda_aligned_smooth_rollout10.json"

BASE_RECTANGLE = np.array([-0.35, 0.35, -0.22, 0.22], dtype=float)
MARGINS = np.array([0.12, 0.12, 0.01, 0.22], dtype=float)
TAU = 0.05

BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#00A83B"
GRAY = "#707070"
GRID = "#D8DCE3"
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"


def load_trajectories() -> tuple[np.ndarray, np.ndarray]:
    acados = json.loads(ACADOS_PATH.read_text(encoding="utf-8"))
    lapanda = json.loads(LAPANDA_PATH.read_text(encoding="utf-8"))

    acados_xy = [[-1.2, 0.0]]
    acados_xy.extend([row["state_x"], row["state_y"]] for row in acados["rows"])
    acados_xy = np.asarray(acados_xy, dtype=float)

    lapanda_xy = np.asarray(lapanda["states"], dtype=float)[:, :2]
    lapanda_xy = lapanda_xy[: len(acados_xy)]
    return lapanda_xy, acados_xy


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


def main() -> None:
    lapanda_xy, acados_xy = load_trajectories()
    xmin, xmax, ymin, ymax = safety_bounds()

    fig, ax = plt.subplots(figsize=(6.75, 3.45))

    gx = np.linspace(-1.32, 0.58, 700)
    gy = np.linspace(-0.47, 0.55, 520)
    xx, yy = np.meshgrid(gx, gy)
    smooth_value = smooth_constraint(xx, yy)
    contour_figure, contour_axis = plt.subplots()
    contour_set = contour_axis.contour(
        xx,
        yy,
        smooth_value,
        levels=[0.0],
    )
    smooth_vertices = np.asarray(contour_set.get_paths()[0].vertices, dtype=float)
    plt.close(contour_figure)
    ax.fill(
        smooth_vertices[:, 0],
        smooth_vertices[:, 1],
        facecolor="#FBE8E8",
        alpha=0.58,
        zorder=0,
    )
    ax.plot(
        smooth_vertices[:, 0],
        smooth_vertices[:, 1],
        color=GREEN,
        linewidth=1.65,
        zorder=3,
    )

    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=GRAY,
            linewidth=1.15,
            linestyle=(0, (3.2, 2.2)),
            label="Rectangular safety boundary",
            zorder=2,
        )
    )

    bxmin, bxmax, bymin, bymax = BASE_RECTANGLE
    bymin += 0.10
    bymax += 0.10
    ax.add_patch(
        Rectangle(
            (bxmin, bymin),
            bxmax - bxmin,
            bymax - bymin,
            facecolor="white",
            edgecolor="black",
            linewidth=1.3,
            hatch="////",
            label="Obstacle",
            zorder=4,
        )
    )

    ax.plot(
        acados_xy[:, 0],
        acados_xy[:, 1],
        color=RED,
        linewidth=1.45,
        linestyle=(0, (5.0, 2.2)),
        marker="s",
        markersize=3.8,
        markevery=(1, 2),
        label="acados",
        zorder=9,
    )
    ax.plot(
        lapanda_xy[:, 0],
        lapanda_xy[:, 1],
        color=BLUE,
        linewidth=2.05,
        marker="o",
        markersize=3.8,
        markevery=2,
        label="lapanda",
        zorder=8,
    )
    ax.scatter(
        [-1.2],
        [0.0],
        marker="^",
        s=34,
        facecolor=GREEN,
        edgecolor=GREEN,
        label="Start",
        zorder=9,
    )

    ax.set_xlim(-1.30, 0.55)
    ax.set_ylim(-0.45, 0.53)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$", fontsize=9.0)
    ax.set_ylabel(r"$p_y$", fontsize=9.0)
    ax.tick_params(axis="both", labelsize=8.0, width=0.8, length=3.5)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.78)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GREEN, linewidth=1.65))
    labels.append(r"Smoothed boundary ($\tau=0.05$)")
    order = [labels.index(name) for name in (
        "lapanda",
        "acados",
        r"Smoothed boundary ($\tau=0.05$)",
        "Rectangular safety boundary",
        "Obstacle",
        "Start",
    )]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        fontsize=7.3,
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=0.96,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.5,
        handletextpad=0.65,
        columnspacing=1.0,
    )

    fig.tight_layout(pad=0.55, rect=(0.0, 0.0, 1.0, 0.90))
    stem = RESULT_DIR / "smoothed_rectangle_lapanda_acados_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
