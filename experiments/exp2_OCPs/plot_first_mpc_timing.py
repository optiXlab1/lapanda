"""Plot the paper first-rollout MPC timing comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

import plot_closed_loop as base


OUT_DIR = base.OUT_DIR

TICK_FONTSIZE = 8.0
LABEL_FONTSIZE = 8.0
LEGEND_FONTSIZE = 8.0
PANEL_LABEL_FONTSIZE = 11.0
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"
LEGEND_FRAME_ALPHA = 0.96


METHODS = [
    {
        "label": "lapanda",
        "color": base.COLORS["alm"],
        "linestyle": "-",
        "linewidth": 1.22,
        "marker": "o",
        "markevery": (0, 8),
        "path_kind": "alm",
    },
    {
        "label": "SafePDP",
        "color": base.COLORS["safepdp_coc"],
        "linestyle": "-",
        "linewidth": 1.22,
        "marker": "s",
        "markevery": (3, 8),
        "path_kind": "safepdp",
    },
    {
        "label": "TurboMPC-CPU",
        "color": base.COLORS["turbo_cpu"],
        "linestyle": "-",
        "linewidth": 1.22,
        "marker": "^",
        "markevery": (6, 8),
        "path_kind": "turbo_cpu_warm",
    },
]


def smooth_log_series(y: np.ndarray, window: int = 5) -> np.ndarray:
    if y.size < 3:
        return y
    window = max(3, min(window, y.size if y.size % 2 == 1 else y.size - 1))
    half = window // 2
    log_y = np.log10(np.maximum(y, 1e-12))
    padded = np.pad(log_y, (half, half), mode="edge")
    med = np.asarray([np.median(padded[i : i + window]) for i in range(y.size)])
    return 10.0**med


def method_path(exp: dict, kind: str) -> Path:
    if kind == "alm":
        return exp["exp_dir"] / exp["alm_mpc"]
    if kind == "safepdp":
        return base.safepdp_mpc_path(exp)
    if kind == "turbo_cpu_warm":
        epoch500_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_mpc_teacher_cpu_warm_epoch500theta.csv"
        if epoch500_path.exists():
            return epoch500_path
        return exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_mpc_teacher_cpu_warm.csv"
    raise ValueError(kind)


def plot_panel(ax, exp: dict, metric: str) -> None:
    positive = []
    for method in METHODS:
        rows = base.read_csv(method_path(exp, method["path_kind"]))
        if not rows:
            continue
        x = np.asarray([base.to_int(row.get("step")) for row in rows], dtype=float)
        y = np.asarray([base.to_float(row.get(metric)) for row in rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if not np.any(mask):
            continue
        x_plot = x[mask]
        y_plot = y[mask]
        if method["path_kind"] == "turbo_cpu_warm":
            y_plot = smooth_log_series(y_plot, window=5)
        positive.extend(y_plot.tolist())
        ax.plot(
            x_plot,
            y_plot,
            color=method["color"],
            linestyle=method["linestyle"],
            linewidth=method["linewidth"],
            marker=method["marker"],
            markersize=3.35,
            markevery=method["markevery"],
            markerfacecolor=method["color"],
            markeredgecolor="white",
            markeredgewidth=0.28,
            solid_capstyle="round",
            label=method["label"],
        )

    ax.set_yscale("log")
    if positive:
        ax.set_ylim(max(min(positive) * 0.55, 2e-5), max(positive) * 1.75)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Time instant", fontsize=LABEL_FONTSIZE, labelpad=1.8)
    style_benchmark_axis(ax)


def style_benchmark_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=50))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#E6E8F0", linestyle="-", linewidth=0.42, alpha=0.70)
    ax.grid(True, which="minor", axis="y", color="#E6E8F0", linestyle="-", linewidth=0.28, alpha=0.32)
    for spine in ax.spines.values():
        spine.set_color("#1F1F1F")
        spine.set_linewidth(0.72)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=2.4, width=0.68, pad=1.3, color="#30343B")
    ax.tick_params(axis="both", which="minor", length=1.3, width=0.45, color="#7A7F89")
    ax.xaxis.label.set_size(LABEL_FONTSIZE)
    ax.yaxis.label.set_size(LABEL_FONTSIZE)
    ax.xaxis.labelpad = 1.8
    ax.yaxis.labelpad = 1.8


def main() -> None:
    base.use_chart_theme()
    fig, axes = plt.subplots(2, 3, figsize=(base.ICLR_TEXT_WIDTH_IN, 2.92), dpi=300, sharex="col")

    for col, exp in enumerate(base.EXPERIMENTS):
        plot_panel(axes[0, col], exp, "forward_time_sec")
        plot_panel(axes[1, col], exp, "backward_time_sec")
        axes[0, col].set_xlabel("")
        axes[0, col].tick_params(labelbottom=False)
        if col == 0:
            axes[0, col].set_ylabel("Fwd. Time (s)", fontsize=LABEL_FONTSIZE, labelpad=1.8)
            axes[1, col].set_ylabel("Bwd. Time (s)", fontsize=LABEL_FONTSIZE, labelpad=1.8)
        else:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")

    handles = [
        Line2D(
            [0],
            [0],
            color=method["color"],
            linestyle=method["linestyle"],
            linewidth=method["linewidth"],
            marker=method["marker"],
            markersize=3.75,
            markerfacecolor=method["color"],
            markeredgecolor="white",
            markeredgewidth=0.28,
            label=method["label"],
        )
        for method in METHODS
        if any(base.read_csv(method_path(exp, method["path_kind"])) for exp in base.EXPERIMENTS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=len(handles),
        frameon=True,
        fancybox=True,
        framealpha=LEGEND_FRAME_ALPHA,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.9,
        columnspacing=0.90,
        borderpad=0.30,
        labelspacing=0.22,
        handletextpad=0.48,
    )
    legend = fig.legends[0]
    legend.get_frame().set_linewidth(0.50)
    legend.get_frame().set_facecolor(LEGEND_FACE)
    legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)

    fig.subplots_adjust(left=0.073, right=0.992, bottom=0.255, top=0.845, wspace=0.24, hspace=0.34)
    panel_labels = ["(a) CartPole", "(b) Quadrotor", "(c) Robot arm"]
    for label, ax in zip(panel_labels, axes[1]):
        box = ax.get_position()
        fig.text(
            0.5 * (box.x0 + box.x1),
            box.y0 - 0.130,
            label,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontfamily="Times New Roman",
            color=base.TOKENS["ink"],
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "ocp_first_mpc_timing"
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
