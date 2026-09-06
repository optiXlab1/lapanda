"""Reproduce the nonlinear horizon-scaling figure from its summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "exp2_OCPs" / "summary_results"
DEFAULT_DATA = RESULTS_DIR / "nonlinear_horizon_scaling_lapanda_turbompc_gpu.csv"
OUTPUT_STEM = "nonlinear_horizon_scaling_lapanda_turbompc_gpu"
HORIZONS = tuple(range(10, 201, 20))

RED = "#D62728"
PURPLE = "#9467BD"
GRID = "#C6C6C6"
PANEL_FACE = "#F3F3F3"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#D3D6DE"

ICLR_TEXT_WIDTH_IN = 5.5
TICK_FONTSIZE = 7.0
LABEL_FONTSIZE = 7.0
LEGEND_FONTSIZE = 7.0

METHODS = (
    ("lapanda", RED, "o", "-", 1.20),
    ("TurboMPC-GPU", PURPLE, "s", "-", 1.00),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    result = {}
    for method, _color, _marker, _linestyle, _linewidth in METHODS:
        selected = {
            int(row["horizon"]): float(row["total_time_ms"])
            for row in rows
            if row["method"] == method and int(row["horizon"]) in HORIZONS
        }
        missing = sorted(set(HORIZONS) - set(selected))
        if missing:
            raise ValueError(f"{path} is missing {method} horizons: {missing}")
        result[method] = (
            np.asarray(HORIZONS, dtype=float),
            np.asarray([selected[horizon] for horizon in HORIZONS], dtype=float),
        )
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_FACE)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, alpha=0.42, linewidth=0.45, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        top=False,
        right=False,
        bottom=True,
        left=True,
        direction="out",
        length=1.8,
        width=0.60,
        color="#20232A",
        pad=1.5,
    )
    ax.tick_params(axis="y", which="minor", top=False, right=False, length=1.0, width=0.40, color="#7A7F89")
    for spine in ax.spines.values():
        spine.set_color("#25282D")
        spine.set_linewidth(0.72)


def main() -> None:
    args = parse_args()
    data = load_summary(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": TICK_FONTSIZE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(ICLR_TEXT_WIDTH_IN, 1.95), dpi=300)
    for method, color, marker, linestyle, linewidth in METHODS:
        horizons, total_time = data[method]
        ax.plot(
            horizons,
            total_time,
            color=color,
            label=method,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=3.0,
            zorder=4,
        )

    style_axis(ax)
    ax.set_xticks(range(10, 191, 20))
    ax.set_xlim(10, 190)
    ax.set_xlabel("Horizon $N$", fontsize=LABEL_FONTSIZE, labelpad=5.0)
    ax.set_ylabel("Total time (ms)", fontsize=LABEL_FONTSIZE, labelpad=1.5)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.018, 0.982),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=0.96,
        borderpad=0.30,
        handlelength=1.65,
        handletextpad=0.48,
        columnspacing=1.00,
    )
    legend.get_frame().set_linewidth(0.75)
    legend.get_frame().set_facecolor(LEGEND_FACE)
    legend.get_frame().set_alpha(0.96)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.955, bottom=0.305)

    stem = args.outdir / OUTPUT_STEM
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
