"""Plot the shared-point Rosenbrock backward comparison."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from .common import RESULTS, RUNS


METHODS = [
    ("cg", "lapanda-ALM", "#D62728", "-", "o"),
    ("minres_fixed", "lapanda-KKT", "#555555", "-", "s"),
    ("minres_matched", "lapanda-KKT (matched)", "#161616", "--", "o"),
    ("sparse_lu", "Sparse LU", "#1F77B4", "--", "s"),
    ("dense_ldlt", "Dense LDLT", "#9467BD", ":", "^"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS / "rosenbrock/main/summary.csv")
    parser.add_argument("--output", type=Path, default=RUNS / "figures")
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))
    metrics = [
        ("median_wall_us", "Backward time (µs)", "(a) Time"),
        ("median_reverse_peak_KiB", "Peak backward allocation (KiB)", "(b) Memory"),
        ("median_gradient_error_percent", "Relative gradient error (%)", "(c) Accuracy"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for method, label, color, linestyle, marker in METHODS:
            rows = data[data.method.eq(method)].sort_values("n")
            ax.plot(rows.n, rows[metric], label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=4.8, markerfacecolor="white",
                    markeredgewidth=.9, linewidth=1.35,
                    zorder=10 if method == "minres_matched" else 5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 200, 500, 1000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Problem dimension n")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, pad=7)
        ax.set_facecolor("#F3F3F3")
        ax.grid(True, which="major", color="#C6C6C6", linewidth=.5, alpha=.45)
        ax.tick_params(direction="out", length=3, width=.7)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
            spine.set_color("#252525")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, 1.015),
               ncol=5, frameon=False, fontsize=8.5, handlelength=2.3, columnspacing=1.25)
    fig.subplots_adjust(left=.067, right=.991, bottom=.19, top=.79, wspace=.40)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"rosenbrock_backward.{suffix}", dpi=180,
                    bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


if __name__ == "__main__":
    main()
