"""Reproduce the paper's Exp.1 nonlinear-constraint activity figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial",
        "mathtext.bf": "Arial",
    }
)


def radius_values(n: int, theta, m: int, constraint_stride: int):
    radii = []
    for i in range(0, n - 1, constraint_stride):
        phase = 2.0 * np.pi * (i + 1) / max(n, 1)
        radii.append(theta[3] + 0.1 * np.sin(phase))
    radii = np.asarray(radii, dtype=float)
    if radii.size != m:
        raise ValueError(f"expected {m} radii, got {radii.size}; check --constraint-stride")
    return radii


def pair_norm_ratio(solution, radii, constraint_stride: int):
    values = []
    for k, i in enumerate(range(0, solution.size - 1, constraint_stride)):
        values.append(np.sqrt(solution[i] ** 2 + solution[i + 1] ** 2) / radii[k])
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "exp1_constraint_sample.npz",
    )
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "exp1_constraint_activity.png",
    )
    args = parser.parse_args()

    data = np.load(args.sample)
    n = int(data["n"])
    theta = np.asarray(data["theta"], dtype=float)
    alm_solution = np.asarray(data["alm_solution"], dtype=float)
    reference_solution = np.asarray(data["casadi_solution"], dtype=float)
    alm_constraints = np.asarray(data["alm_constraints"], dtype=float)
    reference_constraints = np.asarray(data["casadi_constraints"], dtype=float)
    radii = radius_values(n, theta, alm_constraints.size, args.constraint_stride)
    alm_ratio = pair_norm_ratio(alm_solution, radii, args.constraint_stride)
    reference_ratio = pair_norm_ratio(reference_solution, radii, args.constraint_stride)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Match the 0.4-textwidth placement in the ICLR paper without rescaling fonts.
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.91), dpi=300)

    idx = np.arange(alm_ratio.size)
    color_alm = "#D62728"
    color_ref = "#1F77B4"

    y_min = min(float(np.min(alm_ratio)), float(np.min(reference_ratio)))
    y_max = max(float(np.max(alm_ratio)), float(np.max(reference_ratio)), 1.0)
    lower = max(0.90, y_min - 0.015)
    upper = 1.02
    ax.set_ylim(lower, upper)
    ax.set_facecolor("#F3F3F3")
    ax.fill_between(idx, lower, 1.0, color="#00FF00", alpha=0.035, label="Safe region", zorder=0)
    ax.fill_between(idx, 1.0, upper, color="#D62728", alpha=0.050, label="Violation region", zorder=0)
    ax.axhline(1.0, color="#7A7A7A", linestyle=(0, (3.0, 2.0)), linewidth=0.65, zorder=2)

    ax.plot(idx, alm_ratio, "-", label="lapanda", color=color_alm, linewidth=1.20, zorder=3)
    ax.plot(
        idx,
        reference_ratio,
        linestyle=(0, (3.2, 4.8)),
        label="CasADi",
        color=color_ref,
        marker=None,
        linewidth=1.0,
        dash_capstyle="round",
        zorder=4,
    )

    ax.set_xlabel("Constraint index $i$")
    ax.set_ylabel("Normalized constraint value")
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0.90, 0.94, 0.98, 1.02])
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle=(0, (6, 4)), linewidth=0.55, color="#666666", alpha=0.75)

    ax.tick_params(top=False, right=False, length=2.0, width=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.legend(frameon=True, borderpad=0.35, handlelength=2.2, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"), format="svg")
    fig.savefig(args.output.with_suffix(".pdf"), format="pdf")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
