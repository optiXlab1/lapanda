"""Plot nonlinear Quadrotor horizon scaling for lapanda and TurboMPC-GPU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    ROOT
    / "experiments"
    / "exp2_OCPs"
    / "summary_results"
    / "nonlinear_horizon_scaling_mixed_warm_start"
    / "raw"
)
DEFAULT_OUT_DIR = ROOT / "experiments" / "exp2_OCPs" / "summary_results"
HORIZONS = tuple(range(10, 201, 20))
MEASURED_REPETITIONS = 30

BLUE = "#0000FF"
RED = "#FF0000"
BLUE_FILL = "#CCCCFF"
RED_FILL = "#FFCCCC"
GRID = "#B8BDC7"
LEGEND_FACE = "#F2F3F5"
LEGEND_EDGE = "#C9CDD6"

TICK_FONTSIZE = 9.0
LABEL_FONTSIZE = 10.0
LEGEND_FONTSIZE = 10.0
PANEL_LABEL_FONTSIZE = 9.8

METHODS = (
    ("lapanda", "lapanda", BLUE, BLUE_FILL, "o", "-"),
    ("turbompc_gpu", "TurboMPC-GPU", RED, RED_FILL, "s", "--"),
)
TIME_PANELS = (("value_grad_time_ms", "Total time"),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean, max(mean - std, np.finfo(float).tiny), mean + std


def metric_path(data_dir: Path, horizon: int, key: str) -> Path:
    suffix = "lapanda_mixed" if key == "lapanda" else "turbompc_gpu_mixed"
    return data_dir / f"quadrotor_h{horizon}_{suffix}_metrics.csv"


def load_data(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for key, _label, _color, _fill, _marker, _line in METHODS:
        method_data = {name: [] for name, _panel in TIME_PANELS}
        method_data.update({f"{name}_lower": [] for name, _panel in TIME_PANELS})
        method_data.update({f"{name}_upper": [] for name, _panel in TIME_PANELS})
        method_data["constraint_violation_inf"] = []
        for horizon in HORIZONS:
            path = metric_path(data_dir, horizon, key)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows = rows[:MEASURED_REPETITIONS]
            for name, _panel in TIME_PANELS:
                mean, lower, upper = summarize([float(row[name]) for row in rows])
                method_data[name].append(mean)
                method_data[f"{name}_lower"].append(lower)
                method_data[f"{name}_upper"].append(upper)
            method_data["constraint_violation_inf"].append(
                float(np.mean([float(row["constraint_violation_inf"]) for row in rows]))
            )
        output[key] = {name: np.asarray(values) for name, values in method_data.items()}
    return output


def save_summary(data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    rows = []
    for index, horizon in enumerate(HORIZONS):
        for key, label, _color, _fill, _marker, _line in METHODS:
            rows.append(
                {
                    "horizon": horizon,
                    "method": label,
                    "total_time_ms": data[key]["value_grad_time_ms"][index],
                    "total_time_std_ms": (
                        data[key]["value_grad_time_ms_upper"][index]
                        - data[key]["value_grad_time_ms"][index]
                    ),
                    "constraint_violation_inf": data[key]["constraint_violation_inf"][index],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, alpha=0.45, linewidth=0.55, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=2.6, width=0.7, pad=1.5)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.55)
    for spine in ax.spines.values():
        spine.set_color("#25282D")
        spine.set_linewidth(0.72)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir)
    summary_path = args.outdir / "nonlinear_horizon_scaling_lapanda_turbompc_gpu.csv"
    save_summary(data, summary_path)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": TICK_FONTSIZE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.55))
    x = np.asarray(HORIZONS, dtype=float)

    for metric, _panel_label in TIME_PANELS:
        for key, label, color, fill, marker, linestyle in METHODS:
            mean = data[key][metric]
            ax.plot(
                x,
                mean,
                color=color,
                label=label,
                linewidth=1.25,
                linestyle=linestyle,
                marker=marker,
                markersize=4.0,
                zorder=4,
            )
        style_axis(ax)
        ax.set_xticks(range(10, 191, 20))
        ax.set_xlim(10, 190)
        ax.set_xlabel("Horizon $N$", fontsize=LABEL_FONTSIZE, labelpad=1.5)
    ax.set_ylabel("Total time (ms)", fontsize=LABEL_FONTSIZE, labelpad=1.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        framealpha=0.96,
        borderpad=0.42,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.80, bottom=0.18)
    stem = args.outdir / "nonlinear_horizon_scaling_lapanda_turbompc_gpu"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=400 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {summary_path}")
    print(f"wrote {stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
