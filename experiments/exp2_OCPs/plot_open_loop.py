"""Create the paper open-loop OCP summary figure.

Panels are ordered as:

    Cartpole loss | Quadrotor loss | Robot arm loss
    Cartpole timing | Quadrotor timing | Robot arm timing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "exp2_OCPs" / "summary_results"
TURBOMPC_WARM_TAG = "warm_e800_sqp50_admm1000"

BLUE = "#1F77B4"
RED = "#D62728"
GREEN = "#2CA02C"
PURPLE = "#9467BD"
BLUE_FILL = "#C7DDEC"
RED_FILL = "#F5C9C9"
GREEN_FILL = "#CAE7CA"
PURPLE_FILL = "#E4D8EE"

TICK_FONTSIZE = 7.0
LABEL_FONTSIZE = 7.0
LEGEND_FONTSIZE = 7.0
BAR_LABEL_FONTSIZE = 6.2
PANEL_LABEL_FONTSIZE = 8.0
PANEL_FACE = "#F3F3F3"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#C9CDD6"
LEGEND_FRAME_ALPHA = 0.96

METHODS = [
    {
        "key": "alm",
        "label": "lapanda",
        "color": RED,
        "fill": RED_FILL,
        "linestyle": "-",
        "zorder": 3,
    },
    {
        "key": "coc",
        "label": "SafePDP",
        "color": BLUE,
        "fill": BLUE_FILL,
        "linestyle": (0, (5.0, 2.2)),
        "zorder": 4,
    },
    {
        "key": "turbo_cpu",
        "label": "TurboMPC CPU",
        "color": GREEN,
        "fill": GREEN_FILL,
        "linestyle": (0, (4.2, 1.5, 1.0, 1.5)),
        "zorder": 3,
    },
    {
        "key": "turbo_gpu",
        "label": "TurboMPC GPU",
        "color": PURPLE,
        "fill": PURPLE_FILL,
        "linestyle": (0, (1.2, 1.6)),
        "zorder": 2,
    },
]

LOSS_METHODS = [
    {
        "key": "alm",
        "label": "lapanda",
        "color": RED,
        "linestyle": "-",
        "linewidth": 1.20,
        "alpha": 1.0,
        "zorder": 3,
    },
    {
        "key": "coc",
        "label": "SafePDP",
        "color": BLUE,
        "linestyle": (0, (5.0, 2.2)),
        "linewidth": 1.00,
        "alpha": 1.0,
        "zorder": 4,
    },
    {
        "key": "turbo",
        "label": "TurboMPC",
        "color": GREEN,
        "linestyle": (0, (4.0, 1.5, 1.0, 1.5)),
        "linewidth": 1.00,
        "alpha": 0.80,
        "zorder": 5,
    },
]

MODELS = [
    {
        "key": "cartpole",
        "title": "Cartpole",
        "alm": REPO_ROOT
        / "experiments/exp2_OCPs/cartpole/results/exp2_cartpole_open_loop_train_lapanda_fixed32.csv",
        "coc": REPO_ROOT
        / "experiments/exp2_OCPs/cartpole/baselines/results/exp2_cartpole_imitation_safepdp_fixed32_coc.csv",
        "turbo_cpu": REPO_ROOT / "experiments/exp2_OCPs/cartpole/turbompc/results/disabled_cartpole_turbompc_open_cpu.csv",
        "turbo_gpu": REPO_ROOT / "experiments/exp2_OCPs/cartpole/turbompc/results/disabled_cartpole_turbompc_open_gpu.csv",
    },
    {
        "key": "quadrotor",
        "title": "Quadrotor",
        "alm": REPO_ROOT / "experiments/exp2_OCPs/quadrotor/results/exp2_quadrotor_open_loop_train_lapanda_fixed32.csv",
        "coc": REPO_ROOT / "experiments/exp2_OCPs/quadrotor/baselines/results/exp2_quadrotor_imitation_safepdp_fixed32_coc.csv",
        "turbo_cpu": REPO_ROOT / f"experiments/exp2_OCPs/quadrotor/turbompc/results/quadrotor_cpu_open_{TURBOMPC_WARM_TAG}_metrics.csv",
        "turbo_gpu": REPO_ROOT / f"experiments/exp2_OCPs/quadrotor/turbompc/results/quadrotor_gpu_open_{TURBOMPC_WARM_TAG}_metrics.csv",
    },
    {
        "key": "robot_arm",
        "title": "Robot arm",
        "alm": REPO_ROOT / "experiments/exp2_OCPs/robot_arm/results/exp2_robot_arm_open_loop_train_lapanda_fixed32.csv",
        "coc": REPO_ROOT / "experiments/exp2_OCPs/robot_arm/baselines/results/exp2_robot_arm_imitation_safepdp_fixed32_coc.csv",
        "turbo_cpu": REPO_ROOT / f"experiments/exp2_OCPs/robot_arm/turbompc/results/robot_arm_cpu_open_{TURBOMPC_WARM_TAG}_metrics.csv",
        "turbo_gpu": REPO_ROOT / f"experiments/exp2_OCPs/robot_arm/turbompc/results/robot_arm_gpu_open_{TURBOMPC_WARM_TAG}_metrics.csv",
    },
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        _normalize_turbompc_row(row)
    return rows


def _normalize_turbompc_row(row: dict) -> None:
    if "forward_time_sec" not in row and row.get("forward_time_ms") not in (None, ""):
        row["forward_time_sec"] = str(_float(row, "forward_time_ms") / 1000.0)
    if "backward_time_sec" not in row and row.get("backward_time_ms") not in (None, ""):
        row["backward_time_sec"] = str(_float(row, "backward_time_ms") / 1000.0)
    if "loss_mean" not in row and row.get("loss") not in (None, ""):
        row["loss_mean"] = row["loss"]
    if "rss_peak_delta_mb" not in row and row.get("rss_delta_mb") not in (None, ""):
        row["rss_peak_delta_mb"] = row["rss_delta_mb"]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _loss_rows(model: dict, method_key: str) -> list[dict]:
    if method_key == "turbo":
        gpu_rows = _read_csv(model["turbo_gpu"])
        return gpu_rows if gpu_rows else _read_csv(model["turbo_cpu"])
    return _read_csv(model[method_key])


def _float(row: dict, key: str, fallback: str | None = None) -> float:
    try:
        return float(row.get(key, row.get(fallback, "nan") if fallback else "nan"))
    except Exception:
        return float("nan")


def _mean_field(rows: list[dict], key: str, max_epoch: int | None = None) -> float:
    vals = []
    for row in rows:
        if max_epoch is not None and "epoch" in row and int(float(row["epoch"])) > max_epoch:
            continue
        value = _float(row, key)
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def _std_field(rows: list[dict], key: str, max_epoch: int | None = None) -> float:
    vals = []
    for row in rows:
        if max_epoch is not None and "epoch" in row and int(float(row["epoch"])) > max_epoch:
            continue
        value = _float(row, key)
        if np.isfinite(value):
            vals.append(value)
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def _max_field(rows: list[dict], key: str, max_epoch: int | None = None) -> float:
    vals = []
    for row in rows:
        if max_epoch is not None and "epoch" in row and int(float(row["epoch"])) > max_epoch:
            continue
        value = _float(row, key)
        if np.isfinite(value):
            vals.append(value)
    return float(np.max(vals)) if vals else float("nan")


def _last_field(rows: list[dict], key: str, max_epoch: int | None = None) -> float:
    kept = [
        row
        for row in rows
        if max_epoch is None or "epoch" not in row or int(float(row["epoch"])) <= max_epoch
    ]
    return _float(kept[-1], key, "loss") if kept else float("nan")


def _first_field(rows: list[dict], key: str, max_epoch: int | None = None) -> float:
    kept = [
        row
        for row in rows
        if max_epoch is None or "epoch" not in row or int(float(row["epoch"])) <= max_epoch
    ]
    return _float(kept[0], key) if kept else float("nan")


def _epochs_recorded(rows: list[dict], max_epoch: int) -> int:
    epochs = [
        int(float(row["epoch"]))
        for row in rows
        if "epoch" in row and int(float(row["epoch"])) <= max_epoch
    ]
    return max(epochs) if epochs else -1


def _config_path_for_result(result_path: Path) -> Path:
    return result_path.parent / f"config_{result_path.stem}.json"


def _memory_result_path(result_path: Path) -> Path:
    name = result_path.name
    replacements = (
        ("_fixed32_coc.csv", "_memory_coc.csv"),
        ("_fixed32.csv", "_memory.csv"),
    )
    for old, new in replacements:
        if name.endswith(old):
            return result_path.with_name(name[: -len(old)] + new)
    return result_path


def _build_memory_mb(result_path: Path, rows: list[dict]) -> float:
    config = _read_json(_config_path_for_result(result_path))
    for problem_key, solver_key in (
        ("problem_build_rss_tree_peak_delta_mb", "solver_build_rss_tree_peak_delta_mb"),
        ("problem_build_rss_peak_delta_mb", "solver_build_rss_peak_delta_mb"),
    ):
        try:
            problem_value = float(config.get(problem_key, "nan"))
            solver_value = float(config.get(solver_key, "nan"))
            if np.isfinite(problem_value) and np.isfinite(solver_value):
                return problem_value + solver_value
        except Exception:
            pass
    for key in ("build_rss_tree_delta_mb", "build_rss_delta_mb"):
        value = config.get(key, None)
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            pass
    csv_value = _mean_field(rows, "build_rss_delta_mb")
    return csv_value


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _loss_arrays(rows: list[dict], max_epoch: int, smooth_window: int):
    kept = [row for row in rows if int(float(row["epoch"])) <= max_epoch]
    x = np.asarray([int(float(row["epoch"])) for row in kept], dtype=float)
    mean = np.asarray([_float(row, "loss_mean", "loss") for row in kept], dtype=float)
    sem = np.asarray([_float(row, "loss_sem") for row in kept], dtype=float)
    std = np.asarray([_float(row, "loss_std") for row in kept], dtype=float)
    missing_sem = ~np.isfinite(sem)
    if np.any(missing_sem):
        sem[missing_sem] = std[missing_sem] / np.sqrt(32.0)
    sem[~np.isfinite(sem)] = 0.0
    mean = _smooth(mean, smooth_window)
    sem = _smooth(sem, smooth_window)
    return x, mean, np.maximum(mean - sem, 0.0), mean + sem


def _tail_mean(rows: list[dict], key: str, tail_fraction: float, fallback: str | None = None) -> float:
    start = int((1.0 - tail_fraction) * len(rows))
    vals = np.asarray([_float(row, key, fallback) for row in rows[start:]], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _tail_std(rows: list[dict], key: str, tail_fraction: float) -> float:
    start = int((1.0 - tail_fraction) * len(rows))
    vals = np.asarray([_float(row, key) for row in rows[start:]], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def _plot_loss(ax, model: dict, max_epoch: int, smooth_window: int) -> None:
    all_lower = []
    all_upper = []
    for method in LOSS_METHODS[::-1]:
        rows = _loss_rows(model, method["key"])
        if not rows:
            continue
        x, mean, lo, hi = _loss_arrays(rows, max_epoch, smooth_window)
        all_lower.append(lo)
        all_upper.append(hi)
        ax.fill_between(x, lo, hi, color=method["color"], alpha=0.12, linewidth=0.0, zorder=method["zorder"] - 2)
        linewidth = 1.50 if model["key"] == "quadrotor" and method["key"] == "alm" else method["linewidth"]
        ax.plot(
            x,
            mean,
            color=method["color"],
            linestyle=method["linestyle"],
            linewidth=linewidth,
            alpha=method["alpha"],
            zorder=method["zorder"],
            label=method["label"],
        )
    ax.set_xlabel("")
    ax.set_xlim(0, max_epoch)
    if not all_lower:
        ax.text(0.5, 0.5, "Missing loss data", ha="center", va="center", transform=ax.transAxes, fontsize=TICK_FONTSIZE)
    elif model["key"] == "quadrotor":
        ymin = min(float(np.nanmin(values)) for values in all_lower)
        ymax = max(float(np.nanmax(values)) for values in all_upper)
        pad = 0.06 * max(ymax - ymin, 1.0)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    else:
        ax.set_ylim(bottom=0.0)
    if model["key"] == "cartpole":
        ax.set_ylim(0.0, 20.0)
        ax.set_yticks([0, 10, 20])
    elif model["key"] == "quadrotor":
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, 500.0)
        ax.set_yticks([200, 300, 400, 500])
    elif model["key"] == "robot_arm":
        ax.set_ylim(0.0, 30.0)
        ax.set_yticks([0, 10, 20, 30])
    ax.set_xticks([0, max_epoch // 2, max_epoch])
    ax.grid(True, color="#C6C6C6", linestyle="-", linewidth=0.45, alpha=0.42)
    ax.set_facecolor(PANEL_FACE)
    ax.tick_params(labelsize=TICK_FONTSIZE, top=False, right=False, length=2.0, width=0.65)


def _plot_timing(ax, model: dict, tail_fraction: float, show_legend: bool, ymax: float | None) -> float:
    rows_by_method = {method["key"]: _read_csv(model[method["key"]]) for method in METHODS}
    active_methods = [method for method in METHODS if rows_by_method[method["key"]]]
    if not active_methods:
        ax.text(0.5, 0.5, "Missing timing data", ha="center", va="center", transform=ax.transAxes, fontsize=TICK_FONTSIZE)
        return ymax if ymax is not None else 1.0
    fwd = [_mean_field(rows_by_method[method["key"]], "forward_time_sec") for method in active_methods]
    bwd = [_mean_field(rows_by_method[method["key"]], "backward_time_sec") for method in active_methods]
    fwd_std = [_std_field(rows_by_method[method["key"]], "forward_time_sec") for method in active_methods]
    bwd_std = [_std_field(rows_by_method[method["key"]], "backward_time_sec") for method in active_methods]
    values_ms = np.asarray([fwd, bwd], dtype=float) * 1000.0
    errors_ms = np.asarray([fwd_std, bwd_std], dtype=float) * 1000.0

    x = np.asarray([0.0, 1.45])
    group_width = 1.08
    width = group_width / len(active_methods) * 0.78
    offsets = np.linspace(-group_width / 2.0 + width / 2.0, group_width / 2.0 - width / 2.0, len(active_methods))
    all_bars = []
    for method_index, method in enumerate(active_methods):
        bars = ax.bar(
            x + offsets[method_index],
            values_ms[:, method_index],
            width,
            yerr=errors_ms[:, method_index],
            color=method["fill"],
            edgecolor=method["color"],
            linewidth=0.95,
            error_kw={"elinewidth": 0.75, "ecolor": method["color"], "capsize": 1.6, "capthick": 0.75},
            label=method["label"],
            zorder=3,
        )
        all_bars.append(bars)
    label_y_offsets = [2.6, 2.6, 2.6, 2.6]
    for bars_index, bars in enumerate(all_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(
                    0.0,
                    label_y_offsets[bars_index % len(label_y_offsets)],
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                color="#222222",
                clip_on=False,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["Fwd", "Bwd"], fontsize=TICK_FONTSIZE)
    ax.set_xlim(-0.72, 2.28)
    ax.set_xlabel("")
    ax.set_ylabel("Time (ms)", fontsize=LABEL_FONTSIZE, labelpad=1.0)
    if ymax is None:
        finite_values = values_ms[np.isfinite(values_ms)]
        finite_errors = errors_ms[np.isfinite(errors_ms)]
        top = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        err = float(np.nanmax(finite_errors)) if finite_errors.size else 0.0
        headroom = 1.42 if model["key"] == "cartpole" else 1.32
        ymax = (top + err) * headroom
    ax.set_ylim(0.0, ymax)
    if model["key"] == "cartpole":
        ax.set_ylim(0.0, 75.0)
        ax.set_yticks([0, 25, 50, 75])
    elif model["key"] == "quadrotor":
        ax.set_ylim(0.0, 100.0)
        ax.set_yticks([0, 50, 100])
    elif model["key"] == "robot_arm":
        ax.set_ylim(0.0, 75.0)
        ax.set_yticks([0, 25, 50, 75])
    ax.grid(True, axis="y", color="#C6C6C6", linestyle="-", linewidth=0.45, alpha=0.42)
    ax.set_facecolor(PANEL_FACE)
    ax.tick_params(labelsize=TICK_FONTSIZE, top=False, right=False, length=2.0, width=0.65)
    ax.margins(y=0.24)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=LEGEND_FRAME_ALPHA,
            facecolor=LEGEND_FACE,
            edgecolor=LEGEND_EDGE,
            fontsize=LEGEND_FONTSIZE,
            borderpad=0.24,
            handlelength=1.55,
            handletextpad=0.42,
            labelspacing=0.18,
        )
    return ymax


def _summary_rows(max_epoch: int, tail_fraction: float) -> list[dict]:
    rows_out = []
    specs = [
        ("alm", "lapanda", "alm"),
        ("coc", "SafePDP", "coc"),
        ("turbo_cpu", "turboMPC-CPU", "turbo_cpu"),
        ("turbo_gpu", "turboMPC-GPU", "turbo_gpu"),
    ]
    for model in MODELS:
        for _, label, path_key in specs:
            path = model[path_key]
            rows = _read_csv(path)
            if not rows:
                continue
            kept = [
                row
                for row in rows
                if "epoch" not in row or int(float(row["epoch"])) <= max_epoch
            ]
            if not kept:
                continue
            build_memory = _build_memory_mb(path, kept)
            mean_solve_memory = _mean_field(kept, "rss_peak_delta_mb")
            first_solve_memory = _first_field(kept, "rss_peak_delta_mb")
            tail_solve_memory = _tail_mean(kept, "rss_peak_delta_mb", tail_fraction)
            memory_path = _memory_result_path(path)
            memory_rows = _read_csv(memory_path)
            if memory_rows:
                build_memory = _build_memory_mb(memory_path, memory_rows)
                first_solve_memory = _first_field(memory_rows, "rss_peak_delta_mb")
                mean_solve_memory = _mean_field(memory_rows, "rss_peak_delta_mb")
                tail_solve_memory = _tail_mean(memory_rows, "rss_peak_delta_mb", tail_fraction)
            rows_out.append(
                {
                    "ocp": model["key"],
                    "method": label,
                    "status": "ok",
                    "epochs_recorded": _epochs_recorded(kept, max_epoch),
                    "rows": len(kept),
                    "final_loss": _last_field(kept, "loss_mean", max_epoch),
                    "tail20_loss": _tail_mean(kept, "loss_mean", tail_fraction, "loss"),
                    "mean_forward_time_ms": 1000.0 * _mean_field(kept, "forward_time_sec"),
                    "tail20_forward_time_ms": 1000.0 * _tail_mean(kept, "forward_time_sec", tail_fraction),
                    "mean_backward_time_ms": 1000.0 * _mean_field(kept, "backward_time_sec"),
                    "tail20_backward_time_ms": 1000.0 * _tail_mean(kept, "backward_time_sec", tail_fraction),
                    "build_memory_mb": build_memory,
                    "first_solve_memory_mb": first_solve_memory,
                    "mean_solve_memory_mb": mean_solve_memory,
                    "tail20_solve_memory_mb": tail_solve_memory,
                    "total_first_solve_memory_mb": build_memory + first_solve_memory,
                    "total_mean_memory_mb": build_memory + mean_solve_memory,
                    "total_tail20_memory_mb": build_memory + tail_solve_memory,
                    "mean_constraint_violation_inf": _mean_field(kept, "constraint_violation_inf"),
                    "max_constraint_violation_inf": _max_field(kept, "constraint_violation_inf"),
                    "path": str(path.relative_to(REPO_ROOT)),
                    "memory_path": str(memory_path.relative_to(REPO_ROOT)) if memory_rows else "",
                    "build_config_path": str(
                        _config_path_for_result(memory_path if memory_rows else path).relative_to(REPO_ROOT)
                    ),
                }
            )
    return rows_out


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=800)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=2.49)
    parser.add_argument("--smooth-window", type=int, default=31)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(args.width, args.height))
    for i, model in enumerate(MODELS):
        _plot_loss(axes[0, i], model, args.max_epoch, args.smooth_window)
        _plot_timing(
            axes[1, i],
            model,
            args.tail_fraction,
            show_legend=False,
            ymax=None,
        )
        axes[0, i].set_title("")
        axes[0, i].set_ylabel("Imitation loss" if i == 0 else "", fontsize=LABEL_FONTSIZE, labelpad=1.0)
        axes[1, i].set_ylabel("Time (ms)" if i == 0 else "", fontsize=LABEL_FONTSIZE, labelpad=1.0)

    available_timing_keys = {
        method["key"]
        for model in MODELS
        for method in METHODS
        if _read_csv(model[method["key"]])
    }
    timing_handles = [
        Line2D(
            [0],
            [0],
            color=method["color"],
            linestyle="-",
            linewidth=0.0,
            marker="s",
            markersize=5.4,
            markerfacecolor=method["fill"],
            markeredgecolor=method["color"],
            markeredgewidth=0.9,
            label=method["label"],
        )
        for method in METHODS
        if method["key"] in available_timing_keys
    ]
    fig.legend(
        handles=timing_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=min(len(timing_handles), 4),
        frameon=True,
        fancybox=True,
        framealpha=LEGEND_FRAME_ALPHA,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=LEGEND_FONTSIZE,
        handlelength=0.90,
        handletextpad=0.48,
        columnspacing=0.90,
        borderpad=0.34,
        labelspacing=0.18,
    )
    fig.subplots_adjust(left=0.068, right=0.985, top=0.819, bottom=0.171, wspace=0.22, hspace=0.40)
    for i, model in enumerate(MODELS):
        bottom = axes[1, i].get_position()
        x_center = 0.5 * (bottom.x0 + bottom.x1)
        fig.text(
            x_center,
            0.050,
            f"({chr(ord('a') + i)}) {model['title']}",
            ha="center",
            va="center",
            fontsize=PANEL_LABEL_FONTSIZE,
            fontfamily="Times New Roman",
        )

    png = args.out_dir / "ocp_fixed32_2x3_loss_timing.png"
    svg = args.out_dir / "ocp_fixed32_2x3_loss_timing.svg"
    pdf = args.out_dir / "ocp_fixed32_2x3_loss_timing.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    summary_path = args.out_dir / f"open_loop_{args.max_epoch}_timing_summary.csv"
    _write_summary(summary_path, _summary_rows(args.max_epoch, args.tail_fraction))
    print(f"wrote {png}")
    print(f"wrote {svg}")
    print(f"wrote {pdf}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
