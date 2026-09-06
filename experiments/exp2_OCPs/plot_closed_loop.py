"""Build the final Exp.2 OCP baseline figures and table.

Outputs:
- experiments/exp2_OCPs/summary_results/paper_safepdp_timing_comparison.png/svg
- experiments/exp2_OCPs/summary_results/paper_safepdp_loss_comparison.png/svg
- experiments/exp2_OCPs/summary_results/paper_safepdp_constraint_bands.png/svg
- experiments/exp2_OCPs/summary_results/paper_safepdp_vs_alm_table.csv

All reported lapanda and SafePDP runs use tolerance 1e-3.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp2_OCPs.problem_bank import build_problem as build_ocp_problem
from experiments.exp2_OCPs.config import ALM_DEFAULTS, MPC_SOLVER_DEFAULTS


OUT_DIR = REPO_ROOT / "experiments" / "exp2_OCPs" / "summary_results"
TURBOMPC_WARM_TAG = "warm_e500_sqp50_admm1000"
TURBOMPC_OPEN_WARM_TAG = "warm_e800_sqp50_admm1000"
TOLERANCE = 1e-3
ICLR_TEXT_WIDTH_IN = 5.5

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

COLORS = {
    "alm": "#D62728",
    "safepdp": "#1F77B4",
    "safepdp_coc": "#1F77B4",
    "turbompc": "#2CA02C",
    "turbo_cpu": "#2CA02C",
    "turbo_gpu": "#9467BD",
    "forward": "#D62728",
    "backward": "#1F77B4",
    "snapshots": ["#2F6B3F", "#7CA650", "#D7A541", "#C46A55"],
}

OCP_1X6_TICK_FONTSIZE = 7.0
OCP_1X6_LABEL_FONTSIZE = 7.0
OCP_1X6_LOSS_LEGEND_FONTSIZE = 4.6
OCP_1X6_CONSTRAINT_LEGEND_FONTSIZE = 4.6
OCP_1X6_PANEL_LABEL_FONTSIZE = 8.0
OCP_PANEL_FACE = "#F3F3F3"
OCP_LEGEND_FACE = "#FFFFFF"
OCP_LEGEND_EDGE = "#D3D6DE"
OCP_LEGEND_FRAME_ALPHA = 0.96
SAFEPDP_DASH = (0, (3.2, 4.8))

LOSS_STYLES = [
    {
        "key": "safepdp_coc",
        "label": "SafePDP",
        "color": COLORS["safepdp_coc"],
        "linestyle": SAFEPDP_DASH,
        "linewidth": 1.00,
        "marker": None,
        "markersize": 0.0,
        "markevery": None,
        "alpha": 1.0,
        "zorder": 8,
    },
    {
        "key": "turbompc",
        "label": "TurboMPC",
        "color": COLORS["turbompc"],
        "linestyle": (0, (4.0, 1.5, 1.0, 1.5)),
        "linewidth": 1.00,
        "marker": None,
        "markersize": 0.0,
        "markevery": None,
        "alpha": 0.76,
        "zorder": 10,
    },
    {
        "key": "alm",
        "label": "lapanda",
        "color": COLORS["alm"],
        "linestyle": "-",
        "linewidth": 1.20,
        "marker": None,
        "markersize": 0.0,
        "markevery": None,
        "alpha": 1.0,
        "zorder": 6,
    },
]

TRAINING_EPOCHS = 500
CONSTRAINT_EPOCHS = list(range(0, 101, 4)) + list(range(110, TRAINING_EPOCHS + 1, 10))
CONSTRAINT_SNAPSHOT_METHOD = "lapanda"
CONSTRAINT_ROLLOUT_STEPS = 50
CONSTRAINT_SAMPLE_FRACTION = 0.5
CONSTRAINT_ROLLOUT_CACHE_VERSION = 14
QUADROTOR_ANGLE_BASE = 0.38
QUADROTOR_FLOOR_BASE = 0.43
QUADROTOR_CEILING_HEIGHT = 1.20
CARTPOLE_ENERGY_BUDGET = 70.0
TARGET_COLOR = "#D52627"
TARGET_ALPHA = 0.85
TARGET_DASH = (0, (0.55, 2.35))
FIG4_TARGET_DASH = (0, (1.4, 1.4))
BOUNDARY_DASH = (0, (3.3, 1.8))
FIXED_BOUNDARY_COLOR = "#000000"
CONSTRAINT_HISTORY_COLOR = "#6F98E6"
CONSTRAINT_FINAL_COLOR = "#4068E0"
CONSTRAINT_HISTORY_CMAP = LinearSegmentedColormap.from_list(
    "history_epoch_blues",
    ["#DCE9FF", "#8FB1EA", "#2B63BF"],
)
EPOCH_CMAP = LinearSegmentedColormap.from_list(
    "epoch_blue_to_target",
    ["#C9D6EE", "#6E8FCB", "#2A4D8F", "#102B63"],
)

EXPERIMENTS = [
    {
        "model": "cartpole",
        "display": "Cartpole",
        "problem_bank_model": "cartpole",
        "exp_dir": REPO_ROOT / "experiments" / "exp2_OCPs" / "cartpole",
        "prefix": "exp2_cartpole",
        "constraint_sample_index": 15,
        "alm_imitation": "results/exp2_cartpole_closed_loop_train_lapanda_closed_loop.csv",
        "alm_mpc": "results/exp2_cartpole_mpc_lapanda_teacher.csv",
        "constraint_note": "remaining control-energy budget",
    },
    {
        "model": "quadrotor",
        "display": "Quadrotor",
        "problem_bank_model": "quadrotor",
        "exp_dir": REPO_ROOT / "experiments" / "exp2_OCPs" / "quadrotor",
        "prefix": "exp2_quadrotor",
        "constraint_sample_index": 15,
        "alm_imitation": "results/exp2_quadrotor_closed_loop_train_lapanda_closed_loop.csv",
        "alm_mpc": "results/exp2_quadrotor_mpc_lapanda_teacher.csv",
        "constraint_note": "min height/progress/angle margin",
    },
    {
        "model": "robot_arm",
        "display": "Robot arm",
        "problem_bank_model": "robot_arm",
        "exp_dir": REPO_ROOT / "experiments" / "exp2_OCPs" / "robot_arm",
        "prefix": "exp2_robot_arm",
        "constraint_sample_index": 0,
        "alm_imitation": "results/exp2_robot_arm_closed_loop_train_lapanda_closed_loop.csv",
        "alm_mpc": "results/exp2_robot_arm_mpc_lapanda_teacher.csv",
        "constraint_note": "min obstacle/joint margin",
    },
]

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        normalize_turbompc_row(row)
    return rows


def normalize_turbompc_row(row: dict) -> None:
    if "forward_time_sec" not in row and row.get("forward_time_ms") not in (None, ""):
        row["forward_time_sec"] = str(to_float(row.get("forward_time_ms")) / 1000.0)
    if "backward_time_sec" not in row and row.get("backward_time_ms") not in (None, ""):
        row["backward_time_sec"] = str(to_float(row.get("backward_time_ms")) / 1000.0)
    if "loss_mean" not in row and row.get("loss") not in (None, ""):
        row["loss_mean"] = row["loss"]
    if "rss_peak_delta_mb" not in row and row.get("rss_delta_mb") not in (None, ""):
        row["rss_peak_delta_mb"] = row["rss_delta_mb"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def to_int(value, default=-1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_vector(value, default=None) -> np.ndarray:
    if default is None:
        default = []
    if value is None:
        return np.asarray(default, dtype=float)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    try:
        return np.asarray(ast.literal_eval(str(value)), dtype=float).reshape(-1)
    except (SyntaxError, ValueError, TypeError):
        return np.asarray(default, dtype=float).reshape(-1)


def mean_field(rows: list[dict], field: str) -> float:
    values = np.asarray([to_float(row.get(field)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def max_field(rows: list[dict], field: str) -> float:
    values = np.asarray([to_float(row.get(field)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def first_field(rows: list[dict], field: str) -> float:
    if not rows:
        return float("nan")
    return to_float(rows[0].get(field))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def config_path_for_result(result_path: Path) -> Path:
    return result_path.parent / f"config_{result_path.stem}.json"


def memory_result_path(result_path: Path) -> Path:
    name = result_path.name
    replacements = (
        ("_closed_loop_coc.csv", "_memory_coc.csv"),
        ("_closed_loop.csv", "_memory.csv"),
    )
    for old, new in replacements:
        if name.endswith(old):
            return result_path.with_name(name[: -len(old)] + new)
    return result_path


def build_memory_mb(result_path: Path) -> float:
    config = read_json(config_path_for_result(result_path))
    for problem_key, solver_key in (
        ("problem_build_rss_tree_peak_delta_mb", "solver_build_rss_tree_peak_delta_mb"),
        ("problem_build_rss_peak_delta_mb", "solver_build_rss_peak_delta_mb"),
    ):
        problem_value = to_float(config.get(problem_key))
        solver_value = to_float(config.get(solver_key))
        if np.isfinite(problem_value) and np.isfinite(solver_value):
            return problem_value + solver_value
    for key in ("build_rss_tree_delta_mb", "build_rss_delta_mb"):
        value = to_float(config.get(key))
        if np.isfinite(value):
            return value
    return float("nan")


def safepdp_imitation_path(exp: dict) -> Path:
    closed_loop_path = (
        exp["exp_dir"]
        / "baselines"
        / "results"
        / f"{exp['prefix']}_closed_loop_train_safepdp_closed_loop.csv"
    )
    if closed_loop_path.exists():
        return closed_loop_path
    return exp["exp_dir"] / "baselines" / "results" / f"{exp['prefix']}_imitation_safepdp.csv"


def safepdp_coc_imitation_path(exp: dict) -> Path:
    return (
        exp["exp_dir"]
        / "baselines"
        / "results"
        / f"{exp['prefix']}_closed_loop_train_safepdp_closed_loop_coc.csv"
    )


def turbompc_closed_imitation_path(exp: dict) -> Path:
    gpu_warm_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_gpu_closed_{TURBOMPC_WARM_TAG}_metrics.csv"
    cpu_warm_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_cpu_closed_{TURBOMPC_WARM_TAG}_metrics.csv"
    if gpu_warm_path.exists():
        return gpu_warm_path
    if cpu_warm_path.exists():
        return cpu_warm_path
    gpu_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_closed_gpu.csv"
    cpu_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_closed_cpu.csv"
    return gpu_path if gpu_path.exists() else cpu_path


def turbompc_closed_variant_path(exp: dict, variant: str) -> Path:
    warm_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_{variant}_closed_{TURBOMPC_WARM_TAG}_metrics.csv"
    if warm_path.exists():
        return warm_path
    return exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_closed_{variant}.csv"


def turbompc_open_variant_path(exp: dict, variant: str) -> Path:
    open_warm_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_{variant}_open_{TURBOMPC_OPEN_WARM_TAG}_metrics.csv"
    if open_warm_path.exists():
        return open_warm_path
    warm_path = exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_{variant}_open_{TURBOMPC_WARM_TAG}_metrics.csv"
    if warm_path.exists():
        return warm_path
    return exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_open_{variant}.csv"


def turbompc_mpc_variant_path(exp: dict, variant: str) -> Path:
    return exp["exp_dir"] / "turbompc" / "results" / f"{exp['model']}_turbompc_mpc_teacher_{variant}.csv"


def safepdp_mpc_path(exp: dict) -> Path:
    tagged_path = exp["exp_dir"] / "baselines" / "results" / f"{exp['prefix']}_mpc_safepdp_teacher.csv"
    if tagged_path.exists():
        return tagged_path
    return exp["exp_dir"] / "baselines" / "results" / f"{exp['prefix']}_mpc_safepdp.csv"


def result_paths(exp: dict, method: str) -> tuple[Path, Path]:
    if method == "lapanda":
        return exp["exp_dir"] / exp["alm_imitation"], exp["exp_dir"] / exp["alm_mpc"]
    if method == "SafePDP":
        return safepdp_coc_imitation_path(exp), safepdp_mpc_path(exp)
    if method == "SafePDP-COC":
        return safepdp_coc_imitation_path(exp), safepdp_mpc_path(exp)
    if method == "turboMPC-CPU":
        path = turbompc_closed_variant_path(exp, "cpu")
        return path, path
    if method == "turboMPC-GPU":
        path = turbompc_closed_variant_path(exp, "gpu")
        return path, path
    raise ValueError(method)


def collect_method_summary(exp: dict, method: str) -> dict:
    imitation_path, _ = result_paths(exp, method)
    rows = read_csv(imitation_path)
    memory_path = memory_result_path(imitation_path)
    memory_rows = read_csv(memory_path)
    if method in {"turboMPC-CPU", "turboMPC-GPU"}:
        memory_path = imitation_path
        memory_rows = rows
    build_memory = build_memory_mb(memory_path) if memory_rows else float("nan")
    first_solve_memory = first_field(memory_rows, "rss_peak_delta_mb")
    mean_solve_memory = mean_field(memory_rows, "rss_peak_delta_mb")
    mean_gpu_memory = mean_field(memory_rows, "gpu_used_delta_mb")
    peak_solve_memory = max_field(memory_rows, "rss_peak_delta_mb")
    peak_gpu_memory = max_field(memory_rows, "gpu_used_delta_mb")
    if method in {"turboMPC-CPU", "turboMPC-GPU"}:
        variant = "cpu" if method.endswith("CPU") else "gpu"
        build_rows = read_csv(turbompc_open_variant_path(exp, variant))
        build_memory = first_field(build_rows, "build_rss_delta_mb")
        build_gpu_memory = first_field(build_rows, "build_gpu_used_delta_mb")
        first_solve_memory = float("nan")
    total_peak_memory = build_memory + peak_solve_memory
    if not np.isfinite(total_peak_memory):
        total_peak_memory = peak_solve_memory
    status = "ok" if rows else "missing"
    return {
        "ocp": exp["display"],
        "model": exp["model"],
        "method": method,
        "tolerance": TOLERANCE,
        "status": status,
        "mean_forward_time_sec": mean_field(rows, "forward_time_sec"),
        "mean_backward_time_sec": mean_field(rows, "backward_time_sec"),
        "build_memory_mb": build_memory,
        "first_solve_memory_mb": first_solve_memory,
        "mean_solve_memory_mb": mean_solve_memory,
        "peak_solve_memory_mb": peak_solve_memory,
        "build_gpu_memory_delta_mb": build_gpu_memory if method in {"turboMPC-CPU", "turboMPC-GPU"} else float("nan"),
        "mean_gpu_memory_delta_mb": mean_gpu_memory,
        "peak_gpu_memory_delta_mb": peak_gpu_memory,
        "total_peak_memory_mb": total_peak_memory,
        "mean_constraint_violation_inf": mean_field(rows, "constraint_violation_inf"),
        "memory_path": str(memory_path.relative_to(REPO_ROOT)) if memory_rows else "",
    }


def write_safepdp_vs_alm_table() -> None:
    rows = []
    for exp in EXPERIMENTS:
        rows.append(collect_method_summary(exp, "lapanda"))
        rows.append(collect_method_summary(exp, "SafePDP"))
        for method in ("turboMPC-CPU", "turboMPC-GPU"):
            turbo_path, _ = result_paths(exp, method)
            if turbo_path.exists():
                rows.append(collect_method_summary(exp, method))
    write_csv(OUT_DIR / "paper_safepdp_vs_alm_table.csv", rows)


def run_pipeline() -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "exp2_OCPs" / "run_closed_loop.py"),
            "--epochs",
            str(TRAINING_EPOCHS),
            "--learning-rate",
            "0.001",
            "--steps",
            str(CONSTRAINT_ROLLOUT_STEPS),
            "--force-teacher-generation",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": "#50545E",
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "grid.color": "#C6C6C6",
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "grid.alpha": 0.42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial",
            "mathtext.bf": "Arial",
        },
    )


def style_pub_axis(ax, xgrid: bool = True) -> None:
    ax.set_facecolor(OCP_PANEL_FACE)
    ax.xaxis.grid(xgrid, which="major")
    ax.yaxis.grid(True, which="major")
    ax.grid(axis="y", color="#C6C6C6", linewidth=0.45, linestyle="-", alpha=0.42)
    if xgrid:
        ax.grid(axis="x", color="#C6C6C6", linewidth=0.45, linestyle="-", alpha=0.42)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1F1F1F")
        spine.set_linewidth(0.75)
    ax.tick_params(
        axis="both",
        labelsize=5.8,
        top=False,
        right=False,
        bottom=True,
        left=True,
        direction="out",
        length=1.8,
        width=0.6,
        color="#20232A",
    )


def load_mpc_snapshots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    return [dict(item) for item in data["snapshots"].tolist()]


def select_mpc_snapshot(snapshots: list[dict], score_fn=None, feasibility_tol: float = 1e-3) -> dict | None:
    if not snapshots:
        return None
    if score_fn is None:
        return snapshots[0]
    feasible = []
    infeasible = []
    for snapshot in snapshots[1:]:
        score = score_fn(snapshot)
        if score >= -feasibility_tol:
            feasible.append((score, snapshot))
        else:
            infeasible.append((score, snapshot))
    first_score = score_fn(snapshots[0])
    if first_score >= -feasibility_tol:
        feasible.append((first_score, snapshots[0]))
    else:
        infeasible.append((first_score, snapshots[0]))
    if feasible:
        # For display purposes, prefer the snapshot closest to the active boundary
        # while remaining feasible.
        return min(feasible, key=lambda item: item[0])[1]
    # If every candidate violates, show the least violating one.
    return max(infeasible, key=lambda item: item[0])[1]


def _cartpole_constraint_score(snapshot: dict, instance) -> float:
    controls = np.asarray(snapshot["controls"], dtype=float).reshape(-1)
    margin = CARTPOLE_ENERGY_BUDGET - np.cumsum(controls**2)
    return float(np.nanmin(margin))


def _quadrotor_constraint_score(snapshot: dict, instance) -> float:
    states = np.asarray(snapshot["states"], dtype=float)
    floor_height = QUADROTOR_FLOOR_BASE
    ceiling_height = QUADROTOR_CEILING_HEIGHT
    angle_max = 0.34
    values = []
    for x in states[1:]:
        values.extend(
            [
                float(x[1] - floor_height),
                float(ceiling_height - x[1]),
                float(instance.x_target[0] - x[0]),
                float(angle_max - abs(x[4])),
            ]
        )
    return float(np.nanmin(values))


def _arm_constraint_score(snapshot: dict, instance) -> float:
    states = np.asarray(snapshot["states"], dtype=float)
    theta = np.asarray(instance.theta_true, dtype=float)
    obstacle_radius = 0.30 + theta[4]
    q_lower = np.asarray([-1.10, -0.65], dtype=float)
    q_upper = np.asarray([0.75, 1.25], dtype=float)
    values = []
    for q0, q1 in states[1:]:
        ee = np.asarray(
            [
                np.cos(q0) + 0.8 * np.cos(q0 + q1),
                np.sin(q0) + 0.8 * np.sin(q0 + q1),
            ],
            dtype=float,
        )
        values.extend(
            [
                float(np.linalg.norm(ee - np.array([1.30, 0.30])) - obstacle_radius),
                float(q_upper[0] - q0),
                float(q0 - q_lower[0]),
                float(q_upper[1] - q1),
                float(q1 - q_lower[1]),
            ]
        )
    return float(np.nanmin(values))


def select_constraint_snapshot(exp: dict):
    instance = build_ocp_problem(exp["problem_bank_model"])
    candidates = []
    for method in ("lapanda", "SafePDP"):
        _, mpc_path = result_paths(exp, method)
        snapshots = load_mpc_snapshots(mpc_path.with_suffix(".mpc_snapshots.npz"))
        if snapshots:
            candidates.append((method, select_mpc_snapshot(snapshots, lambda s: constraint_score(exp, instance, s))))
    if not candidates:
        return None, None, instance
    # Prefer lapanda when both are available.
    for method, snapshot in candidates:
        if method == "lapanda":
            return method, snapshot, instance
    return candidates[0][0], candidates[0][1], instance


def constraint_score(exp: dict, instance, snapshot: dict) -> float:
    if exp["model"] == "cartpole":
        return _cartpole_constraint_score(snapshot, instance)
    if exp["model"] == "quadrotor":
        return _quadrotor_constraint_score(snapshot, instance)
    if exp["model"] == "robot_arm":
        return _arm_constraint_score(snapshot, instance)
    raise ValueError(exp["model"])


def load_method_rows(exp: dict, method: str, kind: str) -> list[dict]:
    imitation_path, mpc_path = result_paths(exp, method)
    path = imitation_path if kind == "imitation" else mpc_path
    return sorted(read_csv(path), key=lambda row: to_int(row.get("epoch" if kind == "imitation" else "step")))


def training_snapshot_path(exp: dict, method: str) -> Path:
    imitation_path, _ = result_paths(exp, method)
    return imitation_path.with_suffix(".snapshots.npz")


def load_training_snapshots(exp: dict, method: str) -> list[dict]:
    return load_mpc_snapshots(training_snapshot_path(exp, method))


def select_epoch_snapshot(snapshots: list[dict], epoch: int, sample_index: int | None = None) -> dict | None:
    epoch_snapshots = [snapshot for snapshot in snapshots if to_int(snapshot.get("epoch")) == epoch]
    if not epoch_snapshots:
        return None
    if sample_index is None:
        index = int(round((len(epoch_snapshots) - 1) * CONSTRAINT_SAMPLE_FRACTION))
    else:
        index = min(max(int(sample_index), 0), len(epoch_snapshots) - 1)
    return epoch_snapshots[index]


def select_training_epoch_snapshots(exp: dict, method: str, epochs: list[int] = CONSTRAINT_EPOCHS) -> dict[int, dict]:
    snapshots = load_training_snapshots(exp, method)
    if not snapshots:
        return {}
    rows_by_epoch = {
        to_int(row.get("epoch")): row
        for row in load_method_rows(exp, method, "imitation")
    }
    selected = {}
    for epoch in epochs:
        snapshot = select_epoch_snapshot(snapshots, epoch, exp.get("constraint_sample_index"))
        if snapshot is not None:
            if "theta" not in snapshot and epoch in rows_by_epoch:
                theta = parse_vector(rows_by_epoch[epoch].get("theta"))
                if theta.size:
                    snapshot["theta"] = theta
            selected[epoch] = snapshot
    return selected


def selected_epoch_thetas(exp: dict, method: str = CONSTRAINT_SNAPSHOT_METHOD) -> dict[int, np.ndarray]:
    rows_by_epoch = {
        to_int(row.get("epoch")): row
        for row in load_method_rows(exp, method, "imitation")
    }
    thetas = {}
    instance = build_ocp_problem(exp["problem_bank_model"])
    for epoch in CONSTRAINT_EPOCHS:
        row = rows_by_epoch.get(epoch)
        if row is None:
            continue
        theta = parse_vector(row.get("theta"), instance.theta_true)
        if theta.size == instance.theta_true.size:
            thetas[epoch] = theta
    return thetas


def _rollout_cache_key(exp: dict, thetas_by_epoch: dict[int, np.ndarray]) -> str:
    model = exp["model"]
    solver_defaults = MPC_SOLVER_DEFAULTS[model]
    alm_defaults = ALM_DEFAULTS[model]
    pieces = [
        str(CONSTRAINT_ROLLOUT_CACHE_VERSION),
        CONSTRAINT_SNAPSHOT_METHOD,
        model,
        str(CONSTRAINT_ROLLOUT_STEPS),
        str(CONSTRAINT_EPOCHS),
    ]
    for epoch in sorted(thetas_by_epoch):
        pieces.append(f"{epoch}:{np.array2string(thetas_by_epoch[epoch], precision=10, separator=',')}")
    pieces.append(
        "cfg:"
        f"{solver_defaults.backend},"
        f"{solver_defaults.inner_max_iter},"
        f"{solver_defaults.inner_tol:.12g},"
        f"{solver_defaults.max_outer},"
        f"{solver_defaults.alm_tol:.12g},"
        f"{alm_defaults.initial_penalty:.12g},"
        f"{alm_defaults.penalty_update_factor:.12g},"
        f"{alm_defaults.max_penalty:.12g}"
    )
    return "|".join(pieces)


def _load_cached_constraint_rollouts(exp: dict, cache_key: str) -> dict[int, dict] | None:
    path = OUT_DIR / f"{exp['prefix']}_{CONSTRAINT_SNAPSHOT_METHOD.lower().replace('-', '_')}_constraint_rollouts.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    cached_key = str(data.get("cache_key", "")).replace("ALM-PANDA", "lapanda")
    legacy_key = cache_key.rsplit("|cfg:", 1)[0]
    if cached_key not in {cache_key, legacy_key}:
        return None
    return {int(epoch): dict(rollout) for epoch, rollout in data["rollouts"].tolist()}


def _save_cached_constraint_rollouts(exp: dict, cache_key: str, rollouts: dict[int, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{exp['prefix']}_{CONSTRAINT_SNAPSHOT_METHOD.lower().replace('-', '_')}_constraint_rollouts.npz"
    np.savez(
        path,
        cache_key=cache_key,
        rollouts=np.asarray([(epoch, rollout) for epoch, rollout in rollouts.items()], dtype=object),
        allow_pickle=True,
    )


def _safe_shift(u: np.ndarray, nu: int) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(-1)
    if u.size <= nu:
        return np.zeros_like(u)
    return np.concatenate([u[nu:], u[-nu:]])


def _alm_rollout_args(exp: dict) -> argparse.Namespace:
    model = exp["model"]
    solver_defaults = MPC_SOLVER_DEFAULTS[model]
    return argparse.Namespace(
        backend=solver_defaults.backend,
        inner_max_iter=solver_defaults.inner_max_iter,
        inner_tol=solver_defaults.inner_tol,
        max_outer=solver_defaults.max_outer,
        alm_tol=solver_defaults.alm_tol,
        force=False,
    )


def _cartpole_constraint_rollout(theta: np.ndarray, steps: int) -> dict:
    from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver
    from experiments.exp2_OCPs.utils import alm_options, backward_options, inner_options, make_variable_from_state

    instance = build_ocp_problem("cartpole")
    args = _alm_rollout_args({"model": "cartpole"})
    solver = build_solver(
        instance.problem,
        backend=args.backend,
        name=f"exp2_cartpole_constraint_rollout_h{instance.horizon}",
        cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "exp2_cartpole",
        force=args.force,
    )
    state = instance.x_initial.copy()
    states = [state.copy()]
    applied_controls = []
    predicted_energy = []
    warm = np.zeros(instance.horizon * instance.nu)
    multiplier = None
    penalty = None
    for _ in range(steps):
        variable = make_variable_from_state(instance, state, np.zeros(instance.horizon))
        result = solver.solve_lapanda(
            warm,
            theta,
            variable,
            instance.constraint_lower,
            instance.constraint_upper,
            inner_solver_options=inner_options(SolverOptions, args.inner_max_iter, args.inner_tol),
            alm_options=alm_options(
                AlmOptions,
                args.max_outer,
                args.alm_tol,
                config=ALM_DEFAULTS["cartpole"],
            ),
            backward_options=backward_options(BackwardOptions, True, args.alm_tol),
            multiplier0=multiplier,
            penalty0=None,
        )
        solution = np.asarray(result["solution"], dtype=float).reshape(-1)
        applied_controls.append(float(solution[0]))
        predicted_energy.append(float(np.sum(solution**2)))
        rollout = np.asarray(instance.rollout_fun(solution, theta, variable), dtype=float)
        state = rollout[1, : instance.nx]
        states.append(state.copy())
        warm = _safe_shift(solution, instance.nu)
        multiplier = np.asarray(result["multipliers"], dtype=float)
        penalty = np.asarray(result["penalties"], dtype=float)
    return {
        "theta": theta.copy(),
        "states": np.asarray(states, dtype=float),
        "applied_controls": np.asarray(applied_controls, dtype=float),
        "predicted_energy": np.asarray(predicted_energy, dtype=float),
    }


def _physical_constraint_rollout(exp: dict, theta: np.ndarray, steps: int) -> dict:
    from lapanda import AlmOptions, BackwardOptions, SolverOptions, build_solver
    from experiments.exp2_OCPs.utils import (
        alm_options,
        backward_options,
        build_alm_solver,
        first_next_state,
        inner_options,
        make_variable_from_state,
    )

    model = exp["problem_bank_model"]
    instance = build_ocp_problem(model)
    args = _alm_rollout_args(exp)
    solver = build_alm_solver(
        build_solver,
        instance,
        name=f"{exp['prefix']}_constraint_rollout_h{instance.horizon}",
        backend=args.backend,
        force=args.force,
        cache_tag=exp["prefix"],
    )
    state = instance.x_initial.copy()
    states = [state.copy()]
    applied_controls = []
    warm = np.zeros(instance.horizon * instance.nu)
    multiplier = None
    penalty = None
    for _ in range(steps):
        variable = make_variable_from_state(instance, state, np.zeros(instance.horizon * instance.nu))
        result = solver.solve_lapanda(
            warm,
            theta,
            variable,
            instance.constraint_lower,
            instance.constraint_upper,
            inner_solver_options=inner_options(SolverOptions, args.inner_max_iter, args.inner_tol),
            alm_options=alm_options(
                AlmOptions,
                args.max_outer,
                args.alm_tol,
                config=ALM_DEFAULTS[exp["model"]],
            ),
            backward_options=backward_options(BackwardOptions, False, args.alm_tol),
            multiplier0=multiplier,
            penalty0=None,
        )
        solution = np.asarray(result["solution"], dtype=float).reshape(-1)
        applied_controls.append(solution[: instance.nu].copy())
        state = first_next_state(instance, solution, theta, variable)
        states.append(state.copy())
        warm = _safe_shift(solution, instance.nu)
        multiplier = np.asarray(result["multipliers"], dtype=float)
        penalty = np.asarray(result["penalties"], dtype=float)
    return {
        "theta": theta.copy(),
        "states": np.asarray(states, dtype=float),
        "applied_controls": np.asarray(applied_controls, dtype=float),
    }


def load_or_build_constraint_rollouts(exp: dict) -> dict[int, dict]:
    thetas = selected_epoch_thetas(exp, CONSTRAINT_SNAPSHOT_METHOD)
    if not thetas:
        return {}
    cache_key = _rollout_cache_key(exp, thetas)
    cached = _load_cached_constraint_rollouts(exp, cache_key)
    if cached is not None:
        return cached
    rollouts = {}
    for epoch, theta in thetas.items():
        if exp["model"] == "cartpole":
            rollouts[epoch] = _cartpole_constraint_rollout(theta, CONSTRAINT_ROLLOUT_STEPS)
        else:
            rollouts[epoch] = _physical_constraint_rollout(exp, theta, CONSTRAINT_ROLLOUT_STEPS)
    _save_cached_constraint_rollouts(exp, cache_key, rollouts)
    return rollouts


def load_teacher_constraint_series(exp: dict) -> np.ndarray | None:
    _, mpc_path = result_paths(exp, "lapanda")
    if exp["model"] == "cartpole":
        snapshots_path = mpc_path.with_suffix(".mpc_snapshots.npz")
        snapshots = load_mpc_snapshots(snapshots_path) if snapshots_path.exists() else []
        if len(snapshots) < 2:
            from experiments.exp2_OCPs.utils import default_teacher_archive_path

            archive_path = default_teacher_archive_path(
                "cartpole",
                build_ocp_problem(exp["problem_bank_model"]).horizon,
            )
            snapshots = load_mpc_snapshots(archive_path)
        values = [float(np.sum(np.asarray(snapshot["controls"], dtype=float).reshape(-1) ** 2)) for snapshot in snapshots]
        return np.asarray(values[:CONSTRAINT_ROLLOUT_STEPS], dtype=float)

    trajectory_path = mpc_path.with_suffix(".trajectory.npy")
    if trajectory_path.exists():
        states = np.asarray(np.load(trajectory_path), dtype=float)
        if states.shape[0] <= 1:
            return None
        states = states[1 : CONSTRAINT_ROLLOUT_STEPS + 1]
    else:
        snapshots_path = mpc_path.with_suffix(".mpc_snapshots.npz")
        snapshots = load_mpc_snapshots(snapshots_path) if snapshots_path.exists() else []
        next_states = []
        for snapshot in snapshots[:CONSTRAINT_ROLLOUT_STEPS]:
            planned_states = np.asarray(snapshot.get("states", []), dtype=float)
            if planned_states.ndim != 2 or planned_states.shape[0] <= 1:
                return None
            next_states.append(planned_states[1])
        if not next_states:
            return None
        states = np.asarray(next_states, dtype=float)
    if exp["model"] == "quadrotor":
        return states[:, 1]
    if exp["model"] == "robot_arm":
        q0 = states[:, 0]
        q1 = states[:, 1]
        ee = np.asarray(
            [
                np.cos(q0) + 0.8 * np.cos(q0 + q1),
                np.sin(q0) + 0.8 * np.sin(q0 + q1),
            ],
            dtype=float,
        ).T
        return np.linalg.norm(ee - np.array([1.30, 0.30]), axis=1)
    return None


def save_figure(fig, stem: str, tight: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if tight and not fig.get_constrained_layout():
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.svg")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_mpc_timing(ax, exp: dict, metric: str, show_legend: bool) -> None:
    alm_rows = load_method_rows(exp, "lapanda", "mpc")
    safepdp_rows = load_method_rows(exp, "SafePDP", "mpc")
    turbo_cpu_rows = read_csv(turbompc_mpc_variant_path(exp, "cpu"))
    turbo_gpu_rows = read_csv(turbompc_mpc_variant_path(exp, "gpu"))
    if not alm_rows and not safepdp_rows and not turbo_cpu_rows and not turbo_gpu_rows:
        ax.text(0.98, 0.04, "Missing MPC timing", transform=ax.transAxes, ha="right", va="bottom")
        return
    positive_values = []

    def _plot(rows, color, label, zorder, linestyle="-", linewidth=1.05):
        if not rows:
            return
        x = [to_int(row.get("step")) for row in rows]
        y = [to_float(row.get(metric)) for row in rows]
        positive_values.extend([value for value in y if np.isfinite(value) and value > 0.0])
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=zorder)

    _plot(safepdp_rows, COLORS["safepdp_coc"], "SafePDP", zorder=5, linestyle=SAFEPDP_DASH, linewidth=1.05)
    _plot(alm_rows, COLORS["alm"], "lapanda", zorder=4, linestyle="-", linewidth=1.55)
    _plot(turbo_cpu_rows, COLORS["turbo_cpu"], "TurboMPC CPU", zorder=8, linestyle=(0, (3.0, 1.6, 0.8, 1.6)), linewidth=1.10)
    _plot(turbo_gpu_rows, COLORS["turbo_gpu"], "TurboMPC GPU", zorder=7, linestyle=(0, (1.0, 1.45)), linewidth=1.10)
    if positive_values:
        ymin = max(min(positive_values) * 0.7, 1e-5)
        ymax = max(positive_values) * 1.45
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 25, 50])
    ax.margins(x=0.0)
    if show_legend:
        legend = ax.legend(
            frameon=True,
            fancybox=True,
            framealpha=OCP_LEGEND_FRAME_ALPHA,
            facecolor=OCP_LEGEND_FACE,
            edgecolor=OCP_LEGEND_EDGE,
            fontsize=7.0,
            loc="upper right",
            borderpad=0.28,
            labelspacing=0.20,
            handlelength=1.65,
            handletextpad=0.44,
        )
        legend.get_frame().set_linewidth(0.50)


def plot_loss(ax, exp: dict) -> None:
    alm_rows = load_method_rows(exp, "lapanda", "imitation")
    safepdp_coc_rows = load_method_rows(exp, "SafePDP-COC", "imitation")
    turbompc_rows = sorted(read_csv(turbompc_closed_imitation_path(exp)), key=lambda row: to_int(row.get("epoch")))
    if not alm_rows and not safepdp_coc_rows and not turbompc_rows:
        ax.text(0.98, 0.04, "Missing imitation data", transform=ax.transAxes, ha="right", va="bottom")
        return
    def _plot(
        rows,
        color,
        label,
        zorder,
        linestyle,
        marker=None,
        linewidth=1.1,
        markersize=1.45,
        markevery=None,
        alpha=1.0,
    ):
        if not rows:
            return
        rows = [row for row in rows if 0 <= to_int(row.get("epoch")) <= TRAINING_EPOCHS]
        if not rows:
            return
        x = [to_int(row.get("epoch")) for row in rows]
        ax.plot(
            x,
            [to_float(row.get("loss")) for row in rows],
            color=color,
            markersize=markersize,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

    rows_by_key = {
        "safepdp_coc": safepdp_coc_rows,
        "turbompc": turbompc_rows,
        "alm": alm_rows,
    }
    for style in LOSS_STYLES:
        _plot(
            rows_by_key[style["key"]],
            style["color"],
            style["label"],
            zorder=style["zorder"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markevery=style["markevery"],
            alpha=style["alpha"],
        )
    max_epoch = max(
        [
            to_int(row.get("epoch"))
            for row in alm_rows + safepdp_coc_rows + turbompc_rows
            if to_int(row.get("epoch")) >= 0
        ],
        default=TRAINING_EPOCHS,
    )
    max_epoch = min(max_epoch, TRAINING_EPOCHS)
    if exp["model"] == "quadrotor":
        ax.set_ylim(0.0, 55.0)
    elif exp["model"] == "robot_arm":
        ax.set_ylim(0.0, 4.5)
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlim(0, max_epoch)
    ax.set_xticks([0, max_epoch // 2, max_epoch])
    ax.margins(x=0.0)


def plot_timing_figure() -> None:
    use_chart_theme()
    fig, axes = plt.subplots(2, 3, figsize=(ICLR_TEXT_WIDTH_IN, 3.35), sharex=False)
    fig.suptitle(
        "MPC rollout timing comparison: lapanda vs SafePDP",
        fontsize=7.4,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    for col, exp in enumerate(EXPERIMENTS):
        axes[0, col].set_title(exp["display"], fontsize=6.8, color=TOKENS["ink"])
        plot_mpc_timing(axes[0, col], exp, "forward_time_sec", show_legend=False)
        plot_mpc_timing(axes[1, col], exp, "backward_time_sec", show_legend=False)
        axes[0, col].set_xlabel("MPC time instant")
        axes[1, col].set_xlabel("MPC time instant")
        axes[0, col].set_ylabel("Forward time (s)")
        axes[1, col].set_ylabel("Backward time (s)")
        style_pub_axis(axes[0, col])
        style_pub_axis(axes[1, col])
    legend = fig.legend(
        handles=[
            Line2D([0], [0], color=COLORS["safepdp_coc"], linewidth=1.05, linestyle=SAFEPDP_DASH, label="SafePDP"),
            Line2D([0], [0], color=COLORS["alm"], linewidth=1.55, linestyle="-", label="lapanda"),
            Line2D([0], [0], color=COLORS["turbo_cpu"], linewidth=1.10, linestyle=(0, (3.0, 1.6, 0.8, 1.6)), label="TurboMPC CPU"),
            Line2D([0], [0], color=COLORS["turbo_gpu"], linewidth=1.10, linestyle=(0, (1.0, 1.45)), label="TurboMPC GPU"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.887),
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=OCP_LEGEND_FRAME_ALPHA,
        facecolor=OCP_LEGEND_FACE,
        edgecolor=OCP_LEGEND_EDGE,
        fontsize=6.7,
        borderpad=0.25,
        labelspacing=0.18,
        handlelength=1.55,
        handletextpad=0.42,
        columnspacing=0.85,
    )
    legend.get_frame().set_linewidth(0.50)
    save_figure(fig, "paper_safepdp_timing_comparison")


def plot_loss_figure() -> None:
    recovered_path = OUT_DIR / "fig7_barrier_curves_recovered.npz"
    if not recovered_path.exists():
        raise FileNotFoundError(
            f"missing recovered SafePDP-barrier curves: {recovered_path}"
        )

    use_chart_theme()
    recovered = np.load(recovered_path)
    fig, axes = plt.subplots(1, 3, figsize=(ICLR_TEXT_WIDTH_IN, 1.56), dpi=300)
    panel_labels = ["(a) CartPole", "(b) Quadrotor", "(c) Robot arm"]
    y_limits = [(0.0, 20.0), (0.0, 500.0), (0.0, 25.0)]
    y_ticks = [[0, 5, 10, 15, 20], [0, 100, 200, 300, 400, 500], [0, 5, 10, 15, 20, 25]]

    for col, ax in enumerate(axes):
        for color_name, color, alpha in (
            ("blue", COLORS["alm"], 0.16),
            ("red", COLORS["safepdp_coc"], 0.16),
        ):
            ax.fill(
                recovered[f"p{col}_{color_name}_fill_x"],
                recovered[f"p{col}_{color_name}_fill_y"],
                color=color,
                alpha=alpha,
                linewidth=0.0,
                zorder=1,
            )

        ax.plot(
            recovered[f"p{col}_blue_x"],
            recovered[f"p{col}_blue_y"],
            color=COLORS["alm"],
            linestyle="-",
            linewidth=1.20,
            label="lapanda",
            zorder=4,
        )
        ax.plot(
            recovered[f"p{col}_red_x"],
            recovered[f"p{col}_red_y"],
            color=COLORS["safepdp_coc"],
            linestyle="-",
            linewidth=1.00,
            label="SafePDP barrier",
            zorder=3,
        )
        ax.set_xlim(0, 800)
        ax.set_ylim(*y_limits[col])
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_yticks(y_ticks[col])
        ax.set_xlabel("Epoch", fontsize=7.0, labelpad=1.5)
        if col == 0:
            ax.set_ylabel("Imitation loss", fontsize=7.0, labelpad=1.5)
        ax.set_facecolor(OCP_PANEL_FACE)
        ax.grid(True, color="#C6C6C6", linestyle="-", linewidth=0.45, alpha=0.42)
        ax.tick_params(
            axis="both",
            labelsize=7.0,
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
        for spine in ax.spines.values():
            spine.set_linewidth(0.70)
            spine.set_color("#20232A")
        legend = ax.legend(
            loc="upper right",
            fontsize=7.0,
            frameon=True,
            fancybox=True,
            framealpha=OCP_LEGEND_FRAME_ALPHA,
            facecolor=OCP_LEGEND_FACE,
            edgecolor=OCP_LEGEND_EDGE,
            borderpad=0.26,
            labelspacing=0.18,
            handlelength=1.55,
            handletextpad=0.42,
        )
        legend.get_frame().set_linewidth(0.50)
        ax.text(
            0.5,
            -0.36,
            panel_labels[col],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            fontfamily="Times New Roman",
            color=TOKENS["ink"],
        )

    fig.subplots_adjust(left=0.072, right=0.985, top=0.965, bottom=0.315, wspace=0.23)
    save_figure(fig, "paper_safepdp_loss_comparison", tight=False)


def _legend_epochs(ax, epochs: list[int]) -> None:
    return None


def epoch_colorbar_ticks() -> list[int]:
    return [min(CONSTRAINT_EPOCHS), TRAINING_EPOCHS // 2, max(CONSTRAINT_EPOCHS)]


def constraint_epoch_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=CONSTRAINT_HISTORY_CMAP(0.05), linewidth=1.00, alpha=0.70, label="epoch: 0"),
        Line2D([0], [0], color=CONSTRAINT_FINAL_COLOR, linewidth=1.00, label=f"epoch: {TRAINING_EPOCHS}"),
    ]


def constraint_epoch_style(epoch: int) -> dict:
    if epoch == max(CONSTRAINT_EPOCHS):
        return {"color": CONSTRAINT_FINAL_COLOR, "linewidth": 1.20, "alpha": 1.0, "zorder": 12}
    denom = 200.0
    t = min(max((epoch - min(CONSTRAINT_EPOCHS)) / denom, 0.0), 1.0)
    return {
        "color": CONSTRAINT_HISTORY_CMAP(0.05 + 0.90 * t),
        "linewidth": 0.68,
        "alpha": 0.78,
        "zorder": 3,
    }


def _snapshot_theta(snapshot: dict, instance) -> np.ndarray:
    theta = parse_vector(snapshot.get("theta"), instance.theta_true)
    if theta.size != instance.theta_true.size:
        return np.asarray(instance.theta_true, dtype=float)
    return theta


def _cartpole_constraint_axes(
    fig,
    subgs,
    rollouts_by_epoch: dict[int, dict],
    instance,
    target_series=None,
    target_dash=TARGET_DASH,
    target_dash_capstyle="round",
) -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]

    ax_e = fig.add_subplot(subgs)
    budget = CARTPOLE_ENERGY_BUDGET
    ax_e.axhspan(0.0, budget, color="#FFFFFF", alpha=1.0, zorder=0)
    ax_e.axhline(0.0, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    ax_e.axhline(budget, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_e.plot(
            x_target,
            target_series,
            color=TARGET_COLOR,
            linestyle=target_dash,
            linewidth=1.20,
            alpha=TARGET_ALPHA,
            label="target",
            zorder=20,
            dash_capstyle=target_dash_capstyle,
        )
    for epoch in epochs:
        energy = np.asarray(rollouts_by_epoch[epoch]["predicted_energy"], dtype=float).reshape(-1)
        x_e = np.arange(1, energy.size + 1)
        ax_e.plot(x_e, energy, label=f"epoch {epoch}", **constraint_epoch_style(epoch))
    ax_e.set_title("Cartpole control energy", fontsize=7)
    ax_e.set_ylabel("Control energy")
    ax_e.set_xlabel("Closed-loop rollout step")
    ax_e.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_e.set_xticks([0, 25, 50])
    ax_e.set_ylim(-4.0, budget * 1.08)
    ax_e.margins(x=0.0)
    return [ax_e]


def _quadrotor_constraint_axes(
    fig,
    subgs,
    rollouts_by_epoch: dict[int, dict],
    instance,
    target_series=None,
    target_dash=TARGET_DASH,
    target_dash_capstyle="round",
) -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]

    ax_z = fig.add_subplot(subgs)
    floor_height = QUADROTOR_FLOOR_BASE
    ax_z.axhspan(floor_height, QUADROTOR_CEILING_HEIGHT, color="#FFFFFF", alpha=1.0, zorder=0)
    ax_z.axhline(
        floor_height,
        color=FIXED_BOUNDARY_COLOR,
        linestyle=BOUNDARY_DASH,
        linewidth=1.05,
        alpha=0.95,
        zorder=1,
    )
    ax_z.axhline(
        QUADROTOR_CEILING_HEIGHT,
        color=FIXED_BOUNDARY_COLOR,
        linestyle=BOUNDARY_DASH,
        linewidth=1.05,
        alpha=0.95,
        zorder=1,
    )
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_z.plot(
            x_target,
            target_series,
            color=TARGET_COLOR,
            linestyle=target_dash,
            linewidth=1.20,
            alpha=TARGET_ALPHA,
            label="target",
            zorder=20,
            dash_capstyle=target_dash_capstyle,
        )
    for epoch in epochs:
        states = np.asarray(rollouts_by_epoch[epoch]["states"], dtype=float)
        steps = np.arange(1, states.shape[0])
        z = states[1:, 1]
        style = constraint_epoch_style(epoch)
        ax_z.plot(steps, z, label=f"epoch {epoch}", **style)
    ax_z.set_title("Altitude $z_k$ with fixed upper bound", fontsize=7)
    ax_z.set_ylabel("Altitude")
    ax_z.set_xlabel("Closed-loop rollout step")
    ax_z.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_z.set_xticks([0, 25, 50])
    ax_z.set_yticks([0.6, 0.8, 1.0])
    ax_z.margins(x=0.0)
    return [ax_z]


def _robot_arm_constraint_axes(
    fig,
    subgs,
    rollouts_by_epoch: dict[int, dict],
    instance,
    target_series=None,
    target_dash=TARGET_DASH,
    target_dash_capstyle="round",
) -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]
    obstacle_radii = [
        0.29 + float(_snapshot_theta(rollouts_by_epoch[epoch], instance)[4])
        for epoch in epochs
    ]
    max_dist = max(obstacle_radii)

    def _clearance_for_epoch(epoch: int) -> np.ndarray:
        states = np.asarray(rollouts_by_epoch[epoch]["states"], dtype=float)
        q0 = states[1:, 0]
        q1 = states[1:, 1]
        ee = np.asarray(
            [
                np.cos(q0) + 0.8 * np.cos(q0 + q1),
                np.sin(q0) + 0.8 * np.sin(q0 + q1),
            ],
            dtype=float,
        ).T
        return np.linalg.norm(ee - np.array([1.30, 0.30]), axis=1)

    def _show_clearance_epoch(epoch: int, dist: np.ndarray) -> bool:
        return True

    ax_d = fig.add_subplot(subgs)
    if target_series is not None and target_series.size:
        max_dist = max(max_dist, float(np.max(target_series)))
    for epoch in epochs:
        dist = _clearance_for_epoch(epoch)
        if not _show_clearance_epoch(epoch, dist):
            continue
        max_dist = max(max_dist, float(np.max(dist)))
    ax_d.axhspan(min(obstacle_radii), max(max_dist + 0.05, 1.0), color="#FFFFFF", alpha=1.0, zorder=0)
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_d.plot(
            x_target,
            target_series,
            color=TARGET_COLOR,
            linestyle=target_dash,
            linewidth=1.20,
            alpha=TARGET_ALPHA,
            label="target",
            zorder=20,
            dash_capstyle=target_dash_capstyle,
        )
    for epoch in epochs:
        states = np.asarray(rollouts_by_epoch[epoch]["states"], dtype=float)
        steps = np.arange(1, states.shape[0])
        dist = _clearance_for_epoch(epoch)
        if not _show_clearance_epoch(epoch, dist):
            continue
        style = constraint_epoch_style(epoch)
        ax_d.plot(steps, dist, label=f"epoch {epoch}", **style)
        obstacle_radius = 0.29 + float(_snapshot_theta(rollouts_by_epoch[epoch], instance)[4])
        ax_d.axhline(
            obstacle_radius,
            color=style["color"],
            linestyle="--",
            linewidth=max(style["linewidth"] * 0.75, 0.4),
            alpha=style["alpha"],
            zorder=style["zorder"] - 1,
        )
    ax_d.set_title("Obstacle clearance $d_{ee}$", fontsize=7)
    ax_d.set_ylabel("Clearance")
    ax_d.set_xlabel("Closed-loop rollout step")
    ax_d.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_d.set_xticks([0, 25, 50])
    ax_d.margins(x=0.0)
    return [ax_d]


def plot_constraint_figure() -> None:
    from matplotlib.lines import Line2D

    use_chart_theme()
    fig = plt.figure(figsize=(ICLR_TEXT_WIDTH_IN, 2.45))
    outer = fig.add_gridspec(1, 3, left=0.075, right=0.99, bottom=0.20, top=0.70, wspace=0.36)
    fig.suptitle(
        f"Constraint rollouts by OCP ({CONSTRAINT_ROLLOUT_STEPS} closed-loop steps)",
        fontsize=7.6,
        fontweight="semibold",
        color=TOKENS["ink"],
        y=0.975,
    )
    epoch_colors = dict(zip(CONSTRAINT_EPOCHS, sns.color_palette("viridis", len(CONSTRAINT_EPOCHS))))
    fig.legend(
        [
            Line2D([0], [0], color=epoch_colors[epoch], linewidth=1.8)
            for epoch in CONSTRAINT_EPOCHS
        ],
        [f"epoch {epoch}" for epoch in CONSTRAINT_EPOCHS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=6,
        frameon=False,
        fontsize=5.6,
        handlelength=1.35,
        columnspacing=0.65,
        labelspacing=0.45,
    )

    for col, exp in enumerate(EXPERIMENTS):
        instance = build_ocp_problem(exp["problem_bank_model"])
        rollouts_by_epoch = load_or_build_constraint_rollouts(exp)
        if not rollouts_by_epoch:
            ax = fig.add_subplot(outer[0, col])
            ax.text(0.5, 0.5, "Missing training theta records", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        target_series = load_teacher_constraint_series(exp)
        if exp["model"] == "cartpole":
            axes_list = _cartpole_constraint_axes(fig, outer[0, col], rollouts_by_epoch, instance, target_series)
        elif exp["model"] == "quadrotor":
            axes_list = _quadrotor_constraint_axes(fig, outer[0, col], rollouts_by_epoch, instance, target_series)
        else:
            axes_list = _robot_arm_constraint_axes(fig, outer[0, col], rollouts_by_epoch, instance, target_series)

        first_axis_title = axes_list[0].get_title()
        axes_list[0].set_title(
            f"{exp['display']}  ({CONSTRAINT_SNAPSHOT_METHOD})\n{first_axis_title}",
            fontsize=6.8,
            color=TOKENS["ink"],
            pad=12,
        )
        for ax in axes_list:
            ax.tick_params(axis="both", labelsize=5.8)
            ax.xaxis.label.set_size(6.2)
            ax.yaxis.label.set_size(6.2)

    save_figure(fig, "paper_safepdp_constraint_bands", tight=False)


def plot_ocp_4x3_summary_figure() -> None:
    from matplotlib.lines import Line2D

    use_chart_theme()
    fig = plt.figure(figsize=(ICLR_TEXT_WIDTH_IN, 7.15), dpi=300)
    gs = fig.add_gridspec(
        4,
        3,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.865,
        hspace=0.47,
        wspace=0.28,
    )

    row_labels = [
        "Forward time (s)",
        "Backward time (s)",
        "Imitation loss",
        "",
    ]
    axes_grid = [[None for _ in EXPERIMENTS] for _ in row_labels]

    for col, exp in enumerate(EXPERIMENTS):
        ax_forward = fig.add_subplot(gs[0, col])
        axes_grid[0][col] = ax_forward
        ax_forward.set_title(exp["display"], fontsize=7.2, color=TOKENS["ink"], pad=4)
        plot_mpc_timing(ax_forward, exp, "forward_time_sec", show_legend=False)
        ax_forward.set_xlabel("")
        ax_forward.set_ylabel(row_labels[0] if col == 0 else "")

        ax_backward = fig.add_subplot(gs[1, col])
        axes_grid[1][col] = ax_backward
        plot_mpc_timing(ax_backward, exp, "backward_time_sec", show_legend=False)
        ax_backward.set_xlabel("")
        ax_backward.set_ylabel(row_labels[1] if col == 0 else "")

        ax_loss = fig.add_subplot(gs[2, col])
        axes_grid[2][col] = ax_loss
        plot_loss(ax_loss, exp)
        ax_loss.set_xlabel("")
        ax_loss.set_ylabel(row_labels[2] if col == 0 else "")

        instance = build_ocp_problem(exp["problem_bank_model"])
        rollouts_by_epoch = load_or_build_constraint_rollouts(exp)
        if not rollouts_by_epoch:
            ax_constraint = fig.add_subplot(gs[3, col])
            axes_grid[3][col] = ax_constraint
            ax_constraint.text(
                0.5,
                0.5,
                "Missing training theta records",
                ha="center",
                va="center",
                transform=ax_constraint.transAxes,
            )
            ax_constraint.set_axis_off()
        else:
            target_series = load_teacher_constraint_series(exp)
            if exp["model"] == "cartpole":
                axes_list = _cartpole_constraint_axes(fig, gs[3, col], rollouts_by_epoch, instance, target_series)
            elif exp["model"] == "quadrotor":
                axes_list = _quadrotor_constraint_axes(fig, gs[3, col], rollouts_by_epoch, instance, target_series)
            else:
                axes_list = _robot_arm_constraint_axes(fig, gs[3, col], rollouts_by_epoch, instance, target_series)
            ax_constraint = axes_list[0]
            axes_grid[3][col] = ax_constraint
            ax_constraint.set_title("")

        for ax in [ax_forward, ax_backward, ax_loss, ax_constraint]:
            if ax is None or not ax.axison:
                continue
            style_pub_axis(ax)
            ax.tick_params(axis="both", labelsize=5.8, length=1.8, width=0.6)
            ax.xaxis.label.set_size(6.2)
            ax.yaxis.label.set_size(6.2)

    for col in range(3):
        axes_grid[0][col].set_xlabel("")
        axes_grid[1][col].set_xlabel("MPC time instant", fontsize=6.2)
        axes_grid[2][col].set_xlabel("Epoch", fontsize=6.2)
        if axes_grid[3][col] is not None and axes_grid[3][col].axison:
            axes_grid[3][col].set_xlabel("Closed-loop step", fontsize=6.2)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["alm"],
            linestyle="-",
            linewidth=2.25,
            solid_capstyle="round",
            label="lapanda",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["safepdp_coc"],
            linestyle=SAFEPDP_DASH,
            linewidth=2.25,
            solid_capstyle="round",
            label="SafePDP",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["turbompc"],
            linestyle=(0, (1.2, 1.6)),
            linewidth=2.25,
            solid_capstyle="round",
            label="TurboMPC",
        ),
        Line2D(
            [0],
            [0],
            color=TARGET_COLOR,
            linestyle=BOUNDARY_DASH,
            linewidth=2.0,
            alpha=TARGET_ALPHA,
            solid_capstyle="butt",
            label="target",
        ),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.335, 0.986),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=2.45,
        columnspacing=0.85,
        labelspacing=0.35,
    )

    fig.legend(
        handles=constraint_epoch_handles(),
        loc="upper center",
        bbox_to_anchor=(0.79, 0.986),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=2.0,
        columnspacing=0.8,
        labelspacing=0.3,
    )

    # Subtle separators make the four result rows readable at single-column width.
    for row in range(3):
        upper_row = axes_grid[row][0].get_position()
        lower_row = axes_grid[row + 1][0].get_position()
        y = lower_row.y1 + 0.010
        fig.add_artist(
            Line2D(
                [0.075, 0.995],
                [y, y],
                transform=fig.transFigure,
                color="#E1E4EC",
                linewidth=0.55,
                alpha=0.95,
            )
        )

    save_figure(fig, "paper_ocp_4x3_summary", tight=False)


def plot_ocp_1x6_loss_constraints_figure() -> None:
    from matplotlib.lines import Line2D

    use_chart_theme()
    fig = plt.figure(figsize=(ICLR_TEXT_WIDTH_IN, 1.45), dpi=300)
    gs = fig.add_gridspec(
        1,
        6,
        left=0.055,
        right=0.970,
        bottom=0.360,
        top=0.940,
        wspace=0.60,
        width_ratios=[1.0, 1.05, 1.0, 1.05, 1.0, 1.05],
    )

    axes = []
    pair_axes = []
    for col, exp in enumerate(EXPERIMENTS):
        loss_ax = fig.add_subplot(gs[0, 2 * col])
        plot_loss(loss_ax, exp)
        loss_ax.set_title("")
        loss_ax.set_xlabel("Epoch", fontsize=OCP_1X6_LABEL_FONTSIZE)
        loss_ax.set_ylabel("Imitation loss", fontsize=OCP_1X6_LABEL_FONTSIZE)
        axes.append(loss_ax)

        instance = build_ocp_problem(exp["problem_bank_model"])
        rollouts_by_epoch = load_or_build_constraint_rollouts(exp)
        if not rollouts_by_epoch:
            constraint_ax = fig.add_subplot(gs[0, 2 * col + 1])
            constraint_ax.text(
                0.5,
                0.5,
                "Missing rollouts",
                ha="center",
                va="center",
                transform=constraint_ax.transAxes,
                fontsize=OCP_1X6_LABEL_FONTSIZE,
            )
        else:
            target_series = load_teacher_constraint_series(exp)
            if exp["model"] == "cartpole":
                axes_list = _cartpole_constraint_axes(
                    fig, gs[0, 2 * col + 1], rollouts_by_epoch, instance, target_series,
                    target_dash=FIG4_TARGET_DASH, target_dash_capstyle="butt",
                )
            elif exp["model"] == "quadrotor":
                axes_list = _quadrotor_constraint_axes(
                    fig, gs[0, 2 * col + 1], rollouts_by_epoch, instance, target_series,
                    target_dash=FIG4_TARGET_DASH, target_dash_capstyle="butt",
                )
            else:
                axes_list = _robot_arm_constraint_axes(
                    fig, gs[0, 2 * col + 1], rollouts_by_epoch, instance, target_series,
                    target_dash=FIG4_TARGET_DASH, target_dash_capstyle="butt",
                )
            constraint_ax = axes_list[0]
        constraint_ax.set_title("")
        constraint_ax.set_xlabel("Closed-loop step", fontsize=OCP_1X6_LABEL_FONTSIZE)
        if exp["model"] == "cartpole":
            constraint_ax.set_ylabel("Control energy", fontsize=OCP_1X6_LABEL_FONTSIZE)
        elif exp["model"] == "quadrotor":
            constraint_ax.set_ylabel("Altitude", fontsize=OCP_1X6_LABEL_FONTSIZE)
        else:
            constraint_ax.set_ylabel("Clearance", fontsize=OCP_1X6_LABEL_FONTSIZE)
        axes.append(constraint_ax)
        pair_axes.append((loss_ax, constraint_ax))

    y_axis_specs = [
        ((0.0, 5.0), [0, 2.5, 5.0]),
        ((-25.0, 125.0), [-25, 25, 75, 125]),
        ((0.0, 50.0), [0, 25, 50]),
        ((0.2, 1.80), [0.2, 0.6, 1.0, 1.4, 1.8]),
        ((0.0, 4.0), [0, 2, 4]),
        ((0.3, 0.9), [0.3, 0.5, 0.7, 0.9]),
    ]
    for ax, (ylim, yticks) in zip(axes, y_axis_specs):
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)

    for ax in axes:
        if ax is None or not ax.axison:
            continue
        style_pub_axis(ax)
        ax.tick_params(
            axis="both",
            labelsize=OCP_1X6_TICK_FONTSIZE,
            top=False,
            right=False,
            bottom=True,
            left=True,
            direction="out",
            length=2.0,
            width=0.65,
            color="#20232A",
            pad=1.3,
        )
        ymin, ymax = ax.get_ylim()
        yticks = ax.get_yticks()
        has_fractional_ticks = any(not np.isclose(tick, round(tick)) for tick in yticks)
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f" if has_fractional_ticks or (ymax - ymin) <= 5.0 else "%.0f")
        )
        ax.xaxis.label.set_size(OCP_1X6_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(OCP_1X6_LABEL_FONTSIZE)
        ax.yaxis.labelpad = 1.0
        ax.title.set_size(OCP_1X6_LABEL_FONTSIZE)

    def loss_legend_handles(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = {label: handle for handle, label in zip(handles, labels)}
        ordered_labels = [label for label in ("lapanda", "SafePDP", "TurboMPC") if label in by_label]
        return [by_label[label] for label in ordered_labels], ordered_labels
    target_handle = Line2D(
        [0],
        [0],
        color=TARGET_COLOR,
        linestyle=FIG4_TARGET_DASH,
        linewidth=1.00,
        alpha=TARGET_ALPHA,
        dash_capstyle="butt",
        solid_capstyle="butt",
        label="target",
    )
    for idx, (loss_ax, constraint_ax) in enumerate(pair_axes):
        loss_handles, loss_labels = loss_legend_handles(loss_ax)
        loss_legend = loss_ax.legend(
            handles=loss_handles,
            labels=loss_labels,
            loc="upper center",
            frameon=True,
            fancybox=True,
            framealpha=OCP_LEGEND_FRAME_ALPHA,
            facecolor=OCP_LEGEND_FACE,
            edgecolor=OCP_LEGEND_EDGE,
            fontsize=OCP_1X6_LOSS_LEGEND_FONTSIZE,
            handlelength=2.10,
            handletextpad=0.42,
            borderpad=0.24,
            labelspacing=0.16,
        )
        for handle in loss_legend.get_lines():
            handle.set_linewidth(1.00)
        loss_legend.get_frame().set_linewidth(0.50)
        constraint_legend = constraint_ax.legend(
            handles=[*constraint_epoch_handles(), target_handle],
            loc="upper center",
            frameon=True,
            fancybox=True,
            framealpha=OCP_LEGEND_FRAME_ALPHA,
            facecolor=OCP_LEGEND_FACE,
            edgecolor=OCP_LEGEND_EDGE,
            fontsize=OCP_1X6_CONSTRAINT_LEGEND_FONTSIZE,
            handlelength=1.55,
            handletextpad=0.42,
            borderpad=0.22,
            labelspacing=0.18,
        )
        for handle in constraint_legend.get_lines():
            handle.set_linewidth(1.00)
        constraint_legend.get_frame().set_linewidth(0.50)

    panel_labels = ["(a) Cartpole", "(b) Quadrotor", "(c) Robot arm"]
    for label, (left_ax, right_ax) in zip(panel_labels, pair_axes):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        x = 0.5 * (left_box.x0 + right_box.x1)
        fig.text(
            x,
            0.105,
            label,
            ha="center",
            va="top",
            fontsize=OCP_1X6_PANEL_LABEL_FONTSIZE,
            fontfamily="Times New Roman",
            color=TOKENS["ink"],
        )

    save_figure(fig, "paper_ocp_1x6_loss_constraints", tight=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run Exp.2 lapanda and SafePDP scripts before collecting.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.run:
        run_pipeline()
    write_safepdp_vs_alm_table()
    if not args.no_plots:
        plot_timing_figure()
        plot_constraint_figure()
        plot_ocp_1x6_loss_constraints_figure()
        plot_loss_figure()

    print(f"wrote {OUT_DIR / 'paper_safepdp_vs_alm_table.csv'}")
    if not args.no_plots:
        print(f"wrote {OUT_DIR / 'paper_safepdp_timing_comparison.png'}")
        print(f"wrote {OUT_DIR / 'paper_safepdp_constraint_bands.png'}")
        print(f"wrote {OUT_DIR / 'paper_ocp_1x6_loss_constraints.png'}")
        print(f"wrote {OUT_DIR / 'paper_safepdp_loss_comparison.png'}")


if __name__ == "__main__":
    main()
