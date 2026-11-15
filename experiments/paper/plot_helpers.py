from __future__ import annotations
import ast
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
TOKENS = {'surface': '#FCFCFD', 'panel': '#FFFFFF', 'ink': '#1F2430', 'muted': '#6F768A', 'grid': '#E6E8F0', 'axis': '#D7DBE7'}
OCP_PANEL_FACE = '#F3F3F3'
TRAINING_EPOCHS = 500
CONSTRAINT_EPOCHS = list(range(0, 101, 4)) + list(range(110, TRAINING_EPOCHS + 1, 10))
CONSTRAINT_ROLLOUT_STEPS = 50
QUADROTOR_FLOOR_BASE = 0.43
QUADROTOR_CEILING_HEIGHT = 1.2
CARTPOLE_ENERGY_BUDGET = 70.0
TARGET_COLOR = '#D52627'
TARGET_ALPHA = 0.85
TARGET_DASH = (0, (0.55, 2.35))
BOUNDARY_DASH = (0, (3.3, 1.8))
FIXED_BOUNDARY_COLOR = '#000000'
CONSTRAINT_FINAL_COLOR = '#4068E0'
CONSTRAINT_HISTORY_CMAP = LinearSegmentedColormap.from_list('history_epoch_blues', ['#DCE9FF', '#8FB1EA', '#2B63BF'])

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

def use_chart_theme() -> None:
    sns.set_theme(style='whitegrid', rc={'figure.facecolor': TOKENS['surface'], 'savefig.facecolor': TOKENS['surface'], 'axes.facecolor': TOKENS['panel'], 'axes.edgecolor': '#50545E', 'axes.labelcolor': TOKENS['ink'], 'axes.spines.top': True, 'axes.spines.right': True, 'grid.color': '#C6C6C6', 'grid.linewidth': 0.45, 'grid.linestyle': '-', 'grid.alpha': 0.42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], 'axes.labelsize': 6.5, 'axes.titlesize': 7, 'xtick.labelsize': 5.8, 'ytick.labelsize': 5.8, 'legend.fontsize': 6, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial', 'mathtext.bf': 'Arial'})

def style_pub_axis(ax, xgrid: bool=True) -> None:
    ax.set_facecolor(OCP_PANEL_FACE)
    ax.xaxis.grid(xgrid, which='major')
    ax.yaxis.grid(True, which='major')
    ax.grid(axis='y', color='#C6C6C6', linewidth=0.45, linestyle='-', alpha=0.42)
    if xgrid:
        ax.grid(axis='x', color='#C6C6C6', linewidth=0.45, linestyle='-', alpha=0.42)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#1F1F1F')
        spine.set_linewidth(0.75)
    ax.tick_params(axis='both', labelsize=5.8, top=False, right=False, bottom=True, left=True, direction='out', length=1.8, width=0.6, color='#20232A')

def constraint_epoch_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=CONSTRAINT_HISTORY_CMAP(0.05), linewidth=1.0, alpha=0.7, label='epoch: 0'), Line2D([0], [0], color=CONSTRAINT_FINAL_COLOR, linewidth=1.0, label=f'epoch: {TRAINING_EPOCHS}')]

def constraint_epoch_style(epoch: int) -> dict:
    if epoch == max(CONSTRAINT_EPOCHS):
        return {'color': CONSTRAINT_FINAL_COLOR, 'linewidth': 1.2, 'alpha': 1.0, 'zorder': 12}
    denom = 200.0
    t = min(max((epoch - min(CONSTRAINT_EPOCHS)) / denom, 0.0), 1.0)
    return {'color': CONSTRAINT_HISTORY_CMAP(0.05 + 0.9 * t), 'linewidth': 0.68, 'alpha': 0.78, 'zorder': 3}

def _snapshot_theta(snapshot: dict, instance) -> np.ndarray:
    theta = parse_vector(snapshot.get('theta'), instance.theta_true)
    if theta.size != instance.theta_true.size:
        return np.asarray(instance.theta_true, dtype=float)
    return theta

def _cartpole_constraint_axes(fig, subgs, rollouts_by_epoch: dict[int, dict], instance, target_series=None, target_dash=TARGET_DASH, target_dash_capstyle='round') -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]
    ax_e = fig.add_subplot(subgs)
    budget = CARTPOLE_ENERGY_BUDGET
    ax_e.axhspan(0.0, budget, color='#FFFFFF', alpha=1.0, zorder=0)
    ax_e.axhline(0.0, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    ax_e.axhline(budget, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_e.plot(x_target, target_series, color=TARGET_COLOR, linestyle=target_dash, linewidth=1.2, alpha=TARGET_ALPHA, label='target', zorder=20, dash_capstyle=target_dash_capstyle)
    for epoch in epochs:
        energy = np.asarray(rollouts_by_epoch[epoch]['predicted_energy'], dtype=float).reshape(-1)
        x_e = np.arange(1, energy.size + 1)
        ax_e.plot(x_e, energy, label=f'epoch {epoch}', **constraint_epoch_style(epoch))
    ax_e.set_title('Cartpole control energy', fontsize=7)
    ax_e.set_ylabel('Control energy')
    ax_e.set_xlabel('Closed-loop rollout step')
    ax_e.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_e.set_xticks([0, 25, 50])
    ax_e.set_ylim(-4.0, budget * 1.08)
    ax_e.margins(x=0.0)
    return [ax_e]

def _quadrotor_constraint_axes(fig, subgs, rollouts_by_epoch: dict[int, dict], instance, target_series=None, target_dash=TARGET_DASH, target_dash_capstyle='round') -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]
    ax_z = fig.add_subplot(subgs)
    floor_height = QUADROTOR_FLOOR_BASE
    ax_z.axhspan(floor_height, QUADROTOR_CEILING_HEIGHT, color='#FFFFFF', alpha=1.0, zorder=0)
    ax_z.axhline(floor_height, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    ax_z.axhline(QUADROTOR_CEILING_HEIGHT, color=FIXED_BOUNDARY_COLOR, linestyle=BOUNDARY_DASH, linewidth=1.05, alpha=0.95, zorder=1)
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_z.plot(x_target, target_series, color=TARGET_COLOR, linestyle=target_dash, linewidth=1.2, alpha=TARGET_ALPHA, label='target', zorder=20, dash_capstyle=target_dash_capstyle)
    for epoch in epochs:
        states = np.asarray(rollouts_by_epoch[epoch]['states'], dtype=float)
        steps = np.arange(1, states.shape[0])
        z = states[1:, 1]
        style = constraint_epoch_style(epoch)
        ax_z.plot(steps, z, label=f'epoch {epoch}', **style)
    ax_z.set_title('Altitude $z_k$ with fixed upper bound', fontsize=7)
    ax_z.set_ylabel('Altitude')
    ax_z.set_xlabel('Closed-loop rollout step')
    ax_z.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_z.set_xticks([0, 25, 50])
    ax_z.set_yticks([0.6, 0.8, 1.0])
    ax_z.margins(x=0.0)
    return [ax_z]

def _robot_arm_constraint_axes(fig, subgs, rollouts_by_epoch: dict[int, dict], instance, target_series=None, target_dash=TARGET_DASH, target_dash_capstyle='round') -> None:
    epochs = [epoch for epoch in CONSTRAINT_EPOCHS if epoch in rollouts_by_epoch]
    obstacle_radii = [0.29 + float(_snapshot_theta(rollouts_by_epoch[epoch], instance)[4]) for epoch in epochs]
    max_dist = max(obstacle_radii)

    def _clearance_for_epoch(epoch: int) -> np.ndarray:
        states = np.asarray(rollouts_by_epoch[epoch]['states'], dtype=float)
        q0 = states[1:, 0]
        q1 = states[1:, 1]
        ee = np.asarray([np.cos(q0) + 0.8 * np.cos(q0 + q1), np.sin(q0) + 0.8 * np.sin(q0 + q1)], dtype=float).T
        return np.linalg.norm(ee - np.array([1.3, 0.3]), axis=1)

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
    ax_d.axhspan(min(obstacle_radii), max(max_dist + 0.05, 1.0), color='#FFFFFF', alpha=1.0, zorder=0)
    if target_series is not None and target_series.size:
        x_target = np.arange(1, target_series.size + 1)
        ax_d.plot(x_target, target_series, color=TARGET_COLOR, linestyle=target_dash, linewidth=1.2, alpha=TARGET_ALPHA, label='target', zorder=20, dash_capstyle=target_dash_capstyle)
    for epoch in epochs:
        states = np.asarray(rollouts_by_epoch[epoch]['states'], dtype=float)
        steps = np.arange(1, states.shape[0])
        dist = _clearance_for_epoch(epoch)
        if not _show_clearance_epoch(epoch, dist):
            continue
        style = constraint_epoch_style(epoch)
        ax_d.plot(steps, dist, label=f'epoch {epoch}', **style)
        obstacle_radius = 0.29 + float(_snapshot_theta(rollouts_by_epoch[epoch], instance)[4])
        ax_d.axhline(obstacle_radius, color=style['color'], linestyle='--', linewidth=max(style['linewidth'] * 0.75, 0.4), alpha=style['alpha'], zorder=style['zorder'] - 1)
    ax_d.set_title('Obstacle clearance $d_{ee}$', fontsize=7)
    ax_d.set_ylabel('Clearance')
    ax_d.set_xlabel('Closed-loop rollout step')
    ax_d.set_xlim(0, CONSTRAINT_ROLLOUT_STEPS)
    ax_d.set_xticks([0, 25, 50])
    ax_d.margins(x=0.0)
    return [ax_d]
