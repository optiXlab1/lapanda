from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .common import ROOT, RESULTS, RUNS
MODELS = ['cartpole', 'quadrotor', 'robot_arm']
NAMES = ['CartPole', 'Quadrotor', 'Robot arm']
METHODS = ['lapanda-alm', 'lapanda-kkt', 'casadi', 'safepdp', 'acados', 'turbompc-cpu']
LABELS = ['lapanda-ALM', 'lapanda-KKT', 'CasADi/IPOPT', 'SafePDP', 'acados', 'TurboMPC-CPU']

def training(model, protocol, method):
    return RESULTS / 'learning' / model / protocol / method / 'curve.csv'

def plot(learning, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(ROOT))
    from . import plot_helpers as original
    original.use_chart_theme()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42})
    styles = [('#D62728', '-', 1.05), ('#222222', (0, (4.5, 2.8)), 1.65), ('#9467BD', ':', 1.05), ('#1F77B4', '--', 1.05), ('#D89014', '-', 1.05), ('#2CA02C', '-.', 1.05), ('#178F9E', ':', 1.05)]
    fig = plt.figure(figsize=(5.5, 4.5))
    gs = fig.add_gridspec(5, 3, left=0.1, right=0.99, bottom=0.12, top=0.835, height_ratios=[1, 0.5, 1, 0.8, 1.25], hspace=0, wspace=0.42)
    handles = {}
    for i, protocol in enumerate(('open', 'closed')):
        for j, model in enumerate(MODELS):
            ax = fig.add_subplot(gs[2 * i, j])
            original.style_pub_axis(ax)
            for method, label, (color, ls, lw) in zip(METHODS, LABELS, styles):
                part = learning[learning.model.eq(model) & learning.protocol.eq(protocol) & learning.method.eq(method)]
                if part.empty:
                    continue
                path = training(model, protocol, method)
                curve = pd.read_csv(path)
                trace = pd.read_csv(path.with_name('solves.csv'))
                counts = curve.backend_solve_count.cumsum().astype(int).to_numpy()
                assert counts[-1] == len(trace)
                wall = trace.backend_wall_sec.cumsum().to_numpy()[counts - 1]
                if model == 'cartpole' and protocol == 'open' and (method == 'lapanda-alm'):
                    reached = np.flatnonzero(curve.loss.to_numpy() <= float(part.target_loss.iloc[0]))
                    if reached.size:
                        stop = int(reached[0]) + 1
                        curve = curve.iloc[:stop]
                        wall = wall[:stop]
                line, = ax.plot(wall, curve.loss, color=color, ls=ls, lw=lw, label=label, zorder=20 if method == 'lapanda-kkt' else 8 if method == 'lapanda-alm' else 4)
                if method == 'lapanda-kkt':
                    line.set_marker('o')
                    line.set_markevery(np.unique(np.searchsorted(wall, np.geomspace(wall[0], wall[-1], 8))).tolist())
                    line.set_markersize(2.6)
                    line.set_markerfacecolor('white')
                    line.set_markeredgewidth(0.65)
                    line.set_dash_capstyle('butt')
                handles[method] = line
            kkt = [line for line in ax.lines if line.get_label() == 'lapanda-KKT']
            assert not kkt or kkt[0].get_zorder() == 20
            task = learning[learning.model.eq(model) & learning.protocol.eq(protocol)]
            ax.axhline(float(task.target_loss.iloc[0]), color='#555555', ls=(0, (1, 2)), lw=0.65)
            ax.set_xscale('log')
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(NAMES[j], fontsize=8, pad=5)
            ax.set_xlabel('Solver wall time (s)', labelpad=2)
            if j == 0:
                ax.set_ylabel('Open-loop loss' if i == 0 else 'Closed-loop loss', labelpad=2)
    fig.legend([handles[m] for m in METHODS], LABELS, loc='upper center', bbox_to_anchor=(0.55, 0.995), ncol=3, fontsize=6.5, frameon=True, handlelength=2.5, columnspacing=1.0)
    from experiments.exp2_OCPs.problem_bank import build_problem
    for j, model in enumerate(MODELS):
        data = np.load(RESULTS / 'learning/illustrative_rollouts' / f'{model}.npz', allow_pickle=True)
        rollouts = {int(epoch): dict(r) for epoch, r in data['rollouts'].tolist()}
        ins = build_problem(model)
        teacher = data['teacher']
        fn = [original._cartpole_constraint_axes, original._quadrotor_constraint_axes, original._robot_arm_constraint_axes][j]
        ax = fn(fig, gs[4, j], rollouts, ins, teacher)[0]
        original.style_pub_axis(ax)
        ax.set_title('')
        ax.set_xlabel('Closed-loop step', labelpad=2)
        ax.set_xlim(0, 50)
    teacher = Line2D([0], [0], color=original.TARGET_COLOR, linestyle=original.TARGET_DASH, lw=1.2, label='Teacher')
    fig.legend(handles=[*original.constraint_epoch_handles(), teacher], loc='lower center', bbox_to_anchor=(0.55, 0.005), ncol=3, fontsize=6.7, frameon=False, handlelength=2.8)
    output.mkdir(exist_ok=True, parents=True)
    path = output / 'learning.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=0.035)
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUNS / 'figures')
    args = parser.parse_args()
    plot(pd.read_csv(RESULTS / 'learning/summary.csv'), args.output)
