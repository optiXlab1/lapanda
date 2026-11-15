import argparse
import csv
import importlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .common import ROOT, HERE, RUNS, flat, np
from experiments.exp2_OCPs import utils
from threadpoolctl import threadpool_limits

METHODS = ['lapanda-alm', 'lapanda-kkt', 'casadi', 'safepdp', 'acados',
           'turbompc-cpu', 'turbompc-gpu']
TARGETS = {'cartpole': {'open': .5, 'closed': .03},
           'quadrotor': {'open': 190., 'closed': 2.3},
           'robot_arm': {'open': 1., 'closed': .03}}


def main():
    parser = argparse.ArgumentParser(description='Constrained OCP imitation learning.')
    parser.add_argument('--model', required=True, choices=TARGETS)
    parser.add_argument('--protocol', required=True, choices=['open', 'closed'])
    parser.add_argument('--method', default='lapanda-alm', choices=METHODS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--memory-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu', type=int)
    a = parser.parse_args()
    if a.method.startswith('turbompc') and a.model == 'cartpole':
        parser.error('The paper TurboMPC configurations do not support the CartPole energy constraint.')
    if a.cpu is not None and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {a.cpu})
    tolerance = 1e-5 if a.model == 'robot_arm' else 1e-4
    args = SimpleNamespace(**json.loads((HERE/'configs'/f'{a.model}_{a.protocol}.json').read_text()))
    args.teacher_source = str(ROOT/args.teacher_source)
    args.force = False
    args.memory_only = a.memory_only
    args.inner_tol = args.alm_tol = tolerance
    args.backward_tolerance = 1e-3
    args.backward_constraint_penalty_scale = 1.
    args.backward_constraint_penalty_max = 0.
    args.backward_linear_solver = 'minres' if a.method == 'lapanda-kkt' else 'cg'
    args.backward_system = 'original_kkt' if a.method == 'lapanda-kkt' else 'subproblem'
    if a.epochs is not None:
        args.epochs = a.epochs
    output = a.output or RUNS/f'{a.model}_{a.protocol}_{a.method}_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(parents=True, exist_ok=False)
    target = output/'training.csv'
    mod = importlib.import_module(f'experiments.exp2_OCPs.common.{a.protocol}_loop_train')
    mod.output_path = lambda *unused: target
    build_times = []

    def build(ins, model, **unused):
        start = time.perf_counter()
        if a.method.startswith('lapanda'):
            from .native_backend import NativeBackend
            solver = NativeBackend(model, ins, 'c_kkt' if a.method == 'lapanda-kkt' else 'c_final', 1e-3)
        elif a.method in ['casadi', 'safepdp']:
            from .learning_backends import build_learning_backend
            solver = build_learning_backend(a.method, model, ins, tolerance)
        elif a.method == 'acados':
            from .acados_backend import AcadosBackend
            solver = AcadosBackend(model, ins, tolerance, 'exact' if model == 'cartpole' else 'gauss_newton')
        else:
            gpu = a.method.endswith('gpu')
            os.environ['JAX_PLATFORMS'] = 'cuda' if gpu else 'cpu'
            os.environ['JAX_PLATFORM_NAME'] = 'gpu' if gpu else 'cpu'
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            from .turbo_aligned import TurboBackend
            solver = TurboBackend(model, ins, tolerance, False,
                'admm_fused_cudss' if gpu else 'admm_jax_loop_pcg',
                'direct_cudss_ffi' if gpu else 'admm_jax_loop_pcg',
                backward_tolerance=None, pcg_tolerance=None)
            first = utils.load_snapshot_archive(Path(args.teacher_source))[0]
            u0 = flat(first['controls'])
            states = np.asarray(first['states'])
            theta = utils.seeded_theta_initial(ins, args.theta_init, args.seed)
            solver.solve(u0, theta, utils.make_variable_from_state(ins, states[0], u0, states))
        build_times.append(time.perf_counter()-start)
        return solver

    mod.build_lapanda_solver = build
    fields = ['solve', 'status', 'backward_status', 'backend_wall_sec', 'forward_sec',
              'backward_sec', 'violation', 'linear_residual', 'backward_iterations']
    trace = (output/'solves.csv').open('w', newline='', buffering=1)
    writer = csv.DictWriter(trace, fieldnames=fields)
    writer.writeheader()
    samples = []
    batch_wall = []
    batch_failed = []
    count = 0
    snapshots = []
    sample_count = 1 if a.memory_only else (32 if a.protocol == 'open' else args.steps)
    checkpoints = {0, 200, 400, 800} if a.protocol == 'open' else {0, 125, 250, 500}

    def solve(solver, ins, u0, theta, variable, **kwargs):
        nonlocal count
        start = time.perf_counter()
        result = solver.solve(u0, theta, variable, **kwargs)
        wall = time.perf_counter()-start
        status = getattr(solver, 'last_status', None)
        backward_status = getattr(solver, 'last_backward_status', None)
        row = dict(solve=count, status=status, backward_status=backward_status,
                   backend_wall_sec=wall, forward_sec=result['forward_time_sec'],
                   backward_sec=result['backward_time_sec'],
                   violation=utils.violation_inf(ins, result['solution'], theta, variable),
                   linear_residual=result.get('backward_residual', np.nan),
                   backward_iterations=result.get('backward_iterations', np.nan))
        writer.writerow(row)
        samples.append(row)
        batch_wall.append(wall)
        batch_failed.append(status not in [0, None] or backward_status not in [0, None])
        epoch = count//sample_count
        if epoch in checkpoints or epoch == args.epochs:
            snapshots.append(dict(epoch=epoch, sample=count%sample_count,
                warm=flat(u0).copy(), theta=flat(theta).copy(), variable=flat(variable).copy(),
                solution=flat(result['solution']).copy(), gradient=flat(result['grad_theta']).copy(),
                multipliers=flat(result['multipliers']).copy(), penalties=flat(result['penalties']).copy()))
            if (count+1)%sample_count == 0:
                np.savez_compressed(output/f'epoch{epoch}.npz',
                    **{key:np.asarray([r[key] for r in snapshots]) for key in snapshots[0]})
                snapshots.clear()
        count += 1
        if not np.all(np.isfinite(result['grad_theta'])):
            raise RuntimeError('Nonfinite gradient')
        return result

    mod.solve_lapanda = solve
    original_append = mod.append_csv

    def append(path, rows):
        for row in rows:
            row['backend_wall_time_sec'] = sum(batch_wall)
            row['backend_failed_solves'] = sum(batch_failed)
            row['backend_solve_count'] = len(batch_wall)
        original_append(path, rows)
        batch_wall.clear()
        batch_failed.clear()

    mod.append_csv = append
    (output/'settings.json').write_text(json.dumps(dict(model=a.model, protocol=a.protocol,
        method=a.method, forward_tolerance=tolerance, backward_tolerance=1e-3,
        active_tolerance=1e-5, training=vars(args)), indent=2))
    started = time.perf_counter()
    complete = False
    try:
        with threadpool_limits(1):
            getattr(mod, f'run_{a.protocol}_loop_train')(a.model, args)
        complete = True
    finally:
        trace.close()
        summary = dict(complete=complete, solves=count, training_wall_sec=time.perf_counter()-started,
            solver_build_sec=sum(build_times),
            full_solver_wall_sec=sum(r['backend_wall_sec'] for r in samples),
            full_forward_sec=sum(r['forward_sec'] for r in samples),
            full_backward_sec=sum(r['backward_sec'] for r in samples))
        if complete:
            with target.open() as f:
                curve = list(csv.DictReader(f))
            threshold = TARGETS[a.model][a.protocol]
            first = next((i for i,r in enumerate(curve) if float(r['loss']) <= threshold), None)
            summary.update(target_loss=threshold, final_loss=float(curve[-1]['loss']),
                           first_hit_epoch=int(curve[first]['epoch']) if first is not None else None)
            if first is not None:
                calls = sum(int(r['backend_solve_count']) for r in curve[:first+1])
                summary['solver_wall_to_target_sec'] = sum(r['backend_wall_sec'] for r in samples[:calls])
        (output/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
