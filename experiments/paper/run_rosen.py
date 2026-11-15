import argparse
import json
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits
from lapanda import build_solver
from experiments.exp1_rosenbrock_smooth_constraints.problem import build_problem
from .common import ROOT, HERE, RESULTS, RUNS, np, rel
from .native_direct_backends import NativeDirect
from .rosen_backend import reverse

OUT = RESULTS/'rosenbrock/main'


def main():
    parser = argparse.ArgumentParser(description='Backward solvers at shared Rosenbrock forward points.')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 200, 500, 1000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[107, 108, 109, 110, 111])
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--output', type=Path, default=RUNS/'rosenbrock')
    parser.add_argument('--memory', action='store_true')
    a = parser.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    selected = pd.read_csv(OUT/'selected_tolerances.csv')
    rng = np.random.default_rng(20261111)
    rows = []
    with threadpool_limits(1):
        for n in a.sizes:
            ins = build_problem(n)
            solver = build_solver(ins.problem, backend='compiled', name=f'paper_rosen_{n}', cache_dir=HERE/'cache/native_rosen')
            direct = NativeDirect(ins)
            match = float(selected[selected.n.eq(n)].iloc[0].minres_tolerance)
            methods = [('cg', 1e-2), ('minres_fixed', 1e-2), ('minres_matched', match),
                       ('sparse_lu', 0.), ('dense_ldlt', 0.)]
            for seed in a.seeds:
                with np.load(OUT/'states'/f'n{n}_seed{seed}.npz') as saved:
                    state = {k:saved[k].copy() for k in ['solution','multipliers','penalties','final_gamma']}
                    theta, variable, reference = (saved[k].copy() for k in ['theta','variable','reference'])

                def call(method, tolerance):
                    if method in ['sparse_lu', 'dense_ldlt']:
                        return direct.backward(state, theta, variable, method)
                    return reverse(solver, ins, state, theta, variable,
                        'subproblem' if method == 'cg' else 'original_kkt', tolerance)

                for method, tolerance in methods:
                    call(method, tolerance)
                timings = {method:[] for method,_ in methods}
                last = {}
                for repeat in range(a.repeats):
                    for index in rng.permutation(len(methods)):
                        method, tolerance = methods[index]
                        start = time.perf_counter()
                        gradient, info = call(method, tolerance)
                        elapsed = time.perf_counter()-start
                        timings[method].append(elapsed*1e6)
                        last[method] = (gradient, info)
                for method, tolerance in methods:
                    gradient, info = last[method]
                    record = dict(n=n, seed=seed, method=method, tolerance=tolerance,
                        wall_us=float(np.median(timings[method])),
                        gradient_error_percent=100*rel(gradient, reference),
                        iterations=info.get('iterations',1), dimension=info['dimension'],
                        linear_residual=info.get('linear_residual', info.get('residual')),
                        fallback=info.get('fallback',False))
                    if a.memory:
                        import memray
                        capture = a.output/f'memory_n{n}_seed{seed}_{method}.bin'
                        with memray.Tracker(str(capture), native_traces=True, trace_python_allocators=True):
                            call(method, tolerance)
                        reader = memray.FileReader(str(capture))
                        record['peak_allocation_bytes'] = sum(r.size for r in reader.get_high_watermark_allocation_records(merge_threads=True))
                    rows.append(record)
                pd.DataFrame(rows).to_csv(a.output/'points.csv', index=False)
            print('Completed n =', n, flush=True)
    frame = pd.DataFrame(rows)
    fields = ['wall_us', 'gradient_error_percent', 'iterations', 'dimension']
    if a.memory:fields.append('peak_allocation_bytes')
    summary = frame.groupby(['n','method'])[fields].median().reset_index()
    summary.to_csv(a.output/'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
