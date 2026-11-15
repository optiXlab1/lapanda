from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
from . import common as _native_environment
sys.path.insert(0, str(ROOT))
acados_root = Path(os.environ['ACADOS_SOURCE_DIR'])
sys.path.insert(0, str(acados_root / 'interfaces/acados_template'))
import casadi as ca
import numpy as np
from threadpoolctl import threadpool_limits
import lapanda
from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver
from lapanda._lapanda import solve_lapanda_compiled
import lapanda.codegen as codegen
from acados_template import AcadosOcpSolver
SOURCE = ROOT / 'experiments/exp3_embedded_export/rectangle'
spec = importlib.util.spec_from_file_location('smooth_rectangle_original', SOURCE / 'smoothed_rectangle_problem.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
diag.STEER_LIMIT = 0.7

def flat(value):
    return np.asarray(value, dtype=float).reshape(-1)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding='utf-8')

def write_csv(path, rows):
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def build_O3(work_dir, build_dir, cmake_generator):
    subprocess.run(['cmake', '-S', str(work_dir), '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG'], check=True)
    subprocess.run(['cmake', '--build', str(build_dir), '--config', 'Release'], check=True)

def step(state, control):
    v, steer = control
    return state + np.array([diag.DT * v * np.cos(state[2]), diag.DT * v * np.sin(state[2]), diag.DT * v * np.tan(steer) / diag.WHEELBASE])

def shifted_controls(u):
    return np.r_[u[2:], u[-2:]]

class Problem:

    def __init__(self):
        self.teacher = diag.load_teacher()
        self.p = diag.reduced_problem('smooth', 0.05, diag.TEACHER_MARGINS, self.teacher)
        p = self.p
        self.theta = p['theta_value']
        self.lower = np.tile([0.0, -0.7], diag.HORIZON)
        self.upper = np.tile([1.0, 0.7], diag.HORIZON)
        self.grad_cost = ca.Function('retest_cost_gradient', [p['controls'], p['theta'], p['variable']], [ca.gradient(p['cost'], p['controls'])])
        self.initial = diag.handcrafted_feasible_controls()
        self.initial[1::2] *= -1
        data = self.metrics(self.initial, diag.START, np.zeros(diag.HORIZON))
        assert data['violation'] <= 1e-12, data

    def variables(self, state):
        value = self.p['variable_value'].copy()
        value[:3] = state
        return value

    def metrics(self, u, state, multipliers):
        variables = self.variables(state)
        cost, c, J, states = self.p['diagnostic'](u, self.theta, variables)
        c, J, states = (flat(c), np.asarray(J), np.asarray(states))
        grad = flat(self.grad_cost(u, self.theta, variables)) + J.T @ multipliers
        stationarity = float(np.max(np.abs(u - np.clip(u - grad, self.lower, self.upper))))
        violation = float(max(0.0, np.max(-c), np.max(self.lower - u), np.max(u - self.upper)))
        complementarity = float(np.max(np.abs(c * multipliers)))
        dual_violation = float(max(0.0, np.max(multipliers)))
        bounds = diag.rectangle_bounds(diag.TEACHER_MARGINS)
        a, b, lo, hi = bounds
        inside = np.min(np.c_[states[:, 0] - a, b - states[:, 0], states[:, 1] - lo, hi - states[:, 1]], axis=1)
        return dict(cost=float(cost), violation=violation, projected_stationarity=stationarity, complementarity=complementarity, dual_violation=dual_violation, common_original_kkt=max(violation, stationarity, complementarity, dual_violation), margin_penetration=float(max(0.0, np.max(inside))), min_smooth_constraint=float(np.min(c)), imitation_loss=float(0.5 * np.sum((u - self.teacher) ** 2)))

class LapandaBackend:
    name = 'lapanda'

    def __init__(self, common, cache, inner_tolerance):
        self.common = common
        p = common.p
        cp = CasadiProblem(u=p['controls'], theta=p['theta'], variable=p['variable'], cost=p['cost'], box_lower=common.lower, box_upper=common.upper, constraints=p['constraints'], outer_loss=p['outer_loss'])
        codegen._build_shared_library = build_O3
        begin = time.perf_counter()
        self.solver = build_solver(cp, backend='compiled', name='smooth_rectangle_retest_O3', cache_dir=cache, build=True)
        self.init_sec = time.perf_counter() - begin
        self.meta = self.solver.generated_oracle.metadata
        self.so = SolverOptions()
        self.so.max_iterations = 20000
        self.so.tolerance = inner_tolerance
        self.so.buffer_size = 10
        self.so.max_stable_iter = 80
        self.ao = AlmOptions()
        self.ao.max_iterations = 100
        self.ao.tolerance = 0.0001
        self.ao.initial_penalty = 100.0
        self.ao.penalty_update_factor = 10.0
        self.ao.sufficient_decrease_factor = 0.25
        self.ao.warm_start_inner = True
        self.bo = BackwardOptions()
        self.bo.tolerance = 0.001
        self.bo.max_iterations = 200
        self.bo.linear_solver = 'cg'

    def reset(self):
        pass

    def solve(self, guess, state, incoming, backward=True):
        begin = time.perf_counter()
        self.bo.enable = backward
        variable = self.common.variables(state)
        r = solve_lapanda_compiled(str(self.solver.generated_oracle.library_path), self.meta['n'], self.meta['ntheta'], self.meta['nvar'], self.meta['ncon'], self.common.lower, self.common.upper, guess, self.common.theta, variable, self.common.p['lower'], self.common.p['upper'], self.so, self.ao, self.bo, incoming, np.full(self.meta['ncon'], 100.0))
        if backward:
            assert r.get('subproblem_projection_evaluations') == 1, 'Optimized native CG was not loaded'
        u, lam = (flat(r['solution']), flat(r['multipliers']))
        g = flat(r['grad_theta']) if backward else np.zeros(len(self.common.theta))
        row = dict(forward_ms=1000 * float(r['forward_time_sec']), backward_ms=1000 * float(r.get('backward_time_sec', 0.0)), wall_ms=1000 * (time.perf_counter() - begin), forward_status=r.get('status', 'not_exposed'), backward_status=r.get('backward_status', 'not_exposed'), reported_forward_residual=float(r.get('final_residual', np.nan)), backward_residual=float(r.get('backward_residual', np.nan)), backward_fallback=bool(r.get('backward_fallback_used', False)), backward_solver=str(r.get('backward_solver_used', '')), forward_iterations=int(r.get('iterations', -1)), total_inner_iterations=int(np.sum(r.get('inner_iterations', []))), inner_budget_exhaustions=sum((int(t) >= self.so.max_iterations for t in r.get('inner_iterations', []))), backward_iterations=int(r.get('backward_iterations', -1)), max_forward_penalty=float(np.max(r['penalties'])), reverse_penalty_cap_binds=bool(np.max(r['penalties']) > 1000000.0), gradient_finite=bool(np.all(np.isfinite(g))))
        variables = self.common.variables(state)
        c = flat(self.common.p['diagnostic'](u, self.common.theta, variables)[1])
        penalties = flat(r['penalties'])
        effective_lam = lam + penalties * (c - np.clip(c + lam / penalties, self.common.p['lower'], self.common.p['upper']))
        raw = self.common.metrics(u, state, lam)
        row['incoming_multiplier_projected_stationarity'] = raw['projected_stationarity']
        row.update(self.common.metrics(u, state, effective_lam))
        row['incoming_multipliers'] = lam.tolist()
        row['effective_multipliers'] = effective_lam.tolist()
        row['penalties'] = penalties.tolist()
        return (u, lam, g, row)

class AcadosBackend:
    name = 'acados'

    def __init__(self, common, cache):
        self.common = common
        begin = time.perf_counter()
        solvers = []
        for sens in [False, True]:
            dest = cache / ('sens' if sens else 'fwd')
            dest.mkdir(parents=True, exist_ok=True)
            ocp = diag.build_acados_ocp('smooth', 0.05, sens, dest, 'exact', 'kkt')
            ocp.solver_options.cost_scaling = np.ones(diag.HORIZON + 1)
            ocp.model.con_h_expr_0 = ocp.model.con_h_expr
            ocp.constraints.lh_0 = ocp.constraints.lh.copy()
            ocp.constraints.uh_0 = ocp.constraints.uh.copy()
            ocp.constraints.lbu = np.array([0.0, -0.7])
            ocp.constraints.ubu = np.array([1.0, 0.7])
            ocp.solver_options.tol = 0.0001
            ocp.solver_options.ext_fun_compile_flags = '-O3 -DNDEBUG'
            solver = AcadosOcpSolver(ocp, json_file=str(dest / 'ocp.json'), verbose=False)
            solver.set_p_global_and_precompute_dependencies(common.theta)
            solvers.append(solver)
        self.f, self.b = solvers
        self.set_guess(common.initial, diag.START)
        direct_cost = float(common.p['diagnostic'](common.initial, common.theta, common.variables(diag.START))[0])
        error = abs(self.f.get_cost() - direct_cost) / max(1.0, abs(direct_cost))
        assert error < 1e-10, (self.f.get_cost(), direct_cost, error)
        assert self.f.acados_ocp.dims.nh_0 == 1
        self.init_sec = time.perf_counter() - begin
        self.cost_equivalence = [error]

    def reset(self):
        self.f.reset()
        self.b.reset()

    def set_guess(self, guess, state):
        f = self.f
        for solver in [self.f, self.b]:
            solver.constraints_set(0, 'lbx', state)
            solver.constraints_set(0, 'ubx', state)
        x = state.copy()
        f.set(0, 'x', x)
        for k in range(diag.HORIZON):
            control = guess[2 * k:2 * k + 2]
            f.set(k, 'u', control)
            x = step(x, control)
            f.set(k + 1, 'x', x)

    def solve(self, guess, state, incoming, backward=True):
        begin = time.perf_counter()
        f, b = (self.f, self.b)
        self.set_guess(guess, state)
        start = time.perf_counter()
        status = int(f.solve())
        forward_ms = 1000 * (time.perf_counter() - start)
        u = np.concatenate([flat(f.get(k, 'u')) for k in range(diag.HORIZON)])
        lam = []
        for k in range(diag.HORIZON):
            value = flat(f.get(k, 'lam'))
            half = len(value) // 2
            h_index = half - 1
            lam.append(value[half + h_index] - value[h_index])
        lam = np.array(lam)
        backward_ms = 0.0
        backward_status = 'disabled'
        g = np.zeros(len(self.common.theta))
        if backward:
            start = time.perf_counter()
            seeds = [(k, (u[2 * k:2 * k + 2] - self.common.teacher[2 * k:2 * k + 2]).reshape(2, 1)) for k in range(diag.HORIZON)]
            try:
                iterate = f.store_iterate_to_flat_obj()
                b.load_iterate_from_flat_obj(iterate)
                factor = b.setup_qp_matrices_and_factorize()
                g = flat(b.eval_adjoint_solution_sensitivity(seed_x=None, seed_u=seeds, with_respect_to='p_global', sanity_checks=True))
                backward_status = 'ok' if factor in [None, 0] else f'factor_status_{factor}'
            except Exception as exc:
                backward_status = repr(exc)
                g[:] = np.nan
            backward_ms = 1000 * (time.perf_counter() - start)
        row = dict(forward_ms=forward_ms, backward_ms=backward_ms, wall_ms=1000 * (time.perf_counter() - begin), forward_status=status, backward_status=backward_status, reported_forward_residual=float(np.max(f.get_residuals())), forward_iterations=int(np.asarray(f.get_stats('sqp_iter')).reshape(-1)[0]), gradient_finite=bool(np.all(np.isfinite(g))))
        row.update(self.common.metrics(u, state, lam))
        return (u, None, g, row)

def rollout(backend, common, steps):
    backend.reset()
    state = diag.START.copy()
    u, lam, g, initial = backend.solve(common.initial.copy(), state, None, False)
    states = [state.tolist()]
    executed = [u[:2].tolist()]
    state = step(state, u[:2])
    states.append(state.tolist())
    rows = []
    for index in range(1, steps + 1):
        guess = shifted_controls(u)
        shifted_lam = np.r_[lam[1:], 0.0] if lam is not None else None
        u, lam, g, row = backend.solve(guess, state, shifted_lam, True)
        row.update(step=index, gradient=g.tolist(), state_before=state.tolist(), controls=u.tolist())
        executed.append(u[:2].tolist())
        state = step(state, u[:2])
        states.append(state.tolist())
        row['distance_to_target'] = float(np.linalg.norm(state[:2] - diag.TARGET[:2]))
        rows.append(row)
        if not row['gradient_finite'] or (backend.name == 'acados' and row['forward_status'] != 0):
            break
    return dict(solver=backend.name, initial=initial, rows=rows, states=states, executed_controls=executed, reached_target=any((r['distance_to_target'] <= 0.03 for r in rows)), first_target_step=next((r['step'] for r in rows if r['distance_to_target'] <= 0.03), None), final_distance=float(np.linalg.norm(state[:2] - diag.TARGET[:2])), completed_steps=len(rows))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/smooth_rectangle')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--inner-tolerance', type=float, default=0.001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'script_as_used.py').write_bytes(Path(__file__).read_bytes())
    os.sched_setaffinity(0, {2})
    cache = Path(tempfile.mkdtemp(prefix='lapanda_smooth_rectangle_retest_'))
    common = Problem()
    protocol = dict(scope=__doc__, platform=platform.platform(), python=sys.version, casadi=ca.__version__, numpy=np.__version__, lapanda_package=lapanda.__file__, acados_root=str(acados_root), cpu_affinity=[2], blas_threads=1, n=diag.HORIZON, dt=diag.DT, tau=0.05, steps=args.steps, repetitions=args.repetitions, warmups=args.warmups, requested_forward_tolerance=0.0001, lapanda_inner_tolerance=args.inner_tolerance, lapanda_reverse_tolerance=0.001, original_kkt_diagnostic='Common condensed-control projected stationarity, path feasibility, sign and complementarity. lapanda uses final effective ALM multiplier (incoming plus rho times residual), without the reverse penalty cap. Raw incoming multipliers retained separately.', initial_guess='Previously audited feasible reflected handcrafted arc; verified against shared smooth constraints', initial_solve='Separate forward-only initialization, recorded but excluded from warm-step means', warm_start='Both shift the previous controls, repeating the tail. lapanda shifts incoming multipliers; acados retains native dual warm start.', cost='Original unweighted discrete sum, cost_scaling=ones for acados; equivalence checked at common initial guess outside timing', bounds=dict(speed=[0.0, 1.0], steering=[-0.7, 0.7]), stage_zero='Explicit identical x_next obstacle constraint at all N stages, including acados stage 0', backward='Same 0.5*||u-u_teacher||^2, acados exact-Hessian sensitivity with copy/setup/factor/adjoint counted', timing='Serial alternating solver order; perf_counter full-returned-call wall, native component times also retained. Build/init outside per-call wall. Common quality and equivalence diagnostics outside returned-call timing.', runtime_cache=str(cache), source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), SOURCE / 'smoothed_rectangle_problem.py', SOURCE / 'run_smoothed_mpcc_lapanda_rollout.py', SOURCE / 'run_smoothed_mpcc_acados_rollout.py']})
    dump(args.output / 'protocol.json', protocol)
    with threadpool_limits(1):
        backends = [LapandaBackend(common, cache / 'lapanda', args.inner_tolerance), AcadosBackend(common, cache / 'acados')]
        for warm in range(args.warmups):
            for backend in backends:
                record = rollout(backend, common, args.steps)
                dump(args.output / f'warmup_{warm}_{backend.name}.json', record)
                print('WARMUP', backend.name, record['completed_steps'], record['reached_target'], flush=True)
        records = []
        flattened = []
        for repeat in range(args.repetitions):
            for backend in backends if repeat % 2 == 0 else backends[::-1]:
                record = rollout(backend, common, args.steps)
                record['repeat'] = repeat
                dump(args.output / f'repeat_{repeat}_{backend.name}.json', record)
                records.append(record)
                for row in record['rows']:
                    flattened.append(dict(solver=backend.name, repeat=repeat, **{k: v for k, v in row.items() if not isinstance(v, list)}))
                print('REPEAT', repeat, backend.name, record['completed_steps'], record['reached_target'], 'F/B/wall ms', *[round(float(np.mean([r[k] for r in record['rows']])), 4) for k in ['forward_ms', 'backward_ms', 'wall_ms']], flush=True)
                write_csv(args.output / 'steps.csv', flattened)
        summary = []
        for backend in backends:
            rs = [r for r in records if r['solver'] == backend.name]
            rows = [t for r in rs for t in r['rows']]
            result = dict(solver=backend.name, repeats=len(rs), warm_calls=len(rows), completed_rollouts=sum((r['completed_steps'] == args.steps for r in rs)), target_reached_rollouts=sum((r['reached_target'] for r in rs)), initialization_build_sec=backend.init_sec, initial_forward_mean_ms=float(np.mean([r['initial']['forward_ms'] for r in rs])), initial_wall_mean_ms=float(np.mean([r['initial']['wall_ms'] for r in rs])), final_distance_max=max((r['final_distance'] for r in rs)), common_feasibility_passes=sum((r['violation'] <= 0.0001 for r in rows)), common_kkt_passes=sum((r['common_original_kkt'] <= 0.0001 for r in rows)), max_violation=max((r['violation'] for r in rows)), max_margin_penetration=max((r['margin_penetration'] for r in rows)), median_projected_stationarity=float(np.median([r['projected_stationarity'] for r in rows])), max_projected_stationarity=max((r['projected_stationarity'] for r in rows)), fallback_calls=sum((r.get('backward_fallback', False) for r in rows)), cap_binding_calls=sum((r.get('reverse_penalty_cap_binds', False) for r in rows)), gradient_finite_calls=sum((r['gradient_finite'] for r in rows)))
            for key in ['forward_ms', 'backward_ms', 'wall_ms']:
                values = [r[key] for r in rows]
                result['mean_' + key] = float(np.mean(values))
                result['median_' + key] = float(np.median(values))
                result['p90_' + key] = float(np.percentile(values, 90))
            summary.append(result)
        write_csv(args.output / 'summary.csv', summary)
        dump(args.output / 'complete.json', dict(complete=True, summary=summary))
        print(json.dumps(summary, indent=2), flush=True)
if __name__ == '__main__':
    main()
