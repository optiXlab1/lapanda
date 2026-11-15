from .common import *
import argparse, time, importlib
from threadpoolctl import threadpool_limits
from experiments.exp2_OCPs import utils

class AcadosBackend:

    def __init__(self, model, ins, tol, forward_mode='exact'):
        from .run_acados import build
        self.model = model
        self.ins = ins
        self.N = ins.horizon
        _, self.fwd = build(model, self.N, False, forward_mode=forward_mode)
        _, self.sens = build(model, self.N, True)
        for part in ['stat', 'eq', 'ineq', 'comp']:
            self.fwd.options_set('tol_' + part, float(tol))
            self.fwd.options_set('qp_tol_' + part, float(tol))
        self.template = self.fwd.store_iterate_to_flat_obj()
        self.last_status = 0
        self.last_wall = 0.0
        self.last_backward_status = 0

    def solve(self, u0, theta, var, multiplier0=None, **kwargs):
        ins = self.ins
        N = self.N
        f = self.fwd
        s = self.sens
        nu = ins.nu
        nx = ins.nx
        theta = flat(theta)
        var = flat(var)
        u0 = flat(u0)
        x0 = var[:nx]
        target = var[nx:2 * nx]
        du = var[2 * nx:2 * nx + N * nu]
        dx = var[2 * nx + N * nu:].reshape(N, nx)
        started = time.perf_counter()
        f.reset()
        pg = np.r_[theta, target]
        f.set_p_global_and_precompute_dependencies(pg)
        s.set_p_global_and_precompute_dependencies(pg)
        for side in ['lbx', 'ubx']:
            f.constraints_set(0, side, np.r_[x0, 0.0] if self.model == 'cartpole' else x0)
            s.constraints_set(0, side, np.r_[x0, 0.0] if self.model == 'cartpole' else x0)
        states = np.asarray(ins.rollout_fun(u0, theta, var))
        aug = np.r_[0.0, np.cumsum(u0 ** 2)] if self.model == 'cartpole' else None
        it = self.template
        it.x = np.column_stack([states, aug]).reshape(-1) if aug is not None else states.reshape(-1)
        it.u = u0.copy()
        it.pi = np.zeros_like(it.pi)
        it.lam = np.zeros_like(it.lam)
        if multiplier0 is not None:
            m = flat(multiplier0)
            assert m.size == it.pi.size + it.lam.size
            it.pi = m[:it.pi.size].copy()
            it.lam = m[it.pi.size:].copy()
        f.load_iterate_from_flat_obj(it)
        ts = time.perf_counter()
        status = f.solve()
        ft = time.perf_counter() - ts
        sol = f.store_iterate_to_flat_obj()
        u = sol.u.copy()
        xs = sol.x.reshape(N + 1, -1)[:, :nx]
        seeds_u = [(k, (u[k * nu:(k + 1) * nu] - du[k * nu:(k + 1) * nu]).reshape(-1, 1)) for k in range(N)]
        seeds_x = [(k, np.r_[xs[k] - dx[k - 1], 0.0].reshape(-1, 1) if self.model == 'cartpole' else (xs[k] - dx[k - 1]).reshape(-1, 1)) for k in range(1, N + 1)]
        ts = time.perf_counter()
        s.load_iterate_from_flat_obj(sol)
        setup = s.setup_qp_matrices_and_factorize()
        g = flat(s.eval_adjoint_solution_sensitivity(seed_x=seeds_x, seed_u=seeds_u, with_respect_to='p_global', sanity_checks=True))[:theta.size]
        bt = time.perf_counter() - ts
        self.last_status = int(status)
        self.last_backward_status = setup
        self.last_wall = time.perf_counter() - started
        if not np.all(np.isfinite(g)):
            raise RuntimeError('Nonfinite acados sensitivity')
        sqp = int(f.get_stats('sqp_iter'))
        qp = np.asarray(f.get_stats('qp_iter')).reshape(-1)
        return dict(solution=u, grad_theta=g, multipliers=np.r_[sol.pi, sol.lam], penalties=np.zeros(ins.constraint_lower.size), forward_time_sec=ft, backward_time_sec=bt, iterations=sqp, inner_iterations=qp, final_residual=float(np.max(f.get_residuals())), backward_iterations=1, backward_solver_used='acados_exact_adjoint', backward_fallback_used=False)
