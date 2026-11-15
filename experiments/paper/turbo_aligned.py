from .common import *
import time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
TURBO = Path(os.environ['TURBOMPC_SOURCE_DIR'])
sys.path.insert(0, str(TURBO))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from turbompc.solvers import TurboMPCSolver
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.dynamics.base_dynamics import Dynamics
from turbompc.utils.load_params import load_solver_params

def formulation(model, ins):
    nx = ins.nx + (model == 'cartpole')
    nu = ins.nu
    N = ins.horizon
    dt = ins.dt

    def fk(x):
        return jnp.array([jnp.cos(x[0]) + 0.8 * jnp.cos(x[0] + x[1]), jnp.sin(x[0]) + 0.8 * jnp.sin(x[0] + x[1])])

    def step(x, u):
        if model == 'cartpole':
            sn = jnp.sin(x[2])
            cs = jnp.cos(x[2])
            tmp = (u[0] + 0.1 * 0.5 * x[3] ** 2 * sn) / 1.1
            aa = (9.81 * sn - cs * tmp) / (0.5 * (4 / 3 - 0.1 * cs ** 2 / 1.1))
            pa = tmp - 0.1 * 0.5 * aa * cs / 1.1
            v = x[1] + dt * pa
            w = x[3] + dt * aa
            return jnp.array([x[0] + dt * v, v, x[2] + dt * w, w, x[4] + u[0] ** 2])
        if model == 'quadrotor':
            vx = x[2] - dt * u[0] * jnp.sin(x[4])
            vz = x[3] + dt * (u[0] * jnp.cos(x[4]) - 9.81)
            w = x[5] + 4 * dt * u[1]
            return jnp.array([x[0] + dt * vx, x[1] + dt * vz, vx, vz, x[4] + dt * w, w])
        return x + dt * u

    class Dyn(Dynamics):

        def __init__(self):
            super().__init__(dict(num_states=nx, num_controls=nu, names_states=[f'x{i}' for i in range(nx)], names_controls=[f'u{i}' for i in range(nu)]))

        def state_dot(self, state, control, params=None):
            return (step(state, control) - state) / dt

    def state_cost(x, p):
        th = p['_theta']
        e = x - p['reference_state_trajectory']
        if model == 'cartpole':
            return th[0] * e[0] ** 2 + th[2] * e[1] ** 2 + th[1] * e[2] ** 2 + th[3] * e[3] ** 2
        if model == 'quadrotor':
            return th[0] * (e[0] ** 2 + e[1] ** 2) + th[1] * (e[2] ** 2 + e[3] ** 2) + th[2] * e[4] ** 2 + th[3] * e[5] ** 2
        return th[0] * jnp.sum(e ** 2) + th[1] * jnp.sum((fk(x) - fk(p['reference_state_trajectory'])) ** 2)

    class Problem(OptimalControlProblem):

        def stage_cost(self, state, control, next_state, next_control, params):
            th = params['_theta']
            value = state_cost(state, params)
            if model == 'cartpole':
                return value + th[4] * control[0] ** 2
            if model == 'quadrotor':
                return value + th[4] * (control[0] - 9.81) ** 2 + th[5] * control[1] ** 2
            return value + th[2] * control[0] ** 2 + th[3] * control[1] ** 2

        def terminal_cost(self, state, control, params):
            return {'cartpole': 10.0, 'quadrotor': 15.0, 'robot_arm': 20.0}[model] * state_cost(state, params) + 0.0001 * jnp.sum(control ** 2)

        def step_inequality_constraints(self, state, control, params):
            g, lo, hi = super().step_inequality_constraints(state, control, params)
            idx = params['stage_index']
            if model == 'cartpole':
                extra = jnp.array([state[-1] - 70.0])
                enabled = idx == N
            elif model == 'quadrotor':
                extra = jnp.array([0.43 - state[1], state[1] - 1.2, state[0] - params['_target_x'], state[4] - 0.34, -state[4] - 0.34])
                enabled = idx > 0
            else:
                extra = jnp.array([(0.29 + params['_theta'][4]) ** 2 - jnp.sum((fk(state) - jnp.array([1.3, 0.3])) ** 2), state[0] - 0.75, -1.1 - state[0], state[1] - 1.28, -0.65 - state[1]])
                enabled = idx > 0
            extra = jnp.where(enabled, extra, -jnp.ones_like(extra))
            return (jnp.r_[g, extra], jnp.r_[lo, -1e+20 * jnp.ones_like(extra)], jnp.r_[hi, jnp.zeros_like(extra)])
    target = np.r_[ins.x_target, 0.0] if model == 'cartpole' else ins.x_target
    initial = np.r_[ins.x_initial, 0.0] if model == 'cartpole' else ins.x_initial
    params = dict(horizon=N, discretization_resolution=dt, discretization_scheme=0, initial_state=jnp.array(initial), initial_guess_final_state=jnp.array(target), constrain_initial_control=False, initial_control=jnp.zeros(nu), reference_state_trajectory=jnp.tile(jnp.array(target), (N + 1, 1)), reference_control_trajectory=jnp.zeros((N + 1, nu)), penalize_control_reference=False, weights_penalization_reference_state_trajectory=jnp.ones(nx), weights_penalization_final_state=jnp.ones(nx), weights_penalization_control_squared=jnp.ones((N + 1, nu)), weights_penalization_control_rate=jnp.zeros(nu), state_min_bounds=jnp.full(nx, -jnp.inf), state_max_bounds=jnp.full(nx, jnp.inf), control_min_bounds=jnp.array(ins.problem.box_lower[:nu]), control_max_bounds=jnp.array(ins.problem.box_upper[:nu]), rescale_optimization_variables=False, state_rescaling_min=-jnp.ones(nx), state_rescaling_max=jnp.ones(nx), control_rescaling_min=-jnp.ones(nu), control_rescaling_max=jnp.ones(nu), dynamics_state_dot_params={}, _theta=jnp.array(ins.theta_initial), stage_index=jnp.arange(N + 1), _target_x=jnp.array(ins.x_target[0]))
    return (Problem(Dyn(), params), params, step)

class TurboBackend:

    def __init__(self, model, ins, tolerance=0.001, linesearch=False, forward_backend='admm_jax_loop_pcg', backward_backend='admm_jax_loop_pcg', backward_tolerance=None, pcg_tolerance=None):
        self.ins = ins
        self.model = model
        self.tol = tolerance
        self.backward_name = backward_backend
        self.problem, self.base, self.step = formulation(model, ins)
        cfg = load_solver_params('turbompc.yaml')
        cfg['num_sqp_iteration_max'] = 50
        cfg['tol_convergence'] = tolerance
        cfg['convergence_criterion'] = 'first_order'
        cfg['linesearch'] = linesearch
        cfg['admm']['max_iter'] = 1000
        cfg['admm']['eps_abs'] = tolerance
        cfg['admm']['eps_rel'] = tolerance
        cfg['admm'].setdefault('pcg', {})['max_iter'] = 1000
        if pcg_tolerance is not None:
            cfg['admm']['pcg']['tol_epsilon'] = float(pcg_tolerance)
        self.solver = TurboMPCSolver(self.problem, cfg, forward_backend=forward_backend, backward_backend=backward_backend, use_full_hessian=True)
        if backward_tolerance is not None:
            bwd = self.solver._admm_solver_bwd
            if bwd is not None:
                bwd._eps_abs = float(backward_tolerance)
                bwd._eps_rel = float(backward_tolerance)
                schur = self.solver._schur_solver_bwd
                if hasattr(schur, '_pcg'):
                    assert schur._pcg is not self.solver._schur_solver_fwd._pcg
                    schur._pcg._params = dict(schur._pcg.params, tol_epsilon=float(backward_tolerance))
            assert self.solver._admm_solver_fwd.eps_abs == tolerance
        self.precision_settings = dict(forward=tolerance, forward_pcg_epsilon=cfg['admm']['pcg']['tol_epsilon'], backward=backward_tolerance, backward_backend=backward_backend)
        self.template = self.solver.initial_guess(self.base)
        self.admm_flat, self.admm_unravel = ravel_pytree(self.template.admm_state)
        self.fwd = jax.jit(lambda guess, params, theta: self.solver.solve(guess, params, {'_theta': theta}))

        def backward(sol, params, theta, sx, su):
            s = self.solver
            weights = {'_theta': theta}
            p = s.make_params_with_weights(weights, params)
            slack = jnp.zeros_like(sol.slack)
            qp = s._build_backward_qp(sol, sx, su, slack, p)
            b = s._solve_backward_qp_direct(qp, sol, slack) if s._uses_direct_backward() else s._solve_backward_qp_admm(qp, sol, slack, p)
            flatg = s._compute_weight_gradient(sol, p, weights, b)
            return s._unflatten_weight_gradient(weights, flatg, b.dL_dgamma)['_theta']
        self.bwd = jax.jit(backward)
        self.last_status = None
        self.last_backward_status = None
        self.prepared = False
        self.compile_seconds = 0.0

    def solve(self, u0, theta, var, multiplier0=None, **unused):
        ins = self.ins
        N = ins.horizon
        nu = ins.nu
        nx = ins.nx
        var = flat(var)
        theta = jnp.asarray(theta)
        params = dict(self.base)
        x0 = np.r_[var[:nx], 0.0] if self.model == 'cartpole' else var[:nx]
        params['initial_state'] = jnp.array(x0)
        target = np.r_[var[nx:2 * nx], 0.0] if self.model == 'cartpole' else var[nx:2 * nx]
        params['reference_state_trajectory'] = jnp.tile(jnp.array(target), (N + 1, 1))
        params['_target_x'] = jnp.array(var[nx])
        states = np.asarray(ins.rollout_fun(u0, np.asarray(theta), var))
        us = flat(u0).reshape(N, nu)
        if self.model == 'cartpole':
            states = np.column_stack([states, np.r_[0.0, np.cumsum(us[:, 0] ** 2)]])
        guess = self.template._replace(states=jnp.array(states), controls=jnp.vstack([jnp.array(us), jnp.zeros((1, nu))]))
        if multiplier0 is not None and len(multiplier0):
            guess = guess._replace(admm_state=self.admm_unravel(jnp.array(multiplier0)))
        ts = time.perf_counter()
        sol = self.fwd(guess, params, theta)
        jax.block_until_ready(sol)
        ft = time.perf_counter() - ts
        du = var[2 * nx:2 * nx + N * nu].reshape(N, nu)
        dx = var[2 * nx + N * nu:].reshape(N, nx)
        ts = time.perf_counter()
        sx = jnp.zeros_like(sol.states).at[1:, :nx].set(sol.states[1:, :nx] - jnp.array(dx))
        su = jnp.zeros_like(sol.controls).at[:N].set(sol.controls[:N] - jnp.array(du))
        if getattr(self, 'validation_debug', False):
            self.validation_context = (guess, params, theta, sx, su, sol)
        g = self.bwd(sol, params, theta, sx, su)
        jax.block_until_ready(g)
        bt = time.perf_counter() - ts
        if not self.prepared:
            self.compile_seconds = ft + bt
            self.prepared = True
        conv = float(sol.convergence_error)
        self.last_status = 0 if np.isfinite(conv) and conv <= self.tol else 1
        self.last_backward_status = 0 if np.all(np.isfinite(g)) else 1
        return dict(solution=np.asarray(sol.controls[:N]).reshape(-1), grad_theta=np.asarray(g), multipliers=np.asarray(ravel_pytree(sol.admm_state)[0]), penalties=np.zeros(len(ins.constraint_lower)), forward_time_sec=ft, backward_time_sec=bt, iterations=int(sol.num_iter), inner_iterations=np.array([0]), final_residual=conv, backward_iterations=-1, backward_solver_used='turbompc_' + self.backward_name, backward_fallback_used=False)
