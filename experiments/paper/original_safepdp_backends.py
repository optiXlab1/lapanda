from .common import *
import time
import casadi as ca
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import scipy.linalg as la
from experiments.exp2_OCPs import safepdp_baseline as sp

class OriginalSafePDPForward:

    def __init__(self, model, ins, active_tolerance=0.001, tolerance=0.001, cached_symbolic=False):
        self.model = model
        self.ins = ins
        self.coc = sp.build_safepdp_coc(ins)
        self.active_tol = active_tolerance
        self.tolerance = float(tolerance)
        self.last_status = None
        self.last_backward_status = None
        self.forward_policy = dict(implementation='experiments.exp2_OCPs.safepdp_baseline.solve_coc', options={'ipopt.tol': self.tolerance}, numerical_tolerance_override=self.tolerance, formulation='multiple shooting', symbolic_callbacks=True, primal_control_warm_start=True, dual_warm_start=False, nlp_construction='inside every forward call, included in forward wall')
        self.cached_forward = None
        if cached_symbolic:
            from .cached_symbolic_safepdp_forward import CachedSymbolicSafePDPForward
            self.cached_forward = CachedSymbolicSafePDPForward(self.coc, ins.horizon, self.tolerance)
            self.forward_policy.update(implementation='CachedSymbolicSafePDPForward; original multiple-shooting transcription', nlp_construction='once during initialization; separately recorded build wall', generated_code=False, jit=False, build_wall_sec=self.cached_forward.build_wall_sec)

    def forward(self, u0, theta, var):
        ins = self.ins
        state = flat(var)[:ins.nx]
        if self.model == 'cartpole':
            state = np.r_[state, 0.0]
        if self.cached_forward is not None:
            return self.cached_forward.solve(state, theta, u0)
        return sp.solve_coc(self.coc, ins.horizon, state, theta, u0, forward_tolerance=self.tolerance)

    def solve(self, u0, theta, var, multiplier0=None, **unused):
        start = time.perf_counter()
        traj = self.forward(u0, theta, var)
        ft = time.perf_counter() - start
        self.last_trajectory = traj
        stats = traj.get('solver_stats', {})
        self.last_status = (0 if stats['success'] else 1) if 'success' in stats else None
        start = time.perf_counter()
        gradient, reverse_stats = self.backward(traj, flat(theta), flat(var))
        bt = time.perf_counter() - start
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError('Nonfinite original-forward comparator gradient')
        self.last_backward_status = 0
        iterations = stats.get('iter_count', np.nan)
        return dict(solution=np.asarray(traj['control_traj_opt']).reshape(-1), grad_theta=gradient, multipliers=np.zeros(0), penalties=np.zeros(len(self.ins.constraint_lower)), forward_time_sec=ft, backward_time_sec=bt, iterations=iterations, inner_iterations=np.array([iterations]), final_residual=np.nan, backward_iterations=1, backward_solver_used=self.reverse_name, backward_fallback_used=False, **reverse_stats)

class OriginalForwardSafePDP(OriginalSafePDPForward):
    reverse_name = 'submission_safepdp_coc_clqr_original_forward'

    def __init__(self, model, ins, active_tolerance=0.001, tolerance=0.001, reverse_variant='bounds_symmetric', cached_symbolic=False):
        super().__init__(model, ins, active_tolerance, tolerance, cached_symbolic)
        if reverse_variant not in ['bounds_symmetric', 'submission']:
            raise ValueError(reverse_variant)
        self.reverse_variant = reverse_variant
        if reverse_variant == 'bounds_symmetric':
            from .ocp_baselines import safepdp_clqr
            self.clqr = safepdp_clqr()
            self.reverse_name = 'safepdp_coc_bound_aware_symmetric_shared_original_forward'
        else:
            self.clqr = sp.SafePDP.EQCLQR()

    def backward(self, traj, theta, var):
        ins = self.ins
        if self.reverse_variant == 'bounds_symmetric':
            from .ocp_baselines import safepdp_bound_aware_backward
            return (safepdp_bound_aware_backward(ins, self.coc, self.clqr, traj, theta, var, self.active_tol), {})
        demo_u = var[2 * ins.nx:2 * ins.nx + ins.horizon * ins.nu]
        demo_x = var[2 * ins.nx + ins.horizon * ins.nu:].reshape(ins.horizon, ins.nx)
        _, gradient = sp.safepdp_backward(ins, self.coc, self.clqr, traj, demo_u, np.vstack([var[:ins.nx], demo_x]), self.active_tol)
        return (gradient, {})

class OriginalForwardCasadiKKT(OriginalSafePDPForward):
    reverse_name = 'casadi_sparse_multiple_shooting_KKT_original_safepdp_forward'

    def __init__(self, model, ins, active_tolerance=0.001, tolerance=0.001, cached_symbolic=False):
        super().__init__(model, ins, active_tolerance, tolerance, cached_symbolic)
        c = self.coc
        theta = ca.MX.sym('theta', len(ins.theta_true))
        x = ca.MX.sym('x0', c.n_state)
        variables = [x]
        constraints = []
        lower, upper = (list(c.state_lb), list(c.state_ub))
        equalities = []
        cost = 0
        for k in range(ins.horizon):
            u = ca.MX.sym(f'u{k}', c.n_control)
            variables.append(u)
            lower.extend(c.control_lb)
            upper.extend(c.control_ub)
            cost += c.path_cost_fn(x, u, theta)
            if c.n_path_inequ_cstr:
                constraints.append(c.path_inequ_cstr_fn(x, u, theta))
                equalities.extend([False] * c.n_path_inequ_cstr)
            if c.n_path_equ_cstr:
                constraints.append(c.path_equ_cstr_fn(x, u, theta))
                equalities.extend([True] * c.n_path_equ_cstr)
            xn = ca.MX.sym(f'x{k + 1}', c.n_state)
            variables.append(xn)
            lower.extend(c.state_lb)
            upper.extend(c.state_ub)
            constraints.append(c.dyn_fn(x, u, theta) - xn)
            equalities.extend([True] * c.n_state)
            x = xn
        cost += c.final_cost_fn(x, theta)
        if c.n_final_inequ_cstr:
            constraints.append(c.final_inequ_cstr_fn(x, theta))
            equalities.extend([False] * c.n_final_inequ_cstr)
        if c.n_final_equ_cstr:
            constraints.append(c.final_equ_cstr_fn(x, theta))
            equalities.extend([True] * c.n_final_equ_cstr)
        w = ca.vertcat(*variables)
        g = ca.vertcat(*constraints)
        lam = ca.MX.sym('lam', g.numel())
        station = ca.gradient(cost + ca.dot(lam, g), w)
        self.derivatives = ca.Function('original_safe_multiple_shooting_derivatives', [w, theta, lam], [ca.jacobian(station, w), ca.jacobian(g, w), ca.jacobian(station, theta), ca.jacobian(g, theta), g])
        self.lower, self.upper = (np.asarray(lower), np.asarray(upper))
        self.equalities = np.asarray(equalities, dtype=bool)

    def backward(self, traj, theta, var):
        ins, c = (self.ins, self.coc)
        xs, us = (np.asarray(traj['state_traj_opt']), np.asarray(traj['control_traj_opt']))
        primal = np.concatenate([np.r_[xs[k], us[k]] for k in range(ins.horizon)] + [xs[-1]])
        lam = np.concatenate([np.r_[traj['v_path'][k], traj['w_path'][k], traj['costate_traj_opt'][k]] for k in range(ins.horizon)] + [flat(traj['v_final']), flat(traj['w_final'])])
        H, J, B, C, constraints = self.derivatives(primal, theta, lam)
        H, J = (sparse.csc_matrix(H), sparse.csc_matrix(J))
        B, C = (np.asarray(B), np.asarray(C))
        free = (primal > self.lower + self.active_tol) & (primal < self.upper - self.active_tol)
        free[:c.n_state] = False
        free = np.flatnonzero(free)
        active = np.flatnonzero(self.equalities | (flat(constraints) >= -self.active_tol))
        A = J[active][:, free]
        allA = A
        dropped = 0
        if A.shape[0]:
            _, R, piv = la.qr(A.toarray().T, mode='economic', pivoting=True)
            diagonal = abs(np.diag(R))
            rank = int(np.sum(diagonal > 1e-10 * max(1.0, diagonal.max(initial=0.0))))
            if rank < len(active):
                selected = np.sort(piv[:rank])
                dropped = len(active) - rank
                active, A = (active[selected], A[selected])
        q = np.zeros_like(primal)
        demo_u = var[2 * ins.nx:2 * ins.nx + ins.horizon * ins.nu].reshape(ins.horizon, ins.nu)
        demo_x = var[2 * ins.nx + ins.horizon * ins.nu:].reshape(ins.horizon, ins.nx)
        stride = c.n_state + c.n_control
        for k in range(ins.horizon):
            q[k * stride + c.n_state:(k + 1) * stride] = us[k] - demo_u[k]
            q[(k + 1) * stride:(k + 1) * stride + ins.nx] = sp.STATE_IMITATION_WEIGHT * (xs[k + 1, :ins.nx] - demo_x[k])
        K = sparse.bmat([[H[free][:, free], A.T], [A, sparse.csc_matrix((len(active), len(active)))]], format='csc')
        rhs = np.r_[q[free], np.zeros(len(active))]
        adjoint = sla.splu(K).solve(rhs)
        residual = np.linalg.norm(K @ adjoint - rhs) / max(np.linalg.norm(rhs), 1e-300)
        original_residual = np.linalg.norm(np.r_[K @ adjoint - rhs, allA @ adjoint[:len(free)]]) / max(np.linalg.norm(rhs), 1e-300)
        if original_residual > 1e-07:
            raise RuntimeError(f'Original-forward sparse KKT residual {original_residual}')
        gradient = -B[free].T @ adjoint[:len(free)] - C[active].T @ adjoint[len(free):]
        return (gradient, dict(backward_residual=float(residual), original_kkt_residual=float(original_residual), kkt_dimension=K.shape[0], dropped_active_rows=dropped))
