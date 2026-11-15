import time
import casadi as ca
import numpy as np

class CachedSymbolicSafePDPForward:

    def __init__(self, coc, horizon, tolerance):
        start = time.perf_counter()
        self.coc, self.horizon = (coc, horizon)
        self.options = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False, 'ipopt.tol': float(tolerance)}
        theta = ca.MX.sym('forward_theta', coc.n_auxvar)
        x = ca.MX.sym('X0', coc.n_state)
        variables, constraints = ([x], [])
        lower, upper = (list(coc.state_lb), list(coc.state_ub))
        lbg, ubg, cost = ([], [], 0)
        for k in range(horizon):
            u = ca.MX.sym(f'U_{k}', coc.n_control)
            variables.append(u)
            lower.extend(coc.control_lb)
            upper.extend(coc.control_ub)
            if coc.path_inequ_cstr is not None:
                constraints.append(coc.path_inequ_cstr_fn(x, u, theta))
                lbg.extend([-coc.inf] * coc.n_path_inequ_cstr)
                ubg.extend([0.0] * coc.n_path_inequ_cstr)
            if coc.path_equ_cstr is not None:
                constraints.append(coc.path_equ_cstr_fn(x, u, theta))
                lbg.extend([0.0] * coc.n_path_equ_cstr)
                ubg.extend([0.0] * coc.n_path_equ_cstr)
            xn = ca.MX.sym(f'X_{k + 1}', coc.n_state)
            variables.append(xn)
            lower.extend(coc.state_lb)
            upper.extend(coc.state_ub)
            constraints.append(coc.dyn_fn(x, u, theta) - xn)
            lbg.extend([0.0] * coc.n_state)
            ubg.extend([0.0] * coc.n_state)
            cost += coc.path_cost_fn(x, u, theta)
            x = xn
        if coc.final_inequ_cstr is not None:
            constraints.append(coc.final_inequ_cstr_fn(x, theta))
            lbg.extend([-coc.inf] * coc.n_final_inequ_cstr)
            ubg.extend([0.0] * coc.n_final_inequ_cstr)
        if coc.final_equ_cstr is not None:
            constraints.append(coc.final_equ_cstr_fn(x, theta))
            lbg.extend([0.0] * coc.n_final_equ_cstr)
            ubg.extend([0.0] * coc.n_final_equ_cstr)
        cost += coc.final_cost_fn(x, theta)
        self.solver = ca.nlpsol('cached_symbolic_safe_forward', 'ipopt', dict(x=ca.vertcat(*variables), p=theta, f=cost, g=ca.vertcat(*constraints)), self.options)
        self.lower, self.upper = (np.asarray(lower), np.asarray(upper))
        self.lbg, self.ubg = (np.asarray(lbg), np.asarray(ubg))
        self.times = np.arange(horizon + 1)
        self.construction_calls = 1
        self.solve_calls = 0
        self.build_wall_sec = time.perf_counter() - start

    def solve(self, initial_state, theta, controls=None):
        c, N = (self.coc, self.horizon)
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if controls is not None:
            controls = np.asarray(controls, dtype=float).reshape(-1, c.n_control)
            if len(controls) != N:
                controls = None
        lower, upper = (self.lower.copy(), self.upper.copy())
        lower[:c.n_state] = upper[:c.n_state] = initial_state
        guess = np.empty_like(lower)
        stride = c.n_state + c.n_control
        guess[:c.n_state] = initial_state
        state = initial_state
        for k in range(N):
            control = controls[k] if controls is not None else 0.5 * (np.asarray(c.control_lb) + np.asarray(c.control_ub))
            guess[k * stride + c.n_state:(k + 1) * stride] = control
            state = np.asarray(c.dyn_fn(state, control, theta)).reshape(-1) if controls is not None else 0.5 * (np.asarray(c.state_lb) + np.asarray(c.state_ub))
            guess[(k + 1) * stride:(k + 1) * stride + c.n_state] = state
        sol = self.solver(x0=guess, p=theta, lbx=lower, ubx=upper, lbg=self.lbg, ubg=self.ubg)
        self.solve_calls += 1
        primal = np.asarray(sol['x']).reshape(-1)
        lam = np.asarray(sol['lam_g']).reshape(-1)
        g = np.asarray(sol['g']).reshape(-1)
        trajectory = np.r_[primal, np.zeros(c.n_control)].reshape(N + 1, stride)
        path_count = N * (c.n_state + c.n_path_inequ_cstr + c.n_path_equ_cstr)
        path = lam[:path_count].reshape(N, -1)
        values = g[:path_count].reshape(N, -1)
        ni, ne = (c.n_path_inequ_cstr, c.n_path_equ_cstr)
        nf, ef = (c.n_final_inequ_cstr, c.n_final_equ_cstr)
        return dict(state_traj_opt=trajectory[:, :c.n_state], control_traj_opt=trajectory[:-1, c.n_state:], costate_traj_opt=path[:, ni + ne:], inequ_path=values[:, :ni], inequ_final=g[path_count:path_count + nf], v_path=path[:, :ni], w_path=path[:, ni:ni + ne], v_final=lam[path_count:path_count + nf], w_final=lam[path_count + nf:path_count + nf + ef], auxvar_value=theta, time=self.times.copy(), horizon=N, cost=np.asarray(sol['f']), solver_stats=self.solver.stats())
