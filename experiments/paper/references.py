from .common import *
import casadi as ca
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.linalg as la

def sparse(dm):
    r, c = dm.sparsity().get_triplet()
    return sp.csc_matrix((np.asarray(dm.nonzeros()), (r, c)), shape=dm.shape)

class Reference:

    def __init__(self, ins):
        self.ins = ins
        p = ins.problem
        x, t, z = (p.u, p.theta, p.variable)
        y = ca.SX.sym('reference_lambda', p.constraints.numel())
        station = ca.gradient(p.cost + ca.dot(y, p.constraints), x)
        self.der = ca.Function('reference_derivatives', [x, t, z, y], [ca.jacobian(station, x), ca.jacobian(p.constraints, x), ca.jacobian(station, t), ca.jacobian(p.constraints, t), ca.gradient(p.outer_loss, x), ca.gradient(p.outer_loss, t), station])
        self.ips = {}
        for label, tol in [('strict12', 1e-12), ('strict14', 1e-14)]:
            self.ips[label] = ca.nlpsol('original_' + label, 'ipopt', dict(x=x, p=ca.vertcat(t, z), f=p.cost, g=p.constraints), {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'ipopt.tol': tol, 'ipopt.acceptable_iter': 0, 'ipopt.constr_viol_tol': tol, 'ipopt.dual_inf_tol': tol, 'ipopt.compl_inf_tol': tol, 'ipopt.bound_relax_factor': 0.0, 'ipopt.nlp_scaling_method': 'none', 'ipopt.max_iter': 5000})

    def solve(self, start, t, z, label='strict14'):
        p = self.ins.problem
        ip = self.ips[label]
        sol = ip(x0=start, p=np.r_[t, z], lbx=p.box_lower, ubx=p.box_upper, lbg=self.ins.constraint_lower, ubg=self.ins.constraint_upper)
        return (sol, ip.stats())

    def gradient(self, x, t, z, y, lamx=None, active=None, free=None):
        p = self.ins.problem
        n = len(x)
        c = np.asarray(ca.Function('reference_c', [p.u, p.theta, p.variable], [p.constraints])(x, t, z)).reshape(-1)
        if active is None:
            active = np.flatnonzero((abs(y) > 1e-07) | (abs(self.ins.constraint_upper - self.ins.constraint_lower) < 1e-12))
        if free is None:
            free = np.flatnonzero((x > p.box_lower + 1e-08) & (x < p.box_upper - 1e-08))
        H, J, B, C, q, d, station = self.der(x, t, z, y)
        H = sparse(H)[free][:, free]
        J = sparse(J)
        A = J[active][:, free]
        full_active = active.copy()
        fullA = A.copy()
        B = np.asarray(B)[free]
        C = np.asarray(C)
        q = flat(q)[free]
        dropped = 0
        if A.shape[0] and A.shape[1] <= 100:
            _, r, perm = la.qr(A.toarray().T, mode='economic', pivoting=True)
            diagonal = abs(np.diag(r))
            rank = int((diagonal > 1e-11 * max(float(diagonal.max(initial=0)), 1.0)).sum())
            if rank < A.shape[0]:
                choose = np.sort(perm[:rank])
                active = active[choose]
                A = A[choose]
                dropped = len(full_active) - rank
        K = sp.bmat([[H, A.T], [A, sp.csc_matrix((len(active), len(active)))]], format='csc')
        rhs = np.r_[q, np.zeros(len(active))]
        if len(rhs):
            factor = sla.splu(K)
            v = factor.solve(rhs)
            for _ in range(2):
                v += factor.solve(rhs - K @ v)
        else:
            v = np.zeros(0)
        nf = len(free)
        gradient = flat(d) - B.T @ v[:nf] - C[active].T @ v[nf:]
        linear = float(np.linalg.norm(K @ v - rhs) / max(np.linalg.norm(rhs), 1e-300))
        constraints_consistency = float(np.linalg.norm(fullA @ v[:nf]) / max(np.linalg.norm(rhs), 1e-300))
        station = flat(station) + (np.zeros(n) if lamx is None else lamx)
        violation = float(max(np.maximum(self.ins.constraint_lower - c, 0).max(initial=0), np.maximum(c - self.ins.constraint_upper, 0).max(initial=0), np.maximum(p.box_lower - x, 0).max(initial=0), np.maximum(x - p.box_upper, 0).max(initial=0)))
        boundary = np.where(y >= 0, self.ins.constraint_upper, self.ins.constraint_lower)
        comp = float(abs(y * (c - boundary)).max(initial=0))
        info = dict(linear_residual=linear, stationarity_inf=float(abs(station).max(initial=0)), violation_inf=violation, complementarity_inf=comp, dropped_active_rows=dropped, full_active_adjoint_residual=constraints_consistency, dimension=K.shape[0])
        return (gradient, info)

    def resolved(self, start, t, z):
        records = []
        gradients = []
        solutions = []
        for label in ['strict12', 'strict14']:
            sol, stats = self.solve(start, t, z, label)
            x = flat(sol['x'])
            y = flat(sol['lam_g'])
            lx = flat(sol['lam_x'])
            g, info = self.gradient(x, t, z, y, lx)
            records.append(dict(variant=label, ipopt_success=stats['success'], **info))
            gradients.append(g)
            solutions.append(sol)
        change = float(np.linalg.norm(gradients[1] - gradients[0]) / max(np.linalg.norm(gradients[1]), 1e-300))
        qualified = all((r['violation_inf'] <= 1e-09 and r['stationarity_inf'] <= 1e-08 and (r['linear_residual'] <= 1e-08) for r in records))
        qualified &= np.linalg.norm(gradients[1] - gradients[0]) <= 1e-08 + 0.001 * np.linalg.norm(gradients[1])
        return (gradients[-1], solutions[-1], dict(qualified=bool(qualified), reference_relative_change=change, attempts=records))

    def finite_difference(self, sol, t, z, direction):
        p = self.ins.problem
        loss = ca.Function('reference_outer', [p.u, p.theta, p.variable], [p.outer_loss])
        direction = np.asarray(direction)
        direction = direction / np.linalg.norm(direction)
        values = []
        for h in [0.0001, 1e-05, 1e-06]:
            losses = []
            for sign in [-1, 1]:
                tt = t + sign * h * direction
                ss, _ = self.solve(flat(sol['x']), tt, z)
                losses.append(float(loss(ss['x'], tt, z)))
            values.append(dict(step=h, directional_derivative=(losses[1] - losses[0]) / (2 * h)))
        return (direction, values)
