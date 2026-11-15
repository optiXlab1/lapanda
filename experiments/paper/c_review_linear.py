from .c_ablation import *
METHODS = {'cg': 0, 'minres': 1, 'pminres': 2, 'jacobi_pcg': 3, 'lbfgs_pcg': 4}

class CReview(CBackward):

    def __init__(self, model, ins, method='cg', tolerance=0.001, forward_solver=False, kernel_path=None, schur_diagonal=False, max_iterations=10000):
        p = ins.problem
        x, t, z = (p.u, p.theta, p.variable)
        n = x.numel()
        m = p.constraints.numel()
        y = ca.SX.sym('dd_y', m)
        w = ca.SX.sym('dd_w', m)
        H = ca.hessian(p.cost + ca.dot(y, p.constraints), x)[0]
        J = ca.jacobian(p.constraints, x)
        diag = ca.Function('ab_diag', [x, t, z, y, w], [ca.densify(ca.diag(H)), ca.densify(ca.sum2(J * J)), ca.densify(ca.diag(H) + (J * J).T @ w)])
        extras = [diag]
        if schur_diagonal:
            ih = ca.SX.sym('inverse_hdiag', n)
            extras.append(ca.Function('ab_schur_diag', [x, t, z, ih], [ca.densify(J * J @ ih)]))
        super().__init__(model, ins, 'c_final', tolerance, kernel_path or HERE / 'c_review_linear.c', forward_solver=forward_solver, extra_functions=extras)
        self.review_method = method
        self.max_iterations = max_iterations
        arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        self.library.ab_review.argtypes = [ct.c_int, ct.c_int, arr, arr, ct.c_double, ct.c_double] + [arr] * 9 + [ct.c_double, ct.c_int, arr, arr]
        self.library.ab_review.restype = ct.c_int

    def backward(self, r, theta, var):
        ts = time.perf_counter()
        args = [np.ascontiguousarray(flat(a)) for a in [r['solution'], theta, var, r['multipliers'], r['penalties']]]
        S = np.ascontiguousarray(r.get('S', np.empty(0)))
        Y = np.ascontiguousarray(r.get('Y', np.empty(0)))
        g = np.zeros(len(theta))
        info = np.zeros(14)
        st = self.library.ab_review(METHODS[self.review_method], len(S) // len(r['solution']), S, Y, float(r.get('scale', 1)), float(r.get('gamma', 1)), *args, *self.bounds, self.tol, self.max_iterations, g, info)
        return (g, dict(wall_sec=time.perf_counter() - ts, c_sec=info[7], status=st, residual=info[0], iterations=int(info[1]), dimension=int(info[2]), free=int(info[3]), active=int(info[4]), products=int(info[5]), heap_bytes=int(info[6]), original_kkt_residual=info[9], setup_sec=info[10], pairs=int(info[11]), preconditioner_bytes=int(info[12])))
