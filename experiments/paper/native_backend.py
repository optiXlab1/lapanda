from .common import *
from experiments.exp2_OCPs import utils

class NativeBackend:

    def __init__(self, model, ins, method, tolerance):
        self.ins = ins
        self.method = method
        self.tolerance = tolerance
        self.forward = utils.build_lapanda_solver(ins, model, 'native_paper_' + model, 'compiled', False)
        self.last_status = None
        self.last_backward_status = None

    def solve(self, u0, theta, var, **kw):
        kw = dict(kw)
        kkt = self.method == 'c_kkt'
        kw.update(backward_tolerance=self.tolerance, backward_system='original_kkt' if kkt else 'subproblem', backward_linear_solver='minres' if kkt else 'cg', backward_constraint_penalty_scale=1.0, backward_constraint_penalty_max=0.0)
        result = utils.solve_lapanda(self.forward, self.ins, u0, theta, var, **kw)
        if not kkt and kw.get('compute_backward', False):
            assert result.get('subproblem_projection_evaluations') == 1, 'Optimized native CG was not loaded'
        self.last_status = 0 if result['final_residual'] <= kw['alm_tol'] else 1
        self.last_backward_status = 0 if result.get('backward_residual', 0.0) <= self.tolerance else 1
        return result
