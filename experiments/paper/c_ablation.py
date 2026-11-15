from .common import *
import casadi as ca
import ctypes as ct
import subprocess, hashlib, time
from experiments.exp2_OCPs import utils

class CBackward:

    def __init__(self, model, ins, method='c_final', tolerance=0.001, kernel_path=None, forward_solver=None, extra_functions=()):
        self.ins = ins
        self.tol = tolerance
        self.mode = int(method == 'c_kkt')
        self.method = method
        self.forward = utils.build_lapanda_solver(ins, model, 'review_full_' + model, 'compiled', False) if forward_solver is None else forward_solver
        p = ins.problem
        x, t, z = (p.u, p.theta, p.variable)
        n = x.numel()
        m = p.constraints.numel()
        y = ca.SX.sym('y', m)
        v = ca.SX.sym('v', n)
        w = ca.SX.sym('w', m)
        station = ca.gradient(p.cost + ca.dot(y, p.constraints), x)
        fs = [ca.Function('ab_hv', [x, t, z, y, v], [ca.densify(ca.jtimes(station, x, v))]), ca.Function('ab_jv', [x, t, z, v], [ca.densify(ca.jtimes(p.constraints, x, v))]), ca.Function('ab_jt', [x, t, z, w], [ca.densify(ca.jtimes(p.constraints, x, w, True))]), ca.Function('ab_mix', [x, t, z, y, v, w], [ca.densify(ca.jtimes(station, t, v, True) + ca.jtimes(p.constraints, t, w, True))]), ca.Function('ab_setup', [x, t, z], [ca.densify(p.constraints), ca.densify(ca.gradient(p.outer_loss, x)), ca.densify(ca.gradient(p.outer_loss, t))])]
        fs.extend(extra_functions)
        self.fs = {f.name(): f for f in fs}
        kernel = Path(kernel_path) if kernel_path else HERE / 'c_ablation_kernel.c'
        sources = [kernel, ROOT / 'panda/panda_linear_solver.c', ROOT / 'panda/matrix_operations.c']
        dependencies = []
        if kernel_path:
            import re
            seen = {kernel.resolve()}

            def scan(path):
                for name in re.findall('#include\\s+"([^\\"]+)"', path.read_text()):
                    child = (path.parent / name).resolve()
                    if child.exists() and child not in seen:
                        seen.add(child)
                        dependencies.append(child)
                        scan(child)
            scan(kernel)
        fingerprint = hashlib.sha256(b''.join((s.read_bytes() for s in sources)) + b''.join((s.read_bytes() for s in dependencies)) + ''.join((f.serialize() for f in fs)).encode()).hexdigest()[:14]
        dest = HERE / 'cache' / f'c_ablation_{model}_{fingerprint}'
        dest.mkdir(parents=True, exist_ok=True)
        lib = dest / 'ablation.so'
        if not lib.exists():
            gen = ca.CodeGenerator('ab_products.c', {'with_header': True})
            for f in fs:
                gen.add(f)
            gen.generate(str(dest) + os.sep)
            (dest / 'ab_dimensions.h').write_text(f'#define AB_N {n}\n#define AB_M {m}\n#define AB_P {t.numel()}\n#define AB_W {max((f.sz_w() for f in fs))}\n#define AB_IW {max((f.sz_iw() for f in fs))}\n')
            product_id = hashlib.sha256(('gcc-O3-fPIC-' + sys.platform + ''.join((f.serialize() for f in fs))).encode()).hexdigest()[:16]
            objdir = HERE / 'cache' / ('product_objects_' + product_id)
            objdir.mkdir(exist_ok=True)
            obj = objdir / 'products.o'
            compile_cmd = ['gcc', '-O3', '-std=c99', '-fPIC', '-c', str(dest / 'ab_products.c'), '-o', str(obj)]
            if not obj.exists():
                subprocess.run(compile_cmd, check=True)
            cmd = ['gcc', '-O3', '-std=c99', '-D_POSIX_C_SOURCE=200809L', '-shared', '-fPIC', '-I' + str(dest), '-I' + str(ROOT / 'panda'), str(obj), *[str(s) for s in sources], '-lm', '-o', str(lib)]
            subprocess.run(cmd, check=True)
            (dest / 'build.json').write_text(json.dumps(dict(command=cmd, product_compile_command=compile_cmd, fingerprint=fingerprint, scope='Identical compiled operator and -O3 for both backward strategies; common installed native forward.'), indent=2))
        self.library = ct.CDLL(str(lib))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        self.library.ab_backward.argtypes = [ct.c_int] + [arr] * 9 + [ct.c_double, ct.c_int, arr, arr]
        self.library.ab_backward.restype = ct.c_int
        self.bounds = [np.ascontiguousarray(flat(q)) for q in [p.box_lower, p.box_upper, ins.constraint_lower, ins.constraint_upper]]
        self.last_status = None
        self.last_backward_status = None

    def backward(self, r, theta, var):
        ts = time.perf_counter()
        args = [np.ascontiguousarray(flat(q)) for q in [r['solution'], theta, var, r['multipliers'], r['penalties']]]
        grad = np.zeros(len(theta))
        info = np.zeros(10)
        status = self.library.ab_backward(self.mode, *args, *self.bounds, self.tol, 800, grad, info)
        self.last_backward_status = status
        return (grad, dict(backward_time_sec=time.perf_counter() - ts, backward_c_time_sec=info[7], backward_iterations=int(info[1]), backward_residual=info[0], backward_solver_used=self.method, backward_fallback_used=bool(info[8]), backward_peak_workspace_bytes=int(info[6]), kkt_dimension=int(info[2]), free_dimension=int(info[3]), kkt_active=int(info[4]), backward_matvecs=int(info[5]), backward_status=status))

    def solve(self, u0, theta, var, **kw):
        kw = dict(kw)
        kw['compute_backward'] = False
        r = utils.solve_lapanda(self.forward, self.ins, u0, theta, var, **kw)
        gradient, stats = self.backward(r, theta, var)
        r.update(stats)
        r['grad_theta'] = gradient
        return r
