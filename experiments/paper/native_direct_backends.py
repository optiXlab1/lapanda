from .common import *
import ctypes as ct, casadi as ca, subprocess, hashlib, time
import scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg.lapack as lapack

def ints(a):
    return ','.join(map(str, a)) or '0'

def doubles(a):
    return ','.join(('INFINITY' if x == np.inf else '-INFINITY' if x == -np.inf else format(float(x), '.17g') for x in a)) or '0'

class NativeDirect:

    def __init__(self, ins):
        self.ins = ins
        p = ins.problem
        x, t, z = (p.u, p.theta, p.variable)
        self.n = x.numel()
        self.m = p.constraints.numel()
        self.p = t.numel()
        y = ca.SX.sym('native_y', self.m)
        v = ca.SX.sym('native_v', self.n)
        w = ca.SX.sym('native_w', self.m)
        lag = p.cost + ca.dot(y, p.constraints)
        station = ca.gradient(lag, x)
        matrix = ca.Function('dr_matrix', [x, t, z, y], [ca.hessian(lag, x)[0], ca.jacobian(p.constraints, x)])
        hs = matrix.sparsity_out(0)
        js = matrix.sparsity_out(1)
        self.maxnz = hs.nnz() + 2 * js.nnz()
        self.maxdim = self.n + self.m
        fs = [matrix, ca.Function('dr_setup', [x, t, z], [ca.densify(p.constraints), ca.densify(ca.gradient(p.outer_loss, x)), ca.densify(ca.gradient(p.outer_loss, t))]), ca.Function('dr_mix', [x, t, z, y, v, w], [ca.densify(ca.jtimes(station, t, v, True) + ca.jtimes(p.constraints, t, w, True))])]
        header = f'#define DN {self.n}\n#define DM {self.m}\n#define DP {self.p}\n#define DHNZ {hs.nnz()}\n#define DJNZ {js.nnz()}\n#define DW {max(1, max((f.sz_w() for f in fs)))}\n#define DIW {max(1, max((f.sz_iw() for f in fs)))}\n'
        for name, values, ctype in [('HC', hs.colind(), 'int'), ('HR', hs.row(), 'int'), ('JC', js.colind(), 'int'), ('JR', js.row(), 'int'), ('BOX_LO', p.box_lower, 'double'), ('BOX_UP', p.box_upper, 'double'), ('CON_LO', ins.constraint_lower, 'double'), ('CON_UP', ins.constraint_upper, 'double')]:
            header += f'static const {ctype} {name}[]={{' + (ints(values) if ctype == 'int' else doubles(values)) + '};\n'
        kernel = HERE / 'native_direct_kernel.c'
        digest = hashlib.sha256(kernel.read_bytes() + header.encode() + ''.join((f.serialize() for f in fs)).encode()).hexdigest()[:18]
        folder = HERE / 'cache' / ('native_direct_' + digest)
        folder.mkdir(exist_ok=True, parents=True)
        library = folder / 'direct.so'
        if not library.exists():
            gen = ca.CodeGenerator('direct_products.c', {'with_header': True})
            for f in fs:
                gen.add(f)
            gen.generate(str(folder) + os.sep)
            (folder / 'direct_dimensions.h').write_text(header)
            cmd = ['gcc', '-O3', '-std=c99', '-D_POSIX_C_SOURCE=200809L', '-shared', '-fPIC', '-I' + str(folder), str(folder / 'direct_products.c'), str(kernel), '-lm', '-o', str(library)]
            subprocess.run(cmd, check=True)
            (folder / 'build.json').write_text(json.dumps(dict(command=cmd, kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(), products_sha256=hashlib.sha256((folder / 'direct_products.c').read_bytes()).hexdigest(), scope=__doc__), indent=2))
        self.lib = ct.CDLL(str(library))
        D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        I = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
        self.lib.direct_prepare.argtypes = [D, D, D, D, D, D, I, I, D, D, D, I, I, D]
        self.lib.direct_prepare.restype = ct.c_int
        self.lib.direct_dense.argtypes = [ct.c_int, D, I, I, D]
        self.lib.direct_dense.restype = ct.c_int
        self.lib.direct_finish.argtypes = [D, D, D, D, D, ct.c_int, ct.c_int, I, I, D, D, D, I, I, D, D]
        self.lib.direct_finish.restype = ct.c_int
        self.sysv = lapack.get_lapack_funcs('sysv', dtype=np.float64)
        self.sysv_work = lapack.get_lapack_funcs('sysv_lwork', dtype=np.float64)
        self.build = folder / 'build.json'

    def prepare(self, state, theta, z):
        x = np.ascontiguousarray(state['solution'], dtype=float)
        theta = np.ascontiguousarray(theta, dtype=float)
        z = np.ascontiguousarray(z, dtype=float)
        y = np.ascontiguousarray(state['multipliers'], dtype=float)
        R = np.ascontiguousarray(state['penalties'], dtype=float)
        values = np.empty(self.maxnz)
        rows = np.empty(self.maxnz, dtype=np.int32)
        cols = np.empty(self.maxdim + 1, dtype=np.int32)
        rhs = np.empty(self.maxdim)
        yp = np.empty(self.m)
        d = np.empty(self.p)
        free = np.empty(self.n, dtype=np.int32)
        active = np.empty(self.m, dtype=np.int32)
        info = np.empty(5)
        status = self.lib.direct_prepare(x, theta, z, y, R, values, rows, cols, rhs, yp, d, free, active, info)
        if status:
            raise RuntimeError(('Native KKT assembly failed', status))
        nf, na, nnz = map(int, info[:3])
        dim = nf + na
        return dict(x=x, theta=theta, z=z, values=values, rows=rows, cols=cols, rhs=rhs, yp=yp, d=d, free=free, active=active, nf=nf, na=na, nnz=nnz, dim=dim, assembly_C_sec=float(info[3]), assembly_C_heap=float(info[4]))

    def matrix(self, data):
        k = sp.csc_matrix((data['values'][:data['nnz']], data['rows'][:data['nnz']], data['cols'][:data['dim'] + 1]), shape=(data['dim'], data['dim']), copy=False)
        k.has_sorted_indices = True
        k.has_canonical_format = True
        return k

    def backward(self, state, theta, z, method):
        start = time.perf_counter()
        data = self.prepare(state, theta, z)
        prepared = time.perf_counter()
        dim = data['dim']
        factor_time = solve_time = 0.0
        if method == 'sparse_lu':
            K = self.matrix(data)
            converted = time.perf_counter()
            factor = sla.splu(K)
            factored = time.perf_counter()
            adj = np.ascontiguousarray(factor.solve(data['rhs'][:dim]))
            solved = time.perf_counter()
            factor_time = factored - converted
            solve_time = solved - factored
        elif method == 'dense_ldlt':
            K = np.empty((dim, dim), dtype=float, order='F')
            self.lib.direct_dense(dim, data['values'], data['rows'], data['cols'], K.ravel(order='F'))
            converted = time.perf_counter()
            work, code = self.sysv_work(dim, lower=1)
            if code:
                raise RuntimeError(('SYSV workspace query', code))
            matrix, piv, solution, code = self.sysv(K, data['rhs'][:dim, None].copy(order='F'), lower=1, lwork=max(1, int(work)), overwrite_a=1, overwrite_b=1)
            if code:
                raise RuntimeError(('SYSV failed', code))
            adj = np.ascontiguousarray(solution[:, 0])
            solved = time.perf_counter()
            factor_time = solved - converted
        else:
            raise ValueError(method)
        gradient = np.empty(self.p)
        info = np.empty(2)
        status = self.lib.direct_finish(data['x'], data['theta'], data['z'], data['yp'], data['d'], data['nf'], data['na'], data['free'], data['active'], adj, data['rhs'], data['values'], data['rows'], data['cols'], gradient, info)
        stats = dict(status=status, setup_sec=prepared - start, matrix_conversion_sec=converted - prepared, factor_sec=factor_time, solve_sec=solve_time, assembly_C_sec=data['assembly_C_sec'], finish_C_sec=float(info[1]), dimension=dim, free=data['nf'], active=data['na'], nnz=data['nnz'], residual=float(info[0]), original_kkt_residual=float(info[0]), fallback=False, iterations=1, c_sec=np.nan, heap_bytes=np.nan, backend='SuperLU fresh ordering/factor/solve' if method == 'sparse_lu' else 'LAPACK symmetric-indefinite SYSV inplace fresh factor/solve', storage_scope='Complete native allocation peak profiled separately; no factor inspection during call.')
        if method == 'sparse_lu':
            del factor, K
        else:
            del matrix, piv, solution, K
        del adj, data
        end = time.perf_counter()
        stats.update(wall_sec=end - start, finish_sec=end - solved)
        return (gradient, stats)
