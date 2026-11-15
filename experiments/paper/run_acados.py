from .common import *
import argparse, time, traceback
import casadi as ca
import psutil
from threadpoolctl import threadpool_limits
ACADOS = Path(os.environ['ACADOS_SOURCE_DIR'])
sys.path.insert(0, str(ACADOS / 'interfaces/acados_template'))
_dll_handle = os.add_dll_directory(str(ACADOS / 'bin')) if os.name == 'nt' else None
_dll_lib = os.add_dll_directory(str(ACADOS / 'lib')) if os.name == 'nt' else None
if os.name == 'nt':
    import ctypes
    _preloaded = [ctypes.CDLL(str(ACADOS / p)) for p in ['lib/qpOASES_e.dll', 'bin/blasfeo.dll', 'bin/hpipm.dll', 'bin/acados.dll']]
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from experiments.exp2_OCPs.utils import build_instance
from experiments.exp2_OCPs.problem_bank import _cartpole_step, _arm_fk

def formulation(name, N):
    ins = build_instance(name, N)
    nx = ins.nx
    nu = ins.nu
    dt = ins.dt
    x = ca.SX.sym('x', nx + (name == 'cartpole'))
    u = ca.SX.sym('u', nu)
    p = ca.SX.sym('p_global', ins.theta_initial.size + nx)
    t = p[:ins.theta_initial.size]
    target = p[ins.theta_initial.size:]
    e = x[:nx] - target
    if name == 'cartpole':
        q = t[0] * e[0] ** 2 + t[2] * e[1] ** 2 + t[1] * e[2] ** 2 + t[3] * e[3] ** 2
        stage = q + t[4] * u[0] ** 2
        terminal = 10 * q
        xn = ca.vertcat(_cartpole_step(x[:nx], u, dt), x[-1] + u[0] ** 2)
        h = ca.SX.zeros(0, 1)
        he = ca.vertcat(x[-1] - 70)
    elif name == 'quadrotor':
        q = t[0] * (e[0] ** 2 + e[1] ** 2) + t[1] * (e[2] ** 2 + e[3] ** 2) + t[2] * e[4] ** 2 + t[3] * e[5] ** 2
        stage = q + t[4] * (u[0] - 9.81) ** 2 + t[5] * u[1] ** 2
        terminal = 15 * q
        vx = x[2] - dt * u[0] * ca.sin(x[4])
        vz = x[3] + dt * (u[0] * ca.cos(x[4]) - 9.81)
        omega = x[5] + dt * 4 * u[1]
        angle = x[4] + dt * omega
        xn = ca.vertcat(x[0] + dt * vx, x[1] + dt * vz, vx, vz, angle, omega)
        h = ca.vertcat(0.43 - xn[1], xn[1] - 1.2, xn[0] - target[0], xn[4] - 0.34, -xn[4] - 0.34)
        he = ca.SX.zeros(0, 1)
    else:
        ee = _arm_fk(x[0], x[1])
        et = _arm_fk(target[0], target[1])
        q = t[0] * ca.sumsqr(e) + t[1] * ca.sumsqr(ee - et)
        stage = q + t[2] * u[0] ** 2 + t[3] * u[1] ** 2
        terminal = 20 * q
        xn = x + dt * u
        en = _arm_fk(xn[0], xn[1])
        h = ca.vertcat((0.29 + t[4]) ** 2 - ca.sumsqr(en - ca.DM([1.3, 0.3])), xn[0] - 0.75, -1.1 - xn[0], xn[1] - 1.28, -0.65 - xn[1])
        he = ca.SX.zeros(0, 1)
    return (ins, x, u, p, xn, stage, terminal, h, he)

def build(name, N, sens, forward_mode='exact'):
    ins, x, u, p, xn, stage, terminal, h, he = formulation(name, N)
    ocp = AcadosOcp()
    m = AcadosModel()
    m.name = f'rev2_{name}_{N}_' + ('sens' if sens else 'fwd')
    m.x = x
    m.u = u
    m.p_global = p
    m.disc_dyn_expr = xn
    m.cost_expr_ext_cost = stage
    m.cost_expr_ext_cost_e = terminal
    m.con_h_expr = h
    m.con_h_expr_0 = h
    m.con_h_expr_e = he
    ocp.model = m
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * ins.dt
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.cost_scaling = np.ones(N + 1)
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    if not sens and forward_mode == 'gauss_newton':
        theta = p[:ins.theta_initial.size]
        e = x[:ins.nx] - p[ins.theta_initial.size:]
        if name == 'cartpole':
            state_r = ca.vertcat(ca.sqrt(2 * theta[0]) * e[0], ca.sqrt(2 * theta[2]) * e[1], ca.sqrt(2 * theta[1]) * e[2], ca.sqrt(2 * theta[3]) * e[3])
            control_r = ca.sqrt(2 * theta[4]) * u
            terminal_scale = 10.0
        elif name == 'quadrotor':
            state_r = ca.vertcat(ca.sqrt(2 * theta[0]) * e[:2], ca.sqrt(2 * theta[1]) * e[2:4], ca.sqrt(2 * theta[2]) * e[4], ca.sqrt(2 * theta[3]) * e[5])
            control_r = ca.vertcat(ca.sqrt(2 * theta[4]) * (u[0] - 9.81), ca.sqrt(2 * theta[5]) * u[1])
            terminal_scale = 15.0
        else:
            et = _arm_fk(p[ins.theta_initial.size], p[ins.theta_initial.size + 1])
            ee = _arm_fk(x[0], x[1])
            state_r = ca.vertcat(ca.sqrt(2 * theta[0]) * e, ca.sqrt(2 * theta[1]) * (ee - et))
            control_r = ca.vertcat(ca.sqrt(2 * theta[2]) * u[0], ca.sqrt(2 * theta[3]) * u[1])
            terminal_scale = 20.0
        m.cost_y_expr = ca.vertcat(state_r, control_r)
        m.cost_y_expr_e = ca.sqrt(terminal_scale) * state_r
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = np.eye(m.cost_y_expr.numel())
        ocp.cost.W_e = np.eye(m.cost_y_expr_e.numel())
        ocp.cost.yref = np.zeros(m.cost_y_expr.numel())
        ocp.cost.yref_e = np.zeros(m.cost_y_expr_e.numel())
        m.name = m.name + '_gn'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 500
    ocp.solver_options.tol = 1e-07
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_ric_alg = 0
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE' if sens else 'MIRROR'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.print_level = 0
    if not sens and forward_mode == 'gauss_newton':
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.regularize_method = 'NO_REGULARIZE'
    if sens:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True
    ocp.constraints.x0 = np.r_[ins.x_initial, 0.0] if name == 'cartpole' else ins.x_initial
    ocp.constraints.idxbu = np.arange(ins.nu)
    ocp.constraints.lbu = ins.problem.box_lower[:ins.nu]
    ocp.constraints.ubu = ins.problem.box_upper[:ins.nu]
    ocp.constraints.lh = np.full(h.numel(), -1e+20)
    ocp.constraints.uh = np.zeros(h.numel())
    ocp.constraints.lh_e = np.full(he.numel(), -1e+20)
    ocp.constraints.uh_e = np.zeros(he.numel())
    ocp.constraints.lh_0 = np.full(h.numel(), -1e+20)
    ocp.constraints.uh_0 = np.zeros(h.numel())
    ocp.p_global_values = np.r_[ins.theta_initial, ins.x_target]
    dest = HERE / 'cache' / ('linux' if os.name != 'nt' else '') / m.name
    dest.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(dest)
    builder = ocp_get_default_cmake_builder()
    if os.name == 'nt':
        builder.generator = 'Visual Studio 16 2019'
        builder.host = 'x64'
    builder.additional_cmake_options = ''
    builder.build_dir = str(dest / 'build')
    jsonfile = str(dest / 'ocp.json')
    exists = (dest / (f'acados_ocp_solver_{m.name}.dll' if os.name == 'nt' else f'libacados_ocp_solver_{m.name}.so')).exists()
    if not exists:
        AcadosOcpSolver.generate(ocp, json_file=jsonfile, cmake_builder=builder, verbose=False)
        import re
        src = dest / f'acados_solver_{m.name}.c'
        txt = src.read_text()
        txt = re.sub('(=\\s*-?\\d{19,})(;)', '\\1.0\\2', txt)
        src.write_text(txt)
    solver = AcadosOcpSolver(ocp, json_file=jsonfile, cmake_builder=builder, verbose=False, generate=False, build=not exists)
    assert solver.acados_ocp.dims.nh_0 == h.numel(), 'Missing first-stage nonlinear constraints'
    return (ins, solver)
