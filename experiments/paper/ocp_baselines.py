from .common import *
import time, subprocess, inspect, textwrap, types
import casadi as ca
import scipy.sparse.linalg as sla
from experiments.exp2_OCPs import utils

def safepdp_clqr(symmetric_value=True):
    from experiments.exp2_OCPs import safepdp_baseline as sp
    clqr = sp.SafePDP.EQCLQR()
    if symmetric_value:
        source = textwrap.dedent(inspect.getsource(type(clqr).eqctlqrSolver))
        old = 'next_V = curr_Mxx + 2 * curr_Mux.T @ curr_K + curr_K.T @ curr_Muu @ curr_K'
        new = 'next_V = curr_Mxx + curr_Mux.T @ curr_K + curr_K.T @ curr_Mux + curr_K.T @ curr_Muu @ curr_K'
        assert source.count(old) == 1, 'Upstream CLQR changed: re-audit before patching'
        namespace = {'np': np}
        exec(source.replace(old, new), namespace)
        clqr.eqctlqrSolver = types.MethodType(namespace['eqctlqrSolver'], clqr)
    return clqr

def safepdp_bound_aware_backward(ins, coc, clqr, traj, theta, var, active_tol):
    from experiments.exp2_OCPs import safepdp_baseline as sp
    N = ins.horizon
    nx = coc.n_state
    nu = coc.n_control
    aux = coc.getAuxSys(traj, threshold=active_tol)
    for k, uk in enumerate(traj['control_traj_opt']):
        idx = np.flatnonzero((uk <= np.asarray(coc.control_lb) + active_tol) | (uk >= np.asarray(coc.control_ub) - active_tol))
        aux['GbarHx_t'][k] = np.vstack([aux['GbarHx_t'][k], np.zeros((len(idx), nx))])
        aux['GbarHu_t'][k] = np.vstack([aux['GbarHu_t'][k], np.eye(nu)[idx]])
        aux['GbarHe_t'][k] = np.vstack([aux['GbarHe_t'][k], np.zeros((len(idx), len(theta)))])
    clqr.auxsys2Eqctlqr(auxsys=aux)
    sens = clqr.eqctlqrSolver(threshold=active_tol)
    du = var[2 * ins.nx:2 * ins.nx + N * nu]
    dx = var[2 * ins.nx + N * nu:].reshape(N, ins.nx)
    return sp.imitation_loss_and_grad(ins, du, np.vstack([var[:ins.nx], dx]), traj, sens)[1]
