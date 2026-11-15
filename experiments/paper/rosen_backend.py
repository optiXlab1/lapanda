from .common import *
import argparse, time
import pandas as pd
from lapanda import build_solver, SolverOptions, AlmOptions, BackwardOptions
from experiments.exp1_rosenbrock_smooth_constraints.problem import build_problem, initial_point, theta_nominal, sinusoidal_target
from .native_direct_backends import NativeDirect
from .references import Reference
OUT = RESULTS / 'rosenbrock/main'

def options(system, tol, scale=1):
    b = BackwardOptions()
    b.enable = True
    b.system = system
    b.linear_solver = 'cg' if system == 'subproblem' else 'minres'
    b.tolerance = tol
    b.max_iterations = 4000
    b.constraint_penalty_scale = scale
    b.constraint_penalty_max = 0.0
    return b

def reverse(solver, ins, state, t, z, system, tol, scale=1):
    start = time.perf_counter()
    r = solver.backward_lapanda_at_solution(state['solution'], t, z, ins.constraint_lower, ins.constraint_upper, state['multipliers'], state['penalties'], state['final_gamma'], backward_options=options(system, tol, scale))
    if system == 'subproblem':
        assert r.get('subproblem_projection_evaluations') == 1, 'Optimized native subproblem solver was not loaded'
    return (flat(r['grad_theta']), dict(wall_sec=time.perf_counter() - start, kernel_sec=r['backward_time_sec'], iterations=r['backward_iterations'], linear_residual=r['backward_residual'], workspace_bytes=r['backward_peak_workspace_bytes'], fallback=r['backward_fallback_used'], dimension=r.get('backward_system_dimension', len(state['solution'])), solver_used=r['backward_solver_used'], subproblem_projection_evaluations=r.get('subproblem_projection_evaluations'), subproblem_projection_workspace_bytes=r.get('subproblem_projection_workspace_bytes')))
