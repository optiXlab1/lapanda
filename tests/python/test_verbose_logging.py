import os
from pathlib import Path

import casadi as ca
import numpy as np

from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver


def _problem():
    u = ca.SX.sym("u", 2)
    theta = ca.SX.sym("theta", 2)
    variable = ca.SX.sym("variable", 1)
    cost = 0.5 * ca.sumsqr(u - theta)
    constraints = ca.vertcat(u[0] + u[1])
    return CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=cost,
        box_lower=np.array([-2.0, -2.0]),
        box_upper=np.array([2.0, 2.0]),
        constraints=constraints,
    )


def test_solver_does_not_write_csv_logs(tmp_path):
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        solver = build_solver(_problem(), backend="callback")
        solver_options = SolverOptions()
        solver_options.max_iterations = 20
        solver_options.tolerance = 1e-8
        alm_options = AlmOptions()
        alm_options.max_iterations = 3
        alm_options.tolerance = 1e-4
        backward_options = BackwardOptions()

        solver.solve_lapanda(
            np.zeros(2),
            np.array([2.0, -0.25]),
            np.zeros(1),
            np.array([1.0]),
            np.array([1.0]),
            inner_solver_options=solver_options,
            alm_options=alm_options,
            backward_options=backward_options,
        )
        assert not Path("panda_log.csv").exists()
        assert not Path("LAPANDA_log.csv").exists()

        solver_options.verbose = True
        alm_options.verbose = True
        solver.solve_lapanda(
            np.zeros(2),
            np.array([2.0, -0.25]),
            np.zeros(1),
            np.array([1.0]),
            np.array([1.0]),
            inner_solver_options=solver_options,
            alm_options=alm_options,
            backward_options=backward_options,
        )
        assert not Path("panda_log.csv").exists()
        assert not Path("LAPANDA_log.csv").exists()

        solver.solve_panda(
            np.zeros(2),
            np.array([2.0, -0.25]),
            np.zeros(1),
            solver_options=solver_options,
            backward_options=backward_options,
        )
        assert not Path("panda_log.csv").exists()
        assert not Path("LAPANDA_log.csv").exists()
    finally:
        os.chdir(cwd)
