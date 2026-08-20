import json
import subprocess

import casadi as ca
import numpy as np
import pytest

from lapanda import (
    AlmOptions,
    BackwardOptions,
    CasadiProblem,
    CompiledSolver,
    SolverOptions,
    build_solver,
    generate_compiled_oracle,
)


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
        box_lower=np.array([-1.0, -1.0]),
        box_upper=np.array([1.0, 1.0]),
        constraints=constraints,
        outer_loss=0.5 * ca.sumsqr(u),
    )


def test_generate_compiled_oracle_writes_c_sources(tmp_path):
    artifact = generate_compiled_oracle(
        _problem(),
        name="test_oracle",
        cache_dir=tmp_path,
        build=False,
    )

    assert artifact.source_path.exists()
    assert artifact.header_path.exists()
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["n"] == 2
    assert metadata["ntheta"] == 2
    assert metadata["ncon"] == 1
    assert "panda_cost_grad" in metadata["functions"]
    assert "panda_constraint_weighted_hvp" in metadata["functions"]


def test_build_solver_compiled_backend_creates_artifact_shell(tmp_path):
    solver = build_solver(
        _problem(),
        backend="compiled",
        name="test_solver_oracle",
        cache_dir=tmp_path,
        build=False,
    )

    assert solver.generated_oracle.source_path.exists()
    assert solver.compiled_oracle is solver.generated_oracle
    with pytest.raises(RuntimeError, match="without a built library"):
        solver.solve_panda(
            x0=np.zeros(2),
            theta=np.array([0.25, -0.75]),
            variable=np.zeros(1),
        )


def test_build_solver_defaults_to_compiled_backend(tmp_path):
    solver = build_solver(
        _problem(),
        name="test_default_compiled_oracle",
        cache_dir=tmp_path,
        build=False,
    )
    assert isinstance(solver, CompiledSolver)


def test_generate_compiled_oracle_can_build_shared_library(tmp_path):
    try:
        artifact = generate_compiled_oracle(
            _problem(),
            name="test_build_oracle",
            cache_dir=tmp_path,
            build=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"C compiler is not available: {exc}")

    assert artifact.library_path is not None
    assert artifact.library_path.exists()


def test_compiled_solver_matches_callback_panda_forward_backward(tmp_path):
    try:
        callback_solver = build_solver(_problem(), backend="callback")
        compiled_solver = build_solver(
            _problem(),
            backend="compiled",
            name="test_runtime_oracle",
            cache_dir=tmp_path,
            build=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"C compiler is not available: {exc}")

    solver_options = SolverOptions()
    solver_options.max_iterations = 100
    solver_options.tolerance = 1e-9
    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-9
    backward_options.max_iterations = 100

    x0 = np.zeros(2)
    theta = np.array([0.25, -0.75])
    variable = np.zeros(1)
    callback = callback_solver.solve_panda(
        x0, theta, variable, solver_options=solver_options, backward_options=backward_options
    )
    compiled = compiled_solver.solve_panda(
        x0, theta, variable, solver_options=solver_options, backward_options=backward_options
    )

    np.testing.assert_allclose(compiled["solution"], callback["solution"], atol=1e-8)
    np.testing.assert_allclose(compiled["grad_theta"], callback["grad_theta"], atol=1e-8)


def test_compiled_solver_matches_callback_alm_forward_backward(tmp_path):
    try:
        callback_solver = build_solver(_problem(), backend="callback")
        compiled_solver = build_solver(
            _problem(),
            backend="compiled",
            name="test_alm_runtime_oracle",
            cache_dir=tmp_path,
            build=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"C compiler is not available: {exc}")

    solver_options = SolverOptions()
    solver_options.max_iterations = 150
    solver_options.tolerance = 1e-8
    alm_options = AlmOptions()
    alm_options.max_iterations = 12
    alm_options.tolerance = 1e-7
    alm_options.initial_penalty = 1.0
    alm_options.penalty_update_factor = 2.0
    alm_options.sufficient_decrease_factor = 0.25
    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-8
    backward_options.max_iterations = 120

    x0 = np.zeros(2)
    theta = np.array([2.0, -0.25])
    variable = np.zeros(1)
    callback = callback_solver.solve_lapanda(
        x0=x0,
        theta=theta,
        variable=variable,
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=solver_options,
        alm_options=alm_options,
        backward_options=backward_options,
    )
    compiled = compiled_solver.solve_lapanda(
        x0=x0,
        theta=theta,
        variable=variable,
        constraint_lower=np.array([1.0]),
        constraint_upper=np.array([1.0]),
        inner_solver_options=solver_options,
        alm_options=alm_options,
        backward_options=backward_options,
    )

    np.testing.assert_allclose(compiled["solution"], callback["solution"], atol=1e-8)
    np.testing.assert_allclose(compiled["multipliers"], callback["multipliers"], atol=1e-8)
    np.testing.assert_allclose(compiled["grad_theta"], callback["grad_theta"], atol=1e-8)
