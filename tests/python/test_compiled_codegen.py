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
    export_c_project,
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


@pytest.mark.parametrize("split", [False, True])
def test_nonlinear_oracle_products_and_cache(tmp_path, monkeypatch, split):
    u = ca.SX.sym("u", 3)
    theta = ca.SX.sym("theta", 2)
    z = ca.SX.sym("z", 1)
    cost = ca.sin(theta[0]*u[0]*u[1]) + theta[1]*ca.sumsqr(u)
    c = ca.vertcat(u[0]*u[2]*theta[0], ca.sin(u[1]+theta[1]))
    problem = CasadiProblem(u, theta, z, cost, constraints=c)
    artifact = generate_compiled_oracle(problem, cache_dir=tmp_path, split_sources=split, build_jobs=2)
    v = ca.SX.sym("v", 3)
    w = ca.SX.sym("w", 2)
    grad = ca.gradient(cost, u)
    jt = ca.jacobian(c, u).T @ w
    targets = {
        "panda_vjp": ([v,u,theta,z], ca.jacobian(grad,theta).T@v),
        "panda_constraint_jtprod": ([u,theta,z,w], jt),
        "panda_constraint_weighted_hvp": ([v,u,theta,z,w], ca.jtimes(jt,u,v)),
        "panda_constraint_weighted_vjp": ([v,u,theta,z,w], ca.jacobian(jt,theta).T@v),
        "panda_constraint_theta_jtprod": ([u,theta,z,w], ca.jacobian(c,theta).T@w),
    }
    rng = np.random.default_rng(25)
    for name, (inputs, output) in targets.items():
        expected = ca.Function("expected", inputs, [output])
        actual = ca.external(name, str(artifact.library_path))
        for _ in range(5):
            args = [rng.normal(size=x.shape) for x in inputs]
            np.testing.assert_allclose(actual(*args), expected(*args), rtol=1e-12, atol=1e-12)
    from lapanda import codegen
    def unexpected(*args, **kwargs):
        raise AssertionError("cache hit must not regenerate or compile")
    monkeypatch.setattr(codegen, "_make_oracle_functions", unexpected)
    monkeypatch.setattr(codegen, "_build_shared_library", unexpected)
    cached = generate_compiled_oracle(problem, cache_dir=tmp_path, split_sources=split, build_jobs=1)
    assert cached.library_path == artifact.library_path


def test_exported_split_project_builds(tmp_path):
    from lapanda.codegen import _default_cmake_generator
    directory = export_c_project(_problem(), tmp_path / "export")
    configure = ["cmake", "-S", str(directory), "-B", str(directory/"build")]
    generator = _default_cmake_generator()
    if generator:
        configure.extend(["-G", generator])
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(directory/"build"), "--parallel", "2"], check=True)


def test_split_cache_is_distinct_and_jobs_validated(tmp_path):
    a = generate_compiled_oracle(_problem(), cache_dir=tmp_path, build=False)
    b = generate_compiled_oracle(_problem(), cache_dir=tmp_path, build=False, split_sources=False)
    assert a.source_path.parent != b.source_path.parent
    assert len(a.metadata["sources"]) > 1
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="build_jobs"):
            generate_compiled_oracle(_problem(), cache_dir=tmp_path, build_jobs=value)


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
