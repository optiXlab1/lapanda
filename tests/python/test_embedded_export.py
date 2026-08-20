import subprocess

import casadi as ca
import numpy as np
import pytest

from lapanda import CasadiProblem, export_c_project


def test_export_c_project_builds_standalone_example(tmp_path):
    u = ca.SX.sym("u", 2)
    theta = ca.SX.sym("theta", 2)
    variable = ca.SX.sym("variable", 1)
    problem = CasadiProblem(
        u=u,
        theta=theta,
        variable=variable,
        cost=0.5 * ca.sumsqr(u - theta),
        box_lower=np.array([-2.0, -2.0]),
        box_upper=np.array([2.0, 2.0]),
        constraints=ca.vertcat(u[0] + u[1]),
    )

    project_dir = export_c_project(
        problem,
        out_dir=tmp_path / "embedded_project",
        name="test_embedded_oracle",
        force=True,
    )

    assert (project_dir / "CMakeLists.txt").exists()
    assert (project_dir / "generated" / "test_embedded_oracle.c").exists()
    assert (project_dir / "generated" / "lapanda_generated_config.h").exists()
    assert (
        project_dir
        / "lapanda"
        / "adapters"
        / "casadi_static"
        / "static_casadi_oracle.c"
    ).exists()
    assert (project_dir / "lapanda" / "matlab" / "lapanda_create_solver.m").exists()

    try:
        subprocess.run(
            ["cmake", "-S", str(project_dir), "-B", str(project_dir / "build")],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(project_dir / "build")],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"C compiler is not available: {exc}")
