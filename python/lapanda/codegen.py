"""CasADi C-code generation for lapanda oracle functions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import casadi as ca

from .casadi_interface import CasadiProblem, _flat_array, _numel


DEFAULT_ORACLE_NAME = "lapanda_oracle"
DEFAULT_CACHE_DIR = ".lapanda_cache"


@dataclass
class GeneratedOracle:
    """CasADi-generated C oracle files and optional shared library."""

    name: str
    source_path: Path
    header_path: Path
    build_dir: Path
    library_path: Optional[Path]
    metadata_path: Path
    metadata: dict


CompiledOracle = GeneratedOracle


def _default_cache_dir() -> Path:
    return Path.cwd() / DEFAULT_CACHE_DIR


def _problem_fingerprint(problem: CasadiProblem, name: str) -> str:
    h = hashlib.sha256()
    pieces = [
        name,
        str(ca.vec(problem.u)),
        str(ca.vec(problem.theta)),
        str(ca.vec(problem.variable)),
        str(problem.cost),
        str(problem.constraints),
        str(problem.outer_loss),
        str(None if problem.box_lower is None else _flat_array(problem.box_lower).tolist()),
        str(None if problem.box_upper is None else _flat_array(problem.box_upper).tolist()),
    ]
    for piece in pieces:
        h.update(piece.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _shared_library_name(name: str) -> str:
    system = platform.system().lower()
    if system == "windows":
        return f"{name}.dll"
    if system == "darwin":
        return f"lib{name}.dylib"
    return f"lib{name}.so"


def _make_oracle_functions(problem: CasadiProblem) -> Tuple[List[ca.Function], Dict]:
    u = ca.vec(problem.u)
    theta = ca.vec(problem.theta)
    variable = ca.vec(problem.variable)
    cost = problem.cost

    n = _numel(u)
    ntheta = _numel(theta)
    nvar = _numel(variable)

    v = ca.SX.sym("v", n)
    grad = ca.gradient(cost, u)
    functions = [
        ca.Function("panda_cost_grad", [u, theta, variable], [cost, ca.densify(grad)]),
        ca.Function("panda_hvp", [v, u, theta, variable], [ca.densify(ca.jtimes(grad, u, v))]),
        ca.Function(
            "panda_vjp",
            [v, u, theta, variable],
            [ca.densify(ca.jacobian(grad, theta).T @ v)],
        ),
    ]

    if problem.outer_loss is not None:
        loss_grad = ca.gradient(problem.outer_loss, u)
        functions.append(
            ca.Function(
                "panda_loss_grad",
                [u, theta, variable],
                [problem.outer_loss, ca.densify(loss_grad)],
            )
        )

    ncon = 0
    if problem.constraints is not None:
        c = ca.vec(problem.constraints)
        ncon = _numel(c)
        weight = ca.SX.sym("weight", ncon)
        jtprod = ca.jacobian(c, u).T @ weight
        functions.extend(
            [
                ca.Function("panda_constraints", [u, theta, variable], [c]),
                ca.Function(
                    "panda_constraint_jtprod",
                    [u, theta, variable, weight],
                    [ca.densify(jtprod)],
                ),
                ca.Function(
                    "panda_constraint_jv",
                    [v, u, theta, variable],
                    [ca.densify(ca.jtimes(c, u, v))],
                ),
                ca.Function(
                    "panda_constraint_weighted_hvp",
                    [v, u, theta, variable, weight],
                    [ca.densify(ca.jtimes(jtprod, u, v))],
                ),
                ca.Function(
                    "panda_constraint_weighted_vjp",
                    [v, u, theta, variable, weight],
                    [ca.densify(ca.jacobian(jtprod, theta).T @ v)],
                ),
                ca.Function(
                    "panda_constraint_theta_jtprod",
                    [u, theta, variable, weight],
                    [ca.densify(ca.jacobian(c, theta).T @ weight)],
                ),
            ]
        )

    metadata = {
        "n": n,
        "ntheta": ntheta,
        "nvar": nvar,
        "ncon": ncon,
        "has_box": problem.box_lower is not None or problem.box_upper is not None,
        "box_lower": None if problem.box_lower is None else _flat_array(problem.box_lower).tolist(),
        "box_upper": None if problem.box_upper is None else _flat_array(problem.box_upper).tolist(),
        "functions": [f.name() for f in functions],
    }
    return functions, metadata


def generate_compiled_oracle(
    problem: CasadiProblem,
    name: str = DEFAULT_ORACLE_NAME,
    cache_dir: Optional[Path] = None,
    build: bool = True,
    force: bool = False,
    cmake_generator: Optional[str] = None,
) -> GeneratedOracle:
    """Generate CasADi C code and optionally compile it as a shared library.

    The generated library is intentionally runtime-neutral: Python and MATLAB
    frontends can both produce the same C ABI, and the C++ runtime can load it
    directly in the next performance step.
    """

    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    fingerprint = _problem_fingerprint(problem, name)
    work_dir = cache_root / f"{name}_{fingerprint}"
    source_path = work_dir / f"{name}.c"
    header_path = work_dir / f"{name}.h"
    metadata_path = work_dir / "metadata.json"
    build_dir = work_dir / "build"
    library_path = build_dir / _shared_library_name(name)

    if force and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cached_library_path = _find_built_library(build_dir, name) if build else None
    cache_ready = (
        not force
        and source_path.exists()
        and header_path.exists()
        and metadata_path.exists()
        and (not build or cached_library_path is not None)
    )
    if cache_ready:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return GeneratedOracle(
            name=name,
            source_path=source_path,
            header_path=header_path,
            build_dir=build_dir,
            library_path=cached_library_path if build else None,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    functions, metadata = _make_oracle_functions(problem)
    metadata.update({"name": name, "fingerprint": fingerprint})

    if force or not source_path.exists() or not header_path.exists():
        old_cwd = Path.cwd()
        try:
            os.chdir(work_dir)
            generator = ca.CodeGenerator(source_path.name, {"with_header": True})
            for fun in functions:
                generator.add(fun)
            generator.generate()
        finally:
            os.chdir(old_cwd)

    if force or not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if build and (force or not library_path.exists()):
        _write_cmake_project(work_dir, name, source_path.name)
        _build_shared_library(work_dir, build_dir, cmake_generator)

    library_path = _find_built_library(build_dir, name) if build else None

    return GeneratedOracle(
        name=name,
        source_path=source_path,
        header_path=header_path,
        build_dir=build_dir,
        library_path=library_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def export_c_project(
    problem: CasadiProblem,
    out_dir,
    name: str = DEFAULT_ORACLE_NAME,
    force: bool = False,
) -> Path:
    """Export a standalone C project for embedded/Linux deployment."""

    out_path = Path(out_dir)
    if force and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_dir = out_path / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    artifact = generate_compiled_oracle(
        problem,
        name=name,
        cache_dir=generated_dir,
        build=False,
        force=True,
    )
    oracle_source = generated_dir / f"{name}.c"
    oracle_header = generated_dir / f"{name}.h"
    shutil.move(str(artifact.source_path), oracle_source)
    shutil.move(str(artifact.header_path), oracle_header)
    _write_static_config(problem, artifact.metadata, name, generated_dir / "lapanda_generated_config.h")
    _copy_embedded_sources(out_path)
    _write_embedded_cmake(out_path, name)
    _write_example_main(out_path)
    return out_path


def _write_static_config(problem: CasadiProblem, metadata: Dict, name: str, path: Path) -> None:
    lower = metadata["box_lower"] or []
    upper = metadata["box_upper"] or []
    has_lower = int(metadata["box_lower"] is not None)
    has_upper = int(metadata["box_upper"] is not None)
    has_outer_loss = int(problem.outer_loss is not None)

    def array_initializer(values):
        if not values:
            return "{0.0}"
        return "{" + ", ".join(f"{float(v):.17g}" for v in values) + "}"

    text = f"""#ifndef LAPANDA_GENERATED_CONFIG_H
#define LAPANDA_GENERATED_CONFIG_H

#include "{name}.h"

#define LAPANDA_N {metadata["n"]}
#define LAPANDA_NTHETA {metadata["ntheta"]}
#define LAPANDA_NVAR {metadata["nvar"]}
#define LAPANDA_NCON {metadata["ncon"]}
#define LAPANDA_HAS_BOX_LOWER {has_lower}
#define LAPANDA_HAS_BOX_UPPER {has_upper}
#define LAPANDA_HAS_OUTER_LOSS {has_outer_loss}

static const double LAPANDA_BOX_LOWER[LAPANDA_N] = {array_initializer(lower)};
static const double LAPANDA_BOX_UPPER[LAPANDA_N] = {array_initializer(upper)};

#endif
"""
    path.write_text(text, encoding="utf-8")


def _copy_embedded_sources(out_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    for dirname in ["panda", "alm", "include", "globals", "adapters", "matlab"]:
        src = root / dirname
        dst = out_path / "lapanda" / dirname
        if dst.exists():
            shutil.rmtree(dst)
        if dirname == "matlab":
            dst.mkdir(parents=True, exist_ok=True)
            for pattern in ("*.m", "*.c"):
                for file_path in src.glob(pattern):
                    shutil.copy2(file_path, dst / file_path.name)
            continue
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("CMakeLists.txt", "__pycache__"),
        )


def _write_embedded_cmake(out_path: Path, oracle_name: str) -> None:
    text = f"""cmake_minimum_required(VERSION 3.15)
project(lapanda_embedded_example C)

set(LAPANDA_ROOT ${{CMAKE_CURRENT_SOURCE_DIR}}/lapanda)

add_library(lapanda_embedded STATIC
    generated/{oracle_name}.c
    ${{LAPANDA_ROOT}}/adapters/casadi_static/static_casadi_oracle.c
    ${{LAPANDA_ROOT}}/panda/function_evaluator.c
    ${{LAPANDA_ROOT}}/panda/matrix_operations.c
    ${{LAPANDA_ROOT}}/panda/lbfgs.c
    ${{LAPANDA_ROOT}}/panda/panda_eval.c
    ${{LAPANDA_ROOT}}/panda/panda_forward.c
    ${{LAPANDA_ROOT}}/panda/panda_linear_solver.c
    ${{LAPANDA_ROOT}}/panda/panda_backward.c
    ${{LAPANDA_ROOT}}/panda/optimizer.c
    ${{LAPANDA_ROOT}}/alm/alm.c
)

target_include_directories(lapanda_embedded PUBLIC
    ${{CMAKE_CURRENT_SOURCE_DIR}}/generated
    ${{LAPANDA_ROOT}}/adapters/casadi_static
    ${{LAPANDA_ROOT}}/include
    ${{LAPANDA_ROOT}}/panda
    ${{LAPANDA_ROOT}}/alm
    ${{LAPANDA_ROOT}}/globals
)

add_executable(lapanda_embedded_demo main_example.c)
target_link_libraries(lapanda_embedded_demo PRIVATE lapanda_embedded)
"""
    (out_path / "CMakeLists.txt").write_text(text, encoding="utf-8")


def _write_example_main(out_path: Path) -> None:
    text = """#include "static_casadi_oracle.h"
#include "lapanda_generated_config.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    double solution[LAPANDA_N];
    double theta[LAPANDA_NTHETA];
    double variable[LAPANDA_NVAR > 0 ? LAPANDA_NVAR : 1];
    unsigned int i;

    memset(solution, 0, sizeof(solution));
    memset(theta, 0, sizeof(theta));
    memset(variable, 0, sizeof(variable));

    solver_params.max_iterations = 1000;
    solver_params.tolerance = 1e-4;
    solver_params.buffer_size = 10;
    solver_params.max_stable_iter = 80;
    solver_params.verbose = 0;

    backward_params.enable = 0;
    backward_params.tolerance = 1e-8;
    backward_params.max_iterations = 800;

#if LAPANDA_NCON > 0
    {
        alm_problem problem;
        alm_parameters params;
        alm_info info;
        double multipliers[LAPANDA_NCON];
        double constraint_lower[LAPANDA_NCON];
        double constraint_upper[LAPANDA_NCON];

        memset(multipliers, 0, sizeof(multipliers));
        for (i = 0; i < LAPANDA_NCON; ++i) {
            constraint_lower[i] = 0.0;
            constraint_upper[i] = 0.0;
        }

        lapanda_static_init_alm_problem(
            &problem,
            constraint_lower,
            constraint_upper,
            &solver_params,
            &backward_params);

        params.max_iterations = 50;
        params.tolerance = 1e-4;
        params.initial_penalty = 1.0;
        params.penalty_update_factor = 2.0;
        params.max_penalty = 0.0;
        params.sufficient_decrease_factor = 0.25;
        params.verbose = 0;

        (void)alm_solve(&problem, &params, solution, multipliers, theta, variable, &info);
        printf("ALM iterations: %u, residual: %.6e\\n", info.iterations, info.final_residual);
    }
#else
    {
        struct optimizer_problem problem;
        optimizer_solve_info info;

        lapanda_static_init_panda_problem(&problem, &solver_params, &backward_params);
        (void)optimizer_init(&problem);
        (void)solve_problem(solution, theta, variable);
        (void)optimizer_get_forward_info(&info);
        (void)optimizer_cleanup();
        printf("PANDA iterations: %u, residual: %.6e\\n", info.iterations, info.final_residual);
    }
#endif

    for (i = 0; i < LAPANDA_N; ++i) {
        printf("solution[%u] = %.16e\\n", i, solution[i]);
    }
    return 0;
}
"""
    (out_path / "main_example.c").write_text(text, encoding="utf-8")


def _write_cmake_project(work_dir: Path, target_name: str, source_name: str) -> None:
    cmake_text = f"""cmake_minimum_required(VERSION 3.15)
project({target_name}_oracle C)
add_library({target_name} SHARED {source_name})
set_target_properties({target_name} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}
    LIBRARY_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}
    ARCHIVE_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}
)
if(MINGW)
    target_link_options({target_name} PRIVATE -static-libgcc)
endif()
"""
    (work_dir / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")


def _build_shared_library(work_dir: Path, build_dir: Path, cmake_generator: Optional[str]) -> None:
    configure_cmd = ["cmake", "-S", str(work_dir), "-B", str(build_dir)]
    cmake_defines = []
    if cmake_generator is None:
        cmake_generator = _default_cmake_generator()
    if cmake_generator:
        configure_cmd.extend(["-G", cmake_generator])
        if cmake_generator == "MinGW Makefiles":
            gcc = _find_executable("gcc.exe", [])
            make = _find_executable("mingw32-make.exe", [])
            if gcc:
                cmake_defines.append(f"-DCMAKE_C_COMPILER={_cmake_path(gcc)}")
            if make:
                cmake_defines.append(f"-DCMAKE_MAKE_PROGRAM={_cmake_path(make)}")
    configure_cmd.extend(cmake_defines)
    subprocess.run(configure_cmd, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release"], check=True)


def _default_cmake_generator() -> Optional[str]:
    if platform.system().lower() != "windows":
        return None
    if _find_executable("gcc.exe", []) and _find_executable("mingw32-make.exe", []):
        return "MinGW Makefiles"
    return None


def _find_executable(name: str, fallbacks) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for item in fallbacks:
        path = Path(item)
        if path.exists():
            return str(path)
    return None


def _cmake_path(path: str) -> str:
    return Path(path).as_posix()


def _find_built_library(build_dir: Path, name: str) -> Optional[Path]:
    candidates = [_shared_library_name(name)]
    if platform.system().lower() == "windows":
        candidates.append(f"lib{name}.dll")
    for expected in candidates:
        direct = build_dir / expected
        if direct.exists():
            return direct
        matches = list(build_dir.rglob(expected))
        if matches:
            return matches[0]
    return None
