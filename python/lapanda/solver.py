"""Convenience solver objects for callback and generated-oracle workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ._lapanda import (
        AlmOptions,
        BackwardOptions,
        SolverOptions,
        solve_lapanda as solve_lapanda_callback,
        solve_lapanda_compiled,
        solve_panda,
        solve_panda_compiled,
    )
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR = exc

    def _missing_extension(*args, **kwargs):
        raise ImportError("The _lapanda extension is not built yet.") from _IMPORT_ERROR

    AlmOptions = BackwardOptions = SolverOptions = None
    solve_lapanda_callback = solve_lapanda_compiled = solve_panda = solve_panda_compiled = _missing_extension
from .casadi_interface import CasadiProblem, generate_oracle
from .codegen import DEFAULT_ORACLE_NAME, GeneratedOracle, generate_compiled_oracle


class CallbackSolver:
    """Object-style wrapper around the current Python callback oracle."""

    def __init__(self, problem: CasadiProblem):
        self.problem = problem
        self.oracle = generate_oracle(problem)

    def solve_panda(
        self,
        x0,
        theta,
        variable=None,
        solver_options: Optional[SolverOptions] = None,
        backward_options: Optional[BackwardOptions] = None,
        adjoint0=None,
    ):
        if variable is None:
            variable = np.zeros(self.oracle.nvar)
        if solver_options is None:
            solver_options = SolverOptions()
        if backward_options is None:
            backward_options = BackwardOptions()
        del adjoint0
        return solve_panda(self.oracle, x0, theta, variable, solver_options, backward_options)

    def solve_lapanda(
        self,
        x0,
        theta,
        variable,
        constraint_lower,
        constraint_upper,
        inner_solver_options: Optional[SolverOptions] = None,
        alm_options: Optional[AlmOptions] = None,
        backward_options: Optional[BackwardOptions] = None,
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    ):
        if inner_solver_options is None:
            inner_solver_options = SolverOptions()
        if alm_options is None:
            alm_options = AlmOptions()
        if backward_options is None:
            backward_options = BackwardOptions()
        del adjoint0
        return solve_lapanda_callback(
            self.oracle,
            x0,
            theta,
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options,
            alm_options,
            backward_options,
            multiplier0,
            penalty0,
        )

    solve_alm = solve_lapanda


class CompiledSolver(CallbackSolver):
    """Generated-oracle solver using the direct C++ dynamic-loader backend."""

    def __init__(
        self,
        problem: CasadiProblem,
        name: str = DEFAULT_ORACLE_NAME,
        cache_dir: Optional[Path] = None,
        force: bool = False,
        build: bool = True,
        cmake_generator: Optional[str] = None,
    ):
        self.problem = problem
        self.oracle = None
        self.generated_oracle: GeneratedOracle = generate_compiled_oracle(
            problem,
            name=name,
            cache_dir=cache_dir,
            build=build,
            force=force,
            cmake_generator=cmake_generator,
        )
        self.compiled_oracle = self.generated_oracle

    def solve_panda(
        self,
        x0,
        theta,
        variable=None,
        solver_options: Optional[SolverOptions] = None,
        backward_options: Optional[BackwardOptions] = None,
        adjoint0=None,
    ):
        if self.generated_oracle.library_path is None:
            raise RuntimeError("compiled oracle was generated without a built library")
        if variable is None:
            variable = np.zeros(self.generated_oracle.metadata["nvar"])
        if solver_options is None:
            solver_options = SolverOptions()
        if backward_options is None:
            backward_options = BackwardOptions()
        metadata = self.generated_oracle.metadata
        del adjoint0
        return solve_panda_compiled(
            str(self.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["box_lower"],
            metadata["box_upper"],
            x0,
            theta,
            variable,
            solver_options,
            backward_options,
        )

    def solve_lapanda(
        self,
        x0,
        theta,
        variable,
        constraint_lower,
        constraint_upper,
        inner_solver_options: Optional[SolverOptions] = None,
        alm_options: Optional[AlmOptions] = None,
        backward_options: Optional[BackwardOptions] = None,
        multiplier0=None,
        penalty0=None,
        adjoint0=None,
    ):
        if self.generated_oracle.library_path is None:
            raise RuntimeError("compiled oracle was generated without a built library")
        if inner_solver_options is None:
            inner_solver_options = SolverOptions()
        if alm_options is None:
            alm_options = AlmOptions()
        if backward_options is None:
            backward_options = BackwardOptions()
        metadata = self.generated_oracle.metadata
        del adjoint0
        return solve_lapanda_compiled(
            str(self.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["ncon"],
            metadata["box_lower"],
            metadata["box_upper"],
            x0,
            theta,
            variable,
            constraint_lower,
            constraint_upper,
            inner_solver_options,
            alm_options,
            backward_options,
            multiplier0,
            penalty0,
        )

    solve_alm = solve_lapanda


def build_solver(
    problem: CasadiProblem,
    backend: str = "compiled",
    name: str = DEFAULT_ORACLE_NAME,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    build: bool = True,
    cmake_generator: Optional[str] = None,
):
    if backend == "callback":
        return CallbackSolver(problem)
    if backend == "compiled":
        return CompiledSolver(
            problem,
            name=name,
            cache_dir=cache_dir,
            force=force,
            build=build,
            cmake_generator=cmake_generator,
        )
    raise ValueError(f"unknown backend: {backend!r}")
