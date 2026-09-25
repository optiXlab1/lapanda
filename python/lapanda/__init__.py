"""Public Python interface for the lapanda differentiable solver."""

try:
    from ._lapanda import (
        AlmOptions,
        BackwardOptions,
        Oracle,
        SolverOptions,
        solve_dense_minres,
    )
except ImportError as exc:  # pragma: no cover - useful before the extension is built.
    _IMPORT_ERROR = exc

    def _missing_extension(*args, **kwargs):
        raise ImportError(
            "The lapanda native extension is not built yet. Build with "
            "CMake option BUILD_PYTHON=ON and make sure pybind11 is installed."
        ) from _IMPORT_ERROR

    AlmOptions = BackwardOptions = Oracle = SolverOptions = solve_dense_minres = None

from .casadi_interface import CasadiProblem
from .codegen import CompiledOracle, GeneratedOracle, export_c_project, generate_compiled_oracle
from .solver import CallbackSolver, CompiledSolver, build_solver

__all__ = [
    "AlmOptions",
    "BackwardOptions",
    "CallbackSolver",
    "CasadiProblem",
    "CompiledOracle",
    "CompiledSolver",
    "GeneratedOracle",
    "Oracle",
    "SolverOptions",
    "solve_dense_minres",
    "build_solver",
    "export_c_project",
    "generate_compiled_oracle",
]
