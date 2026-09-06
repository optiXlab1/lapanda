# lapanda Python bindings

This directory contains the pybind11 interface for the C PANDA and augmented
Lagrangian solvers used by lapanda.

## Build

Install the Python dependencies:

```powershell
python -m pip install pybind11 casadi numpy
```

Then build the extension from the repository root:

```powershell
cmake -S . -B build-python -DBUILD_PYTHON=ON -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build-python --config Release
```

The public package is `lapanda`. Examples can be run with:

```powershell
$env:PYTHONPATH = "$PWD\python"
jupyter notebook tutorials\python_learning.ipynb
```

## Interface shape

Users define the symbolic problem with CasADi, wrap it as `CasadiProblem`, and
call `build_solver`. The returned solver object provides:

- `solver.solve_panda(...)` for plain PANDA
- `solver.solve_lapanda(...)` for the lapanda augmented Lagrangian solve
- `build_solver(..., backend="compiled")` to generate and build a CasADi C oracle; this is the default
- `build_solver(..., backend="callback")` for the pure Python callback oracle

The first supported nonsmooth term is a box indicator, exposed through
`box_lower` and `box_upper`. Smooth-only problems use the identity prox.

For ALM backward sensitivity, `BackwardOptions.constraint_penalty_scale`
multiplies the final ALM penalty only inside the constraint HVP/VJP callbacks.
`BackwardOptions.constraint_penalty_max` caps that scaled value when positive.
This keeps the forward solve unchanged while making the last ALM subproblem
backward closer to the hard active-constraint sensitivity.

Solver tracing is disabled in the Python package. The `verbose` options are
kept for API compatibility, but the solver does not write CSV traces into the
current working directory.

## Generated C oracle

The compiled backend is the first step toward the performance runtime:

```python
import lapanda

solver = lapanda.build_solver(
    problem,
    backend="compiled",
    name="nmpc_oracle",
)
print(solver.generated_oracle.library_path)
```

This writes CasADi-generated C sources, a small CMake project, metadata, and
the compiled shared library under `.lapanda_cache`. `CompiledSolver` loads
that shared library in C++ and calls the generated CasADi functions directly
for both plain PANDA and ALM+PANDA solves.
