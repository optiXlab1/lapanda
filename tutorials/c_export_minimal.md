# Standalone C Export Tutorial

This tutorial uses the Python CasADi interface to export a standalone C project.
The exported project contains the generated CasADi oracle, the lapanda C
runtime, and a small `main_example.c`.

## 1. Export A C Project

Run the Python tutorial as a starting point, or adapt this pattern:

```python
from pathlib import Path

import casadi as ca
import numpy as np

from lapanda import CasadiProblem, export_c_project

u = ca.SX.sym("u", 2)
theta = ca.SX.sym("theta", 2)
variable = ca.SX.sym("variable", 2)

problem = CasadiProblem(
    u=u,
    theta=theta,
    variable=variable,
    cost=0.5 * ca.sumsqr(u - theta),
    box_lower=np.array([-1.0, -1.0]),
    box_upper=np.array([1.0, 1.0]),
    constraints=ca.vertcat(u[0] + u[1] - 0.5),
    outer_loss=0.5 * ca.sumsqr(u - variable),
)

export_c_project(problem, Path("build/tutorial_c_export"), name="tutorial_oracle")
```

## 2. Build The Exported Project

```powershell
cmake -S build/tutorial_c_export -B build/tutorial_c_export/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/tutorial_c_export/build --config Release --parallel 4
```

On Linux or WSL:

```bash
cmake -S build/tutorial_c_export -B build/tutorial_c_export/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/tutorial_c_export/build --parallel 4
```

## 3. Run The Demo

Windows multi-config generators usually place the executable under `Release`:

```powershell
build\tutorial_c_export\build\Release\lapanda_embedded_demo.exe
```

Makefile-style generators usually place it directly in the build directory:

```bash
./build/tutorial_c_export/build/lapanda_embedded_demo
```

For embedded use, replace the placeholder arrays in `main_example.c` with the
actual runtime values for `theta`, `variable`, constraint bounds, warm starts,
and solver parameters.

The example constraint above is an equality when both bounds are zero. The
generated demo initializes runtime arrays to zero; it demonstrates the build
workflow rather than reproducing the notebook's learning run. Export does not
run a C compiler until the CMake commands above are executed.

The generated CMake project compiles the individual operator sources listed in
the export metadata. A combined C file is retained for compatibility; do not add
it to a target that already compiles the split files, or symbols will be defined
twice. `force=True` deletes and recreates the export directory, so use it only
when intentionally replacing an existing export. On Windows with MinGW, pass
`-G "MinGW Makefiles"` during the initial CMake configure; the executable then
appears directly in the build directory as `lapanda_embedded_demo.exe`.
