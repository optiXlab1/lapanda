# circle_acados_exact

This is an acados C interface package: generated solver sources are
compiled inside this project, while `main_acados.c` is the application-owned
entry point that sets the experiment's `p_global`, initial state constraint and
cold-start guess.

Build:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/circle_acados_exact
```

ROS1 one-shot run:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
./run_ros1_node.sh
```
