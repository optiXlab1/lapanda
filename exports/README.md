# Embedded solver exports

This directory contains the standalone C and ROS1 bundles used in Exp. 3:

- `circle_ros_mpc` and `rectangle_ros_mpc`: lapanda;
- `circle_acados_exact` and `rectangle_acados_gn`: acados.

The recorded settings can be checked before building:

```bash
python3 validate_export_settings.py
```

Run the standalone smoke tests or the repeated lapanda timing benchmark with:

```bash
./run_ros_c_bundle_standalone.sh
bash ./run_lapanda_benchmark.sh 20 /tmp/lapanda_timing.csv
```

The rectangle learning executable is:

```bash
./rectangle_ros_mpc/build/rectangle_imitation_benchmark
```

For acados, define its installation directory first:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}
```

The standalone acados bundles can then be built with CMake, for example:

```bash
cmake -S circle_acados_exact -B circle_acados_exact/build -DCMAKE_BUILD_TYPE=Release
cmake --build circle_acados_exact/build -j2
./circle_acados_exact/build/circle_acados_exact
```

For ROS1, set `ROS_DISTRO=noetic` and run the wrapper supplied by each bundle:

```bash
export ROS_DISTRO=noetic
./circle_ros_mpc/run_ros1_node.sh
./rectangle_ros_mpc/run_ros1_node.sh
./circle_acados_exact/run_ros1_node.sh
./rectangle_acados_gn/run_ros1_node.sh
```

The circle lapanda node subscribes to `state`, `target`, `circle_obstacle`, and
an optional full `theta` vector. It publishes the first control and
`solver_info`. The acados nodes publish the corresponding `solve_info` array.
