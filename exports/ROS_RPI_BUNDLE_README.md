# Raspberry Pi ROS/C solver bundle

This export bundle contains both lapanda and acados C implementations.

## Solvers

- `circle_ros_mpc`: lapanda circle OCP, exported C solver plus ROS1 node.
- `rectangle_ros_mpc`: lapanda rectangle margin-imitation training benchmark, exported C solver plus ROS1 node.
- `circle_acados_exact`: acados circle OCP, `EXACT` Hessian, C interface integration plus ROS1 node.
- `rectangle_acados_gn`: acados rectangle OCP, `GAUSS_NEWTON`, C interface integration plus ROS1 node.

Each solver directory contains a `solver_config.json` with the method, problem,
tolerance setting and expected status. Current exported settings:

- Circle lapanda: `alm_tolerance = 2e-3`.
- Circle acados: `EXACT`, generated from the `2e-3` split-tolerance result.
- Rectangle lapanda: `alm_tolerance = 1e-4`, 120-epoch margin imitation benchmark by default.
- Rectangle acados: `GAUSS_NEWTON`, expected to report failure/max iterations on the current hard-constraint problem.

For the paper table's stricter circle row, use the same circle templates but
switch the circle tolerance at runtime:

```bash
ALM_TOL=1e-4 ./circle_ros_mpc/run_standalone.sh
ACADOS_TOL=1e-4 ./circle_acados_exact/run_standalone.sh
```

For ROS:

```bash
ALM_TOL=1e-4 ./circle_ros_mpc/run_ros1_node.sh
ACADOS_TOL=1e-4 ./circle_acados_exact/run_ros1_node.sh
```

Without these environment variables, the circle export uses the default `2e-3`
row. Rectangle remains `1e-4`.

## Standalone C smoke test

```bash
cd exports
./run_ros_c_bundle_standalone.sh
```

`rectangle_ros_mpc/run_standalone.sh` runs the 120-epoch margin imitation
benchmark, not just a single OCP solve. It prints one CSV-style row per epoch
and summary fields at the end.

For acados, set:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}
```

## ROS1 run

lapanda does not need acados:

```bash
cd exports/circle_ros_mpc
export ROS_DISTRO=noetic
./run_ros1_node.sh

cd ../rectangle_ros_mpc
export ROS_DISTRO=noetic
./run_ros1_node.sh
```

acados needs `ACADOS_SOURCE_DIR`:

```bash
cd exports/circle_acados_exact
export ROS_DISTRO=noetic
export ACADOS_SOURCE_DIR=/path/to/acados
./run_ros1_node.sh

cd ../rectangle_acados_gn
export ROS_DISTRO=noetic
export ACADOS_SOURCE_DIR=/path/to/acados
./run_ros1_node.sh
```

The lapanda ROS nodes publish `solver_info` with:

1. status
2. ALM outer iterations
3. final residual
4. penalty
5. forward time [s]
6. backward time [s]
7. backward iterations

The acados ROS nodes publish `~solve_info` with:

1. solve status
2. SQP iterations
3. acados reported solve time [s]
4. ROS wall-clock solve time [s]
5. KKT infinity norm
6. first control speed
7. first control steering
8. runtime acados tolerance override, or `-1` when using generated defaults
