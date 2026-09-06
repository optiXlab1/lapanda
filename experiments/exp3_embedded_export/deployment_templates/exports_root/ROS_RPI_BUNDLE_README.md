# Raspberry Pi ROS/C solver bundle

This export bundle contains both lapanda and acados C implementations.

## Solvers

- `circle_ros_mpc`: lapanda circle OCP, exported C solver plus ROS1 node.
- `rectangle_ros_mpc`: lapanda rectangle margin-imitation training benchmark, exported C solver plus ROS1 node.
- `circle_acados_exact`: acados circle OCP, `EXACT` Hessian, C interface integration plus ROS1 node.
- `rectangle_acados_gn`: acados rectangle OCP, `GAUSS_NEWTON`, C interface integration plus ROS1 node.

Each solver directory contains a `solver_config.json` or
`experiment_meta.json` describing its generated configuration. Current exported
settings are:

- Circle lapanda: horizon 12, inner tolerance `1e-1`, ALM tolerance `1e-4`.
- Circle acados: horizon 12, `EXACT` Hessian, NLP tolerance `1e-4`.
- Rectangle lapanda: horizon 20, inner tolerance `1e-3`, ALM tolerance `1e-5`.
- Rectangle acados: horizon 20, `GAUSS_NEWTON`, NLP tolerance `1e-5`; it is
  expected to reach the SQP iteration limit on the current hard-constraint
  problem.

For ROS, the circle tolerance can optionally be overridden at runtime:

```bash
ALM_TOL=1e-4 ./circle_ros_mpc/run_ros1_node.sh
ACADOS_TOL=1e-4 ./circle_acados_exact/run_ros1_node.sh
```

## Standalone C smoke test

```bash
cd exports
./run_ros_c_bundle_standalone.sh
```

The per-bundle `run_standalone.sh` scripts run one forward/backward smoke test.
To run the 150-epoch rectangular margin-imitation benchmark after building the
lapanda bundle, use:

```bash
./rectangle_ros_mpc/build/rectangle_imitation_benchmark
```

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
