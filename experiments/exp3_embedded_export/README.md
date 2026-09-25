# Exp. 3: LIMO vehicle experiments

This experiment evaluates lapanda on an AgileX LIMO vehicle running ROS1. The
circle task measures repeated MPC solution performance, while the rectangle
task trains the MPC parameters and deploys the resulting controller for
obstacle avoidance. The Python files in this directory generate the standalone
C and ROS projects; the reported timing and memory results are measured on the
embedded platform rather than through the Python interface.

## Generate the deployment bundle

Run the export scripts from the repository root:

```powershell
python experiments\exp3_embedded_export\circle\export_circle_ros_project.py --force
python experiments\exp3_embedded_export\rectangle\export_rectangle_ros_project.py --force
python experiments\exp3_embedded_export\export_acados_bundles.py --force
```

These commands populate `exports/` with the lapanda and acados standalone C
projects, their ROS1 wrappers, recorded solver configurations, and validation
scripts.

## Run on the embedded platform

Copy the complete `exports/` directory to the LIMO computer. On the device,
first verify that the generated sources and recorded settings agree:

```bash
cd exports
python3 validate_export_settings.py
```

The standalone executables are used to collect solver timing and memory, while
the ROS1 nodes connect the controller to the LIMO state, obstacle, target, and
control topics. Build commands, standalone benchmarks, acados environment
variables, and ROS1 launch commands are collected in `exports/README.md`.

The retained embedded measurements and figures are stored under `results/`;
`results/summary.csv` contains the values reported in the paper.
