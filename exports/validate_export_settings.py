#!/usr/bin/env python3
"""Check that exported embedded benchmarks use their recorded settings."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ERRORS: list[str] = []


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def load(relative: str):
    return json.loads(read(relative))


def require(relative: str, *snippets: str) -> None:
    text = read(relative)
    for snippet in snippets:
        if snippet not in text:
            ERRORS.append(f"{relative}: missing {snippet!r}")


def equal(label: str, actual, expected) -> None:
    if actual != expected:
        ERRORS.append(f"{label}: expected {expected!r}, got {actual!r}")


def close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
        ERRORS.append(f"{label}: expected {expected}, got {actual}")


def validate_lapanda_copies() -> None:
    """Ensure the two standalone bundles carry the same solver core."""
    circle_root = ROOT / "circle_ros_mpc/lapanda"
    rectangle_root = ROOT / "rectangle_ros_mpc/lapanda"
    circle_files = {
        path.relative_to(circle_root) for path in circle_root.rglob("*") if path.is_file()
    }
    rectangle_files = {
        path.relative_to(rectangle_root)
        for path in rectangle_root.rglob("*")
        if path.is_file()
    }
    equal("lapanda bundled file sets", circle_files, rectangle_files)
    for relative in sorted(circle_files & rectangle_files):
        lhs = (circle_root / relative).read_bytes().replace(b"\r\n", b"\n")
        rhs = (rectangle_root / relative).read_bytes().replace(b"\r\n", b"\n")
        if lhs != rhs:
            ERRORS.append(f"lapanda/{relative}: differs between exported bundles")


circle = load("circle_ros_mpc/solver_config.json")
equal("circle horizon", circle["horizon"], 12)
close("circle inner tolerance", circle["tolerance"]["inner_tolerance"], 1e-1)
close("circle ALM tolerance", circle["tolerance"]["alm_tolerance"], 2e-3)
close("circle backward tolerance", circle["tolerance"]["backward_tolerance"], 1e-3)
equal("circle backward solver", circle["backward_solver"], "CG")
require(
    "circle_ros_mpc/main_circle_benchmark.c",
    "constraint_lower[i] = -1.0e20;",
    "solver_params.max_iterations = 2000;",
    "solver_params.tolerance = 1e-1;",
    "backward_params.tolerance = 1e-3;",
    "backward_params.max_iterations = 200;",
    "backward_params.restart = 40;",
    "backward_params.force_solver = PANDA_BACKWARD_SOLVER_CG;",
    "params.max_iterations = 100;",
    "params.tolerance = 2e-3;",
    "params.initial_penalty = 10000.0;",
    "params.penalty_update_factor = 10.0;",
    "params.warm_start_inner = 1;",
)
require(
    "circle_ros_mpc/ros1_circle_mpc/src/circle_mpc_node.cpp",
    'private_nh_.param("alm_tolerance", alm_tolerance_, 2e-3);',
    "backward.force_solver = PANDA_BACKWARD_SOLVER_CG;",
)

circle_meta_files = list((ROOT / "circle_ros_mpc/generated").glob("circle_mpc_oracle_*/metadata.json"))
equal("circle oracle metadata count", len(circle_meta_files), 1)
if circle_meta_files:
    metadata = json.loads(circle_meta_files[0].read_text(encoding="utf-8"))
    equal("circle decision dimension", metadata["n"], 24)
    equal("circle constraint dimension", metadata["ncon"], 12)
    close("circle speed upper bound", metadata["box_upper"][0], 1.5)
    close("circle steering upper bound", metadata["box_upper"][1], 0.4363323129985824)

rectangle = load("rectangle_ros_mpc/solver_config.json")
equal("rectangle horizon", rectangle["horizon"], 20)
close("rectangle inner tolerance", rectangle["tolerance"]["inner_tolerance"], 1e-3)
close("rectangle ALM tolerance", rectangle["tolerance"]["alm_tolerance"], 1e-4)
close("rectangle backward tolerance", rectangle["tolerance"]["backward_tolerance"], 1e-3)
equal("rectangle backward solver", rectangle["backward_solver"], "CG")
equal("rectangle epochs", rectangle["training"]["epochs"], 150)
close("rectangle learning rate", rectangle["training"]["learning_rate"], 1e-3)
equal("rectangle teacher margins", rectangle["training"]["teacher_margins"], [0.12, 0.12, 0.01, 0.22])
equal("rectangle initial margins", rectangle["training"]["initial_margins"], [0.15, 0.15, 0.16, 0.22])
for source in (
    "rectangle_ros_mpc/main_rectangle_benchmark.c",
    "rectangle_ros_mpc/main_rectangle_imitation_benchmark.c",
):
    require(source, "constraint_lower[i] = 0.0;", "constraint_upper[i] = 0.0;")
require(
    "rectangle_ros_mpc/main_rectangle_imitation_benchmark.c",
    "#define EPOCHS 150",
    "#define LR 1.0e-3",
    "solver_params->max_iterations = 2000;",
    "solver_params->tolerance = 1.0e-3;",
    "backward_params->tolerance = 1.0e-3;",
    "backward_params->max_iterations = 200;",
    "backward_params->restart = 40;",
    "backward_params->force_solver = PANDA_BACKWARD_SOLVER_CG;",
    "params->max_iterations = 100;",
    "params->tolerance = 1.0e-4;",
    "params->initial_penalty = 10000.0;",
    "params->penalty_update_factor = 10.0;",
    "params->warm_start_inner = 1;",
)
require(
    "rectangle_ros_mpc/ros1_rectangle_mpc/src/rectangle_mpc_node.cpp",
    'private_nh_.param("alm_tolerance", alm_tolerance_, 1e-4);',
    "backward.force_solver = PANDA_BACKWARD_SOLVER_CG;",
)

rectangle_meta_files = list((ROOT / "rectangle_ros_mpc/generated").glob("rectangle_mpc_oracle_*/metadata.json"))
equal("rectangle oracle metadata count", len(rectangle_meta_files), 1)
if rectangle_meta_files:
    metadata = json.loads(rectangle_meta_files[0].read_text(encoding="utf-8"))
    equal("rectangle decision dimension", metadata["n"], 40)
    equal("rectangle constraint dimension", metadata["ncon"], 20)
    close("rectangle speed upper bound", metadata["box_upper"][0], 1.0)
    close("rectangle steering upper bound", metadata["box_upper"][1], 0.7)

circle_acados = load("circle_acados_exact/solver_config.json")
close("circle acados tolerance", circle_acados["tolerance"]["nlp_tol_reference"], 2e-3)
equal("circle acados NLP iterations", circle_acados["maximum_iterations"]["nlp"], 1000)
equal("circle acados QP iterations", circle_acados["maximum_iterations"]["qp"], 200)
require(
    "circle_acados_exact/c_generated_code/acados_solver_circle_forward.c",
    "double nlp_solver_tol_stat = 0.002;",
    "double nlp_solver_tol_eq = 0.002;",
    "double nlp_solver_tol_ineq = 0.002;",
    "double nlp_solver_tol_comp = 0.002;",
    "int nlp_solver_max_iter = 1000;",
    "int qp_solver_iter_max = 200;",
)
require(
    "circle_acados_exact/c_generated_sensitivity/acados_solver_circle_sensitivity.c",
    "double nlp_solver_tol_stat = 0.002;",
    "double nlp_solver_tol_eq = 0.002;",
    "double nlp_solver_tol_ineq = 0.002;",
    "double nlp_solver_tol_comp = 0.002;",
)

rectangle_acados = load("rectangle_acados_gn/solver_config.json")
close("rectangle acados tolerance", rectangle_acados["tolerance"]["nlp_tol_reference"], 1e-4)
equal("rectangle acados NLP iterations", rectangle_acados["maximum_iterations"]["nlp"], 1000)
equal("rectangle acados QP iterations", rectangle_acados["maximum_iterations"]["qp"], 50)
require(
    "rectangle_acados_gn/c_generated_code/acados_solver_rect_margin_forward.c",
    "double nlp_solver_tol_stat = 0.0001;",
    "double nlp_solver_tol_eq = 0.0001;",
    "double nlp_solver_tol_ineq = 0.0001;",
    "double nlp_solver_tol_comp = 0.0001;",
    "int nlp_solver_max_iter = 1000;",
    "int qp_solver_iter_max = 50;",
)

for script in (
    "circle_ros_mpc/run_standalone.sh",
    "rectangle_ros_mpc/run_standalone.sh",
    "circle_acados_exact/run_standalone.sh",
    "rectangle_acados_gn/run_standalone.sh",
):
    require(script, "-DCMAKE_BUILD_TYPE=Release")

validate_lapanda_copies()

if ERRORS:
    print("Export validation FAILED:", file=sys.stderr)
    for error in ERRORS:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("Export validation passed: circle/rectangle lapanda and acados settings are consistent.")
