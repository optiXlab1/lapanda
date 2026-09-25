"""Generate reproducible acados standalone and ROS deployment bundles.

Run this script from an environment containing ``acados_template``.  It owns
the complete contents of ``exports/circle_acados_exact`` and
``exports/rectangle_acados_gn``: acados generates the solver layer, while the
versioned deployment templates provide the application and ROS layers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
TEMPLATE_ROOT = EXP_DIR / "deployment_templates"
DEFAULT_EXPORTS_ROOT = REPO_ROOT / "exports"
SOURCE_SUFFIXES = {".c", ".h", ".pxd"}

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import config as exp3_config


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_output(path: Path, exports_root: Path, force: bool) -> None:
    resolved = path.resolve()
    resolved_root = exports_root.resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"refusing to export outside {resolved_root}: {resolved}")
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_generated_sources(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in source.rglob("*"):
        if not src.is_file() or src.suffix not in SOURCE_SUFFIXES:
            continue
        dst = destination / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_templates(name: str, destination: Path) -> None:
    shutil.copytree(TEMPLATE_ROOT / name, destination, dirs_exist_ok=True)
    for script in destination.rglob("*.sh"):
        content = script.read_text(encoding="utf-8")
        with script.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def bundle_files(output: Path) -> list[str]:
    return sorted(
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file() and path.name != "export_manifest.json"
    )


def export_circle(exports_root: Path, force: bool) -> None:
    output = exports_root / "circle_acados_exact"
    prepare_output(output, exports_root, force)
    copy_templates("circle_acados_exact", output)

    circle = load_module("exp3_circle_acados_export", EXP_DIR / "circle" / "acados.py")
    case = circle.build_circle_case(
        horizon=exp3_config.CIRCLE_HORIZON,
        speed_limit=exp3_config.CIRCLE_SPEED_LIMIT,
    )
    with tempfile.TemporaryDirectory(prefix="lapanda_circle_acados_export_") as temp:
        from acados_template import AcadosOcpSolver

        temp_path = Path(temp)
        forward_ocp = circle.build_ocp(
            case,
            temp_path,
            "circle_forward",
            False,
            "exact",
            exp3_config.CIRCLE_ACADOS_MAX_ITER,
            "MIRROR",
            "PARTIAL_CONDENSING_HPIPM",
            -1e3,
            exp3_config.CIRCLE_ACADOS_QP_MAX_ITER,
            "MERIT_BACKTRACKING",
            0.0,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
        )
        AcadosOcpSolver.generate(
            forward_ocp,
            json_file=str(temp_path / "acados_forward_ocp.json"),
            verbose=False,
        )
        sensitivity_ocp = circle.build_ocp(
            case,
            temp_path,
            "circle_sensitivity",
            True,
            "exact",
            exp3_config.CIRCLE_ACADOS_MAX_ITER,
            "MIRROR",
            "PARTIAL_CONDENSING_HPIPM",
            -1e3,
            exp3_config.CIRCLE_ACADOS_QP_MAX_ITER,
            "MERIT_BACKTRACKING",
            0.0,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
            exp3_config.CIRCLE_ACADOS_TOL,
        )
        AcadosOcpSolver.generate(
            sensitivity_ocp,
            json_file=str(temp_path / "acados_sensitivity_ocp.json"),
            verbose=False,
        )
        copy_generated_sources(
            temp_path / "acados_codegen_circle_forward",
            output / "c_generated_code",
        )
        copy_generated_sources(
            temp_path / "acados_codegen_circle_sensitivity",
            output / "c_generated_sensitivity",
        )

    ros = output / "ros1_circle_acados"
    shutil.copytree(output / "c_generated_code", ros / "c_generated_code")
    shutil.copytree(output / "c_generated_sensitivity", ros / "c_generated_sensitivity")
    write_json(
        output / "solver_config.json",
        {
            "method": "acados",
            "problem": "circle",
            "backend": "C interface",
            "hessian": "EXACT",
            "horizon": exp3_config.CIRCLE_HORIZON,
            "dynamics": "discrete bicycle",
            "dt": case.dt,
            "wheelbase": circle.WHEELBASE,
            "speed_bounds": [
                -exp3_config.CIRCLE_SPEED_LIMIT,
                exp3_config.CIRCLE_SPEED_LIMIT,
            ],
            "steering_bounds_rad": [-float(circle.STEER_LIMIT), float(circle.STEER_LIMIT)],
            "tolerance": {"nlp_tol_reference": exp3_config.CIRCLE_ACADOS_TOL},
            "maximum_iterations": {
                "qp": exp3_config.CIRCLE_ACADOS_QP_MAX_ITER,
                "nlp": exp3_config.CIRCLE_ACADOS_MAX_ITER,
            },
            "runtime_overrides": {
                "standalone_env": "ACADOS_TOL",
                "ros_private_param": "acados_tol",
            },
            "expected_status": 0,
        },
    )
    write_json(
        output / "export_manifest.json",
        {
            "generator": "experiments/exp3_embedded_export/export_acados_bundles.py",
            "solver_generator": "acados_template",
            "template": "deployment_templates/circle_acados_exact",
            "files": bundle_files(output),
        },
    )


def export_rectangle(exports_root: Path, force: bool) -> None:
    output = exports_root / "rectangle_acados_gn"
    prepare_output(output, exports_root, force)
    copy_templates("rectangle_acados_gn", output)

    rectangle = load_module(
        "exp3_rectangle_acados_export",
        EXP_DIR / "rectangle" / "acados.py",
    )
    settings = SimpleNamespace(
        acados_forward_mode="gn",
        acados_max_iter=exp3_config.RECTANGLE_ACADOS_MAX_ITER,
        acados_regularize="MIRROR",
        acados_tol=exp3_config.RECTANGLE_ACADOS_TOL,
        constraint_power=2,
    )
    case = rectangle.rectangle_case_with_margins(
        exp3_config.RECTANGLE_EXPORT_MARGINS,
        horizon=exp3_config.RECTANGLE_HORIZON,
        speed_limit=exp3_config.RECTANGLE_SPEED_LIMIT,
        steer_limit=exp3_config.RECTANGLE_STEER_LIMIT,
    )
    with tempfile.TemporaryDirectory(prefix="lapanda_rectangle_acados_export_") as temp:
        from acados_template import AcadosOcpSolver

        temp_path = Path(temp)
        forward_ocp = rectangle.build_acados_ocp(
            case,
            "rect_margin_forward",
            temp_path,
            False,
            settings,
        )
        AcadosOcpSolver.generate(
            forward_ocp,
            json_file=str(temp_path / "acados_forward_ocp.json"),
            verbose=False,
        )
        copy_generated_sources(
            temp_path / "acados_codegen_rect_margin_forward",
            output / "c_generated_code",
        )

    ros = output / "ros1_rectangle_acados"
    shutil.copytree(output / "c_generated_code", ros / "c_generated_code")
    write_json(
        output / "solver_config.json",
        {
            "method": "acados",
            "problem": "rectangle",
            "backend": "C interface",
            "hessian": "GAUSS_NEWTON",
            "horizon": exp3_config.RECTANGLE_HORIZON,
            "dynamics": "discrete bicycle",
            "dt": case.dt,
            "wheelbase": rectangle.WHEELBASE,
            "speed_bounds": [
                -exp3_config.RECTANGLE_SPEED_LIMIT,
                exp3_config.RECTANGLE_SPEED_LIMIT,
            ],
            "steering_bounds_rad": [
                -exp3_config.RECTANGLE_STEER_LIMIT,
                exp3_config.RECTANGLE_STEER_LIMIT,
            ],
            "tolerance": {"nlp_tol_reference": exp3_config.RECTANGLE_ACADOS_TOL},
            "maximum_iterations": {
                "qp": exp3_config.RECTANGLE_ACADOS_QP_MAX_ITER,
                "nlp": exp3_config.RECTANGLE_ACADOS_MAX_ITER,
            },
            "expected_status": 2,
            "notes": (
                "The hard-constraint Gauss--Newton baseline is expected to reach "
                "the SQP iteration limit on the current rectangular problem."
            ),
        },
    )
    write_json(
        output / "export_manifest.json",
        {
            "generator": "experiments/exp3_embedded_export/export_acados_bundles.py",
            "solver_generator": "acados_template",
            "template": "deployment_templates/rectangle_acados_gn",
            "files": bundle_files(output),
        },
    )


def copy_root_files(exports_root: Path) -> None:
    for source in (TEMPLATE_ROOT / "exports_root").iterdir():
        destination = exports_root / source.name
        shutil.copy2(source, destination)
        if destination.suffix == ".sh":
            content = destination.read_text(encoding="utf-8")
            with destination.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            destination.chmod(
                destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
    validator = exports_root / "validate_export_settings.py"
    root_files = sorted(
        source.name for source in (TEMPLATE_ROOT / "exports_root").iterdir()
    )
    readme = exports_root / "README.md"
    if readme.is_file():
        root_files.append(readme.name)
    if validator.is_file():
        root_files.append(validator.name)
    root_files.sort()
    write_json(
        exports_root / "export_manifest.json",
        {
            "generator": "experiments/exp3_embedded_export/export_acados_bundles.py",
            "root_files": root_files,
            "bundles": [
                "circle_ros_mpc",
                "circle_acados_exact",
                "rectangle_ros_mpc",
                "rectangle_acados_gn",
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exports-root",
        type=Path,
        default=DEFAULT_EXPORTS_ROOT,
    )
    parser.add_argument(
        "--bundle",
        choices=("all", "circle", "rectangle", "root"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exports_root = args.exports_root.resolve()
    exports_root.mkdir(parents=True, exist_ok=True)
    if args.bundle in {"all", "circle"}:
        export_circle(exports_root, args.force)
    if args.bundle in {"all", "rectangle"}:
        export_rectangle(exports_root, args.force)
    copy_root_files(exports_root)


if __name__ == "__main__":
    main()
