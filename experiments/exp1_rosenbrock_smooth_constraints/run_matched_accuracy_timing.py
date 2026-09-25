"""Run the repeated lapanda and Explicit-KKT timings used by Table 2.

The aligned lapanda cost is reported as the warm-started final-subproblem
refinement plus the subsequent matrix-free Krylov solve.  Explicit KKT uses
the same unrefined lapanda forward result and times derivative evaluation,
dense matrix assembly, and the C MINRES solve.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
for search_path in (REPO_ROOT, PYTHON_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.common import relative_error, write_json
from experiments.exp1_rosenbrock_smooth_constraints.problem import (
    build_problem,
    constraint_violation_inf,
    initial_point,
    sinusoidal_target,
    theta_nominal,
)
from experiments.exp1_rosenbrock_smooth_constraints.kkt_utils import (
    build_kkt_derivative_function,
    cosine_similarity,
    explicit_kkt_minres_backward,
    same_forward_kkt_reference,
    write_csv,
)
from experiments.exp1_rosenbrock_smooth_constraints.post_forward_refinement.run_comparison import (
    solve_aligned_refinement,
    solve_forward,
)
from lapanda import build_solver


def load_reference_gradients(path: Path) -> dict[tuple[int, int], np.ndarray]:
    references: dict[tuple[int, int], np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            references[(int(row["n"]), int(row["trial_id"]))] = np.asarray(
                ast.literal_eval(row["gradient"]), dtype=float
            )
    return references


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (row["method"], row["n"], row["penalty_scale"], row["refinement_tol"])
            for row in rows
        }
    )
    result = []
    for method, n, scale, refinement_tol in keys:
        group = [
            row
            for row in rows
            if row["method"] == method
            and row["n"] == n
            and row["penalty_scale"] == scale
            and row["refinement_tol"] == refinement_tol
        ]
        summary = {
            "method": method,
            "n": n,
            "trials": len(group),
            "penalty_scale": scale,
            "refinement_tol": refinement_tol,
        }
        for field in (
            "gradient_relative_error",
            "gradient_cosine_similarity",
            "final_penalty_max",
            "constraint_violation_inf",
            "base_forward_time_sec",
            "refinement_time_sec",
            "backward_time_sec",
            "combined_backward_time_sec",
            "combined_call_wall_time_sec",
        ):
            values = np.asarray([row.get(field, np.nan) for row in group], dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(np.std(values, ddof=1))
            summary[f"{field}_max"] = float(np.max(values))
        summary["backward_fallback_count"] = sum(
            bool(row.get("backward_fallback_used", False)) for row in group
        )
        result.append(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500, 1000])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--penalty-scales", nargs="+", type=float, default=[10.0])
    parser.add_argument(
        "--refinement-tolerances", nargs="+", type=float, default=[1e-2]
    )
    parser.add_argument("--inner-tol", dest="tol", type=float, default=1e-3)
    parser.add_argument("--alm-tol", type=float, default=1e-3)
    parser.add_argument("--backward-tol", type=float, default=1e-2)
    parser.add_argument("--inner-max-iter", type=int, default=4000)
    parser.add_argument("--max-outer", type=int, default=20)
    parser.add_argument("--initial-penalty", type=float, default=2.0)
    parser.add_argument("--penalty-update-factor", type=float, default=10.0)
    parser.add_argument("--max-penalty", type=float, default=1e8)
    parser.add_argument("--backward-max-iter", type=int, default=200)
    parser.add_argument("--linear-solver", choices=["cg", "minres"], default="cg")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["lapanda", "lapanda_base", "explicit_kkt", "explicit_refined"],
        default=["lapanda", "explicit_kkt"],
        help="Select aligned lapanda, balanced lapanda, and/or Explicit KKT.",
    )
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--constraint-stride", type=int, default=1)
    parser.add_argument("--backend", choices=["compiled", "callback"], default="compiled")
    parser.add_argument("--reference-active-tol", type=float, default=1e-7)
    parser.add_argument("--reference-multiplier-tol", type=float, default=1e-7)
    parser.add_argument("--reference-at-base-point", action="store_true")
    parser.add_argument("--reference-at-refined-point", action="store_true")
    parser.add_argument("--save-base-reference-csv", type=Path)
    parser.add_argument("--save-refined-reference-csv", type=Path)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    references = (
        load_reference_gradients(args.reference_csv) if args.reference_csv else {}
    )
    theta = theta_nominal()
    rows: list[dict] = []
    base_reference_rows: list[dict] = []
    refined_reference_rows: list[dict] = []

    for n in args.sizes:
        instance = build_problem(n, constraint_stride=args.constraint_stride)
        target = sinusoidal_target(n)
        derivative_fun = None
        if (
            "explicit_kkt" in args.methods
            or "explicit_refined" in args.methods
            or args.reference_at_base_point
            or args.reference_at_refined_point
        ):
            derivative_fun = build_kkt_derivative_function(instance, f"recheck_kkt_n{n}")
        solver = build_solver(
            instance.problem,
            backend=args.backend,
            name=f"aligned_explicit_recheck_n{n}",
            cache_dir=Path(tempfile.gettempdir()) / "lapanda_experiments" / "aligned_recheck",
        )

        # Exclude one-time dispatch and workspace initialization from solve timings.
        warm_x0 = initial_point(n, 9999)
        warm_base = solve_forward(
            solver, instance, warm_x0, theta, target, args, backward_scale=None
        )
        if derivative_fun is not None:
            explicit_kkt_minres_backward(
                derivative_fun,
                instance,
                np.asarray(warm_base["solution"], dtype=float),
                theta,
                target,
                np.maximum(np.asarray(warm_base["multipliers"], dtype=float), 0.0),
                args.reference_active_tol,
                args.reference_multiplier_tol,
                args.backward_tol,
                args.backward_max_iter,
            )
        if "lapanda" in args.methods:
            for scale in args.penalty_scales:
                for refinement_tol in args.refinement_tolerances:
                    args.penalty_scale = scale
                    args.refinement_tol = refinement_tol
                    solve_aligned_refinement(
                        solver, instance, warm_base, theta, target, args, enable_backward=False
                    )
                    solve_aligned_refinement(
                        solver, instance, warm_base, theta, target, args, enable_backward=True
                    )
        if "explicit_refined" in args.methods:
            for scale in args.penalty_scales:
                for refinement_tol in args.refinement_tolerances:
                    args.penalty_scale = scale
                    args.refinement_tol = refinement_tol
                    warm_refined = solve_aligned_refinement(
                        solver,
                        instance,
                        warm_base,
                        theta,
                        target,
                        args,
                        enable_backward=False,
                    )
                    explicit_kkt_minres_backward(
                        derivative_fun,
                        instance,
                        np.asarray(warm_refined["solution"], dtype=float),
                        theta,
                        target,
                        np.maximum(
                            np.asarray(warm_refined["multipliers"], dtype=float), 0.0
                        ),
                        args.reference_active_tol,
                        args.reference_multiplier_tol,
                        args.backward_tol,
                        args.backward_max_iter,
                    )

        for repetition in range(args.repetitions):
            for trial_id in range(args.trial_start, args.trial_start + args.trials):
                reference = references.get((n, trial_id))
                x0 = initial_point(n, 100 + trial_id)
                base = solve_forward(
                    solver,
                    instance,
                    x0,
                    theta,
                    target,
                    args,
                    backward_scale=1.0 if "lapanda_base" in args.methods else None,
                )
                if args.reference_at_base_point:
                    base_reference = same_forward_kkt_reference(
                        instance,
                        derivative_fun,
                        np.asarray(base["solution"], dtype=float),
                        theta,
                        target,
                        np.asarray(base["multipliers"], dtype=float),
                        np.asarray(base["penalties"], dtype=float),
                        args.reference_active_tol,
                    )
                    reference = np.asarray(base_reference["gradient"], dtype=float)
                    if repetition == 0:
                        base_reference_rows.append(
                            {
                                "n": n,
                                "trial_id": trial_id,
                                "linear_residual": base_reference["linear_residual"],
                                "gradient": json.dumps(reference.tolist()),
                            }
                        )
                if reference is None and (
                    "lapanda_base" in args.methods or "explicit_kkt" in args.methods
                ):
                    raise ValueError(
                        "Provide --reference-csv or enable --reference-at-base-point"
                    )
                if "lapanda_base" in args.methods:
                    base_gradient = np.asarray(base["grad_theta"], dtype=float)
                    rows.append(
                        {
                            "method": "lapanda balanced",
                            "n": n,
                            "repetition": repetition,
                            "trial_id": trial_id,
                            "penalty_scale": 1.0,
                            "refinement_tol": 0.0,
                            "gradient_relative_error": relative_error(
                                base_gradient, reference
                            ),
                            "gradient_cosine_similarity": cosine_similarity(
                                base_gradient, reference
                            ),
                            "final_penalty_max": float(
                                np.max(np.asarray(base["penalties"], dtype=float))
                            ),
                            "constraint_violation_inf": constraint_violation_inf(
                                instance,
                                np.asarray(base["solution"], dtype=float),
                                theta,
                            ),
                            "base_forward_time_sec": float(base["forward_time_sec"]),
                            "refinement_time_sec": 0.0,
                            "backward_time_sec": float(base["backward_time_sec"]),
                            "combined_backward_time_sec": float(base["backward_time_sec"]),
                            "combined_call_wall_time_sec": float(base["backward_time_sec"]),
                            "backward_solver_used": base.get(
                                "backward_solver_used", "unknown"
                            ),
                            "backward_fallback_used": bool(
                                base.get("backward_fallback_used", False)
                            ),
                        }
                    )
                if "explicit_kkt" in args.methods:
                    explicit = explicit_kkt_minres_backward(
                        derivative_fun,
                        instance,
                        np.asarray(base["solution"], dtype=float),
                        theta,
                        target,
                        np.maximum(np.asarray(base["multipliers"], dtype=float), 0.0),
                        args.reference_active_tol,
                        args.reference_multiplier_tol,
                        args.backward_tol,
                        args.backward_max_iter,
                    )
                    rows.append(
                        {
                            "method": "Explicit KKT",
                            "n": n,
                            "repetition": repetition,
                            "trial_id": trial_id,
                            "penalty_scale": 0.0,
                            "refinement_tol": 0.0,
                            "gradient_relative_error": relative_error(
                                explicit["gradient"], reference
                            ),
                            "base_forward_time_sec": float(base["forward_time_sec"]),
                            "refinement_time_sec": 0.0,
                            "backward_time_sec": float(explicit["total_time_sec"]),
                            "combined_backward_time_sec": float(explicit["total_time_sec"]),
                            "combined_call_wall_time_sec": float(explicit["total_time_sec"]),
                        }
                    )

                needs_refinement = (
                    "lapanda" in args.methods or "explicit_refined" in args.methods
                )
                for scale in args.penalty_scales if needs_refinement else []:
                    for refinement_tol in args.refinement_tolerances:
                        args.penalty_scale = scale
                        args.refinement_tol = refinement_tol
                        start = time.perf_counter()
                        refined_forward = solve_aligned_refinement(
                            solver, instance, base, theta, target, args, enable_backward=False
                        )
                        refinement_time = time.perf_counter() - start
                        aligned_reference = reference
                        reference_residual = float("nan")
                        if args.reference_at_refined_point:
                            refined_reference = same_forward_kkt_reference(
                                instance,
                                derivative_fun,
                                np.asarray(refined_forward["solution"], dtype=float),
                                theta,
                                target,
                                np.asarray(refined_forward["multipliers"], dtype=float),
                                np.asarray(refined_forward["penalties"], dtype=float),
                                args.reference_active_tol,
                            )
                            aligned_reference = np.asarray(
                                refined_reference["gradient"], dtype=float
                            )
                            reference_residual = float(
                                refined_reference["linear_residual"]
                            )
                            if repetition == 0:
                                refined_reference_rows.append(
                                    {
                                        "n": n,
                                        "trial_id": trial_id,
                                        "penalty_scale": scale,
                                        "refinement_tol": refinement_tol,
                                        "linear_residual": reference_residual,
                                        "gradient": json.dumps(
                                            aligned_reference.tolist()
                                        ),
                                    }
                                )
                        if "lapanda" in args.methods:
                            combined_start = time.perf_counter()
                            refined = solve_aligned_refinement(
                                solver,
                                instance,
                                base,
                                theta,
                                target,
                                args,
                                enable_backward=True,
                            )
                            combined_call_wall_time = time.perf_counter() - combined_start
                            backward_time = float(refined["backward_time_sec"])
                            rows.append(
                                {
                                    "method": "lapanda aligned",
                                    "n": n,
                                    "repetition": repetition,
                                    "trial_id": trial_id,
                                    "penalty_scale": scale,
                                    "refinement_tol": refinement_tol,
                                    "gradient_relative_error": relative_error(
                                        np.asarray(refined["grad_theta"], dtype=float),
                                        aligned_reference,
                                    ),
                                    "reference_residual": reference_residual,
                                    "base_forward_time_sec": float(base["forward_time_sec"]),
                                    "refinement_time_sec": refinement_time,
                                    "backward_time_sec": backward_time,
                                    "combined_backward_time_sec": refinement_time + backward_time,
                                    "combined_call_wall_time_sec": combined_call_wall_time,
                                    "backward_solver_used": refined.get(
                                        "backward_solver_used", "unknown"
                                    ),
                                    "backward_fallback_used": bool(
                                        refined.get("backward_fallback_used", False)
                                    ),
                                }
                            )
                        if "explicit_refined" in args.methods:
                            explicit = explicit_kkt_minres_backward(
                                derivative_fun,
                                instance,
                                np.asarray(refined_forward["solution"], dtype=float),
                                theta,
                                target,
                                np.maximum(
                                    np.asarray(
                                        refined_forward["multipliers"], dtype=float
                                    ),
                                    0.0,
                                ),
                                args.reference_active_tol,
                                args.reference_multiplier_tol,
                                args.backward_tol,
                                args.backward_max_iter,
                            )
                            explicit_time = float(explicit["total_time_sec"])
                            rows.append(
                                {
                                    "method": "Explicit KKT aligned",
                                    "n": n,
                                    "repetition": repetition,
                                    "trial_id": trial_id,
                                    "penalty_scale": scale,
                                    "refinement_tol": refinement_tol,
                                    "gradient_relative_error": relative_error(
                                        np.asarray(explicit["gradient"], dtype=float),
                                        aligned_reference,
                                    ),
                                    "reference_residual": reference_residual,
                                    "base_forward_time_sec": float(base["forward_time_sec"]),
                                    "refinement_time_sec": refinement_time,
                                    "backward_time_sec": explicit_time,
                                    "combined_backward_time_sec": refinement_time
                                    + explicit_time,
                                    "combined_call_wall_time_sec": refinement_time
                                    + explicit_time,
                                    "backward_solver_used": "minres",
                                    "backward_fallback_used": False,
                                }
                            )

        print(f"completed n={n}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw.csv", rows)
    summaries = summarize(rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    if args.save_base_reference_csv:
        write_csv(args.save_base_reference_csv, base_reference_rows)
    if args.save_refined_reference_csv:
        write_csv(args.save_refined_reference_csv, refined_reference_rows)
    write_json(
        args.output_dir / "config.json",
        {
            "sizes": args.sizes,
            "trials": args.trials,
            "repetitions": args.repetitions,
            "trial_start": args.trial_start,
            "penalty_scales": args.penalty_scales,
            "refinement_tolerances": args.refinement_tolerances,
            "inner_tolerance": args.tol,
            "alm_tolerance": args.alm_tol,
            "backward_tolerance": args.backward_tol,
            "linear_solver": args.linear_solver,
            "methods": args.methods,
            "timing": {
                "lapanda": "warm-started refinement wall time plus matrix-free backward time",
                "explicit_kkt": "derivative evaluation, dense KKT assembly, and MINRES solve time",
            },
            "reference_csv": str(args.reference_csv) if args.reference_csv else None,
            "reference_at_base_point": args.reference_at_base_point,
            "reference_at_refined_point": args.reference_at_refined_point,
            "reference_timing": "excluded from all reported timings",
        },
    )
    print(f"wrote {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
