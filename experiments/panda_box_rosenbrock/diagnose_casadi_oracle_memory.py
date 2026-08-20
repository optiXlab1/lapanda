"""Diagnose CasADi oracle construction memory for Exp. 1.

This script does not call PANDA. It builds the same CasADi oracle pieces used
by the compiled backend and records RSS after each construction step.

Run:

    python experiments/panda_box_rosenbrock/diagnose_casadi_oracle_memory.py --n 2000
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
from pathlib import Path

import casadi as ca

from experiments.common import append_csv, fresh_output_path, rss_mb, sampled_call
from experiments.panda_box_rosenbrock.problem import build_problem


def record(rows, step, baseline_mb, note=""):
    current = rss_mb()
    rows.append(
        {
            "step": step,
            "rss_mb": current,
            "delta_from_start_mb": current - baseline_mb,
            "note": note,
        }
    )
    print(f"{step:32s} rss={current:.2f} MB delta={current - baseline_mb:.2f} MB {note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--generate-code", action="store_true", default="True")
    parser.add_argument("--free-after-codegen", action="store_true", default="True")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    result_path = fresh_output_path(out_dir / f"exp1_casadi_oracle_memory_n{args.n}.csv")
    rows = []
    baseline = rss_mb()
    record(rows, "start", baseline)

    instance_stats = sampled_call(lambda: build_problem(args.n))
    instance = instance_stats["value"]
    record(
        rows,
        "build_problem",
        baseline,
        f"sample_peak_delta={instance_stats['rss_peak_delta_mb']:.2f}",
    )

    u = ca.vec(instance.problem.u)
    theta = ca.vec(instance.problem.theta)
    variable = ca.vec(instance.problem.variable)
    cost = instance.problem.cost
    v = ca.SX.sym("v", args.n)

    grad_stats = sampled_call(lambda: ca.gradient(cost, u))
    grad = grad_stats["value"]
    record(rows, "ca.gradient(cost,u)", baseline, f"sample_peak_delta={grad_stats['rss_peak_delta_mb']:.2f}")

    cost_grad_stats = sampled_call(
        lambda: ca.Function("panda_cost_grad", [u, theta, variable], [cost, ca.densify(grad)])
    )
    cost_grad_fun = cost_grad_stats["value"]
    record(
        rows,
        "Function cost_grad",
        baseline,
        f"sample_peak_delta={cost_grad_stats['rss_peak_delta_mb']:.2f}",
    )

    hvp_expr_stats = sampled_call(lambda: ca.jtimes(grad, u, v))
    hvp_expr = hvp_expr_stats["value"]
    record(rows, "ca.jtimes grad", baseline, f"sample_peak_delta={hvp_expr_stats['rss_peak_delta_mb']:.2f}")

    hvp_fun_stats = sampled_call(
        lambda: ca.Function("panda_hvp", [v, u, theta, variable], [ca.densify(hvp_expr)])
    )
    hvp_fun = hvp_fun_stats["value"]
    record(rows, "Function hvp", baseline, f"sample_peak_delta={hvp_fun_stats['rss_peak_delta_mb']:.2f}")

    vjp_expr_stats = sampled_call(lambda: ca.jacobian(grad, theta).T @ v)
    vjp_expr = vjp_expr_stats["value"]
    record(
        rows,
        "jacobian(grad,theta).T@v",
        baseline,
        f"sample_peak_delta={vjp_expr_stats['rss_peak_delta_mb']:.2f}",
    )

    vjp_fun_stats = sampled_call(
        lambda: ca.Function("panda_vjp", [v, u, theta, variable], [ca.densify(vjp_expr)])
    )
    vjp_fun = vjp_fun_stats["value"]
    record(rows, "Function vjp", baseline, f"sample_peak_delta={vjp_fun_stats['rss_peak_delta_mb']:.2f}")

    loss_grad_expr_stats = sampled_call(lambda: ca.gradient(instance.problem.outer_loss, u))
    loss_grad_expr = loss_grad_expr_stats["value"]
    record(
        rows,
        "gradient outer_loss",
        baseline,
        f"sample_peak_delta={loss_grad_expr_stats['rss_peak_delta_mb']:.2f}",
    )

    loss_grad_fun_stats = sampled_call(
        lambda: ca.Function(
            "panda_loss_grad",
            [u, theta, variable],
            [instance.problem.outer_loss, ca.densify(loss_grad_expr)],
        )
    )
    loss_grad_fun = loss_grad_fun_stats["value"]
    record(
        rows,
        "Function loss_grad",
        baseline,
        f"sample_peak_delta={loss_grad_fun_stats['rss_peak_delta_mb']:.2f}",
    )

    functions = [cost_grad_fun, hvp_fun, vjp_fun, loss_grad_fun]

    if args.generate_code:
        work_dir = Path(tempfile.mkdtemp(prefix="lapanda_oracle_mem_"))

        def generate():
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(work_dir)
                generator = ca.CodeGenerator("diagnostic_oracle.c", {"with_header": True})
                for fun in functions:
                    generator.add(fun)
                generator.generate()
            finally:
                os.chdir(old_cwd)
            return work_dir / "diagnostic_oracle.c"

        codegen_stats = sampled_call(generate)
        source_path = codegen_stats["value"]
        source_size_mb = source_path.stat().st_size / (1024.0 * 1024.0)
        record(
            rows,
            "CodeGenerator.generate",
            baseline,
            f"sample_peak_delta={codegen_stats['rss_peak_delta_mb']:.2f}; c_size_mb={source_size_mb:.2f}",
        )

    if args.free_after_codegen:
        del functions
        del cost_grad_fun, hvp_fun, vjp_fun, loss_grad_fun
        del grad, hvp_expr, vjp_expr, loss_grad_expr
        del instance
        gc.collect()
        time.sleep(2)
        record(rows, "after del + gc.collect", baseline)

    append_csv(result_path, rows)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
