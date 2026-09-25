"""Assemble the exact records used by the paper's matched-accuracy table."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "matched_accuracy"
SIZES = (100, 200, 500, 1000)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def one_by_n(rows: list[dict[str, str]], n: int) -> dict[str, str]:
    return next(row for row in rows if int(row["n"]) == n)


def one_by_method_n(
    rows: list[dict[str, str]], method: str, n: int
) -> dict[str, str]:
    return next(
        row for row in rows if row["method"] == method and int(row["n"]) == n
    )


def main() -> None:
    aligned = read_rows(ROOT / "lapanda" / "summary.csv")
    explicit_timing = explicit_accuracy = read_rows(
        ROOT / "explicit" / "summary.csv"
    )
    casadi_timing = casadi_accuracy = read_rows(ROOT / "casadi" / "summary.csv")
    memory = read_rows(ROOT / "memory" / "solve_memory_summary.csv")
    output_rows = []

    for n in SIZES:
        lap = one_by_method_n(aligned, "lapanda aligned", n)
        exp_timing = one_by_method_n(explicit_timing, "Explicit KKT", n)
        exp_accuracy = one_by_method_n(explicit_accuracy, "Explicit KKT", n)
        cas_timing = one_by_n(casadi_timing, n)
        cas_accuracy = one_by_n(casadi_accuracy, n)
        method_rows = (
            (
                "lapanda",
                float(lap["gradient_relative_error_mean"]),
                float(lap["gradient_relative_error_max"]),
                float(lap["base_forward_time_sec_mean"]),
                float(lap["combined_backward_time_sec_mean"]),
                "lapanda",
            ),
            (
                "Explicit KKT",
                float(exp_accuracy["gradient_relative_error_mean"]),
                float(exp_accuracy["gradient_relative_error_max"]),
                np.nan,
                float(exp_timing["backward_time_sec_mean"]),
                "Explicit KKT (dense)",
            ),
            (
                "CasADi",
                float(cas_accuracy["gradient_relative_error_mean"]),
                float(cas_accuracy["gradient_relative_error_max"]),
                float(cas_timing["forward_time_sec_mean"]),
                float(cas_timing["backward_time_sec_mean"]),
                "CasADi",
            ),
        )
        for method, error_mean, error_max, forward, backward, memory_method in method_rows:
            memory_row = next(
                row
                for row in memory
                if row["method"] == memory_method and int(row["n"]) == n
            )
            output_rows.append(
                {
                    "n": n,
                    "method": method,
                    "gradient_relative_error_mean": error_mean,
                    "gradient_relative_error_max": error_max,
                    "forward_time_ms": 1e3 * forward if np.isfinite(forward) else "",
                    "backward_time_ms": 1e3 * backward,
                    "total_memory_mb": float(memory_row["total_memory_mb_mean"]),
                }
            )

    path = ROOT / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
