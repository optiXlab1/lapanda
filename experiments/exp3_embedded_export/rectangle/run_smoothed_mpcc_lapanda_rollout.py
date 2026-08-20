"""Run a 30-step lapanda rollout for the smoothed rectangle diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
OUTDIR = EXP_DIR / "results" / "rectangle" / "smoothed_mpcc_full_rollout"

_spec = importlib.util.spec_from_file_location("rectangle_mpcc_diagnostic", THIS_DIR / "diagnose_mpcc.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load diagnose_mpcc.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)
diag.STEER_LIMIT = 0.7

TAU = 0.05
MAX_STEPS = 30
STOP_DISTANCE = 0.03


def shift_controls(controls: np.ndarray) -> np.ndarray:
    shifted = np.vstack((controls[1:], controls[-1]))
    return shifted.reshape(-1)


def bicycle_step(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    speed, steer = control
    return np.array(
        [
            state[0] + diag.DT * speed * np.cos(state[2]),
            state[1] + diag.DT * speed * np.sin(state[2]),
            state[2] + diag.DT * speed * np.tan(steer) / diag.WHEELBASE,
        ],
        dtype=float,
    )


def main() -> None:
    python_dir = REPO_ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from lapanda import AlmOptions, BackwardOptions, CasadiProblem, SolverOptions, build_solver
    from lapanda._lapanda import solve_lapanda_compiled

    teacher = diag.load_teacher()
    problem = diag.reduced_problem("smooth", TAU, diag.TEACHER_MARGINS, teacher)
    casadi_problem = CasadiProblem(
        u=problem["controls"],
        theta=problem["theta"],
        variable=problem["variable"],
        cost=problem["cost"],
        box_lower=np.tile([0.0, -diag.STEER_LIMIT], diag.HORIZON),
        box_upper=np.tile([diag.SPEED_LIMIT, diag.STEER_LIMIT], diag.HORIZON),
        constraints=problem["constraints"],
        outer_loss=problem["outer_loss"],
    )
    solver = build_solver(
        casadi_problem,
        backend="compiled",
        name="rectdiag_lse005_teacher_d07_fwd",
        cache_dir=REPO_ROOT / ".lapanda_cache" / "mpcc" / "smooth" / "teacher_d07_forward",
        build=True,
    )
    metadata = solver.generated_oracle.metadata

    solver_options = SolverOptions()
    solver_options.max_iterations = 20000
    solver_options.tolerance = 1e-3
    solver_options.buffer_size = 10
    solver_options.max_stable_iter = 80
    alm_options = AlmOptions()
    alm_options.max_iterations = 100
    alm_options.tolerance = 1e-4
    alm_options.initial_penalty = 100.0
    alm_options.penalty_update_factor = 10.0
    alm_options.sufficient_decrease_factor = 0.25
    alm_options.warm_start_inner = True
    backward_options = BackwardOptions()
    backward_options.enable = True
    backward_options.tolerance = 1e-3
    backward_options.max_iterations = 200

    def solve(initial_guess: np.ndarray, state: np.ndarray, multipliers, enable_backward: bool):
        backward_options.enable = enable_backward
        variable = problem["variable_value"].copy()
        variable[:3] = state
        return solve_lapanda_compiled(
            str(solver.generated_oracle.library_path),
            metadata["n"],
            metadata["ntheta"],
            metadata["nvar"],
            metadata["ncon"],
            np.tile([0.0, -diag.STEER_LIMIT], diag.HORIZON),
            metadata["box_upper"],
            initial_guess,
            problem["theta_value"],
            variable,
            problem["lower"],
            problem["upper"],
            solver_options,
            alm_options,
            backward_options,
            multipliers,
            np.full(metadata["ncon"], 100.0),
        )

    state = diag.START.copy()
    initial = solve(diag.initial_controls(y_amplitude=0.8), state, None, False)
    print(f"initial forward={1e3 * float(initial['forward_time_sec']):.3f} ms", flush=True)
    solution = np.asarray(initial["solution"], dtype=float)
    multipliers = np.asarray(initial["multipliers"], dtype=float)
    control = solution[:2].copy()
    state = bicycle_step(state, control)
    states = [diag.START.tolist(), state.tolist()]
    controls = [control.tolist()]
    rows = []

    for step in range(1, MAX_STEPS + 1):
        shifted_multipliers = np.r_[multipliers[1:], 0.0]
        result = solve(
            shift_controls(solution.reshape(diag.HORIZON, 2)),
            state,
            shifted_multipliers,
            True,
        )
        solution = np.asarray(result["solution"], dtype=float)
        multipliers = np.asarray(result["multipliers"], dtype=float)
        plan = solution.reshape(diag.HORIZON, 2)
        control = plan[0].copy()
        state = bicycle_step(state, control)
        states.append(state.tolist())
        controls.append(control.tolist())
        rows.append(
            {
                "step": step,
                "forward_ms": 1e3 * float(result["forward_time_sec"]),
                "backward_ms": 1e3 * float(result["backward_time_sec"]),
                "outer": int(result.get("iterations", -1)),
                "inner": int(np.sum(result.get("inner_iterations", []))),
                "backward_iterations": int(result.get("backward_iterations", 0)),
                "residual": float(result.get("backward_residual", np.nan)),
                "distance_to_target": float(np.linalg.norm(state[:2] - diag.TARGET[:2])),
            }
        )
        print(
            f"step={step:02d} forward={rows[-1]['forward_ms']:.3f} ms "
            f"backward={rows[-1]['backward_ms']:.3f} ms",
            flush=True,
        )

    record = {
        "solver": "lapanda",
        "problem": "smooth_tau0.05_teacher_margins",
        "max_steps": MAX_STEPS,
        "initial_solve_counted": False,
        "warm_start": "shifted_previous_solution",
        "reached_target": bool(
            any(row.get("distance_to_target", np.inf) <= STOP_DISTANCE for row in rows)
        ),
        "states": states,
        "controls": controls,
        "rows": rows,
        "mean_forward_ms": float(np.mean([row["forward_ms"] for row in rows])),
        "mean_backward_ms": float(np.mean([row["backward_ms"] for row in rows])),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output = OUTDIR / "lapanda_smooth_rollout_30.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("reached_target", "mean_forward_ms", "mean_backward_ms")}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
