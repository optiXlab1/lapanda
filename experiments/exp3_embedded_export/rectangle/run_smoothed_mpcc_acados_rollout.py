"""Run the aligned acados rollout for the smoothed rectangle diagnostic.

The initial solve uses a feasible nonzero trajectory.  Subsequent MPC solves
shift the previous solution and are timed until the vehicle reaches the target.
This diagnostic is isolated from the published embedded experiment outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
DEFAULT_OUTDIR = (
    EXP_DIR / "results" / "rectangle" / "smoothed_mpcc_full_rollout"
)
_spec = importlib.util.spec_from_file_location("rectangle_mpcc_diagnostic", THIS_DIR / "diagnose_mpcc.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("could not load diagnose_mpcc.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

TAU = 0.05
STEER_LIMIT = 0.7
SPEED_LOWER = 0.0
SPEED_UPPER = 1.0
STOP_DISTANCE = 0.03


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


def set_state_and_guess(solver, initial_state: np.ndarray, controls: np.ndarray) -> None:
    solver.constraints_set(0, "lbx", initial_state)
    solver.constraints_set(0, "ubx", initial_state)
    state = initial_state.copy()
    solver.set(0, "x", state)
    for k in range(diag.HORIZON):
        control = controls[2 * k : 2 * k + 2]
        solver.set(k, "u", control)
        state = bicycle_step(state, control)
        solver.set(k + 1, "x", state)


def solution_arrays(solver) -> tuple[np.ndarray, np.ndarray]:
    controls = np.vstack(
        [np.asarray(solver.get(k, "u"), dtype=float) for k in range(diag.HORIZON)]
    )
    states = np.vstack(
        [np.asarray(solver.get(k, "x"), dtype=float) for k in range(diag.HORIZON + 1)]
    )
    return states, controls


def shifted_controls(controls: np.ndarray) -> np.ndarray:
    shifted = np.vstack((controls[1:], controls[-1]))
    shifted[-1, 0] = 0.0
    return shifted.reshape(-1)


def solve_backward(forward, sensitivity, controls: np.ndarray) -> tuple[float, str]:
    seeds_u = [(k, controls[k].reshape(2, 1)) for k in range(diag.HORIZON)]
    start = time.perf_counter()
    try:
        if hasattr(forward, "get_flat_iterate"):
            iterate = forward.get_flat_iterate()
        else:
            iterate = forward.store_iterate_to_flat_obj()
        sensitivity.load_iterate_from_flat_obj(iterate)
        sensitivity.setup_qp_matrices_and_factorize()
        sensitivity.eval_adjoint_solution_sensitivity(
            seed_x=None,
            seed_u=seeds_u,
            with_respect_to="p_global",
            sanity_checks=False,
        )
        status = "ok"
    except Exception as exc:  # pragma: no cover - depends on acados runtime
        status = repr(exc)
    return 1e3 * (time.perf_counter() - start), status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    from acados_template import AcadosOcpSolver

    diag.STEER_LIMIT = STEER_LIMIT
    workdir = Path(tempfile.mkdtemp(prefix="acados_full_rollout_", dir=str(args.outdir)))
    try:
        forward_ocp = diag.build_acados_ocp("smooth", TAU, False, workdir, "exact", "kkt")
        sensitivity_ocp = diag.build_acados_ocp("smooth", TAU, True, workdir, "exact", "kkt")
        for ocp in (forward_ocp, sensitivity_ocp):
            ocp.constraints.lbu = np.array([SPEED_LOWER, -STEER_LIMIT])
            ocp.constraints.ubu = np.array([SPEED_UPPER, STEER_LIMIT])

        forward = AcadosOcpSolver(
            forward_ocp,
            json_file=str(workdir / "forward.json"),
            verbose=False,
        )
        sensitivity = AcadosOcpSolver(
            sensitivity_ocp,
            json_file=str(workdir / "sensitivity.json"),
            verbose=False,
        )
        parameters = np.r_[diag.THETA_PREFIX, diag.TEACHER_MARGINS]
        forward.set_p_global_and_precompute_dependencies(parameters)
        sensitivity.set_p_global_and_precompute_dependencies(parameters)

        state = diag.START.copy()
        guess = diag.initial_controls(y_amplitude=0.8)
        set_state_and_guess(forward, state, guess)
        initial_start = time.perf_counter()
        initial_status = int(forward.solve())
        initial_forward_ms = 1e3 * (time.perf_counter() - initial_start)
        _, plan_controls = solution_arrays(forward)
        if initial_status != 0:
            raise RuntimeError(f"initial acados solve failed with status {initial_status}")

        executed_states = [state.tolist()]
        executed_controls = []
        rows = []

        # Execute the control from the initialization solve, but do not time it.
        control = plan_controls[0].copy()
        state = bicycle_step(state, control)
        executed_controls.append(control.tolist())
        executed_states.append(state.tolist())
        guess = shifted_controls(plan_controls)

        for step in range(1, args.max_steps + 1):
            set_state_and_guess(forward, state, guess)
            forward.set_p_global_and_precompute_dependencies(parameters)
            start = time.perf_counter()
            status = int(forward.solve())
            forward_ms = 1e3 * (time.perf_counter() - start)
            _, plan_controls = solution_arrays(forward)
            backward_ms, backward_status = solve_backward(forward, sensitivity, plan_controls)

            control = plan_controls[0].copy()
            state = bicycle_step(state, control)
            executed_controls.append(control.tolist())
            executed_states.append(state.tolist())
            distance = float(np.linalg.norm(state[:2] - diag.TARGET[:2]))
            rows.append(
                {
                    "step": step,
                    "forward_ms": forward_ms,
                    "backward_ms": backward_ms,
                    "status": status,
                    "backward_status": backward_status,
                    "sqp_iterations": int(np.asarray(forward.get_stats("sqp_iter")).reshape(-1)[0]),
                    "distance_to_target": distance,
                }
            )
            if status != 0:
                break
            guess = shifted_controls(plan_controls)

        forward_times = np.asarray([row["forward_ms"] for row in rows], dtype=float)
        backward_times = np.asarray([row["backward_ms"] for row in rows], dtype=float)
        record = {
            "solver": "acados",
            "problem": "smooth_tau0.05_teacher_margins_forward_speed",
            "initialization": "configured_nonzero_arc",
            "initial_solve_counted": False,
            "warm_start": "shifted_previous_solution",
            "hessian": "exact",
            "horizon": diag.HORIZON,
            "tau": TAU,
            "speed_bounds": [SPEED_LOWER, SPEED_UPPER],
            "steer_limit": STEER_LIMIT,
            "initial_forward_ms": initial_forward_ms,
            "initial_status": initial_status,
            "reached_target": bool(any(row["distance_to_target"] <= STOP_DISTANCE for row in rows)),
            "final_state": state.tolist(),
            "final_distance": float(np.linalg.norm(state[:2] - diag.TARGET[:2])),
            "states": executed_states,
            "controls": executed_controls,
            "rows": rows,
            "mean_forward_ms": float(np.mean(forward_times)),
            "median_forward_ms": float(np.median(forward_times)),
            "mean_backward_ms": float(np.mean(backward_times)),
            "median_backward_ms": float(np.median(backward_times)),
        }
        output = args.outdir / "acados_smooth_rollout_to_target.json"
        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in record.items() if key not in {"states", "controls", "rows"}}, indent=2))
        print(f"wrote {output}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
