"""Validate and plot a matched Exp.2 run without changing retained raw results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ("cartpole", "quadrotor", "robot_arm")


def check_csv(path: Path, final_epoch: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in rows]
    expected = list(range(final_epoch + 1))
    if epochs != expected:
        raise ValueError(f"{path}: expected epochs 0..{final_epoch}, got {len(epochs)} rows")


def validate(tag: str) -> None:
    for model in MODELS:
        folder = HERE / model
        check_csv(folder / "results" / f"exp2_{model}_open_loop_train_lapanda_{tag}.csv", 800)
        check_csv(folder / "results" / f"exp2_{model}_closed_loop_train_lapanda_{tag}.csv", 500)
        check_csv(folder / "baselines" / "results" / f"exp2_{model}_imitation_safepdp_{tag}_coc.csv", 800)
        check_csv(folder / "baselines" / "results" / f"exp2_{model}_closed_loop_train_safepdp_{tag}_coc.csv", 500)
        if model == "cartpole":
            continue
        for device in ("cpu", "gpu"):
            for mode, final_epoch in (("open", 800), ("closed", 500)):
                stem = f"{model}_{device}_{mode}_{tag}_e{final_epoch}_sqp50_admm1000"
                result_dir = folder / "turbompc" / "results"
                check_csv(result_dir / f"{stem}_metrics.csv", final_epoch)
                summary_path = result_dir / f"{stem}_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("completed_epochs") != final_epoch or summary.get("stopped_reason"):
                    raise ValueError(f"{summary_path}: run did not finish normally")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Shared run tag, e.g. same_time_20260922")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    validate(args.tag)
    print(f"Validated all matched Exp.2 results for {args.tag}", flush=True)
    if args.validate_only:
        return
    for script in ("plot_open_loop.py", "plot_closed_loop.py"):
        subprocess.run(
            [sys.executable, str(HERE / script), "--run-tag", args.tag],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
