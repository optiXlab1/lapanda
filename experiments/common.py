"""Shared helpers for lapanda numerical experiments."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_python_path() -> None:
    """Make local package imports work when experiment scripts are run directly."""

    import sys

    python_dir = repo_root() / "python"
    python_str = str(python_dir)
    if python_str not in sys.path:
        sys.path.insert(0, python_str)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fresh_output_path(path: Path) -> Path:
    """Return a writable output path, avoiding Windows file locks on old CSVs."""

    if not path.exists():
        return path
    try:
        path.unlink()
        return path
    except PermissionError:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def append_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def rss_tree_mb() -> float:
    try:
        import psutil

        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return total / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def timed_call(fn):
    before_mem = rss_mb()
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    after_mem = rss_mb()
    return value, elapsed, max(before_mem, after_mem)


def sampled_call(fn, sample_interval_sec: float = 0.002, sample_tree: bool = False):
    """Run fn while sampling process RSS in a background thread."""

    samples = []
    tree_samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            samples.append(rss_mb())
            if sample_tree:
                tree_samples.append(rss_tree_mb())
            stop_event.wait(sample_interval_sec)

    before_mem = rss_mb()
    before_tree_mem = rss_tree_mb() if sample_tree else float("nan")
    samples.append(before_mem)
    if sample_tree:
        tree_samples.append(before_tree_mem)
    thread = threading.Thread(target=sampler, daemon=True)
    start = time.perf_counter()
    thread.start()
    try:
        value = fn()
    finally:
        after_mem = rss_mb()
        after_tree_mem = rss_tree_mb() if sample_tree else float("nan")
        samples.append(after_mem)
        if sample_tree:
            tree_samples.append(after_tree_mem)
        stop_event.set()
        thread.join()
    elapsed = time.perf_counter() - start
    peak_mem = float(np.nanmax(samples)) if samples else max(before_mem, after_mem)
    peak_tree_mem = float(np.nanmax(tree_samples)) if tree_samples else float("nan")
    return {
        "value": value,
        "elapsed_sec": elapsed,
        "rss_before_mb": before_mem,
        "rss_after_mb": after_mem,
        "rss_peak_mb": peak_mem,
        "rss_peak_delta_mb": peak_mem - before_mem,
        "rss_tree_before_mb": before_tree_mem,
        "rss_tree_after_mb": after_tree_mem,
        "rss_tree_peak_mb": peak_tree_mem,
        "rss_tree_peak_delta_mb": peak_tree_mem - before_tree_mem,
        "rss_samples": len(samples),
    }


def profiled_call(fn, measure_memory: bool = False, sample_tree: bool = False):
    """Run fn with timing, and optionally sample RSS memory."""

    if measure_memory:
        return sampled_call(fn, sample_tree=sample_tree)
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return {
        "value": value,
        "elapsed_sec": elapsed,
        "rss_before_mb": float("nan"),
        "rss_after_mb": float("nan"),
        "rss_peak_mb": float("nan"),
        "rss_peak_delta_mb": float("nan"),
        "rss_tree_before_mb": float("nan"),
        "rss_tree_after_mb": float("nan"),
        "rss_tree_peak_mb": float("nan"),
        "rss_tree_peak_delta_mb": float("nan"),
        "rss_samples": 0,
    }


def relative_error(value, reference, eps: float = 1e-12) -> float:
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(value - reference) / (np.linalg.norm(reference) + eps))


def finite_difference_gradient(loss_at_theta, theta, epsilon: float = 1e-5) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = epsilon
        grad[i] = (loss_at_theta(theta + step) - loss_at_theta(theta - step)) / (2.0 * epsilon)
    return grad
