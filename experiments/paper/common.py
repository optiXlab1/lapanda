import csv
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import lapanda
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RUNS = ROOT / 'runs'
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')


def flat(value):
    return np.asarray(value, dtype=float).reshape(-1)


def rel(value, reference):
    return float(np.linalg.norm(value-reference) / max(np.linalg.norm(reference), np.finfo(float).eps))
