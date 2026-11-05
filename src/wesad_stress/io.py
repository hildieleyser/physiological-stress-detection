"""Load WESAD recordings and cut them into labelled windows.

WESAD ships as Python pickles, and unpickling can run arbitrary code. The
loader below only allows the handful of NumPy names the files need.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
CHEST_FS = 700
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
WINDOW_S = 60
STEP_S = 15
CALIBRATION_S = 300

_ALLOWED = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")


def load_subject(data_dir: str | Path, subject: str) -> dict:
    path = Path(data_dir) / subject / f"{subject}.pkl"
    with open(path, "rb") as handle:
        return SafeUnpickler(handle, encoding="latin1").load()


@dataclass(frozen=True)
class Window:
    label: int
    start_s: float
    run_start_s: float
    is_calibration: bool


def label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (label, start_index, end_index) for each unbroken run of a kept label."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(labels)]
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends) if int(labels[s]) in LABELS]


def make_windows(labels: np.ndarray, fs: int = CHEST_FS) -> list[Window]:
    """Cut windows that sit wholly inside one condition.

    The first CALIBRATION_S seconds of the first baseline are marked. They stand
    in for a short resting recording taken when someone first wears the device,
    and they are never scored.
    """
    windows: list[Window] = []
    first_baseline_seen = False
    for label, start, end in label_runs(labels):
        run_start = start / fs
        run_end = end / fs
        is_first_baseline = label == 1 and not first_baseline_seen
        first_baseline_seen = first_baseline_seen or label == 1
        t = run_start
        while t + WINDOW_S <= run_end:
            calibration = is_first_baseline and t + WINDOW_S <= run_start + CALIBRATION_S
            windows.append(Window(label, t, run_start, calibration))
            t += STEP_S
    return windows
