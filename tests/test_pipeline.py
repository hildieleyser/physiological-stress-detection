import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from wesad_stress.evaluate import calibrate, run_loso
from wesad_stress.features import bvp_peaks, ecg_peaks, hrv
from wesad_stress.io import SafeUnpickler, make_windows


class Evil:
    def __reduce__(self):
        return (os.system, ("echo hacked",))


def test_unpickler_refuses_code():
    payload = pickle.dumps({"x": Evil()})
    with pytest.raises(pickle.UnpicklingError):
        SafeUnpickler(io.BytesIO(payload)).load()


def test_unpickler_loads_arrays():
    arr = np.arange(5.0)
    out = SafeUnpickler(io.BytesIO(pickle.dumps({"x": arr}))).load()
    assert np.array_equal(out["x"], arr)


def test_windows_never_cross_conditions():
    fs = 700
    labels = np.r_[np.zeros(10 * fs), np.ones(400 * fs), np.full(60 * fs, 6), np.full(200 * fs, 2)].astype(int)
    windows = make_windows(labels, fs)
    for w in windows:
        block = labels[int(w.start_s * fs) : int((w.start_s + 60) * fs)]
        assert set(block) == {w.label}


def test_calibration_only_in_first_baseline():
    fs = 700
    labels = np.r_[np.ones(400 * fs), np.full(100 * fs, 2), np.ones(400 * fs)].astype(int)
    windows = make_windows(labels, fs)
    flagged = [w for w in windows if w.is_calibration]
    assert flagged
    assert all(w.label == 1 and w.start_s + 60 <= 300 for w in flagged)


def test_ecg_peaks_recover_heart_rate():
    fs = 700
    t = np.arange(0, 120, 1 / fs)
    ecg = np.zeros_like(t)
    for beat in np.arange(0.5, 119.5, 0.75):
        ecg += np.exp(-0.5 * ((t - beat) / 0.008) ** 2)
    features = hrv(ecg_peaks(ecg, fs), 0, 120, "x")
    assert features["x_hr_mean"] == pytest.approx(80, abs=1)
    assert features["x_rmssd"] < 5


def test_bvp_peaks_recover_heart_rate():
    fs = 64
    t = np.arange(0, 120, 1 / fs)
    features = hrv(bvp_peaks(np.sin(2 * np.pi * 1.25 * t), fs), 0, 120, "x")
    assert features["x_hr_mean"] == pytest.approx(75, abs=1)


def _toy_table():
    rng = np.random.default_rng(0)
    frames = []
    for subject in ["A", "B", "C", "D"]:
        for label in [1, 2, 3]:
            n = 40
            frames.append(pd.DataFrame({
                "subject": subject, "label": label, "start_s": np.arange(n) * 15.0,
                "is_calibration": (label == 1) & (np.arange(n) < 10),
                "signal": rng.normal(label == 2, 0.2, n), "noise": rng.normal(0, 1, n),
            }))
    return pd.concat(frames, ignore_index=True)


def test_loso_scores_every_person_once_and_finds_real_signal():
    per_person, _, _ = run_loso(_toy_table(), ["signal", "noise"], "binary", "logistic_regression")
    assert sorted(per_person["subject"]) == ["A", "B", "C", "D"]
    assert per_person["macro_f1"].mean() > 0.9


def test_noise_only_is_no_better_than_chance():
    table = _toy_table()
    per_person, _, _ = run_loso(table, ["noise"], "three_class", "logistic_regression")
    assert per_person["balanced_accuracy"].mean() < 0.5


def test_calibration_uses_only_that_persons_rest():
    table = _toy_table()
    table.loc[table["subject"] == "A", "signal"] += 100
    out = calibrate(table, ["signal"])
    rest = out[(out["subject"] == "A") & out["is_calibration"]]["signal"]
    assert abs(rest.mean()) < 1e-9
