"""Named features for each window, grouped by what they measure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage, signal

from .io import CHEST_FS, SUBJECTS, WINDOW_S, WRIST_FS, load_subject, make_windows

EDA_FS = 4
RESP_FS = 10


def _filter(x, fs, low=None, high=None, order=3):
    nyquist = fs / 2
    if low and high:
        sos = signal.butter(order, [low / nyquist, high / nyquist], "band", output="sos")
    elif high:
        sos = signal.butter(order, high / nyquist, "low", output="sos")
    else:
        sos = signal.butter(order, low / nyquist, "high", output="sos")
    return signal.sosfiltfilt(sos, x)


def block_mean(x, factor):
    n = len(x) // factor
    return x[: n * factor].reshape(n, factor).mean(axis=1)


def ecg_peaks(ecg, fs):
    """R peak times in seconds, from the band-passed, squared and smoothed ECG."""
    band = _filter(ecg, fs, 5, 15)
    energy = band**2
    energy = ndimage.uniform_filter1d(energy, int(0.08 * fs))
    floor = np.median(energy)
    spread = np.median(np.abs(energy - floor))
    threshold = max(floor + 5 * spread, 0.1 * np.percentile(energy, 99))
    peaks, _ = signal.find_peaks(energy, distance=int(0.33 * fs), height=threshold)
    reach = int(0.06 * fs)
    refined = [max(p - reach, 0) + int(np.argmax(band[max(p - reach, 0) : p + reach])) for p in peaks]
    return np.unique(refined) / fs


def bvp_peaks(bvp, fs):
    clean = _filter(bvp, fs, 0.5, 4)
    peaks, _ = signal.find_peaks(clean, distance=int(0.33 * fs), prominence=0.2 * np.std(clean))
    return peaks / fs


def hrv(peak_times, start, stop, prefix):
    """Heart rate and variability from beats inside a window."""
    beats = peak_times[(peak_times >= start) & (peak_times < stop)]
    ibi = np.diff(beats)
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    names = ["hr_mean", "hr_std", "sdnn", "rmssd", "pnn50"]
    if len(ibi) < 10:
        return {f"{prefix}_{n}": np.nan for n in names}
    previous, current = ibi[:-1], ibi[1:]
    keep = np.abs(current - previous) / previous < 0.2
    steps = (current - previous)[keep]
    if len(steps) < 5:
        return {f"{prefix}_{n}": np.nan for n in names}
    rate = 60 / ibi
    return {
        f"{prefix}_hr_mean": rate.mean(),
        f"{prefix}_hr_std": rate.std(),
        f"{prefix}_sdnn": ibi.std() * 1000,
        f"{prefix}_rmssd": np.sqrt(np.mean(steps**2)) * 1000,
        f"{prefix}_pnn50": np.mean(np.abs(steps) > 0.05),
    }


def slope(x, fs):
    t = np.arange(len(x)) / fs
    return np.polyfit(t, x, 1)[0]


def eda_features(eda, phasic, fs, prefix):
    peaks, _ = signal.find_peaks(phasic, prominence=0.05)
    return {
        f"{prefix}_mean": eda.mean(),
        f"{prefix}_std": eda.std(),
        f"{prefix}_slope": slope(eda, fs),
        f"{prefix}_scr_count": float(len(peaks)),
        f"{prefix}_phasic_std": phasic.std(),
    }


def temp_features(temp, fs, prefix):
    return {f"{prefix}_mean": temp.mean(), f"{prefix}_std": temp.std(), f"{prefix}_slope": slope(temp, fs)}


def acc_features(acc, prefix):
    magnitude = np.linalg.norm(acc, axis=1)
    return {
        f"{prefix}_mag_mean": magnitude.mean(),
        f"{prefix}_mag_std": magnitude.std(),
        f"{prefix}_x_mean": acc[:, 0].mean(),
        f"{prefix}_y_mean": acc[:, 1].mean(),
        f"{prefix}_z_mean": acc[:, 2].mean(),
    }


def extract_subject(data_dir, subject):
    record = load_subject(data_dir, subject)
    labels = np.asarray(record["label"]).ravel().astype(int)
    chest = {k: np.asarray(v, dtype=float) for k, v in record["signal"]["chest"].items()}
    wrist = {k: np.asarray(v, dtype=float) for k, v in record["signal"]["wrist"].items()}
    del record

    ecg_times = ecg_peaks(chest["ECG"].ravel(), CHEST_FS)
    bvp_times = bvp_peaks(wrist["BVP"].ravel(), WRIST_FS["BVP"])

    chest_eda = block_mean(chest["EDA"].ravel(), CHEST_FS // EDA_FS)
    chest_eda = _filter(chest_eda, EDA_FS, high=1)
    chest_phasic = chest_eda - ndimage.median_filter(chest_eda, size=8 * EDA_FS, mode="nearest")
    wrist_eda = _filter(wrist["EDA"].ravel(), EDA_FS, high=1)
    wrist_phasic = wrist_eda - ndimage.median_filter(wrist_eda, size=8 * EDA_FS, mode="nearest")

    chest_temp = block_mean(chest["Temp"].ravel(), CHEST_FS // EDA_FS)
    wrist_temp = wrist["TEMP"].ravel()

    resp = _filter(block_mean(chest["Resp"].ravel(), CHEST_FS // RESP_FS), RESP_FS, 0.1, 0.7)
    breaths, _ = signal.find_peaks(resp, distance=int(1.5 * RESP_FS), prominence=0.25 * resp.std())

    emg = _filter(chest["EMG"].ravel(), CHEST_FS, low=20)
    emg_energy = np.r_[0, np.cumsum(emg**2)]
    del emg

    chest_acc = chest["ACC"][:: CHEST_FS // 32]
    wrist_acc = wrist["ACC"] / 64

    rows = []
    for window in make_windows(labels):
        t0, t1 = window.start_s, window.start_s + WINDOW_S
        row = {
            "subject": subject,
            "label": window.label,
            "start_s": t0,
            "is_calibration": window.is_calibration,
        }
        row.update(hrv(ecg_times, t0, t1, "chest_ecg"))
        row.update(hrv(bvp_times, t0, t1, "wrist_bvp"))

        a, b = int(t0 * EDA_FS), int(t1 * EDA_FS)
        row.update(eda_features(chest_eda[a:b], chest_phasic[a:b], EDA_FS, "chest_eda"))
        row.update(eda_features(wrist_eda[a:b], wrist_phasic[a:b], EDA_FS, "wrist_eda"))
        row.update(temp_features(chest_temp[a:b], EDA_FS, "chest_temp"))
        row.update(temp_features(wrist_temp[a:b], EDA_FS, "wrist_temp"))

        r0, r1 = int(t0 * RESP_FS), int(t1 * RESP_FS)
        segment = resp[r0:r1]
        inside = breaths[(breaths >= r0) & (breaths < r1)]
        row["chest_resp_rate"] = len(inside) * 60 / WINDOW_S
        row["chest_resp_std"] = segment.std()

        e0, e1 = int(t0 * CHEST_FS), int(t1 * CHEST_FS)
        row["chest_emg_rms"] = np.sqrt((emg_energy[e1] - emg_energy[e0]) / (e1 - e0))

        q0, q1 = int(t0 * 32), int(t1 * 32)
        row.update(acc_features(chest_acc[q0:q1], "chest_acc"))
        row.update(acc_features(wrist_acc[q0:q1], "wrist_acc"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder holding S2, S3 and so on")
    parser.add_argument("--out", default="results/features.csv")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    frames = Parallel(n_jobs=args.jobs)(delayed(extract_subject)(args.data, s) for s in SUBJECTS)
    table = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.groupby(["subject", "label"]).size().unstack())


if __name__ == "__main__":
    main()
