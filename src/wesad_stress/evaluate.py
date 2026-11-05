"""Evaluate on held-out people, with baselines and a look at what the models use.

All settings are fixed here and nothing is tuned, so held-out people are never
used to choose a model. Scores are computed per person and then summarised, as
the person is the unit we want to generalise to.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 0
META = ["subject", "label", "start_s", "is_calibration"]
TASKS = {
    "binary": {1: 0, 3: 0, 2: 1},
    "three_class": {1: 0, 2: 1, 3: 2},
}
MODELS = {
    "majority_class": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, eval_metric="logloss"),
}


def feature_groups(columns):
    motion = [c for c in columns if "_acc_" in c]
    chest = [c for c in columns if c.startswith("chest_")]
    wrist = [c for c in columns if c.startswith("wrist_")]
    return {
        "all_sensors": list(columns),
        "no_motion": [c for c in columns if c not in motion],
        "motion_only": motion,
        "chest_only": chest,
        "wrist_only": wrist,
        "wrist_no_motion": [c for c in wrist if c not in motion],
        "heart_only": [c for c in columns if "_ecg_" in c or "_bvp_" in c],
        "skin_conductance_only": [c for c in columns if "_eda_" in c],
    }


def calibrate(table, features):
    """Express each person's features relative to their own resting recording."""
    out = table.copy()
    out[features] = out[features].astype(float)
    for _, group in table.groupby("subject"):
        rest = group.loc[group["is_calibration"], features]
        spread = rest.std().replace(0, np.nan)
        out.loc[group.index, features] = (group[features] - rest.mean()) / spread
    return out


def fit_predict(model, x_train, y_train, x_test):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(model))
    pipe.fit(x_train, y_train)
    return pipe.predict(x_test)


def score(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
    }


def prepare(table, task):
    return table[table["label"].isin(TASKS[task])], table["label"].map(TASKS[task])


def run_loso(table, features, task, model_name):
    keep, y_all = prepare(table[~table["is_calibration"]], task)
    y_all = y_all.loc[keep.index].to_numpy().astype(int)
    x = keep[features].to_numpy()
    groups = keep["subject"].to_numpy()
    y_pred = np.zeros_like(y_all)
    for train, test in LeaveOneGroupOut().split(x, y_all, groups):
        assert not set(groups[train]) & set(groups[test])
        y_pred[test] = fit_predict(MODELS[model_name], x[train], y_all[train], x[test])
    rows = []
    for subject in pd.unique(groups):
        mask = groups == subject
        rows.append({"subject": subject, **score(y_all[mask], y_pred[mask])})
    return pd.DataFrame(rows), y_all, y_pred


def run_random_kfold(table, features, task, model_name, n_splits=5):
    """The flawed design, kept only to measure how much it flatters a model."""
    keep, y_all = prepare(table[~table["is_calibration"]], task)
    y_all = y_all.loc[keep.index].to_numpy().astype(int)
    x = keep[features].to_numpy()
    y_pred = np.zeros_like(y_all)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for train, test in splitter.split(x, y_all):
        y_pred[test] = fit_predict(MODELS[model_name], x[train], y_all[train], x[test])
    return score(y_all, y_pred)


def bootstrap_ci(values, n=10000):
    rng = np.random.default_rng(SEED)
    values = np.asarray(values)
    means = rng.choice(values, (n, len(values))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarise(per_subject, **labels):
    low, high = bootstrap_ci(per_subject["macro_f1"])
    return {
        **labels,
        "macro_f1": per_subject["macro_f1"].mean(),
        "macro_f1_ci_low": low,
        "macro_f1_ci_high": high,
        "macro_f1_worst_person": per_subject["macro_f1"].min(),
        "balanced_accuracy": per_subject["balanced_accuracy"].mean(),
        "accuracy": per_subject["accuracy"].mean(),
    }


def _job(table, features, task, model, setting, group):
    per_subject, y_true, y_pred = run_loso(table, features, task, model)
    tags = dict(task=task, model=model, setting=setting, features=group)
    return summarise(per_subject, **tags), per_subject.assign(**tags), (tags, y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="results/features.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    raw = pd.read_csv(args.features)
    columns = [c for c in raw.columns if c not in META]
    groups = feature_groups(columns)
    tables = {"raw": raw, "calibrated": calibrate(raw, columns)}

    jobs = []
    for task in TASKS:
        for setting, table in tables.items():
            for model in MODELS:
                jobs.append((table, columns, task, model, setting, "all_sensors"))
            for group, features in groups.items():
                if group != "all_sensors":
                    jobs.append((table, features, task, "xgboost", setting, group))
    done = Parallel(n_jobs=args.jobs)(delayed(_job)(*job) for job in jobs)

    out = Path(args.out)
    summary = pd.DataFrame([d[0] for d in done])
    summary.to_csv(out / "loso_summary.csv", index=False, float_format="%.4f")
    pd.concat([d[1] for d in done]).to_csv(out / "loso_per_person.csv", index=False, float_format="%.4f")

    confusion = []
    for _, _, (tags, y_true, y_pred) in done:
        if tags["model"] != "majority_class" and tags["features"] == "all_sensors":
            names = ["non_stress", "stress"] if tags["task"] == "binary" else ["baseline", "stress", "amusement"]
            matrix = confusion_matrix(y_true, y_pred, labels=range(len(names)))
            for i, truth in enumerate(names):
                for j, predicted in enumerate(names):
                    confusion.append({**tags, "true": truth, "predicted": predicted, "windows": int(matrix[i, j])})
    pd.DataFrame(confusion).to_csv(out / "loso_confusion.csv", index=False)

    leak = []
    for task in TASKS:
        for model in ["logistic_regression", "random_forest", "xgboost"]:
            leak.append({"task": task, "model": model, "design": "random_window_5fold", **run_random_kfold(raw, columns, task, model)})
    pd.DataFrame(leak).to_csv(out / "random_kfold.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 100)
    print(summary.round(3).to_string())
    print(pd.DataFrame(leak).round(3).to_string())


if __name__ == "__main__":
    main()
