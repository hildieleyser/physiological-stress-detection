"""Build the paired comparisons and the figure used in the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#E6E4DC"
MARK = "#2a78d6"
SEED = 0


def paired(people, first, second):
    a = people.query(first).set_index("subject")["macro_f1"]
    b = people.query(second).set_index("subject")["macro_f1"]
    difference = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    means = rng.choice(difference, (10000, len(difference))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return difference.mean(), low, high, len(difference)


def comparisons(people):
    rows = []
    for task in ["binary", "three_class"]:
        base = f"task == '{task}' and features == 'all_sensors'"
        pairs = {
            "calibrated minus raw, random forest": (
                f"{base} and setting == 'calibrated' and model == 'random_forest'",
                f"{base} and setting == 'raw' and model == 'random_forest'",
            ),
            "all sensors minus no motion, xgboost": (
                f"{base} and setting == 'raw' and model == 'xgboost'",
                f"task == '{task}' and features == 'no_motion' and setting == 'raw' and model == 'xgboost'",
            ),
        }
        for name, (first, second) in pairs.items():
            mean, low, high, n = paired(people, first, second)
            rows.append({"task": task, "comparison": name, "mean_difference": mean, "ci_low": low, "ci_high": high, "people": n})
    return pd.DataFrame(rows)


def figure(people, kfold, out):
    query = "task == 'binary' and model == 'random_forest' and setting == 'raw' and features == 'all_sensors'"
    scores = people.query(query).sort_values("macro_f1")
    leaky = kfold.query("task == 'binary' and model == 'random_forest'")["macro_f1"].iloc[0]
    majority = people.query("task == 'binary' and model == 'majority_class' and setting == 'raw'")["macro_f1"].mean()
    mean = scores["macro_f1"].mean()

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    y = np.arange(len(scores))
    ax.hlines(y, 0.3, scores["macro_f1"], color=GRID, linewidth=1, zorder=1)
    ax.scatter(scores["macro_f1"], y, s=46, color=MARK, zorder=3, edgecolor="white", linewidth=1.2)
    references = {majority: f"Always guess\nmost common {majority:.2f}", mean: f"Mean, unseen\npeople {mean:.2f}", leaky: f"Random split\n{leaky:.3f}"}
    for x, text in references.items():
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.text(x, len(scores) - 0.1, text, ha="right" if x == mean else "center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, scores["subject"])
    ax.set_xlim(0.3, 1.04)
    ax.set_ylim(-0.7, len(scores) - 0.3)
    ax.set_xlabel("Macro F1 on the held-out person, stress against non-stress", color=INK, fontsize=9.5)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Random splitting hides how much the model varies between people", x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor="white")


def main():
    results = Path("results")
    people = pd.read_csv(results / "loso_per_person.csv")
    kfold = pd.read_csv(results / "random_kfold.csv")
    table = comparisons(people)
    table.to_csv(results / "paired_comparisons.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string())
    figure(people, kfold, results / "per_person_scores.png")


if __name__ == "__main__":
    main()
