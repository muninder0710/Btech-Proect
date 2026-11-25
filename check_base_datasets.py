"""Checks of base_datasets_demo.py, to run before the experiment.

    python check_base_datasets.py        # a few minutes; exits non-zero on the first failure

1. Every one of the nine datasets must load with the size of Khaseeb et al.'s Table 1, hold no missing value,
   and have the content checksum that fetch_gems.py recorded in data/gems/SOURCES.txt.
2. HoldoutImbalance must return the imbalance-aware score of the reported rows, computed here a second way
   with scikit-learn's KNN.
3. On Arrhythmia, where Amendment 5 already ran the base paper's protocol, the job of base_datasets_demo.py
   must reproduce the saved rows of results/base_protocol exactly, for the two arms that both scripts run.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

import base_datasets_demo as demo
from arrfs.data import load_task
from arrfs.fitness import class_scores
from base_protocol_demo import K, N_RUNS, TEST_SIZE

SAVED = Path(__file__).resolve().parent / "results" / "base_protocol" / "base_protocol_runs.csv"


def check_datasets() -> None:
    recorded = dict(reversed(line.split()[1:3]) for line in (demo.DATA / "SOURCES.txt").read_text().splitlines()
                    if line.strip().startswith("content"))
    for name, (n, genes, classes) in demo.DATASETS.items():
        X, y, names = demo.load_dataset(name)
        digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
        assert recorded[f"{name}.npz"] == digest, f"{name}.npz does not hold the arrays fetch_gems.py recorded"
        assert X.shape == (n, genes) and np.isfinite(X).all(), (name, X.shape)
        counts = np.bincount(y, minlength=classes)
        assert len(names) == classes and counts.min() > 0, (name, counts)
        print(f"{name:15s} ok  {n} x {genes}, {classes} classes with {counts.min()} to {counts.max()} patients")


def check_holdout_imbalance() -> None:
    rng = np.random.default_rng(0)
    Xtr, Xte, C = rng.random((40, 12)), rng.random((15, 12)), 3
    ytr, yte = rng.integers(C, size=40), rng.integers(C, size=15)
    fitness = demo.HoldoutImbalance(Xtr, ytr, Xte, yte, C, k=K)
    for _ in range(20):
        mask = rng.random(12) < 0.5
        mask[rng.integers(12)] = True
        s = class_scores(yte, KNeighborsClassifier(K).fit(Xtr[:, mask], ytr).predict(Xte[:, mask]), C)
        expected = 0.7 * (1 - s["macro_f1"]) + 0.3 * (1 - s["bacc"]) + 0.01 * mask.sum() / 12
        assert np.isclose(fitness(mask), expected, rtol=0, atol=1e-12), (fitness(mask), expected)
    assert fitness(np.zeros(12, bool)) == demo.EMPTY_PENALTY
    print("HoldoutImbalance ok  equals 0.7 (1 - macro-F1) + 0.3 (1 - balanced accuracy) + 0.01 |S| / D on the reported rows, 20 masks")


def check_matches_amendment_5() -> None:
    saved = pd.read_csv(SAVED, float_precision="round_trip").set_index(["task", "run", "reading", "algorithm"])
    columns = ["n_features", "best_fitness", "acc", "bacc", "macro_f1"]
    real_loader = demo.load_dataset
    try:
        for task in ("arrhythmia-multiclass", "arrhythmia-binary"):
            data = load_task(task)
            demo.load_dataset = lambda name, data=data: (data.X, data.y, data.class_names)
            splits = list(ShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=42).split(np.zeros(len(data.y))))
            for run in (0, 17):
                _, _, rows = demo.run_job((task, run, *splits[run]))
                for row in rows:
                    if row["fitness"] == "accuracy" and row["start"] == "random":
                        want = saved.loc[(task, run, row["reading"], row["algorithm"]), columns]
                        got = pd.Series({c: row[c] for c in columns})
                        assert (want.to_numpy(float) == got.to_numpy(float)).all(), (task, run, row["arm"], want, got)
                print(f"{task}, run {run}: both arms shared with Amendment 5 reproduce its saved rows exactly, in both readings")
    finally:
        demo.load_dataset = real_loader


if __name__ == "__main__":
    check_datasets()
    check_holdout_imbalance()
    check_matches_amendment_5()
