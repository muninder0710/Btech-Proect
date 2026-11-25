"""How much does leakage inflate scores on UCI Arrhythmia?

Clean: preprocess, SMOTE, select 20 features and KNN are all fitted on the outer-train rows only.
Leaky (the mistake in several papers): preprocess, SMOTE and feature selection run on all 452
records first, then cross-validation runs on the resampled, already-selected data.
Random-label control: the same two variants with the labels shuffled (once per repeat).

Run from code/:  .venv/bin/python leakage_demo.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from arrfs.data import FoldPreprocessor, load_raw, make_task
from arrfs.fitness import class_scores

OUT = Path(__file__).resolve().parent / "results" / "leakage"
N_REPEATS, K_FEATURES = 5, 20


class Prep(BaseEstimator, TransformerMixin):
    """FoldPreprocessor in scikit-learn form; remembers how many rows it was fitted on."""

    def fit(self, X, y=None):
        self.pre_ = FoldPreprocessor().fit(X)
        self.n_fit_ = len(X)
        return self

    def transform(self, X):
        return self.pre_.transform(X)


class AutoSMOTE(SMOTE):
    """SMOTE whose k_neighbors adapts to the rarest class of the data it is fitted on."""

    def fit_resample(self, X, y):
        self.k_neighbors = min(5, np.bincount(y)[np.bincount(y) > 0].min() - 1)
        self.n_fit_ = len(X)
        return super().fit_resample(X, y)


def new_pipeline():
    return Pipeline([("prep", Prep()), ("smote", AutoSMOTE(random_state=0)),
                     ("select", SelectKBest(f_classif, k=K_FEATURES)),
                     ("knn", KNeighborsClassifier(5))])


def clean_scores(X, y, C, splits):
    for tr, te in splits:
        pipe = new_pipeline().fit(X[tr], y[tr])
        assert pipe["prep"].n_fit_ == pipe["smote"].n_fit_ == len(tr), "SMOTE saw non-train rows"
        yield class_scores(y[te], pipe.predict(X[te]), C)


def leaky_scores(X, y, C, n_repeats, seed):
    Xp = Prep().fit_transform(X)
    Xr, yr = AutoSMOTE(random_state=0).fit_resample(Xp, y)
    Xr = SelectKBest(f_classif, k=K_FEATURES).fit_transform(Xr, yr)
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(Xr, yr):
        knn = KNeighborsClassifier(5).fit(Xr[tr], yr[tr])
        yield class_scores(yr[te], knn.predict(Xr[te]), C)


def run_task(task, X, y_raw):
    y, _ = make_task(y_raw, task)
    C = y.max() + 1
    rng = np.random.default_rng(0)
    # (label kind, first repeat number, labels, repeats, split seed). True labels use the main
    # experiment's folds; each permuted repeat gets its own shuffle and its own stratified folds.
    runs = [("true", 0, y, N_REPEATS, 42)]
    runs += [("permuted", r, rng.permutation(y), 1, 42 + r) for r in range(N_REPEATS)]
    rows = []
    for labels, r0, yv, reps, seed in runs:
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed).split(X, yv))
        for variant, scores in [("clean", clean_scores(X, yv, C, splits)),
                                ("leaky", leaky_scores(X, yv, C, reps, seed))]:
            for i, s in enumerate(scores):
                rows.append(dict(task=task, labels=labels, variant=variant, repeat=r0 + i // 5,
                                 fold=i % 5, acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    counts = np.bincount(y)
    return rows, dict(chance_bacc=1 / C, majority_acc=counts.max() / len(y))


def summarize(folds, chance):
    metrics = ["acc", "bacc", "macro_f1"]
    g = folds.groupby(["task", "labels", "variant"], sort=False)[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g = g.reset_index()
    diffs = []
    for (task, labels), d in g.groupby(["task", "labels"], sort=False):
        d = d.set_index("variant")
        diffs.append({"task": task, "labels": labels, "variant": "leaky_minus_clean",
                      **{f"{m}_mean": d.loc["leaky", f"{m}_mean"] - d.loc["clean", f"{m}_mean"] for m in metrics}})
    g = pd.concat([g, pd.DataFrame(diffs)], ignore_index=True)
    g["chance_bacc"] = g["task"].map(lambda t: chance[t]["chance_bacc"])
    g["majority_acc"] = g["task"].map(lambda t: chance[t]["majority_acc"])
    return g


def plot(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    colors = {"clean": "#2a6f97", "leaky": "#d1495b"}
    for ax, task in zip(axes, ["binary", "multiclass"]):
        for j, variant in enumerate(["clean", "leaky"]):
            d = summary[(summary.task == task) & (summary.variant == variant)].set_index("labels").loc[["true", "permuted"]]
            ax.bar(np.arange(2) + (j - 0.5) * 0.38, d.macro_f1_mean, 0.38, yerr=d.macro_f1_std,
                   capsize=3, color=colors[variant], label=variant)
        ax.set_xticks(range(2), ["true labels", "random labels"])
        ax.set_title(task)
    axes[0].set_ylabel("macro-F1 (mean, error bar = sd over 25 folds)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "leakage.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y_raw = load_raw()
    rows, chance = [], {}
    for task in ["binary", "multiclass"]:
        r, chance[task] = run_task(task, X, y_raw)
        rows += r
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "leakage_folds.csv", index=False)
    summary = summarize(folds, chance)
    summary.to_csv(OUT / "leakage_summary.csv", index=False)
    plot(summary)

    for task in chance:
        c = summary[(summary.task == task) & (summary.labels == "permuted") & (summary.variant == "clean")]
        assert c.bacc_mean.iloc[0] < chance[task]["chance_bacc"] + 0.05, f"{task}: clean pipeline beats chance on random labels"
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
