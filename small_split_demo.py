"""Experiment 3(c): how much can one 42-patient test split tell us about a 95% accuracy claim?

Replicates the pipeline Dhanka & Maini 2025 describe for normal vs arrhythmia, without PSO tuning.
Paper-style arm: the fitted steps (zero-count filter, correlation filter, min-max scaling) are fitted
on all 420 records before splitting. Clean arm: they are fitted on the 378 training records of each split.
HistGradientBoostingClassifier stands in for XGBoost, which is not installed.

Design: design/amendment-4-review-and-split.md, section "Experiment 3(c): one small test split".
Run from code/:  .venv/bin/python small_split_demo.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from arrfs.data import load_raw, make_task  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "small_split"
N_SPLITS, TEST_SIZE = 200, 0.1
N_ZERO_KEEP, N_CORR_KEEP = 189, 100
N_BOOT, BOOT_SEED = 2000, 20261
CLAIM_CORRECT, N_TEST = 40, 42
CLASSIFIERS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=5000),
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "GaussianNB": lambda: GaussianNB(),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=0),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=0),
}


def load_420():
    """Binary arrhythmia without attribute 14 and without records missing attribute 11, 12, 13 or 15."""
    X, y_raw = load_raw()
    y, _ = make_task(y_raw, "binary")
    rows = ~np.isnan(X[:, [10, 11, 12, 14]]).any(1)  # attributes are 1-indexed, columns 0-indexed
    X = np.delete(X, 13, axis=1)[rows]
    y = y[rows]
    assert X.shape == (420, 278) and not np.isnan(X).any(), X.shape
    assert np.bincount(y).tolist() == [237, 183], np.bincount(y)
    return X, y


class Steps:
    """Zero-count filter, then correlation filter, then min-max scaling, all fitted on the rows given to fit."""

    def fit(self, X, y):
        self.n_fit_ = len(X)
        zeros = (X == 0).sum(0)
        self.zero_keep_ = np.sort(np.argsort(zeros, kind="stable")[:N_ZERO_KEEP])
        Xz = X[:, self.zero_keep_]
        yc = y - y.mean()
        xc = Xz - Xz.mean(0)
        denom = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
        constant = np.ptp(Xz, axis=0) == 0  # a constant column's float mean is not exact, so test the range
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):  # spurious Accelerate flags
            cov = xc.T @ yc
        corr = np.where(constant, 0.0, np.abs(cov) / np.where(constant, 1.0, denom))
        self.corr_keep_ = np.sort(np.argsort(-corr, kind="stable")[:N_CORR_KEEP])
        Xc = Xz[:, self.corr_keep_]
        self.lo_, hi = Xc.min(0), Xc.max(0)
        self.rng_ = np.where(hi > self.lo_, hi - self.lo_, 1.0)
        return self

    def transform(self, X):
        return (X[:, self.zero_keep_][:, self.corr_keep_] - self.lo_) / self.rng_  # test rows are not clipped


def correct_counts(Xtr, ytr, Xte, yte):
    """Correct test predictions of every classifier."""
    # macOS Accelerate BLAS raises spurious FP flags in float64 matmul (see arrfs/fitness.py).
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return {name: int((make().fit(Xtr, ytr).predict(Xte) == yte).sum()) for name, make in CLASSIFIERS.items()}


def evaluate(X, y, tr, te, arm, steps_all):
    """Dict classifier -> correct predictions out of len(te) for one split and one arm."""
    if arm == "paper":
        steps = steps_all
    else:
        steps = Steps().fit(X[tr], y[tr])
        assert steps.n_fit_ == len(tr), "clean arm fitted a step on non-train rows"
    return correct_counts(steps.transform(X[tr]), y[tr], steps.transform(X[te]), y[te])


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bootstrap_ci(diff):
    """95% percentile bootstrap interval of the mean of paired differences."""
    rng = np.random.default_rng(BOOT_SEED)
    means = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def plot(acc, best):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    edges = (np.arange(N_TEST + 2) - 0.5) / N_TEST
    for ax, arm, color in zip(axes, ["clean", "paper"], ["#2a6f97", "#d1495b"]):
        ax.hist(acc[(arm, best)], bins=edges, color=color)
        ax.axvline(CLAIM_CORRECT / N_TEST, color="black", ls="--", lw=1.2)
        ax.set_title("paper-style" if arm == "paper" else "clean", fontsize=11)
        ax.set_xlabel("test accuracy (42 patients)")
    axes[0].set_xlim(0.6, 1.0)
    axes[0].set_ylabel("splits (of 200)")
    axes[0].text(CLAIM_CORRECT / N_TEST, axes[0].get_ylim()[1] * 0.95, " 95.24%", ha="left", va="top")
    fig.suptitle(f"{best}: 200 stratified 90:10 splits", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "small_split.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    X, y = load_420()
    warnings.simplefilter("error", ConvergenceWarning)  # the logistic regression must actually converge
    steps_all = Steps().fit(X, y)
    assert steps_all.n_fit_ == 420

    splits = list(StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42).split(X, y))
    assert all(len(te) == N_TEST and len(tr) == 378 for tr, te in splits)
    rows = []
    for i, (tr, te) in enumerate(splits):
        for arm in ["paper", "clean"]:
            for clf, k in evaluate(X, y, tr, te, arm, steps_all).items():
                rows.append(dict(split=i, arm=arm, classifier=clf, correct=k, acc=k / N_TEST))
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "small_split_splits.csv", index=False)
    acc = {key: g.sort_values("split").acc.to_numpy() for key, g in folds.groupby(["arm", "classifier"])}

    q = []
    for arm in ["paper", "clean"]:
        for clf in CLASSIFIERS:
            a = acc[(arm, clf)]
            p5, med, p95 = np.percentile(a, [5, 50, 95])
            q.append(dict(arm=arm, classifier=clf, min=a.min(), p5=p5, median=med, p95=p95, max=a.max(),
                          share_at_least_40_of_42=(np.round(a * N_TEST) >= CLAIM_CORRECT).mean()))
    summary = pd.DataFrame(q)

    clean_median = summary[summary.arm == "clean"].set_index("classifier")["median"]
    best = clean_median.idxmax()  # a tie goes to the first classifier in CLASSIFIERS order
    summary["best_clean_median"] = summary.classifier == best
    summary.to_csv(OUT / "small_split_summary.csv", index=False)

    d = []
    for clf in CLASSIFIERS:
        diff = acc[("paper", clf)] - acc[("clean", clf)]
        lo, hi = bootstrap_ci(diff)
        d.append(dict(classifier=clf, mean_diff=diff.mean(), ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0),
                      best_clean_median=clf == best))
    diffs = pd.DataFrame(d)
    diffs.to_csv(OUT / "small_split_paired_diff.csv", index=False)

    c = summary[(summary.arm == "clean") & (summary.classifier == best)].iloc[0]
    claims = pd.DataFrame([
        dict(claim="(i) 42-patient test set too small for a 95% claim", classifier=best,
             value=c.p95 - c.p5, threshold=0.10, holds=bool(c.p95 - c.p5 > 0.10)),
        *[dict(claim="(ii) selecting on all records inflated accuracy", classifier=r.classifier, value=r.mean_diff,
               threshold=0.0, holds=r.excludes_zero and r.mean_diff > 0) for r in diffs.itertuples()],
    ])
    claims["best_clean_median"] = claims.classifier == best
    claims.to_csv(OUT / "small_split_claims.csv", index=False)

    lo, hi = wilson(CLAIM_CORRECT, N_TEST)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X, y)
    ref_acc = {clf: [] for clf in CLASSIFIERS}
    for tr, te in cv:
        for clf, k in evaluate(X, y, tr, te, "clean", steps_all).items():
            ref_acc[clf].append(k / len(te))
    ref = pd.DataFrame([dict(classifier=clf, cv_acc_mean=np.mean(a), cv_acc_sd=np.std(a, ddof=1), n_folds=len(a))
                        for clf, a in ref_acc.items()])
    ref.to_csv(OUT / "small_split_reference_cv.csv", index=False)

    tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=0)  # the paper's split, unstratified
    paper_split = pd.DataFrame([dict(arm=arm, test_size=len(te), classifier=clf, acc=k / len(te))
                                for arm in ["paper", "clean"] for clf, k in evaluate(X, y, tr, te, arm, steps_all).items()])
    paper_split.to_csv(OUT / "small_split_papers_split.csv", index=False)

    plot(acc, best)

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    print(claims.round(4).to_string(index=False))
    print(ref.round(4).to_string(index=False))
    print(paper_split.round(4).to_string(index=False))
    print(f"Wilson 95% interval for {CLAIM_CORRECT} of {N_TEST}: {lo:.4f} to {hi:.4f}")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
