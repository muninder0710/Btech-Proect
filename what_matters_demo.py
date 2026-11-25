"""Experiment 13: which choices matter on UCI Arrhythmia, missing values, feature selection or the classifier?

Every outer fold scores 2 missing-value levels x 8 selectors x 4 classifiers for two and twelve classes. The
questions, rules and figures were fixed before the run in design/amendment-13-what-matters.md.

Run from code/:
    .venv/bin/python what_matters_demo.py --smoke   # synthetic data, 1 repeat, small budget -> results/what_matters_smoke/
    .venv/bin/python what_matters_demo.py           # UCI Arrhythmia, 5 x 5 folds            -> results/what_matters/
A stopped run resumes: folds already saved in <out>/folds/ are not recomputed.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

# Importing this module restarts the process once with scikit-learn's libomp when Homebrew's is missing.
import tuning_on_test_demo  # noqa: E402, F401

import argparse  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import extra_optimizers  # noqa: E402
from arrfs import optimizers  # noqa: E402
from arrfs.data import FoldPreprocessor, load_task  # noqa: E402
from arrfs.experiment import FoldKey  # noqa: E402
from arrfs.fitness import WrapperFitness, class_scores  # noqa: E402
from informed_init_demo import informed_start  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = {"two-class": "arrhythmia-binary", "twelve-class": "arrhythmia-multiclass"}
OUTER_SEED, REPEATS, SPLITS, INNER_SPLITS = 2027, 5, 5, 3
POPULATION, BUDGET = 20, 2000
MISSING = ("median", "median + indicators")
FILTERS = ("ReliefF", "mutual information", "mRMR")
WRAPPERS = {  # selector -> (algorithm, fitness kind, start)
    "IBGWO4 as published": ("IBGWO4", "accuracy", "random"),
    "GWO-CPO guided": ("GWO-CPO", "imbalance", "guided"),
    "BPSO guided": ("BPSO", "imbalance", "guided"),
    "GA guided": ("GA", "imbalance", "guided"),
}
SELECTORS = ("none", *FILTERS, *WRAPPERS)
CLASSIFIERS = ("5-NN", "SVM", "random forest", "XGBoost")
K_GRID = (5, 10, 20, 30, 50, 100)
RELIEFF_NEIGHBOURS, MRMR_BINS = 10, 10


# ---------- data ----------

def synthetic(task):
    """Stand-in data with the real shape problems (missing values that carry class information, rare classes),
    so the pipeline can be tested without touching the arrhythmia results."""
    rng = np.random.default_rng(7)
    if task == "two-class":
        X, y = make_classification(452, 60, n_informative=8, n_classes=2, weights=[0.54], random_state=7)
    else:
        weights = np.array([245, 50, 44, 25, 22, 15, 15, 13, 9, 5, 5, 4]) / 452
        X, y = make_classification(452, 60, n_informative=8, n_classes=12, n_clusters_per_class=1,
                                   weights=weights, random_state=7)
    X[rng.random(452) < 0.8, 3] = np.nan                       # mostly missing, like the J angle
    X[(y == 1) & (rng.random(452) < 0.3), 5] = np.nan          # missing more often in one class
    return X, y, len(np.unique(y))


def load(task, smoke):
    if smoke:
        return synthetic(task)
    data = load_task(TASKS[task])
    return data.X, data.y, len(data.class_names)


class Preprocessor:
    """FoldPreprocessor, optionally followed by one 0/1 column per raw attribute with a missing value in the
    training rows (1 = missing). Indicators are the last columns and survive even when the values are dropped."""

    def __init__(self, indicators):
        self.indicators = indicators

    def fit(self, X):
        self.base_ = FoldPreprocessor().fit(X)
        miss = np.isnan(X)
        cols = np.flatnonzero(miss.any(0)) if self.indicators else np.array([], dtype=int)
        self.ind_ = cols[miss[:, cols].min(0) != miss[:, cols].max(0)]  # a constant indicator says nothing
        return self

    def transform(self, X):
        return np.hstack([self.base_.transform(X), np.isnan(X[:, self.ind_]).astype(float)])


# ---------- filters: each returns column indices, best first ----------

def relieff_rank(X, y, seed=None):
    """ReliefF (Kononenko 1994): k nearest hits and misses per class, Manhattan distance on min-max scaled
    columns, misses weighted by the class prior. Every training row is a sample."""
    n, d = X.shape
    classes, counts = np.unique(y, return_counts=True)
    prior = dict(zip(classes, counts / n))
    dist = cdist(X, X, "cityblock")
    np.fill_diagonal(dist, np.inf)
    members = {c: np.flatnonzero(y == c) for c in classes}
    w = np.zeros(d)
    for i in range(n):
        for c in classes:
            idx = members[c][members[c] != i]
            if len(idx) == 0:
                continue
            near = idx[np.argsort(dist[i, idx], kind="stable")[:RELIEFF_NEIGHBOURS]]
            diff = np.abs(X[near] - X[i]).mean(0)
            w += -diff if c == y[i] else prior[c] / (1 - prior[y[i]]) * diff
    return np.argsort(-w / n, kind="stable")


def mi_rank(X, y, seed):
    return np.argsort(-mutual_info_classif(X, y, random_state=seed), kind="stable")


def _bins(X):
    """Each column cut into at most MRMR_BINS equal-frequency bins, as integer codes."""
    out = np.empty(X.shape, dtype=np.int64)
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, MRMR_BINS + 1)[1:-1]))
        out[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return out


def _mi(a, b):
    """Mutual information (nats) of two integer-coded vectors."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    joint = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).reshape(a.max() + 1, -1)
    p = joint / joint.sum()
    outer = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / outer[nz])).sum())


def mrmr_rank(X, y, seed=None):
    """Greedy mRMR, difference form (Peng, Long & Ding 2005): relevance I(f; class) minus the mean I(f; s)
    over the features s already chosen. Ranks the first max(K_GRID) features."""
    B = _bins(X)
    d = B.shape[1]
    relevance = np.array([_mi(B[:, j], y) for j in range(d)])
    redundancy = np.zeros(d)
    chosen = [int(np.argmax(relevance))]
    free = np.ones(d, bool)
    free[chosen[0]] = False
    while len(chosen) < min(max(K_GRID), d):
        last = B[:, chosen[-1]]
        redundancy[free] += [_mi(B[:, j], last) for j in np.flatnonzero(free)]
        score = np.where(free, relevance - redundancy / len(chosen), -np.inf)
        chosen.append(int(np.argmax(score)))
        free[chosen[-1]] = False
    return np.array(chosen)


RANKERS = {"ReliefF": relieff_rank, "mutual information": mi_rank, "mRMR": mrmr_rank}


# ---------- classifiers ----------

def fit_predict(name, Xa, ya, Xb, seed):
    assert Xa.shape[1] > 0
    if name == "XGBoost":  # XGBoost needs labels 0..m-1 with none missing
        classes, codes = np.unique(ya, return_inverse=True)
        return classes[XGBClassifier(n_jobs=1, random_state=0).fit(Xa, codes).predict(Xb)]
    model = {"5-NN": lambda: KNeighborsClassifier(5),
             "SVM": lambda: SVC(kernel="rbf", class_weight="balanced"),
             "random forest": lambda: RandomForestClassifier(300, random_state=seed, n_jobs=1)}[name]()
    return model.fit(Xa, ya).predict(Xb)


# ---------- one outer fold ----------

def run_fold(task, repeat, fold, tr, te, smoke):
    warnings.filterwarnings("ignore", message="The least populated class")
    X, y, C = load(task, smoke)
    key = FoldKey(task, repeat, fold)
    ytr, yte = y[tr], y[te]
    budget = 200 if smoke else BUDGET
    rows, preds, curves = [], [], []
    for missing in MISSING:
        prep = Preprocessor(missing != "median").fit(X[tr])
        Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
        D, n_ind = Xtr.shape[1], len(prep.ind_)
        ind_attr = np.concatenate([np.full(D - n_ind, -1), prep.ind_ + 1])  # raw attribute number of an indicator
        inner = list(StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=key.seed()).split(Xtr, ytr))

        # subsets: {selector: {classifier: (columns, chosen k or None)}}
        subsets = {"none": {c: (np.arange(D), None) for c in CLASSIFIERS}}
        for f, rank in RANKERS.items():
            seed = key.seed(f, missing)
            inner_ranks = [rank(Xtr[a], ytr[a], seed) for a, _ in inner]
            outer_rank = rank(Xtr, ytr, seed)
            subsets[f] = {}
            for clf in CLASSIFIERS:
                grid = sorted({min(k, len(outer_rank)) for k in K_GRID})
                score = {k: np.mean([class_scores(ytr[b], fit_predict(clf, Xtr[a][:, r[:k]], ytr[a], Xtr[b][:, r[:k]],
                                                                      key.seed("rf")), C)["macro_f1"]
                                     for (a, b), r in zip(inner, inner_ranks)]) for k in grid}
                best = max(grid, key=lambda k: (score[k], -k))
                subsets[f][clf] = (np.sort(outer_rank[:best]), best)
        for sel, (alg, kind, start) in WRAPPERS.items():
            fitness = WrapperFitness(Xtr, ytr, C, inner, kind, k=5)
            X0 = (informed_start(key, Xtr, ytr) if start == "guided"
                  else np.random.default_rng(key.seed("X0", missing)).random((POPULATION, D)))
            engine = optimizers if alg in optimizers.ALGORITHMS else extra_optimizers
            t0 = time.perf_counter()
            res = engine.search(alg, fitness, X0, budget, np.random.default_rng(key.seed(alg, kind, missing)))
            subsets[sel] = {c: (np.flatnonzero(res.mask), None) for c in CLASSIFIERS}
            curves.append(dict(task=task, repeat=repeat, fold=fold, missing=missing, selector=sel,
                               seconds=time.perf_counter() - t0, curve=res.curve))

        for sel in SELECTORS:
            for clf in CLASSIFIERS:
                cols, k = subsets[sel][clf]
                yhat = fit_predict(clf, Xtr[:, cols], ytr, Xte[:, cols], key.seed("rf"))
                s = class_scores(yte, yhat, C)
                kept_ind = ind_attr[cols][ind_attr[cols] > 0]
                rows.append(dict(task=task, repeat=repeat, fold=fold, missing=missing, selector=sel, classifier=clf,
                                 n_features=len(cols), n_columns=D, n_indicator_columns=n_ind,
                                 n_indicators_kept=len(kept_ind), indicators_kept=";".join(map(str, kept_ind)),
                                 k_chosen=k, acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"],
                                 **{f"recall_{c}": s["recall"][c] for c in range(C)}))
                preds.append(dict(task=task, repeat=repeat, fold=fold, missing=missing, selector=sel, classifier=clf,
                                  patient=te, y=yte, yhat=yhat))
    return rows, preds, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 1 repeat, budget 200")
    args = ap.parse_args()
    out = HERE / "results" / ("what_matters_smoke" if args.smoke else "what_matters")
    (out / "folds").mkdir(parents=True, exist_ok=True)
    repeats = 1 if args.smoke else REPEATS
    jobs = []
    for task in TASKS:
        y = load(task, args.smoke)[1]
        cv = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=repeats, random_state=OUTER_SEED)
        jobs += [(task, i // SPLITS, i % SPLITS, tr, te) for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y))]
    todo = [j for j in jobs if not (out / "folds" / f"{j[0]}_r{j[1]}_f{j[2]}.pkl").exists()]
    print(f"{len(jobs)} folds, {len(jobs) - len(todo)} already saved", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(run_fold, *job, args.smoke): job for job in todo}
        for i, fut in enumerate(futures, 1):
            task, repeat, fold = futures[fut][:3]
            (out / "folds" / f"{task}_r{repeat}_f{fold}.pkl").write_bytes(pickle.dumps(fut.result()))
            print(f"[{i}/{len(todo)}] {task} r{repeat} f{fold} saved at {time.perf_counter() - start:.0f}s", flush=True)

    parts = [pickle.loads((out / "folds" / f"{t}_r{r}_f{f}.pkl").read_bytes()) for t, r, f, *_ in jobs]
    rows = pd.DataFrame([r for p in parts for r in p[0]])
    assert len(rows) == len(jobs) * len(MISSING) * len(SELECTORS) * len(CLASSIFIERS)
    assert not rows.duplicated(["task", "repeat", "fold", "missing", "selector", "classifier"]).any()
    rows.to_csv(out / "what_matters_rows.csv", index=False)
    preds = pd.DataFrame([p for part in parts for p in part[1]]).explode(["patient", "y", "yhat"])
    preds.to_csv(out / "what_matters_predictions.csv", index=False)
    curves = pd.DataFrame([c for part in parts for c in part[2]])
    curves = curves.assign(point=[list(range(len(c))) for c in curves.curve]).explode(["point", "curve"])
    curves.rename(columns={"curve": "best_fitness"}).to_csv(out / "what_matters_curves.csv", index=False)
    print(rows.groupby(["task", "missing", "selector", "classifier"], sort=False).macro_f1.mean().round(3)
          .unstack("classifier").to_string() if args.smoke else "rows saved; run analyze_what_matters.py")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
