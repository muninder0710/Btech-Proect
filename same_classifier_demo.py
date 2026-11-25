"""Experiment 4: the proposed feature selection against the base paper's filter, with the same classifier.

Dhanka & Maini 2025 pick features with two filters and classify with XGBoost. The proposed model picks features
with an informed-start GWO-CPO search. Here both feed the same default XGBoost on 25 fresh outer folds, for two
classes (the paper's 420 patients) and twelve classes (the project's 452 patients). The proposed search wraps a
light XGBoost, so the subset is chosen for the classifier that will use it.

Design: design/amendment-8-same-classifier.md.
Run from code/:  .venv/bin/python same_classifier_demo.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

# Importing this module restarts the process once with scikit-learn's libomp when Homebrew's is missing.
import tuning_on_test_demo as tuning  # noqa: E402

import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from analyze import _paired  # noqa: E402
from arrfs.data import FoldPreprocessor, load_task  # noqa: E402
from arrfs.experiment import FoldKey  # noqa: E402
from arrfs.fitness import WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import search  # noqa: E402
from informed_init_demo import informed_start  # noqa: E402
from small_split_demo import N_CORR_KEEP, N_ZERO_KEEP, Steps, load_420  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "same_classifier"
TASKS = {"two-class": ("accuracy", "acc"), "twelve-class": ("imbalance", "macro_f1")}  # fitness kind, primary metric
OUTER_SEED, REPEATS, SPLITS = 2026, 5, 5
BUDGET = 2000
LIGHT = dict(n_estimators=50, max_depth=3)  # the XGBoost inside the search
MARGIN, FEWER_THAN = 0.02, 50
METRICS = ["acc", "bacc", "macro_f1"]
COMPARISONS = [("proposed", "base filter"), ("proposed", "all features"), ("base filter", "all features"),
               ("proposed, 5-NN inside", "base filter"), ("proposed, tuned", "base filter, tuned")]


def load(task):
    """(attributes with possible missing values, labels, number of classes)."""
    if task == "two-class":
        X, y = load_420()
        return X, y, 2
    data = load_task("arrhythmia-multiclass")
    return data.X, data.y, len(data.class_names)


def xgb_predict(Xtr, ytr, Xte, **params):
    """Fit XGBoost and predict. Labels are re-indexed, because XGBoost needs 0..m-1 with none missing."""
    classes, codes = np.unique(ytr, return_inverse=True)
    return classes[XGBClassifier(n_jobs=1, random_state=0, **params).fit(Xtr, codes).predict(Xte)]


class XGBFitness(WrapperFitness):
    """The wrapper fitness with a light XGBoost in place of 5-NN."""

    def inner_predict(self, mask):
        yhat = np.empty_like(self.y)
        for tr, va in self.folds:
            yhat[va] = xgb_predict(self.X[tr][:, mask], self.y[tr], self.X[va][:, mask], **LIGHT)
        return yhat


def base_filter_columns(X_raw_tr, kept, Xtr, ytr):
    """The paper's two filters for a many-valued label: fewest zeros, then the largest ANOVA F."""
    zeros = (X_raw_tr[:, kept] == 0).sum(0)
    few_zeros = np.sort(np.argsort(zeros, kind="stable")[:N_ZERO_KEEP])
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.nan_to_num(f_classif(Xtr[:, few_zeros], ytr)[0], nan=0.0)
    return few_zeros[np.sort(np.argsort(-f, kind="stable")[:N_CORR_KEEP])]


def run_fold(task, repeat, fold, tr, te):
    """Every arm on one outer fold. Only the outer-train rows fit anything."""
    warnings.filterwarnings("ignore", message="The least populated class")
    warnings.filterwarnings("ignore", message="Features .* are constant")
    kind, _ = TASKS[task]
    X, y, C = load(task)
    key = FoldKey(task, repeat, fold)
    ytr, yte = y[tr], y[te]
    prep = FoldPreprocessor().fit(X[tr])
    Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
    inner = list(StratifiedKFold(3, shuffle=True, random_state=key.seed()).split(Xtr, ytr))

    if task == "two-class":
        steps = Steps().fit(X[tr], ytr)
        base = steps.transform(X[tr]), steps.transform(X[te])
    else:
        cols = base_filter_columns(X[tr], np.flatnonzero(prep.keep_), Xtr, ytr)
        base = Xtr[:, cols], Xte[:, cols]

    X0 = informed_start(key, Xtr, ytr)
    arms = {"all features": (Xtr, Xte), "base filter": base}
    searched = {}
    for arm, fitness in [("proposed", XGBFitness(Xtr, ytr, C, inner, kind)),
                         ("proposed, 5-NN inside", WrapperFitness(Xtr, ytr, C, inner, kind, k=5))]:
        start = time.perf_counter()
        result = search("GWO-CPO", fitness, X0, BUDGET, np.random.default_rng(key.seed("GWO-CPO", arm)))
        cols = np.flatnonzero(result.mask)
        arms[arm] = Xtr[:, cols], Xte[:, cols]
        searched[arm] = dict(best_fitness=result.fitness, search_seconds=time.perf_counter() - start)

    rows = []

    def add(arm, A_tr, A_te, **params):
        s = class_scores(yte, xgb_predict(A_tr, ytr, A_te, **params), C)
        rows.append(dict(task=task, repeat=repeat, fold=fold, arm=arm, n_features=A_tr.shape[1], acc=s["acc"],
                         bacc=s["bacc"], macro_f1=s["macro_f1"], **searched.get(arm.removesuffix(", tuned"), {})))

    for arm, (A_tr, A_te) in arms.items():
        add(arm, A_tr, A_te)
    if task == "two-class":
        for arm in ("base filter", "proposed"):
            A_tr, A_te = arms[arm]

            def cv_accuracy(u):
                return float(np.mean([(xgb_predict(A_tr[a], ytr[a], A_tr[b], **tuning.settings(u)) == ytr[b]).mean()
                                      for a, b in inner]))

            u, _ = tuning.pso(cv_accuracy, np.random.default_rng(key.seed(arm, "pso")))
            add(f"{arm}, tuned", A_tr, A_te, **tuning.settings(u))
    return rows


def main():
    warnings.filterwarnings("ignore", message="The least populated class")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    jobs = []
    for task in TASKS:
        y = load(task)[1]
        cv = RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=OUTER_SEED)
        jobs += [(task, i // SPLITS, i % SPLITS, tr, te) for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y))]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_fold, *job) for job in jobs]
        rows = pd.DataFrame([r for f in futures for r in f.result()])
    assert not rows.duplicated(["task", "repeat", "fold", "arm"]).any()
    rows.to_csv(OUT / "same_classifier_rows.csv", index=False)

    means = rows.groupby(["task", "arm"], sort=False)[METRICS + ["n_features"]].agg(["mean", "std"])
    means.columns = [f"{m}_{s}" for m, s in means.columns]
    means = means.reset_index()
    means.to_csv(OUT / "same_classifier_means.csv", index=False)

    compared, claims = [], []
    for task, (_, primary) in TASKS.items():
        view = rows[rows.task == task]
        for a, b in COMPARISONS:
            if not {a, b} <= set(view.arm):
                continue
            for metric in METRICS:
                wide = view.pivot_table(index=["repeat", "fold"], columns="arm", values=metric)
                diff = (wide[a] - wide[b]).to_numpy()
                assert len(diff) == REPEATS * SPLITS and not np.isnan(diff).any(), (task, a, b)
                compared.append(dict(task=task, a=a, b=b, metric=metric, primary=metric == primary,
                                     mean_a=wide[a].mean(), mean_b=wide[b].mean(), **_paired(diff)))
        c = next(r for r in compared if (r["task"], r["a"], r["b"], r["metric"]) == (task, "proposed", "base filter", primary))
        features = view[view.arm == "proposed"].n_features.mean()
        beats = bool(c["mean_diff"] > 0 and c["p_exploratory"] < 0.05)
        matches = bool(c["ci_low"] > -MARGIN and features < FEWER_THAN)
        verdict = ("scores higher than the base paper's filter" if beats else
                   f"matches the base paper's filter with {features:.0f} features in place of {N_CORR_KEEP}" if matches else
                   "the base paper's filter does better" if c["ci_high"] < 0 else "no clear difference")
        claims.append(dict(task=task, metric=primary, proposed=c["mean_a"], base_filter=c["mean_b"], mean_diff=c["mean_diff"],
                           ci_low=c["ci_low"], ci_high=c["ci_high"], p=c["p_exploratory"], proposed_features=features,
                           beats=beats, matches_with_fewer_features=matches, verdict=verdict))
    compared, claims = pd.DataFrame(compared), pd.DataFrame(claims)
    compared.to_csv(OUT / "same_classifier_compare.csv", index=False)
    claims.to_csv(OUT / "same_classifier_claims.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(means.round(3).to_string(index=False))
    print(compared[compared.primary].drop(columns=["primary", "n_folds"]).round(4).to_string(index=False))
    print(claims.round(4).to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
