"""Protocol checks for the experiment (failure modes 6 to 12 in design/rubric.md).

    python check_protocol.py        # about 30 seconds; exits non-zero on the first failure

Runs a tiny grid (budget 200, one repeat, two algorithms, one Arrhythmia task
and one small dataset) through the real runner in a temporary directory, then
checks leakage, resume, the results table, and the two pieces of
arrfs/fitness.py the whole study rests on.
"""
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from analyze import load_methods
from arrfs.data import TASKS, FoldPreprocessor, load_task
from arrfs.experiment import (FITNESS_KINDS, NO_SEARCH, PROPOSED, REFERENCES, FoldKey, Settings,
                              _outer_split, fold_keys, run_fold)
from arrfs.fitness import WrapperFitness, class_scores
from run_experiments import run

TINY = Settings(repeats=1, budget=200, algorithms=("IBGWO4", PROPOSED[0]),
                tasks=("arrhythmia-multiclass", "lymphography"))


def check_class_scores() -> str:
    """class_scores against scikit-learn on random label vectors, including
    classes never predicted and predicted classes absent from y_true."""
    rng = np.random.default_rng(0)
    extra_class_cases = 0
    warnings.filterwarnings("ignore", module="sklearn.metrics")  # these vectors are degenerate on purpose
    for trial in range(300):
        C = int(rng.choice([2, 5, 12]))
        n = int(rng.choice([5, 20, 90]))
        y = rng.choice(rng.choice(C, size=max(1, C - int(rng.integers(0, 3))), replace=False), n)
        yhat = rng.integers(0, C, n) if trial % 2 else rng.choice(np.unique(y), n)
        present = np.unique(y)
        s = class_scores(y, yhat, C)
        assert np.isclose(s["acc"], accuracy_score(y, yhat)), trial
        assert np.isclose(s["bacc"], balanced_accuracy_score(y, yhat)), trial
        assert np.isclose(s["macro_f1"], f1_score(y, yhat, labels=present, average="macro", zero_division=0)), trial
        assert np.allclose(s["recall"][present], recall_score(y, yhat, labels=present, average=None, zero_division=0)), trial
        extra_class_cases += not np.isin(yhat, present).all()
    return (f"class_scores matches scikit-learn on 300 random label vectors "
            f"({extra_class_cases} of them predict a class absent from y_true; macro-F1 averages over the classes in y_true)")


def check_knn() -> str:
    """The numpy 5-NN inside WrapperFitness against KNeighborsClassifier on random subsets of every task.

    When the 5th and 6th nearest neighbours are equally far, which neighbour
    votes is arbitrary in both implementations. With a handful of mostly
    discrete attributes such ties are common, so the contract is: identical
    predictions wherever the 5 nearest neighbours are uniquely defined.
    """
    rng = np.random.default_rng(1)
    subsets = predictions = tied = tied_mismatches = 0
    for task in TASKS:
        data = load_task(task)
        X = FoldPreprocessor().fit_transform(data.X)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, data.y))
        fitness = WrapperFitness(X, data.y, len(data.class_names), folds)
        for density in (0.02, 0.1, 0.5, 0.9):
            for _ in range(10):
                mask = rng.random(X.shape[1]) < density
                mask[rng.integers(X.shape[1])] = True
                ours = fitness.inner_predict(mask)
                subsets += 1
                for tr, va in folds:
                    knn = KNeighborsClassifier(5).fit(X[tr][:, mask], data.y[tr])
                    distance, _ = knn.kneighbors(X[va][:, mask], 6)
                    tie = np.isclose(distance[:, 4], distance[:, 5])
                    differ = ours[va] != knn.predict(X[va][:, mask])
                    assert not (differ & ~tie).any(), f"{task}: KNN differs from scikit-learn without a distance tie"
                    predictions += len(va)
                    tied += int(tie.sum())
                    tied_mismatches += int(differ.sum())
    return (f"numpy 5-NN matches scikit-learn on every untied record over {subsets} random subsets of the "
            f"{len(TASKS)} tasks: {predictions - tied} untied predictions, 0 mismatches; {tied} records have a "
            f"5th/6th-neighbour distance tie and {tied_mismatches} of those break it differently")


def check_smoke_is_repeat_zero() -> str:
    for task in TASKS:
        y = load_task(task).y
        for fold in range(5):
            key = FoldKey(task, 0, fold)
            assert np.array_equal(_outer_split(y, key, Settings(repeats=1))[1], _outer_split(y, key, Settings())[1])
    return f"the smoke run's outer folds are the full run's repeat 0, for all {len(TASKS)} tasks"


def check_table(out: Path) -> str:
    """Every key once, every record predicted once per repeat, ANOVA-k uses the proposed row's k."""
    m = load_methods(out)
    key = ["task", "repeat", "fold", "method", "fitness"]
    per_fold = len(TINY.algorithms) * len(FITNESS_KINDS) + len(REFERENCES)
    assert not m.duplicated(key).any(), "a (task, repeat, fold, method, fitness) key repeats"
    assert len(m) == len(fold_keys(TINY)) * per_fold, f"{len(m)} rows"
    records = {task: len(load_task(task).y) for task in TINY.tasks}
    for (task, repeat, lab), g in m.groupby(["task", "repeat", "label"]):
        assert np.array_equal(np.sort(np.concatenate(g.test_idx.tolist())), np.arange(records[task])), (task, lab)
        assert all(len(p) == len(t) for preds, t in zip(g.y_pred, g.test_idx) for p in preds.values()), (task, lab)
    search = m[m.fitness != NO_SEARCH]
    assert (search.calls == TINY.budget).all() and all(len(c) == 100 for c in search.curve)
    for _, g in m.groupby(["task", "repeat", "fold"]):
        k = g[(g.method == PROPOSED[0]) & (g.fitness == PROPOSED[1])].n_features.item()
        assert g[g.method == "ANOVA-k"].n_features.item() == k, "ANOVA-k used a different k"
    return (f"{len(m)} rows over {', '.join(TINY.tasks)}: every key once, every record predicted once per "
            f"repeat and classifier, every search spent {TINY.budget} calls, ANOVA-k matches the proposed "
            f"row's k in every fold")


def _without_seconds(path: Path) -> dict:
    d = json.loads(path.read_text())
    for m in d["methods"]:
        del m["seconds"]
    return d


def check_resume(out: Path, workers: int) -> str:
    """Deleting one fold file recomputes only that fold, with identical results;
    a run with nothing missing computes nothing."""
    first = {p.name: _without_seconds(p) for p in (out / "folds").glob("*.json")}
    victim = FoldKey(TINY.tasks[-1], 0, 3)
    (out / "folds" / f"{victim.name}.json").unlink()
    computed, failed = run(TINY, out, workers)
    assert computed == [victim] and not failed, f"recomputed {computed}"
    second = {p.name: _without_seconds(p) for p in (out / "folds").glob("*.json")}
    assert second == first, "the recomputed fold differs from the first run"
    computed, failed = run(TINY, out, workers)
    assert computed == [] and not failed, f"a complete run recomputed {computed}"
    return f"resume recomputed only {victim.name}, its results match the first run exactly, and a complete run computes nothing"


def check_no_test_leakage(task: str) -> str:
    """Noise in the outer-test rows changes predictions but no selected subset."""
    key = FoldKey(task, 0, 1)
    clean = run_fold(key, TINY)
    X = load_task(task).X.copy()
    X[clean.test_idx] = np.random.default_rng(2).normal(0, 1000, (len(clean.test_idx), X.shape[1]))
    noisy = run_fold(key, TINY, X)
    changed_predictions = 0
    for a, b in zip(clean.methods, noisy.methods):
        assert np.array_equal(a.selected, b.selected), f"{a.method}/{a.fitness} selected a different subset"
        if a.curve is not None:
            assert a.best_fitness == b.best_fitness and np.array_equal(a.curve, b.curve), f"{a.method}/{a.fitness}"
        changed_predictions += sum(int((a.y_pred[c] != b.y_pred[c]).sum()) for c in a.y_pred)
    assert changed_predictions > 0, "the noise never reached the outer-test rows"
    return (f"noise in {key.name}'s {len(clean.test_idx)} outer-test rows leaves all {len(clean.methods)} "
            f"subsets unchanged and changes {changed_predictions} test predictions")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The least populated class")  # expected for the rare classes; see README
    print(check_class_scores())
    print(check_knn())
    print(check_smoke_is_repeat_zero())
    workers = os.cpu_count()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        computed, failed = run(TINY, out, workers)
        assert len(computed) == len(fold_keys(TINY)) and not failed
        print(check_table(out))
        print(check_resume(out, workers))
    for task in TINY.tasks:
        print(check_no_test_leakage(task))
    print("all protocol checks passed")
