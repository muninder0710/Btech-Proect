"""Experiment 3(b): what happens when the subset is chosen on the same rows the score is reported on?

Paper-style arm: the search fitness trains 5-NN on the 70% rows and scores the 30% rows, and the
reported number is the chosen subset's score on those same 30% rows.
Clean arm: the fitness sees only the 70% rows (inner stratified 3-fold CV), and the chosen subset
is scored once on the 30% rows.
Random-label control: both arms again with the labels permuted once per split.
Both arms are reported with scikit-learn's KNeighborsClassifier(5) so the reporting code is identical.

Design: design/amendment-3-contributions.md, section "Experiment for contribution 3(b)".
Run from code/:  .venv/bin/python test_reuse_demo.py
"""
import os

# One BLAS thread per process, set before numpy loads, as in run_experiments.py.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import hashlib  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from arrfs.data import FoldPreprocessor, load_task  # noqa: E402
from arrfs.fitness import EMPTY_PENALTY, WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import search  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "test_reuse"
TASKS = ["arrhythmia-multiclass", "arrhythmia-binary"]
ALGORITHMS = ["BCPO", "GWO-CPO"]
ARMS = ["paper", "clean"]
LABELS = ["true", "permuted"]
METRICS = ["acc", "bacc", "macro_f1"]
N_SPLITS, POPULATION, BUDGET, K = 25, 20, 2000, 5
N_BOOT, BOOT_SEED, PERMUTE_SEED = 2000, 20260, 7


def seed_of(*parts) -> int:
    """Stable 32-bit seed from sha256 (Python's hash() of a string differs between processes)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


class HoldoutFitness(WrapperFitness):
    """The paper-style fitness: 5-NN trained on the first n_train rows, scored on the rest.

    WrapperFitness scores every row of its y against the predictions, and a single (train, test)
    fold leaves the train rows' predictions unfilled. So the score here uses the test rows only.
    """

    def __init__(self, Xtr, ytr, Xte, yte, n_classes, k=K):
        n = len(ytr)
        super().__init__(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), n_classes,
                         [(np.arange(n), np.arange(n, n + len(yte)))], "accuracy", k)
        self.n_train = n

    def __call__(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.calls += 1
        key = mask.tobytes()
        if key not in self.cache:
            n_sel = int(mask.sum())
            if n_sel == 0:
                f = EMPTY_PENALTY
            else:
                acc = (self.inner_predict(mask)[self.n_train:] == self.y[self.n_train:]).mean()
                f = 0.99 * (1 - acc) + 0.01 * n_sel / self.D
            self.cache[key] = f
        return self.cache[key]


def choose_subset(arm, Xtr, ytr, Xte, yte, C, seed_parts):
    """The mask the search picks. In the clean arm the test rows never reach the fitness."""
    X0 = np.random.default_rng(seed_of(*seed_parts, "X0")).random((POPULATION, Xtr.shape[1]))
    rng = np.random.default_rng(seed_of(*seed_parts, "search"))
    if arm == "paper":
        fitness = HoldoutFitness(Xtr, ytr, Xte, yte, C)
    else:
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed_of(*seed_parts, "inner")).split(Xtr, ytr))
        fitness = WrapperFitness(Xtr, ytr, C, inner, "accuracy", k=K)
        assert fitness.X.shape[0] == len(ytr) and fitness.y.shape[0] == len(ytr), "clean fitness holds non-train rows"
        assert np.array_equal(np.sort(np.concatenate([va for _, va in fitness.folds])), np.arange(len(ytr)))
    return search(seed_parts[2], fitness, X0, BUDGET, rng).mask



def run_split(task, split, tr, te):
    """All 8 searches (2 algorithms x 2 arms x 2 label kinds) of one (task, split)."""
    warnings.filterwarnings("ignore", message="The least populated class")  # rare classes, inner 3-fold
    data = load_task(task)
    C = len(data.class_names)
    permuted = np.random.default_rng(seed_of(task, split, "permute", PERMUTE_SEED)).permutation(data.y)
    prep = FoldPreprocessor().fit(data.X[tr])
    Xtr, Xte = prep.transform(data.X[tr]), prep.transform(data.X[te])
    rows = []
    for labels, y in [("true", data.y), ("permuted", permuted)]:
        ytr, yte = y[tr], y[te]
        for algo in ALGORITHMS:
            for arm in ARMS:
                mask = choose_subset(arm, Xtr, ytr, Xte, yte, C, (task, split, algo, arm, labels))
                cols = np.flatnonzero(mask)
                yhat = KNeighborsClassifier(K).fit(Xtr[:, cols], ytr).predict(Xte[:, cols])
                s = class_scores(yte, yhat, C)
                rows.append(dict(task=task, split=split, labels=labels, algorithm=algo, arm=arm,
                                 n_features=len(cols), acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    return rows


def bootstrap_ci(diff):
    """95% percentile bootstrap interval of the mean of paired differences."""
    rng = np.random.default_rng(BOOT_SEED)
    means = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def summarize(folds):
    keys = ["task", "labels", "arm", "algorithm"]
    g = folds.groupby(keys, sort=False)[METRICS].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    arms = g.reset_index()

    wide = folds.pivot_table(index=["task", "labels", "algorithm", "split"], columns="arm", values=METRICS)
    diffs = []
    for (task, labels, algo), d in wide.groupby(level=["task", "labels", "algorithm"], sort=False):
        for m in METRICS:
            diff = (d[(m, "paper")] - d[(m, "clean")]).to_numpy()
            assert len(diff) == N_SPLITS
            lo, hi = bootstrap_ci(diff)
            diffs.append(dict(task=task, labels=labels, algorithm=algo, metric=m, mean_diff=diff.mean(),
                              ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    return arms, pd.DataFrame(diffs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    jobs = []
    for task in TASKS:
        y = load_task(task).y
        splits = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.3, random_state=42).split(np.zeros(len(y)), y)
        jobs += [(task, i, tr, te) for i, (tr, te) in enumerate(splits)]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_split, *job) for job in jobs]
        rows = [r for f in futures for r in f.result()]
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "test_reuse_splits.csv", index=False)
    arms, diffs = summarize(folds)
    arms.to_csv(OUT / "test_reuse_arms.csv", index=False)

    chance = {t: 1 / len(load_task(t).class_names) for t in TASKS}
    clean_perm = arms[(arms.labels == "permuted") & (arms.arm == "clean")].copy()
    clean_perm["chance"] = clean_perm.task.map(chance)
    clean_perm["gap"] = clean_perm.bacc_mean - clean_perm.chance
    clean_perm.to_csv(OUT / "test_reuse_chance_check.csv", index=False)
    assert (clean_perm.gap.abs() <= 0.05).all(), f"clean arm not within 0.05 of chance:\n{clean_perm}"

    true = diffs.labels == "true"
    diffs["verdict"] = np.where(~true, "control (no claim)", np.where(~diffs.excludes_zero, "not supported",
                                np.where(diffs.mean_diff > 0, "inflates", "deflates")))
    diffs.to_csv(OUT / "test_reuse_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(arms.round(3).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    print(clean_perm[["task", "algorithm", "bacc_mean", "chance", "gap"]].round(3).to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
