"""More optimisers on the main experiment's Arrhythmia folds, as comparison rows for the table and graphs.

The optimisers of extra_optimizers.py (Amendments 9 and 10) run from the random start of the main run and
from the informed start of Amendment 6, with both fitness kinds. Everything else is the main run's: outer
folds, preprocessing, inner split, population, budget, search seeds and classifiers. No claim is tested.

The five optimisers of arrfs are not scored again. Their rows come from results/full and results/informed_init.
Their informed-start searches are rerun only to record the convergence curves, which the Amendment 6 run did
not save, and the rerun must reproduce that run's subset sizes and fitness values exactly.

Written to results/more_optimizers/:
    more_optimizers_rows.csv     the added optimisers, one row per fold, start, fitness and classifier
    all_optimizers_rows.csv      the same rows together with the five optimisers of results/full and results/informed_init
    all_optimizers_summary.csv   mean and sd over the 25 folds for every optimiser, start, fitness and classifier
    all_optimizers_curves.csv    mean best-so-far fitness of every optimiser at 100 points of the budget

Design: design/amendment-9-more-optimisers.md and design/amendment-10-wider-group.md.
Run from code/:  .venv/bin/python more_optimizers_demo.py     then     .venv/bin/python plot_optimizers.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import extra_optimizers  # noqa: E402
from arrfs.experiment import _CLASSIFIERS, FITNESS_KINDS, fold_keys, load_run  # noqa: E402
from arrfs.fitness import WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import ALGORITHMS, CURVE_POINTS  # noqa: E402
from informed_init_demo import SETTINGS, TASKS, fold_context, informed_start, select  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "more_optimizers"
MAIN = HERE / "results" / "full"
INFORMED_ROWS = HERE / "results" / "informed_init" / "informed_init_rows.csv"
SEARCH = ["task", "repeat", "fold", "method", "fitness"]
KEYS = SEARCH + ["classifier", "start"]
METRICS = ["acc", "bacc", "macro_f1", "n_features"]
CURVE = ["task", "method", "fitness", "start"]


def run_fold(key):
    """One outer fold: (score rows of the added optimisers, curves, reruns of the five informed-start searches)."""
    warnings.filterwarnings("ignore", message="The least populated class")
    Xtr, ytr, Xte, yte, inner, _, C = fold_context(key)
    starts = {"random": np.random.default_rng(key.seed()).random((SETTINGS.population, Xtr.shape[1])),
              "informed": informed_start(key, Xtr, ytr)}
    rows, curves, reruns = [], [], []
    for start, X0 in starts.items():
        for kind in FITNESS_KINDS:
            for name in extra_optimizers.ALGORITHMS:
                fitness = WrapperFitness(Xtr, ytr, C, inner, kind, k=SETTINGS.neighbours)
                res = extra_optimizers.search(name, fitness, X0, SETTINGS.budget,
                                              np.random.default_rng(key.seed(name, kind)))
                cols = np.flatnonzero(res.mask)
                where = dict(task=key.task, repeat=key.repeat, fold=key.fold, method=name, fitness=kind, start=start)
                curves.append(dict(**where, curve=res.curve))
                for clf, make in _CLASSIFIERS.items():
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):  # spurious Accelerate flags
                        yhat = make(key.seed(), SETTINGS.neighbours).fit(Xtr[:, cols], ytr).predict(Xte[:, cols])
                    s = class_scores(yte, yhat, C)
                    rows.append(dict(**where, classifier=clf, n_features=len(cols), best_fitness=res.fitness,
                                     acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    for kind in FITNESS_KINDS:
        for name in ALGORITHMS:
            res = select(key, Xtr, ytr, inner, C, starts["informed"], name, kind)
            where = dict(task=key.task, repeat=key.repeat, fold=key.fold, method=name, fitness=kind, start="informed")
            curves.append(dict(**where, curve=res.curve))
            reruns.append(dict(**where, n_features=int(res.mask.sum()), best_fitness=res.fitness))
    return rows, curves, reruns


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        results = list(pool.map(run_fold, fold_keys(SETTINGS)))
    new = pd.DataFrame([r for rows, _, _ in results for r in rows])
    n_folds = len(fold_keys(SETTINGS))
    assert len(new) == n_folds * 2 * len(FITNESS_KINDS) * len(extra_optimizers.ALGORITHMS) * len(_CLASSIFIERS)
    new.to_csv(OUT / "more_optimizers_rows.csv", index=False)

    informed = pd.read_csv(INFORMED_ROWS, float_precision="round_trip")
    saved = informed[informed.classifier == "KNN"].set_index(SEARCH)[["n_features", "best_fitness"]].sort_index()
    reruns = pd.DataFrame([r for _, _, fold_reruns in results for r in fold_reruns])
    pd.testing.assert_frame_equal(reruns.set_index(SEARCH)[["n_features", "best_fitness"]].sort_index(), saved,
                                  check_exact=True)
    print(f"the {len(reruns)} rerun informed-start searches reproduce results/informed_init exactly", flush=True)

    _, main_folds = load_run(MAIN)
    curves = pd.DataFrame(
        [c for _, fold_curves, _ in results for c in fold_curves]
        + [dict(task=r.key.task, repeat=r.key.repeat, fold=r.key.fold, method=m.method, fitness=m.fitness,
                start="random", curve=m.curve)
           for r in main_folds if r.key.task in TASKS for m in r.methods if m.method in ALGORITHMS])
    assert not curves.duplicated(SEARCH + ["start"]).any()
    assert (curves.groupby(CURVE).size() == n_folds // len(TASKS)).all()
    mean_curves = pd.DataFrame([dict(zip(CURVE, cell), point=i + 1, best_fitness=value)
                                for cell, g in curves.groupby(CURVE, sort=False)
                                for i, value in enumerate(np.mean(np.stack(g.curve.to_list()), axis=0))])
    n_optimisers = len(ALGORITHMS) + len(extra_optimizers.ALGORITHMS)
    assert len(mean_curves) == len(TASKS) * n_optimisers * len(FITNESS_KINDS) * 2 * CURVE_POINTS
    mean_curves.to_csv(OUT / "all_optimizers_curves.csv", index=False)

    old = pd.concat([pd.read_csv(MAIN / "summary" / "rows.csv", float_precision="round_trip").assign(start="random"),
                     informed], ignore_index=True)
    old = old[old.method.isin(ALGORITHMS) & old.task.isin(TASKS)]
    everything = pd.concat([old[KEYS + METRICS], new[KEYS + METRICS]], ignore_index=True)
    assert not everything.duplicated(KEYS).any()
    assert (everything.groupby(["task", "method", "fitness", "classifier", "start"]).size() == n_folds // len(TASKS)).all()
    everything.to_csv(OUT / "all_optimizers_rows.csv", index=False)

    summary = everything.groupby(["task", "start", "fitness", "classifier", "method"], sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "all_optimizers_summary.csv", index=False)

    pd.set_option("display.width", 220)
    knn = summary[summary.classifier == "KNN"]
    for (task, kind), metric in [(("arrhythmia-multiclass", "imbalance"), "macro_f1_mean"), (("arrhythmia-binary", "accuracy"), "acc_mean")]:
        view = knn[(knn.task == task) & (knn.fitness == kind)].pivot(index="method", columns="start", values=[metric, "n_features_mean"])
        print(f"\n{task}, {kind} fitness, 5-NN: {metric} and features by start")
        print(view.round(3).to_string())
        for s in ("random", "informed"):
            col = view[(metric, s)]
            print(f"   {s} start: best {col.idxmax()} {col.max():.3f}, worst {col.idxmin()} {col.min():.3f}, spread {col.max() - col.min():.3f}")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
