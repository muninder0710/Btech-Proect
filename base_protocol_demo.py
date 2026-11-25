"""GWO-CPO against IBGWO4 under the base paper's own protocol.

Khaseeb et al. 2025 (pp. 12 to 14) use a random 80/20 split, 5-NN, the fitness 0.99 x error + 0.01 x |S| / D,
a population of 10, 100 iterations, and test accuracy averaged over 30 runs, with one initial population shared
by every method. The paper does not say which rows the fitness is scored on, so both readings run.
Paper-style scores the fitness on the 20% rows that are reported. Clean scores it inside the 80% rows.

Also written: GWO-CPO against IBGWO4 with the same accuracy fitness on the main experiment's 25 folds,
read from results/full with no new search.

Design: design/amendment-5-base-protocol.md.
Run from code/:  .venv/bin/python base_protocol_demo.py
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
from sklearn.model_selection import ShuffleSplit, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from analyze import _paired, holm  # noqa: E402
from arrfs.data import FoldPreprocessor, load_task  # noqa: E402
from arrfs.fitness import WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import search  # noqa: E402
from test_reuse_demo import HoldoutFitness, seed_of  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "base_protocol"
MAIN_ROWS = HERE / "results" / "full" / "summary" / "rows.csv"
TASKS = ["arrhythmia-multiclass", "arrhythmia-binary"]
ALGORITHMS = ["IBGWO4", "GWO-CPO"]
READINGS = ["paper", "clean"]
METRICS = ["acc", "bacc", "macro_f1"]
N_RUNS, TEST_SIZE, POPULATION, K = 30, 0.2, 10, 5
BUDGET = 2000  # P + 2PT = 2,010 for T = 100; search() needs a multiple of 100


def make_fitness(reading, Xtr, ytr, Xte, yte, C, inner_seed):
    """The search objective. In the clean reading the 20% rows never reach it."""
    if reading == "paper":
        return HoldoutFitness(Xtr, ytr, Xte, yte, C, k=K)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=inner_seed).split(Xtr, ytr))
    fitness = WrapperFitness(Xtr, ytr, C, inner, "accuracy", k=K)
    assert fitness.X.shape[0] == len(ytr), "clean fitness holds non-train rows"
    return fitness


def run_split(task, run, tr, te):
    """Both methods under both readings on one random 80/20 split, from one shared initial population."""
    warnings.filterwarnings("ignore", message="The least populated class")  # rare classes, inner 3-fold
    data = load_task(task)
    C = len(data.class_names)
    prep = FoldPreprocessor().fit(data.X[tr])
    Xtr, Xte = prep.transform(data.X[tr]), prep.transform(data.X[te])
    ytr, yte = data.y[tr], data.y[te]
    X0 = np.random.default_rng(seed_of(task, run, "X0")).random((POPULATION, Xtr.shape[1]))
    rows = []
    for reading in READINGS:
        for algo in ALGORITHMS:
            fitness = make_fitness(reading, Xtr, ytr, Xte, yte, C, seed_of(task, run, "inner"))
            result = search(algo, fitness, X0, BUDGET, np.random.default_rng(seed_of(task, run, reading, algo)))
            cols = np.flatnonzero(result.mask)
            yhat = KNeighborsClassifier(K).fit(Xtr[:, cols], ytr).predict(Xte[:, cols])
            s = class_scores(yte, yhat, C)
            rows.append(dict(task=task, run=run, reading=reading, algorithm=algo, n_features=len(cols),
                             best_fitness=result.fitness, acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    return rows


def gwo_cpo_minus_ibgwo4(rows, blocks, pair_keys, method_col):
    """Paired GWO-CPO minus IBGWO4 per block and metric, with Holm over the blocks of each metric."""
    out = []
    for keys, d in rows.groupby(blocks, sort=False):
        for m in METRICS:
            wide = d.pivot_table(index=pair_keys, columns=method_col, values=m)
            diff = (wide["GWO-CPO"] - wide["IBGWO4"]).to_numpy()
            assert not np.isnan(diff).any(), (keys, m)
            out.append(dict(zip(blocks, keys), metric=m, **_paired(diff)))
    table = pd.DataFrame(out).rename(columns={"n_folds": "n_pairs"})
    for m in METRICS:
        sel = table.metric == m
        table.loc[sel, "p_holm"] = holm(table.loc[sel, "p_exploratory"].to_numpy())
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    jobs = []
    for task in TASKS:
        n = len(load_task(task).y)
        splits = ShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=42).split(np.zeros(n))
        jobs += [(task, run, tr, te) for run, (tr, te) in enumerate(splits)]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_split, *job) for job in jobs]
        runs = pd.DataFrame([r for f in futures for r in f.result()])
    assert len(runs) == len(TASKS) * N_RUNS * len(READINGS) * len(ALGORITHMS)
    assert not runs.duplicated(["task", "run", "reading", "algorithm"]).any()
    runs.to_csv(OUT / "base_protocol_runs.csv", index=False)

    arms = runs.groupby(["task", "reading", "algorithm"], sort=False)[METRICS + ["n_features"]].agg(["mean", "std"])
    arms.columns = [f"{m}_{s}" for m, s in arms.columns]
    arms = arms.reset_index()
    arms.to_csv(OUT / "base_protocol_arms.csv", index=False)

    diffs = gwo_cpo_minus_ibgwo4(runs, ["task", "reading"], ["run"], "algorithm")
    diffs["share_runs_gwo_cpo_higher"] = diffs.wins / diffs.n_pairs
    holds = (diffs.mean_diff > 0) & (diffs.p_exploratory < 0.05)
    diffs["verdict"] = np.where(diffs.metric != "acc", "secondary (no claim)",
                                np.where(holds, "holds", "does not hold"))
    diffs.to_csv(OUT / "base_protocol_summary.csv", index=False)

    main_rows = pd.read_csv(MAIN_ROWS)
    main_rows = main_rows[(main_rows.classifier == "KNN") & (main_rows.fitness == "accuracy")
                          & main_rows.method.isin(ALGORITHMS) & main_rows.task.isin(TASKS)]
    same_fitness = gwo_cpo_minus_ibgwo4(main_rows, ["task"], ["repeat", "fold"], "method")
    assert (same_fitness.n_pairs == 25).all(), same_fitness.n_pairs
    same_fitness.to_csv(OUT / "main_protocol_same_fitness.csv", index=False)

    pd.set_option("display.width", 220)
    print(arms.round(3).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    print(same_fitness.round(4).to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
