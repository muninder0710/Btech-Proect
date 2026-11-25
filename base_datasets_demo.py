"""The base paper's method and the proposed model on the base paper's own nine datasets, under its own protocol.

Khaseeb et al. 2025 test IBGWO4 on nine cancer gene-expression datasets (their Table 1) with a random 80/20
split, 5-NN, a population of 10, 100 iterations and 30 runs. This script runs four arms on those datasets with
that protocol (the constants of base_protocol_demo.py), in the two readings of Amendment 5: `paper` scores the
fitness on the 20% rows that are reported, and `clean` scores it inside the 80% rows.

    base as published       IBGWO4,  accuracy fitness,  random start
    hybrid alone            GWO-CPO, accuracy fitness,  random start
    base with smart start   IBGWO4,  accuracy fitness,  informed start of Amendment 6
    proposed                GWO-CPO, imbalance fitness, informed start

Written to results/base_datasets/:
    jobs/<dataset>_r<run>.csv    one finished dataset and run, the unit of resuming
    base_datasets_runs.csv       every row
    base_datasets_arms.csv       mean and sd per dataset, reading and arm
    base_datasets_compare.csv    the three comparisons per dataset and reading, paired over the runs
    base_datasets_across.csv     the same comparisons across the datasets, with their verdicts

Design: design/amendment-12-base-datasets.md.
Run from code/:  .venv/bin/python fetch_gems.py     then     .venv/bin/python base_datasets_demo.py
A stopped run can be started again and keeps its finished jobs. `base_datasets_demo.py summary` only rebuilds
the four summary files from the jobs that are there.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.model_selection import ShuffleSplit, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from analyze import _paired  # noqa: E402
from arrfs.data import FoldPreprocessor  # noqa: E402
from arrfs.experiment import FoldKey  # noqa: E402
from arrfs.fitness import EMPTY_PENALTY, WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import search  # noqa: E402
from base_protocol_demo import BUDGET, K, N_RUNS, POPULATION, READINGS, TEST_SIZE  # noqa: E402
from informed_init_demo import informed_start  # noqa: E402
from test_reuse_demo import HoldoutFitness, seed_of  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "gems"
OUT = HERE / "results" / "base_datasets"
DATASETS = {  # name in Khaseeb et al. 2025, Table 1: (instances, genes, classes)
    "SRBCT": (83, 2308, 4),
    "Brain_Tumor2": (50, 10367, 4),
    "Leukemia1": (72, 5327, 3),
    "DLBCL": (77, 5469, 2),
    "Brain_Tumor1": (90, 5920, 5),
    "Leukemia2": (72, 11225, 3),
    "Prostate_Tumor": (102, 10509, 2),
    "11_Tumors": (174, 12533, 11),
    "14_Tumors": (308, 15009, 26),
}
ARMS = {  # arm: (algorithm, fitness kind, start)
    "base as published": ("IBGWO4", "accuracy", "random"),
    "hybrid alone": ("GWO-CPO", "accuracy", "random"),
    "base with smart start": ("IBGWO4", "accuracy", "informed"),
    "proposed": ("GWO-CPO", "imbalance", "informed"),
}
BASE = "base as published"
COMPARISONS = {  # comparison: (arm compared with BASE, its primary metric)
    "hybrid alone": ("hybrid alone", "acc"),
    "start alone": ("base with smart start", "acc"),
    "proposed model": ("proposed", "macro_f1"),
}
METRICS = ["acc", "macro_f1", "bacc", "n_features"]
ALPHA = 0.05


class HoldoutImbalance(HoldoutFitness):
    """HoldoutFitness with the imbalance-aware score of the proposed model, on the same reported rows."""

    def __call__(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.calls += 1
        key = mask.tobytes()
        if key not in self.cache:
            n_sel = int(mask.sum())
            if n_sel == 0:
                self.cache[key] = EMPTY_PENALTY
            else:
                s = class_scores(self.y[self.n_train:], self.inner_predict(mask)[self.n_train:], self.C)
                self.cache[key] = (self.alpha * (1 - s["macro_f1"]) + self.beta * (1 - s["bacc"])
                                   + self.gamma * n_sel / self.D)
        return self.cache[key]


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(expression values float[n, genes], class index int[n], class names) as fetch_gems.py stored them."""
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as stored:
        X, y, classes = stored["X"].astype(float), stored["y"], stored["classes"].tolist()
    assert (len(y), X.shape[1], len(classes)) == DATASETS[name], (name, X.shape, len(classes))
    return X, y, classes


def make_fitness(reading, kind, Xtr, ytr, Xte, yte, C, inner):
    """The search objective. In the clean reading the 20% rows never reach it."""
    if reading == "paper":
        return (HoldoutFitness if kind == "accuracy" else HoldoutImbalance)(Xtr, ytr, Xte, yte, C, k=K)
    fitness = WrapperFitness(Xtr, ytr, C, inner, kind, k=K)
    assert fitness.X.shape[0] == len(ytr), "clean fitness holds non-train rows"
    return fitness


def run_job(job):
    """Every arm under both readings on one random 80/20 split of one dataset: (dataset, run, rows)."""
    name, run, tr, te = job
    warnings.filterwarnings("ignore", message="The least populated class")  # rare classes, inner 3-fold
    X, y, classes = load_dataset(name)
    C = len(classes)
    prep = FoldPreprocessor().fit(X[tr])
    Xtr, Xte, ytr, yte = prep.transform(X[tr]), prep.transform(X[te]), y[tr], y[te]
    starts = {"random": np.random.default_rng(seed_of(name, run, "X0")).random((POPULATION, Xtr.shape[1])),
              "informed": informed_start(FoldKey(name, run, 0), Xtr, ytr)[:POPULATION]}
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed_of(name, run, "inner")).split(Xtr, ytr))
    rows = []
    for reading in READINGS:
        for arm, (algorithm, kind, start) in ARMS.items():
            fitness = make_fitness(reading, kind, Xtr, ytr, Xte, yte, C, inner)
            # the search seeds of base_protocol_demo.py for the arms it also runs; the start never enters a seed
            search_seed = seed_of(name, run, reading, algorithm, *([] if kind == "accuracy" else [kind]))
            result = search(algorithm, fitness, starts[start], BUDGET, np.random.default_rng(search_seed))
            cols = np.flatnonzero(result.mask)
            assert len(cols), (name, run, reading, arm)
            yhat = KNeighborsClassifier(K).fit(Xtr[:, cols], ytr).predict(Xte[:, cols])
            s = class_scores(yte, yhat, C)
            rows.append(dict(dataset=name, run=run, reading=reading, arm=arm, algorithm=algorithm, fitness=kind,
                             start=start, start_size=(starts[start] > 0.5).sum(1).mean(), n_features=len(cols),
                             best_fitness=result.fitness, acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    return name, run, rows


def jobs_to_do() -> list[tuple]:
    """Unfinished jobs, run by run, so that a stopped run leaves the same number of runs for every dataset."""
    splits = {name: list(ShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=42).split(np.zeros(n)))
              for name, (n, _, _) in DATASETS.items()}
    return [(name, run, *splits[name][run]) for run in range(N_RUNS) for name in DATASETS
            if not (OUT / "jobs" / f"{name}_r{run}.csv").exists()]


def summarise() -> None:
    runs = pd.concat([pd.read_csv(path, float_precision="round_trip") for path in sorted((OUT / "jobs").glob("*.csv"))],
                     ignore_index=True)
    assert not runs.duplicated(["dataset", "run", "reading", "arm"]).any()
    done = runs.groupby("dataset").run.nunique().reindex(list(DATASETS), fill_value=0)
    complete = bool((done == N_RUNS).all())
    runs.to_csv(OUT / "base_datasets_runs.csv", index=False)

    arms = runs.groupby(["dataset", "reading", "arm"], sort=False)[METRICS + ["best_fitness", "start_size"]].agg(["mean", "std"])
    arms.columns = [f"{m}_{s}" for m, s in arms.columns]
    arms.reset_index().to_csv(OUT / "base_datasets_arms.csv", index=False)

    compare, across = [], []
    for reading in READINGS:
        for comparison, (arm, primary) in COMPARISONS.items():
            for metric in METRICS:
                means = []
                for name in done.index[done > 0]:
                    block = runs[(runs.dataset == name) & (runs.reading == reading)]
                    wide = block.pivot(index="run", columns="arm", values=metric)
                    diff = (wide[arm] - wide[BASE]).to_numpy()
                    means.append(diff.mean())
                    compare.append(dict(reading=reading, comparison=comparison, metric=metric, dataset=name,
                                        mean_arm=wide[arm].mean(), mean_base=wide[BASE].mean(), **_paired(diff)))
                means = np.array(means)
                with warnings.catch_warnings():  # scipy falls back to an approximation when a difference is exactly 0
                    warnings.filterwarnings("ignore", message="Sample size too small")
                    p = wilcoxon(means).pvalue if means.any() else 1.0
                verdict = "no clear difference" if p >= ALPHA else "higher" if np.median(means) > 0 else "lower"
                across.append(dict(reading=reading, comparison=comparison, metric=metric, primary=metric == primary,
                                   datasets=len(means), higher_in=int((means > 0).sum()), lower_in=int((means < 0).sum()),
                                   mean_diff=means.mean(), median_diff=np.median(means), p=p, verdict=verdict))
    compare = pd.DataFrame(compare).rename(columns={"n_folds": "n_runs"})
    across = pd.DataFrame(across)
    compare.to_csv(OUT / "base_datasets_compare.csv", index=False)
    across.to_csv(OUT / "base_datasets_across.csv", index=False)

    pd.set_option("display.width", 230)
    print("runs per dataset:", done.to_dict(), "" if complete else "  (NOT COMPLETE: a partial result)")
    view = runs.pivot_table(index="dataset", columns=["reading", "arm"], values="acc", sort=False).reindex(list(DATASETS))
    print("\nmean accuracy on the reported 20% rows")
    print(view.round(3).to_string())
    for reading in READINGS:
        published = view[(reading, BASE)]
        print(f"   {BASE}, {reading} reading: mean over the datasets {published.mean():.3f}, "
              f"{int((published > 0.995).sum())} datasets at 1.00")
    genes = runs.pivot_table(index="dataset", columns=["reading", "arm"], values="n_features", sort=False).reindex(list(DATASETS))
    print("\nmean number of genes kept")
    print(genes.round(0).to_string())
    show = ["reading", "comparison", "metric", "higher_in", "lower_in", "mean_diff", "median_diff", "p", "verdict"]
    print("\nacross the datasets, each arm minus the base as published, on its primary metric")
    print(across[across.primary][show].round(4).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in ([], ["summary"]):
        print(__doc__)
        return 2
    (OUT / "jobs").mkdir(parents=True, exist_ok=True)
    start, jobs = time.perf_counter(), [] if argv else jobs_to_do()
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            name, run, rows = future.result()
            partial = OUT / "jobs" / f"{name}_r{run}.partial"
            pd.DataFrame(rows).to_csv(partial, index=False)
            os.replace(partial, OUT / "jobs" / f"{name}_r{run}.csv")
            if n % 9 == 0 or n == len(futures):
                print(f"{n} of {len(futures)} jobs done after {time.perf_counter() - start:.0f}s", flush=True)
    summarise()
    print(f"done in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
