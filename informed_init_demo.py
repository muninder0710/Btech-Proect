"""Does an informed start let GWO-CPO beat the base paper's method?

Li, Xue & Zhang 2021 (Applied Soft Computing 106:107302, Section 3.2) start their search from masks that keep
feature d with probability p_d = lambda / (1 + exp(sigma - 2 sigma (w_d - min w) / (max w - min w))), where w_d
is the mutual information between feature d and the class (their Eq. 14, with lambda = 0.5 and sigma = 10).

This script reruns the main experiment's Arrhythmia folds with that start in place of the random one, for all
five algorithms and both fitness kinds. Everything else is the main run's: outer folds, preprocessing, inner
split, population, budget, search seeds and classifiers. Before it runs, it checks that with the random start
it picks the main run's saved subsets exactly, so the start is the only difference.

Design: design/amendment-6-informed-start.md.
Run from code/:  .venv/bin/python informed_init_demo.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from analyze import _paired, holm  # noqa: E402
from arrfs.data import FoldPreprocessor, load_task  # noqa: E402
from arrfs.experiment import (_CLASSIFIERS, FITNESS_KINDS, PROPOSED, FoldKey, FoldResult, Settings,  # noqa: E402
                              _outer_split, fold_keys)
from arrfs.fitness import WrapperFitness, class_scores  # noqa: E402
from arrfs.optimizers import ALGORITHMS, search  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "informed_init"
MAIN = HERE / "results" / "full"
TASKS = ("arrhythmia-multiclass", "arrhythmia-binary")
SETTINGS = Settings(tasks=TASKS)
BASE = ("IBGWO4", "accuracy")  # the base paper's method as published
LAMBDA, SIGMA = 0.5, 10.0      # Li, Xue & Zhang 2021, Section 4.3
METRICS = ["macro_f1", "acc", "bacc", "n_features"]
CHECK_KEYS = [FoldKey(task, repeat, fold) for task in TASKS for repeat, fold in [(0, 0), (3, 2)]]


def arm(method, fitness, start):
    return f"{method} / {fitness} / {start}"


def fold_context(key):
    """The main run's view of one outer fold: (Xtr, ytr, Xte, yte, inner folds, kept raw attributes, classes)."""
    data = load_task(key.task)
    train, test = _outer_split(data.y, key, SETTINGS)
    prep = FoldPreprocessor().fit(data.X[train])
    Xtr, Xte, ytr = prep.transform(data.X[train]), prep.transform(data.X[test]), data.y[train]
    inner = list(StratifiedKFold(SETTINGS.inner_splits, shuffle=True, random_state=key.seed()).split(Xtr, ytr))
    return Xtr, ytr, Xte, data.y[test], inner, np.flatnonzero(prep.keep_), len(data.class_names)


def informed_start(key, Xtr, ytr):
    """X0 whose masks (X0 > 0.5) keep feature d with the probability of Li, Xue & Zhang's Eq. 14."""
    w = mutual_info_classif(Xtr, ytr, random_state=key.seed("mi"))
    assert w.max() > w.min(), key.name
    z = 2 * SIGMA * (w - w.min()) / (w.max() - w.min()) - SIGMA
    p = LAMBDA / (1 + np.exp(-z))
    keep = np.random.default_rng(key.seed("fwdi")).random((SETTINGS.population, len(p))) < p
    return np.where(keep, 0.75, 0.25)


def select(key, Xtr, ytr, inner, C, X0, name, kind):
    fitness = WrapperFitness(Xtr, ytr, C, inner, kind, k=SETTINGS.neighbours)
    return search(name, fitness, X0, SETTINGS.budget, np.random.default_rng(key.seed(name, kind)))


def check_matches_main_run(key):
    """With the random start, this script's fold pipeline must pick the main run's saved subsets."""
    warnings.filterwarnings("ignore", message="The least populated class")
    saved = FoldResult.from_json((MAIN / "folds" / f"{key.name}.json").read_text())
    Xtr, ytr, _, _, inner, kept, C = fold_context(key)
    X0 = np.random.default_rng(key.seed()).random((SETTINGS.population, Xtr.shape[1]))
    for name, kind in [PROPOSED, BASE]:
        mask = select(key, Xtr, ytr, inner, C, X0, name, kind).mask
        want = next(m.selected for m in saved.methods if (m.method, m.fitness) == (name, kind))
        assert np.array_equal(kept[np.flatnonzero(mask)], want), (key.name, name, kind)
    return key.name


def run_fold(key):
    """Every algorithm and fitness kind on one outer fold, from one shared informed start."""
    warnings.filterwarnings("ignore", message="The least populated class")
    Xtr, ytr, Xte, yte, inner, _, C = fold_context(key)
    X0 = informed_start(key, Xtr, ytr)
    rows = []
    for kind in FITNESS_KINDS:
        for name in ALGORITHMS:
            res = select(key, Xtr, ytr, inner, C, X0, name, kind)
            cols = np.flatnonzero(res.mask)
            for clf, make in _CLASSIFIERS.items():
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):  # spurious Accelerate flags
                    yhat = make(key.seed(), SETTINGS.neighbours).fit(Xtr[:, cols], ytr).predict(Xte[:, cols])
                s = class_scores(yte, yhat, C)
                rows.append(dict(task=key.task, repeat=key.repeat, fold=key.fold, method=name, fitness=kind,
                                 classifier=clf, start="informed", start_size=(X0 > 0.5).sum(1).mean(),
                                 n_features=len(cols), best_fitness=res.fitness,
                                 acc=s["acc"], bacc=s["bacc"], macro_f1=s["macro_f1"]))
    return rows


def difference(rows, task, classifier, a, b, metric):
    """Arm a minus arm b, paired over the 25 folds, with the main tables' statistics."""
    view = rows[(rows.task == task) & (rows.classifier == classifier)]
    wide = view.pivot_table(index=["repeat", "fold"], columns="arm", values=metric)
    diff = (wide[a] - wide[b]).to_numpy()
    assert len(diff) == SETTINGS.repeats * SETTINGS.outer_splits and not np.isnan(diff).any(), (task, a, b, metric)
    return dict(task=task, classifier=classifier, a=a, b=b, metric=metric,
                mean_a=wide[a].mean(), mean_b=wide[b].mean(), **_paired(diff))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    saved_settings = json.loads((MAIN / "manifest.json").read_text())["settings"]
    ours = json.loads(json.dumps(asdict(SETTINGS)))
    assert {k: v for k, v in saved_settings.items() if k != "tasks"} == {k: v for k, v in ours.items() if k != "tasks"}

    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        checked = list(pool.map(check_matches_main_run, CHECK_KEYS))
        print(f"random start reproduces the main run's subsets on {len(checked)} folds: {', '.join(checked)}", flush=True)
        futures = [pool.submit(run_fold, key) for key in fold_keys(SETTINGS)]
        informed = pd.DataFrame([r for f in futures for r in f.result()])
    assert len(informed) == len(fold_keys(SETTINGS)) * len(ALGORITHMS) * len(FITNESS_KINDS) * len(_CLASSIFIERS)
    informed.to_csv(OUT / "informed_init_rows.csv", index=False)

    random = pd.read_csv(MAIN / "summary" / "rows.csv", float_precision="round_trip")
    random = random[random.method.isin(ALGORITHMS) & random.task.isin(TASKS)].assign(start="random")
    rows = pd.concat([random, informed], ignore_index=True)
    rows["arm"] = [arm(m, f, s) for m, f, s in zip(rows.method, rows.fitness, rows.start)]
    assert not rows.duplicated(["task", "repeat", "fold", "arm", "classifier"]).any()

    knn = rows[rows.classifier == "KNN"]
    means = knn.groupby(["task", "arm"], sort=False)[METRICS + ["best_fitness", "start_size"]].mean().reset_index()
    means.to_csv(OUT / "informed_init_means.csv", index=False)

    proposed, base = arm(*PROPOSED, "informed"), arm(*BASE, "random")
    claim = pd.DataFrame([difference(rows, t, c, proposed, base, m) for t in TASKS for c in _CLASSIFIERS for m in METRICS])
    primary = (claim.classifier == "KNN") & (claim.metric == "macro_f1")
    holds = (claim.mean_diff > 0) & (claim.p_exploratory < 0.05)
    claim["verdict"] = np.where(~primary, "secondary or exploratory (no claim)", np.where(holds, "holds", "does not hold"))
    claim.to_csv(OUT / "informed_init_claim.csv", index=False)

    pairs = [(arm(*BASE, "informed"), base), (proposed, arm(*BASE, "informed")), (proposed, arm(*PROPOSED, "random"))]
    attribution = pd.DataFrame([difference(rows, t, "KNN", a, b, m) for t in TASKS for a, b in pairs for m in METRICS])
    attribution.to_csv(OUT / "informed_init_attribution.csv", index=False)

    effect = pd.DataFrame([difference(rows, t, "KNN", arm(n, k, "informed"), arm(n, k, "random"), m)
                           for t in TASKS for m in METRICS + ["best_fitness"] for k in FITNESS_KINDS for n in ALGORITHMS])
    for _, block in effect.groupby(["task", "metric"]):
        effect.loc[block.index, "p_holm"] = holm(block.p_exploratory.to_numpy())
    effect.to_csv(OUT / "informed_init_effect.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    show = ["task", "classifier", "a", "b", "metric", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
            "wins", "ties", "losses", "p_exploratory"]
    print(means.round(3).to_string(index=False))
    print(claim[show + ["verdict"]].round(4).to_string(index=False))
    print(attribution[show].round(4).to_string(index=False))
    print(effect[show + ["p_holm"]].round(4).to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
