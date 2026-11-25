"""Turn a run directory into the review's tables, tests and figures.

    python analyze.py results/smoke     # writes results/smoke/summary/

Works on a partial run and says how many folds it found. Every metric comes
from the saved predictions through fitness.class_scores, so per-fold and
pooled numbers share one definition.

Written to <run>/summary/:
    rows.csv                      one row per (task, repeat, fold, method, fitness, classifier)
    <classifier>/slide_<task>.csv     mean ± sd over folds, plus pooled out-of-fold scores
    <classifier>/recall_<task>.csv    pooled per-class recall
    <classifier>/compare_<task>.csv   GWO-CPO / imbalance against every other row, paired by fold
    <classifier>/ablation_<task>.csv  imbalance fitness minus accuracy fitness, per algorithm
    <classifier>/cross_dataset.csv    macro-F1 per task and method, and the average rank over tasks
    stability_<task>.csv          Nogueira stability and mean pairwise Jaccard of the fold subsets
    convergence_<task>.png        mean best-so-far fitness per algorithm
    leads_<task>.csv / .png       selection rate per ECG lead (Arrhythmia tasks only)
    features_<task>.csv           selection frequency per raw attribute (Arrhythmia tasks only; input to the SHAP step)

KNN is the classifier inside the search, so summary/KNN/ holds the headline
tables. The Majority row (a most-frequent-class dummy) appears in every
classifier's tables. p-values are exploratory: folds of repeated
cross-validation share training records, so they are not independent.
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from arrfs.data import TASKS, Dataset, load_task  # noqa: E402
from arrfs.experiment import (CLASSIFIERS, DUMMY, FITNESS_KINDS, NO_SEARCH, PROPOSED,  # noqa: E402
                              fold_keys, load_run)
from arrfs.fitness import class_scores  # noqa: E402

METRICS = ("acc", "bacc", "macro_f1")
BOOTSTRAP_RESAMPLES = 2000
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def label(method: str, fitness: str) -> str:
    """'GWO-CPO / imbalance' for a search row, the method name for a reference."""
    return method if fitness == NO_SEARCH else f"{method} / {fitness}"


PROPOSED_LABEL = label(*PROPOSED)


def load_methods(run_dir: Path) -> pd.DataFrame:
    """One row per (task, repeat, fold, method, fitness), in grid order, with
    the saved arrays (selected, kept, curve, test_idx, y_true, y_pred) kept as
    object columns. Prints how many of the grid's folds it found."""
    settings, folds = load_run(run_dir)
    print(f"found {len(folds)} of {len(fold_keys(settings))} folds in {run_dir}")
    return pd.DataFrame([
        dict(task=r.key.task, repeat=r.key.repeat, fold=r.key.fold, method=m.method, fitness=m.fitness,
             label=label(m.method, m.fitness), n_features=len(m.selected), seconds=m.seconds,
             best_fitness=m.best_fitness, calls=m.calls, selected=m.selected, kept=r.kept, curve=m.curve,
             test_idx=r.test_idx, y_true=r.y_true, y_pred=m.y_pred, class_names=r.class_names)
        for r in folds for m in r.methods
    ])


def _scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """class_scores, with the recall of a class absent from y_true set to NaN
    (class_scores reports 0 there, which would read as a miss)."""
    s = class_scores(y_true, y_pred, n_classes)
    recall = np.where(np.bincount(y_true, minlength=n_classes) > 0, s["recall"], np.nan)
    return {"acc": s["acc"], "bacc": s["bacc"], "macro_f1": s["macro_f1"],
            "worst_recall": s["worst_recall"], "recall": recall}


def score_rows(methods: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, repeat, fold, label, classifier) with acc, bacc,
    macro_f1, worst_recall and per-class recall on that outer-test fold."""
    return pd.DataFrame([
        dict(task=r.task, repeat=r.repeat, fold=r.fold, label=r.label, method=r.method, fitness=r.fitness,
             classifier=clf, n_features=r.n_features, seconds=r.seconds, best_fitness=r.best_fitness,
             calls=r.calls, **_scores(r.y_true, pred, len(r.class_names)))
        for r in methods.itertuples() for clf, pred in r.y_pred.items()
    ])


def pooled_scores(methods: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, label, classifier, repeat), scored on that repeat's
    concatenated out-of-fold predictions. Each record is predicted once per
    repeat, so a class with 4 records counts in full here, while a single test
    fold holds 0 or 1 of them."""
    rows = []
    for (task, lab, repeat), g in methods.groupby(["task", "label", "repeat"], sort=False):
        y_true = np.concatenate(g.y_true.tolist())
        for clf in g.y_pred.iloc[0]:
            y_pred = np.concatenate([p[clf] for p in g.y_pred])
            rows.append(dict(task=task, label=lab, classifier=clf, repeat=repeat, n_folds=len(g),
                             **_scores(y_true, y_pred, len(g.class_names.iloc[0]))))
    return pd.DataFrame(rows)


def _view(df: pd.DataFrame, task: str, classifier: str) -> pd.DataFrame:
    """The rows one classifier's tables show: that classifier, plus the Majority row."""
    return df[(df.task == task) & df.classifier.isin([classifier, DUMMY])]


def _mean_sd(x: pd.Series, digits: int = 3) -> str:
    return f"{x.mean():.{digits}f}" if len(x) < 2 else f"{x.mean():.{digits}f} ± {x.std():.{digits}f}"


def slide_table(scores: pd.DataFrame, pooled: pd.DataFrame, task: str, classifier: str) -> pd.DataFrame:
    """The review table for one task and classifier, one row per label in grid
    order. Acc, BAcc, Macro-F1, Features and Time (s, to choose the subset) are
    mean ± sd over folds; the pooled columns are mean ± sd over repeats."""
    s, p = _view(scores, task, classifier), _view(pooled, task, classifier)
    return pd.DataFrame([{
        "Method": lab,
        "Acc": _mean_sd(a.acc), "BAcc": _mean_sd(a.bacc), "Macro-F1": _mean_sd(a.macro_f1),
        "Features": _mean_sd(a.n_features, 1), "Time (s)": _mean_sd(a.seconds, 1),
        "Pooled BAcc": _mean_sd(p[p.label == lab].bacc), "Pooled Macro-F1": _mean_sd(p[p.label == lab].macro_f1),
    } for lab, a in s.groupby("label", sort=False)])


def recall_table(pooled: pd.DataFrame, class_names: list[str], task: str, classifier: str) -> pd.DataFrame:
    """Pooled per-class recall, mean over repeats: one row per label, one column per class."""
    p = _view(pooled, task, classifier)
    return pd.DataFrame([[lab, *np.mean(np.stack(g.recall.tolist()), axis=0).round(3)]
                         for lab, g in p.groupby("label", sort=False)], columns=["Method", *class_names])


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the same order as p."""
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def _paired(d: np.ndarray) -> dict:
    """Summary of paired differences d (one per fold): mean, 95% percentile
    bootstrap interval of the mean, wins/ties/losses, and the two-sided
    Wilcoxon signed-rank p (1.0 when every difference is zero)."""
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (BOOTSTRAP_RESAMPLES, len(d)))].mean(axis=1)
    tie = np.isclose(d, 0, atol=1e-12)
    return dict(n_folds=len(d), mean_diff=d.mean(), ci_low=np.percentile(means, 2.5),
                ci_high=np.percentile(means, 97.5), wins=int((~tie & (d > 0)).sum()),
                ties=int(tie.sum()), losses=int((~tie & (d < 0)).sum()),
                p_exploratory=1.0 if tie.all() else wilcoxon(d[~tie]).pvalue)


def _compare(rows: pd.DataFrame, pairs: list[tuple[str, str, str]], metrics: tuple[str, ...]) -> pd.DataFrame:
    """For each metric and each (name, a, b): a minus b paired by (repeat, fold),
    with Holm adjustment over the comparisons of that metric."""
    out = []
    for metric in metrics:
        wide = rows.pivot_table(index=["repeat", "fold"], columns="label", values=metric)
        block = [dict(metric=metric, comparison=name, **_paired((wide[a] - wide[b]).dropna().to_numpy()))
                 for name, a, b in pairs]
        for row, p in zip(block, holm(np.array([r["p_exploratory"] for r in block]))):
            row["p_holm_exploratory"] = p
        out += block
    return pd.DataFrame(out).round(4)


def compare_to_proposed(scores: pd.DataFrame, task: str, classifier: str) -> pd.DataFrame:
    """PROPOSED minus every other label; a positive mean_diff favours the proposed model."""
    s = _view(scores, task, classifier)
    pairs = [(f"{PROPOSED_LABEL} vs {lab}", PROPOSED_LABEL, lab) for lab in pd.unique(s.label) if lab != PROPOSED_LABEL]
    return _compare(s, pairs, METRICS)


def fitness_ablation(scores: pd.DataFrame, task: str, classifier: str) -> pd.DataFrame:
    """For each algorithm, the proposed fitness kind minus each other kind,
    paired by (repeat, fold), on the metrics and the number of features.
    This is the evidence for the imbalance-aware fitness."""
    s = _view(scores, task, classifier)
    pairs = [(f"{alg}: {PROPOSED[1]} minus {base}", label(alg, PROPOSED[1]), label(alg, base))
             for alg in pd.unique(s.method[s.fitness != NO_SEARCH])
             for base in FITNESS_KINDS if base != PROPOSED[1]]
    return _compare(s, pairs, (*METRICS, "n_features"))


def cross_dataset_table(scores: pd.DataFrame, tasks: list[str], classifier: str) -> pd.DataFrame:
    """Macro-F1 (mean ± sd over folds), one row per label and one column per
    task, plus each label's rank averaged over the tasks, where rank 1 is the
    highest mean macro-F1 on a task and tied means share a rank. The average
    rank is descriptive. No test stands behind it."""
    s = scores[scores.classifier.isin([classifier, DUMMY])]
    labels = list(pd.unique(s.label))
    cells = {key: g.macro_f1 for key, g in s.groupby(["label", "task"], sort=False)}
    table = pd.DataFrame({task: [_mean_sd(cells[lab, task]) for lab in labels] for task in tasks}, index=labels)
    means = pd.DataFrame({task: [cells[lab, task].mean() for lab in labels] for task in tasks}, index=labels)
    table[f"Average rank over {len(tasks)} tasks (descriptive)"] = means.rank(ascending=False).mean(axis=1).round(2)
    return table.rename_axis("Method").reset_index()


def _raw_masks(selected: pd.Series, n_raw: int) -> np.ndarray:
    Z = np.zeros((len(selected), n_raw), dtype=bool)
    for row, idx in zip(Z, selected):
        row[idx] = True
    return Z


def nogueira(Z: np.ndarray) -> float:
    """Stability of M subsets (bool[M, d]) by Nogueira, Sechidis and Brown
    (2018, JMLR 18:174): 1 minus the mean unbiased per-feature variance over
    its value under random selection of the same mean size. 1 means identical
    subsets, about 0 means chance. NaN when every subset is empty or full."""
    M, d = Z.shape
    k = Z.sum(axis=1).mean()
    if k in (0, d):
        return np.nan
    p = Z.mean(axis=0)
    return 1 - (M / (M - 1) * p * (1 - p)).mean() / (k / d * (1 - k / d))


def mean_jaccard(Z: np.ndarray) -> float:
    """Mean |A ∩ B| / |A ∪ B| over all pairs of subsets; NaN when all are empty."""
    ratios = [(a & b).sum() / (a | b).sum() for a, b in combinations(Z, 2) if (a | b).any()]
    return float(np.mean(ratios)) if ratios else np.nan


def stability(methods: pd.DataFrame, task: str, n_raw: int) -> pd.DataFrame:
    """Per label: Nogueira stability and mean pairwise Jaccard over the fold
    subsets of each repeat, in the task's raw-attribute space, mean ± sd over repeats."""
    rows = []
    for lab, g in methods[methods.task == task].groupby("label", sort=False):
        per_repeat = [(nogueira(Z), mean_jaccard(Z)) for Z in
                      (_raw_masks(r.selected, n_raw) for _, r in g.groupby("repeat"))]
        nog, jac = (pd.Series(v) for v in zip(*per_repeat))
        rows.append({"Method": lab, "Nogueira": _mean_sd(nog), "Jaccard": _mean_sd(jac),
                     "Features": _mean_sd(g.n_features, 1)})
    return pd.DataFrame(rows)


def lead_frequency(methods: pd.DataFrame, task: str, leads: np.ndarray) -> pd.DataFrame:
    """Selection rate per ECG lead: one row per label, one column per lead.

    Each cell is the mean over folds of (attributes selected in the lead) /
    (attributes of the lead that survived preprocessing in that fold). Dividing
    by what was available matters: "General" has 15 attributes, each lead 22,
    and preprocessing drops a different few per fold.
    """
    columns = list(pd.unique(leads))

    def rates(r) -> list[float]:
        return [np.isin(r.selected, np.flatnonzero(leads == c)).sum() / np.isin(r.kept, np.flatnonzero(leads == c)).sum()
                for c in columns]

    m = methods[methods.task == task]
    return pd.DataFrame([[lab, *np.mean([rates(r) for r in g.itertuples()], axis=0).round(3)]
                         for lab, g in m.groupby("label", sort=False)], columns=["Method", *columns])


def feature_frequency(methods: pd.DataFrame, task: str, data: Dataset) -> pd.DataFrame:
    """One row per raw attribute (name, lead), one column per search label:
    the fraction of folds in which that label selected the attribute."""
    m = methods[(methods.task == task) & (methods.fitness != NO_SEARCH)]
    freq = {lab: _raw_masks(g.selected, len(data.feature_names)).mean(axis=0).round(3)
            for lab, g in m.groupby("label", sort=False)}
    return pd.DataFrame({"attribute": data.feature_names, "lead": data.groups, **freq})


def plot_convergence(methods: pd.DataFrame, task: str, path: Path) -> None:
    """Mean best-so-far fitness against the share of the evaluation budget,
    one line per algorithm with a ±1 sd band over folds, one panel per fitness
    kind because the two fitness scales differ."""
    m = methods[(methods.task == task) & (methods.fitness != NO_SEARCH)]
    algorithms = list(pd.unique(m.method))
    fig, axes = plt.subplots(1, len(FITNESS_KINDS), figsize=(11, 4), constrained_layout=True)
    for ax, kind in zip(axes, FITNESS_KINDS):
        for color, alg in zip(SERIES_COLORS, algorithms):
            curves = np.stack(m[(m.method == alg) & (m.fitness == kind)].curve.tolist())
            x = np.arange(1, curves.shape[1] + 1) * 100 / curves.shape[1]
            mean, sd = curves.mean(axis=0), curves.std(axis=0)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, label=alg)
        ax.set(title=f"{kind} fitness", xlabel="% of the fitness-evaluation budget",
               ylabel="best fitness so far (lower is better)")
        ax.grid(color="#e5e5e2", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Convergence, {task} task (mean ± sd over {m.groupby(['repeat', 'fold']).ngroups} folds)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_leads(leads: pd.DataFrame, task: str, path: Path) -> None:
    """Grouped bars of the selection rate per lead for the proposed algorithm
    under each fitness kind."""
    shown = [label(PROPOSED[0], kind) for kind in FITNESS_KINDS]
    table = leads.set_index("Method").loc[shown]
    x = np.arange(table.shape[1])
    width = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for j, (color, lab) in enumerate(zip(SERIES_COLORS, shown)):
        ax.bar(x + (j - (len(shown) - 1) / 2) * width, table.loc[lab], width * 0.9, color=color, label=lab)
    ax.set_xticks(x, table.columns)
    ax.set(ylabel="share of the lead's attributes selected", title=f"Selected attributes per ECG lead, {task} task")
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    run_dir = Path(argv[0])
    out = run_dir / "summary"
    out.mkdir(exist_ok=True)
    methods = load_methods(run_dir)
    scores, pooled = score_rows(methods), pooled_scores(methods)
    scores.drop(columns="recall").to_csv(out / "rows.csv", index=False)
    tasks = [task for task in TASKS if task in set(methods.task)]
    for task in tasks:
        data = load_task(task)
        class_names = methods[methods.task == task].class_names.iloc[0]
        for clf in CLASSIFIERS:
            (out / clf).mkdir(exist_ok=True)
            slide = slide_table(scores, pooled, task, clf)
            slide.to_csv(out / clf / f"slide_{task}.csv", index=False)
            recall_table(pooled, class_names, task, clf).to_csv(out / clf / f"recall_{task}.csv", index=False)
            compare_to_proposed(scores, task, clf).to_csv(out / clf / f"compare_{task}.csv", index=False)
            fitness_ablation(scores, task, clf).to_csv(out / clf / f"ablation_{task}.csv", index=False)
            if clf == CLASSIFIERS[0]:
                print(f"\n{task}, {clf}\n{slide.to_string(index=False)}")
        stability(methods, task, len(data.feature_names)).to_csv(out / f"stability_{task}.csv", index=False)
        plot_convergence(methods, task, out / f"convergence_{task}.png")
        if data.groups is not None:
            leads = lead_frequency(methods, task, data.groups)
            leads.to_csv(out / f"leads_{task}.csv", index=False)
            plot_leads(leads, task, out / f"leads_{task}.png")
            feature_frequency(methods, task, data).to_csv(out / f"features_{task}.csv", index=False)
    for clf in CLASSIFIERS:
        table = cross_dataset_table(scores, tasks, clf)
        table.to_csv(out / clf / "cross_dataset.csv", index=False)
        if clf == CLASSIFIERS[0]:
            print(f"\nmacro-F1 across tasks, {clf}\n{table.to_string(index=False)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
