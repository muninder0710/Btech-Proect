"""Turn a run directory into the review's tables, tests and figures.

    python analyze.py results/full      # writes results/full/summary/

Works on a partial run and says how many folds it found. Every metric comes
from the saved predictions through fitness.class_scores, so per-fold and
pooled numbers share one definition.

Written to <run>/summary/:
    rows.csv                  one row per (task, repeat, fold, method, fitness), arrays left out
    slide_<task>.csv          mean ± sd over folds, plus pooled out-of-fold scores
    recall_<task>.csv         pooled per-class recall
    wilcoxon_<task>.csv       GWO-CPO/imbalance against every other row, paired by fold
    ablation_<task>.csv       imbalance fitness minus accuracy fitness, per algorithm
    leads_<task>.csv/.png     selection rate per ECG lead
    features_<task>.csv       selection frequency per raw attribute (input to the SHAP step)
    convergence_<task>.png    mean best-so-far fitness per algorithm
"""
from pathlib import Path

import numpy as np
import pandas as pd

from arrfs.experiment import PROPOSED, Task

METRICS = ("acc", "bacc", "macro_f1")


def load_rows(run_dir: Path) -> pd.DataFrame:
    """One row per (task, repeat, fold, method, fitness) of a run directory.
    Prints "found 48 of 50 folds" when the run is incomplete, using the manifest.

    Columns: task, repeat, fold, method, fitness, label ("GWO-CPO / imbalance"
    for search rows, the method name for references), acc, bacc, macro_f1,
    worst_recall, recall (float[C], NaN for a class with no record in that
    test fold; class_scores reports 0 there), n_features, seconds,
    best_fitness and calls (NaN for references), and the array columns
    selected, kept, curve, test_idx, y_true, y_pred, class_names.
    Row order is grid order (search rows, then references), and every table
    below keeps it, so no list of method names exists outside the runner.
    """
    raise NotImplementedError


def pooled_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, label, repeat) with acc, bacc, macro_f1 and recall
    computed on that repeat's concatenated out-of-fold predictions.

    Each record is predicted exactly once per repeat, so a class with 4 or 5
    records counts in full here, while in a single test fold it has 0 or 1
    records and its recall is NaN or 0 or 1.
    """
    raise NotImplementedError


def slide_table(rows: pd.DataFrame, pooled: pd.DataFrame, task: Task) -> pd.DataFrame:
    """The review table for one task, one row per label.

    Columns as "mean ± sd" strings: Acc, BAcc, Macro-F1, Features and Time (s)
    over the folds, then Pooled BAcc and Pooled Macro-F1 over the repeats.
    """
    raise NotImplementedError


def recall_table(pooled: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Pooled per-class recall, mean over repeats: one row per label, one column per class name."""
    raise NotImplementedError


def wilcoxon_vs_proposed(rows: pd.DataFrame, task: Task,
                         metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    """PROPOSED against every other label of `task`, paired by (repeat, fold).

    One row per (label, metric): median difference (proposed minus other),
    wins, ties, losses, two-sided p from scipy.stats.wilcoxon, and p_holm,
    Holm-adjusted over the comparisons of that metric. A label with every
    difference zero gets p = 1. Folds of repeated CV share training records,
    so these p-values are optimistic; the README says so next to the table.
    """
    raise NotImplementedError


def fitness_ablation(rows: pd.DataFrame, task: Task) -> pd.DataFrame:
    """For each algorithm, imbalance fitness minus accuracy fitness, paired by
    (repeat, fold): mean ± sd of the difference in acc, bacc, macro_f1 and
    n_features, plus Wilcoxon p on bacc and macro_f1. This is the evidence for
    the second contribution, the imbalance-aware fitness."""
    raise NotImplementedError


def lead_frequency(rows: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Selection rate per ECG lead: one row per label, one column per lead
    ("General", "DI", ..., "V6").

    Each cell is the mean over folds of (attributes selected in the lead) /
    (attributes of the lead that survived preprocessing in that fold). Dividing
    by what was available matters: "General" has 15 attributes, each lead 22,
    and the preprocessing drops a different few per fold. Leads come from
    data.feature_names_and_leads(), indexed by the stored raw attribute indices.
    """
    raise NotImplementedError


def feature_frequency(rows: pd.DataFrame, task: Task) -> pd.DataFrame:
    """279 rows (attribute name, lead), one column per search label: the
    fraction of folds in which that label selected the attribute."""
    raise NotImplementedError


def plot_convergence(rows: pd.DataFrame, task: Task, path: Path) -> None:
    """Mean best-so-far fitness against iteration (0 .. T), one line per
    algorithm with a ±1 sd band over folds, one panel per fitness kind
    because the two fitness scales differ."""
    raise NotImplementedError


def plot_leads(leads: pd.DataFrame, path: Path) -> None:
    """Bar chart of `lead_frequency` for PROPOSED, with GWO-CPO/accuracy beside it."""
    raise NotImplementedError


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the same order as p."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Load the run, write every output listed in the module docstring, and
    print the slide tables. Takes one argument, the run directory."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
