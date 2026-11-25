"""Turn a results directory into slide tables and figures.

    python analyze.py results/full          # writes results/full/report/*.csv, *.png

Every function takes the DataFrame from results.load_rows and returns a
DataFrame (or a Figure), so each can be called from a notebook as well.
Labels and the proposed-method row come from optimizers.REGISTRY / PROPOSED.
"""
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def slide_table(rows: pd.DataFrame, task: str) -> pd.DataFrame:
    """One line per (method, fitness): mean +- sd over folds of acc, bacc,
    macro_f1, n_features, seconds. Strings like '0.71 +- 0.05' ready to paste."""
    raise NotImplementedError


def pooled_table(out: Path, task: str) -> pd.DataFrame:
    """Per (method, fitness): class_scores on out-of-fold predictions pooled
    over each repeat's five folds, then mean +- sd over repeats. This is the
    honest number for rare classes, which sit in only a few test folds."""
    raise NotImplementedError


def wilcoxon_vs_proposed(rows: pd.DataFrame, task: str, metric: str = "macro_f1") -> pd.DataFrame:
    """scipy.stats.wilcoxon, GWO-CPO + imbalance against every other
    (method, fitness), paired by (repeat, fold). Columns: other, n_pairs,
    mean_diff, statistic, p_value, p_holm (Holm over the rows of this call).
    All-zero differences give p = 1, not an exception. Folds within a repeat
    share data, so p-values are optimistic; the docstring of the output says so."""
    raise NotImplementedError


def fitness_ablation(rows: pd.DataFrame, task: str) -> pd.DataFrame:
    """Per algorithm, imbalance minus accuracy fitness on acc, bacc, macro_f1,
    n_features (mean diff, Wilcoxon p paired by fold). Innovation 2's evidence."""
    raise NotImplementedError


def lead_frequency(rows: pd.DataFrame, task: str, method: str = "gwo_cpo",
                   fitness: str = "imbalance") -> pd.DataFrame:
    """Fraction of folds in which each feature was selected, and the same
    summed per ECG lead (data.feature_names_and_leads), normalised by lead size.
    Uses Row.selected, which is already in original-column coordinates."""
    raise NotImplementedError


def convergence_figure(rows: pd.DataFrame, task: str, fitness: str) -> Figure:
    """Mean best-so-far curve per algorithm with a +- sd band, over folds."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Load rows once, call each function above per task and fitness kind,
    write CSVs and PNGs to <dir>/report/."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
