"""Result records and their on-disk form.

Layout:  <out>/manifest.json
         <out>/<task>/r<repeat>_f<fold>.pkl      one file per outer fold

One file per fold is the whole concurrency story: exactly one worker ever
writes a given file, so there is nothing to lock and nothing to merge at write
time. The analysis merges at read time (load_rows / load_oof).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


class MethodKey(NamedTuple):
    """Identifies one row inside a fold. Reference models use fitness='none'."""
    method: str     # a REGISTRY name, or "knn_all" | "logreg_all" | "svm_all" | "anova_knn"
    fitness: str    # "accuracy" | "imbalance" | "none"


@dataclass(frozen=True)
class Row:
    """One (task, repeat, fold, method, fitness) result. Becomes one DataFrame row."""
    task: str
    repeat: int
    fold: int
    method: str
    fitness: str
    acc: float
    bacc: float
    macro_f1: float
    worst_recall: float
    recall: np.ndarray              # (C,) per-class test recall, 0 for absent classes
    n_features: int
    selected: np.ndarray            # int indices into the ORIGINAL 279 columns (not the preprocessed ones)
    seconds: float                  # wall time of search + final fit; includes cross-process contention
    calls: int                      # fitness calls, cache hits included; 0 for references
    best_fitness: float             # nan for references
    curve: np.ndarray | None        # (iters + 1,) best-so-far; None for references


@dataclass
class FoldResult:
    task: str
    repeat: int
    fold: int
    test_idx: np.ndarray                    # row indices into the 452 records
    rows: dict[MethodKey, Row]
    oof: dict[MethodKey, np.ndarray]        # test-fold predictions, aligned with test_idx


def fold_path(out: Path, task: str, repeat: int, fold: int) -> Path:
    raise NotImplementedError


def save_fold(path: Path, result: FoldResult) -> None:
    """Write to a temp file in the same directory, then os.replace. A crash
    leaves either the old file or the new one, never a half-written one."""
    raise NotImplementedError


def load_fold(path: Path) -> FoldResult | None:
    """None when the file is missing. A file that fails to unpickle is treated
    as missing (and reported), so a corrupt fold is redone, not fatal."""
    raise NotImplementedError


def load_rows(out: Path) -> pd.DataFrame:
    """All folds, one row per Row, columns = Row fields. The only reader the
    analysis uses for metrics, curves and masks."""
    raise NotImplementedError


def load_oof(out: Path) -> dict[tuple[str, int], dict[MethodKey, tuple[np.ndarray, np.ndarray]]]:
    """(task, repeat) -> key -> (test_idx, preds) concatenated over that repeat's folds."""
    raise NotImplementedError
