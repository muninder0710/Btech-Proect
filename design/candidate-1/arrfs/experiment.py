"""The experiment grid: its settings, one outer fold, and the results files.

`run_fold` turns one (task, repeat, fold) into a FoldResult. `prepare_run_dir`,
`save_fold` and `load_run` are the only code that touches a run directory, so
the file format has one owner.

Run directory:
    <out>/manifest.json                          settings + sha256 of the arrfs/ source
    <out>/folds/<task>_r<repeat>_f<fold>.json    one FoldResult, written atomically

A fold file either exists complete or does not exist. That single fact is the
resume rule and also why the analysis can read a run that is still going.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.base import ClassifierMixin

from .optimizers import ALGORITHMS

Task = Literal["binary", "multiclass"]
FitnessKind = Literal["accuracy", "imbalance"]

PROPOSED: tuple[str, FitnessKind] = ("GWO-CPO", "imbalance")


@dataclass(frozen=True)
class Settings:
    """Every number that defines the grid, fixed before any run. Nothing here
    is chosen by looking at outer-test results. Saved in manifest.json, so a
    results folder always records how it was made."""

    repeats: int = 5
    outer_splits: int = 5
    outer_seed: int = 42
    inner_splits: int = 3
    population: int = 20
    iterations: int = 100
    neighbours: int = 5      # k of the KNN, in the fitness and in the final evaluation
    tasks: tuple[Task, ...] = ("binary", "multiclass")
    fitness_kinds: tuple[FitnessKind, ...] = ("accuracy", "imbalance")
    algorithms: tuple[str, ...] = ALGORITHMS


@dataclass(frozen=True, order=True)
class FoldKey:
    """One cell of the outer loop. Its `name` is the fold file's stem."""

    task: Task
    repeat: int
    fold: int

    @property
    def name(self) -> str:
        """For example 'multiclass_r3_f1'."""
        raise NotImplementedError

    def seed(self, *extra: str) -> int:
        """Stable 32-bit seed from (task, repeat, fold, *extra) through sha256.

        Python's hash() of a string changes between processes, so it is not
        used. key.seed() seeds the fold (inner folds and X0).
        key.seed(algorithm, kind) seeds one search.
        """
        raise NotImplementedError


def fold_keys(settings: Settings) -> list[FoldKey]:
    """Every fold of the grid, in (task, repeat, fold) order."""
    raise NotImplementedError


@dataclass(frozen=True)
class MethodResult:
    """One method on one outer fold: one row of the results table.

    Metrics are not stored. The analysis derives them from y_pred with
    fitness.class_scores, the same function for per-fold and pooled numbers.
    """

    method: str                             # an ALGORITHMS name, or "KNN-all", "LogReg-all", "SVM-all", "ANOVA-KNN"
    fitness: FitnessKind | Literal["none"]  # "none" for the four reference models
    y_pred: np.ndarray                      # int[n_test], aligned with FoldResult.test_idx
    selected: np.ndarray                    # int[k], indices into the 279 raw attributes
    seconds: float                          # wall time to produce y_pred; the only field that varies between runs
    best_fitness: float | None = None       # search rows only
    calls: int | None = None                # search rows only
    curve: np.ndarray | None = None         # search rows only, float[T + 1]


@dataclass(frozen=True)
class FoldResult:
    """Everything one outer fold produced. Facts about the fold are stored
    once here, facts about each method in `methods`."""

    key: FoldKey
    class_names: list[str]
    test_idx: np.ndarray        # int[n_test], rows of the 452-record file
    y_true: np.ndarray          # int[n_test]
    kept: np.ndarray            # int[D], raw attributes that survived FoldPreprocessor on this fold's training rows
    methods: list[MethodResult]  # 5 algorithms x 2 fitness kinds, then the 4 references

    def to_json(self) -> dict:
        """Plain dict with arrays as lists. Floats round-trip exactly through json."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, d: dict) -> "FoldResult":
        raise NotImplementedError


def run_fold(key: FoldKey, settings: Settings) -> FoldResult:
    """Run every method of the grid on one outer fold.

    Pure apart from reading the dataset file: the same (key, settings) gives
    the same FoldResult on the same machine, `seconds` aside. Only the
    outer-train rows fit the preprocessing, feed the fitness and fit models.
    The outer-test rows are only transformed and predicted.
    """
    raise NotImplementedError
    # TODO
    #   X_raw, y_raw = load_raw() ; y, class_names = make_task(y_raw, key.task)
    #   train, test = _outer_split(y, key, settings)
    #   prep = FoldPreprocessor().fit(X_raw[train]) ; Xtr, Xte = prep.transform(X_raw[train]), prep.transform(X_raw[test])
    #   kept = flatnonzero(prep.keep_) ; D = len(kept)
    #   inner = list(StratifiedKFold(inner_splits, shuffle=True, random_state=key.seed()).split(Xtr, y[train]))
    #   X0 = default_rng(key.seed()).random((population, D))
    #   for kind in fitness_kinds, for name in algorithms:
    #       fitness = WrapperFitness(Xtr, y[train], C, inner, kind, k=neighbours)
    #           a fresh one per search: a shared cache would make later searches look faster
    #       timed: res = search(name, fitness, X0, iterations, default_rng(key.seed(name, kind)))
    #              y_pred = _knn_predict(Xtr[:, res.mask], y[train], Xte[:, res.mask], neighbours)
    #       MethodResult(name, kind, y_pred, kept[res.mask], seconds, res.fitness, res.calls, res.curve)
    #   k = number of attributes the PROPOSED row selected in this fold
    #   for ref, model in _reference_models(k, neighbours).items():
    #       timed: model.fit(Xtr, y[train]) ; y_pred = model.predict(Xte)
    #       MethodResult(ref, "none", y_pred, kept[_columns_used(model, D)], seconds)


def _outer_split(y: np.ndarray, key: FoldKey, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) of fold key.fold in repeat key.repeat of
    RepeatedStratifiedKFold(outer_splits, repeats, random_state=outer_seed),
    stratified on this task's labels.

    Repeat r's splits do not depend on how many repeats follow (checked on
    scikit-learn 1.5.2), so the smoke run's folds are the full run's repeat 0.
    """
    raise NotImplementedError


def _knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    """scikit-learn KNeighborsClassifier(k) fitted on outer-train rows, predicting outer-test rows."""
    raise NotImplementedError


def _reference_models(k_features: int, neighbours: int) -> dict[str, ClassifierMixin]:
    """The four no-search rows, in table order: KNN-all, LogReg-all
    (class_weight="balanced", max_iter=5000), SVM-all (RBF,
    class_weight="balanced") and ANOVA-KNN (SelectKBest(f_classif, k_features)
    then KNN). ANOVA-KNN keeps as many attributes as GWO-CPO/imbalance chose in
    the same fold, so the comparison is about which attributes, not how many."""
    raise NotImplementedError


def _columns_used(model: ClassifierMixin, D: int) -> np.ndarray:
    """Column indices (into the D kept columns) the fitted model reads:
    the SelectKBest support for ANOVA-KNN, all D for the others."""
    raise NotImplementedError


def prepare_run_dir(out: Path, settings: Settings) -> list[FoldKey]:
    """Make `out` a run directory for exactly these settings and this code,
    and return the folds not yet saved there, in grid order.

    Idempotent. A missing directory is created with its manifest. A matching
    manifest keeps every finished fold (resume). A different manifest (another
    setting, or any arrfs/*.py file edited) renames the old directory to
    <out>.stale-<YYYYmmdd-HHMMSS> and starts empty, so one table never mixes
    two versions of an algorithm. Nothing is deleted. Leftover *.tmp files
    from a crash are ignored, because only finished *.json files count.
    """
    raise NotImplementedError


def _manifest(settings: Settings) -> dict:
    """{"settings": the Settings as a dict, "source_sha256": sha256 over the
    arrfs/*.py files, read in name order}."""
    raise NotImplementedError


def save_fold(result: FoldResult, out: Path) -> None:
    """Write folds/<key.name>.json through a temp file and os.replace, so a
    crash leaves either no file or a complete one. Only the parent process
    calls this; workers never write."""
    raise NotImplementedError


def load_run(out: Path) -> tuple[Settings, list[FoldResult]]:
    """Read manifest.json and every saved fold, sorted by key. A run that is
    still going loads fine and simply has fewer folds than fold_keys(settings)."""
    raise NotImplementedError
