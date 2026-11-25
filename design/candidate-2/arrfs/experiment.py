"""The protocol of grounding.md as code: settings, seeds, jobs, and run_fold.

run_fold is the unit of work, of parallelism and of restart. It is a pure
function of (job, settings, what is already done): it reads the dataset, does
no I/O besides that, and returns a FoldResult. The caller saves it.
"""
from dataclasses import dataclass

import numpy as np

from .optimizers import REGISTRY
from .results import FoldResult, MethodKey

TASKS = ("binary", "multiclass")
FITNESS_KINDS = ("accuracy", "imbalance")
REFERENCES = ("knn_all", "logreg_all", "svm_all", "anova_knn")


@dataclass(frozen=True)
class Settings:
    """Everything that changes a number. Written to manifest.json; a run
    directory refuses to resume under different settings. `repeats` is
    deliberately absent: a smoke run (1 repeat) can be extended to the full
    run in the same directory."""
    pop: int = 20
    iters: int = 100
    outer_splits: int = 5
    inner_splits: int = 3
    knn_k: int = 5
    base_seed: int = 42


@dataclass(frozen=True)
class FoldJob:
    task: str
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def expected_keys() -> list[MethodKey]:
    """Every MethodKey a complete fold holds: REGISTRY x FITNESS_KINDS, plus
    REFERENCES with fitness 'none'. Derived, so adding to REGISTRY extends the grid."""
    raise NotImplementedError


def make_jobs(y_by_task: dict[str, np.ndarray], settings: Settings, repeats: int) -> list[FoldJob]:
    """RepeatedStratifiedKFold(outer_splits, n_repeats=repeats, random_state=42)
    per task; fold index = position within the repeat."""
    raise NotImplementedError


def seed_for(settings: Settings, job: FoldJob, *tags: str) -> np.random.SeedSequence:
    """SeedSequence([base_seed, crc32(task), repeat, fold, *crc32(tags)]).
    crc32, not hash(): Python's str hash changes between processes. Algorithms
    are identified by name, not list position, so adding one never shifts the
    seed of another. tags: () for inner folds + X0 ("shared"), (alg, kind) per search."""
    raise NotImplementedError


def run_fold(job: FoldJob, settings: Settings, existing: FoldResult | None = None) -> FoldResult:
    """Compute every expected key not already in `existing`; return the merge.

    Steps
    - load data, FoldPreprocessor.fit on train rows, transform train and test
    - inner folds: StratifiedKFold(inner_splits, shuffle=True, random_state=<seed_for(job)>)
    - X0 = rng(seed_for(job, "x0")).random((pop, D))   one per fold, shared
    - one WrapperFitness per kind, reused by all algorithms in this fold
      (its mask cache is deterministic, only `calls` bookkeeping differs)
    - for each missing (alg, kind): Problem -> alg.run(problem, Generator(seed_for(job, alg, kind)));
      calls = fitness.calls delta; then fit sklearn 5-NN on train[:, mask], predict test
    - references; anova_knn uses k = n_features of the ("gwo_cpo", "imbalance") row,
      from `existing` if present, else from this call
    - map each selected mask back to original column indices via preprocessor.keep_
    """
    raise NotImplementedError


def final_scores(y_test: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """fitness.class_scores on the outer-test fold. The one place test metrics are computed."""
    raise NotImplementedError
