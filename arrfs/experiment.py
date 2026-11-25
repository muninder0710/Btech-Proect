"""The experiment grid: its settings, one outer fold, and the run directory.

`run_fold` turns one (task, repeat, fold) into a FoldResult. `prepare_run_dir`,
`save_fold` and `load_run` are the only code that touches a run directory, so
the file format has one owner.

Run directory:
    <out>/manifest.json                          settings + sha256 of the arrfs/ source
    <out>/folds/<task>_r<repeat>_f<fold>.json    one FoldResult, written atomically

A fold file either exists complete or does not exist. That single fact is the
resume rule, and it is why the analysis can read a run that is still going.
"""
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .data import TASKS, FoldPreprocessor, Task, load_task
from .fitness import WrapperFitness
from .optimizers import ALGORITHMS, search

FitnessKind = Literal["accuracy", "imbalance"]  # the kinds arrfs.fitness.WrapperFitness implements
FITNESS_KINDS: tuple[FitnessKind, ...] = get_args(FitnessKind)
NO_SEARCH = "none"  # the `fitness` of a reference row

PROPOSED: tuple[str, FitnessKind] = ("GWO-CPO", "imbalance")

_Model = Callable[[int, int], ClassifierMixin]  # (fold seed, neighbours) -> unfitted classifier

_CLASSIFIERS: dict[str, _Model] = {
    "KNN": lambda seed, k: KNeighborsClassifier(k),
    "LogReg": lambda seed, k: LogisticRegression(class_weight="balanced", max_iter=5000),
    "SVM": lambda seed, k: SVC(class_weight="balanced"),
    "RF": lambda seed, k: RandomForestClassifier(300, class_weight="balanced", random_state=seed),
}
CLASSIFIERS: tuple[str, ...] = tuple(_CLASSIFIERS)
DUMMY = "Dummy"

_Columns = Callable[[np.ndarray, np.ndarray, int], np.ndarray]  # (Xtr, ytr, k) -> kept-column indices

_REFERENCES: dict[str, tuple[_Columns, dict[str, _Model]]] = {
    "All features": (lambda Xtr, ytr, k: np.arange(Xtr.shape[1]), _CLASSIFIERS),
    "ANOVA-k": (lambda Xtr, ytr, k: SelectKBest(f_classif, k=k).fit(Xtr, ytr).get_support(indices=True),
                _CLASSIFIERS),
    "Majority": (lambda Xtr, ytr, k: np.arange(0),
                 {DUMMY: lambda seed, k: DummyClassifier(strategy="most_frequent")}),
}
REFERENCES: tuple[str, ...] = tuple(_REFERENCES)


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
    budget: int = 2000       # fitness calls per search, the same for every algorithm
    neighbours: int = 5      # k of the KNN, in the fitness and in the final evaluation
    algorithms: tuple[str, ...] = ALGORITHMS
    tasks: tuple[Task, ...] = TASKS

    def __post_init__(self):
        assert PROPOSED[0] in self.algorithms, "ANOVA-k takes its k from the proposed row"


@dataclass(frozen=True)
class FoldKey:
    """One cell of the outer loop. Its `name` is the fold file's stem."""

    task: Task
    repeat: int
    fold: int

    @property
    def name(self) -> str:
        return f"{self.task}_r{self.repeat}_f{self.fold}"

    def seed(self, *extra: str) -> int:
        """Stable 32-bit seed from (task, repeat, fold, *extra) through sha256.

        Python's hash() of a string changes between processes, so it is not
        used. key.seed() seeds the fold (inner folds, X0, the random forest).
        key.seed(algorithm, kind) seeds one search.
        """
        text = "|".join(map(str, (self.task, self.repeat, self.fold, *extra)))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def fold_keys(settings: Settings) -> list[FoldKey]:
    """Every fold of the grid, in (task, repeat, fold) order."""
    return [FoldKey(t, r, f) for t in settings.tasks for r in range(settings.repeats)
            for f in range(settings.outer_splits)]


@dataclass(frozen=True)
class MethodResult:
    """One method on one outer fold.

    Metrics are not stored. The analysis derives them from y_pred with
    fitness.class_scores, the same function for per-fold and pooled numbers.
    """

    method: str                    # an ALGORITHMS name or a REFERENCES name
    fitness: str                   # a FITNESS_KINDS name, or NO_SEARCH for a reference
    y_pred: dict[str, np.ndarray]  # classifier name -> int[n_test], aligned with FoldResult.test_idx
    selected: np.ndarray           # int[k], indices into the task's raw attributes
    seconds: float                 # wall time to choose the subset; the only field that varies between runs
    best_fitness: float | None = None  # search rows only
    calls: int | None = None           # search rows only
    curve: np.ndarray | None = None    # search rows only, float[100]


@dataclass(frozen=True)
class FoldResult:
    """Everything one outer fold produced. Facts about the fold are stored
    once here, facts about each method in `methods`."""

    key: FoldKey
    class_names: list[str]
    test_idx: np.ndarray         # int[n_test], rows of the task's Dataset
    y_true: np.ndarray           # int[n_test]
    kept: np.ndarray             # int[D], raw attributes that survived FoldPreprocessor on this fold's training rows
    methods: list[MethodResult]  # algorithms x fitness kinds, then REFERENCES

    def to_json(self) -> str:
        """Arrays become lists. Python floats round-trip exactly through json."""
        return json.dumps(asdict(self), default=lambda a: a.tolist())

    @classmethod
    def from_json(cls, text: str) -> "FoldResult":
        d = json.loads(text)
        return cls(
            key=FoldKey(**d["key"]),
            class_names=d["class_names"],
            test_idx=np.array(d["test_idx"], dtype=int),
            y_true=np.array(d["y_true"], dtype=int),
            kept=np.array(d["kept"], dtype=int),
            methods=[MethodResult(**{
                **m,
                "y_pred": {c: np.array(p, dtype=int) for c, p in m["y_pred"].items()},
                "selected": np.array(m["selected"], dtype=int),
                "curve": None if m["curve"] is None else np.array(m["curve"]),
            }) for m in d["methods"]],
        )


def run_fold(key: FoldKey, settings: Settings, X_raw: np.ndarray | None = None) -> FoldResult:
    """Run every method of the grid on one outer fold.

    `X_raw` replaces the dataset's feature values (the leakage check passes
    noise in the outer-test rows); None reads them from the files. The same
    (key, settings, X_raw) gives the same FoldResult on the same machine,
    `seconds` aside. Only the outer-train rows fit the preprocessing, feed the
    fitness and fit models. The outer-test rows are only transformed and predicted.
    """
    data = load_task(key.task)
    X_raw = data.X if X_raw is None else X_raw
    y, class_names = data.y, data.class_names
    train, test = _outer_split(y, key, settings)
    prep = FoldPreprocessor().fit(X_raw[train])
    Xtr, Xte, ytr = prep.transform(X_raw[train]), prep.transform(X_raw[test]), y[train]
    kept = np.flatnonzero(prep.keep_)
    inner = list(StratifiedKFold(settings.inner_splits, shuffle=True, random_state=key.seed()).split(Xtr, ytr))
    X0 = np.random.default_rng(key.seed()).random((settings.population, len(kept)))

    def score(columns: np.ndarray, models: dict[str, _Model]) -> dict[str, np.ndarray]:
        # macOS Accelerate BLAS raises spurious FP flags on every float64 matmul
        # (X @ 0 flags overflow and returns exact zeros), as in arrfs/fitness.py.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return {name: make(key.seed(), settings.neighbours).fit(Xtr[:, columns], ytr).predict(Xte[:, columns])
                    for name, make in models.items()}

    methods = []
    for kind in FITNESS_KINDS:
        for name in settings.algorithms:
            # A fresh fitness per search: a shared cache would make later searches look faster.
            fitness = WrapperFitness(Xtr, ytr, len(class_names), inner, kind, k=settings.neighbours)
            start = time.perf_counter()
            res = search(name, fitness, X0, settings.budget, np.random.default_rng(key.seed(name, kind)))
            seconds = time.perf_counter() - start
            columns = np.flatnonzero(res.mask)
            methods.append(MethodResult(name, kind, score(columns, _CLASSIFIERS), kept[columns], seconds,
                                        res.fitness, res.calls, res.curve))

    k = next(len(m.selected) for m in methods if (m.method, m.fitness) == PROPOSED)
    for name, (choose, models) in _REFERENCES.items():
        start = time.perf_counter()
        columns = choose(Xtr, ytr, k)
        seconds = time.perf_counter() - start
        methods.append(MethodResult(name, NO_SEARCH, score(columns, models), kept[columns], seconds))
    return FoldResult(key, class_names, test, y[test], kept, methods)


def _outer_split(y: np.ndarray, key: FoldKey, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) of fold key.fold in repeat key.repeat of
    RepeatedStratifiedKFold(outer_splits, repeats, random_state=outer_seed).

    Repeat r's splits do not depend on how many repeats follow (check_protocol.py
    checks this), so the smoke run's folds are the full run's repeat 0.
    """
    cv = RepeatedStratifiedKFold(n_splits=settings.outer_splits, n_repeats=settings.repeats,
                                 random_state=settings.outer_seed)
    return next(islice(cv.split(np.zeros(len(y)), y), key.repeat * settings.outer_splits + key.fold, None))


def _manifest(settings: Settings) -> dict:
    sources = sorted(Path(__file__).parent.glob("*.py"))
    digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sources)).hexdigest()
    return json.loads(json.dumps({"settings": asdict(settings), "source_sha256": digest}))


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def prepare_run_dir(out: Path, settings: Settings) -> list[FoldKey]:
    """Make `out` a run directory for exactly these settings and this code,
    and return the folds not yet saved there, in grid order.

    Idempotent. A missing directory is created with its manifest. A matching
    manifest keeps every finished fold (resume). A different manifest (another
    setting, or any arrfs/*.py file edited) renames the old directory to
    <out>.stale-<YYYYmmdd-HHMMSS> and starts empty, so one table never mixes
    two versions of an algorithm. Nothing is deleted. Leftover *.tmp files
    from a crash do not count, because only finished *.json files do.
    """
    manifest = _manifest(settings)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        stale = out.with_name(f"{out.name}.stale-{datetime.now():%Y%m%d-%H%M%S}")
        out.rename(stale)
        print(f"{out} was made with other settings or code; moved it to {stale}")
    (out / "folds").mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_atomic(path, json.dumps(manifest, indent=1))
    done = {p.stem for p in (out / "folds").glob("*.json")}
    return [k for k in fold_keys(settings) if k.name not in done]


def save_fold(result: FoldResult, out: Path) -> None:
    """Write folds/<key.name>.json through a temp file and os.replace, so a
    crash leaves either no file or a complete one. Only the parent process
    calls this; workers never write."""
    _write_atomic(out / "folds" / f"{result.key.name}.json", result.to_json())


def load_run(out: Path) -> tuple[Settings, list[FoldResult]]:
    """Read manifest.json and every saved fold, in grid order. A run that is
    still going loads fine and simply has fewer folds than fold_keys(settings)."""
    raw = json.loads((out / "manifest.json").read_text())["settings"]
    settings = Settings(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    folds = [FoldResult.from_json(p.read_text()) for p in (out / "folds").glob("*.json")]
    grid_order = {key: i for i, key in enumerate(fold_keys(settings))}
    return settings, sorted(folds, key=lambda r: grid_order[r.key])
