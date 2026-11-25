"""The study's datasets and the per-fold preprocessing.

`load_task(task)` returns one of the seven tasks in TASKS as a `Dataset`. The
files come from the UCI Machine Learning Repository: fetch_data.py downloads
each dataset into data/<dataset>/ and records its source in data/SOURCES.txt.
Every loader checks the size the study expects and fails on anything else.

Categorical attributes keep their numeric codes, as the base papers do, so
they are scaled like any other number.

Arrhythmia facts (checked on the raw file):
- 452 records, 279 attributes, label in column 280.
- Labels 11, 12, 13 never occur. Label 8 has 2 records and label 7 has 3.
- Attribute 14 (J angle) is missing in 376 of 452 records.

Everything that learns from data (dropping mostly-missing or constant columns,
median imputation, min-max scaling) is fitted on the training fold only.
"""
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, get_args

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

Task = Literal["arrhythmia-binary", "arrhythmia-multiclass", "lymphography", "dermatology",
               "spect", "statlog-heart", "wdbc"]
TASKS: tuple[Task, ...] = get_args(Task)


@dataclass(frozen=True)
class Dataset:
    """One task's records, as read from its files."""

    X: np.ndarray              # float[n, d], NaN where the file has '?'
    y: np.ndarray              # int[n], class index 0..C-1
    class_names: list[str]     # C names, in class-index order
    feature_names: np.ndarray  # str[d]
    groups: np.ndarray | None  # str[d], the ECG lead of each Arrhythmia attribute ("General" or a lead); None elsewhere


LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

_GENERAL = ["Age", "Sex", "Height", "Weight", "QRS duration", "P-R interval",
            "Q-T interval", "T interval", "P interval", "QRS angle", "T angle",
            "P angle", "QRST angle", "J angle", "Heart rate"]
_WIDTH = ["Q width", "R width", "S width", "R' width", "S' width",
          "Intrinsic deflections", "Ragged R", "Diphasic R", "Ragged P",
          "Diphasic P", "Ragged T", "Diphasic T"]
_AMP = ["JJ amp", "Q amp", "R amp", "S amp", "R' amp", "S' amp", "P amp",
        "T amp", "QRSA", "QRSTA"]

# Names from each dataset's .names or .doc file, with the spelling of the
# dermatology names corrected.
_LYMPHOGRAPHY = ["lymphatics", "block of affere", "bl. of lymph. c", "bl. of lymph. s", "by pass",
                 "extravasates", "regeneration of", "early uptake in", "lym.nodes dimin",
                 "lym.nodes enlar", "changes in lym.", "defect in node", "changes in node",
                 "changes in stru", "special forms", "dislocation of", "exclusion of no",
                 "no. of nodes in"]
_DERMATOLOGY = ["erythema", "scaling", "definite borders", "itching", "koebner phenomenon",
                "polygonal papules", "follicular papules", "oral mucosal involvement",
                "knee and elbow involvement", "scalp involvement", "family history",
                "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate",
                "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis",
                "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges",
                "thinning of the suprapapillary epidermis", "spongiform pustule", "Munro microabscess",
                "focal hypergranulosis", "disappearance of the granular layer",
                "vacuolisation and damage of basal layer", "spongiosis", "saw-tooth appearance of retes",
                "follicular horn plug", "perifollicular parakeratosis",
                "inflammatory mononuclear infiltrate", "band-like infiltrate", "age"]
_HEART = ["age", "sex", "chest pain type", "resting blood pressure", "serum cholesterol",
          "fasting blood sugar > 120 mg/dl", "resting ECG", "maximum heart rate",
          "exercise induced angina", "oldpeak", "slope of peak exercise ST segment",
          "major vessels coloured by fluoroscopy", "thal"]
_WDBC_MEASURES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                  "concavity", "concave points", "symmetry", "fractal dimension"]


def feature_names_and_leads():
    """Return (names, lead) for the 279 attributes; lead is 'General' or a lead name."""
    names, leads = list(_GENERAL), ["General"] * 15
    for lead in LEADS:                      # attributes 16-159: 12 per lead
        names += [f"{lead} {n}" for n in _WIDTH]
        leads += [lead] * 12
    for lead in LEADS:                      # attributes 160-279: 10 per lead
        names += [f"{lead} {n}" for n in _AMP]
        leads += [lead] * 10
    assert len(names) == 279
    return np.array(names), np.array(leads)


def _rows(path: Path, sep: str | None = ",") -> list[list[str]]:
    """The fields of every non-blank line of a text file, split on `sep` (None: on whitespace)."""
    return [line.strip().split(sep) for line in path.read_text().splitlines() if line.strip()]


def _floats(rows: list[list[str]]) -> np.ndarray:
    return np.array([[np.nan if v == "?" else float(v) for v in r] for r in rows])


def load_raw(path=None):
    path = Path(path) if path else DATA_DIR / "arrhythmia" / "arrhythmia.data"
    rows = _rows(path)
    X = _floats([r[:-1] for r in rows])
    y = np.array([int(r[-1]) for r in rows])
    assert X.shape == (452, 279), X.shape
    return X, y


def make_task(y_raw, task):
    """'binary': normal (label 1) vs any arrhythmia.
    'multiclass': the 13 labels present, with 7 and 8 (both premature
    contractions, 3 and 2 records) merged, giving 12 classes."""
    if task == "binary":
        y = (y_raw != 1).astype(int)
        class_names = ["Normal", "Arrhythmia"]
    elif task == "multiclass":
        merged = np.where(y_raw == 8, 7, y_raw)
        labels = np.unique(merged)
        y = np.searchsorted(labels, merged)
        class_names = [f"Class {l}" if l != 7 else "Class 7+8" for l in labels]
    else:
        raise ValueError(task)
    return y, class_names


def _arrhythmia(task: str) -> Dataset:
    X, y_raw = load_raw()
    y, class_names = make_task(y_raw, task)
    names, leads = feature_names_and_leads()
    return Dataset(X, y, class_names, names, leads)


def _labelled(X: np.ndarray, labels: list[str], classes: dict[str, str], feature_names: list[str],
              shape: tuple[int, int]) -> Dataset:
    """A Dataset whose class index is the position of each label among the keys
    of `classes` (file code -> class name). Fails on an unknown label, a missing
    class, or a shape other than `shape`."""
    codes = list(classes)
    y = np.array([codes.index(label) for label in labels])
    assert X.shape == shape == (len(y), len(feature_names)), (X.shape, shape)
    assert np.bincount(y, minlength=len(codes)).min() > 0, "a class has no records"
    return Dataset(X, y, list(classes.values()), np.array(feature_names), None)


def _lymphography() -> Dataset:
    rows = _rows(DATA_DIR / "lymphography" / "lymphography.data")
    return _labelled(_floats([r[1:] for r in rows]), [r[0] for r in rows],
                     {"1": "normal find", "2": "metastases", "3": "malign lymph", "4": "fibrosis"},
                     _LYMPHOGRAPHY, (148, 18))


def _dermatology() -> Dataset:
    rows = _rows(DATA_DIR / "dermatology" / "dermatology.data")
    return _labelled(_floats([r[:-1] for r in rows]), [r[-1] for r in rows],
                     {"1": "psoriasis", "2": "seborrheic dermatitis", "3": "lichen planus",
                      "4": "pityriasis rosea", "5": "chronic dermatitis", "6": "pityriasis rubra pilaris"},
                     _DERMATOLOGY, (366, 34))


def _spect() -> Dataset:
    """SPECT.names calls the two classes normal and abnormal but does not say
    which code is which, so the classes keep their codes."""
    rows = _rows(DATA_DIR / "spect" / "SPECT.train") + _rows(DATA_DIR / "spect" / "SPECT.test")
    return _labelled(_floats([r[1:] for r in rows]), [r[0] for r in rows],
                     {"0": "diagnosis 0", "1": "diagnosis 1"}, [f"F{j}" for j in range(1, 23)], (267, 22))


def _statlog_heart() -> Dataset:
    rows = _rows(DATA_DIR / "statlog-heart" / "heart.dat", None)
    return _labelled(_floats([r[:-1] for r in rows]), [r[-1] for r in rows],
                     {"1": "absence", "2": "presence"}, _HEART, (270, 13))


def _wdbc() -> Dataset:
    rows = _rows(DATA_DIR / "wdbc" / "wdbc.data")
    names = ([f"mean {m}" for m in _WDBC_MEASURES] + [f"{m} SE" for m in _WDBC_MEASURES]
             + [f"worst {m}" for m in _WDBC_MEASURES])
    return _labelled(_floats([r[2:] for r in rows]), [r[1] for r in rows],
                     {"B": "benign", "M": "malignant"}, names, (569, 30))


_LOADERS: dict[Task, Callable[[], Dataset]] = {
    "arrhythmia-binary": partial(_arrhythmia, "binary"),
    "arrhythmia-multiclass": partial(_arrhythmia, "multiclass"),
    "lymphography": _lymphography,
    "dermatology": _dermatology,
    "spect": _spect,
    "statlog-heart": _statlog_heart,
    "wdbc": _wdbc,
}
assert tuple(_LOADERS) == TASKS, "every Task needs one loader, in the same order"


def load_task(task: Task) -> Dataset:
    return _LOADERS[task]()


class FoldPreprocessor:
    """Fit on the training fold, then apply to train and test."""

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X):
        miss = np.isnan(X).mean(0)
        keep = miss <= self.max_missing
        med = np.nanmedian(np.where(keep, X, 0.0), axis=0)
        Xi = np.where(np.isnan(X), med, X)
        lo, hi = Xi.min(0), Xi.max(0)
        keep &= hi > lo                       # drop constant columns
        self.keep_, self.med_, self.lo_, self.rng_ = keep, med, lo, np.where(hi > lo, hi - lo, 1.0)
        return self

    def transform(self, X):
        Xi = np.where(np.isnan(X), self.med_, X)
        Xs = (Xi - self.lo_) / self.rng_
        return Xs[:, self.keep_]

    def fit_transform(self, X):
        return self.fit(X).transform(X)
