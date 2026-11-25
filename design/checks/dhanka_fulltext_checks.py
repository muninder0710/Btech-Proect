"""Checks of Dhanka & Maini 2025 (Int. J. Cardiol. 432:133266) against its own full text and our rebuild.

Numbers typed from the paper: Fig. 6 and Fig. 9 (confusion matrices), Fig. 5 legend (the 100 kept attributes).
Run from code/:  .venv/bin/python design/checks/dhanka_fulltext_checks.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from small_split_demo import Steps, load_420

# (TP, TN, FP, FN) as printed under each bar chart.
EXPERIMENT_1 = {"LR": (22, 8, 5, 7), "LDA": (16, 9, 8, 9), "GNB": (22, 13, 3, 4), "DT": (21, 12, 5, 4),
                "XGBoost": (23, 11, 3, 5)}
EXPERIMENT_2 = {"PSO+LR": (24, 12, 2, 4), "PSO+LDA": (24, 13, 2, 3), "PSO+GNB": (25, 14, 1, 2),
                "PSO+DT": (22, 15, 2, 3), "PSO+XGBoost": (26, 14, 1, 1)}
REPORTED_ACC = {"LR": 71.43, "LDA": 59.52, "GNB": 83.33, "DT": 78.57, "XGBoost": 80.95, "PSO+LR": 85.71,
                "PSO+LDA": 88.10, "PSO+GNB": 92.86, "PSO+DT": 88.10, "PSO+XGBoost": 95.24}
REPORTED_SENS = {"LR": 75.86, "LDA": 64.00, "GNB": 84.62, "DT": 84.00, "XGBoost": 82.14, "PSO+LR": 85.71,
                 "PSO+LDA": 88.89, "PSO+GNB": 92.59, "PSO+DT": 88.00, "PSO+XGBoost": 96.30}
FIG5_ATTRIBUTES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 18, 21, 30, 33, 53, 57, 65, 66, 69, 76, 78, 81,
                   88, 90, 91, 93, 100, 103, 105, 112, 113, 114, 117, 124, 126, 136, 138, 141, 148, 153, 160, 161,
                   163, 167, 169, 171, 172, 173, 174, 177, 178, 179, 181, 183, 190, 191, 192, 193, 197, 198, 199,
                   202, 203, 207, 209, 211, 213, 218, 219, 221, 222, 224, 227, 229, 231, 238, 239, 241, 243, 248,
                   249, 251, 253, 257, 258, 259, 261, 263, 267, 268, 269, 270, 271, 273, 277, 279]

print("1. The paper's own confusion matrices")
for name, (tp, tn, fp, fn) in {**EXPERIMENT_1, **EXPERIMENT_2}.items():
    assert tp + tn + fp + fn == 42, name
    assert abs(100 * (tp + tn) / 42 - REPORTED_ACC[name]) < 0.01, name
    assert abs(100 * tp / (tp + fn) - REPORTED_SENS[name]) < 0.01, name  # so positives = TP + FN in the paper's usage
    print(f"   {name:12s} total 42, positives in the test set (TP + FN) = {tp + fn}, negatives (TN + FP) = {tn + fp}")
positives = sorted({tp + fn for tp, tn, fp, fn in {**EXPERIMENT_1, **EXPERIMENT_2}.values()})
print(f"   distinct numbers of positive test patients across the ten models: {positives}")

X, y = load_420()
attribute = np.delete(np.arange(1, 280), 13)  # original attribute number of each of the 278 columns
assert len(FIG5_ATTRIBUTES) == 100 and set(FIG5_ATTRIBUTES) <= set(attribute.tolist())

print("\n2. Our rebuild of the two filters against the 100 attributes in the paper's Fig. 5")
steps = Steps().fit(X, y)
ours = set(attribute[steps.zero_keep_][steps.corr_keep_].tolist())
after_zero_filter = set(attribute[steps.zero_keep_].tolist())
paper = set(FIG5_ATTRIBUTES)
print(f"   attributes in both lists: {len(ours & paper)} of 100")
print(f"   paper's attributes that our zero-count filter had already dropped: {len(paper - after_zero_filter)}")
zeros = (X == 0).mean(0)
print(f"   share of zero values, paper's 100: median {np.median(zeros[np.isin(attribute, FIG5_ATTRIBUTES)]):.2f}, "
      f"max {zeros[np.isin(attribute, FIG5_ATTRIBUTES)].max():.2f}; all 278: median {np.median(zeros):.2f}")
corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.ptp(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])])
in_paper = np.isin(attribute, FIG5_ATTRIBUTES)
print(f"   |Pearson r| with the label, paper's 100: median {np.median(corr[in_paper]):.3f}, min {corr[in_paper].min():.3f}; "
      f"the other 178: median {np.median(corr[~in_paper]):.3f}, max {corr[~in_paper].max():.3f}")

print("\n3. The paper's split as we read it: train_test_split(test_size=0.1, random_state=0), the paper's 100 attributes")
Xp = X[:, in_paper]
Xp = (Xp - Xp.min(0)) / np.where(np.ptp(Xp, axis=0) > 0, np.ptp(Xp, axis=0), 1.0)  # min-max on all 420 rows, as in the paper
for stratify, label in [(None, "unstratified"), (y, "stratified")]:
    tr, te = train_test_split(np.arange(len(y)), test_size=0.1, random_state=0, stratify=stratify)
    print(f"   {label}: test set has {int((y[te] == 0).sum())} normal and {int((y[te] == 1).sum())} arrhythmia patients")
    warnings.simplefilter("ignore", ConvergenceWarning)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for name, model in [("LR", LogisticRegression()), ("LDA", LinearDiscriminantAnalysis()), ("GNB", GaussianNB()),
                            ("DT", DecisionTreeClassifier(random_state=0))]:
            correct = int((model.fit(Xp[tr], y[tr]).predict(Xp[te]) == y[te]).sum())
            print(f"      default {name:4s} {correct}/42 = {100 * correct / 42:.2f}%   (paper: {REPORTED_ACC[name]:.2f}%)")
