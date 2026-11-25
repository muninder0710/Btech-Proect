"""Wrapper fitness: KNN (k=5) with inner stratified cross-validation.

The KNN is written in numpy so that one fitness call takes about a
millisecond. Its predictions match scikit-learn's KNeighborsClassifier
(uniform weights, Euclidean distance, a vote tie going to the smaller class
index) when the 5th and 6th neighbours are not tied.

Two fitness functions:
- "accuracy" (base papers, Emary 2016 / Al-Tashi 2019 / Liu & Tian 2025):
      F = 0.99 * (1 - accuracy) + 0.01 * |S| / D
- "imbalance" (proposed):
      F = a * (1 - macro-F1) + b * (1 - balanced accuracy) + g * |S| / D
  Worst-class recall was tried first. On the 12-class task it was 0 for every
  random subset (classes with 3-4 training patients are never recovered by
  5-NN), so it gave the search no signal.
"""
import numpy as np

EMPTY_PENALTY = 2.0  # worse than any real fitness value


def class_scores(y, yhat, n_classes):
    """Accuracy, balanced accuracy, macro-F1 and per-class recall (classes present in y)."""
    cm = np.bincount(y * n_classes + yhat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred = cm.sum(0).astype(float)
    present = support > 0
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "acc": tp.sum() / len(y),
        "bacc": rec[present].mean(),
        "macro_f1": f1[present].mean(),
        "worst_recall": rec[present].min(),
        "recall": rec,
    }


class WrapperFitness:
    def __init__(self, X, y, n_classes, inner_folds, kind="accuracy", k=5,
                 alpha=0.7, beta=0.3, gamma=0.01):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.C = n_classes
        self.folds = [(np.asarray(tr), np.asarray(va)) for tr, va in inner_folds]
        self.kind, self.k = kind, k
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.D = X.shape[1]
        self.cache = {}
        self.calls = 0

    def inner_predict(self, mask):
        Xs = self.X[:, mask]
        yhat = np.empty_like(self.y)
        sq = (Xs * Xs).sum(1)
        for tr, va in self.folds:
            # macOS Accelerate BLAS raises spurious FP flags in float64 matmul.
            # Predictions were checked to match scikit-learn exactly.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                d = sq[va][:, None] + sq[tr][None, :] - 2.0 * (Xs[va] @ Xs[tr].T)
            nn = np.argpartition(d, self.k - 1, axis=1)[:, : self.k]
            lab = self.y[tr][nn]
            counts = np.stack([(lab == c).sum(1) for c in range(self.C)], axis=1)
            yhat[va] = counts.argmax(1)
        return yhat

    def __call__(self, mask):
        mask = np.asarray(mask, dtype=bool)
        self.calls += 1
        key = mask.tobytes()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        n_sel = int(mask.sum())
        if n_sel == 0:
            f = EMPTY_PENALTY
        else:
            s = class_scores(self.y, self.inner_predict(mask), self.C)
            size = n_sel / self.D
            if self.kind == "accuracy":
                f = 0.99 * (1 - s["acc"]) + 0.01 * size
            elif self.kind == "imbalance":
                f = (self.alpha * (1 - s["macro_f1"]) + self.beta * (1 - s["bacc"])
                     + self.gamma * size)
            else:
                raise ValueError(self.kind)
        self.cache[key] = f
        return f
