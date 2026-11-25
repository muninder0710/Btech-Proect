"""Checks of what_matters_demo.py, to run before the experiment. Exits non-zero on the first failure.

    .venv/bin/python check_what_matters.py

1. Indicator columns are fitted on training rows only, keep the dropped J angle, and equal "value was missing".
2. ReliefF and mutual information rank the one informative column of a toy problem first.
3. mRMR skips an exact copy of the column it already chose; plain mutual information does not.
4. One synthetic fold gives identical rows when run twice.
"""
import numpy as np

import what_matters_demo as wm
from arrfs.data import load_task


def check_indicators():
    X = load_task("arrhythmia-binary").X
    train, test = np.arange(300), np.arange(300, 452)
    prep = wm.Preprocessor(True).fit(X[train])
    plain = wm.Preprocessor(False).fit(X[train])
    assert 14 - 1 in prep.ind_ and not plain.base_.keep_[13], "J angle values dropped, its indicator kept"
    T = prep.transform(X[test])
    assert np.array_equal(T[:, -len(prep.ind_):], np.isnan(X[test][:, prep.ind_]))
    assert np.array_equal(T[:, :-len(prep.ind_)], plain.transform(X[test]))
    trained_on = set(np.flatnonzero(np.isnan(X[train]).any(0)))
    assert set(prep.ind_) <= trained_on, "an indicator exists only where the training rows had a missing value"
    print(f"1. indicators ok: {len(prep.ind_)} columns for raw attributes {list(prep.ind_ + 1)}")


def check_rankers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    X = rng.random((200, 6))
    X[:, 4] = y / 2 + rng.normal(0, 0.05, 200)
    for name in ("ReliefF", "mutual information", "mRMR"):
        assert wm.RANKERS[name](X, y, 0)[0] == 4, name
    Z = np.column_stack([X[:, 4], X[:, 4], (y == 2) + rng.normal(0, 0.2, 200), X[:, :3]])
    assert list(wm.mi_rank(Z, y, 0)[:2]) in ([0, 1], [1, 0]), "plain MI keeps the copy"
    assert 1 not in list(wm.mrmr_rank(Z, y)[:2]), "mRMR must skip the copy"
    print("2-3. ReliefF, mutual information and mRMR ok")


def check_determinism():
    X, y, _ = wm.synthetic("two-class")
    tr, te = np.arange(0, 452, 2)[:200], np.arange(1, 452, 2)[:100]
    a, b = wm.run_fold("two-class", 0, 0, tr, te, True)[0], wm.run_fold("two-class", 0, 0, tr, te, True)[0]
    assert a == b and len(a) == 2 * len(wm.SELECTORS) * len(wm.CLASSIFIERS)
    print(f"4. one synthetic fold repeats exactly ({len(a)} rows)")


if __name__ == "__main__":
    check_indicators()
    check_rankers()
    check_determinism()
    print("all checks passed")
