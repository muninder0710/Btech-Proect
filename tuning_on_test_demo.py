"""Experiment 3(d): what does tuning on the test patients do to the base paper's pipeline?

Dhanka & Maini 2025 tune XGBoost with PSO and report 95.24% on 42 test patients. The paper's text leaves open
whether the tuning saw those 42 patients. Every split here also keeps 42 patients aside that no step sees.
Shortcut arm: filters fitted on training and test patients together, PSO fitness = accuracy on the test patients.
Fair arm: filters fitted on training patients, PSO fitness = inner 3-fold CV on training patients.
Default arm: filters fitted on training patients, XGBoost with default settings.
Each arm is scored on the 42 test patients and on the 42 untouched patients.

Design: design/amendment-7-tuning-on-test.md.
Run from code/:  .venv/bin/python tuning_on_test_demo.py
"""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"


def _use_bundled_openmp():
    """xgboost on macOS needs libomp. Without Homebrew's copy, restart once with the copy scikit-learn ships."""
    homebrew = any(Path(p, "opt/libomp/lib/libomp.dylib").exists() for p in ("/opt/homebrew", "/usr/local"))
    if sys.platform == "darwin" and not homebrew and "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ:
        import sklearn
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = str(Path(sklearn.__file__).parent / ".dylibs")
        os.execv(sys.executable, [sys.executable, *sys.argv])


_use_bundled_openmp()

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from analyze import _paired  # noqa: E402
from small_split_demo import Steps, load_420  # noqa: E402
from test_reuse_demo import seed_of  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "tuning_on_test"
ARMS = ["shortcut", "fair", "default"]
N_SPLITS, N_TEST, N_NEW = 40, 42, 42
CLAIM_CORRECT = 40  # 40 of 42 is the paper's 95.24%
PARTICLES, ITERATIONS, VMAX = 20, 20, 0.25
N_SETTINGS = 8


def settings(u):
    """XGBoost settings at point u of the unit cube. The ranges are ours; the paper gives none."""
    return dict(n_estimators=int(round(50 + 350 * u[0])), max_depth=int(round(2 + 6 * u[1])),
                learning_rate=float(0.01 * 30 ** u[2]), subsample=float(0.5 + 0.5 * u[3]),
                colsample_bytree=float(0.3 + 0.7 * u[4]), min_child_weight=float(1 + 9 * u[5]),
                gamma=float(5 * u[6]), reg_lambda=float(10 * u[7]))


def xgb(u=None):
    return XGBClassifier(n_jobs=1, random_state=0, **({} if u is None else settings(u)))


def pso(fitness, rng):
    """Standard PSO on the unit cube, maximising `fitness` with PARTICLES x ITERATIONS calls."""
    x = rng.random((PARTICLES, N_SETTINGS))
    v = np.zeros_like(x)
    f = np.array([fitness(p) for p in x])
    best, best_f = x.copy(), f.copy()
    for t in range(1, ITERATIONS):
        leader = best[best_f.argmax()]
        inertia = 0.9 - 0.5 * t / (ITERATIONS - 1)
        v = np.clip(inertia * v + 2 * rng.random(x.shape) * (best - x) + 2 * rng.random(x.shape) * (leader - x),
                    -VMAX, VMAX)
        x = np.clip(x + v, 0, 1)
        f = np.array([fitness(p) for p in x])
        improved = f > best_f
        best[improved], best_f[improved] = x[improved], f[improved]
    return best[best_f.argmax()], float(best_f.max())


def run_split(split):
    """The three arms on one split into 336 training, 42 test and 42 untouched patients."""
    X, y = load_420()
    order = np.random.default_rng(seed_of("3d", split, "split")).permutation(len(y))
    new, test, train = order[:N_NEW], order[N_NEW:N_NEW + N_TEST], order[N_NEW + N_TEST:]
    ytr = y[train]

    def row(arm, steps, u, fitness):
        fitted = xgb(u).fit(steps.transform(X[train]), ytr)
        correct = {name: int((fitted.predict(steps.transform(X[rows])) == y[rows]).sum())
                   for name, rows in [("test", test), ("new", new)]}
        return dict(split=split, arm=arm, correct_test=correct["test"], correct_new=correct["new"],
                    acc_test=correct["test"] / N_TEST, acc_new=correct["new"] / N_NEW, fitness=fitness,
                    **(settings(u) if u is not None else {}))

    fair_steps = Steps().fit(X[train], ytr)
    assert fair_steps.n_fit_ == len(train) == 336
    Xtr = fair_steps.transform(X[train])
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed_of("3d", split, "inner")).split(Xtr, ytr))

    def fair_fitness(u):
        return float(np.mean([(xgb(u).fit(Xtr[a], ytr[a]).predict(Xtr[b]) == ytr[b]).mean() for a, b in inner]))

    seen = np.concatenate([train, test])
    shortcut_steps = Steps().fit(X[seen], y[seen])
    assert shortcut_steps.n_fit_ == 378
    Xtr_s, Xte_s = shortcut_steps.transform(X[train]), shortcut_steps.transform(X[test])

    def shortcut_fitness(u):
        return float((xgb(u).fit(Xtr_s, ytr).predict(Xte_s) == y[test]).mean())

    u_fair, f_fair = pso(fair_fitness, np.random.default_rng(seed_of("3d", split, "pso")))
    u_short, f_short = pso(shortcut_fitness, np.random.default_rng(seed_of("3d", split, "pso")))
    rows = [row("shortcut", shortcut_steps, u_short, f_short), row("fair", fair_steps, u_fair, f_fair),
            row("default", fair_steps, None, np.nan)]
    assert abs(rows[0]["acc_test"] - f_short) < 1e-12, "the printed number is the shortcut fitness"
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        splits = pd.DataFrame([r for rows in pool.map(run_split, range(N_SPLITS)) for r in rows])
    assert len(splits) == N_SPLITS * len(ARMS)
    splits.to_csv(OUT / "tuning_on_test_splits.csv", index=False)

    by_arm = {arm: splits[splits.arm == arm].sort_values("split") for arm in ARMS}
    summary = []
    for arm, d in by_arm.items():
        gap = _paired((d.acc_test - d.acc_new).to_numpy())
        summary.append(dict(arm=arm, acc_test_mean=d.acc_test.mean(), acc_test_sd=d.acc_test.std(),
                            acc_new_mean=d.acc_new.mean(), acc_new_sd=d.acc_new.std(),
                            unseen_mean=np.nan if arm == "shortcut" else ((d.acc_test + d.acc_new) / 2).mean(),
                            share_test_at_least_40_of_42=(d.correct_test >= CLAIM_CORRECT).mean(),
                            best_test=d.acc_test.max(), gap_mean=gap["mean_diff"], gap_ci_low=gap["ci_low"],
                            gap_ci_high=gap["ci_high"]))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "tuning_on_test_summary.csv", index=False)

    printed_minus_fair = _paired(by_arm["shortcut"].acc_test.to_numpy() - by_arm["fair"].acc_test.to_numpy())
    s = summary.set_index("arm")
    claims = pd.DataFrame([
        dict(claim="tuning on the test patients inflates the printed accuracy", value=s.gap_mean["shortcut"],
             ci_low=s.gap_ci_low["shortcut"], ci_high=s.gap_ci_high["shortcut"], holds=bool(s.gap_ci_low["shortcut"] > 0)),
        dict(claim="fair tuning shows a gap between test and new patients", value=s.gap_mean["fair"],
             ci_low=s.gap_ci_low["fair"], ci_high=s.gap_ci_high["fair"],
             holds=bool(s.gap_ci_low["fair"] > 0 or s.gap_ci_high["fair"] < 0)),
        dict(claim="no claim: shortcut minus fair accuracy on the same test patients", value=printed_minus_fair["mean_diff"],
             ci_low=printed_minus_fair["ci_low"], ci_high=printed_minus_fair["ci_high"], holds=np.nan),
    ])
    claims.to_csv(OUT / "tuning_on_test_claims.csv", index=False)

    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(claims.round(4).to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
