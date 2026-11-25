"""Checks of benchmark_functions_demo.py, to run before the experiment.

    python check_benchmark_functions.py      # under a minute; exits non-zero on the first failure

1. Every function is written a second time here, as a literal loop translation of Get_Functions_details.m.
   The two versions must agree on random points, and each must return its published optimum value at its
   published optimum point.
2. A shifted function must return the same optimum at the moved point, which must lie inside the bounds, and
   must no longer return it at the old point.
3. Each optimiser in continuous form must spend exactly its budget, keep every point inside the bounds, give
   the same result twice, improve on its start and beat random search with the same budget. After a run the
   bit flip of arrfs must be back in place.
"""
import math

import numpy as np

from arrfs import optimizers as base
from benchmark_functions_demo import (BUDGET, DIM, FUNCTIONS, OPTIMISERS, POPULATION, Objective, minimise, offset,
                                      seed)


def _u(x: float, a: float, k: float, m: int) -> float:
    return k * (x - a) ** m if x > a else k * (-x - a) ** m if x < -a else 0.0


def _f12(x: list[float]) -> float:
    n = len(x)
    total = 10 * math.sin(math.pi * (1 + (x[0] + 1) / 4)) ** 2
    for i in range(n - 1):
        total += ((x[i] + 1) / 4) ** 2 * (1 + 10 * math.sin(math.pi * (1 + (x[i + 1] + 1) / 4)) ** 2)
    total += ((x[n - 1] + 1) / 4) ** 2
    return (math.pi / n) * total + sum(_u(v, 10, 100, 4) for v in x)


def _f13(x: list[float]) -> float:
    n = len(x)
    total = math.sin(3 * math.pi * x[0]) ** 2
    for i in range(n - 1):
        total += (x[i] - 1) ** 2 * (1 + math.sin(3 * math.pi * x[i + 1]) ** 2)
    total += (x[n - 1] - 1) ** 2 * (1 + math.sin(2 * math.pi * x[n - 1]) ** 2)
    return 0.1 * total + sum(_u(v, 5, 100, 4) for v in x)


LITERAL = {  # the noise of F7 is left out here and in Function.value
    "F1": lambda x: sum(v ** 2 for v in x),
    "F2": lambda x: sum(abs(v) for v in x) + math.prod(abs(v) for v in x),
    "F3": lambda x: sum(sum(x[:i]) ** 2 for i in range(1, len(x) + 1)),
    "F4": lambda x: max(abs(v) for v in x),
    "F5": lambda x: sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)),
    "F6": lambda x: sum(abs(v + 0.5) ** 2 for v in x),
    "F7": lambda x: sum((i + 1) * v ** 4 for i, v in enumerate(x)),
    "F8": lambda x: sum(-v * math.sin(math.sqrt(abs(v))) for v in x),
    "F9": lambda x: sum(v ** 2 - 10 * math.cos(2 * math.pi * v) for v in x) + 10 * len(x),
    "F10": lambda x: (-20 * math.exp(-0.2 * math.sqrt(sum(v ** 2 for v in x) / len(x)))
                      - math.exp(sum(math.cos(2 * math.pi * v) for v in x) / len(x)) + 20 + math.e),
    "F11": lambda x: sum(v ** 2 for v in x) / 4000 - math.prod(math.cos(v / math.sqrt(i + 1)) for i, v in enumerate(x)) + 1,
    "F12": _f12,
    "F13": _f13,
}


def check_functions() -> None:
    rng = np.random.default_rng(0)
    for fn in FUNCTIONS:
        for x in fn.lb + (fn.ub - fn.lb) * rng.random((50, DIM)):
            ours, literal = float(fn.value(x)), LITERAL[fn.name](x.tolist())
            assert math.isclose(ours, literal, rel_tol=1e-9, abs_tol=1e-9), (fn.name, ours, literal)
        best_point = np.full(DIM, float(fn.best_at))
        assert math.isclose(float(fn.value(best_point)), fn.best, abs_tol=1e-2), (fn.name, fn.value(best_point))

        moved = offset(fn, "shifted")
        assert not offset(fn, "published").any() and moved.any() == fn.centred, fn.name
        assert np.all((best_point + moved >= fn.lb) & (best_point + moved <= fn.ub)), fn.name
        shifted = Objective(fn, "shifted", np.random.default_rng(0))
        noise = 1.0 if fn.noisy else 0.0
        assert fn.best <= shifted(best_point + moved) <= fn.best + noise + 1e-2, fn.name
        if fn.centred:
            assert shifted(best_point) > fn.best + noise + 1e-3, f"{fn.name}: the shift did not move the optimum"
        assert shifted(np.full(DIM, np.nan)) == np.inf and shifted.not_finite == 1, fn.name
        print(f"{fn.name:4s} ok  agrees with the literal translation on 50 points; optimum {fn.best:.4f} at "
              f"{fn.best_at:g}; largest shift {np.abs(moved).max():.3g} of half-range {(fn.ub - fn.lb) / 2:g}")


def check_optimisers() -> None:
    real_flip, real_vmax = base._flip, base._PSO_VMAX
    for fn, form in [(FUNCTIONS[0], "published"), (FUNCTIONS[8], "shifted")]:
        for run in range(2):
            X0 = fn.lb + (fn.ub - fn.lb) * np.random.default_rng(seed(fn.name, form, run)).random((POPULATION, DIM))
            scorer = Objective(fn, form, np.random.default_rng(0))
            sample = fn.lb + (fn.ub - fn.lb) * np.random.default_rng([run, 1]).random((BUDGET, DIM))
            random_search = min(scorer(x) for x in sample)
            for name in OPTIMISERS:
                results = []
                for _ in range(2):
                    log: list[float] = []

                    def watched(x: np.ndarray) -> float:
                        assert np.all((x >= fn.lb) & (x <= fn.ub)), f"{name} left the bounds of {fn.name}"
                        log.append(scorer(x))
                        return log[-1]

                    best, curve = minimise(name, watched, X0, fn.lb, fn.ub, BUDGET, np.random.default_rng(seed(fn.name, form, run, name)))
                    assert len(log) == BUDGET, f"{name}: {len(log)} calls"
                    assert np.all(np.diff(curve) <= 0) and curve[-1] == best == min(log), f"{name}: curve or best is wrong"
                    assert curve[0] == min(log[:BUDGET // len(curve)]), f"{name}: first curve point"
                    assert best < min(log[:POPULATION]), f"{name}: no better than its start on {fn.name}"
                    assert best < random_search, f"{name}: {best} is no better than random search {random_search} on {fn.name}"
                    results.append((best, curve))
                assert results[0][0] == results[1][0] and np.array_equal(results[0][1], results[1][1]), f"{name}: not repeatable"
                assert base._flip is real_flip and base._PSO_VMAX == real_vmax, "the bit flip of arrfs was not put back"
                print(f"{name:8s} ok  {fn.name} {form}, run {run}: best {results[0][0]:.3g}, random search {random_search:.3g}")
            assert scorer.not_finite == 0


if __name__ == "__main__":
    check_functions()
    check_optimisers()
