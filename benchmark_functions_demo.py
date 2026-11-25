"""The five optimisers of the main experiment on standard benchmark functions. A side check, not an arrhythmia result.

PSO, GWO, CPO, GWO + PSO and GWO + CPO minimise F1 to F13 in 30 dimensions, the scalable test functions of the
GWO and WOA papers, each as published and with its optimum shifted away from the origin. The optimisers run the
update code of arrfs/optimizers.py unchanged: only its bit flip is replaced by the ordinary position update
(`minimise`). Every optimiser gets the same number of fitness calls and, within a run, the same start.

Written to results/benchmark_functions/:
    benchmark_runs.csv      best value found, one row per function, form, run and optimiser
    benchmark_summary.csv   mean, sd, median and mean rank per function, form and optimiser
    benchmark_ranks.csv     the mean rank over the thirteen functions, per form and optimiser
    benchmark_tests.csv     GWO-CPO against each other optimiser, per function and form, paired over the runs
    benchmark_verdicts.csv  how often GWO-CPO is better, worse or not clearly different, per form and competitor
    benchmark_curves.csv    median best-so-far value at 100 points of the budget

Design: design/amendment-11-benchmark-functions.md. Checks: check_benchmark_functions.py.
Run from code/:  .venv/bin/python benchmark_functions_demo.py     then     .venv/bin/python plot_benchmark_functions.py

    .venv/bin/python benchmark_functions_demo.py 0.1     the sensitivity run of the amendment's addendum: PSO's
                                                         velocity limit as that fraction of the range, written to
                                                         results/benchmark_functions/pso_vmax_0.1/
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, wilcoxon  # noqa: E402

from analyze import holm  # noqa: E402
from arrfs import optimizers as base  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "benchmark_functions"
DIM = 30             # GWO paper, Tables 1 and 2
POPULATION = 30      # 30 agents and 500 iterations in the GWO and WOA papers
BUDGET = 15_000      # fitness calls per run, the same for every optimiser
RUNS = 30
FORMS = ("published", "shifted")
SHIFT_FRACTION = 0.8  # of the half-range, in every dimension
PSO_VMAX = 0.5        # PSO's velocity limit as a fraction of the range: half the range is Vmax = Xmax (Shi & Eberhart 1999)
OPTIMISERS = {"PSO": "BPSO", "GWO": "BGWO", "CPO": "BCPO", "GWO-PSO": "IBGWO4", "GWO-CPO": "GWO-CPO"}  # name here -> arrfs
PROPOSED = "GWO-CPO"
ALPHA = 0.05
_INDEX = np.arange(1, DIM + 1)


def _penalty(x: np.ndarray, a: float, k: float, m: int) -> np.ndarray:
    return k * (x - a) ** m * (x > a) + k * (-x - a) ** m * (x < -a)


def _f12(x: np.ndarray) -> float:
    y = 1 + (x + 1) / 4
    return (np.pi / DIM) * (10 * np.sin(np.pi * y[0]) ** 2
                            + np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
                            + (y[-1] - 1) ** 2) + np.sum(_penalty(x, 10, 100, 4))


def _f13(x: np.ndarray) -> float:
    return 0.1 * (np.sin(3 * np.pi * x[0]) ** 2
                  + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
                  + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)) + np.sum(_penalty(x, 5, 100, 4))


@dataclass(frozen=True)
class Function:
    """One test function as Get_Functions_details.m of the GWO-family code defines it, in DIM dimensions."""

    name: str
    lb: float
    ub: float
    best_at: float                       # the published optimum has this value in every coordinate
    best: float                          # the published optimum value, without the noise of a noisy function
    value: Callable[[np.ndarray], float]
    noisy: bool = False                  # adds uniform noise in [0, 1) to every evaluation
    centred: bool = True                 # the optimum lies at or next to the origin, so the shifted form moves it


FUNCTIONS = (
    Function("F1", -100, 100, 0, 0, lambda x: np.sum(x ** 2)),
    Function("F2", -10, 10, 0, 0, lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x))),
    Function("F3", -100, 100, 0, 0, lambda x: np.sum(np.cumsum(x) ** 2)),
    Function("F4", -100, 100, 0, 0, lambda x: np.max(np.abs(x))),
    Function("F5", -30, 30, 1, 0, lambda x: np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)),
    Function("F6", -100, 100, -0.5, 0, lambda x: np.sum((x + 0.5) ** 2)),
    Function("F7", -1.28, 1.28, 0, 0, lambda x: np.sum(_INDEX * x ** 4), noisy=True),
    Function("F8", -500, 500, 420.9687, -418.9829 * DIM, lambda x: np.sum(-x * np.sin(np.sqrt(np.abs(x)))), centred=False),
    Function("F9", -5.12, 5.12, 0, 0, lambda x: np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * DIM),
    Function("F10", -32, 32, 0, 0, lambda x: (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / DIM))
                                               - np.exp(np.sum(np.cos(2 * np.pi * x)) / DIM) + 20 + np.e)),
    Function("F11", -600, 600, 0, 0, lambda x: np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(_INDEX))) + 1),
    Function("F12", -50, 50, -1, 0, _f12),
    Function("F13", -50, 50, 1, 0, _f13),
)
_BY_NAME = {fn.name: fn for fn in FUNCTIONS}


def seed(*parts: object) -> int:
    """Stable 32-bit seed from the parts through sha256, as arrfs.experiment.FoldKey.seed builds its seeds."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def offset(fn: Function, form: str) -> np.ndarray:
    """How far the optimum is moved in this form: nothing as published, and nothing for a function that is not centred."""
    if form == "published" or not fn.centred:
        return np.zeros(DIM)
    return np.random.default_rng(seed(fn.name, "shift")).uniform(-SHIFT_FRACTION, SHIFT_FRACTION, DIM) * (fn.ub - fn.lb) / 2


class Objective:
    """A function in one form, as an optimiser calls it. A proposal that is not a finite point scores +infinity
    and is counted, so that one overflow inside an optimiser cannot poison a run."""

    def __init__(self, fn: Function, form: str, noise: np.random.Generator):
        self.fn, self.offset, self.noise, self.not_finite = fn, offset(fn, form), noise, 0

    def __call__(self, x: np.ndarray) -> float:
        if not np.isfinite(x).all():
            self.not_finite += 1
            return np.inf
        value = float(self.fn.value(x - self.offset))
        return value + self.noise.random() if self.fn.noisy else value


def minimise(name: str, f: Callable[[np.ndarray], float], X0: np.ndarray, lb: float, ub: float, budget: int,
             rng: np.random.Generator, pso_vmax: float = PSO_VMAX) -> tuple[float, np.ndarray]:
    """Run one arrfs optimiser on real vectors from the population X0: (best value found, best-so-far curve).

    arrfs's update code proposes a step and hands it to `_flip`. For the length of this call `_flip` is the
    ordinary position update, X + step kept inside [lb, ub], and PSO's velocity limit is `pso_vmax` of the range.
    """
    flip, vmax = base._flip, base._PSO_VMAX
    base._flip = lambda X, step, rng: np.clip(X + step, lb, ub)
    base._PSO_VMAX = pso_vmax * (ub - lb)
    rec = base._Recorder(f, budget)
    try:
        with np.errstate(over="ignore", invalid="ignore"):  # CPO's weight exp(f_i / sum f) can overflow when f changes sign
            base._ALGORITHMS[OPTIMISERS[name]](rec, X0.copy(), rng)
    except base._BudgetSpent:
        pass
    finally:
        base._flip, base._PSO_VMAX = flip, vmax
    assert rec.calls == budget and len(rec.curve) == base.CURVE_POINTS, (name, rec.calls)
    return rec.best_fitness, np.array(rec.curve)


def run_task(task: tuple[str, str, int, float]) -> list[dict]:
    """One run of one function in one form: every optimiser from the same start."""
    name, form, run, pso_vmax = task
    fn = _BY_NAME[name]
    X0 = fn.lb + (fn.ub - fn.lb) * np.random.default_rng(seed(name, form, run)).random((POPULATION, DIM))
    rows = []
    for method in OPTIMISERS:
        f = Objective(fn, form, np.random.default_rng(seed(name, form, run, method, "noise")))
        best, curve = minimise(method, f, X0, fn.lb, fn.ub, BUDGET, np.random.default_rng(seed(name, form, run, method)),
                               pso_vmax)
        rows.append(dict(function=name, form=form, run=run, method=method, best=best, not_finite=f.not_finite, curve=curve))
    return rows


def compare(runs: pd.DataFrame) -> pd.DataFrame:
    """PROPOSED against every other optimiser, per form and function: paired Wilcoxon signed-rank over the runs,
    Holm-corrected over the functions of one form and competitor."""
    wide = runs.pivot(index=["form", "function", "run"], columns="method", values="best")
    tests = []
    for competitor in [m for m in OPTIMISERS if m != PROPOSED]:
        for (form, function), block in wide.groupby(level=["form", "function"], sort=False):
            d = (block[PROPOSED] - block[competitor]).to_numpy()
            moved = d[d != 0]
            p = wilcoxon(moved).pvalue if len(moved) else 1.0
            lower = np.sum(np.sign(d) * rankdata(np.abs(d))) < 0
            tests.append(dict(form=form, function=function, competitor=competitor, lower_in=int((d < 0).sum()),
                              equal_in=int((d == 0).sum()), higher_in=int((d > 0).sum()), p=p, lower=lower))
    tests = pd.DataFrame(tests)
    for _, block in tests.groupby(["form", "competitor"]):
        tests.loc[block.index, "p_holm"] = holm(block.p.to_numpy())
    clear = tests.p_holm < ALPHA
    tests["verdict"] = np.where(~clear, "no clear difference", np.where(tests.lower, "better", "worse"))
    return tests.drop(columns="lower")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        print(__doc__)
        return 2
    pso_vmax = float(argv[0]) if argv else PSO_VMAX
    out = OUT if pso_vmax == PSO_VMAX else OUT / f"pso_vmax_{pso_vmax:g}"
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tasks = [(fn.name, form, run, pso_vmax) for fn in FUNCTIONS for form in FORMS for run in range(RUNS)]
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        rows = [row for task_rows in pool.map(run_task, tasks, chunksize=4) for row in task_rows]
    runs = pd.DataFrame(rows)
    assert len(runs) == len(FUNCTIONS) * len(FORMS) * RUNS * len(OPTIMISERS) and np.isfinite(runs.best).all()

    cell = ["function", "form", "method"]
    curves = pd.DataFrame([dict(zip(cell, key), point=i + 1, best=value)
                           for key, block in runs.groupby(cell, sort=False)
                           for i, value in enumerate(np.median(np.stack(block.curve.to_list()), axis=0))])
    curves.to_csv(out / "benchmark_curves.csv", index=False)
    runs = runs.drop(columns="curve")
    runs["rank"] = runs.groupby(["function", "form", "run"]).best.rank(method="average")
    runs.to_csv(out / "benchmark_runs.csv", index=False)

    summary = runs.groupby(cell, sort=False).agg(mean=("best", "mean"), sd=("best", "std"), median=("best", "median"),
                                                 mean_rank=("rank", "mean"), not_finite=("not_finite", "sum")).reset_index()
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    ranks = summary.pivot_table(index="method", columns="form", values="mean_rank", sort=False)[list(FORMS)]
    ranks.to_csv(out / "benchmark_ranks.csv")
    tests = compare(runs)
    tests.to_csv(out / "benchmark_tests.csv", index=False)
    verdicts = (tests.groupby(["form", "competitor", "verdict"], sort=False).size().unstack(fill_value=0)
                .reindex(columns=["better", "worse", "no clear difference"], fill_value=0))
    verdicts.to_csv(out / "benchmark_verdicts.csv")

    pd.set_option("display.width", 200)
    print(f"PSO velocity limit: {pso_vmax:g} of the range. Files in {out}")
    print(f"mean rank over the {len(FUNCTIONS)} functions (1 is best of {len(OPTIMISERS)})")
    print(ranks.round(2).to_string())
    print(f"\n{PROPOSED} against each other optimiser: number of functions, paired Wilcoxon with Holm at {ALPHA}")
    print(verdicts.to_string())
    lost = summary[summary.not_finite > 0]
    if len(lost):
        print("\nproposals that were not finite points and scored +infinity")
        print(lost[cell + ["not_finite"]].to_string(index=False))
    print(f"done in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
