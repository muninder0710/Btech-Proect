"""Run the experiment grid. Run the same command again to resume after a crash.

    python run_experiments.py --smoke     # 1 repeat  -> results/smoke/
    python run_experiments.py             # 5 repeats -> results/full/

`--out` defaults to results/<smoke|full> under the current directory, so on
Kaggle `cd /kaggle/working` first and the code folder can stay read-only.
"""
import os

# One BLAS thread per process. This must run before numpy is imported, here and
# in every spawned worker, which inherits this environment.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

from arrfs.experiment import (PROPOSED, FoldKey, FoldResult, Settings, fold_keys,  # noqa: E402
                              prepare_run_dir, run_fold, save_fold)
from arrfs.fitness import class_scores  # noqa: E402


def run(settings: Settings, out: Path, workers: int) -> tuple[list[FoldKey], list[FoldKey]]:
    """Run every fold not yet saved in `out` and return (computed, failed) keys.

    Workers only compute (`run_fold`). This process is the only writer of the
    run directory (`save_fold`). A fold that raises is printed and skipped, and
    the next run retries only the missing folds. The pool uses the spawn start
    method on every platform, so macOS and Kaggle's Linux behave the same.
    """
    pending = prepare_run_dir(out, settings)
    print(f"{len(pending)} of {len(fold_keys(settings))} folds to run, {workers} workers -> {out}", flush=True)
    computed, failed = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(run_fold, key, settings): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failed.append(key)
                print(f"FAILED {key.name}: {error!r}", flush=True)
                continue
            save_fold(result, out)
            computed.append(key)
            print(_progress_line(result, len(computed), len(pending), time.perf_counter() - start), flush=True)
    print(f"done: {len(computed)} folds in {time.perf_counter() - start:.0f} s, {len(failed)} failed", flush=True)
    return computed, failed


def _progress_line(result: FoldResult, done: int, total: int, elapsed: float) -> str:
    """For example '[3/10] binary_r0_f2 at 41 s: GWO-CPO / imbalance, KNN macro-F1 0.712, 38 features'."""
    m = next(m for m in result.methods if (m.method, m.fitness) == PROPOSED)
    f1 = class_scores(result.y_true, m.y_pred["KNN"], len(result.class_names))["macro_f1"]
    return (f"[{done}/{total}] {result.key.name} at {elapsed:.0f} s: {m.method} / {m.fitness}, "
            f"KNN macro-F1 {f1:.3f}, {len(m.selected)} features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="1 repeat instead of 5")
    parser.add_argument("--out", type=Path, help="run directory (default results/smoke or results/full)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes (default: all cores)")
    args = parser.parse_args(argv)
    settings = Settings(repeats=1) if args.smoke else Settings()
    out = args.out or Path("results") / ("smoke" if args.smoke else "full")
    _, failed = run(settings, out, args.workers)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
