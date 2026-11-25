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
from pathlib import Path  # noqa: E402

from arrfs.experiment import FoldResult, Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Turn the flags into Settings and an output path, then run every pending
    fold on a process pool and save each one as it finishes.

    Flags: --smoke (Settings(repeats=1)), --out PATH, --workers N (default
    os.cpu_count()). The pool uses the spawn start method on every platform,
    so macOS and Kaggle's Linux behave the same. Workers only compute
    (`run_fold`). This process is the only writer of the run directory
    (`save_fold`). A fold that raises is printed with its key and skipped. The
    exit code is then 1, and the next run retries only the missing folds.
    """
    raise NotImplementedError
    # TODO
    #   pending = prepare_run_dir(out, settings) ; print how many of len(fold_keys(settings)) remain
    #   with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
    #       futures = {pool.submit(run_fold, key, settings): key for key in pending}
    #       for each future as it completes:
    #           on exception: remember the key, print it, continue
    #           save_fold(result, out) ; print(_progress_line(result, done, len(pending)))
    #   print failed keys ; return 1 if any failed else 0


def _progress_line(result: FoldResult, done: int, total: int) -> str:
    """For example '[12/50] multiclass_r2_f0  GWO-CPO/imbalance bacc 0.41  12 features  31 s'.
    The bacc comes from fitness.class_scores, the same code the analysis uses."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
