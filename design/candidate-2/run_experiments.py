"""Run the experiment grid.

    python run_experiments.py --repeats 1 --out results/smoke     # smoke run
    python run_experiments.py --repeats 5 --out results/full      # full run
    (re-run the same command after a crash or Ctrl-C: finished folds are skipped)

Thin shell: parse args, make jobs, skip finished ones, fan out, save. All
science lives in arrfs/experiment.py.
"""
import argparse
from pathlib import Path

import arrfs  # noqa: F401  sets BLAS threads to 1 before numpy loads
from arrfs.experiment import FoldJob, Settings


def check_manifest(out: Path, settings: Settings) -> None:
    """Write out/manifest.json on first use. On later use, raise if it differs
    from `settings`, with a message telling the student to pick a new --out."""
    raise NotImplementedError


def pending_jobs(out: Path, jobs: list[FoldJob]) -> list[FoldJob]:
    """Jobs whose fold file is missing, unreadable, or lacks any key from
    expected_keys(). Complete folds are skipped, which is the whole restart logic."""
    raise NotImplementedError


def _work(job: FoldJob, settings: Settings, out: Path) -> str:
    """Runs in a worker process: load existing file, run_fold, save_fold
    (atomic). Returns a one-line progress string. Each worker writes only its own file."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Flags: --repeats R (default 5), --out DIR (required), --workers N
    (default os.cpu_count()), --tasks binary multiclass.
    ProcessPoolExecutor with mp_context=spawn. A failing fold is printed and
    skipped; the exit code is 1 if any failed, so a rerun retries only those."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
