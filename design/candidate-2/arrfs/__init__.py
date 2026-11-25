"""arrfs: feature selection for UCI Arrhythmia.

The thread limits below must be set before numpy is imported anywhere. With
macOS "spawn" workers, every worker imports this package first (to unpickle
the job function), so setting them here covers parent and workers alike.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
