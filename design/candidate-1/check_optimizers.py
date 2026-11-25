"""Behaviour checks for the five optimisers on a toy problem with a known answer.

    python check_optimizers.py        # a few seconds; exits non-zero on the first failure

The toy fitness is the fraction of positions where a mask differs from a
planted target mask, so the optimum is 0 and progress is easy to read. These
checks do not touch the Arrhythmia data, so a failure points at the optimiser,
not at the data or the KNN.
"""
import numpy as np

from arrfs.optimizers import ALGORITHMS, search


def planted_fitness(target: np.ndarray):
    """mask -> fraction of positions where mask != target; 0 only at the target."""
    raise NotImplementedError


def check_contract(name: str) -> None:
    """On the toy problem (D = 40, N = 20, T = 50): the curve has T + 1 points
    and never rises, curve[-1] equals the returned fitness, the returned mask
    has that fitness, and calls is at least N."""
    raise NotImplementedError


def check_deterministic(name: str) -> None:
    """Two searches with equal X0 and equal seeds return identical masks, curves and calls."""
    raise NotImplementedError


def check_improves(name: str) -> None:
    """The final fitness is below the best fitness of the initial population
    on each of 5 seeds, so an update rule that does not move towards good
    masks fails here before it reaches the real experiment."""
    raise NotImplementedError


if __name__ == "__main__":
    for name in ALGORITHMS:
        check_contract(name)
        check_deterministic(name)
        check_improves(name)
        print(f"{name}: ok")
