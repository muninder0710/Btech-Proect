"""The five search algorithms behind one interface.

Every algorithm is a plain function  (Problem, rng) -> SearchResult  that
minimises Problem.fitness over D binary decisions, and is listed in REGISTRY.
Nothing else in the code base names an algorithm: the experiment grid, the
analysis tables and the plot labels are all derived from REGISTRY.

Invariants every algorithm must keep (checked by _Tracker, not by convention):
- It starts from Problem.X0 (continuous algorithms use it directly, binary
  ones use X0 > 0.5). It never draws its own initial population.
- All randomness comes from the `rng` argument. No global numpy state.
- It never calls fitness on an empty mask on purpose (the fitness penalises it).
- SearchResult.curve has length iters + 1 and is non-increasing.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .fitness import WrapperFitness


@dataclass(frozen=True)
class Problem:
    """One search task: minimise `fitness(mask)` starting from `X0`.

    X0 is (N, D) uniform in [0, 1). The runner builds one X0 per outer fold and
    hands the same array to every algorithm, so differences between algorithms
    are not differences in starting luck. `pop` and `iters` are read from here
    so an algorithm never hard-codes N = 20 or T = 100.
    """
    fitness: WrapperFitness
    X0: np.ndarray
    iters: int

    @property
    def pop(self) -> int: return self.X0.shape[0]

    @property
    def dim(self) -> int: return self.X0.shape[1]


@dataclass(frozen=True)
class SearchResult:
    """What an algorithm hands back. Fitness calls and wall time are measured
    by the runner (single owner), not reported by the algorithm."""
    best_mask: np.ndarray        # bool (D,), best by fitness over everything evaluated
    best_fitness: float
    curve: np.ndarray            # float (iters + 1,), best-so-far; entry 0 is the initial population


Optimizer = Callable[[Problem, np.random.Generator], SearchResult]


@dataclass(frozen=True)
class Algorithm:
    name: str       # stable key used in seeds, file contents and tables ("gwo_cpo")
    label: str      # slide label ("GWO-CPO (proposed)")
    run: Optimizer


REGISTRY: dict[str, Algorithm] = {}
PROPOSED = "gwo_cpo"   # the row every Wilcoxon test compares against


def register(name: str, label: str) -> Callable[[Optimizer], Optimizer]:
    """Decorator that adds an algorithm to REGISTRY. Duplicate names raise."""
    raise NotImplementedError


class _Tracker:
    """Evaluates masks, remembers the best one, and records the curve.

    The only place that touches Problem.fitness during a search, so the
    best-so-far and curve invariants hold for every algorithm without each one
    re-implementing them. Ties keep the first mask found (deterministic).
    """

    def __init__(self, problem: Problem): raise NotImplementedError

    def evaluate(self, mask: np.ndarray) -> float:
        """Fitness of `mask`; updates the best-so-far."""
        raise NotImplementedError

    def end_iteration(self) -> None:
        """Append the current best-so-far to the curve. Call once per iteration
        and once after the initial population (that is entry 0)."""
        raise NotImplementedError

    def result(self) -> SearchResult: raise NotImplementedError


def _decode(x: np.ndarray) -> np.ndarray:
    """Continuous position to mask: x > 0.5."""
    raise NotImplementedError


def _sigmoid10(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-10 (z - 0.5)))."""
    raise NotImplementedError


def _gwo_move(X: np.ndarray, leaders: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    """Mirjalili three-leader step, vectorised over rows of X (n, D) with
    leaders (3, D): mean over leaders of L - A * |C * L - X|, clipped to [0, 1].
    Used by both `gwo` and `gwo_cpo`, so the exploitation step has one definition."""
    raise NotImplementedError


def _cpo_search(problem: Problem, rng: np.random.Generator,
                exploit: Literal["cpo", "gwo"]) -> SearchResult:
    """CPO main loop (grounding section 4). `exploit` picks what the
    non-exploration branch does: CPO's odor / physical attack, or `_gwo_move`
    with leaders from the personal-best memory. Everything else (sight and
    sound exploration, greedy acceptance, cyclic population reduction, random
    reset of out-of-range coordinates, intended +-1 for U2) is shared, so
    `cpo` and `gwo_cpo` differ in exactly one branch by construction."""
    # TODO per iteration t (1-based), active = cyclic_size(t); only X[:active] move
    #   r2 = rng.random(); for i in range(active): draw U1, then explore vs exploit
    #   new position -> reset coords outside [0,1] to uniform -> evaluate decoded mask
    #   greedy: pbest[i], f_pbest[i] updated only if new fitness <= old
    #   leaders (gwo branch) = three best rows of pbest by f_pbest
    #   tracker.end_iteration()
    raise NotImplementedError


@register("bpso", "Binary PSO")
def bpso(problem: Problem, rng: np.random.Generator) -> SearchResult:
    """Kennedy and Eberhart 1997 binary PSO (grounding section 1)."""
    raise NotImplementedError


@register("bpso_gwo", "Binary PSO-GWO (Al-Tashi 2019)")
def bpso_gwo(problem: Problem, rng: np.random.Generator) -> SearchResult:
    """Al-Tashi 2019 equations 16 to 21. Initial velocity 0.3 * N(0, 1) is our
    choice (the paper is silent) and belongs in the method notes."""
    raise NotImplementedError


@register("gwo", "GWO")
def gwo(problem: Problem, rng: np.random.Generator) -> SearchResult:
    """Mirjalili 2014 with threshold decoding. Leaders are best-three-so-far."""
    raise NotImplementedError


@register("cpo", "CPO")
def cpo(problem: Problem, rng: np.random.Generator) -> SearchResult:
    return _cpo_search(problem, rng, exploit="cpo")


@register(PROPOSED, "GWO-CPO (proposed)")
def gwo_cpo(problem: Problem, rng: np.random.Generator) -> SearchResult:
    return _cpo_search(problem, rng, exploit="gwo")
