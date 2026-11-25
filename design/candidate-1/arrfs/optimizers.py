"""Five feature-selection optimisers behind one function, `search`.

Every algorithm minimises a fitness over D yes/no decisions (keep attribute d
or not). In one outer fold all algorithms start from the same population
X0 = rng.random((N, D)). The binary algorithms (BPSO, BPSO-GWO) start from
X0 > 0.5. The continuous ones (GWO, CPO, GWO-CPO) move positions inside [0, 1]
and evaluate the mask x > 0.5.

Iteration counting is the same for all five. Evaluating the initial population
is iteration 0, then T iterations follow, and each one ends with
`rec.end_iteration()`. So every convergence curve has T + 1 points and the
curves of different algorithms can be averaged and plotted together.

To add an algorithm, write one function with the `_Algorithm` signature and
add one line to `_ALGORITHMS`. The runner's grid and the analysis tables take
their names from `ALGORITHMS`, so nothing else changes.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

Mask = np.ndarray                    # bool[D]
Objective = Callable[[Mask], float]  # lower is better; arrfs.fitness.WrapperFitness is one

_BPSO_C = 2.0          # c1 = c2
_BPSO_VMAX = 6.0
_BPSO_W = (0.9, 0.4)   # inertia at the first and at the last iteration
_HYBRID_C = 0.5        # BPSO-GWO: C, and c1 = c2 = c3
_HYBRID_V0_SD = 0.3    # BPSO-GWO initial velocity sd; not given in the paper
_CPO_MIN_FRACTION = 0.8   # N_min = round(0.8 N)
_CPO_CYCLES = 2
_CPO_ALPHA = 0.2
_CPO_TF = 0.8          # probability of the odor defence inside exploitation


@dataclass(frozen=True)
class SearchResult:
    """What one search returns. `_Recorder` fills every field, never the
    algorithm, so each field means the same thing for all five algorithms."""

    mask: Mask         # best mask evaluated during the search; never empty
    fitness: float     # fitness of `mask`
    curve: np.ndarray  # float[T + 1]: best fitness so far after each iteration, non-increasing
    calls: int         # fitness calls requested, cache hits included


def search(name: str, fitness: Objective, X0: np.ndarray, T: int,
           rng: np.random.Generator) -> SearchResult:
    """Run algorithm `name` for T iterations from X0 (float[N, D] in [0, 1)).

    The result depends only on (name, the fitness values, X0, T, rng state).
    Raises KeyError for a name not in ALGORITHMS. Asserts that the algorithm
    ended exactly T + 1 iterations and that the best mask is not empty, so an
    algorithm that miscounts fails here instead of producing a short curve.
    """
    raise NotImplementedError


@dataclass
class _Recorder:
    """The objective each algorithm calls in place of the raw fitness.

    It is the only owner of the best mask so far, the curve and the call
    count. Algorithms keep their own memories (pbest, leaders, the porcupine
    memory) because their equations need them, but they never report results.
    """

    fitness: Objective
    best_mask: Mask | None = None
    best_fitness: float = np.inf
    curve: list[float] = field(default_factory=list)
    calls: int = 0

    def __call__(self, mask: Mask) -> float:
        """Evaluate `mask` and return its fitness. Only a strictly lower
        fitness replaces the best, so of two equal masks the earlier is kept."""
        raise NotImplementedError

    def end_iteration(self) -> None:
        """Append the best fitness so far to the curve."""
        raise NotImplementedError


_Algorithm = Callable[[_Recorder, np.ndarray, int, np.random.Generator], None]


def sigmoid10(z: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-10 (z - 0.5))), the transfer function of Al-Tashi 2019."""
    raise NotImplementedError


def _decode(x: np.ndarray) -> Mask:
    """Continuous position in [0, 1] to mask: keep attribute d when x[d] > 0.5."""
    raise NotImplementedError


def _best_three(leaders: np.ndarray, leader_fitness: np.ndarray,
                X: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the three lowest-fitness distinct positions among the current
    leaders (shape (0, D) before the first call) and a newly evaluated
    population X with fitness f.

    The sort is stable with the current leaders first, so an earlier find wins
    a tie. Distinctness matters for BPSO-GWO, whose wolves often land on the
    same mask. Without it alpha, beta and delta can be one point, and the
    three-leader average turns into a pull towards a single leader.
    """
    raise NotImplementedError


def _gwo_move(x: np.ndarray, leaders: np.ndarray, a: float,
              rng: np.random.Generator) -> np.ndarray:
    """One GWO position update for one wolf x (Mirjalili et al. 2014).

    For each of the three leaders L, with fresh vectors r1, r2 = rand(D):
    A = 2 a r1 - a, C = 2 r2, X_L' = X_L - A |C X_L - x|. Returns the mean of
    the three X_L'. The result is not bounded: GWO clips it to [0, 1], while
    the CPO family applies its own random-reset rule. Shared by `_gwo` and
    `_gwo_hunt`, so plain GWO and GWO-CPO exploit with identical code.
    """
    raise NotImplementedError


def _bpso(rec: _Recorder, X0: np.ndarray, T: int, rng: np.random.Generator) -> None:
    """Binary PSO (Kennedy and Eberhart 1997).

    Positions are masks starting at X0 > 0.5, and velocities start at 0.
    v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), with w falling linearly
    from 0.9 at the first iteration to 0.4 at the last, c1 = c2 = 2, v clipped
    to [-6, 6], and x = rand(D) < 1 / (1 + exp(-v)). pbest and gbest are the
    best masks found so far.
    """
    raise NotImplementedError


def _bpso_gwo(rec: _Recorder, X0: np.ndarray, T: int, rng: np.random.Generator) -> None:
    """Binary PSO-GWO, the method base paper (Al-Tashi et al. 2019, eqs. 16 to 21).

    Positions are masks starting at X0 > 0.5. Velocities start at
    0.3 N(0, 1), which the paper does not specify (stated in the method notes).
    Leaders alpha, beta, delta come from `_best_three`.
    """
    raise NotImplementedError
    # TODO per iteration t = 0 .. T-1
    #   a = 2 - 2 t / T ; w = 0.5 + rand() / 2              scalars, once per iteration
    #   for each wolf X (bool[D]):
    #       for each leader L in (alpha, beta, delta):
    #           A = 2 a r - a with r = rand(D), fresh per leader ; C = 0.5
    #           D_L   = |C X_L - w X|
    #           bstep = sigmoid10(A D_L) >= rand(D)
    #           x_L   = (X_L + bstep) >= 1                    a logical OR
    #       v = w (v + c r1 (x_alpha - X) + c r2 (x_beta - X) + c r3 (x_delta - X)), c = 0.5
    #       X = sigmoid10((x_alpha + x_beta + x_delta) / 3 + v) >= rand(D)
    #   evaluate every wolf, merge into the leaders with _best_three, rec.end_iteration()


def _gwo(rec: _Recorder, X0: np.ndarray, T: int, rng: np.random.Generator) -> None:
    """GWO with threshold decoding (Mirjalili et al. 2014).

    Continuous positions in [0, 1] starting at X0. Leaders are the three best
    distinct positions found so far (`_best_three`). In iteration
    t = 0 .. T-1, a = 2 - 2 t / T and every wolf moves to
    clip(_gwo_move(x, leaders, a), 0, 1).
    """
    raise NotImplementedError


@dataclass(frozen=True)
class _CpoStep:
    """Per-iteration values that an exploitation step may read."""

    t: int         # 1 .. T, 1-based as in the CPO MATLAB code
    T: int
    r2: float      # drawn once per iteration, used by the physical attack
    n_active: int  # porcupines that move this iteration; random partners X_r come from these


_Exploit = Callable[[int, np.ndarray, np.ndarray, _CpoStep, np.random.Generator], np.ndarray]


def _cpo_family(rec: _Recorder, X0: np.ndarray, T: int, rng: np.random.Generator,
                exploit: _Exploit) -> None:
    """Crested Porcupine Optimizer (Abdel-Basset et al. 2024; checked against
    porcupy 0.2.0) with the exploitation step passed in. CPO passes
    `_odor_or_attack` and GWO-CPO passes `_gwo_hunt`.

    Shared by both: exploration (sight and sound defences), greedy acceptance,
    the random reset of coordinates outside [0, 1], and the cyclic population
    reduction from N to N_min = round(0.8 N), two cycles over T.

    Memory: X (float[N, D]) and f (float[N]) hold each porcupine's personal
    best. Under greedy acceptance a porcupine's current position equals its
    personal best after its turn, so one array is enough, and the global best
    is always X[argmin f]. Neither is stored twice.
    """
    raise NotImplementedError
    # TODO
    #   f = [rec(_decode(x)) for x in X] ; rec.end_iteration()
    #   n_active = N
    #   for t = 1 .. T:
    #       step = _CpoStep(t, T, r2=rand(), n_active)
    #       for i in 0 .. n_active-1:            sequential: porcupine i sees the updates of 0 .. i-1
    #           g = X[argmin f]                   live global best
    #           if rand() < rand():               exploration
    #               every X_r below is a fresh random row among the first n_active
    #               if rand() < rand():           sight
    #                   y = (X_i + X_r) / 2 ; new = X_i + randn() |2 rand() g - y|
    #               else:                         sound
    #                   U1 = rand(D) > rand() ; y = (X_i + X_r) / 2
    #                   new = U1 X_i + (1 - U1)(y + rand() (X_r1 - X_r2))
    #           else:
    #               new = exploit(i, X, f, step, rng)
    #           each coordinate of new outside [0, 1] becomes rand()
    #           nf = rec(_decode(new))
    #           if nf <= f[i]: X[i], f[i] = new, nf          keep the old one only if the new one is worse
    #       n_active = floor(N_min + (N - N_min)(1 - (t mod (T / cycles)) / (T / cycles)))
    #       rec.end_iteration()


def _odor_or_attack(i: int, X: np.ndarray, f: np.ndarray, step: _CpoStep,
                    rng: np.random.Generator) -> np.ndarray:
    """CPO's exploitation: the third (odor) and fourth (physical attack) defences.

    Yt = 2 rand() (1 - t/T)^(t/T), U2 = +1 or -1 per dimension, S = rand() U2,
    weight = exp(f_i / (sum of all N memories' f + eps)), g = X[argmin f].
    With probability Tf, odor: U1 = rand(D) > rand(),
        new = (1 - U1) X_i + U1 (X_r1 + weight (X_r2 - X_r3) - S Yt weight).
    Otherwise physical attack: Ft = rand(D) weight (X_r - X_i),
        new = g + (alpha (1 - r2) + r2)(U2 g - X_i) - S Yt Ft.
    The MATLAB line `U2=rand(1,dim)<0.5*2-1` is all zeros by operator
    precedence. This uses the intended +1 or -1, as porcupy does.
    """
    raise NotImplementedError


def _gwo_hunt(i: int, X: np.ndarray, f: np.ndarray, step: _CpoStep,
              rng: np.random.Generator) -> np.ndarray:
    """GWO-CPO's exploitation, in place of `_odor_or_attack`.

    `_gwo_move(X_i, leaders, a)` with a = 2 - 2 t / T and leaders = the three
    lowest-f rows of the whole porcupine memory, read when porcupine i moves,
    so they are as current as CPO's global best. No clipping here: the shared
    CPO loop applies its random reset, exactly as for the other branches.
    """
    raise NotImplementedError


_ALGORITHMS: dict[str, _Algorithm] = {
    "BPSO": _bpso,
    "BPSO-GWO": _bpso_gwo,
    "GWO": _gwo,
    "CPO": partial(_cpo_family, exploit=_odor_or_attack),
    "GWO-CPO": partial(_cpo_family, exploit=_gwo_hunt),
}
ALGORITHMS: tuple[str, ...] = tuple(_ALGORITHMS)
