"""Five binary feature-selection optimisers behind one function, `search`.

Every algorithm minimises a fitness over D yes/no decisions (keep attribute d
or not) and gets exactly the same number of fitness calls, `budget`. In one
outer fold all five start from the same N masks, X0 > 0.5, where the fold
shares X0 = rng.random((N, D)).

Positions are masks. Every operator proposes a continuous position Y from the
current mask X, and bit d flips with probability |tanh(Y_d - X_d)| (`_flip`).
This is the V-shaped transfer function that Khaseeb et al. 2025 found best,
applied as Mirjalili and Lewis 2013 apply it. For PSO, Y - X is the velocity.
One rule for all five means the comparison is between update rules, not
between binarisations.

Algorithms hold no budget logic and no result bookkeeping. They loop forever
and evaluate through a `_Recorder`, which stops them when the budget is spent.
Every schedule that a paper writes with t / T reads `rec.progress` instead
(calls / budget). The CPO paper itself counts t in function evaluations
(Abdel-Basset et al. 2024, Eq. 3 and Algorithm 1), so for CPO this is the
paper's own clock.

To add an algorithm, write one function `(rec, X, rng) -> None` and add one
entry to `_ALGORITHMS`. The grid, the analysis and check_optimizers.py read
`ALGORITHMS`, so nothing else changes.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Mask = np.ndarray                    # bool[D]
Objective = Callable[[Mask], float]  # lower is better; arrfs.fitness.WrapperFitness is one

CURVE_POINTS = 100

_PSO_C = 2.0               # c1 = c2
_PSO_VMAX = 6.0
_PSO_W = (0.9, 0.4)        # inertia at the start and at the end of the budget
# The settings recommended by Abdel-Basset et al. 2024, Table 2.
_CPO_ALPHA = 0.1           # alpha, the convergence rate of the physical attack
_CPO_TF = 0.5              # Tf, the probability that exploitation uses odor
_CPO_CYCLES = 2            # T, the cycles of population reduction
_CPO_MIN_FRACTION = 2 / 3  # N_min = round(2/3 N), as N_min = 80 of N' = 120


@dataclass(frozen=True)
class SearchResult:
    """What one search returns. `_Recorder` fills every field, never the
    algorithm, so each field means the same thing for all five algorithms."""

    mask: Mask         # best mask evaluated; the earliest one on a tie
    fitness: float     # fitness of `mask`, the minimum over every call
    curve: np.ndarray  # float[CURVE_POINTS]: best so far after each budget/CURVE_POINTS calls
    calls: int         # fitness calls requested, cache hits included; always == budget


class _BudgetSpent(Exception):
    pass


class _Recorder:
    """The objective each algorithm calls in place of the raw fitness.

    The only owner of the best mask, the curve, the call count and the budget.
    Algorithms keep their own memories (pbest, leaders, the porcupine memory)
    because their equations need them, but they never report results.
    """

    def __init__(self, fitness: Objective, budget: int):
        self.fitness = fitness
        self.budget = budget
        self.calls = 0
        self.best_mask: Mask | None = None
        self.best_fitness = np.inf
        self.curve: list[float] = []

    @property
    def progress(self) -> float:
        """calls / budget, in [0, 1]."""
        return self.calls / self.budget

    def __call__(self, mask: Mask) -> float:
        if self.calls == self.budget:
            raise _BudgetSpent
        f = self.fitness(mask)
        self.calls += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        if self.calls % (self.budget // CURVE_POINTS) == 0:
            self.curve.append(self.best_fitness)
        return f


_Algorithm = Callable[[_Recorder, np.ndarray, np.random.Generator], None]


def search(name: str, fitness: Objective, X0: np.ndarray, budget: int,
           rng: np.random.Generator) -> SearchResult:
    """Run algorithm `name` from the masks X0 > 0.5 (X0 float[N, D] in [0, 1))
    for exactly `budget` fitness calls, and return the best mask it evaluated.

    The result depends only on (name, the fitness values, X0, budget, rng state).
    """
    assert budget % CURVE_POINTS == 0 and budget >= len(X0), budget
    rec = _Recorder(fitness, budget)
    try:
        _ALGORITHMS[name](rec, X0 > 0.5, rng)
    except _BudgetSpent:
        pass
    assert rec.calls == budget and len(rec.curve) == CURVE_POINTS, (name, rec.calls)
    return SearchResult(rec.best_mask, rec.best_fitness, np.array(rec.curve), rec.calls)


def _evaluate(rec: _Recorder, X: np.ndarray) -> np.ndarray:
    return np.array([rec(x) for x in X])


def _flip(X: np.ndarray, step: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The shared V-shaped binarisation, for one mask or a population of masks.

    Bit d of X flips with probability |tanh(step_d)|, where step = Y - X for
    the operator's proposed continuous position Y. For PSO the step is the velocity.
    """
    return X ^ (rng.random(X.shape) < np.abs(np.tanh(step)))


def _best_three(P: np.ndarray, pf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The three lowest-fitness distinct rows of P, and their fitness.

    The sort is stable, so with the current leaders stacked first an earlier
    find wins a tie. Binary wolves often land on the same mask: without
    distinctness alpha, beta and delta can be one mask, and the three-leader
    mean becomes a pull towards a single leader. Fewer than three rows come
    back when P holds fewer distinct masks.
    """
    picked: list[int] = []
    for j in np.argsort(pf, kind="stable"):
        if not any(np.array_equal(P[j], P[k]) for k in picked):
            picked.append(j)
            if len(picked) == 3:
                break
    return P[picked], pf[picked]


def _keep_best(P: int, *populations: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The P lowest-fitness masks of the union of (masks, fitness) populations,
    and their fitness. The sort is stable, so on a tie the earlier population
    and the earlier row win. Duplicate masks are kept."""
    X = np.vstack([X for X, _ in populations])
    f = np.concatenate([f for _, f in populations])
    keep = np.argsort(f, kind="stable")[:P]
    return X[keep], f[keep]


def _update_pbest(Pb: np.ndarray, pbf: np.ndarray, X: np.ndarray, f: np.ndarray) -> None:
    """Replace each slot's personal best by its new mask where that is strictly better."""
    better = f < pbf
    Pb[better], pbf[better] = X[better], f[better]


def _gwo_step(X: np.ndarray, leaders: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    """One GWO move of every wolf (Mirjalili et al. 2014), binarised by `_flip`.

    For each leader L, with fresh r1, r2 = rand(N, D): A = 2 a r1 - a,
    C = 2 r2 and X_L = L - A |C L - X|. Y is the mean of the leaders' X_L.
    """
    Y = np.mean([L - (2 * a * rng.random(X.shape) - a) * np.abs(2 * rng.random(X.shape) * L - X)
                 for L in leaders], axis=0)
    return _flip(X, Y - X, rng)


def _pso_step(X: np.ndarray, V: np.ndarray, Pb: np.ndarray, g: Mask, progress: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One binary PSO move of every particle (Kennedy and Eberhart 1997),
    binarised by `_flip` with the velocity as the step. Returns (masks, velocities).

    V = w V + c1 r1 (Pb - X) + c2 r2 (g - X), with w falling linearly from 0.9
    to 0.4 over the budget, c1 = c2 = 2 and V clipped to [-6, 6].
    """
    w = _PSO_W[0] - (_PSO_W[0] - _PSO_W[1]) * progress
    x = X.astype(float)
    V = np.clip(w * V + _PSO_C * rng.random(X.shape) * (Pb - x) + _PSO_C * rng.random(X.shape) * (g - x),
                -_PSO_VMAX, _PSO_VMAX)
    return _flip(X, V, rng), V


def _cpo_move(i: int, X: np.ndarray, f: np.ndarray, g: Mask, progress: float, tau4: float,
              rng: np.random.Generator) -> np.ndarray:
    """Porcupine i's proposed position Y under the four defences of CPO
    (Abdel-Basset et al. 2024). The caller binarises it with `_flip`.

    X (bool[n, D]) is the current population and f its fitness: x = X[i],
    every x_r is a random row of X, and g is the best mask found so far (x_CP
    in the paper). The branch follows Eq. (14). With probability rand < rand CPO explores,
    with y = (x + x_r) / 2 (Eq. 5):
        sight, when rand < rand:  Y = x + N(0, 1) |2 rand g - y|                Eq. (4)
        sound, otherwise:         Y = (1 - U1) x + U1 (y + rand (x_r1 - x_r2))  Eq. (6)
    Otherwise it exploits, with gamma = 2 rand (1 - p)^p for p = progress
    (Eq. 9), delta = +1 or -1 per dimension (Eq. 8), and the weight
    w = exp(f_i / (sum of f + eps)), which is S in Eq. (10) and m in Eq. (12):
        odor, with probability Tf: Y = (1 - U1) x + U1 (x_r1 + w (x_r2 - x_r3) - rand delta gamma w)
                                                                               Eq. (7)
        physical attack, otherwise: Y = g + (alpha (1 - tau4) + tau4)(delta g - x) - rand delta gamma F,
                                    with F = rand(D) w (x_r - x)               Eqs. (11), (13)
    U1 = rand(D) > rand: the paper says only that U1 is a random binary vector.
    The paper does not say when tau4 is drawn; the callers draw it once per generation.
    """
    X = X.astype(float)
    n, D = X.shape
    x = X[i]

    def partner() -> np.ndarray:
        return X[rng.integers(n)]

    if rng.random() < rng.random():
        y = (x + partner()) / 2
        if rng.random() < rng.random():
            return x + rng.standard_normal() * np.abs(2 * rng.random() * g - y)
        U1 = rng.random(D) > rng.random()
        return (1 - U1) * x + U1 * (y + rng.random() * (partner() - partner()))
    gamma = 2 * rng.random() * (1 - progress) ** progress
    delta = np.where(rng.random(D) < 0.5, 1.0, -1.0)
    w = np.exp(f[i] / (f.sum() + np.finfo(float).eps))
    if rng.random() < _CPO_TF:
        U1 = rng.random(D) > rng.random()
        return (1 - U1) * x + U1 * (partner() + w * (partner() - partner()) - rng.random() * delta * gamma * w)
    force = rng.random(D) * w * (partner() - x)
    return g + (_CPO_ALPHA * (1 - tau4) + tau4) * (delta * g - x) - rng.random() * delta * gamma * force


def _bpso(rec: _Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary PSO (Kennedy and Eberhart 1997) with the V-shaped flip.

    Velocities start at 0. pbest is each particle's best mask and gbest the
    best pbest, both updated after every move.
    """
    V = np.zeros(X.shape)
    Pb, pbf = X.copy(), _evaluate(rec, X)
    while True:
        X, V = _pso_step(X, V, Pb, Pb[np.argmin(pbf)], rec.progress, rng)
        _update_pbest(Pb, pbf, X, _evaluate(rec, X))


def _bgwo(rec: _Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary GWO (the step of Mirjalili et al. 2014) with the V-shaped flip.

    Leaders are the three best distinct masks found so far, and a = 2 - 2 progress.
    """
    f = _evaluate(rec, X)
    leaders, lf = _best_three(X, f)
    while True:
        X = _gwo_step(X, leaders, 2 - 2 * rec.progress, rng)
        f = _evaluate(rec, X)
        leaders, lf = _best_three(np.vstack([leaders, X]), np.concatenate([lf, f]))


def _bcpo(rec: _Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary CPO (Abdel-Basset et al. 2024, Algorithm 1) with the V-shaped flip.

    X (bool[N, D]) and f hold each porcupine's best mask. Algorithm 1 keeps the
    old position only when the new one is worse, so a porcupine's position is
    its memory once its turn ends. Porcupines move one at a time by
    `_cpo_move`, and each sees the moves made before it. In a generation only
    the first n porcupines move. n is the current population of Eq. (3), which
    falls from N to N_min = round(2/3 N) and back, twice over the budget.
    Partners and the sums in Eqs. (10) and (12) come from those n, and g is the
    best of all N memories.
    """
    N = len(X)
    n_min = round(_CPO_MIN_FRACTION * N)
    f = _evaluate(rec, X)
    n = N
    while True:
        tau4 = rng.random()
        for i in range(n):
            Y = _cpo_move(i, X[:n], f[:n], X[np.argmin(f)], rec.progress, tau4, rng)
            new = _flip(X[i], Y - X[i], rng)
            new_f = rec(new)
            if new_f <= f[i]:
                X[i], f[i] = new, new_f
        n = int(n_min + (N - n_min) * (1 - (rec.progress * _CPO_CYCLES) % 1))


def _ibgwo4(rec: _Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """IBGWO4, the best variant of Khaseeb et al. 2025 (Applied Sciences 15:489), V-shaped.

    Each generation:
    1. G = one GWO step on the population X, led by the three best distinct
       masks of X, with a = 2 - 2 progress. Evaluate G.
    2. Q = one PSO step on G, with gbest the best mask found so far, which is
       the best of X and G. Evaluate Q.
    3. The next X is the best P masks of X, G and Q together (`_keep_best`).
    Velocity and pbest belong to a population slot and stay with the slot when
    X is re-ranked. Velocities start at 0 and pbest at the first masks. pbest
    takes in G before the PSO step reads it, and Q after.
    """
    f = _evaluate(rec, X)
    V, Pb, pbf = np.zeros(X.shape), X.copy(), f.copy()
    while True:
        G = _gwo_step(X, _best_three(X, f)[0], 2 - 2 * rec.progress, rng)
        fG = _evaluate(rec, G)
        _update_pbest(Pb, pbf, G, fG)
        g = _keep_best(1, (X, f), (G, fG))[0][0]
        Q, V = _pso_step(G, V, Pb, g, rec.progress, rng)
        fQ = _evaluate(rec, Q)
        _update_pbest(Pb, pbf, Q, fQ)
        X, f = _keep_best(len(X), (X, f), (G, fG), (Q, fQ))


def _gwo_cpo(rec: _Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """GWO-CPO, the proposed model: `_ibgwo4` with its PSO step replaced by one CPO step.

    Steps 1 and 3 are IBGWO4's. Step 2 moves every member of G by `_cpo_move`
    with G as the population, so partners come from G and the sums in
    Eqs. (10) and (12) run over G's fitness. gbest is the best mask found so
    far, the best of X and G. There is no population reduction and no greedy
    step, because the ranking in step 3 already keeps the best.
    """
    f = _evaluate(rec, X)
    while True:
        G = _gwo_step(X, _best_three(X, f)[0], 2 - 2 * rec.progress, rng)
        fG = _evaluate(rec, G)
        g = _keep_best(1, (X, f), (G, fG))[0][0]
        tau4 = rng.random()
        Y = np.array([_cpo_move(i, G, fG, g, rec.progress, tau4, rng) for i in range(len(G))])
        C = _flip(G, Y - G, rng)
        fC = _evaluate(rec, C)
        X, f = _keep_best(len(X), (X, f), (G, fG), (C, fC))


_ALGORITHMS: dict[str, _Algorithm] = {
    "BPSO": _bpso,
    "BGWO": _bgwo,
    "BCPO": _bcpo,
    "IBGWO4": _ibgwo4,
    "GWO-CPO": _gwo_cpo,
}
ALGORITHMS: tuple[str, ...] = tuple(_ALGORITHMS)
