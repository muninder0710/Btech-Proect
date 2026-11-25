"""More binary optimisers for the comparison table and graphs, kept outside arrfs/ so that results/full stays valid.

They follow the rules of arrfs/optimizers.py. Positions are masks. Every operator proposes a continuous
position Y from a mask X, and bit d flips with probability |tanh(Y_d - X_d)| (`_flip`). Every schedule reads
`rec.progress` (calls / budget), and the same `_Recorder` owns the budget and the result. GA is the one
exception: it works on the masks themselves and has no continuous position (`NATIVE_BINARY`).

Amendment 9 (design/amendment-9-more-optimisers.md):
BCS      cuckoo search (Yang & Deb 2009) with the exponentially increasing switching parameter of
         Mareli & Twala 2018 (their Eq. 4)
BWOA     whale optimisation (Mirjalili & Lewis 2016) with the Tent chaotic map of Kaur & Arora 2018 as p
GWO-CS   the two-step frame of Khaseeb et al. 2025 with a cuckoo move as the second step
GWO-WOA  the same frame with a whale move as the second step

Amendment 10 (design/amendment-10-wider-group.md). GA has no reference code: its settings come from an earlier
arrhythmia paper and its operators are the textbook ones. The other seven were checked against their authors'
reference code or original paper:
GA       genetic algorithm with rank-based selection, uniform crossover and bit-flip mutation
BDE      differential evolution, DE/rand/1/bin (Storn & Price 1997)
BBA      binary bat algorithm (Mirjalili, Mirjalili & Yang 2014)
BDA      binary dragonfly algorithm (Mirjalili 2016)
BSCA     sine cosine algorithm (Mirjalili 2016)
BSSA     salp swarm algorithm (Mirjalili et al. 2017)
BHHO     Harris hawks optimisation (Heidari et al. 2019)
BMPA     marine predators algorithm (Faramarzi et al. 2020)

The binary versions and the two hybrids are ours. The two amendments list every choice the sources leave
open, and every place where a paper and its authors' code disagree.
"""
from math import gamma, pi, sin

import numpy as np

from arrfs import optimizers as base
from arrfs.optimizers import CURVE_POINTS, Mask, Objective, SearchResult

_CS_PA_MAX = 0.25  # Mareli & Twala 2018, Section 4.1
_LEVY_BETA = 1.5   # Levy exponent of cuckoo search, Harris hawks and marine predators
_LEVY_SIGMA = (gamma(1 + _LEVY_BETA) * sin(pi * _LEVY_BETA / 2)
               / (gamma((1 + _LEVY_BETA) / 2) * _LEVY_BETA * 2 ** ((_LEVY_BETA - 1) / 2))) ** (1 / _LEVY_BETA)
_WOA_B = 1.0       # spiral constant b of Mirjalili & Lewis 2016
_GA_ELITES = 2      # Atik & Dalyan 2023, Table 2, who also set the crossover probability to 1
_DE_F, _DE_CR = 0.5, 0.9  # the values of Carrillo-Alarcon et al. 2020, within the advice of Storn & Price 1997, Section 4
_BA_Q = (0.0, 2.0)  # Qmin and Qmax in the authors' BBA.m
_BA_PULSE = 0.1     # pulse rate r in its Main.m
_DA_STEP_MAX = 6.0  # limit of the step vector in the authors' BDA.m
_MPA_P = 0.5        # Faramarzi et al. 2020, Eq. 12
_MPA_FADS = 0.2     # probability of the FADs effect, their Eq. 16
_MPA_LEVY = 0.05    # scale of the Levy numbers, their Eq. 5


def _levy(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Levy-distributed numbers with exponent 1.5 by Mantegna's algorithm."""
    return rng.standard_normal(shape) * _LEVY_SIGMA / np.abs(rng.standard_normal(shape)) ** (1 / _LEVY_BETA)


def _levy_term(X: np.ndarray, g: Mask, rng: np.random.Generator) -> np.ndarray:
    """Step of cuckoo search's global random walk: levy (X - g) N(0, 1), where g is the best nest.
    The best nest itself does not move."""
    return _levy(X.shape, rng) * (X.astype(float) - g) * rng.standard_normal(X.shape)


def _local_term(X: np.ndarray, pa: float, rng: np.random.Generator) -> np.ndarray:
    """Step of cuckoo search's local random walk (Mareli & Twala Eq. 1): rand H(pa - rand) (X_j - X_k),
    with j and k two random orderings of the nests."""
    x = X.astype(float)
    return rng.random(X.shape) * (rng.random(X.shape) < pa) * (x[rng.permutation(len(X))] - x[rng.permutation(len(X))])


def _switch(progress: float) -> float:
    """Mareli & Twala's exponentially increasing switching parameter (their Eq. 4): 0.25 at the start, 0.68 at the end."""
    return _CS_PA_MAX * np.exp(progress)


def _tent(c: float) -> float:
    """Tent map of Kaur & Arora 2018 (Table 1, map 8)."""
    return c / 0.7 if c < 0.7 else (10 / 3) * (1 - c)


def _woa_step(X: np.ndarray, best: Mask, a: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """One whale move of every agent (Mirjalili & Lewis 2016), binarised by `_flip`.

    p >= 0.5: the spiral, Y = |best - X| e^(b l) cos(2 pi l) + best, with l uniform in [-1, 1]   Eq. (5)
    p < 0.5:  Y = R - A |C R - X|, with A = 2 a rand - a and C = 2 rand, one number per whale.
              R is `best` when |A| < 1 (Eq. 2) and a random whale otherwise (Eq. 8).
    """
    x, n = X.astype(float), len(X)
    if p >= 0.5:
        turn = rng.uniform(-1, 1, (n, 1))
        Y = np.abs(best - x) * np.exp(_WOA_B * turn) * np.cos(2 * np.pi * turn) + best
    else:
        A = 2 * a * rng.random((n, 1)) - a
        ref = np.where(np.abs(A) < 1, best.astype(float), x[rng.integers(n, size=n)])
        Y = ref - A * np.abs(2 * rng.random((n, 1)) * ref - x)
    return base._flip(X, Y - x, rng)


def _bcs(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary cuckoo search with an increasing switching parameter.

    Each generation makes a Levy flight of every nest, then the local random walk on a fraction pa
    of the dimensions. After each of the two, a nest takes its new mask only where that is not worse.
    """
    f = base._evaluate(rec, X)
    while True:
        for walk in ("levy", "local"):
            step = (_levy_term(X, X[np.argmin(f)], rng) if walk == "levy"
                    else _local_term(X, _switch(rec.progress), rng))
            new = base._flip(X, step, rng)
            new_f = base._evaluate(rec, new)
            keep = new_f <= f
            X[keep], f[keep] = new[keep], new_f[keep]


def _bwoa(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary whale optimisation with the Tent map in place of the random number p.

    The map moves once per generation, so all whales take the same branch in a generation. It starts
    from a random value. a = 2 - 2 progress. `best` is the best mask found so far.
    """
    f = base._evaluate(rec, X)
    best, best_f, c = X[np.argmin(f)].copy(), f.min(), rng.random()
    while True:
        c = _tent(c)
        X = _woa_step(X, best, 2 - 2 * rec.progress, c, rng)
        f = base._evaluate(rec, X)
        if f.min() < best_f:
            best, best_f = X[np.argmin(f)].copy(), f.min()


def _gwo_cs(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """`_ibgwo4` with its PSO step replaced by one cuckoo move: the Levy flight plus the switched
    local walk, as one proposal from the GWO output G."""
    f = base._evaluate(rec, X)
    while True:
        G = base._gwo_step(X, base._best_three(X, f)[0], 2 - 2 * rec.progress, rng)
        fG = base._evaluate(rec, G)
        g = base._keep_best(1, (X, f), (G, fG))[0][0]
        C = base._flip(G, _levy_term(G, g, rng) + _local_term(G, _switch(rec.progress), rng), rng)
        fC = base._evaluate(rec, C)
        X, f = base._keep_best(len(X), (X, f), (G, fG), (C, fC))


def _gwo_woa(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """`_ibgwo4` with its PSO step replaced by one whale move of G around the best mask found so far."""
    f = base._evaluate(rec, X)
    c = rng.random()
    while True:
        G = base._gwo_step(X, base._best_three(X, f)[0], 2 - 2 * rec.progress, rng)
        fG = base._evaluate(rec, G)
        g = base._keep_best(1, (X, f), (G, fG))[0][0]
        c = _tent(c)
        W = _woa_step(G, g, 2 - 2 * rec.progress, c, rng)
        fW = base._evaluate(rec, W)
        X, f = base._keep_best(len(X), (X, f), (G, fG), (W, fW))


def _ga(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Generational genetic algorithm on the masks themselves, with no continuous position and no shared flip.

    The two best masks pass to the next generation unchanged. Every other place is filled by a child of
    two different parents, each drawn with probability proportional to its rank (the best of N has rank N).
    A pair gives two children by uniform crossover: one takes each bit from either parent with probability
    1/2, and the other takes the remaining bits. Every bit of a child then flips with probability 1/D.
    """
    N, D = X.shape
    f = base._evaluate(rec, X)
    rank_share = np.arange(N, 0, -1) / (N * (N + 1) / 2)
    while True:
        best_first = np.argsort(f, kind="stable")
        children = []
        while len(children) < N - _GA_ELITES:
            mother, father = X[best_first[rng.choice(N, 2, replace=False, p=rank_share)]]
            from_mother = rng.random(D) < 0.5
            children += [np.where(from_mother, mother, father), np.where(from_mother, father, mother)]
        children = np.array(children[:N - _GA_ELITES])
        children ^= rng.random(children.shape) < 1 / D
        elites = best_first[:_GA_ELITES]
        X, f = np.vstack([X[elites], children]), np.concatenate([f[elites], base._evaluate(rec, children)])


def _bde(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary differential evolution, DE/rand/1/bin (Storn & Price 1997).

    For each target X_i the mutant is V = X_a + F (X_b - X_c), with a, b, c three different members other
    than i (Eq. 2). The trial takes V in every dimension where rand <= CR and in one dimension drawn at
    random, and X_i elsewhere (Eq. 4). It is binarised by the shared flip and replaces X_i when its fitness
    is not higher, as in the paper's own code (Figure 3). The paper's text says "smaller". All trials of a
    generation come from the same population.
    """
    N, D = X.shape
    f = base._evaluate(rec, X)
    while True:
        x, trials = X.astype(float), np.empty_like(X)
        for i in range(N):
            a, b, c = rng.choice(np.delete(np.arange(N), i), 3, replace=False)
            from_mutant = rng.random(D) <= _DE_CR
            from_mutant[rng.integers(D)] = True
            trials[i] = base._flip(X[i], from_mutant * (x[a] + _DE_F * (x[b] - x[c]) - x[i]), rng)
        trial_f = base._evaluate(rec, trials)
        not_worse = trial_f <= f
        X[not_worse], f[not_worse] = trials[not_worse], trial_f[not_worse]


def _bba(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary bat algorithm (Mirjalili, Mirjalili & Yang 2014), following the authors' BBA.m.

    Bats move one at a time. For every bit, with Q uniform between Qmin - Qmax and Qmin:
        V = V + (X - best) Q              the velocity, which is the step of the shared flip
        the bit then takes best's value with probability 1 - r, the pulse rate
    `best` is the best mask found so far, and a later mask replaces it on a tie. Velocities start at 0.
    BBA.m never rejects a move: its loudness test changes a stored fitness and nothing else, so loudness
    plays no part in the search and is left out.
    """
    V = np.zeros(X.shape)
    f = base._evaluate(rec, X)
    best, best_f = X[np.argmin(f)].copy(), f.min()
    while True:
        for i in range(len(X)):
            Q = _BA_Q[0] + (_BA_Q[0] - _BA_Q[1]) * rng.random(X.shape[1])
            V[i] += (X[i].astype(float) - best) * Q
            X[i] = np.where(rng.random(X.shape[1]) > _BA_PULSE, best, base._flip(X[i], V[i], rng))
            new_f = rec(X[i])
            if new_f <= best_f:
                best, best_f = X[i].copy(), new_f


def _bda(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary dragonfly algorithm (Mirjalili 2016), following the authors' BDA.m.

    Every dragonfly has all the others as neighbours. Each generation evaluates the swarm, then moves
    the dragonflies one at a time, each seeing the moves made before it:
        S = sum of (X_i - X_j)                separation                      Eq. (3.1)
        A = mean step vector of the others    alignment                       Eq. (3.2)
        C = mean position of the others - X_i cohesion                        Eq. (3.3)
        F = food - X_i,  E = enemy + X_i                                      Eqs. (3.4), (3.5)
        DX_i = s S + a A + c C + f F + e E + w DX_i, limited to [-6, 6]       Eq. (3.6)
    DX_i is the step of the shared flip. Food and enemy are the best and the worst mask found so far.
    One draw per generation of s, a, c = 2 rand m and f = 2 rand, with e = m, m = max(0.1 - 0.2 progress, 0)
    and w = 0.9 - 0.5 progress. Step vectors start as random 0s and 1s.
    """
    N = len(X)
    DX = (rng.random(X.shape) > 0.5).astype(float)
    enemy, enemy_f = np.zeros(X.shape[1]), -np.inf
    while True:
        w, m = 0.9 - 0.5 * rec.progress, max(0.1 - 0.2 * rec.progress, 0.0)
        s, a, c, to_food = 2 * rng.random() * m, 2 * rng.random() * m, 2 * rng.random() * m, 2 * rng.random()
        f = base._evaluate(rec, X)
        food = rec.best_mask.astype(float)
        if f.max() > enemy_f:
            enemy, enemy_f = X[np.argmax(f)].astype(float), f.max()
        for i in range(N):
            x, others = X.astype(float), np.arange(N) != i
            S = (x[i] - x[others]).sum(axis=0)
            A = DX[others].mean(axis=0)
            C = x[others].mean(axis=0) - x[i]
            DX[i] = np.clip(s * S + a * A + c * C + to_food * (food - x[i]) + m * (enemy + x[i]) + w * DX[i],
                            -_DA_STEP_MAX, _DA_STEP_MAX)
            X[i] = base._flip(X[i], DX[i], rng)


def _bsca(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary sine cosine algorithm (Mirjalili 2016), following the authors' SCA.m.

    Y = X + r1 sin(r2) |r3 P - X| or X + r1 cos(r2) |r3 P - X|, each with probability 1/2 per dimension
    (Eqs. 3.1 to 3.3). P is the best mask found so far, r2 is uniform in [0, 2 pi], r3 in [0, 2], and
    r1 = 2 - 2 progress (Eq. 3.4). No move is rejected.
    """
    base._evaluate(rec, X)
    while True:
        r1, x, r2 = 2 - 2 * rec.progress, X.astype(float), 2 * np.pi * rng.random(X.shape)
        swing = np.where(rng.random(X.shape) < 0.5, np.sin(r2), np.cos(r2))
        X = base._flip(X, r1 * swing * np.abs(2 * rng.random(X.shape) * rec.best_mask - x), rng)
        base._evaluate(rec, X)


def _bssa(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary salp swarm algorithm (Mirjalili et al. 2017), following the authors' SSA.m.

    The first half of the chain are leaders. A leader moves around the food F, the best mask found so far:
    Y = F + c1 c2 or F - c1 c2, each with probability 1/2 per dimension, with c2 uniform in [0, 1] and
    c1 = 2 exp(-(4 progress)^2) (Eqs. 3.1 and 3.2 between the bounds 0 and 1). A follower moves to the
    midpoint of itself and the salp ahead of it, which has already moved (Eq. 3.4). No move is rejected.
    """
    N, D = X.shape
    base._evaluate(rec, X)
    while True:
        c1, food = 2 * np.exp(-(4 * rec.progress) ** 2), rec.best_mask.astype(float)
        for i in range(N):
            x = X[i].astype(float)
            if i < N // 2:
                Y = food + np.where(rng.random(D) < 0.5, c1, -c1) * rng.random(D)
            else:
                Y = (x + X[i - 1]) / 2
            X[i] = base._flip(X[i], Y - x, rng)
        base._evaluate(rec, X)


def _bhho(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary Harris hawks optimisation (Heidari et al. 2019), following the authors' HHO.m.

    Hawks move one at a time and see the moves made before them. The rabbit is the best mask found when
    the generation starts. E = 2 (1 - progress)(2 rand - 1) is the rabbit's energy (Eq. 3), J = 2 (1 - rand),
    X_m is the mean of the hawks and every rand is one number per hawk.
    |E| >= 1, exploration (Eq. 1), each rule with probability 1/2:
        Y = X_r - rand |X_r - 2 rand X|, with X_r a random hawk
        Y = (rabbit - X_m) - rand rand, which is the paper's rule between the bounds 0 and 1
    |E| < 1 and r >= 0.5:
        |E| >= 0.5, soft besiege:  Y = (rabbit - X) - E |J rabbit - X|        Eqs. (4), (5)
        |E| <  0.5, hard besiege:  Y = rabbit - E |rabbit - X|                Eq. (6)
    |E| < 1 and r < 0.5, rapid dives:
        Y = rabbit - E |J rabbit - X| when |E| >= 0.5, and rabbit - E |J rabbit - X_m| otherwise
                                                                               Eqs. (7), (12)
        The hawk takes the mask from Y if that is better than its own. If not, it tries
        Z = Y + rand(D) Levy(D) and takes that mask if it is better                Eqs. (8), (10), (13)
    Every other move is taken whatever its fitness. The Levy numbers carry no factor 0.01, as in HHO.m.
    """
    N, D = X.shape
    f = base._evaluate(rec, X)
    while True:
        rabbit, energy = rec.best_mask.astype(float), 2 * (1 - rec.progress)
        for i in range(N):
            x, E = X[i].astype(float), energy * (2 * rng.random() - 1)
            if abs(E) < 1 and rng.random() < 0.5:
                anchor = x if abs(E) >= 0.5 else X.mean(axis=0)
                Y = rabbit - E * np.abs(2 * (1 - rng.random()) * rabbit - anchor)
                new = base._flip(X[i], Y - x, rng)
                new_f = rec(new)
                if not new_f < f[i]:
                    new = base._flip(X[i], Y + rng.random(D) * _levy((D,), rng) - x, rng)
                    new_f = rec(new)
                if new_f < f[i]:
                    X[i], f[i] = new, new_f
                continue
            if abs(E) >= 1:
                if rng.random() < 0.5:
                    perch = X[rng.integers(N)].astype(float)
                    Y = perch - rng.random() * np.abs(perch - 2 * rng.random() * x)
                else:
                    Y = (rabbit - X.mean(axis=0)) - rng.random() * rng.random()
            elif abs(E) >= 0.5:
                Y = (rabbit - x) - E * np.abs(2 * (1 - rng.random()) * rabbit - x)
            else:
                Y = rabbit - E * np.abs(rabbit - x)
            X[i] = base._flip(X[i], Y - x, rng)
            f[i] = rec(X[i])


def _bmpa(rec: base._Recorder, X: np.ndarray, rng: np.random.Generator) -> None:
    """Binary marine predators algorithm (Faramarzi et al. 2020), following the authors' MPA.m.

    Each generation makes two moves, and after each a prey keeps its old mask only when that was
    strictly better (the marine memory). The elite is the best mask found so far, R_B is normal, R_L is
    0.05 times a Levy number, R is uniform, all per prey and dimension, and CF = (1 - p)^(2p), p = progress.
    Move 1, by phase:
        p < 1/3, or the first half of the prey while p < 2/3:
            Y = X + 0.5 R step, step = R_B (elite - R_B X) in phase 1 and R_L (elite - R_L X) in phase 2
                                                                               Eqs. (12), (13)
        otherwise:
            Y = elite + 0.5 CF step, step = R_B (R_B elite - X) in phase 2 and R_L (R_L elite - X) in phase 3
                                                                               Eqs. (14), (15)
    Move 2, the FADs effect (Eq. 16), one draw for the whole population as in MPA.m:
        with probability 0.2:  Y = X + CF rand U, with U one in a fifth of the dimensions
        otherwise:             Y = X + (0.2 (1 - r) + r)(X_j - X_k), with j and k two random orderings
    The paper's text gives U four fifths of ones. MPA.m gives it one fifth, and this follows MPA.m.
    """
    N, D = X.shape
    f = base._evaluate(rec, X)

    def move(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new = base._flip(X, Y - X, rng)
        new_f = base._evaluate(rec, new)
        old_better = f < new_f
        return np.where(old_better[:, None], X, new), np.where(old_better, f, new_f)

    while True:
        p, elite, x = rec.progress, rec.best_mask.astype(float), X.astype(float)
        CF = (1 - p) ** (2 * p)
        RL, RB, R = _MPA_LEVY * _levy((N, D), rng), rng.standard_normal((N, D)), rng.random((N, D))
        prey_led = np.full((N, 1), p < 1 / 3) | ((np.arange(N) < N // 2)[:, None] & (p < 2 / 3))
        walk = RB if p < 1 / 3 else RL
        chase = RB if p < 2 / 3 else RL
        X, f = move(np.where(prey_led, x + _MPA_P * R * walk * (elite - walk * x),
                             elite + _MPA_P * CF * chase * (chase * elite - x)))
        x = X.astype(float)
        if rng.random() < _MPA_FADS:
            X, f = move(x + CF * rng.random((N, D)) * (rng.random((N, D)) < _MPA_FADS))
        else:
            r = rng.random()
            X, f = move(x + (_MPA_FADS * (1 - r) + r) * (x[rng.permutation(N)] - x[rng.permutation(N)]))


_ALGORITHMS = {"BCS": _bcs, "BWOA": _bwoa, "GWO-CS": _gwo_cs, "GWO-WOA": _gwo_woa,
               "GA": _ga, "BDE": _bde, "BBA": _bba, "BDA": _bda, "BSCA": _bsca, "BSSA": _bssa, "BHHO": _bhho,
               "BMPA": _bmpa}
ALGORITHMS: tuple[str, ...] = tuple(_ALGORITHMS)
NATIVE_BINARY: tuple[str, ...] = ("GA",)  # optimisers that never call the shared flip


def search(name: str, fitness: Objective, X0: np.ndarray, budget: int, rng: np.random.Generator) -> SearchResult:
    """`arrfs.optimizers.search` for the algorithms of this module: the same contract and recorder."""
    assert budget % CURVE_POINTS == 0 and budget >= len(X0), budget
    rec = base._Recorder(fitness, budget)
    try:
        _ALGORITHMS[name](rec, X0 > 0.5, rng)
    except base._BudgetSpent:
        pass
    assert rec.calls == budget and len(rec.curve) == CURVE_POINTS, (name, rec.calls)
    return SearchResult(rec.best_mask, rec.best_fitness, np.array(rec.curve), rec.calls)
