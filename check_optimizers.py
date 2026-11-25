"""Behaviour checks for every registered optimiser on a toy problem with a known answer.

    python check_optimizers.py        # about 1 second; exits non-zero on the first failure

The toy fitness is the fraction of positions where a mask differs from a
planted target mask, so the optimum is 0. These checks never touch a dataset,
so a failure points at the optimiser, not at the data or the KNN.

The checks are checked too: with the shared flip rule broken on purpose
(the impostors), every optimiser must fail them.
"""
import numpy as np

from arrfs import optimizers
from arrfs.optimizers import ALGORITHMS, CURVE_POINTS, search

D, N, BUDGET, SEEDS = 100, 20, 2000, range(5)

IMPOSTOR_FLIPS = {
    "never flips a bit": lambda X, step, rng: X.copy(),
    "only turns bits on": lambda X, step, rng: X | (rng.random(X.shape) < np.abs(np.tanh(step))),
}


class PlantedFitness:
    """mask -> fraction of positions where mask != target. Logs every value it returns."""

    def __init__(self, target: np.ndarray):
        self.target = target
        self.values: list[float] = []

    def __call__(self, mask: np.ndarray) -> float:
        self.values.append(float(np.mean(mask != self.target)))
        return self.values[-1]


def problem(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(target mask with about 30% ones, shared initial population X0).

    Problem, search and random search draw from separate streams, so no
    optimiser can read the target out of its own random numbers.
    """
    rng = np.random.default_rng([seed, 0])
    return rng.random(D) < 0.3, rng.random((N, D))


def random_search_best(target: np.ndarray, seed: int) -> float:
    masks = np.random.default_rng([seed, 1]).random((BUDGET, D)) < 0.5
    return float(np.mean(masks != target, axis=1).min())


def check(name: str) -> str:
    finals, randoms = [], []
    for seed in SEEDS:
        target, X0 = problem(seed)
        fitness = PlantedFitness(target)
        res = search(name, fitness, X0, BUDGET, np.random.default_rng([seed, 2]))
        initial_best = min(np.mean((X0 > 0.5) != target, axis=1))

        assert res.calls == BUDGET == len(fitness.values), f"{name}: {res.calls} calls, budget {BUDGET}"
        assert len(res.curve) == CURVE_POINTS, f"{name}: curve has {len(res.curve)} points"
        assert np.all(np.diff(res.curve) <= 0), f"{name}: curve rises"
        assert res.fitness == res.curve[-1] == min(fitness.values), f"{name}: returned fitness is not the best evaluated"
        assert res.fitness == np.mean(res.mask != target), f"{name}: returned fitness does not match the returned mask"
        assert res.curve[0] == initial_best, f"{name}: first curve point {res.curve[0]} is not the shared population's best {initial_best}"
        assert res.fitness < initial_best, f"{name}: no better than the initial population on seed {seed}"
        rs = random_search_best(target, seed)
        assert res.fitness < rs, f"{name}: {res.fitness} is no better than random search {rs} on seed {seed}"

        again = search(name, PlantedFitness(target), X0, BUDGET, np.random.default_rng([seed, 2]))
        assert np.array_equal(again.mask, res.mask) and np.array_equal(again.curve, res.curve) \
            and again.calls == res.calls, f"{name}: same seed, different result on seed {seed}"
        finals.append(res.fitness)
        randoms.append(rs)
    return (f"{name:8s} ok  final fitness {min(finals):.2f} to {max(finals):.2f}, "
            f"random search {min(randoms):.2f} to {max(randoms):.2f}, over {len(SEEDS)} seeds")


def check_impostors() -> list[str]:
    """Every impostor flip rule must make `check` fail for every optimiser."""
    lines, real_flip = [], optimizers._flip
    try:
        for label, impostor in IMPOSTOR_FLIPS.items():
            optimizers._flip = impostor
            caught = []
            for name in ALGORITHMS:
                try:
                    check(name)
                except AssertionError as failure:
                    caught.append(str(failure))
                    continue
                raise AssertionError(f"{name} passed the checks with a flip rule that {label}")
            lines.append(f"impostor that {label}: rejected for all {len(caught)} optimisers, "
                         f"for example '{caught[-1]}'")
    finally:
        optimizers._flip = real_flip
    return lines


if __name__ == "__main__":
    print(f"toy problem: D = {D}, N = {N}, budget = {BUDGET} calls, optimum 0.00")
    for name in ALGORITHMS:
        print(check(name))
    for line in check_impostors():
        print(line)
