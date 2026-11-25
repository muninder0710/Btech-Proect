"""The toy checks of check_optimizers.py, applied to the optimisers of extra_optimizers.py.

    python check_extra_optimizers.py        # a few seconds; exits non-zero on the first failure

The checks are checked too. With the shared flip rule broken on purpose, every optimiser that uses it must
either fail the checks or end worse than with the real rule on every seed. The second case is for an
optimiser with an operator of its own next to the flip, which a broken flip slows down without stopping.
Optimisers that never call the shared flip (extra_optimizers.NATIVE_BINARY) are left out of this part.
"""
import numpy as np

import check_optimizers
import extra_optimizers
from arrfs import optimizers


def final_fitness(name: str) -> list[float]:
    """The best fitness `name` reaches on the toy problem of each seed."""
    finals = []
    for seed in check_optimizers.SEEDS:
        target, X0 = check_optimizers.problem(seed)
        fitness = check_optimizers.PlantedFitness(target)
        rng = np.random.default_rng([seed, 2])
        finals.append(extra_optimizers.search(name, fitness, X0, check_optimizers.BUDGET, rng).fitness)
    return finals


def main() -> None:
    check_optimizers.search = extra_optimizers.search
    for name in extra_optimizers.ALGORITHMS:
        print(check_optimizers.check(name))
    flipping = [name for name in extra_optimizers.ALGORITHMS if name not in extra_optimizers.NATIVE_BINARY]
    real = {name: final_fitness(name) for name in flipping}
    real_flip = optimizers._flip
    try:
        for label, impostor in check_optimizers.IMPOSTOR_FLIPS.items():
            optimizers._flip = impostor
            slowed = []
            for name in flipping:
                try:
                    check_optimizers.check(name)
                except AssertionError:
                    continue
                assert all(broken > intact for broken, intact in zip(final_fitness(name), real[name])), \
                    f"{name} did as well with a flip rule that {label}"
                slowed.append(name)
            print(f"impostor that {label}: caught for all {len(flipping)} optimisers that use the shared flip"
                  + (f" ({', '.join(slowed)} passed the checks but ended worse on every seed)" if slowed else ""))
    finally:
        optimizers._flip = real_flip
    print(f"not put through the impostors, because they never call the shared flip: {', '.join(extra_optimizers.NATIVE_BINARY)}")


if __name__ == "__main__":
    main()
