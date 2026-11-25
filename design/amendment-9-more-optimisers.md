# Amendment 9: four more optimisers as comparison rows

Decided on 2026-11-21, before any of the four had run. The students asked for cuckoo search and other optimisers "for graphs' sake". This adds rows to the fair comparison. It tests no claim.

## What this is and is not

- It is a wider comparison: nine optimisers in place of five, on the same folds, with both starts.
- It is not a search for a winner. The stop rule of Amendment 8 stands. None of the four can become the proposed model, and no "beats the base paper" claim is made from them, whatever they score.
- Every row is reported, including any that do badly.

## The four optimisers

All four live in `extra_optimizers.py`, outside `arrfs/`, and follow the rules of `arrfs/optimizers.py`: positions are masks, an operator proposes a continuous position Y from a mask X, bit d flips with probability |tanh(Y_d - X_d)|, schedules read calls / budget, and the same recorder owns the budget.

1. **BCS**, cuckoo search (Yang & Deb 2009) with the exponentially increasing switching parameter of Mareli & Twala 2018 (their Eq. 4, pa = 0.25 exp(progress)). Each generation makes a Lévy flight of every nest (their Eq. 2, with Yang's step proportional to the distance from the best nest, Lévy exponent 1.5 by Mantegna's algorithm), then the local random walk of their Eq. 1 on a fraction pa of the dimensions. Each of the two is evaluated and kept only where it is not worse.
2. **BWOA**, whale optimisation (Mirjalili & Lewis 2016) with the Tent chaotic map of Kaur & Arora 2018 in place of the random number p. With p below 0.5 a whale encircles the best mask when |A| < 1 and a random whale otherwise. With p at or above 0.5 it follows the spiral (b = 1, l uniform in [-1, 1]). a falls from 2 to 0.
3. **GWO-CS**, the two-step frame of Khaseeb et al. 2025 with a cuckoo move as the second step (the Lévy flight plus the switched local walk in one proposal).
4. **GWO-WOA**, the same frame with a whale move as the second step.

Choices that are ours, because the source papers work in continuous space:
- The step scale is 1, not the 0.01 that Mareli & Twala use for wide continuous ranges. In the unit cube 0.01 would freeze the search.
- The Tent map starts from a random value, as Kaur & Arora's Algorithm 2 says. Their Section 3 names 0.7, which sends the map to 0 after two steps.
- A and C in the whale move are one number per whale, as in Mirjalili's code.
- These binary versions and the two hybrids are our constructions. They are not taken from a published feature-selection paper.

## Design

- The main experiment's protocol and folds: both Arrhythmia tasks, the 25 outer folds of `RepeatedStratifiedKFold(5, 5, random_state=42)`, population 20, 2,000 fitness calls, both fitness kinds, 5-NN inside the search, four classifiers at the end.
- Both starts: the random start of the main run and the informed start of Amendment 6.
- The five existing optimisers are not rerun. Their rows come from `results/full` and `results/informed_init`.
- Before the run, each new optimiser must pass the toy checks of `check_optimizers.py` (budget, curve, best mask, determinism, better than random search).

## Reported

One table and a set of graphs with all nine optimisers: macro-F1, accuracy and number of features by optimiser and start, and convergence curves. One descriptive line: the spread between the best and the worst optimiser under each start. No test, no claim.

The script is `more_optimizers_demo.py`. Results and graphs go to `results/more_optimizers/`.

## Addendum, 2026-11-21: curves for the graphs

Written after the first run of the four new optimisers and before any graph was drawn.

- The Amendment 6 run did not save convergence curves. To draw all nine optimisers from both starts, `more_optimizers_demo.py` now reruns the five existing optimisers from the informed start and records their curves. The rerun must reproduce the subset sizes and fitness values saved in `results/informed_init` exactly, and the script stops if it does not. No score is taken from the rerun, so the sentence above still holds for scores: the five are not scored again.
- The curves of all nine optimisers go to `all_optimizers_curves.csv`, which replaces `more_optimizers_curves.csv`. The random-start curves of the five existing optimisers are read from the fold files of `results/full`.
- The graphs (`plot_optimizers.py`) show 5-NN, the classifier inside the search, and the imbalance-aware fitness, which is the proposed model's fitness (`PROPOSED` in `arrfs/experiment.py`). The same graphs for the accuracy fitness are written next to them, so no fitness kind is left out.
- A graph carries the descriptive line promised above, the spread between the best and the worst optimiser under each start. It carries no test.
