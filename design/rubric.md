# Rubric, failure modes, and the decision each result drives

Written before any implementation exists. Nothing here may change after outer-test results are seen.

## Rubric for picking the design (arena Phase A)

1. Caller usage. One command runs the experiment and one command analyses it. A teammate can read every new file in one sitting.
2. Leakage is structurally impossible. Search code never receives outer-test rows. Preprocessing is fitted on outer-train rows only.
3. Fair comparison by structure. Every algorithm in a fold gets the same initial population, the same inner folds and the same fitness object type. Seeds are derived from the job key, not from global state.
4. Restartable and safe in parallel. Each fold writes its own file. A rerun skips finished folds and produces identical bytes. No two workers write the same file.
5. One list of algorithms and fitness kinds. The grid, the analysis tables and the method names all derive from it. Adding an algorithm means writing one function and one registry entry.
6. The results schema answers every analysis question (pooled out-of-fold metrics, masks, curves, per-class recall) without rerunning the search.

## Failure modes the tests must catch (written before the code)

Optimiser contract, checked on every registered algorithm:

1. The returned best fitness is not the minimum the algorithm evaluated, or does not equal fitness(returned mask).
2. The convergence curve rises at any point, or starts above the best fitness in the shared initial population.
3. The algorithm spends far more or far fewer fitness calls than its budget (N x (T + 1), less for CPO's population reduction).
4. The same seed gives a different mask or curve.
5. On a synthetic problem with a known best subset, the algorithm does no better than random search with the same number of calls.

Experiment protocol:

6. Search depends on outer-test rows. Replacing the outer-test feature values with noise must leave every selected mask unchanged.
7. A rerun after deleting one fold's file recomputes only that fold, and the merged table is byte-identical to the first run.
8. A (task, repeat, fold, method, fitness) key appears more or less than once in the merged table.
9. Each record is predicted exactly once per (task, repeat, method, fitness) in the out-of-fold predictions.
10. The ANOVA reference in a fold uses a k different from the number of features GWO-CPO with the imbalance fitness picked in that fold.

Fitness:

11. `class_scores` disagrees with scikit-learn's accuracy, balanced accuracy or macro-F1 on random label vectors (including classes missing from predictions).
12. The numpy 5-NN disagrees with scikit-learn's `KNeighborsClassifier` on any random subset.

## Pre-registered claims

Primary metric: macro-F1 on outer-test folds. Secondary: balanced accuracy, accuracy, number of features. Unit of pairing: the 25 outer folds (5 x 5 repeated stratified CV) per task. Test: two-sided Wilcoxon signed-rank, alpha 0.05, no multiple-comparison correction claimed beyond reporting all p-values.

| Claim | Holds only if | If it fails, the slides say |
|---|---|---|
| GWO-CPO beats the base paper's PSO-GWO | GWO-CPO + imbalance fitness has higher mean macro-F1 than PSO-GWO + accuracy fitness, p < 0.05 | "comparable to the base paper" with the numbers |
| The hybrid beats its parts | GWO-CPO beats both GWO and CPO under the same fitness, p < 0.05 | the hybrid is dropped as a contribution, and CPO alone is reported |
| The imbalance fitness helps | macro-F1 rises with the imbalance fitness for at least 4 of 5 algorithms, each p < 0.05 or a consistent sign | "no measurable effect" with the numbers |
| Feature selection beats no selection | GWO-CPO + imbalance beats 5-NN on all features, p < 0.05 | "same score with far fewer features", with the counts |
| The wrapper beats a simple filter | GWO-CPO + imbalance beats ANOVA top-k + 5-NN with the same k, p < 0.05 | "a filter does as well", reported plainly |

Noise floor: the spread of fold-level differences. A mean gap smaller than its standard error is reported as no difference regardless of the sign.

## Amendment 1 (2026-11-20, before any search result existed)

The method base paper is now Khaseeb et al. 2025, and the algorithm list is the one in `amendment-khaseeb.md`. In the claims table, "the base paper's PSO-GWO" means IBGWO4 run with the accuracy fitness, and "its parts" means BGWO and BCPO. All other rows stand.

## Amendment 3 (2026-11-20, before any result of the amended design existed)

Contributions are narrowed to three (see `amendment-3-contributions.md`). The "imbalance fitness helps" row is now read as a head-to-head test whose either outcome is reported as a finding. New row: choosing the subset on the reported split inflates the score only if the paired paper-style minus clean difference has a 95% bootstrap interval excluding 0 over 25 splits; otherwise the slides say no measurable inflation.

## Amendment 4 (2026-11-20, before experiment 3(c) existed)

Contribution 1 is claimed only as the first binary GWO + CPO hybrid for feature selection (see `amendment-4-review-and-split.md`). New experiment 3(c), a single small test split, replicates the pipeline Dhanka & Maini 2025 describe. Its two claims and their pass conditions are fixed in that file: the clean arm's 5th-to-95th-percentile accuracy range over 200 stratified 42-patient test sets must exceed 10 points, and the paired paper-style minus clean difference needs a 95% bootstrap interval excluding 0.

## Amendment 5 (2026-11-20, after the main results, before this check existed)

An exploratory check reruns IBGWO4 and GWO-CPO under the base paper's own protocol (see `amendment-5-base-protocol.md`). Per task and per reading of where the fitness is scored, it holds only if GWO-CPO's mean paired test-accuracy gain over 30 random 80/20 runs is above 0 with Wilcoxon p below 0.05. Claim 1's verdict does not change whatever it shows.

## Amendment 6 (2026-11-21, after claim 1 and Amendment 5, before any search with the new start)

One literature-motivated variant is tested once (see `amendment-6-informed-start.md`). The informed start of Li, Xue & Zhang 2021 (their Eq. 14, λ = 0.5, σ = 10) replaces the random start for all five algorithms on the main experiment's Arrhythmia folds. New row: GWO-CPO with the imbalance fitness and the informed start beats the base paper's method as published (IBGWO4, accuracy fitness, random start) only if its mean KNN macro-F1 is higher over the 25 folds with Wilcoxon p below 0.05. Stop rule: no further variant is tried on these folds for that claim.

## Amendment 7 (2026-11-21, before experiment 3(d) existed)

New experiment 3(d) measures what tuning on the test patients does to the base paper's pipeline (see `amendment-7-tuning-on-test.md`). New row: tuning on the test patients inflates the printed accuracy only if, over 40 splits, the shortcut arm's accuracy on its 42 test patients minus its accuracy on 42 untouched patients has a 95% bootstrap interval above 0. No claim is made about what the authors of the base paper did.

## Amendment 8 (2026-11-21, before the same-classifier experiment existed)

Last planned experiment (see `amendment-8-same-classifier.md`). On 25 fresh outer folds per task, the proposed selection (informed-start GWO-CPO wrapping a light XGBoost) and the base paper's filter both feed a default XGBoost. Two rows per task: the proposed selection beats the base filter only if the mean paired difference in the primary metric is above 0 with Wilcoxon p below 0.05; it matches with fewer features only if the 95% bootstrap interval lies above -0.02 and it uses fewer than 50 features on average. Stop rule: no further variant is tested for a "beats the base paper" claim after this.

## Amendment 9 (2026-11-21, before any of the four new optimisers had run)

Four more optimisers (cuckoo search, whale optimisation, and GWO followed by each) are added as comparison rows for the graphs (see `amendment-9-more-optimisers.md`). They test no claim. The stop rule of Amendment 8 stands: none of them can become the proposed model or support a "beats the base paper" claim.


## Amendment 10 (2026-11-22, before any of the eight new optimisers had run on the data)

Eight more optimisers (genetic algorithm, differential evolution, bat, dragonfly, sine cosine, salp swarm, Harris hawks and marine predators) are added as comparison rows (see `amendment-10-wider-group.md`). The rule that chose them is fixed there: an arrhythmia paper in the students' notes used the optimiser, its update rule maps onto the shared flip without redesign, and it was checked against its authors' code or paper. They test no claim. The stop rule of Amendment 8 stands: none of them can become the proposed model or support a "beats the base paper" claim, whatever they score.

## Amendment 11 (2026-11-23, before any optimiser had run on any function)

The five optimisers of the main experiment are run on thirteen standard benchmark functions, as published and with the optimum shifted away from the origin (see `amendment-11-benchmark-functions.md`). It is a side check asked for by the students and belongs in an appendix. It is the same five optimisers on a different test bed, not a new variant, and its outcome cannot support a "beats the base paper" claim in the arrhythmia paper. The stop rule of Amendment 8 stands.

## Amendment 12 (2026-11-23, before any optimiser had run on any of the nine datasets)

The base method and the proposed model of Amendment 6, unchanged, are run on the nine gene-expression datasets of the base paper under the base paper's protocol, in the two readings of Amendment 5 (see `amendment-12-base-datasets.md`). It is a replication on new data, not a new variant. The claim of Amendment 6 replicates there only if, in the clean reading, the proposed model is higher than the base method on macro-F1 across the nine datasets by a Wilcoxon signed-rank test at 5%. If it is not, the write-up says that the arrhythmia result did not carry over.
